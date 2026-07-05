#!/usr/bin/env python3
"""
Experiment: Course Start Percentile Sensitivity Analysis

Tests how different percentiles for defining "course start" affect
model performance at early cutoffs (weeks 2, 4, 6).

Hypothesis: Using 10% percentile instead of 5% may improve early
prediction by ensuring more students have started activity.

Output:
    data/analysis/percentile_sensitivity_results.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta
from collections import Counter
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from xgboost import XGBClassifier

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/analysis/percentile_sensitivity_results.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Experiment parameters
PERCENTILES = [0.05, 0.10, 0.15, 0.20]
CUTOFF_WEEKS = [2, 4, 6]
SESSION_GAP_MINUTES = 30

# XGBoost parameters
XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'n_estimators': 100,
    'subsample': 0.8,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'random_state': 42,
}


def normalize_user_id(user_id):
    """Normalize Canvas user_id to short format."""
    if user_id > 10000000000:
        return user_id % 10000000000
    return user_id


def load_data():
    """Load page views and enrollments."""
    print("Loading data...")
    df = pd.read_parquet(PAGE_VIEWS_FILE)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df[df['course_id'].isin(COURSES)].copy()
    df['user_id'] = df['user_id'].apply(normalize_user_id)

    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    df_enroll['failed'] = (df_enroll['final_score'] < 57).astype(int)

    print(f"  Loaded {len(df):,} page views")
    return df, df_enroll


def get_course_starts(df, percentile):
    """Calculate course start dates using given percentile."""
    course_starts = {}
    for course_id in COURSES:
        df_course = df[df['course_id'] == course_id]
        if len(df_course) > 0:
            course_starts[course_id] = df_course['created_at'].quantile(percentile)
    return course_starts


def filter_by_cutoff(df, course_starts, cutoff_weeks):
    """Filter page views to first N weeks from course start."""
    filtered = []
    for course_id in COURSES:
        if course_id not in course_starts:
            continue
        start = course_starts[course_id]
        cutoff_date = start + timedelta(weeks=cutoff_weeks)
        df_course = df[(df['course_id'] == course_id) & (df['created_at'] <= cutoff_date)]
        filtered.append(df_course)

    if not filtered:
        return pd.DataFrame()
    return pd.concat(filtered, ignore_index=True)


def calculate_simple_features(df, course_starts):
    """Calculate simplified features for quick experimentation."""
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        if len(group) < 2:
            continue

        group = group.sort_values('created_at')
        timestamps = pd.to_datetime(group['created_at'])
        course_start = course_starts.get(course_id, timestamps.min())

        # Session features
        gaps = timestamps.diff().dt.total_seconds() / 60
        session_starts = gaps >= SESSION_GAP_MINUTES
        session_starts.iloc[0] = True
        session_ids = session_starts.cumsum()
        n_sessions = session_ids.max()

        total_span = max((timestamps.max() - course_start).days / 7, 1)

        features = {
            'user_id': user_id,
            'course_id': course_id,
            'total_views': len(group),
            'n_sessions': n_sessions,
            'sessions_per_week': n_sessions / total_span,
        }

        # Category features
        for cat in ['files', 'discussions', 'quizzes', 'assignments', 'modules', 'grades']:
            cat_data = group[group['resource_type'] == cat]
            features[f'{cat}_views'] = len(cat_data)
            features[f'{cat}_unique'] = cat_data['resource_id'].nunique() if 'resource_id' in cat_data.columns else 0

        # Time features
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofweek
        total = len(hours)

        features['pct_weekend'] = sum(days >= 5) / total * 100 if total > 0 else 0
        features['pct_evening'] = sum((hours >= 18) & (hours < 24)) / total * 100 if total > 0 else 0

        # Navigation features
        types = group['resource_type'].tolist()
        bigrams = [(types[i], types[i+1]) for i in range(len(types)-1)]
        features['total_transitions'] = len(bigrams)
        features['unique_transitions'] = len(set(bigrams))

        # Temporal features
        weeks = ((timestamps - course_start).dt.days // 7 + 1).clip(lower=1)
        features['active_weeks'] = weeks.nunique()
        features['first_active_week'] = weeks.min()
        features['last_active_week'] = weeks.max()

        results.append(features)

    return pd.DataFrame(results)


def calculate_znorm_features(df):
    """Add z-score normalized features per course."""
    exclude_cols = ['user_id', 'course_id']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    znorm_data = []
    for course_id in df['course_id'].unique():
        course_df = df[df['course_id'] == course_id].copy()
        for col in feature_cols:
            col_data = course_df[col]
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                course_df[f'{col}_znorm'] = (col_data - mean_val) / std_val
            else:
                course_df[f'{col}_znorm'] = 0
        znorm_data.append(course_df)

    return pd.concat(znorm_data, ignore_index=True)


def train_and_evaluate(df_features, df_enroll):
    """Train XGBoost and evaluate."""
    # Merge with enrollments
    df = df_features.merge(
        df_enroll[['user_id', 'course_id', 'failed']],
        on=['user_id', 'course_id'],
        how='inner'
    )
    df = df.dropna(subset=['failed'])

    if len(df) < 50:
        return None

    # Prepare features
    exclude_cols = ['user_id', 'course_id', 'failed']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['failed'].values

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Keep only numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    if len(numeric_cols) < 5:
        return None

    # Train with CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)

    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        return {
            'n_samples': int(len(X)),
            'n_features': len(numeric_cols),
            'failure_rate': float(y.mean()),
            'roc_auc': float(roc_auc_score(y, y_pred_proba)),
            'accuracy': float(accuracy_score(y, y_pred)),
            'recall': float(recall_score(y, y_pred, zero_division=0)),
            'precision': float(precision_score(y, y_pred, zero_division=0)),
            'f1': float(f1_score(y, y_pred, zero_division=0)),
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def run_experiment(df_full, df_enroll, percentile, cutoff_weeks):
    """Run a single experiment."""
    # Get course starts with this percentile
    course_starts = get_course_starts(df_full, percentile)

    # Filter data
    df_filtered = filter_by_cutoff(df_full, course_starts, cutoff_weeks)

    if len(df_filtered) == 0:
        return None

    # Calculate features
    df_features = calculate_simple_features(df_filtered, course_starts)

    if len(df_features) == 0:
        return None

    # Add z-norm
    df_features = calculate_znorm_features(df_features)

    # Train and evaluate
    return train_and_evaluate(df_features, df_enroll)


def main():
    print("=" * 70)
    print("EXPERIMENT: Course Start Percentile Sensitivity Analysis")
    print("=" * 70)
    print()

    df_full, df_enroll = load_data()

    results = {
        'experiments': [],
        'summary': {}
    }

    # Run all experiments
    print("\nRunning experiments...")
    print("-" * 70)

    for percentile in PERCENTILES:
        print(f"\nPercentile: {int(percentile*100)}%")
        for cutoff in CUTOFF_WEEKS:
            print(f"  Week {cutoff}...", end=" ")

            metrics = run_experiment(df_full, df_enroll, percentile, cutoff)

            if metrics:
                exp = {
                    'percentile': percentile,
                    'cutoff_weeks': cutoff,
                    **metrics
                }
                results['experiments'].append(exp)
                print(f"ROC-AUC={metrics['roc_auc']:.3f}, Acc={metrics['accuracy']*100:.1f}%, Samples={metrics['n_samples']}")
            else:
                print("FAILED")

    # Create summary table
    print("\n" + "=" * 70)
    print("SUMMARY: ROC-AUC by Percentile and Cutoff Week")
    print("=" * 70)
    print()

    header = f"{'Percentile':<12}"
    for cutoff in CUTOFF_WEEKS:
        header += f"{'Week '+str(cutoff):>12}"
    print(header)
    print("-" * (12 + 12 * len(CUTOFF_WEEKS)))

    for percentile in PERCENTILES:
        row = f"{int(percentile*100)}%{'':<9}"
        for cutoff in CUTOFF_WEEKS:
            exp = next((e for e in results['experiments']
                       if e['percentile'] == percentile and e['cutoff_weeks'] == cutoff), None)
            if exp:
                row += f"{exp['roc_auc']:>12.3f}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Accuracy table
    print()
    print("SUMMARY: Accuracy by Percentile and Cutoff Week")
    print("-" * (12 + 12 * len(CUTOFF_WEEKS)))

    for percentile in PERCENTILES:
        row = f"{int(percentile*100)}%{'':<9}"
        for cutoff in CUTOFF_WEEKS:
            exp = next((e for e in results['experiments']
                       if e['percentile'] == percentile and e['cutoff_weeks'] == cutoff), None)
            if exp:
                row += f"{exp['accuracy']*100:>11.1f}%"
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Sample sizes
    print()
    print("SUMMARY: Sample Sizes by Percentile and Cutoff Week")
    print("-" * (12 + 12 * len(CUTOFF_WEEKS)))

    for percentile in PERCENTILES:
        row = f"{int(percentile*100)}%{'':<9}"
        for cutoff in CUTOFF_WEEKS:
            exp = next((e for e in results['experiments']
                       if e['percentile'] == percentile and e['cutoff_weeks'] == cutoff), None)
            if exp:
                row += f"{exp['n_samples']:>12}"
            else:
                row += f"{'N/A':>12}"
        print(row)

    # Find best configuration for each cutoff
    print()
    print("=" * 70)
    print("BEST CONFIGURATION PER CUTOFF")
    print("=" * 70)

    for cutoff in CUTOFF_WEEKS:
        cutoff_exps = [e for e in results['experiments'] if e['cutoff_weeks'] == cutoff]
        if cutoff_exps:
            best = max(cutoff_exps, key=lambda x: x['roc_auc'])
            print(f"\nWeek {cutoff}:")
            print(f"  Best percentile: {int(best['percentile']*100)}%")
            print(f"  ROC-AUC: {best['roc_auc']:.3f}")
            print(f"  Accuracy: {best['accuracy']*100:.1f}%")
            print(f"  Recall: {best['recall']*100:.1f}%")
            print(f"  Samples: {best['n_samples']}")

            results['summary'][f'week_{cutoff}'] = {
                'best_percentile': best['percentile'],
                'roc_auc': best['roc_auc'],
                'accuracy': best['accuracy'],
                'recall': best['recall'],
                'n_samples': best['n_samples']
            }

    # Compare to baseline (5% percentile)
    print()
    print("=" * 70)
    print("IMPROVEMENT OVER BASELINE (5% Percentile)")
    print("=" * 70)

    for cutoff in CUTOFF_WEEKS:
        baseline = next((e for e in results['experiments']
                        if e['percentile'] == 0.05 and e['cutoff_weeks'] == cutoff), None)
        best = results['summary'].get(f'week_{cutoff}')

        if baseline and best:
            delta_auc = best['roc_auc'] - baseline['roc_auc']
            delta_acc = (best['accuracy'] - baseline['accuracy']) * 100

            print(f"\nWeek {cutoff}:")
            print(f"  Baseline (5%): ROC-AUC={baseline['roc_auc']:.3f}, Acc={baseline['accuracy']*100:.1f}%")
            print(f"  Best ({int(best['best_percentile']*100)}%): ROC-AUC={best['roc_auc']:.3f}, Acc={best['accuracy']*100:.1f}%")
            print(f"  Delta: ROC-AUC {delta_auc:+.3f}, Acc {delta_acc:+.1f}pp")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
