#!/usr/bin/env python3
"""
Train Optimal Early Warning Model

Combines all optimizations:
- Configurable percentile for course start (default: 20%)
- Full feature engineering including z-normalization
- Threshold optimization for F2 score
- Tests WITH and WITHOUT assessment features

Output:
    data/analysis/optimal_early_model_results.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, recall_score, precision_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/analysis/optimal_early_model_results.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Configuration
START_PERCENTILE = 0.20  # Use 20% percentile based on experiment results
CUTOFF_WEEKS = [2, 4, 6]  # Include week 2 for complete analysis
SESSION_GAP_MINUTES = 30

# Assessment-related feature patterns
ASSESSMENT_PATTERNS = [
    'quiz', 'quizzes', 'assi', 'assignment',
    'grade', 'grad', 'score', 'submission'
]

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
    print(f"  Loaded {len(df_enroll):,} enrollments")
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


def calculate_comprehensive_features(df, course_starts):
    """Calculate comprehensive features matching the production pipeline."""
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        if len(group) < 2:
            continue

        group = group.sort_values('created_at')
        timestamps = pd.to_datetime(group['created_at'])
        course_start = course_starts.get(course_id, timestamps.min())

        # === SESSION FEATURES ===
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
            'views_per_session': len(group) / max(n_sessions, 1),
        }

        # Session durations
        session_durations = []
        for sid in range(1, n_sessions + 1):
            session_mask = session_ids == sid
            if session_mask.sum() > 1:
                session_ts = timestamps[session_mask]
                duration = (session_ts.max() - session_ts.min()).total_seconds() / 60
                session_durations.append(duration)

        if session_durations:
            features['avg_session_duration'] = np.mean(session_durations)
            features['max_session_duration'] = np.max(session_durations)
            features['total_session_time'] = np.sum(session_durations)
        else:
            features['avg_session_duration'] = 0
            features['max_session_duration'] = 0
            features['total_session_time'] = 0

        # === CATEGORY FEATURES ===
        categories = ['files', 'discussions', 'quizzes', 'assignments', 'modules', 'grades', 'pages']
        for cat in categories:
            cat_data = group[group['resource_type'] == cat]
            features[f'{cat}_views'] = len(cat_data)
            features[f'{cat}_unique'] = cat_data['resource_id'].nunique() if 'resource_id' in cat_data.columns else 0
            features[f'{cat}_pct'] = len(cat_data) / len(group) * 100 if len(group) > 0 else 0

        # === TIME FEATURES ===
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofweek
        total = len(hours)

        # Time blocks
        features['pct_morning'] = sum((hours >= 6) & (hours < 12)) / total * 100 if total > 0 else 0
        features['pct_afternoon'] = sum((hours >= 12) & (hours < 18)) / total * 100 if total > 0 else 0
        features['pct_evening'] = sum((hours >= 18) & (hours < 24)) / total * 100 if total > 0 else 0
        features['pct_night'] = sum((hours >= 0) & (hours < 6)) / total * 100 if total > 0 else 0
        features['pct_weekend'] = sum(days >= 5) / total * 100 if total > 0 else 0

        # Hour entropy
        hour_counts = hours.value_counts(normalize=True)
        features['hour_entropy'] = -sum(p * np.log2(p + 1e-10) for p in hour_counts)

        # Day entropy
        day_counts = days.value_counts(normalize=True)
        features['day_entropy'] = -sum(p * np.log2(p + 1e-10) for p in day_counts)

        # === TEMPORAL TRAJECTORY FEATURES ===
        weeks = ((timestamps - course_start).dt.days // 7 + 1).clip(lower=1)
        features['active_weeks'] = weeks.nunique()
        features['first_active_week'] = weeks.min()
        features['last_active_week'] = weeks.max()
        features['week_span'] = weeks.max() - weeks.min() + 1

        # Weekly activity trajectory
        weekly_counts = weeks.value_counts().sort_index()
        if len(weekly_counts) >= 2:
            features['weekly_trend'] = np.polyfit(range(len(weekly_counts)), weekly_counts.values, 1)[0]
        else:
            features['weekly_trend'] = 0

        # DCT coefficients for trajectory
        if len(weekly_counts) >= 4:
            padded = np.zeros(8)
            padded[:min(len(weekly_counts), 8)] = weekly_counts.values[:8]
            dct_coeffs = dct(padded, norm='ortho')
            features['dct_0'] = dct_coeffs[0]
            features['dct_1'] = dct_coeffs[1]
            features['dct_2'] = dct_coeffs[2]
        else:
            features['dct_0'] = 0
            features['dct_1'] = 0
            features['dct_2'] = 0

        # === NAVIGATION FEATURES ===
        types = group['resource_type'].tolist()
        bigrams = [(types[i], types[i+1]) for i in range(len(types)-1)]
        features['total_transitions'] = len(bigrams)
        features['unique_transitions'] = len(set(bigrams))
        features['transition_diversity'] = len(set(bigrams)) / max(len(bigrams), 1)

        # === PROACTIVITY FEATURES ===
        # Days until first activity
        days_to_first = (timestamps.min() - course_start).days
        features['days_to_first_activity'] = max(days_to_first, 0)

        # Activity consistency (std of daily views)
        daily_counts = group.groupby(timestamps.dt.date).size()
        features['daily_consistency'] = 1 / (1 + daily_counts.std()) if len(daily_counts) > 1 else 1

        # Active days
        features['active_days'] = len(daily_counts)
        features['views_per_active_day'] = len(group) / max(len(daily_counts), 1)

        results.append(features)

    return pd.DataFrame(results)


def calculate_znorm_features(df):
    """Add z-score normalized features per course."""
    exclude_cols = ['user_id', 'course_id']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.endswith('_znorm')]

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


def filter_assessment_features(columns):
    """Filter out assessment-related features."""
    filtered = []
    for col in columns:
        col_lower = col.lower()
        is_excluded = any(pattern in col_lower for pattern in ASSESSMENT_PATTERNS)
        if not is_excluded:
            filtered.append(col)
    return filtered


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Calculate all metrics at a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    return {
        'threshold': float(threshold),
        'recall': float(recall),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'f1': float(f1),
        'f2': float(f2),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal thresholds by various criteria."""
    thresholds = np.arange(0.10, 0.71, 0.01)
    results = [calculate_metrics_at_threshold(y_true, y_pred_proba, t) for t in thresholds]
    df_results = pd.DataFrame(results)

    # Baseline (0.50)
    baseline_mask = df_results['threshold'].round(2) == 0.50
    baseline = df_results[baseline_mask].iloc[0].to_dict() if baseline_mask.any() else None

    # Max F2 (prioritizes recall)
    idx_f2 = df_results['f2'].idxmax()
    best_f2 = df_results.loc[idx_f2].to_dict()

    # Best accuracy with recall >= 70%
    recall_70 = df_results[df_results['recall'] >= 0.70]
    best_r70_acc = recall_70.loc[recall_70['accuracy'].idxmax()].to_dict() if len(recall_70) > 0 else None

    # Best accuracy with recall >= 80%
    recall_80 = df_results[df_results['recall'] >= 0.80]
    best_r80_acc = recall_80.loc[recall_80['accuracy'].idxmax()].to_dict() if len(recall_80) > 0 else None

    return {
        'baseline_0.50': baseline,
        'max_f2': best_f2,
        'recall_70_best_acc': best_r70_acc,
        'recall_80_best_acc': best_r80_acc
    }


def train_and_evaluate(df_features, df_enroll, include_assessment=True):
    """Train XGBoost with comprehensive evaluation."""
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

    if not include_assessment:
        feature_cols = filter_assessment_features(feature_cols)

    X = df[feature_cols].copy()
    y = df['failed'].values

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Keep only numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_cols = numeric_cols

    if len(feature_cols) < 5:
        return None

    # Train with CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)

    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Find optimal thresholds
        optimal = find_optimal_thresholds(y, y_pred_proba)

        # Get feature importances
        model.fit(X, y)
        importances = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
        top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:15])

        return {
            'n_samples': int(len(X)),
            'n_features': len(feature_cols),
            'failure_rate': float(y.mean()),
            'roc_auc': float(roc_auc),
            'thresholds': optimal,
            'top_features': top_features,
            'feature_cols': feature_cols
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    print("=" * 70)
    print("OPTIMAL EARLY WARNING MODEL TRAINING")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Start Percentile: {int(START_PERCENTILE*100)}%")
    print(f"  Cutoff Weeks: {CUTOFF_WEEKS}")
    print()

    df_full, df_enroll = load_data()

    # Get course starts with optimal percentile
    course_starts = get_course_starts(df_full, START_PERCENTILE)

    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'start_percentile': START_PERCENTILE,
            'cutoff_weeks': CUTOFF_WEEKS,
            'session_gap_minutes': SESSION_GAP_MINUTES
        },
        'models': []
    }

    print("Training models...")
    print("-" * 70)

    for cutoff in CUTOFF_WEEKS:
        print(f"\n=== WEEK {cutoff} ===")

        # Filter data
        df_filtered = filter_by_cutoff(df_full, course_starts, cutoff)
        print(f"  Page views: {len(df_filtered):,}")

        # Calculate features
        df_features = calculate_comprehensive_features(df_filtered, course_starts)
        print(f"  Students with features: {len(df_features)}")

        # Add z-normalization
        df_features = calculate_znorm_features(df_features)

        # Train WITH assessment features
        print("\n  Training WITH assessment features...")
        result_with = train_and_evaluate(df_features, df_enroll, include_assessment=True)

        if result_with:
            model_info = {
                'cutoff_weeks': cutoff,
                'include_assessment': True,
                'label': f'week_{cutoff}_with_assessment',
                **result_with
            }
            results['models'].append(model_info)

            baseline = result_with['thresholds']['baseline_0.50']
            best_f2 = result_with['thresholds']['max_f2']
            best_r70 = result_with['thresholds'].get('recall_70_best_acc')

            print(f"    ROC-AUC: {result_with['roc_auc']:.3f}")
            print(f"    Baseline (t=0.50): Acc={baseline['accuracy']*100:.1f}%, Recall={baseline['recall']*100:.1f}%")
            print(f"    Max F2 (t={best_f2['threshold']:.2f}): Acc={best_f2['accuracy']*100:.1f}%, Recall={best_f2['recall']*100:.1f}%")
            if best_r70:
                print(f"    R≥70% Best Acc (t={best_r70['threshold']:.2f}): Acc={best_r70['accuracy']*100:.1f}%, Recall={best_r70['recall']*100:.1f}%")

        # Train WITHOUT assessment features
        print("\n  Training WITHOUT assessment features...")
        result_without = train_and_evaluate(df_features, df_enroll, include_assessment=False)

        if result_without:
            model_info = {
                'cutoff_weeks': cutoff,
                'include_assessment': False,
                'label': f'week_{cutoff}_without_assessment',
                **result_without
            }
            results['models'].append(model_info)

            baseline = result_without['thresholds']['baseline_0.50']
            best_f2 = result_without['thresholds']['max_f2']
            best_r70 = result_without['thresholds'].get('recall_70_best_acc')

            print(f"    ROC-AUC: {result_without['roc_auc']:.3f}")
            print(f"    Baseline (t=0.50): Acc={baseline['accuracy']*100:.1f}%, Recall={baseline['recall']*100:.1f}%")
            print(f"    Max F2 (t={best_f2['threshold']:.2f}): Acc={best_f2['accuracy']*100:.1f}%, Recall={best_f2['recall']*100:.1f}%")
            if best_r70:
                print(f"    R≥70% Best Acc (t={best_r70['threshold']:.2f}): Acc={best_r70['accuracy']*100:.1f}%, Recall={best_r70['recall']*100:.1f}%")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: OPTIMAL EARLY MODELS (Percentile 20%)")
    print("=" * 100)
    print()
    print(f"{'Model':<40} {'ROC-AUC':>10} {'Acc (0.50)':>12} {'Rec (0.50)':>12} {'Best Thr':>10} {'Best Acc':>10} {'Best Rec':>10}")
    print("-" * 100)

    for model in results['models']:
        baseline = model['thresholds']['baseline_0.50']
        best_r70 = model['thresholds'].get('recall_70_best_acc') or model['thresholds']['max_f2']

        print(f"{model['label']:<40} {model['roc_auc']:>10.3f} "
              f"{baseline['accuracy']*100:>11.1f}% {baseline['recall']*100:>11.1f}% "
              f"{best_r70['threshold']:>10.2f} {best_r70['accuracy']*100:>9.1f}% {best_r70['recall']*100:>9.1f}%")

    # Find best model for each week
    print("\n" + "=" * 70)
    print("RECOMMENDED CONFIGURATIONS")
    print("=" * 70)

    for cutoff in CUTOFF_WEEKS:
        models = [m for m in results['models'] if m['cutoff_weeks'] == cutoff]
        if models:
            best = max(models, key=lambda x: x['roc_auc'])
            best_r70 = best['thresholds'].get('recall_70_best_acc') or best['thresholds']['max_f2']

            print(f"\nWeek {cutoff}:")
            print(f"  Best config: {'WITH' if best['include_assessment'] else 'WITHOUT'} assessment features")
            print(f"  ROC-AUC: {best['roc_auc']:.3f}")
            print(f"  Recommended threshold: {best_r70['threshold']:.2f}")
            print(f"  Expected performance: Accuracy {best_r70['accuracy']*100:.1f}%, Recall {best_r70['recall']*100:.1f}%")
            print(f"  Top 5 features:")
            for i, (feat, imp) in enumerate(list(best['top_features'].items())[:5]):
                print(f"    {i+1}. {feat}: {imp:.3f}")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Remove feature_cols from output (too large)
    for model in results['models']:
        if 'feature_cols' in model:
            del model['feature_cols']

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
