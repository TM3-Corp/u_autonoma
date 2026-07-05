#!/usr/bin/env python3
"""
Comprehensive Week 4 & 6 Optimization

Tests all combinations of:
- Percentiles: 5%, 10%, 15%, 20%
- Thresholds: 0.10 to 0.60
- With/Without assessment features
- With feature selection

Finds the absolute best configuration for each week.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import timedelta
from scipy.fftpack import dct
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/analysis/comprehensive_optimization_results.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Configuration
PERCENTILES = [0.05, 0.10, 0.15, 0.20]
CUTOFF_WEEKS = [4, 6]
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

# Feature selection parameters
IMPORTANCE_THRESHOLD = 0.005
CORRELATION_THRESHOLD = 0.85


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
        for sid in range(1, min(n_sessions + 1, 100)):  # Limit for performance
            session_mask = session_ids == sid
            if session_mask.sum() > 1:
                session_ts = timestamps[session_mask]
                duration = (session_ts.max() - session_ts.min()).total_seconds() / 60
                session_durations.append(duration)

        if session_durations:
            features['avg_session_duration'] = np.mean(session_durations)
            features['max_session_duration'] = np.max(session_durations)
            features['total_session_time'] = np.sum(session_durations)
            features['session_duration_std'] = np.std(session_durations) if len(session_durations) > 1 else 0
        else:
            features['avg_session_duration'] = 0
            features['max_session_duration'] = 0
            features['total_session_time'] = 0
            features['session_duration_std'] = 0

        # === CATEGORY FEATURES ===
        categories = ['files', 'discussions', 'quizzes', 'assignments', 'modules', 'grades', 'pages', 'announcements']
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

        # Unique hours and days
        features['unique_hours'] = hours.nunique()
        features['unique_days'] = days.nunique()

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
            features['weekly_std'] = weekly_counts.std()
            features['weekly_mean'] = weekly_counts.mean()
        else:
            features['weekly_trend'] = 0
            features['weekly_std'] = 0
            features['weekly_mean'] = weekly_counts.mean() if len(weekly_counts) > 0 else 0

        # DCT coefficients for trajectory
        if len(weekly_counts) >= 4:
            padded = np.zeros(8)
            padded[:min(len(weekly_counts), 8)] = weekly_counts.values[:8]
            dct_coeffs = dct(padded, norm='ortho')
            features['dct_0'] = dct_coeffs[0]
            features['dct_1'] = dct_coeffs[1]
            features['dct_2'] = dct_coeffs[2]
            features['dct_3'] = dct_coeffs[3] if len(dct_coeffs) > 3 else 0
        else:
            features['dct_0'] = 0
            features['dct_1'] = 0
            features['dct_2'] = 0
            features['dct_3'] = 0

        # === NAVIGATION FEATURES ===
        types = group['resource_type'].tolist()
        bigrams = [(types[i], types[i+1]) for i in range(len(types)-1)]
        features['total_transitions'] = len(bigrams)
        features['unique_transitions'] = len(set(bigrams))
        features['transition_diversity'] = len(set(bigrams)) / max(len(bigrams), 1)

        # Specific bigrams (assessment-related)
        if bigrams:
            bigram_counts = pd.Series(bigrams).value_counts(normalize=True)
            for bg in [('assignments', 'assignments'), ('assignments', 'grades'),
                       ('grades', 'assignments'), ('quizzes', 'quizzes')]:
                features[f'bigram_{bg[0]}_to_{bg[1]}'] = bigram_counts.get(bg, 0)

        # === PROACTIVITY FEATURES ===
        days_to_first = (timestamps.min() - course_start).days
        features['days_to_first_activity'] = max(days_to_first, 0)

        # Activity consistency
        daily_counts = group.groupby(timestamps.dt.date).size()
        features['daily_consistency'] = 1 / (1 + daily_counts.std()) if len(daily_counts) > 1 else 1
        features['active_days'] = len(daily_counts)
        features['views_per_active_day'] = len(group) / max(len(daily_counts), 1)

        # Gap analysis
        if len(gaps) > 1:
            valid_gaps = gaps[gaps > 0]
            if len(valid_gaps) > 0:
                features['mean_gap_hours'] = valid_gaps.mean() / 60
                features['max_gap_hours'] = valid_gaps.max() / 60
                features['gap_std_hours'] = valid_gaps.std() / 60 if len(valid_gaps) > 1 else 0

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


def select_features_by_importance(X, y, feature_cols, threshold=0.005):
    """Select features with importance above threshold."""
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_))
    selected = [f for f, imp in importances.items() if imp >= threshold]
    return selected, importances


def remove_correlated_features(X, feature_cols, threshold=0.85):
    """Remove highly correlated features."""
    if len(feature_cols) <= 1:
        return feature_cols
    corr_matrix = X[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [f for f in feature_cols if f not in to_drop]
    return selected


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
    """Find optimal thresholds for various targets."""
    thresholds = np.arange(0.10, 0.61, 0.01)
    results = [calculate_metrics_at_threshold(y_true, y_pred_proba, t) for t in thresholds]
    df_results = pd.DataFrame(results)

    optimal = {}

    # Max Accuracy
    idx = df_results['accuracy'].idxmax()
    optimal['max_accuracy'] = df_results.loc[idx].to_dict()

    # Max F2
    idx = df_results['f2'].idxmax()
    optimal['max_f2'] = df_results.loc[idx].to_dict()

    # Recall >= 70% with max accuracy
    subset = df_results[df_results['recall'] >= 0.70]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['recall_70'] = subset.loc[idx].to_dict()

    # Recall >= 75% with max accuracy
    subset = df_results[df_results['recall'] >= 0.75]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['recall_75'] = subset.loc[idx].to_dict()

    # Recall >= 80% with max accuracy
    subset = df_results[df_results['recall'] >= 0.80]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['recall_80'] = subset.loc[idx].to_dict()

    return optimal, df_results


def run_experiment(df_filtered, df_enroll, course_starts, include_assessment=True):
    """Run a single experiment with feature selection."""
    # Calculate features
    df_features = calculate_comprehensive_features(df_filtered, course_starts)

    if len(df_features) < 50:
        return None

    # Add z-normalization
    df_features = calculate_znorm_features(df_features)

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

    if len(numeric_cols) < 5:
        return None

    # Feature selection
    selected_by_importance, importances = select_features_by_importance(X, y, numeric_cols)
    if len(selected_by_importance) < 5:
        selected_by_importance = numeric_cols[:20]  # Fallback

    selected_final = remove_correlated_features(X, selected_by_importance)
    if len(selected_final) < 5:
        selected_final = selected_by_importance[:20]

    X = X[selected_final]

    # Train with CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)

    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)

        # Find optimal thresholds
        optimal, df_thresholds = find_optimal_thresholds(y, y_pred_proba)

        # Get feature importances for final model
        model.fit(X, y)
        final_importances = dict(zip(selected_final, [float(x) for x in model.feature_importances_]))
        top_features = dict(sorted(final_importances.items(), key=lambda x: -x[1])[:10])

        return {
            'n_samples': int(len(X)),
            'n_features_initial': len(numeric_cols),
            'n_features_selected': len(selected_final),
            'failure_rate': float(y.mean()),
            'roc_auc': float(roc_auc),
            'thresholds': optimal,
            'top_features': top_features,
            'all_thresholds': df_thresholds.to_dict('records')
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


def main():
    print("=" * 80)
    print("COMPREHENSIVE OPTIMIZATION: Weeks 4 & 6")
    print("=" * 80)
    print(f"\nTesting:")
    print(f"  Percentiles: {[int(p*100) for p in PERCENTILES]}%")
    print(f"  Weeks: {CUTOFF_WEEKS}")
    print(f"  Variants: WITH and WITHOUT assessment features")
    print()

    df_full, df_enroll = load_data()

    all_results = {
        'experiments': [],
        'best_per_week': {}
    }

    # Run all experiments
    for cutoff in CUTOFF_WEEKS:
        print(f"\n{'='*80}")
        print(f"WEEK {cutoff}")
        print(f"{'='*80}")

        week_results = []

        for percentile in PERCENTILES:
            print(f"\n  Percentile {int(percentile*100)}%:")

            # Get course starts
            course_starts = get_course_starts(df_full, percentile)

            # Filter data
            df_filtered = filter_by_cutoff(df_full, course_starts, cutoff)
            print(f"    Page views: {len(df_filtered):,}")

            for include_assessment in [True, False]:
                label = f"week_{cutoff}_p{int(percentile*100)}_{'with' if include_assessment else 'without'}_assessment"
                variant = "WITH" if include_assessment else "WITHOUT"

                result = run_experiment(df_filtered, df_enroll, course_starts, include_assessment)

                if result:
                    result['label'] = label
                    result['cutoff'] = cutoff
                    result['percentile'] = percentile
                    result['include_assessment'] = include_assessment

                    all_results['experiments'].append(result)
                    week_results.append(result)

                    # Show key metrics
                    r70 = result['thresholds'].get('recall_70', result['thresholds']['max_f2'])
                    print(f"      {variant:7} assessment: ROC-AUC={result['roc_auc']:.3f}, "
                          f"t={r70['threshold']:.2f} → Acc={r70['accuracy']*100:.1f}%, Rec={r70['recall']*100:.1f}%")

        # Find best for this week
        if week_results:
            # Best by ROC-AUC
            best_auc = max(week_results, key=lambda x: x['roc_auc'])

            # Best by Recall≥70% Accuracy
            best_r70 = None
            best_r70_acc = 0
            for r in week_results:
                if 'recall_70' in r['thresholds']:
                    if r['thresholds']['recall_70']['accuracy'] > best_r70_acc:
                        best_r70_acc = r['thresholds']['recall_70']['accuracy']
                        best_r70 = r

            all_results['best_per_week'][f'week_{cutoff}'] = {
                'best_by_auc': best_auc['label'],
                'best_by_r70_accuracy': best_r70['label'] if best_r70 else None
            }

    # Print comprehensive summary
    print("\n" + "=" * 100)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 100)

    for cutoff in CUTOFF_WEEKS:
        print(f"\n{'─'*100}")
        print(f"WEEK {cutoff}")
        print(f"{'─'*100}")
        print(f"\n{'Config':<45} {'ROC-AUC':>8} {'Thr':>6} {'Acc':>8} {'Recall':>8} {'Prec':>8} {'Features':>10}")
        print("-" * 100)

        week_exps = [e for e in all_results['experiments'] if e['cutoff'] == cutoff]
        week_exps_sorted = sorted(week_exps, key=lambda x: -x['roc_auc'])

        for exp in week_exps_sorted:
            r70 = exp['thresholds'].get('recall_70', exp['thresholds']['max_f2'])
            p_str = f"P{int(exp['percentile']*100)}%"
            a_str = "WITH" if exp['include_assessment'] else "WITHOUT"
            config = f"{p_str} {a_str} assessment"

            print(f"{config:<45} {exp['roc_auc']:>8.3f} {r70['threshold']:>6.2f} "
                  f"{r70['accuracy']*100:>7.1f}% {r70['recall']*100:>7.1f}% "
                  f"{r70['precision']*100:>7.1f}% {exp['n_features_selected']:>10}")

    # Best configurations
    print("\n" + "=" * 100)
    print("BEST CONFIGURATIONS (Recall ≥ 70%)")
    print("=" * 100)

    for cutoff in CUTOFF_WEEKS:
        week_exps = [e for e in all_results['experiments'] if e['cutoff'] == cutoff]

        # Find best WITH assessment
        with_assessment = [e for e in week_exps if e['include_assessment']]
        if with_assessment:
            best_with = max(with_assessment, key=lambda x: x['thresholds'].get('recall_70', {'accuracy': 0})['accuracy']
                          if 'recall_70' in x['thresholds'] else 0)
            r70 = best_with['thresholds'].get('recall_70', best_with['thresholds']['max_f2'])
            print(f"\nWeek {cutoff} WITH assessment:")
            print(f"  Percentile: {int(best_with['percentile']*100)}%")
            print(f"  ROC-AUC: {best_with['roc_auc']:.3f}")
            print(f"  Threshold: {r70['threshold']:.2f}")
            print(f"  Accuracy: {r70['accuracy']*100:.1f}%")
            print(f"  Recall: {r70['recall']*100:.1f}%")
            print(f"  Top features: {list(best_with['top_features'].keys())[:5]}")

        # Find best WITHOUT assessment
        without_assessment = [e for e in week_exps if not e['include_assessment']]
        if without_assessment:
            best_without = max(without_assessment, key=lambda x: x['thresholds'].get('recall_70', {'accuracy': 0})['accuracy']
                              if 'recall_70' in x['thresholds'] else 0)
            r70 = best_without['thresholds'].get('recall_70', best_without['thresholds']['max_f2'])
            print(f"\nWeek {cutoff} WITHOUT assessment:")
            print(f"  Percentile: {int(best_without['percentile']*100)}%")
            print(f"  ROC-AUC: {best_without['roc_auc']:.3f}")
            print(f"  Threshold: {r70['threshold']:.2f}")
            print(f"  Accuracy: {r70['accuracy']*100:.1f}%")
            print(f"  Recall: {r70['recall']*100:.1f}%")
            print(f"  Top features: {list(best_without['top_features'].keys())[:5]}")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Remove all_thresholds to reduce file size
    for exp in all_results['experiments']:
        if 'all_thresholds' in exp:
            del exp['all_thresholds']

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
