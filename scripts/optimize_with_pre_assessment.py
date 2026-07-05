#!/usr/bin/env python3
"""
Comprehensive Optimization with Pre-Assessment Features

Tests all combinations of:
- Weeks: 2, 4, 6 (and full semester)
- Percentiles: 5%, 10%, 15%, 20%
- With/Without assessment features
- With/Without pre-assessment features (NEW!)
- Thresholds: 0.10 to 0.60

Compares against documented baselines from MODEL_RESULTS_REFERENCE.md:
- Week 2: ROC-AUC 0.743 (P5%, with assessment)
- Week 4 SIN assessment: ROC-AUC 0.741, Recall 80.4% (P20%)
- Week 6 CON assessment: ROC-AUC 0.822, Recall 86.2% (P10%)
- Full: ROC-AUC 0.902
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
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_XGBOOST = False

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
ASSIGNMENTS_FILE = BASE_DIR / "data/assignment_analytics.json"
OUTPUT_FILE = BASE_DIR / "data/analysis/pre_assessment_optimization_results.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Configuration
PERCENTILES = [0.05, 0.10, 0.15, 0.20]
CUTOFF_WEEKS = [2, 4, 6, 'full']
SESSION_GAP_MINUTES = 30

# Assessment-related feature patterns (for filtering)
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

# Documented baselines for comparison
BASELINES = {
    2: {'roc_auc': 0.743, 'recall': 0.817, 'config': 'P5%, with assessment'},
    4: {'roc_auc': 0.756, 'recall': 0.703, 'config': 'P20%, with assessment (best ROC)'},
    '4_no_asmt': {'roc_auc': 0.741, 'recall': 0.804, 'config': 'P20%, NO assessment'},
    6: {'roc_auc': 0.822, 'recall': 0.862, 'config': 'P10%, with assessment'},
    'full': {'roc_auc': 0.902, 'recall': 0.859, 'config': 'Full, with assessment'},
}


def normalize_user_id(user_id):
    """Normalize Canvas user_id to short format."""
    if user_id > 10000000000:
        return user_id % 10000000000
    return user_id


def load_data():
    """Load page views, enrollments, and assignments."""
    print("Loading data...")

    # Page views
    df = pd.read_parquet(PAGE_VIEWS_FILE)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
    df = df[df['course_id'].isin(COURSES)].copy()
    df['user_id'] = df['user_id'].apply(normalize_user_id)
    print(f"  Page views: {len(df):,}")

    # Enrollments
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    df_enroll['failed'] = (df_enroll['final_score'] < 57).astype(int)
    print(f"  Enrollments: {len(df_enroll):,}")

    # Assignments with due dates
    with open(ASSIGNMENTS_FILE) as f:
        assignments = json.load(f)
    assignments = [a for a in assignments
                   if a.get('course_id') in COURSES and a.get('due_at')]
    assignments_df = pd.DataFrame(assignments)
    if len(assignments_df) > 0:
        assignments_df['due_at'] = pd.to_datetime(assignments_df['due_at']).dt.tz_localize(None)
    print(f"  Assignments with due dates: {len(assignments_df)}")

    return df, df_enroll, assignments_df


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
    if cutoff_weeks == 'full':
        return df.copy()

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


def calculate_base_features(df, course_starts):
    """Calculate base features (sessions, categories, time, trajectory)."""
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
        for sid in range(1, min(n_sessions + 1, 100)):
            session_mask = session_ids == sid
            if session_mask.sum() > 1:
                session_ts = timestamps[session_mask]
                duration = (session_ts.max() - session_ts.min()).total_seconds() / 60
                session_durations.append(duration)

        if session_durations:
            features['avg_session_duration'] = np.mean(session_durations)
            features['total_session_time'] = np.sum(session_durations)
            features['session_duration_std'] = np.std(session_durations) if len(session_durations) > 1 else 0
        else:
            features['avg_session_duration'] = 0
            features['total_session_time'] = 0
            features['session_duration_std'] = 0

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

        features['pct_morning'] = sum((hours >= 6) & (hours < 12)) / total * 100 if total > 0 else 0
        features['pct_afternoon'] = sum((hours >= 12) & (hours < 18)) / total * 100 if total > 0 else 0
        features['pct_evening'] = sum((hours >= 18) & (hours < 24)) / total * 100 if total > 0 else 0
        features['pct_weekend'] = sum(days >= 5) / total * 100 if total > 0 else 0

        features['unique_hours'] = hours.nunique()
        features['unique_days'] = days.nunique()

        # === TEMPORAL TRAJECTORY ===
        weeks = ((timestamps - course_start).dt.days // 7 + 1).clip(lower=1)
        features['active_weeks'] = weeks.nunique()
        features['first_active_week'] = weeks.min()

        weekly_counts = weeks.value_counts().sort_index()
        if len(weekly_counts) >= 2:
            features['weekly_trend'] = np.polyfit(range(len(weekly_counts)), weekly_counts.values, 1)[0]
            features['weekly_std'] = weekly_counts.std()
        else:
            features['weekly_trend'] = 0
            features['weekly_std'] = 0

        # DCT coefficients
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
        features['unique_transitions'] = len(set(bigrams))

        # === PROACTIVITY FEATURES ===
        days_to_first = (timestamps.min() - course_start).days
        features['days_to_first_activity'] = max(days_to_first, 0)

        daily_counts = group.groupby(timestamps.dt.date).size()
        features['active_days'] = len(daily_counts)
        features['views_per_active_day'] = len(group) / max(len(daily_counts), 1)

        results.append(features)

    return pd.DataFrame(results)


def calculate_pre_assessment_features(df, assignments_df, course_starts, cutoff_weeks):
    """Calculate pre-assessment features with time cutoff awareness."""
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        if len(group) == 0:
            continue

        group = group.sort_values('created_at')
        timestamps = pd.to_datetime(group['created_at'])
        course_start = course_starts.get(course_id, timestamps.min())

        # Get data range for this cutoff
        if cutoff_weeks == 'full':
            cutoff_date = timestamps.max() + timedelta(days=365)  # No cutoff
        else:
            cutoff_date = course_start + timedelta(weeks=cutoff_weeks)

        features = {
            'user_id': user_id,
            'course_id': course_id,
        }

        # === DEADLINE-RELATIVE FEATURES ===
        course_assignments = assignments_df[
            (assignments_df['course_id'] == course_id) &
            (assignments_df['due_at'] <= cutoff_date)  # Only deadlines within cutoff
        ]

        activity_24h = 0
        activity_48h = 0
        activity_72h = 0
        files_72h = 0

        for _, assignment in course_assignments.iterrows():
            due_at = assignment['due_at']
            window_24h = (due_at - timedelta(hours=24), due_at)
            window_48h = (due_at - timedelta(hours=48), due_at)
            window_72h = (due_at - timedelta(hours=72), due_at)

            activity_24h += ((timestamps >= window_24h[0]) & (timestamps <= window_24h[1])).sum()
            activity_48h += ((timestamps >= window_48h[0]) & (timestamps <= window_48h[1])).sum()
            activity_72h += ((timestamps >= window_72h[0]) & (timestamps <= window_72h[1])).sum()

            # File accesses before deadline
            file_ts = pd.to_datetime(group[group['resource_type'] == 'files']['created_at'])
            if len(file_ts) > 0:
                files_72h += ((file_ts >= window_72h[0]) & (file_ts <= window_72h[1])).sum()

        n_deadlines = len(course_assignments)
        features['activity_24h_before'] = activity_24h
        features['activity_48h_before'] = activity_48h
        features['activity_72h_before'] = activity_72h
        features['files_72h_before'] = files_72h
        features['n_deadlines'] = n_deadlines

        if n_deadlines > 0:
            features['activity_per_deadline'] = activity_72h / n_deadlines
        else:
            features['activity_per_deadline'] = 0

        # Preparation intensity
        total_activity = len(group)
        features['preparation_intensity'] = activity_72h / total_activity if total_activity > 0 else 0

        # === QAT (Quiz Access Time) FEATURES ===
        quiz_views = group[group['resource_type'] == 'quizzes']
        if len(quiz_views) > 0:
            first_quiz = pd.to_datetime(quiz_views['created_at']).min()
            features['first_quiz_access_days'] = (first_quiz - course_start).days
            features['quiz_access_count'] = len(quiz_views)
            features['unique_quizzes'] = quiz_views['resource_id'].nunique()
            features['quiz_revisits'] = len(quiz_views) / max(quiz_views['resource_id'].nunique(), 1)
        else:
            features['first_quiz_access_days'] = 999  # Never accessed
            features['quiz_access_count'] = 0
            features['unique_quizzes'] = 0
            features['quiz_revisits'] = 0

        # === ASSIGNMENT ACCESS FEATURES ===
        assgn_views = group[group['resource_type'] == 'assignments']
        if len(assgn_views) > 0:
            first_assgn = pd.to_datetime(assgn_views['created_at']).min()
            features['first_assgn_access_days'] = (first_assgn - course_start).days
            features['assgn_access_count'] = len(assgn_views)
            features['unique_assignments'] = assgn_views['resource_id'].nunique()
            features['assgn_revisits'] = len(assgn_views) / max(assgn_views['resource_id'].nunique(), 1)
        else:
            features['first_assgn_access_days'] = 999
            features['assgn_access_count'] = 0
            features['unique_assignments'] = 0
            features['assgn_revisits'] = 0

        # Assessment diversity
        features['assessment_diversity'] = features['unique_quizzes'] + features['unique_assignments']

        # === TEMPORAL PREPARATION PATTERNS ===
        if cutoff_weeks != 'full' and cutoff_weeks > 0:
            cutoff_days = cutoff_weeks * 7
            midpoint = course_start + timedelta(days=cutoff_days / 2)
            early_activity = (timestamps <= midpoint).sum()
            features['early_half_pct'] = early_activity / total_activity if total_activity > 0 else 0
        else:
            course_end = timestamps.max()
            course_duration = (course_end - course_start).days
            midpoint = course_start + timedelta(days=course_duration / 2)
            early_activity = (timestamps <= midpoint).sum()
            features['early_half_pct'] = early_activity / total_activity if total_activity > 0 else 0

        # === FILE PREPARATION PATTERNS ===
        file_views = group[group['resource_type'] == 'files']
        features['files_total'] = len(file_views)
        features['files_diversity'] = file_views['resource_id'].nunique() if len(file_views) > 0 else 0

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

    # Youden's J
    youden_j = recall + specificity - 1

    return {
        'threshold': float(threshold),
        'recall': float(recall),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'f1': float(f1),
        'f2': float(f2),
        'youden_j': float(youden_j),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal thresholds."""
    thresholds = np.arange(0.10, 0.61, 0.01)
    results = [calculate_metrics_at_threshold(y_true, y_pred_proba, t) for t in thresholds]
    df_results = pd.DataFrame(results)

    optimal = {}

    # Max F2 (balanced toward recall)
    idx = df_results['f2'].idxmax()
    optimal['max_f2'] = df_results.loc[idx].to_dict()

    # Max Youden's J (statistical optimum)
    idx = df_results['youden_j'].idxmax()
    optimal['max_youden'] = df_results.loc[idx].to_dict()

    # Recall >= 80% with max accuracy
    subset = df_results[df_results['recall'] >= 0.80]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['recall_80'] = subset.loc[idx].to_dict()

    # Recall >= 85% with max accuracy
    subset = df_results[df_results['recall'] >= 0.85]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['recall_85'] = subset.loc[idx].to_dict()

    return optimal


def run_experiment(df_filtered, df_enroll, assignments_df, course_starts, cutoff_weeks,
                   include_assessment=True, include_pre_assessment=True):
    """Run a single experiment."""
    # Calculate base features
    df_base = calculate_base_features(df_filtered, course_starts)

    if len(df_base) < 50:
        return None

    # Calculate pre-assessment features if requested
    if include_pre_assessment:
        df_pre = calculate_pre_assessment_features(df_filtered, assignments_df, course_starts, cutoff_weeks)
        df_features = df_base.merge(df_pre, on=['user_id', 'course_id'], how='inner')
    else:
        df_features = df_base

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

    # Get feature columns
    exclude_cols = ['user_id', 'course_id', 'failed']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Filter assessment features if requested
    if not include_assessment:
        feature_cols = filter_assessment_features(feature_cols)

    # Prepare data
    X = df[feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
    y = df['failed'].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    if HAS_XGBOOST:
        model = XGBClassifier(**XGBOOST_PARAMS)
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y, y_pred_proba)

    # Find optimal thresholds
    optimal = find_optimal_thresholds(y, y_pred_proba)

    # Get feature importances
    model.fit(X_scaled, y)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:10]

    return {
        'roc_auc': float(roc_auc),
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'failure_rate': float(y.mean()),
        'optimal_thresholds': optimal,
        'top_features': [{'name': f, 'importance': float(i)} for f, i in top_features],
    }


def main():
    print("=" * 70)
    print("COMPREHENSIVE OPTIMIZATION WITH PRE-ASSESSMENT FEATURES")
    print("=" * 70)
    print()

    # Load data
    df, df_enroll, assignments_df = load_data()
    print()

    # Results storage
    all_results = []
    summary = []

    # Test all combinations
    total_experiments = len(CUTOFF_WEEKS) * len(PERCENTILES) * 4  # 4 = combinations of assessment + pre-assessment
    exp_num = 0

    for cutoff_weeks in CUTOFF_WEEKS:
        for percentile in PERCENTILES:
            # Get course starts and filter data
            course_starts = get_course_starts(df, percentile)
            df_filtered = filter_by_cutoff(df, course_starts, cutoff_weeks)

            if len(df_filtered) == 0:
                continue

            for include_assessment in [True, False]:
                for include_pre_assessment in [True, False]:
                    exp_num += 1
                    print(f"\n[{exp_num}/{total_experiments}] Week {cutoff_weeks}, P{int(percentile*100)}%, "
                          f"Assessment={include_assessment}, PreAssmt={include_pre_assessment}")

                    result = run_experiment(
                        df_filtered, df_enroll, assignments_df, course_starts, cutoff_weeks,
                        include_assessment, include_pre_assessment
                    )

                    if result is None:
                        print("  Skipped (insufficient data)")
                        continue

                    # Store config
                    config = {
                        'week': str(cutoff_weeks),
                        'percentile': percentile,
                        'include_assessment': include_assessment,
                        'include_pre_assessment': include_pre_assessment,
                    }
                    result['config'] = config

                    all_results.append(result)

                    # Print summary
                    opt = result['optimal_thresholds']
                    best = opt.get('max_f2', opt.get('max_youden', {}))
                    print(f"  ROC-AUC: {result['roc_auc']:.4f}, "
                          f"Samples: {result['n_samples']}, "
                          f"Best Recall: {best.get('recall', 0):.3f}")

                    # Track for summary
                    summary.append({
                        'week': str(cutoff_weeks),
                        'percentile': int(percentile * 100),
                        'assessment': include_assessment,
                        'pre_assessment': include_pre_assessment,
                        'roc_auc': result['roc_auc'],
                        'recall_at_f2': best.get('recall', 0),
                        'accuracy_at_f2': best.get('accuracy', 0),
                        'threshold': best.get('threshold', 0.5),
                    })

    # Print summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: IMPACT OF PRE-ASSESSMENT FEATURES")
    print("=" * 70)

    summary_df = pd.DataFrame(summary)

    for week in CUTOFF_WEEKS:
        week_df = summary_df[summary_df['week'] == str(week)]
        if len(week_df) == 0:
            continue

        print(f"\n--- WEEK {week} ---")

        # Find best for each configuration
        for asmt in [True, False]:
            asmt_label = "WITH" if asmt else "WITHOUT"

            base = week_df[(week_df['assessment'] == asmt) & (week_df['pre_assessment'] == False)]
            enhanced = week_df[(week_df['assessment'] == asmt) & (week_df['pre_assessment'] == True)]

            if len(base) == 0 or len(enhanced) == 0:
                continue

            best_base = base.loc[base['roc_auc'].idxmax()]
            best_enhanced = enhanced.loc[enhanced['roc_auc'].idxmax()]

            roc_diff = best_enhanced['roc_auc'] - best_base['roc_auc']
            recall_diff = best_enhanced['recall_at_f2'] - best_base['recall_at_f2']

            print(f"\n  {asmt_label} Assessment Features:")
            print(f"    Baseline (no pre-asmt):  ROC={best_base['roc_auc']:.4f}, "
                  f"Recall={best_base['recall_at_f2']:.3f}, P{int(best_base['percentile'])}%")
            print(f"    + Pre-Assessment:        ROC={best_enhanced['roc_auc']:.4f}, "
                  f"Recall={best_enhanced['recall_at_f2']:.3f}, P{int(best_enhanced['percentile'])}%")
            print(f"    Improvement:             ROC {roc_diff:+.4f}, Recall {recall_diff:+.3f}")

    # Compare against documented baselines
    print("\n" + "=" * 70)
    print("COMPARISON VS DOCUMENTED BASELINES")
    print("=" * 70)

    for week_key, baseline in BASELINES.items():
        week = str(week_key).replace('_no_asmt', '')
        week_df = summary_df[summary_df['week'] == week]

        if len(week_df) == 0:
            continue

        # Find best with pre-assessment
        best = week_df[week_df['pre_assessment'] == True]
        if len(best) == 0:
            continue

        best = best.loc[best['roc_auc'].idxmax()]

        roc_diff = best['roc_auc'] - baseline['roc_auc']

        print(f"\n  Week {week_key}:")
        print(f"    Baseline ({baseline['config']}): ROC={baseline['roc_auc']:.3f}")
        print(f"    Best + Pre-Assessment:          ROC={best['roc_auc']:.4f} ({roc_diff:+.4f})")

        if roc_diff > 0.01:
            print(f"    ✓ IMPROVED by {roc_diff:.4f}")
        elif roc_diff < -0.01:
            print(f"    ✗ Decreased by {abs(roc_diff):.4f}")
        else:
            print(f"    ≈ Minimal change ({roc_diff:+.4f})")

    # Save results
    output = {
        'all_results': all_results,
        'summary': summary,
        'baselines': BASELINES,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n\nResults saved to {OUTPUT_FILE}")

    # Find absolute best configuration
    print("\n" + "=" * 70)
    print("BEST CONFIGURATIONS WITH PRE-ASSESSMENT FEATURES")
    print("=" * 70)

    for week in CUTOFF_WEEKS:
        week_df = summary_df[(summary_df['week'] == str(week)) & (summary_df['pre_assessment'] == True)]
        if len(week_df) == 0:
            continue

        best = week_df.loc[week_df['roc_auc'].idxmax()]
        print(f"\n  Week {week}:")
        print(f"    Best ROC-AUC: {best['roc_auc']:.4f}")
        print(f"    Config: P{int(best['percentile'])}%, Assessment={best['assessment']}")
        print(f"    Recall: {best['recall_at_f2']:.3f}, Accuracy: {best['accuracy_at_f2']:.3f}")


if __name__ == '__main__':
    main()
