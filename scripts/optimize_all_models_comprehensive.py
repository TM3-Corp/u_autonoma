#!/usr/bin/env python3
"""
Comprehensive Threshold Optimization for ALL Time-Limited Models

Tests all combinations of:
- Cutoff weeks: 2, 4, 6, 8, full
- Percentiles: 5%, 10%, 15%, 20%
- Assessment features: With/Without
- Thresholds: 0.10 to 0.70 in 0.01 increments

New optimization criteria:
- Youden's J (clinical standard)
- Matthews Correlation Coefficient (MCC)
- G-Mean (geometric mean of sensitivity/specificity)
- F3 Score (heavily recall-weighted)
- Cost-optimal (FN=3x and 5x FP)
- Recall thresholds (80%, 85%, 90%)

Generates comprehensive report with best configuration per scenario.
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
ENRICHED_DIR = BASE_DIR / "data/enriched_features"
OUTPUT_DIR = BASE_DIR / "data/report/models/comprehensive_optimization"
OUTPUT_FILE = OUTPUT_DIR / "all_models_optimized.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Configuration
PERCENTILES = [0.05, 0.10, 0.15, 0.20]
CUTOFF_WEEKS = [2, 4, 6, 8, 'full']
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


def load_page_views():
    """Load page views data."""
    print("Loading page views...")
    df = pd.read_parquet(PAGE_VIEWS_FILE)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df[df['course_id'].isin(COURSES)].copy()
    df['user_id'] = df['user_id'].apply(normalize_user_id)
    print(f"  Loaded {len(df):,} page views")
    return df


def load_enrollments():
    """Load enrollment data with target variable."""
    df = pd.read_csv(ENROLLMENTS_FILE)
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


def load_enriched_features(cutoff='full'):
    """Load pre-computed enriched features for a specific cutoff."""
    if cutoff == 'full':
        feature_dir = ENRICHED_DIR
    else:
        feature_dir = ENRICHED_DIR / f'cutoff_week_{cutoff}'

    if not feature_dir.exists():
        return None

    feature_files = [
        'session_features.parquet',
        'category_features.parquet',
        'proactivity_features.parquet',
        'pca_features.parquet',
        'weekly_features.parquet',
        'ngram_features.parquet',
        'graph_features.parquet',
        'time_features.parquet',
    ]

    # Add normalized features for full cutoff
    if cutoff == 'full':
        feature_files.append('normalized_features.parquet')

    dfs = []
    for fname in feature_files:
        fpath = feature_dir / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            if fname == 'normalized_features.parquet':
                znorm_cols = [c for c in df.columns if c.endswith('_znorm')]
                df = df[['user_id', 'course_id'] + znorm_cols]
            dfs.append(df)

    if not dfs:
        return None

    # Merge all features
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
        df_merged = df_merged.drop(columns=dup_cols)

    return df_merged


def get_course_starts(df_pv, percentile):
    """Calculate course start dates using given percentile."""
    course_starts = {}
    for course_id in COURSES:
        df_course = df_pv[df_pv['course_id'] == course_id]
        if len(df_course) > 0:
            course_starts[course_id] = df_course['created_at'].quantile(percentile)
    return course_starts


def filter_by_cutoff(df_pv, course_starts, cutoff_weeks):
    """Filter page views to first N weeks from course start."""
    filtered = []
    for course_id in COURSES:
        if course_id not in course_starts:
            continue
        start = course_starts[course_id]
        cutoff_date = start + timedelta(weeks=cutoff_weeks)
        df_course = df_pv[(df_pv['course_id'] == course_id) & (df_pv['created_at'] <= cutoff_date)]
        filtered.append(df_course)

    if not filtered:
        return pd.DataFrame()
    return pd.concat(filtered, ignore_index=True)


def calculate_features_from_pageviews(df_pv, course_starts):
    """Calculate comprehensive features from filtered page views."""
    results = []

    for (user_id, course_id), group in df_pv.groupby(['user_id', 'course_id']):
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
        for sid in range(1, min(n_sessions + 1, 50)):
            session_mask = session_ids == sid
            if session_mask.sum() > 1:
                session_ts = timestamps[session_mask]
                duration = (session_ts.max() - session_ts.min()).total_seconds() / 60
                session_durations.append(duration)

        if session_durations:
            features['total_session_time'] = sum(session_durations)
            features['avg_session_duration'] = np.mean(session_durations)
            features['std_session_duration'] = np.std(session_durations) if len(session_durations) > 1 else 0
        else:
            features['total_session_time'] = 0
            features['avg_session_duration'] = 0
            features['std_session_duration'] = 0

        # === CATEGORY FEATURES ===
        for cat in ['modules', 'pages', 'files', 'discussions', 'quizzes', 'assignments', 'grades']:
            cat_mask = group['resource_type'] == cat
            features[f'{cat}_views'] = cat_mask.sum()
            features[f'{cat}_unique'] = group.loc[cat_mask, 'http_request'].nunique() if cat_mask.any() else 0

        # Percentages
        total = len(group)
        for cat in ['modules', 'pages', 'files', 'discussions']:
            features[f'{cat}_pct'] = features[f'{cat}_views'] / total if total > 0 else 0

        # === TIME FEATURES ===
        hours = timestamps.dt.hour
        features['morning_views'] = ((hours >= 6) & (hours < 12)).sum()
        features['afternoon_views'] = ((hours >= 12) & (hours < 18)).sum()
        features['evening_views'] = ((hours >= 18) & (hours < 24)).sum()
        features['night_views'] = ((hours >= 0) & (hours < 6)).sum()

        weekdays = timestamps.dt.dayofweek
        features['weekday_views'] = (weekdays < 5).sum()
        features['weekend_views'] = (weekdays >= 5).sum()

        # === PROACTIVITY FEATURES ===
        days_active = timestamps.dt.date.nunique()
        features['days_active'] = days_active
        features['activity_density'] = len(group) / max(days_active, 1)

        # First activity relative to course start
        first_activity = timestamps.min()
        days_to_first = (first_activity - course_start).days
        features['days_to_first_activity'] = days_to_first

        # === TRANSITION FEATURES ===
        if len(group) > 1:
            categories = group['resource_type'].values
            transitions = [f"{categories[i]}->{categories[i+1]}" for i in range(len(categories)-1)]
            features['unique_transitions'] = len(set(transitions))
        else:
            features['unique_transitions'] = 0

        # === DCT FEATURES (temporal patterns) ===
        if len(group) >= 8:
            # Bin activity into weekly buckets
            days_from_start = (timestamps - course_start).dt.days
            weekly_bins = (days_from_start // 7).clip(lower=0)
            weekly_counts = weekly_bins.value_counts().sort_index()
            weekly_array = np.zeros(max(8, weekly_counts.index.max() + 1))
            for week, count in weekly_counts.items():
                if 0 <= week < len(weekly_array):
                    weekly_array[week] = count

            if weekly_array.sum() > 0:
                weekly_array = weekly_array / weekly_array.sum()
                dct_coeffs = dct(weekly_array[:8], norm='ortho')
                for i in range(min(4, len(dct_coeffs))):
                    features[f'dct_{i}'] = dct_coeffs[i]

        results.append(features)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


def calculate_znorm_features(df, feature_cols):
    """Calculate z-score normalized features per course."""
    znorm_data = []

    for course_id in df['course_id'].unique():
        course_df = df[df['course_id'] == course_id].copy()

        for col in feature_cols:
            if col in course_df.columns:
                col_data = course_df[col]
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    course_df[f'{col}_znorm'] = (col_data - mean_val) / std_val
                else:
                    course_df[f'{col}_znorm'] = 0

        znorm_data.append(course_df)

    return pd.concat(znorm_data, ignore_index=True)


def filter_assessment_features(feature_cols, include_assessment=True):
    """Filter out assessment-related features if needed."""
    if include_assessment:
        return feature_cols

    filtered = []
    for col in feature_cols:
        col_lower = col.lower()
        if not any(pattern in col_lower for pattern in ASSESSMENT_PATTERNS):
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
        return feature_cols, []

    corr_matrix = X[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [f for f in feature_cols if f not in to_drop]
    return selected, to_drop


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
    f3 = 10 * precision * recall / (9 * precision + recall) if (9 * precision + recall) > 0 else 0

    # Advanced metrics
    youden_j = recall + specificity - 1
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0
    g_mean = np.sqrt(recall * specificity)
    balanced_accuracy = (recall + specificity) / 2

    return {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'specificity': specificity,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'youden_j': youden_j,
        'mcc': mcc,
        'g_mean': g_mean,
        'balanced_accuracy': balanced_accuracy,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def calculate_cost(fn, fp, cost_fn=3.0, cost_fp=1.0):
    """Calculate weighted cost of misclassification."""
    return fn * cost_fn + fp * cost_fp


def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal thresholds by various criteria."""
    thresholds = np.arange(0.10, 0.71, 0.01)
    results = []

    for t in thresholds:
        metrics = calculate_metrics_at_threshold(y_true, y_pred_proba, t)
        metrics['cost_3x'] = calculate_cost(metrics['fn'], metrics['fp'], 3.0, 1.0)
        metrics['cost_5x'] = calculate_cost(metrics['fn'], metrics['fp'], 5.0, 1.0)
        results.append(metrics)

    df_results = pd.DataFrame(results)

    # Find optimal thresholds for different criteria
    optimal = {}

    # Baseline
    baseline_idx = df_results[df_results['threshold'].round(2) == 0.50].index
    if len(baseline_idx) > 0:
        optimal['baseline_0.50'] = df_results.loc[baseline_idx[0]].to_dict()

    # Max metrics
    for metric in ['f2', 'f3', 'youden_j', 'mcc', 'g_mean', 'accuracy']:
        idx = df_results[metric].idxmax()
        optimal[f'max_{metric}'] = df_results.loc[idx].to_dict()

    # Min cost
    idx_cost_3x = df_results['cost_3x'].idxmin()
    optimal['cost_optimal_3x'] = df_results.loc[idx_cost_3x].to_dict()

    idx_cost_5x = df_results['cost_5x'].idxmin()
    optimal['cost_optimal_5x'] = df_results.loc[idx_cost_5x].to_dict()

    # Recall thresholds
    for recall_target in [0.80, 0.85, 0.90]:
        recall_filtered = df_results[df_results['recall'] >= recall_target]
        if len(recall_filtered) > 0:
            best_idx = recall_filtered['accuracy'].idxmax()
            optimal[f'recall_{int(recall_target*100)}'] = df_results.loc[best_idx].to_dict()

    return optimal, df_results.to_dict('records')


def run_experiment(df_features, df_enroll, cutoff, percentile, include_assessment):
    """Run a single experiment with given configuration."""
    # Merge with enrollments
    df = df_features.merge(
        df_enroll[['user_id', 'course_id', 'failed', 'final_score']],
        on=['user_id', 'course_id'],
        how='inner'
    )
    df = df.dropna(subset=['failed'])

    if len(df) < 50:
        return None

    # Get feature columns
    exclude_cols = ['user_id', 'course_id', 'failed', 'final_score', 'enrollment_state']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Filter assessment features
    feature_cols = filter_assessment_features(feature_cols, include_assessment)

    X = df[feature_cols].copy()
    y = df['failed'].values

    # Handle missing/infinite values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_cols = numeric_cols

    if len(feature_cols) < 5:
        return None

    # Feature selection
    if len(feature_cols) > 10:
        selected_by_importance, importances = select_features_by_importance(X, y, feature_cols)
        if len(selected_by_importance) > 0:
            selected_final, _ = remove_correlated_features(X, selected_by_importance)
            if len(selected_final) > 0:
                feature_cols = selected_final
                X = X[feature_cols]

    # Train model with CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)

    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        roc_auc = roc_auc_score(y, y_pred_proba)
    except Exception as e:
        print(f"    Error: {e}")
        return None

    # Find optimal thresholds
    optimal_thresholds, all_thresholds = find_optimal_thresholds(y, y_pred_proba)

    # Get feature importances
    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10])

    return {
        'cutoff': cutoff,
        'percentile': percentile,
        'include_assessment': include_assessment,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'failure_rate': float(y.mean()),
        'roc_auc': float(roc_auc),
        'optimal_thresholds': optimal_thresholds,
        'top_features': top_features,
        'all_thresholds': all_thresholds
    }


def print_summary_table(experiments):
    """Print summary table of results."""
    print()
    print('=' * 100)
    print('SUMMARY: BEST CONFIGURATIONS BY WEEK')
    print('=' * 100)

    # Group by cutoff
    by_cutoff = {}
    for exp in experiments:
        cutoff = exp['cutoff']
        if cutoff not in by_cutoff:
            by_cutoff[cutoff] = []
        by_cutoff[cutoff].append(exp)

    print()
    print(f'{"Cutoff":<10} {"Percentile":<12} {"Assessment":<12} {"ROC-AUC":<10} {"Best t":<8} {"Recall":<10} {"Accuracy":<10} {"Criterion":<15}')
    print('-' * 100)

    best_overall = None
    best_per_week = {}

    for cutoff in ['2', '4', '6', '8', 'full']:
        cutoff_key = int(cutoff) if cutoff != 'full' else 'full'
        if cutoff_key not in by_cutoff:
            continue

        exps = by_cutoff[cutoff_key]

        # Find best by ROC-AUC
        best_exp = max(exps, key=lambda x: x['roc_auc'])
        best_per_week[cutoff_key] = best_exp

        # Get best threshold metrics
        if 'max_youden_j' in best_exp['optimal_thresholds']:
            best_t = best_exp['optimal_thresholds']['max_youden_j']
            criterion = 'Youden J'
        else:
            best_t = best_exp['optimal_thresholds'].get('max_f2', {})
            criterion = 'Max F2'

        print(f'{cutoff:<10} {best_exp["percentile"]*100:.0f}%{"":<9} {"Yes" if best_exp["include_assessment"] else "No":<12} '
              f'{best_exp["roc_auc"]:.3f}{"":<5} {best_t.get("threshold", 0):.2f}{"":<4} '
              f'{best_t.get("recall", 0)*100:.1f}%{"":<5} {best_t.get("accuracy", 0)*100:.1f}%{"":<5} {criterion}')

        if best_overall is None or best_exp['roc_auc'] > best_overall['roc_auc']:
            best_overall = best_exp

    return best_per_week, best_overall


def print_deployment_recommendations(best_per_week):
    """Print deployment recommendations."""
    print()
    print('=' * 100)
    print('DEPLOYMENT RECOMMENDATIONS BY INTERVENTION TIMING')
    print('=' * 100)

    recommendations = [
        ('2', 'Week 2-3: Early Watch List', 'Initial identification, high uncertainty'),
        ('4', 'Week 4-5: First Intervention', 'Good balance of early detection and reliability'),
        ('6', 'Week 6-7: Active Intervention', 'Strong predictions, 10+ weeks remaining'),
        ('8', 'Week 8+: Intensive Support', 'High reliability, mid-semester checkpoint'),
        ('full', 'End of Term: Final Review', 'Maximum accuracy for grade prediction'),
    ]

    print()
    for cutoff, timing, description in recommendations:
        cutoff_key = int(cutoff) if cutoff != 'full' else 'full'
        if cutoff_key not in best_per_week:
            continue

        exp = best_per_week[cutoff_key]
        opt = exp['optimal_thresholds']

        # Get aggressive, balanced, conservative thresholds
        aggressive = opt.get('cost_optimal_5x', opt.get('max_f3', {}))
        balanced = opt.get('max_youden_j', opt.get('max_f2', {}))
        conservative = opt.get('max_mcc', opt.get('max_accuracy', {}))

        print(f'{timing}')
        print(f'  Description: {description}')
        print(f'  ROC-AUC: {exp["roc_auc"]:.3f} | Samples: {exp["n_samples"]} | Features: {exp["n_features"]}')
        print(f'  Config: Percentile={exp["percentile"]*100:.0f}%, Assessment={"Yes" if exp["include_assessment"] else "No"}')
        print()
        print(f'  Thresholds:')
        print(f'    Aggressive (t={aggressive.get("threshold", 0):.2f}): Recall={aggressive.get("recall", 0)*100:.1f}%, Accuracy={aggressive.get("accuracy", 0)*100:.1f}%')
        print(f'    Balanced   (t={balanced.get("threshold", 0):.2f}): Recall={balanced.get("recall", 0)*100:.1f}%, Accuracy={balanced.get("accuracy", 0)*100:.1f}%')
        print(f'    Conservative (t={conservative.get("threshold", 0):.2f}): Recall={conservative.get("recall", 0)*100:.1f}%, Accuracy={conservative.get("accuracy", 0)*100:.1f}%')
        print()


def main():
    print('=' * 100)
    print('COMPREHENSIVE THRESHOLD OPTIMIZATION - ALL TIME-LIMITED MODELS')
    print('=' * 100)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_pv = load_page_views()
    df_enroll = load_enrollments()

    experiments = []
    total_configs = len(CUTOFF_WEEKS) * len(PERCENTILES) * 2  # 2 for with/without assessment
    config_num = 0

    for cutoff in CUTOFF_WEEKS:
        print(f'\n{"="*70}')
        print(f'CUTOFF: {"Week " + str(cutoff) if cutoff != "full" else "FULL SEMESTER"}')
        print(f'{"="*70}')

        for percentile in PERCENTILES:
            # Calculate course starts for this percentile
            course_starts = get_course_starts(df_pv, percentile)

            # Filter page views by cutoff (or use all for 'full')
            if cutoff == 'full':
                # Try to load pre-computed enriched features
                df_features = load_enriched_features('full')
                if df_features is None:
                    print(f"  No enriched features for full cutoff, skipping...")
                    continue
            else:
                # Filter page views and calculate features
                df_filtered = filter_by_cutoff(df_pv, course_starts, cutoff)
                if len(df_filtered) < 100:
                    print(f"  Insufficient data for cutoff={cutoff}, percentile={percentile}")
                    continue

                df_features = calculate_features_from_pageviews(df_filtered, course_starts)
                if len(df_features) < 50:
                    print(f"  Insufficient features for cutoff={cutoff}, percentile={percentile}")
                    continue

                # Calculate z-norm features
                numeric_cols = [c for c in df_features.columns
                               if c not in ['user_id', 'course_id'] and df_features[c].dtype in ['float64', 'int64']]
                df_features = calculate_znorm_features(df_features, numeric_cols)

            for include_assessment in [True, False]:
                config_num += 1
                label = f"week_{cutoff}_p{int(percentile*100)}_{'with' if include_assessment else 'without'}_assessment"
                print(f'\n  [{config_num}/{total_configs}] {label}')

                result = run_experiment(df_features, df_enroll, cutoff, percentile, include_assessment)

                if result:
                    experiments.append(result)
                    opt = result['optimal_thresholds'].get('max_youden_j', result['optimal_thresholds'].get('max_f2', {}))
                    print(f'    ROC-AUC: {result["roc_auc"]:.3f} | Samples: {result["n_samples"]} | Features: {result["n_features"]}')
                    print(f'    Best threshold (Youden J): t={opt.get("threshold", 0):.2f} -> Recall={opt.get("recall", 0)*100:.1f}%, Accuracy={opt.get("accuracy", 0)*100:.1f}%')
                else:
                    print(f'    SKIPPED (insufficient data)')

    # Print summary
    if experiments:
        best_per_week, best_overall = print_summary_table(experiments)
        print_deployment_recommendations(best_per_week)

        # Save results
        results = {
            'experiments': experiments,
            'best_per_week': {str(k): v for k, v in best_per_week.items()},
            'best_overall': best_overall,
            'configuration': {
                'percentiles': PERCENTILES,
                'cutoff_weeks': [str(c) for c in CUTOFF_WEEKS],
                'xgboost_params': XGBOOST_PARAMS,
                'importance_threshold': IMPORTANCE_THRESHOLD,
                'correlation_threshold': CORRELATION_THRESHOLD
            }
        }

        with open(OUTPUT_FILE, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print()
        print(f'Results saved to: {OUTPUT_FILE}')
    else:
        print('\nNo experiments completed successfully.')


if __name__ == '__main__':
    main()
