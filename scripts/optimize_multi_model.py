#!/usr/bin/env python3
"""
Multi-Model Optimization for Early Warning System

Tests multiple model architectures to find the best accuracy-recall balance:
1. XGBoost (current baseline)
2. LightGBM
3. Random Forest with class weights
4. Gradient Boosting
5. SVM with RBF kernel
6. Multi-Layer Perceptron (MLP)
7. 1D CNN on temporal features
8. Voting Ensemble
9. Stacking Ensemble

Optimization criteria:
- G-Mean: sqrt(Recall × Specificity) - balances both metrics
- Balanced Accuracy: (Recall + Specificity) / 2
- Custom: Maximize Accuracy subject to Recall >= 75%
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
from sklearn.metrics import roc_auc_score, confusion_matrix, make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
ASSIGNMENTS_FILE = BASE_DIR / "data/assignment_analytics.json"
OUTPUT_FILE = BASE_DIR / "data/analysis/multi_model_optimization_results.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

SESSION_GAP_MINUTES = 30

# Feature patterns to exclude for "without assessment" experiments
ASSESSMENT_PATTERNS = ['quiz', 'quizzes', 'assi', 'assignment', 'grade', 'grad', 'score', 'submission']


def normalize_user_id(user_id):
    if user_id > 10000000000:
        return user_id % 10000000000
    return user_id


def load_data():
    """Load all data sources."""
    print("Loading data...")

    df = pd.read_parquet(PAGE_VIEWS_FILE)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)
    df = df[df['course_id'].isin(COURSES)].copy()
    df['user_id'] = df['user_id'].apply(normalize_user_id)

    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    df_enroll['failed'] = (df_enroll['final_score'] < 57).astype(int)

    with open(ASSIGNMENTS_FILE) as f:
        assignments = json.load(f)
    assignments = [a for a in assignments if a.get('course_id') in COURSES and a.get('due_at')]
    assignments_df = pd.DataFrame(assignments)
    if len(assignments_df) > 0:
        assignments_df['due_at'] = pd.to_datetime(assignments_df['due_at']).dt.tz_localize(None)

    print(f"  Page views: {len(df):,}")
    print(f"  Students: {len(df_enroll)}")

    return df, df_enroll, assignments_df


def get_course_starts(df, percentile):
    course_starts = {}
    for course_id in COURSES:
        df_course = df[df['course_id'] == course_id]
        if len(df_course) > 0:
            course_starts[course_id] = df_course['created_at'].quantile(percentile)
    return course_starts


def filter_by_cutoff(df, course_starts, cutoff_weeks):
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

    return pd.concat(filtered, ignore_index=True) if filtered else pd.DataFrame()


def calculate_features(df, assignments_df, course_starts, cutoff_weeks):
    """Calculate comprehensive features including pre-assessment."""
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
        n_sessions = session_starts.sum()

        total_span = max((timestamps.max() - course_start).days / 7, 1)

        features = {
            'user_id': user_id,
            'course_id': course_id,
            'total_views': len(group),
            'n_sessions': n_sessions,
            'sessions_per_week': n_sessions / total_span,
            'views_per_session': len(group) / max(n_sessions, 1),
        }

        # Category features
        categories = ['files', 'discussions', 'quizzes', 'assignments', 'modules', 'grades', 'pages']
        for cat in categories:
            cat_data = group[group['resource_type'] == cat]
            features[f'{cat}_views'] = len(cat_data)
            features[f'{cat}_pct'] = len(cat_data) / len(group) * 100 if len(group) > 0 else 0

        # Time features
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofweek
        total = len(hours)

        features['pct_morning'] = sum((hours >= 6) & (hours < 12)) / total * 100 if total > 0 else 0
        features['pct_afternoon'] = sum((hours >= 12) & (hours < 18)) / total * 100 if total > 0 else 0
        features['pct_evening'] = sum((hours >= 18) & (hours < 24)) / total * 100 if total > 0 else 0
        features['pct_weekend'] = sum(days >= 5) / total * 100 if total > 0 else 0
        features['unique_hours'] = hours.nunique()
        features['unique_days'] = days.nunique()

        # Trajectory features
        weeks = ((timestamps - course_start).dt.days // 7 + 1).clip(lower=1)
        features['active_weeks'] = weeks.nunique()
        features['first_active_week'] = weeks.min()

        weekly_counts = weeks.value_counts().sort_index()
        if len(weekly_counts) >= 2:
            features['weekly_trend'] = np.polyfit(range(len(weekly_counts)), weekly_counts.values, 1)[0]
        else:
            features['weekly_trend'] = 0

        # DCT
        if len(weekly_counts) >= 4:
            padded = np.zeros(8)
            padded[:min(len(weekly_counts), 8)] = weekly_counts.values[:8]
            dct_coeffs = dct(padded, norm='ortho')
            features['dct_0'] = dct_coeffs[0]
            features['dct_1'] = dct_coeffs[1]
        else:
            features['dct_0'] = 0
            features['dct_1'] = 0

        # Pre-assessment features
        if cutoff_weeks == 'full':
            cutoff_date = timestamps.max() + timedelta(days=365)
        else:
            cutoff_date = course_start + timedelta(weeks=cutoff_weeks)

        course_assignments = assignments_df[
            (assignments_df['course_id'] == course_id) &
            (assignments_df['due_at'] <= cutoff_date)
        ]

        activity_72h = 0
        for _, assignment in course_assignments.iterrows():
            due_at = assignment['due_at']
            window = (due_at - timedelta(hours=72), due_at)
            activity_72h += ((timestamps >= window[0]) & (timestamps <= window[1])).sum()

        features['activity_72h_before'] = activity_72h
        features['preparation_intensity'] = activity_72h / len(group) if len(group) > 0 else 0

        # Quiz/Assignment access
        quiz_views = group[group['resource_type'] == 'quizzes']
        features['quiz_access_count'] = len(quiz_views)
        features['unique_quizzes'] = quiz_views['resource_id'].nunique() if len(quiz_views) > 0 else 0

        assgn_views = group[group['resource_type'] == 'assignments']
        features['assgn_access_count'] = len(assgn_views)
        features['unique_assignments'] = assgn_views['resource_id'].nunique() if len(assgn_views) > 0 else 0

        features['assessment_diversity'] = features['unique_quizzes'] + features['unique_assignments']

        results.append(features)

    return pd.DataFrame(results)


def add_znorm(df):
    """Add z-normalized features per course."""
    exclude = ['user_id', 'course_id']
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]

    result = []
    for course_id in df['course_id'].unique():
        cdf = df[df['course_id'] == course_id].copy()
        for col in numeric:
            mean, std = cdf[col].mean(), cdf[col].std()
            cdf[f'{col}_znorm'] = (cdf[col] - mean) / std if std > 0 else 0
        result.append(cdf)

    return pd.concat(result, ignore_index=True)


def get_models():
    """Get all models to test."""
    models = {}

    # 1. XGBoost
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, eval_metric='logloss', verbosity=0, random_state=42
        )
        models['XGBoost_balanced'] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, scale_pos_weight=1.5,  # Weight positive class
            eval_metric='logloss', verbosity=0, random_state=42
        )

    # 2. LightGBM
    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, verbosity=-1, random_state=42
        )
        models['LightGBM_balanced'] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, class_weight='balanced', verbosity=-1, random_state=42
        )

    # 3. Random Forest
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42
    )
    models['RandomForest_balanced'] = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight='balanced', random_state=42
    )

    # 4. Gradient Boosting
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )

    # 5. SVM
    models['SVM_RBF'] = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        cv=3
    )
    models['SVM_balanced'] = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
        cv=3
    )

    # 6. MLP Neural Network
    models['MLP'] = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu',
        learning_rate_init=0.001, max_iter=500, random_state=42
    )
    models['MLP_deep'] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        learning_rate_init=0.001, max_iter=500, random_state=42
    )

    # 7. Logistic Regression (baseline)
    models['LogisticRegression'] = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42
    )
    models['LogisticRegression_balanced'] = LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=42
    )

    return models


def create_ensemble(X, y):
    """Create ensemble models."""
    ensembles = {}

    base_estimators = [
        ('xgb', XGBClassifier(n_estimators=50, max_depth=4, verbosity=0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)),
    ]

    # Voting ensemble
    ensembles['VotingEnsemble'] = VotingClassifier(
        estimators=base_estimators, voting='soft'
    )

    # Stacking ensemble
    ensembles['StackingEnsemble'] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )

    return ensembles


def calculate_metrics(y_true, y_pred_proba, thresholds):
    """Calculate metrics at multiple thresholds."""
    results = []

    for t in thresholds:
        y_pred = (y_pred_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        # G-Mean: geometric mean of recall and specificity
        g_mean = np.sqrt(recall * specificity)

        # Balanced accuracy
        balanced_acc = (recall + specificity) / 2

        # F1 and F2
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

        results.append({
            'threshold': t,
            'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'g_mean': g_mean,
            'balanced_acc': balanced_acc,
            'f1': f1,
            'f2': f2,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })

    return pd.DataFrame(results)


def find_best_thresholds(metrics_df):
    """Find optimal thresholds for different criteria."""
    optimal = {}

    # Max G-Mean (best balance)
    idx = metrics_df['g_mean'].idxmax()
    optimal['max_g_mean'] = metrics_df.loc[idx].to_dict()

    # Max Balanced Accuracy
    idx = metrics_df['balanced_acc'].idxmax()
    optimal['max_balanced_acc'] = metrics_df.loc[idx].to_dict()

    # Max Accuracy with Recall >= 75%
    subset = metrics_df[metrics_df['recall'] >= 0.75]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['acc_at_recall_75'] = subset.loc[idx].to_dict()

    # Max Accuracy with Recall >= 80%
    subset = metrics_df[metrics_df['recall'] >= 0.80]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['acc_at_recall_80'] = subset.loc[idx].to_dict()

    # Max F1 (balance precision/recall)
    idx = metrics_df['f1'].idxmax()
    optimal['max_f1'] = metrics_df.loc[idx].to_dict()

    return optimal


def train_and_evaluate(model, X, y, model_name):
    """Train model and return evaluation metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    except Exception as e:
        print(f"    Error with {model_name}: {e}")
        return None

    roc_auc = roc_auc_score(y, y_pred_proba)

    # Calculate metrics at various thresholds
    thresholds = np.arange(0.15, 0.65, 0.01)
    metrics_df = calculate_metrics(y, y_pred_proba, thresholds)

    # Find optimal thresholds
    optimal = find_best_thresholds(metrics_df)

    return {
        'roc_auc': float(roc_auc),
        'optimal': optimal,
    }


def filter_assessment_features(feature_cols, with_assessment=True):
    """Filter out assessment-related features if with_assessment is False."""
    if with_assessment:
        return feature_cols

    filtered = []
    for col in feature_cols:
        col_lower = col.lower()
        exclude = False
        for pattern in ASSESSMENT_PATTERNS:
            if pattern in col_lower:
                exclude = True
                break
        if not exclude:
            filtered.append(col)
    return filtered


def run_week_experiment(df, df_enroll, assignments_df, cutoff_weeks, percentile, with_assessment=True):
    """Run all models for a specific week/percentile configuration."""
    # Prepare data
    course_starts = get_course_starts(df, percentile)
    df_filtered = filter_by_cutoff(df, course_starts, cutoff_weeks)

    if len(df_filtered) < 100:
        return None

    # Calculate features
    df_features = calculate_features(df_filtered, assignments_df, course_starts, cutoff_weeks)
    df_features = add_znorm(df_features)

    # Merge with enrollments
    df_merged = df_features.merge(
        df_enroll[['user_id', 'course_id', 'failed']],
        on=['user_id', 'course_id'], how='inner'
    ).dropna(subset=['failed'])

    if len(df_merged) < 100:
        return None

    # Prepare X, y
    exclude = ['user_id', 'course_id', 'failed']
    feature_cols = [c for c in df_merged.columns if c not in exclude]

    # Filter out assessment features if needed
    feature_cols = filter_assessment_features(feature_cols, with_assessment)

    X = df_merged[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df_merged['failed'].values

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Get models
    models = get_models()

    # Add ensembles
    if HAS_XGBOOST:
        ensembles = create_ensemble(X_scaled, y)
        models.update(ensembles)

    results = {
        'n_samples': len(y),
        'failure_rate': float(y.mean()),
        'n_features': len(feature_cols),
        'with_assessment': with_assessment,
        'feature_cols': feature_cols,
        'models': {}
    }

    for model_name, model in models.items():
        print(f"    Testing {model_name}...")
        model_result = train_and_evaluate(model, X_scaled, y, model_name)
        if model_result:
            results['models'][model_name] = model_result

    return results


def main():
    print("=" * 70)
    print("MULTI-MODEL OPTIMIZATION FOR EARLY WARNING SYSTEM")
    print("=" * 70)
    print()

    # Load data
    df, df_enroll, assignments_df = load_data()
    print()

    # Load existing results to preserve them
    existing_results = {}
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            existing_results = json.load(f)
        print(f"Loaded {len(existing_results)} existing experiments")

    # NEW comprehensive experiment configurations
    configs = [
        # Week 8 (completely missing - all 4 variants)
        {'week': 8, 'percentile': 0.10, 'with_assessment': True},
        {'week': 8, 'percentile': 0.10, 'with_assessment': False},
        {'week': 8, 'percentile': 0.20, 'with_assessment': True},
        {'week': 8, 'percentile': 0.20, 'with_assessment': False},
        # Full data with P20% (fair comparison)
        {'week': 'full', 'percentile': 0.20, 'with_assessment': True},
        {'week': 'full', 'percentile': 0.20, 'with_assessment': False},
        # WITHOUT assessment for existing weeks
        {'week': 2, 'percentile': 0.10, 'with_assessment': False},
        {'week': 4, 'percentile': 0.20, 'with_assessment': False},
        {'week': 6, 'percentile': 0.20, 'with_assessment': False},
    ]

    all_results = existing_results.copy()

    for config in configs:
        week = config['week']
        pct = config['percentile']
        with_asmt = config['with_assessment']

        # Build experiment name
        asmt_suffix = '_with_assessment' if with_asmt else '_without_assessment'
        exp_name = f'week_{week}_p{int(pct*100)}{asmt_suffix}'

        # Skip if already exists
        if exp_name in all_results:
            print(f"\nSkipping {exp_name} (already exists)")
            continue

        print(f"\n{'='*70}")
        print(f"WEEK {week}, Percentile {int(pct*100)}%, Assessment: {with_asmt}")
        print(f"Experiment: {exp_name}")
        print(f"{'='*70}")

        result = run_week_experiment(df, df_enroll, assignments_df, week, pct, with_asmt)

        if result is None:
            print("  Skipped (insufficient data)")
            continue

        all_results[exp_name] = result

        # Print summary
        print(f"\n  Samples: {result['n_samples']}, Failure rate: {result['failure_rate']:.1%}")
        print(f"  Features: {result['n_features']} ({'with' if with_asmt else 'without'} assessment)")
        print(f"\n  Model Comparison (sorted by G-Mean):")
        print(f"  {'Model':<25} {'ROC-AUC':>8} {'G-Mean':>8} {'Recall':>8} {'Accuracy':>8} {'Threshold':>9}")
        print(f"  {'-'*70}")

        # Sort by G-Mean
        model_summary = []
        for model_name, model_result in result['models'].items():
            g_mean_opt = model_result['optimal'].get('max_g_mean', {})
            model_summary.append({
                'name': model_name,
                'roc_auc': model_result['roc_auc'],
                'g_mean': g_mean_opt.get('g_mean', 0),
                'recall': g_mean_opt.get('recall', 0),
                'accuracy': g_mean_opt.get('accuracy', 0),
                'threshold': g_mean_opt.get('threshold', 0.5),
            })

        model_summary.sort(key=lambda x: -x['g_mean'])

        for m in model_summary[:10]:  # Top 10
            print(f"  {m['name']:<25} {m['roc_auc']:>8.4f} {m['g_mean']:>8.4f} "
                  f"{m['recall']:>8.1%} {m['accuracy']:>8.1%} {m['threshold']:>9.2f}")

        # Best accuracy at recall >= 80%
        print(f"\n  Best Accuracy at Recall >= 80%:")
        best_acc_80 = []
        for model_name, model_result in result['models'].items():
            opt = model_result['optimal'].get('acc_at_recall_80', {})
            if opt:
                best_acc_80.append({
                    'name': model_name,
                    'accuracy': opt.get('accuracy', 0),
                    'recall': opt.get('recall', 0),
                    'threshold': opt.get('threshold', 0.5),
                })

        best_acc_80.sort(key=lambda x: -x['accuracy'])
        for m in best_acc_80[:5]:
            print(f"    {m['name']:<25} Acc={m['accuracy']:.1%}, Recall={m['recall']:.1%}, t={m['threshold']:.2f}")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nResults saved to {OUTPUT_FILE}")

    # Final summary
    print("\n" + "=" * 70)
    print("BEST MODELS SUMMARY")
    print("=" * 70)

    for config_name, result in all_results.items():
        print(f"\n{config_name}:")

        # Find best by different criteria
        best_roc = max(result['models'].items(), key=lambda x: x[1]['roc_auc'])
        best_gmean = max(result['models'].items(),
                        key=lambda x: x[1]['optimal'].get('max_g_mean', {}).get('g_mean', 0))

        print(f"  Best ROC-AUC: {best_roc[0]} ({best_roc[1]['roc_auc']:.4f})")

        gm = best_gmean[1]['optimal'].get('max_g_mean', {})
        print(f"  Best G-Mean:  {best_gmean[0]} (G={gm.get('g_mean', 0):.4f}, "
              f"Acc={gm.get('accuracy', 0):.1%}, Rec={gm.get('recall', 0):.1%})")

        # Best accuracy at recall >= 80%
        best_acc = None
        best_acc_val = 0
        for model_name, model_result in result['models'].items():
            opt = model_result['optimal'].get('acc_at_recall_80', {})
            if opt and opt.get('accuracy', 0) > best_acc_val:
                best_acc_val = opt['accuracy']
                best_acc = (model_name, opt)

        if best_acc:
            print(f"  Best Acc@80%Rec: {best_acc[0]} (Acc={best_acc[1]['accuracy']:.1%}, "
                  f"Rec={best_acc[1]['recall']:.1%})")

    # Comparison summary: WITH vs WITHOUT assessment
    print("\n" + "=" * 70)
    print("WITH vs WITHOUT ASSESSMENT COMPARISON")
    print("=" * 70)
    print(f"\n{'Experiment':<35} {'ROC-AUC (with)':>15} {'ROC-AUC (wo)':>15} {'Diff':>10}")
    print("-" * 75)

    weeks_to_compare = [2, 4, 6, 8, 'full']
    pcts = {2: 10, 4: 20, 6: 20, 8: 10, 'full': 20}

    for week in weeks_to_compare:
        pct = pcts.get(week, 20)
        with_key = f'week_{week}_p{pct}_with_assessment'
        wo_key = f'week_{week}_p{pct}_without_assessment'

        with_result = all_results.get(with_key)
        wo_result = all_results.get(wo_key)

        if with_result and wo_result:
            with_best = max(with_result['models'].values(), key=lambda x: x['roc_auc'])['roc_auc']
            wo_best = max(wo_result['models'].values(), key=lambda x: x['roc_auc'])['roc_auc']
            diff = with_best - wo_best
            print(f"Week {week} (P{pct}%){'':<20} {with_best:>15.4f} {wo_best:>15.4f} {diff:>+10.4f}")
        elif with_result:
            with_best = max(with_result['models'].values(), key=lambda x: x['roc_auc'])['roc_auc']
            print(f"Week {week} (P{pct}%){'':<20} {with_best:>15.4f} {'N/A':>15} {'N/A':>10}")
        elif wo_result:
            wo_best = max(wo_result['models'].values(), key=lambda x: x['roc_auc'])['roc_auc']
            print(f"Week {week} (P{pct}%){'':<20} {'N/A':>15} {wo_best:>15.4f} {'N/A':>10}")


if __name__ == '__main__':
    main()
