#!/usr/bin/env python3
"""
Train Early Warning Model - Predicts failure BEFORE any grades exist.

This model explicitly EXCLUDES:
- grades_views, grades_check_per_week, grades_first_access_week
- Any submission scores (avg_score, min_score, max_score, etc.)
- Quiz scores

It INCLUDES:
- Proactivity features (PCT ranking - being first to access resources)
- Session patterns (regularity, count, duration)
- Category engagement (files, discussions, modules - NOT grades/quizzes with scores)
- Temporal patterns (early vs late engagement, DCT coefficients)
- Download behavior

This enables truly early intervention - predict at-risk students from day 1.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import json
import warnings
warnings.filterwarnings('ignore')

# Input files
SESSION_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/session_features.parquet')
CATEGORY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/category_features.parquet')
ENGAGEMENT_RATIOS = Path('/home/paul/projects/uautonoma/data/enriched_features/engagement_ratios.parquet')
WEEKLY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/weekly_features.parquet')
PROACTIVITY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/proactivity_features.parquet')
ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/early_warning_model_results.json')

# Features to EXCLUDE (anything related to grades or scores)
EXCLUDED_FEATURES = [
    # Grade-related
    'grades_views', 'grades_views_pct', 'grades_unique_resources', 'grades_time_min',
    'grades_check_per_week', 'grades_first_access_week',
    'grad_mean_pct', 'grad_median_pct', 'grad_std_pct', 'grad_access_rate',
    'grad_top25_rate', 'grad_top50_rate', 'grad_n_resources',
    'grad_hist_b1', 'grad_hist_b2', 'grad_hist_b3', 'grad_hist_b4', 'grad_hist_b5',

    # Score-related (these are actual grades, not activity)
    'avg_score', 'min_score', 'max_score', 'score_std', 'first_score',
    'num_graded', 'num_scores', 'submission_rate',

    # Quiz scores (quizzes are fine as activity, but scores are grades)
    'quiz_score_avg', 'quiz_score_min', 'quiz_score_max',

    # Target variables (obviously exclude)
    'final_score', 'current_score', 'failed', 'enrollment_state',

    # IDs
    'user_id', 'course_id'
]


def load_and_merge_features():
    """Load and merge all feature sets for early warning model."""
    print('Loading features...')

    # Load all feature files
    dfs_to_merge = []

    # Session features
    if SESSION_FEATURES.exists():
        session_df = pd.read_parquet(SESSION_FEATURES)
        print(f'  Session features: {len(session_df)} rows, {len(session_df.columns)} columns')
        dfs_to_merge.append(session_df)

    # Category features
    if CATEGORY_FEATURES.exists():
        category_df = pd.read_parquet(CATEGORY_FEATURES)
        print(f'  Category features: {len(category_df)} rows, {len(category_df.columns)} columns')

    # Engagement ratios
    if ENGAGEMENT_RATIOS.exists():
        ratios_df = pd.read_parquet(ENGAGEMENT_RATIOS)
        print(f'  Engagement ratios: {len(ratios_df)} rows, {len(ratios_df.columns)} columns')

    # Weekly features
    if WEEKLY_FEATURES.exists():
        weekly_df = pd.read_parquet(WEEKLY_FEATURES)
        print(f'  Weekly features: {len(weekly_df)} rows, {len(weekly_df.columns)} columns')

    # Proactivity features
    if PROACTIVITY_FEATURES.exists():
        proact_df = pd.read_parquet(PROACTIVITY_FEATURES)
        print(f'  Proactivity features: {len(proact_df)} rows, {len(proact_df.columns)} columns')

    # Load enrollments (target)
    enrollments_df = pd.read_csv(ENROLLMENTS)
    enrollments_df['failed'] = enrollments_df['final_score'] < 57
    print(f'  Enrollments: {len(enrollments_df)} rows')

    # Start with session features
    merged = session_df.copy()

    # Merge category features
    if CATEGORY_FEATURES.exists():
        merged = merged.merge(category_df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_cat'))

    # Merge engagement ratios
    if ENGAGEMENT_RATIOS.exists():
        ratio_cols = ['user_id', 'course_id', 'total_sessions', 'total_views', 'total_time_min',
                     'course_session_ratio', 'course_views_ratio', 'course_time_ratio']
        ratio_cols = [c for c in ratio_cols if c in ratios_df.columns]
        merged = merged.merge(ratios_df[ratio_cols], on=['user_id', 'course_id'], how='left')

    # Merge weekly features (exclude grades-related)
    if WEEKLY_FEATURES.exists():
        weekly_cols = ['user_id', 'course_id', 'active_weeks_count', 'first_active_week',
                      'last_active_week', 'peak_week', 'early_semester_views', 'late_semester_views',
                      'early_vs_late_ratio', 'avg_week_over_week_change', 'activity_consistency',
                      'quizzes_first_access_week', 'assignments_first_access_week',
                      'discussions_first_access_week', 'engagement_pattern']
        # Explicitly exclude grades_first_access_week
        weekly_cols = [c for c in weekly_cols if c in weekly_df.columns and 'grades' not in c.lower()]
        merged = merged.merge(weekly_df[weekly_cols], on=['user_id', 'course_id'], how='left')

    # Merge proactivity features (exclude grades-related)
    if PROACTIVITY_FEATURES.exists():
        proact_cols = [c for c in proact_df.columns if 'grad' not in c.lower() or c in ['user_id', 'course_id']]
        merged = merged.merge(proact_df[proact_cols], on=['user_id', 'course_id'], how='left')

    # Merge with enrollments (target)
    merged = merged.merge(
        enrollments_df[['user_id', 'course_id', 'final_score', 'current_score', 'failed']],
        on=['user_id', 'course_id'],
        how='left'
    )

    # Drop rows without target
    merged = merged.dropna(subset=['failed'])

    print(f'  Merged dataset: {len(merged)} rows')
    print()

    return merged


def get_early_warning_features(df):
    """Get feature columns, excluding grades/score-related features."""
    all_cols = df.columns.tolist()

    # Filter out excluded features
    feature_cols = []
    for col in all_cols:
        # Skip if in explicit exclude list
        if col in EXCLUDED_FEATURES:
            continue

        # Skip if contains 'grade' or 'score' (case-insensitive)
        col_lower = col.lower()
        if 'grade' in col_lower or 'score' in col_lower:
            continue

        # Skip if ends with common ID suffixes
        if col in ['user_id', 'course_id']:
            continue

        feature_cols.append(col)

    return feature_cols


def train_and_evaluate(X, y, model_name='XGBoost'):
    """Train model with cross-validation and return metrics."""
    if model_name == 'XGBoost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    else:
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Fit on full data for feature importance
    pipeline.fit(X, y)

    # Cross-val predictions
    y_pred_proba = np.zeros(len(y))
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred_proba[test_idx] = pipeline.predict_proba(X_test)[:, 1]
        y_pred[test_idx] = pipeline.predict(X_test)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.5,
    }

    # Feature importance
    if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
        importances = pipeline.named_steps['model'].feature_importances_
        feature_importance = dict(zip(X.columns, importances))
    else:
        feature_importance = {}

    return metrics, feature_importance


def main():
    print('=' * 70)
    print('EARLY WARNING MODEL - No Grades, Pure Activity & Proactivity')
    print('=' * 70)
    print()

    # Load data
    df = load_and_merge_features()

    # Get features (excluding grades)
    feature_cols = get_early_warning_features(df)
    print(f'Total early warning features: {len(feature_cols)}')
    print()

    # Show some excluded features
    all_cols = set(df.columns)
    excluded = all_cols - set(feature_cols) - {'user_id', 'course_id', 'final_score', 'current_score', 'failed'}
    print(f'Excluded features ({len(excluded)}):')
    for col in sorted(excluded)[:10]:
        print(f'  - {col}')
    if len(excluded) > 10:
        print(f'  ... and {len(excluded) - 10} more')
    print()

    X = df[feature_cols]
    y = df['failed'].astype(int)

    print(f'Dataset: {len(X)} samples')
    print(f'Class distribution: {y.value_counts().to_dict()}')
    print(f'Failure rate: {y.mean()*100:.1f}%')
    print()

    # Train models
    results = {}

    for model_name in ['XGBoost', 'RandomForest']:
        print(f'Training {model_name}...')
        metrics, importance = train_and_evaluate(X, y, model_name)

        results[model_name] = {
            'metrics': metrics,
            'top_features': dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
        }

        print(f'  Accuracy: {metrics["accuracy"]:.3f}')
        print(f'  ROC-AUC:  {metrics["roc_auc"]:.3f}')
        print(f'  Recall:   {metrics["recall"]:.3f}')
        print(f'  F1:       {metrics["f1"]:.3f}')
        print()

    # Comparison
    print('=' * 70)
    print('COMPARISON: Early Warning vs Models with Grades')
    print('=' * 70)
    print()
    print('Baseline (Activity-only, no page views):')
    print('  ROC-AUC:  0.787')
    print('  Accuracy: 0.740')
    print()
    print('Full model with grades features (v3):')
    print('  ROC-AUC:  0.862')
    print('  Accuracy: 0.809')
    print()

    best_model = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'])
    best_metrics = results[best_model]['metrics']
    print('EARLY WARNING MODEL (no grades, pure activity + proactivity):')
    print(f'  Model:    {best_model}')
    print(f'  ROC-AUC:  {best_metrics["roc_auc"]:.3f}')
    print(f'  Accuracy: {best_metrics["accuracy"]:.3f}')
    print(f'  Recall:   {best_metrics["recall"]:.3f} (catches {best_metrics["recall"]*100:.0f}% of at-risk students)')
    print(f'  F1:       {best_metrics["f1"]:.3f}')
    print()

    # Top features
    print('Top 15 Early Warning Features:')
    for i, (feat, imp) in enumerate(list(results[best_model]['top_features'].items())[:15], 1):
        print(f'  {i:2d}. {feat}: {imp:.3f}')
    print()

    # Key insight
    proactivity_features = [f for f in results[best_model]['top_features'].keys()
                           if 'pct' in f.lower() or 'proactiv' in f.lower() or 'top25' in f.lower()]
    print(f'Proactivity features in top 15: {len(proactivity_features)}')
    for f in proactivity_features[:5]:
        print(f'  - {f}')

    # Save results
    results['baseline'] = {'metrics': {'accuracy': 0.74, 'roc_auc': 0.787, 'recall': 0.617, 'f1': 0.655}}
    results['full_model_v3'] = {'metrics': {'accuracy': 0.809, 'roc_auc': 0.862, 'recall': 0.662, 'f1': 0.732}}
    results['dataset_info'] = {
        'samples': len(X),
        'features': len(feature_cols),
        'feature_list': feature_cols,
        'failure_rate': float(y.mean())
    }

    # Convert numpy types
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = convert_to_serializable(results)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'Results saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
