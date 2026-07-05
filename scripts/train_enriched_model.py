#!/usr/bin/env python3
"""
Train predictive model with enriched features from page views.

Compares:
- Baseline model (activity-only from analytics API)
- Enriched model (session + category features from page views)

Target: Binary classification (failed = final_score < 57% or grade < 4.0)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import json
import warnings
warnings.filterwarnings('ignore')

# Input files
SESSION_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/session_features.parquet')
CATEGORY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/category_features.parquet')
ENGAGEMENT_RATIOS = Path('/home/paul/projects/uautonoma/data/enriched_features/engagement_ratios.parquet')
WEEKLY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/weekly_features.parquet')
ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/model_results_v3.json')


def load_and_merge_features():
    """Load and merge all feature sets."""
    print('Loading features...')

    # Load session features
    session_df = pd.read_parquet(SESSION_FEATURES)
    print(f'  Session features: {len(session_df)} rows, {len(session_df.columns)} columns')

    # Load category features
    category_df = pd.read_parquet(CATEGORY_FEATURES)
    print(f'  Category features: {len(category_df)} rows, {len(category_df.columns)} columns')

    # Load engagement ratios
    if ENGAGEMENT_RATIOS.exists():
        ratios_df = pd.read_parquet(ENGAGEMENT_RATIOS)
        print(f'  Engagement ratios: {len(ratios_df)} rows, {len(ratios_df.columns)} columns')
    else:
        ratios_df = None
        print('  Engagement ratios: NOT FOUND')

    # Load weekly features
    if WEEKLY_FEATURES.exists():
        weekly_df = pd.read_parquet(WEEKLY_FEATURES)
        print(f'  Weekly features: {len(weekly_df)} rows, {len(weekly_df.columns)} columns')
    else:
        weekly_df = None
        print('  Weekly features: NOT FOUND')

    # Load enrollments (has grades)
    enrollments_df = pd.read_csv(ENROLLMENTS)
    print(f'  Enrollments: {len(enrollments_df)} rows')

    # Create target variable
    enrollments_df['failed'] = enrollments_df['final_score'] < 57

    # Merge features
    merged = session_df.merge(
        category_df,
        on=['user_id', 'course_id'],
        how='outer',
        suffixes=('_session', '_category')
    )

    # Merge engagement ratios if available
    if ratios_df is not None:
        # Select only the ratio columns (avoid duplicating views/time columns)
        ratio_cols = ['user_id', 'course_id', 'total_sessions', 'total_views', 'total_time_min',
                     'course_session_ratio', 'course_views_ratio', 'course_time_ratio']
        ratio_cols = [c for c in ratio_cols if c in ratios_df.columns]
        merged = merged.merge(
            ratios_df[ratio_cols],
            on=['user_id', 'course_id'],
            how='left'
        )

    # Merge weekly features if available
    if weekly_df is not None:
        # Select temporal features (avoid duplicating total_views, total_sessions)
        weekly_cols = ['user_id', 'course_id', 'active_weeks_count', 'first_active_week',
                      'last_active_week', 'peak_week', 'early_semester_views', 'late_semester_views',
                      'early_vs_late_ratio', 'avg_week_over_week_change', 'activity_consistency',
                      'quizzes_first_access_week', 'assignments_first_access_week',
                      'discussions_first_access_week', 'grades_first_access_week', 'engagement_pattern']
        weekly_cols = [c for c in weekly_cols if c in weekly_df.columns]
        merged = merged.merge(
            weekly_df[weekly_cols],
            on=['user_id', 'course_id'],
            how='left'
        )

    # Merge with enrollments
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


def get_feature_columns(df):
    """Get list of feature columns (excluding IDs and target)."""
    exclude = ['user_id', 'course_id', 'final_score', 'current_score', 'failed', 'enrollment_state']
    return [c for c in df.columns if c not in exclude]


def train_and_evaluate(X, y, model_name='XGBoost'):
    """Train model with cross-validation and return metrics."""
    # Create pipeline
    if model_name == 'XGBoost':
        model = XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)

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

    return metrics, feature_importance, pipeline


def main():
    print('=' * 60)
    print('Training Enriched Predictive Model')
    print('=' * 60)
    print()

    # Load data
    df = load_and_merge_features()

    # Get features
    feature_cols = get_feature_columns(df)
    print(f'Total features: {len(feature_cols)}')
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
        metrics, importance, pipeline = train_and_evaluate(X, y, model_name)

        results[model_name] = {
            'metrics': metrics,
            'top_features': dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15])
        }

        print(f'  Accuracy: {metrics["accuracy"]:.3f}')
        print(f'  ROC-AUC:  {metrics["roc_auc"]:.3f}')
        print(f'  Recall:   {metrics["recall"]:.3f}')
        print(f'  F1:       {metrics["f1"]:.3f}')
        print()

    # Compare with baseline
    print('=' * 60)
    print('COMPARISON WITH BASELINE')
    print('=' * 60)
    print()
    print('Baseline (Activity-only model):')
    print('  Accuracy: 0.740')
    print('  ROC-AUC:  0.787')
    print('  Recall:   0.617')
    print('  F1:       0.655')
    print()
    print('Enriched model (Best):')
    best_model = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'])
    best_metrics = results[best_model]['metrics']
    print(f'  Model:    {best_model}')
    print(f'  Accuracy: {best_metrics["accuracy"]:.3f} ({"+" if best_metrics["accuracy"] > 0.74 else ""}{(best_metrics["accuracy"] - 0.74)*100:.1f}%)')
    print(f'  ROC-AUC:  {best_metrics["roc_auc"]:.3f} ({"+" if best_metrics["roc_auc"] > 0.787 else ""}{(best_metrics["roc_auc"] - 0.787)*100:.1f}%)')
    print(f'  Recall:   {best_metrics["recall"]:.3f}')
    print(f'  F1:       {best_metrics["f1"]:.3f}')
    print()

    # Top features
    print('Top 10 Features:')
    for i, (feat, imp) in enumerate(list(results[best_model]['top_features'].items())[:10], 1):
        print(f'  {i}. {feat}: {imp:.3f}')
    print()

    # Save results
    results['baseline'] = {
        'metrics': {'accuracy': 0.74, 'roc_auc': 0.787, 'recall': 0.617, 'f1': 0.655}
    }
    results['dataset_info'] = {
        'samples': len(X),
        'features': len(feature_cols),
        'failure_rate': float(y.mean())
    }

    # Convert numpy types to Python types for JSON serialization
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
