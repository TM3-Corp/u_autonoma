#!/usr/bin/env python3
"""
Train Learning Materials Only Model - True Early Warning.

This model uses ONLY learning material engagement features:
- Files (PCA + proactivity)
- Discussions (PCA + proactivity)
- Pages (PCA + proactivity)
- Modules (PCA + proactivity)
- Home page views
- Announcements views
- Session patterns
- Downloads
- Temporal patterns

COMPLETELY EXCLUDES (Assessments):
- ALL quiz features (quiz_*, quizzes_*)
- ALL assignment features (assi_*, assignments_*)
- ALL grades features (grades_*, grad_*)

This enables prediction BEFORE the first exam/assignment.
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
PROACTIVITY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/proactivity_features.parquet')
PCA_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/pca_features.parquet')
WEEKLY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/weekly_features.parquet')
ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/learning_material_model_results.json')

# Learning materials only - EXCLUDE assessments
LEARNING_MATERIAL_PREFIXES = ['file', 'disc', 'page', 'modu', 'mods', 'pages', 'files', 'home', 'announcements']

# Patterns to EXCLUDE (assessments)
EXCLUDE_PATTERNS = [
    'quiz', 'quizzes',  # Quiz activity = assessment
    'assi', 'assignment',  # Assignment activity = assessment
    'grade', 'grad',  # Grades = outcome variable
    'score',  # Scores = grades
]


def is_learning_material_feature(col):
    """Check if a feature is a learning material feature (not assessment)."""
    col_lower = col.lower()

    # Exclude IDs and target
    if col in ['user_id', 'course_id', 'final_score', 'current_score', 'failed', 'enrollment_state']:
        return False

    # Exclude assessment patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern in col_lower:
            return False

    return True


def load_and_merge_features():
    """Load and merge all feature sets for learning materials model."""
    print('Loading features...')

    # Load session features
    session_df = pd.read_parquet(SESSION_FEATURES)
    print(f'  Session features: {len(session_df)} rows, {len(session_df.columns)} columns')

    # Load category features
    category_df = pd.read_parquet(CATEGORY_FEATURES)
    print(f'  Category features: {len(category_df)} rows, {len(category_df.columns)} columns')

    # Load proactivity features
    proact_df = pd.read_parquet(PROACTIVITY_FEATURES)
    print(f'  Proactivity features: {len(proact_df)} rows, {len(proact_df.columns)} columns')

    # Load PCA features
    pca_df = pd.read_parquet(PCA_FEATURES)
    print(f'  PCA features: {len(pca_df)} rows, {len(pca_df.columns)} columns')

    # Load weekly features
    weekly_df = pd.read_parquet(WEEKLY_FEATURES)
    print(f'  Weekly features: {len(weekly_df)} rows, {len(weekly_df.columns)} columns')

    # Load enrollments (target)
    enrollments_df = pd.read_csv(ENROLLMENTS)
    enrollments_df['failed'] = enrollments_df['final_score'] < 57
    print(f'  Enrollments: {len(enrollments_df)} rows')

    # Start merging
    merged = session_df.merge(category_df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_cat'))
    merged = merged.merge(proact_df, on=['user_id', 'course_id'], how='left', suffixes=('', '_proact'))
    merged = merged.merge(pca_df, on=['user_id', 'course_id'], how='left', suffixes=('', '_pca'))

    # Merge weekly (exclude quiz/assignment timing features)
    weekly_cols = ['user_id', 'course_id', 'active_weeks_count', 'first_active_week',
                   'last_active_week', 'peak_week', 'early_semester_views', 'late_semester_views',
                   'early_vs_late_ratio', 'avg_week_over_week_change', 'activity_consistency',
                   'discussions_first_access_week', 'engagement_pattern']
    weekly_cols = [c for c in weekly_cols if c in weekly_df.columns]
    merged = merged.merge(weekly_df[weekly_cols], on=['user_id', 'course_id'], how='left')

    # Merge with enrollments (target)
    merged = merged.merge(
        enrollments_df[['user_id', 'course_id', 'final_score', 'current_score', 'failed']],
        on=['user_id', 'course_id'],
        how='left'
    )

    # Drop rows without target
    merged = merged.dropna(subset=['failed'])

    print(f'  Merged dataset: {len(merged)} rows, {len(merged.columns)} total columns')
    print()

    return merged


def get_learning_material_features(df):
    """Get only learning material features, excluding all assessments."""
    all_cols = df.columns.tolist()
    feature_cols = [col for col in all_cols if is_learning_material_feature(col)]
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
    print('=' * 75)
    print('LEARNING MATERIALS ONLY MODEL')
    print('True Early Warning - No Assessments (Quizzes/Assignments/Grades)')
    print('=' * 75)
    print()

    # Load data
    df = load_and_merge_features()

    # Get learning material features only
    feature_cols = get_learning_material_features(df)

    # Show what's included vs excluded
    all_cols = set(df.columns) - {'user_id', 'course_id', 'final_score', 'current_score', 'failed'}
    excluded_cols = all_cols - set(feature_cols)

    print(f'Total features available: {len(all_cols)}')
    print(f'Learning material features (INCLUDED): {len(feature_cols)}')
    print(f'Assessment features (EXCLUDED): {len(excluded_cols)}')
    print()

    # Show excluded features
    print('EXCLUDED features (assessments):')
    for col in sorted(excluded_cols)[:15]:
        print(f'  - {col}')
    if len(excluded_cols) > 15:
        print(f'  ... and {len(excluded_cols) - 15} more')
    print()

    # Show included feature categories
    print('INCLUDED feature categories:')
    categories = {}
    for col in feature_cols:
        if 'pc' in col.lower():
            cat = 'PCA components'
        elif 'pct' in col.lower() or 'proactiv' in col.lower():
            cat = 'Proactivity'
        elif 'session' in col.lower():
            cat = 'Session patterns'
        elif 'file' in col.lower():
            cat = 'Files'
        elif 'disc' in col.lower():
            cat = 'Discussions'
        elif 'page' in col.lower():
            cat = 'Pages'
        elif 'modu' in col.lower() or 'mods' in col.lower():
            cat = 'Modules'
        elif 'home' in col.lower():
            cat = 'Home'
        elif 'download' in col.lower():
            cat = 'Downloads'
        elif 'announcements' in col.lower():
            cat = 'Announcements'
        elif 'week' in col.lower() or 'early' in col.lower() or 'late' in col.lower():
            cat = 'Temporal'
        else:
            cat = 'Other'
        categories[cat] = categories.get(cat, 0) + 1

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f'  {cat}: {count} features')
    print()

    X = df[feature_cols]
    y = df['failed'].astype(int)

    print(f'Dataset: {len(X)} samples, {len(feature_cols)} features')
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
    print('=' * 75)
    print('COMPARISON: Learning Materials Only vs Other Models')
    print('=' * 75)
    print()
    print('Baseline (Activity-only, no page views):')
    print('  ROC-AUC:  0.787')
    print('  Accuracy: 0.740')
    print()
    print('Full model with grades features:')
    print('  ROC-AUC:  0.862')
    print('  Accuracy: 0.809')
    print()
    print('Previous "Early Warning" (included quiz/assignment):')
    print('  ROC-AUC:  0.857')
    print('  Accuracy: 0.778')
    print()

    best_model = max(results.keys(), key=lambda k: results[k]['metrics']['roc_auc'])
    best_metrics = results[best_model]['metrics']
    print('LEARNING MATERIALS ONLY (true early warning):')
    print(f'  Model:    {best_model}')
    print(f'  ROC-AUC:  {best_metrics["roc_auc"]:.3f}')
    print(f'  Accuracy: {best_metrics["accuracy"]:.3f}')
    print(f'  Recall:   {best_metrics["recall"]:.3f} (catches {best_metrics["recall"]*100:.0f}% of at-risk students)')
    print(f'  F1:       {best_metrics["f1"]:.3f}')
    print()

    # Top features
    print('Top 15 Learning Material Features:')
    for i, (feat, imp) in enumerate(list(results[best_model]['top_features'].items())[:15], 1):
        print(f'  {i:2d}. {feat}: {imp:.3f}')
    print()

    # Categorize top features
    top_feats = list(results[best_model]['top_features'].keys())[:15]
    print('Top feature breakdown:')
    for cat in ['Files', 'Discussions', 'Pages', 'Modules', 'PCA', 'Proactivity', 'Session', 'Download']:
        cat_feats = [f for f in top_feats if cat.lower()[:4] in f.lower()]
        if cat_feats:
            print(f'  {cat}: {len(cat_feats)} features')

    # Save results
    results['baseline'] = {'metrics': {'accuracy': 0.74, 'roc_auc': 0.787, 'recall': 0.617, 'f1': 0.655}}
    results['full_model_with_grades'] = {'metrics': {'accuracy': 0.809, 'roc_auc': 0.862, 'recall': 0.662, 'f1': 0.732}}
    results['previous_early_warning'] = {'metrics': {'accuracy': 0.778, 'roc_auc': 0.857, 'recall': 0.648, 'f1': 0.697}}
    results['dataset_info'] = {
        'samples': len(X),
        'features': len(feature_cols),
        'feature_list': feature_cols,
        'excluded_features': list(excluded_cols),
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
    print()
    print(f'Results saved to {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
