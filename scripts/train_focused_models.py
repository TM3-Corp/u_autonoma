"""
Focused Model Training: 2 Completed Courses (77 students)

This script trains and compares two model types on the completed courses
with real class diversity (FUNDAMENTOS DE MICROECONOMÍA P01 & P03):

1. ACTIVITY-ONLY: Pure LMS engagement data
   - page_views, participations, total_activity_time
   - page_views_level, participations_level
   - NO grades, NO submission/deadline info

2. ALL-DATA: Including submission/assignment features
   - Activity features + submission counts, scores, tardiness

Uses cross-validation for more reliable estimates on small dataset.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import API_URL, API_TOKEN, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
PASS_THRESHOLD = 57.0

# Only completed courses with real failures
COMPLETED_COURSES = {
    84936: {'name': 'FUNDAMENTOS DE MICROECONOMÍA-P03', 'fail_rate': 0.27},
    84941: {'name': 'FUNDAMENTOS DE MICROECONOMÍA-P01', 'fail_rate': 0.61},
}

# Feature definitions
ACTIVITY_ONLY_FEATURES = [
    'page_views',
    'participations',
    'total_activity_time',
    'page_views_level',
    'participations_level',
]

ALL_DATA_FEATURES = ACTIVITY_ONLY_FEATURES + [
    'on_time', 'late', 'missing', 'floating',
    'on_time_rate', 'late_rate', 'missing_rate',
    'num_submissions', 'submission_rate',
    'avg_score', 'min_score', 'max_score', 'score_std',
    'first_score', 'num_graded', 'num_scores',
]


def load_course_data(course_id: int) -> pd.DataFrame:
    """Load and build features from previously extracted course data."""
    raw_path = os.path.join(DATA_DIR, 'baseline', f'course_{course_id}_raw.json')

    with open(raw_path, 'r') as f:
        data = json.load(f)

    enrollments = data['enrollments']
    summaries = data['student_summaries']
    submissions = data['submissions']
    assignments = data['assignments']

    # Build base dataframe from enrollments
    df = pd.DataFrame([
        {
            'user_id': e['user_id'],
            'course_id': course_id,
            'final_score': e.get('grades', {}).get('final_score'),
            'total_activity_time': e.get('total_activity_time', 0) or 0,
        }
        for e in enrollments
        if e.get('type') == 'StudentEnrollment'
    ])

    if df.empty:
        return df

    # Add activity data from summaries
    summary_df = pd.DataFrame([
        {
            'user_id': s['id'],
            'page_views': s.get('page_views', 0) or 0,
            'page_views_level': s.get('page_views_level', 0) or 0,
            'participations': s.get('participations', 0) or 0,
            'participations_level': s.get('participations_level', 0) or 0,
            'on_time': s.get('tardiness_breakdown', {}).get('on_time', 0) or 0,
            'late': s.get('tardiness_breakdown', {}).get('late', 0) or 0,
            'missing': s.get('tardiness_breakdown', {}).get('missing', 0) or 0,
            'floating': s.get('tardiness_breakdown', {}).get('floating', 0) or 0,
        }
        for s in summaries
    ])

    if not summary_df.empty:
        df = df.merge(summary_df, on='user_id', how='left')

    # Process submissions per user
    sub_df = pd.DataFrame(submissions)
    if not sub_df.empty and 'user_id' in sub_df.columns:
        user_submissions = sub_df.groupby('user_id').agg({
            'score': ['count', 'mean', 'min', 'max', 'std', 'first'],
            'workflow_state': lambda x: (x == 'graded').sum()
        }).reset_index()

        user_submissions.columns = [
            'user_id', 'num_submissions', 'avg_score', 'min_score', 'max_score',
            'score_std', 'first_score', 'num_graded'
        ]

        scores_df = sub_df[sub_df['score'].notna()].groupby('user_id').size().reset_index(name='num_scores')
        user_submissions = user_submissions.merge(scores_df, on='user_id', how='left')
        user_submissions['num_scores'] = user_submissions['num_scores'].fillna(0)

        df = df.merge(user_submissions, on='user_id', how='left')

    # Calculate derived features
    total_assignments = len(assignments) if assignments else 1
    df['submission_rate'] = df.get('num_submissions', 0).fillna(0) / max(total_assignments, 1)

    # Tardiness rates
    total_required = df['on_time'].fillna(0) + df['late'].fillna(0) + df['missing'].fillna(0)
    df['on_time_rate'] = np.where(total_required > 0, df['on_time'] / total_required, 0)
    df['late_rate'] = np.where(total_required > 0, df['late'] / total_required, 0)
    df['missing_rate'] = np.where(total_required > 0, df['missing'] / total_required, 0)

    # Fill NaN values
    for col in ALL_DATA_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Set target
    df['target'] = df['final_score']
    df['passed'] = (df['final_score'] >= PASS_THRESHOLD).astype(int)

    return df


def evaluate_regression_cv(X: np.ndarray, y: np.ndarray, model, cv: int = 5) -> Dict:
    """Evaluate regression model using cross-validation."""
    # R² scores from cross-validation
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')

    # Get predictions for all folds
    y_pred = cross_val_predict(model, X, y, cv=cv)

    return {
        'r2_mean': float(np.mean(r2_scores)),
        'r2_std': float(np.std(r2_scores)),
        'r2_scores': r2_scores.tolist(),
        'rmse': float(np.sqrt(mean_squared_error(y, y_pred))),
        'mae': float(mean_absolute_error(y, y_pred)),
    }


def evaluate_classification_cv(X: np.ndarray, y: np.ndarray, model, cv: int = 5) -> Dict:
    """Evaluate classification model using cross-validation."""
    # Use stratified k-fold for classification
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Get predictions
    y_pred = cross_val_predict(model, X, y, cv=skf)

    # Cross-validation scores
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')

    return {
        'accuracy_mean': float(np.mean(accuracy_scores)),
        'accuracy_std': float(np.std(accuracy_scores)),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'precision': float(precision_score(y, y_pred, zero_division=0)),
        'recall': float(recall_score(y, y_pred, zero_division=0)),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist(),
    }


def train_and_evaluate(df: pd.DataFrame, feature_list: List[str], model_name: str) -> Dict:
    """Train models and evaluate using cross-validation."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Training {model_name} model")
    logger.info(f"Features: {feature_list}")
    logger.info(f"{'='*50}")

    # Prepare data
    df_valid = df[df['target'].notna()].copy()

    available_features = [f for f in feature_list if f in df_valid.columns]
    X = df_valid[available_features].fillna(0).values
    y_reg = df_valid['target'].values
    y_clf = df_valid['passed'].values

    logger.info(f"Samples: {len(df_valid)}")
    logger.info(f"Features used: {len(available_features)}")
    logger.info(f"Class balance: {sum(y_clf)} pass / {len(y_clf) - sum(y_clf)} fail")

    results = {
        'model_type': model_name,
        'n_samples': len(df_valid),
        'n_features': len(available_features),
        'features_used': available_features,
        'class_balance': {
            'pass': int(sum(y_clf)),
            'fail': int(len(y_clf) - sum(y_clf)),
            'pass_rate': float(np.mean(y_clf))
        }
    }

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Regression models
    logger.info("\nRegression models (predicting final_score):")
    results['regression'] = {}

    # Linear Regression
    lr = LinearRegression()
    lr_results = evaluate_regression_cv(X_scaled, y_reg, lr, cv=5)
    results['regression']['linear'] = lr_results
    logger.info(f"  Linear:        R² = {lr_results['r2_mean']:.3f} ± {lr_results['r2_std']:.3f}")

    # Ridge Regression (better for multicollinearity)
    ridge = Ridge(alpha=1.0)
    ridge_results = evaluate_regression_cv(X_scaled, y_reg, ridge, cv=5)
    results['regression']['ridge'] = ridge_results
    logger.info(f"  Ridge:         R² = {ridge_results['r2_mean']:.3f} ± {ridge_results['r2_std']:.3f}")

    # Random Forest Regression
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_results = evaluate_regression_cv(X, y_reg, rf_reg, cv=5)
    results['regression']['random_forest'] = rf_results
    logger.info(f"  RandomForest:  R² = {rf_results['r2_mean']:.3f} ± {rf_results['r2_std']:.3f}")

    # Feature importance from RF
    rf_reg.fit(X, y_reg)
    results['regression']['feature_importance'] = [
        {'feature': f, 'importance': float(i)}
        for f, i in sorted(zip(available_features, rf_reg.feature_importances_),
                          key=lambda x: x[1], reverse=True)
    ]

    # Classification models
    logger.info("\nClassification models (predicting pass/fail):")
    results['classification'] = {}

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_results = evaluate_classification_cv(X_scaled, y_clf, log_reg, cv=5)
    results['classification']['logistic'] = log_results
    logger.info(f"  Logistic:      F1 = {log_results['f1_mean']:.3f} ± {log_results['f1_std']:.3f}")

    # Random Forest Classification
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_clf_results = evaluate_classification_cv(X, y_clf, rf_clf, cv=5)
    results['classification']['random_forest'] = rf_clf_results
    logger.info(f"  RandomForest:  F1 = {rf_clf_results['f1_mean']:.3f} ± {rf_clf_results['f1_std']:.3f}")

    # Feature importance from RF classifier
    rf_clf.fit(X, y_clf)
    results['classification']['feature_importance'] = [
        {'feature': f, 'importance': float(i)}
        for f, i in sorted(zip(available_features, rf_clf.feature_importances_),
                          key=lambda x: x[1], reverse=True)
    ]

    return results


def analyze_feature_correlations(df: pd.DataFrame, features: List[str]) -> Dict:
    """Analyze feature correlations and multicollinearity."""
    available_features = [f for f in features if f in df.columns]
    X = df[available_features].fillna(0)

    # Correlation matrix
    corr_matrix = X.corr()

    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > 0.7:
                high_corr_pairs.append({
                    'feature1': available_features[i],
                    'feature2': available_features[j],
                    'correlation': float(corr)
                })

    # Correlation with target
    target_corr = df[available_features + ['target']].corr()['target'].drop('target')

    return {
        'high_correlation_pairs': high_corr_pairs,
        'target_correlations': [
            {'feature': f, 'correlation': float(c)}
            for f, c in sorted(target_corr.items(), key=lambda x: abs(x[1]), reverse=True)
        ]
    }


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("FOCUSED MODEL TRAINING: 2 COMPLETED COURSES (77 STUDENTS)")
    logger.info("="*70)
    logger.info("\nComparing two model types:")
    logger.info("  1. ACTIVITY-ONLY: Pure LMS engagement (no grades/submissions)")
    logger.info("  2. ALL-DATA: Including submission/assignment features")
    logger.info("="*70)

    # Load data from both courses
    logger.info("\nLoading course data...")
    all_dfs = []

    for course_id, config in COMPLETED_COURSES.items():
        logger.info(f"  Loading {config['name']} (ID: {course_id})...")
        df = load_course_data(course_id)
        df['course_name'] = config['name']
        all_dfs.append(df)
        logger.info(f"    -> {len(df)} students, {df['passed'].sum()} pass / {len(df) - df['passed'].sum()} fail")

    # Combine datasets
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_df = combined_df[combined_df['target'].notna()]

    logger.info(f"\nCombined dataset: {len(combined_df)} students")
    logger.info(f"  Pass: {combined_df['passed'].sum()} ({combined_df['passed'].mean()*100:.1f}%)")
    logger.info(f"  Fail: {len(combined_df) - combined_df['passed'].sum()} ({(1-combined_df['passed'].mean())*100:.1f}%)")
    logger.info(f"  Target mean: {combined_df['target'].mean():.1f}%")
    logger.info(f"  Target std: {combined_df['target'].std():.1f}%")

    results = {
        'analysis_time': datetime.now().isoformat(),
        'dataset': {
            'courses': list(COMPLETED_COURSES.keys()),
            'n_students': len(combined_df),
            'pass_count': int(combined_df['passed'].sum()),
            'fail_count': int(len(combined_df) - combined_df['passed'].sum()),
            'pass_rate': float(combined_df['passed'].mean()),
            'target_mean': float(combined_df['target'].mean()),
            'target_std': float(combined_df['target'].std()),
        },
        'models': {}
    }

    # Analyze correlations
    logger.info("\n" + "="*50)
    logger.info("FEATURE CORRELATION ANALYSIS")
    logger.info("="*50)

    # Activity features correlations
    activity_corr = analyze_feature_correlations(combined_df, ACTIVITY_ONLY_FEATURES)
    results['activity_correlations'] = activity_corr

    logger.info("\nActivity features - correlations with target (final_score):")
    for item in activity_corr['target_correlations']:
        logger.info(f"  {item['feature']}: r = {item['correlation']:.3f}")

    if activity_corr['high_correlation_pairs']:
        logger.info("\nHighly correlated pairs (|r| > 0.7):")
        for pair in activity_corr['high_correlation_pairs']:
            logger.info(f"  {pair['feature1']} <-> {pair['feature2']}: r = {pair['correlation']:.3f}")

    # All-data correlations
    all_data_corr = analyze_feature_correlations(combined_df, ALL_DATA_FEATURES)
    results['all_data_correlations'] = all_data_corr

    logger.info("\nAll-data features - top correlations with target:")
    for item in all_data_corr['target_correlations'][:10]:
        logger.info(f"  {item['feature']}: r = {item['correlation']:.3f}")

    # Train ACTIVITY-ONLY model
    activity_results = train_and_evaluate(combined_df, ACTIVITY_ONLY_FEATURES, 'ACTIVITY-ONLY')
    results['models']['activity_only'] = activity_results

    # Train ALL-DATA model
    all_data_results = train_and_evaluate(combined_df, ALL_DATA_FEATURES, 'ALL-DATA')
    results['models']['all_data'] = all_data_results

    # Summary comparison
    logger.info("\n" + "="*70)
    logger.info("SUMMARY COMPARISON")
    logger.info("="*70)

    logger.info("\nRegression (predicting final_score):")
    logger.info(f"                        ACTIVITY-ONLY    ALL-DATA")
    logger.info(f"  Linear R²:            {activity_results['regression']['linear']['r2_mean']:.3f} ± {activity_results['regression']['linear']['r2_std']:.3f}      {all_data_results['regression']['linear']['r2_mean']:.3f} ± {all_data_results['regression']['linear']['r2_std']:.3f}")
    logger.info(f"  Ridge R²:             {activity_results['regression']['ridge']['r2_mean']:.3f} ± {activity_results['regression']['ridge']['r2_std']:.3f}      {all_data_results['regression']['ridge']['r2_mean']:.3f} ± {all_data_results['regression']['ridge']['r2_std']:.3f}")
    logger.info(f"  RandomForest R²:      {activity_results['regression']['random_forest']['r2_mean']:.3f} ± {activity_results['regression']['random_forest']['r2_std']:.3f}      {all_data_results['regression']['random_forest']['r2_mean']:.3f} ± {all_data_results['regression']['random_forest']['r2_std']:.3f}")

    logger.info("\nClassification (predicting pass/fail):")
    logger.info(f"                        ACTIVITY-ONLY    ALL-DATA")
    logger.info(f"  Logistic F1:          {activity_results['classification']['logistic']['f1_mean']:.3f} ± {activity_results['classification']['logistic']['f1_std']:.3f}      {all_data_results['classification']['logistic']['f1_mean']:.3f} ± {all_data_results['classification']['logistic']['f1_std']:.3f}")
    logger.info(f"  RandomForest F1:      {activity_results['classification']['random_forest']['f1_mean']:.3f} ± {activity_results['classification']['random_forest']['f1_std']:.3f}      {all_data_results['classification']['random_forest']['f1_mean']:.3f} ± {all_data_results['classification']['random_forest']['f1_std']:.3f}")

    logger.info("\nTop predictive features (Random Forest importance):")
    logger.info("\n  ACTIVITY-ONLY:")
    for item in activity_results['regression']['feature_importance'][:5]:
        logger.info(f"    {item['feature']}: {item['importance']:.3f}")

    logger.info("\n  ALL-DATA:")
    for item in all_data_results['regression']['feature_importance'][:5]:
        logger.info(f"    {item['feature']}: {item['importance']:.3f}")

    # Save results
    output_path = os.path.join(DATA_DIR, 'baseline', 'focused_models_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("="*70)

    return results


if __name__ == '__main__':
    main()
