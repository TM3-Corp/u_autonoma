#!/usr/bin/env python3
"""
Prediction Model for Student Pass/Fail using Tier 1 Activity Features.

Features used (from student_summaries):
- tardiness_missing: Number of missing assignments
- on_time: Number of on-time submissions
- participations: Total participation events
- page_views: Total page views (Canvas summary)

Usage:
    python prediction_model.py --course-dir exploratory/data/courses/course_86676
    python prediction_model.py --course-dir exploratory/data/courses/course_86676 --save-model
"""

import argparse
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error
)

# Tier 1 features (highest correlation with final_score)
TIER1_FEATURES = ['tardiness_missing', 'on_time', 'participations', 'page_views']

# Pass threshold (Chilean grading scale)
PASS_THRESHOLD = 57.0


def load_data(course_dir: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and prepare data from course directory."""
    # Try consolidated CSV first, fall back to parquet files
    consolidated_path = os.path.join(course_dir, 'student_consolidated.csv')

    if os.path.exists(consolidated_path):
        df = pd.read_csv(consolidated_path)
    else:
        # Load from parquet files
        enrollments = pd.read_parquet(os.path.join(course_dir, 'enrollments.parquet'))
        summaries = pd.read_parquet(os.path.join(course_dir, 'student_summaries.parquet'))

        # Merge
        df = enrollments.merge(summaries, on=['user_id', 'course_id'], how='left')

    # Prepare features and targets
    X = df[TIER1_FEATURES].copy().fillna(0)
    y_class = (df['final_score'] >= PASS_THRESHOLD).astype(int)
    y_reg = df['final_score']

    return X, y_class, y_reg


def train_classification_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train and evaluate classification model using LOO-CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    lr = LogisticRegression(random_state=42, max_iter=1000)
    y_pred = cross_val_predict(lr, X_scaled, y, cv=loo)

    # Metrics
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=['FAIL', 'PASS'], output_dict=True)

    # Train final model
    lr.fit(X_scaled, y)

    # Feature importance from Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=3)
    rf.fit(X_scaled, y)
    feature_importance = dict(zip(TIER1_FEATURES, rf.feature_importances_))

    return {
        'model': lr,
        'scaler': scaler,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'classification_report': report,
        'feature_importance': feature_importance,
        'predictions': y_pred
    }


def train_regression_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train and evaluate regression model using LOO-CV."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Leave-One-Out Cross-Validation
    loo = LeaveOneOut()
    ridge = Ridge(alpha=1.0)
    y_pred = cross_val_predict(ridge, X_scaled, y, cv=loo)

    # Metrics
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    # Train final model
    ridge.fit(X_scaled, y)
    coefficients = dict(zip(TIER1_FEATURES, ridge.coef_))

    return {
        'model': ridge,
        'scaler': scaler,
        'r2_score': r2,
        'mae': mae,
        'coefficients': coefficients,
        'intercept': ridge.intercept_,
        'predictions': y_pred
    }


def print_results(class_results: Dict, reg_results: Dict, y_class: pd.Series):
    """Print model evaluation results."""
    print('=' * 60)
    print('PREDICTION MODEL RESULTS - TIER 1 FEATURES')
    print('=' * 60)
    print(f'\nFeatures: {TIER1_FEATURES}')
    print(f'Samples: {len(y_class)}')
    print(f'Class distribution: PASS={y_class.sum()}, FAIL={len(y_class) - y_class.sum()}')

    print('\n' + '=' * 60)
    print('1. CLASSIFICATION (PASS/FAIL)')
    print('=' * 60)

    print(f'\nLOO-CV Accuracy: {class_results["accuracy"]:.1%}')

    cm = class_results['confusion_matrix']
    print(f'\nConfusion Matrix:')
    print(f'                 Predicted')
    print(f'                 FAIL  PASS')
    print(f'  Actual FAIL    {cm[0,0]:>4}  {cm[0,1]:>4}')
    print(f'  Actual PASS    {cm[1,0]:>4}  {cm[1,1]:>4}')

    report = class_results['classification_report']
    print(f'\nPrecision/Recall:')
    print(f'  FAIL: precision={report["FAIL"]["precision"]:.2f}, recall={report["FAIL"]["recall"]:.2f}, f1={report["FAIL"]["f1-score"]:.2f}')
    print(f'  PASS: precision={report["PASS"]["precision"]:.2f}, recall={report["PASS"]["recall"]:.2f}, f1={report["PASS"]["f1-score"]:.2f}')

    print(f'\nFeature Importance:')
    for feat, imp in sorted(class_results['feature_importance'].items(), key=lambda x: -x[1]):
        print(f'  {feat:<20} {imp:.3f}')

    print('\n' + '=' * 60)
    print('2. REGRESSION (final_score)')
    print('=' * 60)

    print(f'\nLOO-CV R² Score: {reg_results["r2_score"]:.3f}')
    print(f'LOO-CV MAE: {reg_results["mae"]:.2f} percentage points')

    print(f'\nCoefficients (standardized):')
    for feat, coef in sorted(reg_results['coefficients'].items(), key=lambda x: -abs(x[1])):
        print(f'  {feat:<20} {coef:>8.2f}')

    print('\n' + '=' * 60)
    print('3. SUMMARY')
    print('=' * 60)

    baseline_acc = max(y_class.mean(), 1 - y_class.mean())
    improvement = (class_results['accuracy'] / baseline_acc - 1) * 100

    print(f'\nClassification:')
    print(f'  Model Accuracy: {class_results["accuracy"]:.1%}')
    print(f'  Baseline (always FAIL): {baseline_acc:.1%}')
    print(f'  Improvement: {improvement:+.1f}%')

    print(f'\nRegression:')
    print(f'  R² Score: {reg_results["r2_score"]:.3f} ({reg_results["r2_score"]*100:.1f}% variance explained)')
    print(f'  MAE: {reg_results["mae"]:.1f} percentage points')


def save_model(class_results: Dict, reg_results: Dict, output_dir: str):
    """Save trained models to disk."""
    model_data = {
        'classification': {
            'model': class_results['model'],
            'scaler': class_results['scaler'],
            'accuracy': class_results['accuracy']
        },
        'regression': {
            'model': reg_results['model'],
            'scaler': reg_results['scaler'],
            'r2_score': reg_results['r2_score']
        },
        'features': TIER1_FEATURES,
        'pass_threshold': PASS_THRESHOLD
    }

    model_path = os.path.join(output_dir, 'prediction_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f'\nModel saved to: {model_path}')


def main():
    parser = argparse.ArgumentParser(description='Train prediction model using Tier 1 features')
    parser.add_argument('--course-dir', type=str, required=True,
                        help='Path to course data directory')
    parser.add_argument('--save-model', action='store_true',
                        help='Save trained model to disk')

    args = parser.parse_args()

    # Load data
    X, y_class, y_reg = load_data(args.course_dir)

    # Train models
    class_results = train_classification_model(X, y_class)
    reg_results = train_regression_model(X, y_reg)

    # Print results
    print_results(class_results, reg_results, y_class)

    # Save model if requested
    if args.save_model:
        save_model(class_results, reg_results, args.course_dir)


if __name__ == '__main__':
    main()
