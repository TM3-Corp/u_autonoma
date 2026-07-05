#!/usr/bin/env python3
"""
Compare Model Performance: With and Without Pre-Assessment Features.

This script:
1. Trains the baseline model (existing pure activity features)
2. Trains an enhanced model (+ pre-assessment features)
3. Compares ROC-AUC, Recall, and feature importances
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not available, using Random Forest")

# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# Baseline pure activity features (from pooled_binary_classifier.py)
BASELINE_FEATURES = [
    # Session regularity
    'session_count', 'session_gap_min', 'session_gap_max',
    'session_gap_mean', 'session_gap_std', 'session_regularity',
    'sessions_per_week',
    # Time preferences
    'weekday_morning_pct', 'weekday_afternoon_pct',
    'weekday_evening_pct', 'weekday_night_pct',
    'weekend_morning_pct', 'weekend_afternoon_pct',
    'weekend_evening_pct', 'weekend_night_pct',
    # Weekly pattern consistency
    'weekday_morning_sd', 'weekday_afternoon_sd',
    'weekday_evening_sd', 'weekday_night_sd',
    'weekend_morning_sd', 'weekend_afternoon_sd',
    'weekend_evening_sd', 'weekend_night_sd',
    'weekend_total_sd',
    # Weekly rhythm (DCT coefficients)
    'dct_coef_0', 'dct_coef_1', 'dct_coef_2', 'dct_coef_3',
    'dct_coef_4', 'dct_coef_5', 'dct_coef_6', 'dct_coef_7',
    'dct_coef_8', 'dct_coef_9', 'dct_coef_10', 'dct_coef_11',
    # Engagement trajectory
    'engagement_velocity', 'engagement_acceleration',
    'weekly_cv', 'weekly_range', 'trend_reversals',
    'early_engagement_ratio', 'late_surge',
    # Workload dynamics
    'peak_count_type1', 'peak_count_type2', 'peak_count_type3',
    'peak_ratio', 'max_positive_slope', 'max_negative_slope',
    'slope_std', 'positive_slope_sum', 'negative_slope_sum',
    # Time-to-access
    'first_access_day', 'first_module_day', 'first_assignment_day',
    'access_time_pct',
    # Raw aggregates
    'activity_span_days', 'unique_active_hours',
    'total_page_views',
]

# NEW pre-assessment features to add
PRE_ASSESSMENT_FEATURES = [
    # Deadline-relative (for courses with due dates)
    'activity_24h_before', 'activity_48h_before', 'activity_72h_before',
    'files_24h_before', 'files_48h_before', 'files_72h_before',
    'preparation_intensity',
    'activity_24h_per_deadline', 'activity_48h_per_deadline', 'activity_72h_per_deadline',
    # QAT features
    'first_quiz_access_days', 'first_assessment_access_days',
    'quiz_access_pct', 'early_quiz_accessor',
    # Assessment patterns
    'unique_quizzes_accessed', 'unique_assignments_accessed',
    'assessment_diversity', 'quiz_revisits', 'assignment_revisits',
    'quiz_sessions', 'assignment_sessions',
    # Temporal preparation
    'early_half_activity_pct', 'late_surge_ratio', 'activity_acceleration',
    'consistent_preparer',
    # File preparation
    'files_early_pct', 'files_per_week', 'files_diversity',
]


def load_merged_data():
    """Load and merge baseline features with pre-assessment features."""
    print("Loading data...")

    # Baseline features
    df_base = pd.read_csv('data/engagement_dynamics/student_features.csv')
    df_base = df_base[df_base['final_score'].notna()].copy()
    df_base['failed'] = (df_base['final_score'] < 57).astype(int)
    print(f"  Baseline: {len(df_base)} students")

    # Pre-assessment features
    df_pre = pd.read_parquet('data/enriched_features/pre_assessment_features.parquet')
    print(f"  Pre-assessment: {len(df_pre)} students")

    # Merge
    df_merged = df_base.merge(
        df_pre,
        on=['user_id', 'course_id'],
        how='inner',
        suffixes=('', '_pre')
    )
    print(f"  Merged: {len(df_merged)} students")

    # Filter to good diversity courses
    try:
        with open('data/engagement_dynamics/pure_activity_analysis.json') as f:
            course_analysis = json.load(f)
        good_courses = [
            int(c['course_id']) for c in course_analysis
            if c.get('class_diversity') == 'GOOD'
        ]
        df_merged = df_merged[df_merged['course_id'].isin(good_courses)]
        print(f"  After filtering to GOOD diversity courses: {len(df_merged)} students")
    except FileNotFoundError:
        print("  Warning: Using all courses (no diversity filter)")

    return df_merged


def prepare_features(df, feature_list):
    """Prepare feature matrix with proper handling of missing values."""
    available = [f for f in feature_list if f in df.columns]
    missing = [f for f in feature_list if f not in df.columns]

    if missing:
        print(f"  Missing features: {len(missing)} ({missing[:5]}...)")

    X = df[available].copy()

    # Handle missing/infinite values
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())
    X = X.replace([np.inf, -np.inf], 0).fillna(0)

    return X, available


def train_and_evaluate(X, y, model_name="Model"):
    """Train model with cross-validation and return metrics."""
    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Choose model
    if HAS_XGBOOST:
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    else:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            random_state=42
        )

    # Cross-validation predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y, y_pred_proba),
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
    }

    # Calculate optimal threshold metrics
    for threshold in [0.3, 0.4, 0.5]:
        y_t = (y_pred_proba >= threshold).astype(int)
        metrics[f'recall_t{int(threshold*100)}'] = recall_score(y, y_t)
        metrics[f'accuracy_t{int(threshold*100)}'] = accuracy_score(y, y_t)

    # Train final model for feature importances
    model.fit(X_scaled, y)

    return metrics, y_pred_proba, model


def get_feature_importance(model, feature_names):
    """Extract feature importances from trained model."""
    if HAS_XGBOOST:
        importances = model.feature_importances_
    else:
        importances = model.feature_importances_

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    return importance_df


def main():
    print("=" * 70)
    print("COMPARISON: Baseline vs Pre-Assessment Enhanced Model")
    print("=" * 70)
    print()

    # Load merged data
    df = load_merged_data()

    y = df['failed'].values
    print(f"\nTarget distribution: {y.sum()} failures / {len(y)} total ({y.mean()*100:.1f}%)")

    # ==========================================================================
    # BASELINE MODEL (existing features only)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODEL 1: BASELINE (Pure Activity Features)")
    print("=" * 70)

    X_base, base_features = prepare_features(df, BASELINE_FEATURES)
    print(f"Features: {len(base_features)}")

    base_metrics, base_proba, base_model = train_and_evaluate(X_base, y, "Baseline")

    print(f"\n  ROC-AUC:  {base_metrics['roc_auc']:.4f}")
    print(f"  Accuracy: {base_metrics['accuracy']:.4f}")
    print(f"  Recall:   {base_metrics['recall']:.4f}")
    print(f"  F1:       {base_metrics['f1']:.4f}")

    # ==========================================================================
    # ENHANCED MODEL (baseline + pre-assessment)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODEL 2: ENHANCED (+ Pre-Assessment Features)")
    print("=" * 70)

    all_features = BASELINE_FEATURES + PRE_ASSESSMENT_FEATURES
    X_enhanced, enhanced_features = prepare_features(df, all_features)
    print(f"Features: {len(enhanced_features)}")

    enhanced_metrics, enhanced_proba, enhanced_model = train_and_evaluate(X_enhanced, y, "Enhanced")

    print(f"\n  ROC-AUC:  {enhanced_metrics['roc_auc']:.4f}")
    print(f"  Accuracy: {enhanced_metrics['accuracy']:.4f}")
    print(f"  Recall:   {enhanced_metrics['recall']:.4f}")
    print(f"  F1:       {enhanced_metrics['f1']:.4f}")

    # ==========================================================================
    # COMPARISON
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    roc_diff = enhanced_metrics['roc_auc'] - base_metrics['roc_auc']
    recall_diff = enhanced_metrics['recall'] - base_metrics['recall']

    print(f"""
    {'Metric':<20} {'Baseline':>12} {'Enhanced':>12} {'Change':>12}
    {'-'*56}
    {'ROC-AUC':<20} {base_metrics['roc_auc']:>12.4f} {enhanced_metrics['roc_auc']:>12.4f} {roc_diff:>+12.4f}
    {'Accuracy':<20} {base_metrics['accuracy']:>12.4f} {enhanced_metrics['accuracy']:>12.4f} {enhanced_metrics['accuracy']-base_metrics['accuracy']:>+12.4f}
    {'Recall':<20} {base_metrics['recall']:>12.4f} {enhanced_metrics['recall']:>12.4f} {recall_diff:>+12.4f}
    {'Precision':<20} {base_metrics['precision']:>12.4f} {enhanced_metrics['precision']:>12.4f} {enhanced_metrics['precision']-base_metrics['precision']:>+12.4f}
    {'F1':<20} {base_metrics['f1']:>12.4f} {enhanced_metrics['f1']:>12.4f} {enhanced_metrics['f1']-base_metrics['f1']:>+12.4f}
    """)

    print(f"\n  At threshold 0.30:")
    print(f"    Baseline: Recall={base_metrics['recall_t30']:.3f}, Acc={base_metrics['accuracy_t30']:.3f}")
    print(f"    Enhanced: Recall={enhanced_metrics['recall_t30']:.3f}, Acc={enhanced_metrics['accuracy_t30']:.3f}")

    # ==========================================================================
    # FEATURE IMPORTANCE ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TOP 15 FEATURES IN ENHANCED MODEL")
    print("=" * 70)

    importance_df = get_feature_importance(enhanced_model, enhanced_features)
    print()
    for i, row in importance_df.head(15).iterrows():
        # Mark pre-assessment features
        marker = " [NEW]" if row['feature'] in PRE_ASSESSMENT_FEATURES else ""
        print(f"  {row['feature']:40s} {row['importance']:8.4f}{marker}")

    # Count new features in top 15
    top15 = importance_df.head(15)['feature'].tolist()
    new_in_top15 = [f for f in top15 if f in PRE_ASSESSMENT_FEATURES]
    print(f"\n  Pre-assessment features in top 15: {len(new_in_top15)}/15")
    if new_in_top15:
        print(f"  Specifically: {', '.join(new_in_top15)}")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    results = {
        'baseline': {
            'n_features': len(base_features),
            'metrics': base_metrics,
        },
        'enhanced': {
            'n_features': len(enhanced_features),
            'metrics': enhanced_metrics,
            'new_features_in_top15': new_in_top15,
        },
        'improvement': {
            'roc_auc': roc_diff,
            'recall': recall_diff,
        },
        'sample_size': len(df),
        'failure_rate': float(y.mean()),
    }

    output_dir = Path('data/report/analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'pre_assessment_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_dir / 'pre_assessment_comparison.json'}")

    # ==========================================================================
    # VERDICT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if roc_diff > 0.01:
        print(f"\n  ✓ Pre-assessment features IMPROVED the model!")
        print(f"    ROC-AUC improved by {roc_diff:.4f} ({roc_diff/base_metrics['roc_auc']*100:.1f}%)")
    elif roc_diff < -0.01:
        print(f"\n  ✗ Pre-assessment features DEGRADED the model")
        print(f"    ROC-AUC decreased by {abs(roc_diff):.4f}")
    else:
        print(f"\n  ≈ Pre-assessment features had MINIMAL impact")
        print(f"    ROC-AUC change: {roc_diff:+.4f}")

    if len(new_in_top15) >= 3:
        print(f"\n  ✓ {len(new_in_top15)} pre-assessment features ranked in top 15!")
        print("    This confirms they capture important predictive signal.")

    return results


if __name__ == '__main__':
    main()
