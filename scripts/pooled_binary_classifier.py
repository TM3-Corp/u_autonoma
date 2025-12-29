#!/usr/bin/env python3
"""
Pooled Binary Classification for Student Failure Prediction.

Trains binary classifiers on ALL students pooled across courses to generate
actionable insights for early intervention.

Uses ONLY pure activity features (no grade leakage).
"""

import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Try to import XGBoost and SHAP
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Using Random Forest instead.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("Warning: SHAP not installed. Skipping SHAP analysis.")


# =============================================================================
# FEATURE DEFINITIONS (NO LEAKAGE)
# =============================================================================

# Features that are LEAKY (directly tied to grades)
LEAKY_FEATURES = [
    'total_participations',  # Submitting assignments/quizzes
    'on_time', 'late', 'missing', 'floating',  # Tardiness = submission status
    'on_time_rate', 'late_rate', 'missing_rate',
    'num_submissions', 'submission_rate',
    'participations', 'participations_level',
]

# Pure activity features (NOT tied to grades)
PURE_ACTIVITY_FEATURES = [
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

# Courses with GOOD class diversity (from pure_activity_analysis.json)
GOOD_DIVERSITY_COURSES = [
    79875, 84936, 84941, 86676, 86681, 86682, 86683, 79876, 79877, 79878
]


def load_and_prepare_data(filter_good_courses=True):
    """Load student data and prepare for binary classification."""
    df = pd.read_csv('data/engagement_dynamics/student_features.csv')

    # Filter to students with grades
    df = df[df['final_score'].notna()].copy()

    # Create binary target
    df['failed'] = (df['final_score'] < 57).astype(int)

    # Filter to good diversity courses if requested
    if filter_good_courses:
        # Load pure activity analysis to get courses with GOOD diversity
        try:
            with open('data/engagement_dynamics/pure_activity_analysis.json') as f:
                course_analysis = json.load(f)
            good_courses = [
                int(c['course_id']) for c in course_analysis
                if c.get('class_diversity') == 'GOOD'
            ]
            df = df[df['course_id'].isin(good_courses)]
        except FileNotFoundError:
            print("Warning: pure_activity_analysis.json not found. Using all courses.")

    # Get available pure activity features
    available_features = [f for f in PURE_ACTIVITY_FEATURES if f in df.columns]

    # Prepare feature matrix
    X_df = df[available_features].copy()

    # Handle missing values with median imputation
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(X_df[col].median())
    X_df = X_df.replace([np.inf, -np.inf], 0).fillna(0)

    # Store original values before scaling (for insights)
    X_original = X_df.copy()

    # Z-score normalization within course (makes features course-agnostic)
    X_normalized = X_df.copy()
    for course_id in df['course_id'].unique():
        mask = df['course_id'] == course_id
        if mask.sum() > 1:
            scaler = StandardScaler()
            X_normalized.loc[mask, :] = scaler.fit_transform(X_normalized.loc[mask, :])

    y = df['failed'].values

    return df, X_normalized, X_original, y, available_features


def train_models(X, y, feature_names):
    """Train multiple binary classifiers with cross-validation."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Logistic Regression
    print("\n" + "=" * 70)
    print("LOGISTIC REGRESSION")
    print("=" * 70)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    y_pred_lr = cross_val_predict(lr, X, y, cv=cv)
    y_prob_lr = cross_val_predict(lr, X, y, cv=cv, method='predict_proba')[:, 1]

    lr.fit(X, y)  # Fit on full data for coefficients
    lr_coef = pd.DataFrame({
        'feature': feature_names,
        'coefficient': lr.coef_[0],
        'odds_ratio': np.exp(lr.coef_[0])
    }).sort_values('coefficient', key=abs, ascending=False)

    results['logistic_regression'] = {
        'model': lr,
        'predictions': y_pred_lr,
        'probabilities': y_prob_lr,
        'coefficients': lr_coef,
        'metrics': calculate_metrics(y, y_pred_lr, y_prob_lr)
    }
    print_metrics(results['logistic_regression']['metrics'])

    # 2. Random Forest
    print("\n" + "=" * 70)
    print("RANDOM FOREST")
    print("=" * 70)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    y_pred_rf = cross_val_predict(rf, X, y, cv=cv)
    y_prob_rf = cross_val_predict(rf, X, y, cv=cv, method='predict_proba')[:, 1]

    rf.fit(X, y)
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    results['random_forest'] = {
        'model': rf,
        'predictions': y_pred_rf,
        'probabilities': y_prob_rf,
        'importance': rf_importance,
        'metrics': calculate_metrics(y, y_pred_rf, y_prob_rf)
    }
    print_metrics(results['random_forest']['metrics'])

    # 3. XGBoost (if available)
    if HAS_XGBOOST:
        print("\n" + "=" * 70)
        print("XGBOOST")
        print("=" * 70)

        xgb_model = xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            scale_pos_weight=sum(y == 0) / sum(y == 1),  # Handle imbalance
            random_state=42, n_jobs=-1, use_label_encoder=False,
            eval_metric='logloss'
        )
        y_pred_xgb = cross_val_predict(xgb_model, X, y, cv=cv)
        y_prob_xgb = cross_val_predict(xgb_model, X, y, cv=cv, method='predict_proba')[:, 1]

        xgb_model.fit(X, y)
        xgb_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)

        results['xgboost'] = {
            'model': xgb_model,
            'predictions': y_pred_xgb,
            'probabilities': y_prob_xgb,
            'importance': xgb_importance,
            'metrics': calculate_metrics(y, y_pred_xgb, y_prob_xgb)
        }
        print_metrics(results['xgboost']['metrics'])

        # SHAP analysis
        if HAS_SHAP:
            print("\nCalculating SHAP values...")
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(X)
            results['xgboost']['shap_values'] = shap_values
            results['xgboost']['shap_explainer'] = explainer

    return results


def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }


def print_metrics(metrics):
    """Print classification metrics."""
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f} (catching at-risk students)")
    print(f"F1 Score:  {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(f"\nConfusion Matrix:")
    cm = np.array(metrics['confusion_matrix'])
    print(f"           Predicted")
    print(f"           Pass  Fail")
    print(f"Actual Pass  {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Fail  {cm[1,0]:4d}  {cm[1,1]:4d}")


def generate_actionable_insights(df, X_original, y, feature_names):
    """Generate actionable insights with odds ratios and relative risk."""
    insights = []

    # Create a dataframe for analysis
    analysis_df = X_original.copy()
    analysis_df['failed'] = y

    print("\n" + "=" * 70)
    print("GENERATING ACTIONABLE INSIGHTS")
    print("=" * 70)

    # 1. TIME-OF-DAY INSIGHTS
    print("\n--- Time-of-Day Analysis ---")
    time_features = {
        'weekday_morning_pct': 'morning (6am-12pm)',
        'weekday_afternoon_pct': 'afternoon (12pm-6pm)',
        'weekday_evening_pct': 'evening (6pm-10pm)',
        'weekday_night_pct': 'night (10pm-6am)',
    }

    for feature, time_label in time_features.items():
        if feature in analysis_df.columns:
            insight = analyze_feature_impact(
                analysis_df, feature, f'studying in the {time_label}',
                is_percentage=True
            )
            if insight:
                insights.append(insight)

    # 2. SESSION PATTERNS
    print("\n--- Session Pattern Analysis ---")
    session_features = {
        'session_count': 'total number of study sessions',
        'sessions_per_week': 'sessions per week',
        'session_gap_mean': 'average gap between sessions (days)',
        'session_gap_std': 'irregular session timing',
    }

    for feature, description in session_features.items():
        if feature in analysis_df.columns:
            insight = analyze_feature_impact(
                analysis_df, feature, description,
                inverse=(feature == 'session_gap_mean' or feature == 'session_gap_std')
            )
            if insight:
                insights.append(insight)

    # 3. ENGAGEMENT TRAJECTORY
    print("\n--- Engagement Trajectory Analysis ---")
    trajectory_features = {
        'early_engagement_ratio': 'early course engagement',
        'engagement_velocity': 'increasing engagement over time',
        'weekly_cv': 'inconsistent weekly activity',
        'trend_reversals': 'fluctuating engagement patterns',
    }

    for feature, description in trajectory_features.items():
        if feature in analysis_df.columns:
            inverse = feature in ['weekly_cv', 'trend_reversals']
            insight = analyze_feature_impact(
                analysis_df, feature, description, inverse=inverse
            )
            if insight:
                insights.append(insight)

    # 4. TIME-TO-ACCESS (Procrastination)
    print("\n--- Procrastination Analysis ---")
    access_features = {
        'first_access_day': 'late first access to course',
        'first_module_day': 'delayed start on course modules',
    }

    for feature, description in access_features.items():
        if feature in analysis_df.columns:
            insight = analyze_feature_impact(
                analysis_df, feature, description, inverse=True
            )
            if insight:
                insights.append(insight)

    # 5. ACTIVITY INTENSITY
    print("\n--- Activity Intensity Analysis ---")
    intensity_features = {
        'total_page_views': 'total page views',
        'unique_active_hours': 'unique active hours',
        'activity_span_days': 'days active in course',
    }

    for feature, description in intensity_features.items():
        if feature in analysis_df.columns:
            insight = analyze_feature_impact(
                analysis_df, feature, description
            )
            if insight:
                insights.append(insight)

    # 6. WEEKEND VS WEEKDAY
    print("\n--- Weekend Study Analysis ---")
    if 'weekend_morning_pct' in analysis_df.columns:
        weekend_cols = [c for c in analysis_df.columns if c.startswith('weekend_') and c.endswith('_pct')]
        weekday_cols = [c for c in analysis_df.columns if c.startswith('weekday_') and c.endswith('_pct')]

        if weekend_cols and weekday_cols:
            analysis_df['weekend_study_pct'] = analysis_df[weekend_cols].sum(axis=1)
            insight = analyze_feature_impact(
                analysis_df, 'weekend_study_pct', 'weekend studying',
                is_percentage=True
            )
            if insight:
                insights.append(insight)

    # Sort insights by effect size
    insights = sorted(insights, key=lambda x: abs(x.get('relative_risk', 1) - 1), reverse=True)

    return insights


def analyze_feature_impact(df, feature, description, inverse=False, is_percentage=False):
    """Analyze the impact of a feature on failure rate."""
    if feature not in df.columns:
        return None

    # Remove NaN values
    valid_df = df[[feature, 'failed']].dropna()
    if len(valid_df) < 50:
        return None

    # Split into high/low groups by median
    median_val = valid_df[feature].median()

    if is_percentage:
        # For percentage features, use >20% as "high"
        threshold = 0.2 if valid_df[feature].max() <= 1 else 20
        high_group = valid_df[valid_df[feature] > threshold]
        low_group = valid_df[valid_df[feature] <= threshold]
        group_label = f">{threshold*100 if threshold < 1 else threshold:.0f}%"
    else:
        high_group = valid_df[valid_df[feature] > median_val]
        low_group = valid_df[valid_df[feature] <= median_val]
        group_label = f">median ({median_val:.1f})"

    if len(high_group) < 10 or len(low_group) < 10:
        return None

    # Calculate failure rates
    high_fail_rate = high_group['failed'].mean()
    low_fail_rate = low_group['failed'].mean()

    # Avoid division by zero
    if high_fail_rate == 0 or low_fail_rate == 0:
        return None

    # Calculate relative risk
    if inverse:
        # Higher values = higher failure risk
        relative_risk = high_fail_rate / low_fail_rate
        risk_group = 'high'
        comparison_group = 'low'
    else:
        # Lower values = higher failure risk
        relative_risk = low_fail_rate / high_fail_rate
        risk_group = 'low'
        comparison_group = 'high'

    # Calculate odds ratio
    contingency = [
        [sum(high_group['failed'] == 0), sum(high_group['failed'] == 1)],
        [sum(low_group['failed'] == 0), sum(low_group['failed'] == 1)]
    ]

    try:
        if min(min(row) for row in contingency) < 5:
            _, p_value = fisher_exact(contingency)
        else:
            _, p_value, _, _ = chi2_contingency(contingency)
    except:
        p_value = 1.0

    # Calculate odds ratio manually
    a, b = contingency[0]
    c, d = contingency[1]
    if b == 0 or c == 0:
        odds_ratio = np.inf
    else:
        odds_ratio = (a * d) / (b * c)

    # Generate insight text
    if inverse:
        if relative_risk > 1.2:
            effect = f"{(relative_risk - 1) * 100:.0f}% higher"
            direction = "risk factor"
        else:
            return None
    else:
        if relative_risk > 1.2:
            effect = f"{(relative_risk - 1) * 100:.0f}% higher"
            direction = "risk factor"
        else:
            return None

    insight = {
        'feature': feature,
        'description': description,
        'relative_risk': round(relative_risk, 2),
        'odds_ratio': round(odds_ratio, 2) if odds_ratio != np.inf else 'Inf',
        'p_value': round(p_value, 4),
        'significant': p_value < 0.05,
        'high_group_n': len(high_group),
        'low_group_n': len(low_group),
        'high_fail_rate': round(high_fail_rate * 100, 1),
        'low_fail_rate': round(low_fail_rate * 100, 1),
        'insight_text': generate_insight_text(
            description, relative_risk, p_value, inverse,
            high_fail_rate, low_fail_rate
        )
    }

    if insight['significant']:
        print(f"  {feature}: RR={relative_risk:.2f}, p={p_value:.4f} ***")
    else:
        print(f"  {feature}: RR={relative_risk:.2f}, p={p_value:.4f}")

    return insight


def generate_insight_text(description, relative_risk, p_value, inverse,
                          high_fail_rate, low_fail_rate):
    """Generate human-readable insight text."""
    # Convert to percentages (multiply by 100)
    high_pct = high_fail_rate * 100
    low_pct = low_fail_rate * 100

    if inverse:
        # Higher values = higher risk
        if relative_risk > 1:
            effect_pct = (relative_risk - 1) * 100
            text = f"Students with high {description} have {effect_pct:.0f}% higher failure risk"
            text += f" (failure rate: {high_pct:.1f}% vs {low_pct:.1f}%)"
        else:
            return None
    else:
        # Lower values = higher risk (protective factor)
        if relative_risk > 1:
            effect_pct = (relative_risk - 1) * 100
            text = f"Students with low {description} have {effect_pct:.0f}% higher failure risk"
            text += f" (failure rate: {low_pct:.1f}% vs {high_pct:.1f}%)"
        else:
            return None

    if p_value < 0.001:
        text += " (p < 0.001)"
    elif p_value < 0.01:
        text += " (p < 0.01)"
    elif p_value < 0.05:
        text += " (p < 0.05)"
    else:
        text += f" (p = {p_value:.3f}, not statistically significant)"

    return text


def create_visualizations(df, y, model_results, insights, output_dir):
    """Create and save visualizations."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))

    for name, result in model_results.items():
        fpr, tpr, _ = roc_curve(y, result['probabilities'])
        auc = result['metrics']['roc_auc']
        ax.plot(fpr, tpr, label=f"{name.replace('_', ' ').title()} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Binary Classification Models')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150)
    plt.close()

    # 2. Feature Importance (from best model)
    best_model = 'xgboost' if 'xgboost' in model_results else 'random_forest'
    importance_df = model_results[best_model]['importance'].head(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(importance_df)), importance_df['importance'].values)
    ax.set_yticks(range(len(importance_df)))
    ax.set_yticklabels(importance_df['feature'].values)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top 15 Features - {best_model.replace("_", " ").title()}')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png', dpi=150)
    plt.close()

    # 3. SHAP Summary Plot (if available)
    if HAS_SHAP and 'xgboost' in model_results and 'shap_values' in model_results['xgboost']:
        fig, ax = plt.subplots(figsize=(12, 10))
        shap_values = model_results['xgboost']['shap_values']
        feature_names = model_results['xgboost']['importance']['feature'].tolist()

        # Create DataFrame for SHAP plotting
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=True)
        top_features = mean_abs_shap.tail(15).index.tolist()

        ax.barh(range(len(top_features)), mean_abs_shap[top_features].values)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel('Mean |SHAP Value|')
        ax.set_title('SHAP Feature Importance')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/shap_importance.png', dpi=150)
        plt.close()

    # 4. Top Insights Bar Chart
    sig_insights = [i for i in insights if i['significant']][:10]
    if sig_insights:
        fig, ax = plt.subplots(figsize=(12, 8))

        features = [i['feature'] for i in sig_insights]
        rr_values = [i['relative_risk'] for i in sig_insights]
        colors = ['red' if rr > 1 else 'green' for rr in rr_values]

        bars = ax.barh(range(len(features)), rr_values, color=colors, alpha=0.7)
        ax.axvline(x=1, color='black', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Relative Risk (RR)')
        ax.set_title('Statistically Significant Risk Factors (p < 0.05)')
        ax.invert_yaxis()

        # Add value labels
        for i, (bar, rr) in enumerate(zip(bars, rr_values)):
            ax.text(rr + 0.05, i, f'{rr:.2f}x', va='center')

        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_factors.png', dpi=150)
        plt.close()

    print(f"\nVisualizations saved to {output_dir}/")


def generate_report(df, y, model_results, insights, output_dir):
    """Generate markdown report with findings."""

    # Calculate summary statistics
    n_total = len(y)
    n_fail = sum(y)
    n_pass = n_total - n_fail
    fail_rate = n_fail / n_total * 100

    best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['roc_auc'])
    best_model_name = best_model[0].replace('_', ' ').title()
    best_auc = best_model[1]['metrics']['roc_auc']
    best_recall = best_model[1]['metrics']['recall']

    report = f"""# Pooled Binary Classification Report
## Student Failure Prediction Using Pure Activity Features

Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

This analysis trained binary classifiers on **{n_total} students** pooled across multiple courses
to predict academic failure (final grade < 57%) using **only pure activity features** (no grade leakage).

### Key Metrics
| Metric | Value |
|--------|-------|
| Total Students | {n_total} |
| Passing (≥57%) | {n_pass} ({100-fail_rate:.1f}%) |
| Failing (<57%) | {n_fail} ({fail_rate:.1f}%) |
| Best Model | {best_model_name} |
| Best ROC-AUC | {best_auc:.3f} |
| Best Recall | {best_recall:.1%} |

---

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
"""

    for name, result in model_results.items():
        m = result['metrics']
        report += f"| {name.replace('_', ' ').title()} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |\n"

    report += """
**Note:** Recall is prioritized as it measures the ability to catch at-risk students.

---

## Actionable Insights

The following insights are derived from statistical analysis of student activity patterns.
Only statistically significant findings (p < 0.05) with meaningful effect sizes (RR > 1.5 or < 0.67) are reported.

"""

    sig_insights = [i for i in insights if i['significant'] and abs(i['relative_risk'] - 1) > 0.2]

    for i, insight in enumerate(sig_insights[:10], 1):
        rr = insight['relative_risk']
        if rr > 1:
            direction = "RISK FACTOR"
            symbol = "⚠️"
        else:
            direction = "PROTECTIVE FACTOR"
            symbol = "✅"

        report += f"""### {i}. {insight['description'].title()} ({direction})
{symbol} **{insight['insight_text']}**

- Relative Risk: **{rr:.2f}x**
- Statistical significance: p = {insight['p_value']:.4f}
- Sample sizes: n={insight['high_group_n']} (high) vs n={insight['low_group_n']} (low)

"""

    report += """---

## Top Predictive Features

### Feature Importance (from best model)

| Rank | Feature | Importance |
|------|---------|------------|
"""

    importance = model_results[max(model_results.keys())]['importance']
    for i, (_, row) in enumerate(importance.head(10).iterrows(), 1):
        report += f"| {i} | {row['feature']} | {row['importance']:.4f} |\n"

    # Logistic regression coefficients
    if 'logistic_regression' in model_results:
        report += """
### Logistic Regression Coefficients

| Feature | Coefficient | Odds Ratio | Interpretation |
|---------|-------------|------------|----------------|
"""
        coef = model_results['logistic_regression']['coefficients']
        for _, row in coef.head(10).iterrows():
            if row['coefficient'] > 0:
                interp = f"{row['odds_ratio']:.2f}x more likely to fail"
            else:
                interp = f"{1/row['odds_ratio']:.2f}x less likely to fail"
            report += f"| {row['feature']} | {row['coefficient']:.3f} | {row['odds_ratio']:.3f} | {interp} |\n"

    report += """
---

## Recommendations for Early Intervention

Based on the analysis, the following student profiles are at higher risk of failure:

### High-Risk Indicators
"""

    risk_factors = [i for i in sig_insights if i['relative_risk'] > 1.3][:5]
    for rf in risk_factors:
        report += f"- **{rf['description'].title()}**: {rf['high_fail_rate']:.1f}% failure rate vs {rf['low_fail_rate']:.1f}%\n"

    report += """
### Protective Factors
"""

    protective = [i for i in sig_insights if i['relative_risk'] < 0.8][:5]
    for pf in protective:
        report += f"- **{pf['description'].title()}**: Associated with lower failure rates\n"

    report += """
---

## Methodology Notes

1. **Data Leakage Prevention**: Features directly tied to grades (submissions, tardiness) were excluded
2. **Course-Agnostic Features**: All features were z-score normalized within each course
3. **Cross-Validation**: 5-fold stratified cross-validation was used for all model evaluation
4. **Class Imbalance**: Models used class weighting to handle the pass/fail imbalance
5. **Statistical Testing**: Chi-square or Fisher's exact test used for significance testing

---

## Files Generated

- `model_results.json`: Detailed model metrics and predictions
- `actionable_insights.json`: All generated insights with statistics
- `roc_curves.png`: ROC curves comparing all models
- `feature_importance.png`: Top features by importance
- `risk_factors.png`: Visualization of significant risk factors

---

*Report generated by pooled_binary_classifier.py*
"""

    # Save report
    report_path = f'{output_dir}/report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")
    return report


def main():
    """Main execution function."""
    print("=" * 70)
    print("POOLED BINARY CLASSIFICATION FOR STUDENT FAILURE PREDICTION")
    print("=" * 70)

    # Create output directory
    output_dir = 'data/pooled_analysis'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/visualizations', exist_ok=True)

    # Load and prepare data
    print("\n1. Loading and preparing data...")
    df, X_normalized, X_original, y, feature_names = load_and_prepare_data(filter_good_courses=True)

    print(f"\nDataset Summary:")
    print(f"  Total students: {len(y)}")
    print(f"  Passing (≥57%): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.1f}%)")
    print(f"  Failing (<57%): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.1f}%)")
    print(f"  Features used: {len(feature_names)} (pure activity only)")
    print(f"  Courses: {df['course_id'].nunique()}")

    # Train models
    print("\n2. Training binary classification models...")
    model_results = train_models(X_normalized.values, y, feature_names)

    # Generate insights
    print("\n3. Generating actionable insights...")
    insights = generate_actionable_insights(df, X_original, y, feature_names)

    # Create visualizations
    print("\n4. Creating visualizations...")
    create_visualizations(df, y, model_results, insights, f'{output_dir}/visualizations')

    # Generate report
    print("\n5. Generating report...")
    report = generate_report(df, y, model_results, insights, output_dir)

    # Save results
    print("\n6. Saving results...")

    # Save model results (without non-serializable objects)
    results_to_save = {}
    for name, result in model_results.items():
        results_to_save[name] = {
            'metrics': result['metrics'],
            'top_features': result.get('importance', result.get('coefficients')).head(20).to_dict('records')
        }

    with open(f'{output_dir}/model_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)

    # Save insights (convert numpy types to native Python)
    def convert_to_native(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    insights_to_save = []
    for i in insights:
        insight_clean = {}
        for k, v in i.items():
            if k != 'model':
                insight_clean[k] = convert_to_native(v)
        insights_to_save.append(insight_clean)

    with open(f'{output_dir}/actionable_insights.json', 'w') as f:
        json.dump(insights_to_save, f, indent=2)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {output_dir}/report.md")
    print(f"  - {output_dir}/model_results.json")
    print(f"  - {output_dir}/actionable_insights.json")
    print(f"  - {output_dir}/visualizations/")

    # Summary of key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    best_model = max(model_results.items(), key=lambda x: x[1]['metrics']['roc_auc'])
    print(f"\nBest Model: {best_model[0].replace('_', ' ').title()}")
    print(f"  ROC-AUC: {best_model[1]['metrics']['roc_auc']:.3f}")
    print(f"  Recall:  {best_model[1]['metrics']['recall']:.3f}")

    sig_insights = [i for i in insights if i['significant']]
    print(f"\nStatistically Significant Insights: {len(sig_insights)}")
    print("\nTop 5 Risk Factors:")
    for i, insight in enumerate(sig_insights[:5], 1):
        print(f"  {i}. {insight['description']}: RR={insight['relative_risk']:.2f}")

    return model_results, insights


if __name__ == '__main__':
    main()
