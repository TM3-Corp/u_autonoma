#!/usr/bin/env python3
"""
Per-course analysis using ONLY pure activity features.

EXCLUDES leaky features that are directly tied to grades:
- total_participations (counts assignment/quiz submissions)
- tardiness features (on_time, late, missing = submission status)

Uses only behavioral patterns that are independent of graded work.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

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
    # Session regularity (how they use LMS)
    'session_count', 'session_gap_min', 'session_gap_max',
    'session_gap_mean', 'session_gap_std', 'session_regularity',
    'sessions_per_week',

    # Time preferences (WHEN they study)
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

    # Engagement trajectory (how activity changes over time)
    'engagement_velocity', 'engagement_acceleration',
    'weekly_cv', 'weekly_range', 'trend_reversals',
    'early_engagement_ratio', 'late_surge',

    # Workload dynamics (peaks and slopes)
    'peak_count_type1', 'peak_count_type2', 'peak_count_type3',
    'peak_ratio', 'max_positive_slope', 'max_negative_slope',
    'slope_std', 'positive_slope_sum', 'negative_slope_sum',

    # Time-to-access (procrastination indicators)
    'first_access_day', 'first_module_day', 'first_assignment_day',
    'access_time_pct',

    # Raw aggregates (passive engagement)
    'activity_span_days', 'unique_active_hours',
    'total_page_views',  # Viewing is NOT graded, submissions are
]


def analyze_pure_activity():
    """Analyze per-course using only pure activity features."""
    df = pd.read_csv('data/engagement_dynamics/student_features.csv')

    # Get only pure activity features that exist in the data
    available_features = [f for f in PURE_ACTIVITY_FEATURES if f in df.columns]

    print("=" * 90)
    print("PURE ACTIVITY ANALYSIS (No Leaky Features)")
    print("=" * 90)
    print(f"\nUsing {len(available_features)} pure activity features")
    print(f"EXCLUDED: {', '.join(LEAKY_FEATURES[:5])}...")
    print("=" * 90)

    results = []

    for course_id in sorted(df['course_id'].unique()):
        course_df = df[df['course_id'] == course_id].copy()
        course_df = course_df[course_df['final_score'].notna()]
        n_students = len(course_df)

        if n_students < 10:
            continue

        y = course_df['final_score'].values
        grade_std = np.std(y)
        grade_mean = np.mean(y)

        if grade_std < 5:
            continue

        # Calculate pass rate
        pass_rate = sum(1 for g in y if g >= 57) / len(y)

        # Check if we have meaningful class diversity
        if pass_rate == 0 or pass_rate == 1:
            diversity_warning = "LOW DIVERSITY"
        elif pass_rate < 0.2 or pass_rate > 0.8:
            diversity_warning = "MODERATE"
        else:
            diversity_warning = "GOOD"

        # Prepare features
        X_df = course_df[available_features].copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].fillna(X_df[col].median())
        X_df = X_df.replace([np.inf, -np.inf], 0).fillna(0)
        X = X_df.values

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate correlations
        correlations = {}
        for i, col in enumerate(available_features):
            r, p = stats.spearmanr(X[:, i], y)
            if not np.isnan(r):
                correlations[col] = {'r': round(r, 3), 'p': round(p, 4)}

        # Sort by absolute correlation
        top_features = sorted(correlations.items(),
                            key=lambda x: abs(x[1]['r']), reverse=True)[:10]

        # Random Forest R²
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        try:
            cv_folds = min(5, n_students // 3)
            if cv_folds >= 2:
                cv_scores = cross_val_score(rf, X_scaled, y, cv=cv_folds, scoring='r2')
                r2_mean = np.mean(cv_scores)
                r2_std = np.std(cv_scores)
            else:
                r2_mean = np.nan
                r2_std = np.nan
        except:
            r2_mean = np.nan
            r2_std = np.nan

        # Fit full model for feature importance
        rf.fit(X_scaled, y)
        importances = pd.DataFrame({
            'feature': available_features,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        result = {
            'course_id': str(course_id),
            'n_students': n_students,
            'grade_mean': round(grade_mean, 1),
            'grade_std': round(grade_std, 1),
            'pass_rate': round(pass_rate * 100, 1),
            'class_diversity': diversity_warning,
            'r2_mean': round(r2_mean, 3) if not np.isnan(r2_mean) else None,
            'r2_std': round(r2_std, 3) if not np.isnan(r2_std) else None,
            'top_correlations': top_features[:5],
            'top_importances': [(f, round(i, 4))
                               for f, i in importances.head(5).values]
        }
        results.append(result)

        # Print detailed results
        print(f"\n{'='*90}")
        print(f"COURSE {course_id} | n={n_students} | μ={result['grade_mean']} | "
              f"σ={result['grade_std']} | Pass={result['pass_rate']}% | "
              f"Diversity: {diversity_warning}")
        print("=" * 90)

        if result['r2_mean'] is not None:
            print(f"\nPure Activity R² = {result['r2_mean']:.3f} (±{result['r2_std']:.3f})")
            print(f"Variance explained by PURE activity: {result['r2_mean']*100:.1f}%")
        else:
            print("\nR² = N/A (insufficient folds)")

        print(f"\nTop Correlations (Pure Activity):")
        print(f"{'Feature':<30} {'Spearman r':>12} {'p-value':>10}")
        print("-" * 55)
        for feat, vals in top_features[:8]:
            sig = "***" if vals['p'] < 0.001 else "**" if vals['p'] < 0.01 else "*" if vals['p'] < 0.05 else ""
            print(f"{feat:<30} {vals['r']:>+12.3f} {vals['p']:>10.4f} {sig}")

        print(f"\nTop Feature Importances:")
        print(f"{'Feature':<30} {'Importance':>12}")
        print("-" * 45)
        for feat, imp in importances.head(8).values:
            print(f"{feat:<30} {imp:>12.4f}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY: PURE ACTIVITY VARIANCE EXPLAINED")
    print("(Excluding total_participations and tardiness features)")
    print("=" * 100)

    print(f"\n{'Course':<10} {'n':>4} {'μ':>6} {'σ':>6} {'Pass%':>6} {'Diversity':<12} "
          f"{'R² Mean':>8} {'R² Std':>8} {'Top Predictor':<25}")
    print("-" * 100)

    for r in sorted(results, key=lambda x: x['r2_mean'] if x['r2_mean'] else -999, reverse=True):
        top_pred = r['top_correlations'][0][0] if r['top_correlations'] else 'N/A'
        r2_str = f"{r['r2_mean']:.3f}" if r['r2_mean'] else "N/A"
        r2_std_str = f"±{r['r2_std']:.3f}" if r['r2_std'] else ""
        print(f"{r['course_id']:<10} {r['n_students']:>4} {r['grade_mean']:>6.1f} "
              f"{r['grade_std']:>6.1f} {r['pass_rate']:>5.1f}% {r['class_diversity']:<12} "
              f"{r2_str:>8} {r2_std_str:>8} {top_pred:<25}")

    # Save results
    output_path = 'data/engagement_dynamics/pure_activity_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Overall statistics
    valid_r2 = [r['r2_mean'] for r in results if r['r2_mean'] is not None]
    if valid_r2:
        print(f"\n{'='*90}")
        print("OVERALL STATISTICS (Pure Activity Only)")
        print("=" * 90)
        print(f"Courses analyzed: {len(results)}")
        print(f"Average R²: {np.mean(valid_r2):.3f}")
        print(f"Best R²: {max(valid_r2):.3f}")
        print(f"Worst R²: {min(valid_r2):.3f}")
        print(f"Median R²: {np.median(valid_r2):.3f}")

        # Compare with previous (leaky) results
        print("\n" + "=" * 90)
        print("COMPARISON: With vs Without Leaky Features")
        print("=" * 90)
        try:
            with open('data/engagement_dynamics/per_course_analysis.json') as f:
                leaky_results = json.load(f)
            leaky_r2 = [r['r2_mean'] for r in leaky_results if r['r2_mean']]
            print(f"WITH total_participations:    Avg R² = {np.mean(leaky_r2):.3f}")
            print(f"WITHOUT (pure activity only): Avg R² = {np.mean(valid_r2):.3f}")
            print(f"Difference: {np.mean(leaky_r2) - np.mean(valid_r2):.3f} "
                  f"({(np.mean(leaky_r2)/np.mean(valid_r2) - 1)*100:.0f}% inflation)")
        except:
            pass

    return results


if __name__ == '__main__':
    analyze_pure_activity()
