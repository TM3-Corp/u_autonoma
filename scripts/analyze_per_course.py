#!/usr/bin/env python3
"""
Per-course analysis of engagement dynamics features.

Calculates:
- Per-course correlations with final_score
- Per-course R² prediction performance
- Variance explained analysis
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import requests
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}


def get_courses_from_account(account_id):
    """Get all courses from an account."""
    r = requests.get(
        f'{API_URL}/api/v1/accounts/{account_id}/courses',
        headers=headers,
        params={'per_page': 100}
    )
    return r.json() if r.status_code == 200 else []


def check_course_quality(course_id):
    """Check if course has adequate data for analysis."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 100, 'include[]': 'grades'}
    )
    if r.status_code != 200:
        return None

    enrollments = r.json()
    grades = [e['grades'].get('final_score') for e in enrollments
              if e.get('grades') and e['grades'].get('final_score') is not None
              and 0 < e['grades'].get('final_score') <= 100]

    if len(grades) < 15:
        return None

    return {
        'n_students': len(grades),
        'grade_mean': round(np.mean(grades), 1),
        'grade_std': round(np.std(grades), 1),
        'min_grade': round(min(grades), 1),
        'max_grade': round(max(grades), 1),
        'pass_rate': round(sum(1 for g in grades if g >= 57) / len(grades), 2)
    }


def find_quality_courses():
    """Find all Control de Gestión courses with adequate data."""
    print("Scanning Control de Gestión courses for adequate data...")
    print("=" * 90)

    cdg_accounts = [719, 718]
    good_courses = []

    for account_id in cdg_accounts:
        courses = get_courses_from_account(account_id)
        for course in courses:
            course_id = course['id']
            name = course.get('name', 'Unknown')

            stats = check_course_quality(course_id)
            if stats and stats['grade_std'] > 5:
                stats['course_id'] = course_id
                stats['name'] = name[:50]
                good_courses.append(stats)
                print(f"✓ {course_id}: {name[:40]:<40} | n={stats['n_students']:>2} | "
                      f"μ={stats['grade_mean']:>5.1f} | σ={stats['grade_std']:>5.1f} | "
                      f"pass={stats['pass_rate']:.0%}")

    print("\n" + "=" * 90)
    print(f"Found {len(good_courses)} courses with adequate data (n≥15, σ>5)")
    return good_courses


def analyze_existing_data():
    """Analyze per-course performance from existing feature data."""
    df = pd.read_csv('data/engagement_dynamics/student_features.csv')

    # Get feature columns
    exclude = ['course_id', 'user_id', 'user_role', 'final_score', 'failed']
    feature_cols = [c for c in df.columns
                   if c not in exclude
                   and not c.endswith('_norm')
                   and df[c].notna().sum() > 10]

    print("\n" + "=" * 90)
    print("PER-COURSE ANALYSIS FROM EXISTING DATA")
    print("=" * 90)

    results = []

    for course_id in df['course_id'].unique():
        course_df = df[df['course_id'] == course_id].copy()

        # Filter to students with grades
        course_df = course_df[course_df['final_score'].notna()]
        n_students = len(course_df)

        if n_students < 10:
            continue

        y = course_df['final_score'].values
        grade_std = np.std(y)

        if grade_std < 5:
            continue

        # Prepare features
        X_df = course_df[feature_cols].copy()
        for col in X_df.columns:
            X_df[col] = X_df[col].fillna(X_df[col].median())
        X_df = X_df.replace([np.inf, -np.inf], 0).fillna(0)
        X = X_df.values

        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate correlations
        correlations = {}
        for i, col in enumerate(feature_cols):
            r, p = stats.spearmanr(X[:, i], y)
            if not np.isnan(r):
                correlations[col] = {'r': round(r, 3), 'p': round(p, 4)}

        # Sort by absolute correlation
        top_features = sorted(correlations.items(), key=lambda x: abs(x[1]['r']), reverse=True)[:10]

        # Random Forest R²
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        try:
            cv_scores = cross_val_score(rf, X_scaled, y, cv=min(5, n_students // 3), scoring='r2')
            r2_mean = np.mean(cv_scores)
            r2_std = np.std(cv_scores)
        except:
            r2_mean = np.nan
            r2_std = np.nan

        # Fit full model for feature importance
        rf.fit(X_scaled, y)
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        result = {
            'course_id': course_id,
            'n_students': n_students,
            'grade_mean': round(np.mean(y), 1),
            'grade_std': round(grade_std, 1),
            'r2_mean': round(r2_mean, 3) if not np.isnan(r2_mean) else None,
            'r2_std': round(r2_std, 3) if not np.isnan(r2_std) else None,
            'top_correlations': top_features[:5],
            'top_importances': list(importances.head(5).to_records(index=False))
        }
        results.append(result)

        print(f"\n{'='*90}")
        print(f"COURSE {course_id} | n={n_students} | μ={result['grade_mean']} | σ={result['grade_std']}")
        print(f"{'='*90}")
        print(f"\nRandom Forest R² = {result['r2_mean']:.3f} (±{result['r2_std']:.3f})" if result['r2_mean'] else "R² = N/A")
        print(f"\nVariance explained by engagement features: {result['r2_mean']*100:.1f}%" if result['r2_mean'] else "")

        print(f"\nTop Correlations with Final Score:")
        print(f"{'Feature':<35} {'Spearman r':>12} {'p-value':>10}")
        print("-" * 60)
        for feat, vals in top_features[:8]:
            sig = "***" if vals['p'] < 0.001 else "**" if vals['p'] < 0.01 else "*" if vals['p'] < 0.05 else ""
            print(f"{feat:<35} {vals['r']:>+12.3f} {vals['p']:>10.4f} {sig}")

        print(f"\nTop Feature Importances (Random Forest):")
        print(f"{'Feature':<35} {'Importance':>12}")
        print("-" * 50)
        for _, feat, imp in list(importances.head(8).to_records()):
            print(f"{feat:<35} {imp:>12.4f}")

    return results


def main():
    # First find quality courses
    quality_courses = find_quality_courses()

    # Analyze existing data
    results = analyze_existing_data()

    # Summary table
    print("\n" + "=" * 90)
    print("SUMMARY: VARIANCE EXPLAINED PER COURSE")
    print("=" * 90)

    print(f"\n{'Course ID':<12} {'n':>4} {'Grade μ':>8} {'Grade σ':>8} {'R² Mean':>10} {'R² Std':>8} {'Top Predictor':<30}")
    print("-" * 90)

    for r in sorted(results, key=lambda x: x['r2_mean'] if x['r2_mean'] else 0, reverse=True):
        top_pred = r['top_correlations'][0][0] if r['top_correlations'] else 'N/A'
        r2_str = f"{r['r2_mean']:.3f}" if r['r2_mean'] else "N/A"
        r2_std_str = f"±{r['r2_std']:.3f}" if r['r2_std'] else ""
        print(f"{r['course_id']:<12} {r['n_students']:>4} {r['grade_mean']:>8.1f} {r['grade_std']:>8.1f} "
              f"{r2_str:>10} {r2_std_str:>8} {top_pred:<30}")

    # Save results
    output_path = 'data/engagement_dynamics/per_course_analysis.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Overall statistics
    valid_r2 = [r['r2_mean'] for r in results if r['r2_mean'] is not None]
    if valid_r2:
        print(f"\n{'='*90}")
        print("OVERALL STATISTICS")
        print(f"{'='*90}")
        print(f"Courses analyzed: {len(results)}")
        print(f"Average R²: {np.mean(valid_r2):.3f}")
        print(f"Best R²: {max(valid_r2):.3f}")
        print(f"Worst R²: {min(valid_r2):.3f}")
        print(f"Median R²: {np.median(valid_r2):.3f}")


if __name__ == '__main__':
    main()
