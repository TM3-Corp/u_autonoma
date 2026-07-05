#!/usr/bin/env python3
"""
Analyze resource type impact on academic performance.

Compares engagement metrics between Aprobados (passed) and Reprobados (failed)
for each resource type: modules, quizzes, discussions, files, assignments,
pages, grades, announcements.

Statistical tests:
- Mann-Whitney U test (non-parametric)
- Effect size (Cohen's d)
- Correlation with final_score
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

# Input/Output paths
CATEGORY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/category_features.parquet')
SESSION_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/session_features.parquet')
ENGAGEMENT_RATIOS = Path('/home/paul/projects/uautonoma/data/enriched_features/engagement_ratios.parquet')
ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/analysis/resource_impact_analysis.json')
OUTPUT_DIR = Path('/home/paul/projects/uautonoma/data/analysis')

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Passing threshold (57% or above)
PASS_THRESHOLD = 57.0

# Resource types to analyze
RESOURCE_TYPES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules', 'grades', 'announcements', 'home']

# Course names for reporting
COURSE_NAMES = {
    79875: 'TALLER DE COMP DIGITALES-P01',
    79913: 'FUND. DE BUSINESS ANALYTICS-P01',
    84936: 'FUNDAMENTOS DE MICROECONOMÍA-P03',
    84941: 'FUNDAMENTOS DE MICROECONOMÍA-P01',
    84944: 'FUNDAMENTOS DE MACROECONOMÍA-P03',
    86020: 'TALL DE COMPETENCIAS DIGITALES-P02',
    86676: 'FUND DE BUSINESS ANALYTICS-P01',
    88381: 'MATEMÁTICAS PARA LOS NEGOCIOS-P01',
    89099: 'TALLER DE COMP DIGITALES-P01',
    89390: 'GESTIÓN DEL TALENTO-P01',
}


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0

    var1 = group1.var()
    var2 = group2.var()

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def interpret_effect_size(d):
    """Interpret Cohen's d value."""
    d = abs(d)
    if d < 0.2:
        return 'negligible'
    elif d < 0.5:
        return 'small'
    elif d < 0.8:
        return 'medium'
    else:
        return 'large'


def interpret_p_value(p):
    """Interpret p-value."""
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'ns'


def analyze_resource_type(df, resource_type, metric_suffix='_views'):
    """
    Analyze a specific resource type comparing Aprobados vs Reprobados.

    Returns dict with statistical results.
    """
    col_name = f'{resource_type}{metric_suffix}'

    if col_name not in df.columns:
        return None

    # Get values for each group
    aprobados = df[df['passed'] == True][col_name].dropna()
    reprobados = df[df['passed'] == False][col_name].dropna()

    if len(aprobados) < 5 or len(reprobados) < 5:
        return None

    # Mann-Whitney U test
    try:
        stat, p_value = stats.mannwhitneyu(aprobados, reprobados, alternative='two-sided')
    except Exception:
        stat, p_value = 0, 1.0

    # Effect size
    d = cohens_d(aprobados, reprobados)

    # Correlation with final_score
    valid_data = df[[col_name, 'final_score']].dropna()
    if len(valid_data) > 5:
        corr, corr_p = stats.pearsonr(valid_data[col_name], valid_data['final_score'])
    else:
        corr, corr_p = 0, 1.0

    return {
        'resource_type': resource_type,
        'metric': col_name,
        'n_aprobados': len(aprobados),
        'n_reprobados': len(reprobados),
        'mean_aprobados': float(aprobados.mean()),
        'mean_reprobados': float(reprobados.mean()),
        'median_aprobados': float(aprobados.median()),
        'median_reprobados': float(reprobados.median()),
        'std_aprobados': float(aprobados.std()),
        'std_reprobados': float(reprobados.std()),
        'mann_whitney_u': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'significance_level': interpret_p_value(p_value),
        'cohens_d': float(d),
        'effect_size_interpretation': interpret_effect_size(d),
        'correlation_with_grade': float(corr),
        'correlation_p_value': float(corr_p),
        'direction': 'Aprobados > Reprobados' if aprobados.mean() > reprobados.mean() else 'Reprobados > Aprobados'
    }


def analyze_per_course(df, resource_type, metric_suffix='_views'):
    """Analyze resource type impact per course."""
    col_name = f'{resource_type}{metric_suffix}'

    if col_name not in df.columns:
        return {}

    results = {}
    for course_id in df['course_id'].unique():
        course_df = df[df['course_id'] == course_id]

        aprobados = course_df[course_df['passed'] == True][col_name].dropna()
        reprobados = course_df[course_df['passed'] == False][col_name].dropna()

        if len(aprobados) < 3 or len(reprobados) < 3:
            continue

        # Mann-Whitney U test
        try:
            stat, p_value = stats.mannwhitneyu(aprobados, reprobados, alternative='two-sided')
        except Exception:
            continue

        d = cohens_d(aprobados, reprobados)

        results[int(course_id)] = {
            'course_name': COURSE_NAMES.get(course_id, str(course_id)),
            'n_aprobados': len(aprobados),
            'n_reprobados': len(reprobados),
            'mean_aprobados': float(aprobados.mean()),
            'mean_reprobados': float(reprobados.mean()),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'cohens_d': float(d),
            'effect_size': interpret_effect_size(d)
        }

    return results


def calculate_risk_factors(df):
    """
    Calculate risk factors by resource type engagement.
    Low engagement = below median, High engagement = above median.
    """
    results = []

    for resource in RESOURCE_TYPES:
        col_name = f'{resource}_views'
        if col_name not in df.columns:
            continue

        # Skip if all zeros
        if df[col_name].sum() == 0:
            continue

        median_val = df[col_name].median()

        # Split by engagement level
        low_engagement = df[df[col_name] <= median_val]
        high_engagement = df[df[col_name] > median_val]

        if len(low_engagement) < 10 or len(high_engagement) < 10:
            continue

        # Calculate failure rates
        low_failure_rate = (low_engagement['passed'] == False).mean()
        high_failure_rate = (high_engagement['passed'] == False).mean()

        # Risk ratio
        if high_failure_rate > 0:
            risk_ratio = low_failure_rate / high_failure_rate
        else:
            risk_ratio = float('inf') if low_failure_rate > 0 else 1.0

        # Chi-square test
        try:
            contingency = pd.crosstab(df[col_name] <= median_val, df['passed'])
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        except Exception:
            chi2, p_value = 0, 1.0

        results.append({
            'resource_type': resource,
            'metric': col_name,
            'median_threshold': float(median_val),
            'n_low_engagement': len(low_engagement),
            'n_high_engagement': len(high_engagement),
            'failure_rate_low_engagement': float(low_failure_rate),
            'failure_rate_high_engagement': float(high_failure_rate),
            'risk_ratio': float(risk_ratio) if risk_ratio != float('inf') else 999.0,
            'chi2': float(chi2),
            'p_value': float(p_value),
            'significant': p_value < 0.05
        })

    return results


def main():
    print('=' * 60)
    print('Resource Impact Analysis: Aprobados vs Reprobados')
    print('=' * 60)
    print()

    # Load data
    print('Loading data...')
    category_df = pd.read_parquet(CATEGORY_FEATURES)
    print(f'  Category features: {len(category_df)} rows')

    enrollments_df = pd.read_csv(ENROLLMENTS)
    print(f'  Enrollments: {len(enrollments_df)} rows')

    # Merge with grades
    df = category_df.merge(
        enrollments_df[['user_id', 'course_id', 'final_score']],
        on=['user_id', 'course_id'],
        how='left'
    )

    # Create pass/fail column
    df['passed'] = df['final_score'] >= PASS_THRESHOLD
    df = df.dropna(subset=['final_score'])

    print(f'  Merged dataset: {len(df)} rows')
    print(f'  Aprobados: {df["passed"].sum()} ({df["passed"].mean()*100:.1f}%)')
    print(f'  Reprobados: {(~df["passed"]).sum()} ({(~df["passed"]).mean()*100:.1f}%)')
    print()

    # Store all results
    results = {
        'dataset_info': {
            'total_students': len(df),
            'n_aprobados': int(df['passed'].sum()),
            'n_reprobados': int((~df['passed']).sum()),
            'pass_rate': float(df['passed'].mean()),
            'pass_threshold': PASS_THRESHOLD,
            'n_courses': int(df['course_id'].nunique())
        },
        'resource_analysis': {},
        'risk_factors': [],
        'per_course_analysis': {}
    }

    # Analyze each resource type
    print('Analyzing resource types...')
    print()
    print('=' * 80)
    print(f'{"Resource":<15} {"Mean Apr":>10} {"Mean Rep":>10} {"p-value":>10} {"Cohen d":>10} {"Sig":>5}')
    print('=' * 80)

    for resource in RESOURCE_TYPES:
        # Analyze views
        result = analyze_resource_type(df, resource, '_views')
        if result:
            results['resource_analysis'][f'{resource}_views'] = result
            print(f'{resource:<15} {result["mean_aprobados"]:>10.1f} {result["mean_reprobados"]:>10.1f} '
                  f'{result["p_value"]:>10.4f} {result["cohens_d"]:>10.2f} {result["significance_level"]:>5}')

        # Analyze unique resources
        result_unique = analyze_resource_type(df, resource, '_unique_resources')
        if result_unique:
            results['resource_analysis'][f'{resource}_unique_resources'] = result_unique

        # Analyze time
        result_time = analyze_resource_type(df, resource, '_time_min')
        if result_time:
            results['resource_analysis'][f'{resource}_time_min'] = result_time

        # Per-course analysis
        course_results = analyze_per_course(df, resource, '_views')
        if course_results:
            results['per_course_analysis'][resource] = course_results

    print('=' * 80)
    print()

    # Calculate risk factors
    print('Calculating risk factors...')
    risk_factors = calculate_risk_factors(df)
    results['risk_factors'] = risk_factors

    # Print risk factors
    print()
    print('=' * 80)
    print('RISK FACTORS (Low vs High Engagement)')
    print('=' * 80)
    print(f'{"Resource":<15} {"Low Fail%":>10} {"High Fail%":>10} {"Risk Ratio":>12} {"Sig":>5}')
    print('-' * 80)

    for rf in sorted(risk_factors, key=lambda x: x['risk_ratio'], reverse=True):
        if rf['significant']:
            print(f'{rf["resource_type"]:<15} {rf["failure_rate_low_engagement"]*100:>9.1f}% '
                  f'{rf["failure_rate_high_engagement"]*100:>9.1f}% '
                  f'{rf["risk_ratio"]:>11.2f}x {"*" if rf["significant"] else ""}')

    print()

    # Summary of significant findings
    print('=' * 60)
    print('SIGNIFICANT FINDINGS (p < 0.05)')
    print('=' * 60)

    significant = [r for r in results['resource_analysis'].values() if r.get('significant', False)]
    significant = sorted(significant, key=lambda x: abs(x.get('cohens_d', 0)), reverse=True)

    for r in significant[:10]:
        direction = '+' if r['mean_aprobados'] > r['mean_reprobados'] else '-'
        print(f"  {r['metric']}: {direction} (d={r['cohens_d']:.2f}, p={r['p_value']:.4f})")

    print()

    # Correlation summary
    print('=' * 60)
    print('CORRELATION WITH FINAL GRADE')
    print('=' * 60)

    correlations = [(k, v.get('correlation_with_grade', 0), v.get('correlation_p_value', 1))
                   for k, v in results['resource_analysis'].items()]
    correlations = sorted(correlations, key=lambda x: abs(x[1]), reverse=True)

    for metric, corr, p in correlations[:10]:
        sig = '*' if p < 0.05 else ''
        print(f'  {metric}: r={corr:.3f} {sig}')

    print()

    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, (np.floating, float)):
            if np.isnan(obj):
                return None
            return float(obj)
        elif isinstance(obj, (np.integer, int)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results = convert_to_serializable(results)

    # Save results
    print(f'Saving results to {OUTPUT_FILE}...')
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print('Done!')


if __name__ == '__main__':
    main()
