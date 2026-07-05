#!/usr/bin/env python3
"""
Analyze course-level factors that correlate with student failure rates.

This analysis focuses on instructional design insights:
- Does delayed student engagement correlate with failure?
- What course characteristics predict higher failure rates?
- How can course design be improved based on these findings?

Output:
    data/report/analysis/course_level_factors.json
    data/report/analysis/course_failure_correlation.png
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
BASE_DIR = Path(__file__).parent.parent
STATS_FILE = BASE_DIR / "data/report/analysis/first_interaction_stats.json"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
COURSES_FILE = BASE_DIR / "data/courses_raw.json"
OUTPUT_DIR = BASE_DIR / "data/report/analysis"

# Model courses
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Courses with incorrect term dates (Periodo Completo that started mid-semester)
EXCLUDE_TERM_ANALYSIS = [86020, 86676]  # Their "days from term" metrics are misleading


def load_first_interaction_stats():
    """Load first interaction statistics."""
    with open(STATS_FILE) as f:
        stats = json.load(f)

    # Convert to dataframe
    rows = []
    for course_id, data in stats.items():
        row = {'course_id': int(course_id)}
        row.update(data)
        rows.append(row)

    return pd.DataFrame(rows)


def load_enrollments():
    """Load enrollment data with grades."""
    df = pd.read_csv(ENROLLMENTS_FILE)
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


def load_page_views():
    """Load page view data."""
    df = pd.read_parquet(PAGE_VIEWS_FILE)
    return df


def load_courses():
    """Load course metadata."""
    with open(COURSES_FILE) as f:
        courses = json.load(f)
    return {c['id']: c for c in courses}


def calculate_course_metrics(df_stats, df_enroll, df_pv, courses_meta):
    """Calculate comprehensive course-level metrics."""

    metrics = []

    for course_id in MODEL_COURSES:
        course_stats = df_stats[df_stats['course_id'] == course_id].iloc[0] if course_id in df_stats['course_id'].values else None
        course_enroll = df_enroll[df_enroll['course_id'] == course_id]
        course_pv = df_pv[df_pv['course_id'] == course_id]
        course_meta = courses_meta.get(course_id, {})

        if course_stats is None or len(course_enroll) == 0:
            continue

        n_students = len(course_enroll)
        n_failed = course_enroll['failed'].sum()
        failure_rate = n_failed / n_students if n_students > 0 else 0

        # Page view metrics per course
        if len(course_pv) > 0:
            total_pv = len(course_pv)
            pv_per_student = total_pv / n_students

            # Activity diversity (unique controllers/actions)
            unique_controllers = course_pv['controller'].nunique() if 'controller' in course_pv.columns else 0
            unique_actions = course_pv['action'].nunique() if 'action' in course_pv.columns else 0

            # Temporal spread of activity
            if 'created_at' in course_pv.columns:
                course_pv_dt = pd.to_datetime(course_pv['created_at'])
                activity_span_days = (course_pv_dt.max() - course_pv_dt.min()).days
            else:
                activity_span_days = 0
        else:
            total_pv = 0
            pv_per_student = 0
            unique_controllers = 0
            unique_actions = 0
            activity_span_days = 0

        # Grade statistics
        avg_grade = course_enroll['final_score'].mean()
        std_grade = course_enroll['final_score'].std()
        min_grade = course_enroll['final_score'].min()
        max_grade = course_enroll['final_score'].max()

        row = {
            'course_id': course_id,
            'name': course_stats['name'],
            'term_name': course_stats['term_name'],
            'n_students': n_students,
            'n_passed': n_students - n_failed,
            'n_failed': n_failed,
            'failure_rate': failure_rate,
            'failure_pct': failure_rate * 100,

            # First interaction timing
            'days_to_5th_pct': course_stats.get('days_from_term_to_5th_pct'),
            'days_to_median': course_stats.get('days_from_term_to_median'),
            'days_to_earliest': course_stats.get('days_from_term_to_earliest'),

            # Activity metrics
            'total_page_views': total_pv,
            'pv_per_student': pv_per_student,
            'activity_controllers': unique_controllers,
            'activity_actions': unique_actions,
            'activity_span_days': activity_span_days,

            # Grade distribution
            'avg_grade': avg_grade,
            'std_grade': std_grade,
            'min_grade': min_grade,
            'max_grade': max_grade,
            'grade_range': max_grade - min_grade,

            # Term type
            'is_periodo_completo': 'Periodo Completo' in str(course_stats.get('term_name', '')),
            'has_valid_term_dates': course_id not in EXCLUDE_TERM_ANALYSIS,
        }

        metrics.append(row)

    return pd.DataFrame(metrics)


def analyze_correlations(df):
    """Analyze correlations between course metrics and failure rate."""

    # Only use courses with valid term dates for timing correlations
    df_valid = df[df['has_valid_term_dates']].copy()

    correlations = {}

    # Timing metrics (only for courses with valid term dates)
    timing_metrics = ['days_to_5th_pct', 'days_to_median', 'days_to_earliest']
    for metric in timing_metrics:
        if metric in df_valid.columns and df_valid[metric].notna().sum() >= 3:
            valid_data = df_valid[[metric, 'failure_rate']].dropna()
            if len(valid_data) >= 3:
                r, p = stats.pearsonr(valid_data[metric], valid_data['failure_rate'])
                correlations[metric] = {'r': r, 'p': p, 'n': len(valid_data)}

    # Activity metrics (all courses)
    activity_metrics = ['pv_per_student', 'activity_controllers', 'activity_actions', 'activity_span_days']
    for metric in activity_metrics:
        if metric in df.columns and df[metric].notna().sum() >= 3:
            valid_data = df[[metric, 'failure_rate']].dropna()
            if len(valid_data) >= 3:
                r, p = stats.pearsonr(valid_data[metric], valid_data['failure_rate'])
                correlations[metric] = {'r': r, 'p': p, 'n': len(valid_data)}

    # Grade distribution metrics
    grade_metrics = ['std_grade', 'grade_range']
    for metric in grade_metrics:
        if metric in df.columns and df[metric].notna().sum() >= 3:
            valid_data = df[[metric, 'failure_rate']].dropna()
            if len(valid_data) >= 3:
                r, p = stats.pearsonr(valid_data[metric], valid_data['failure_rate'])
                correlations[metric] = {'r': r, 'p': p, 'n': len(valid_data)}

    return correlations


def create_visualization(df, correlations):
    """Create visualization of course-level factors vs failure rate."""

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Course-Level Factors vs Student Failure Rate\n(Instructional Design Analysis)',
                 fontsize=14, fontweight='bold')

    # Only courses with valid term dates for timing plots
    df_valid = df[df['has_valid_term_dates']].copy()

    # 1. Days to Median vs Failure Rate
    ax = axes[0, 0]
    if len(df_valid) > 0:
        ax.scatter(df_valid['days_to_median'], df_valid['failure_pct'],
                   s=df_valid['n_students']*3, alpha=0.7, c='steelblue', edgecolors='navy')

        # Add course labels
        for _, row in df_valid.iterrows():
            short_name = row['name'][:20] + '...' if len(row['name']) > 20 else row['name']
            ax.annotate(short_name, (row['days_to_median'], row['failure_pct']),
                       fontsize=7, alpha=0.8, ha='center', va='bottom')

        # Add correlation info
        if 'days_to_median' in correlations:
            corr = correlations['days_to_median']
            sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
            ax.text(0.05, 0.95, f"r = {corr['r']:.3f}{sig}\np = {corr['p']:.3f}\nn = {corr['n']}",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Days from Term Start to Median First Interaction')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Engagement Timing')
    ax.grid(True, alpha=0.3)

    # 2. Days to 5th Percentile vs Failure Rate
    ax = axes[0, 1]
    if len(df_valid) > 0:
        ax.scatter(df_valid['days_to_5th_pct'], df_valid['failure_pct'],
                   s=df_valid['n_students']*3, alpha=0.7, c='coral', edgecolors='darkred')

        if 'days_to_5th_pct' in correlations:
            corr = correlations['days_to_5th_pct']
            sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
            ax.text(0.05, 0.95, f"r = {corr['r']:.3f}{sig}\np = {corr['p']:.3f}",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Days from Term Start to 5th Pct First Interaction')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Early Adopters Timing')
    ax.grid(True, alpha=0.3)

    # 3. Page Views per Student vs Failure Rate
    ax = axes[0, 2]
    ax.scatter(df['pv_per_student'], df['failure_pct'],
               s=df['n_students']*3, alpha=0.7, c='green', edgecolors='darkgreen')

    if 'pv_per_student' in correlations:
        corr = correlations['pv_per_student']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
        ax.text(0.05, 0.95, f"r = {corr['r']:.3f}{sig}\np = {corr['p']:.3f}",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Page Views per Student')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Student Activity Level')
    ax.grid(True, alpha=0.3)

    # 4. Activity Diversity (Controllers) vs Failure Rate
    ax = axes[1, 0]
    ax.scatter(df['activity_controllers'], df['failure_pct'],
               s=df['n_students']*3, alpha=0.7, c='purple', edgecolors='indigo')

    if 'activity_controllers' in correlations:
        corr = correlations['activity_controllers']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
        ax.text(0.05, 0.95, f"r = {corr['r']:.3f}{sig}\np = {corr['p']:.3f}",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Activity Types (Unique Controllers)')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Content Diversity')
    ax.grid(True, alpha=0.3)

    # 5. Grade Standard Deviation vs Failure Rate
    ax = axes[1, 1]
    ax.scatter(df['std_grade'], df['failure_pct'],
               s=df['n_students']*3, alpha=0.7, c='orange', edgecolors='darkorange')

    if 'std_grade' in correlations:
        corr = correlations['std_grade']
        sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
        ax.text(0.05, 0.95, f"r = {corr['r']:.3f}{sig}\np = {corr['p']:.3f}",
               transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Grade Standard Deviation')
    ax.set_ylabel('Failure Rate (%)')
    ax.set_title('Grade Dispersion')
    ax.grid(True, alpha=0.3)

    # 6. Summary Bar Chart - Courses by Failure Rate
    ax = axes[1, 2]
    df_sorted = df.sort_values('failure_pct', ascending=True)
    colors = ['green' if x < 30 else 'orange' if x < 50 else 'red' for x in df_sorted['failure_pct']]
    short_names = [n[:15] + '...' if len(n) > 15 else n for n in df_sorted['name']]

    bars = ax.barh(short_names, df_sorted['failure_pct'], color=colors, alpha=0.7, edgecolor='black')
    ax.axvline(x=40, color='red', linestyle='--', alpha=0.5, label='Overall avg (40%)')
    ax.set_xlabel('Failure Rate (%)')
    ax.set_title('Courses Ranked by Failure Rate')
    ax.legend()

    # Add value labels
    for bar, val in zip(bars, df_sorted['failure_pct']):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.0f}%',
               va='center', fontsize=8)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'course_failure_correlation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved visualization: {output_path}")
    return output_path


def create_detailed_scatter(df):
    """Create detailed scatter plot with regression line."""

    df_valid = df[df['has_valid_term_dates']].copy()

    if len(df_valid) < 3:
        print("Not enough courses with valid term dates for regression")
        return

    fig, ax = plt.subplots(figsize=(10, 8))

    # Scatter with size by student count
    scatter = ax.scatter(df_valid['days_to_median'], df_valid['failure_pct'],
                        s=df_valid['n_students']*5,
                        c=df_valid['avg_grade'], cmap='RdYlGn',
                        alpha=0.7, edgecolors='black', linewidth=1)

    # Add regression line if enough points
    if len(df_valid) >= 3:
        x = df_valid['days_to_median'].values
        y = df_valid['failure_pct'].values

        # Remove NaN
        mask = ~(np.isnan(x) | np.isnan(y))
        x, y = x[mask], y[mask]

        if len(x) >= 3:
            slope, intercept, r, p, se = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, 'r--', linewidth=2, alpha=0.7,
                   label=f'Trend: r={r:.3f}, p={p:.3f}')

    # Add course labels with smart positioning
    for _, row in df_valid.iterrows():
        short_name = row['name'].replace('FUNDAMENTOS DE ', '').replace('TALLER DE ', '')[:18]
        ax.annotate(short_name,
                   (row['days_to_median'], row['failure_pct']),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Average Grade (%)', fontsize=10)

    ax.set_xlabel('Days from Term Start to Median First Interaction', fontsize=12)
    ax.set_ylabel('Course Failure Rate (%)', fontsize=12)
    ax.set_title('Course Engagement Timing vs Student Failure Rate\n(Bubble size = # students, Color = avg grade)',
                fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add interpretation box
    if len(df_valid) >= 3:
        r, p = stats.pearsonr(df_valid['days_to_median'].dropna(),
                             df_valid['failure_pct'].dropna())

        if r > 0:
            interpretation = "Positive correlation: Later engagement → Higher failure"
        else:
            interpretation = "Negative correlation: Later engagement → Lower failure (!)"

        sig_text = "Statistically significant" if p < 0.05 else "Not statistically significant"

        textstr = f"Correlation: r = {r:.3f}\np-value = {p:.3f} ({sig_text})\n\n{interpretation}"
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=props)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'course_timing_vs_failure_detailed.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved detailed scatter: {output_path}")
    return output_path


def main():
    print("=" * 70)
    print("Course-Level Factors Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df_stats = load_first_interaction_stats()
    df_enroll = load_enrollments()
    df_pv = load_page_views()
    courses_meta = load_courses()

    print(f"  First interaction stats: {len(df_stats)} courses")
    print(f"  Enrollments: {len(df_enroll)} records")
    print(f"  Page views: {len(df_pv)} records")
    print(f"  Course metadata: {len(courses_meta)} courses")

    # Calculate metrics
    print("\nCalculating course-level metrics...")
    df_metrics = calculate_course_metrics(df_stats, df_enroll, df_pv, courses_meta)
    print(f"  Analyzed {len(df_metrics)} courses")

    # Analyze correlations
    print("\nAnalyzing correlations with failure rate...")
    correlations = analyze_correlations(df_metrics)

    # Print correlation results
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS RESULTS")
    print("=" * 70)

    print("\n1. TIMING METRICS (courses with valid term dates only):")
    print("-" * 50)

    for metric in ['days_to_5th_pct', 'days_to_median', 'days_to_earliest']:
        if metric in correlations:
            corr = correlations[metric]
            sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
            direction = "Later engagement → MORE failures" if corr['r'] > 0 else "Later engagement → FEWER failures"
            print(f"\n  {metric}:")
            print(f"    Correlation: r = {corr['r']:+.3f}{sig}")
            print(f"    p-value: {corr['p']:.4f}")
            print(f"    Interpretation: {direction}")

    print("\n2. ACTIVITY METRICS (all courses):")
    print("-" * 50)

    for metric in ['pv_per_student', 'activity_controllers', 'activity_actions', 'activity_span_days']:
        if metric in correlations:
            corr = correlations[metric]
            sig = '***' if corr['p'] < 0.001 else '**' if corr['p'] < 0.01 else '*' if corr['p'] < 0.05 else ''
            print(f"\n  {metric}:")
            print(f"    Correlation: r = {corr['r']:+.3f}{sig}")
            print(f"    p-value: {corr['p']:.4f}")

    # Print course summary table
    print("\n" + "=" * 70)
    print("COURSE SUMMARY TABLE")
    print("=" * 70)

    df_display = df_metrics[['name', 'n_students', 'failure_pct', 'days_to_median',
                             'pv_per_student', 'avg_grade']].copy()
    df_display = df_display.sort_values('failure_pct', ascending=False)
    df_display['name'] = df_display['name'].str[:30]
    df_display['failure_pct'] = df_display['failure_pct'].round(1)
    df_display['pv_per_student'] = df_display['pv_per_student'].round(0)
    df_display['avg_grade'] = df_display['avg_grade'].round(1)

    print(f"\n{'Course':<32} {'N':>4} {'Fail%':>6} {'DaysToMed':>10} {'PV/Stud':>8} {'AvgGrade':>9}")
    print("-" * 75)

    for _, row in df_display.iterrows():
        days = f"{row['days_to_median']:.0f}" if pd.notna(row['days_to_median']) else "N/A"
        print(f"{row['name']:<32} {row['n_students']:>4} {row['failure_pct']:>5.1f}% {days:>10} "
              f"{row['pv_per_student']:>8.0f} {row['avg_grade']:>8.1f}%")

    # Create visualizations
    print("\n" + "=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)

    create_visualization(df_metrics, correlations)
    create_detailed_scatter(df_metrics)

    # Save results
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'n_courses_analyzed': len(df_metrics),
        'n_courses_with_valid_term': len(df_metrics[df_metrics['has_valid_term_dates']]),
        'correlations': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv
                            for kk, vv in v.items()}
                        for k, v in correlations.items()},
        'course_metrics': df_metrics.to_dict('records'),
        'key_findings': []
    }

    # Add key findings
    df_valid = df_metrics[df_metrics['has_valid_term_dates']]

    if 'days_to_median' in correlations:
        corr = correlations['days_to_median']
        if corr['r'] < 0:
            results['key_findings'].append({
                'finding': 'Counterintuitive: Later median engagement correlates with LOWER failure',
                'r': corr['r'],
                'p': corr['p'],
                'implication': 'Course design quality may matter more than timing. Well-designed courses '
                              'may not require immediate engagement to be effective.'
            })
        elif corr['r'] > 0.3 and corr['p'] < 0.1:
            results['key_findings'].append({
                'finding': 'Later engagement correlates with higher failure',
                'r': corr['r'],
                'p': corr['p'],
                'implication': 'Courses should encourage early engagement through onboarding activities.'
            })

    # Find outliers
    high_failure_courses = df_metrics[df_metrics['failure_pct'] > 60]
    low_failure_courses = df_metrics[df_metrics['failure_pct'] < 25]

    if len(high_failure_courses) > 0:
        results['high_risk_courses'] = high_failure_courses[['name', 'failure_pct', 'days_to_median']].to_dict('records')

    if len(low_failure_courses) > 0:
        results['successful_courses'] = low_failure_courses[['name', 'failure_pct', 'days_to_median']].to_dict('records')

    output_file = OUTPUT_DIR / 'course_level_factors.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")

    # Print key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FOR INSTRUCTIONAL DESIGN")
    print("=" * 70)

    for finding in results['key_findings']:
        print(f"\n• {finding['finding']}")
        print(f"  Correlation: r = {finding['r']:.3f}, p = {finding['p']:.4f}")
        print(f"  Implication: {finding['implication']}")

    if 'high_risk_courses' in results:
        print("\n• HIGH-RISK COURSES (>60% failure):")
        for course in results['high_risk_courses']:
            print(f"  - {course['name']}: {course['failure_pct']:.1f}% failure")

    if 'successful_courses' in results:
        print("\n• SUCCESSFUL COURSES (<25% failure):")
        for course in results['successful_courses']:
            print(f"  - {course['name']}: {course['failure_pct']:.1f}% failure")


if __name__ == "__main__":
    main()
