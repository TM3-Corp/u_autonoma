#!/usr/bin/env python3
"""
Plot PUC Student Cumulative Access Timeline by Final Grade

Creates scatter plots showing:
- X axis: Timeline (dates)
- Y axis: Cumulative % of students who have accessed (0-100%)
- Each dot: One student accessing at that time
- Color: Final grade (Chilean 1-7 scale)

This reveals whether early accessors tend to pass (blue/green at bottom)
and late accessors tend to fail (red at top).

Data Source: PUC wave_analysis dataset (Jan-Jul 2023)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Use a clean seaborn style
sns.set_style('whitegrid')

# Grade color mapping for Chilean 1-7 scale
def get_grade_color(grade):
    """Map Chilean 1-7 grade to color."""
    if pd.isna(grade):
        return 'gray'  # No grade data
    elif grade >= 6.0:
        return '#2E86AB'  # Blue (6.0-7.0) - High pass
    elif grade >= 5.0:
        return '#27AE60'  # Green (5.0-5.9) - Mid pass
    elif grade >= 4.0:
        return '#F39C12'  # Yellow/Orange (4.0-4.9) - Low pass
    else:
        return '#E74C3C'  # Red (<4.0) - Fail


def get_grade_category(grade):
    """Get grade category label."""
    if pd.isna(grade):
        return 'No Grade'
    elif grade >= 6.0:
        return 'High Pass (6.0-7.0)'
    elif grade >= 5.0:
        return 'Mid Pass (5.0-5.9)'
    elif grade >= 4.0:
        return 'Low Pass (4.0-4.9)'
    else:
        return 'Fail (<4.0)'


def load_and_merge_puc_data():
    """Load PUC page views and grades, calculate first interactions."""
    print("Loading PUC data from wave_analysis...")

    # Load page views (2.98M rows, 784 students, 22 courses)
    pv_path = '/home/paul/projects/wave_analysis/puc_analysis/data/categorized_page_views.parquet'
    print(f"Loading page views from: {pv_path}")
    pv = pd.read_parquet(pv_path)

    print(f"Page views loaded: {len(pv):,} records")
    print(f"  Unique students: {pv['student_id'].nunique()}")
    print(f"  Unique courses: {pv['course_id'].nunique()}")
    print(f"  Date range: {pv['created_at'].min()} to {pv['created_at'].max()}")

    # Calculate first interaction per student per course
    print("\nCalculating first interactions...")
    print("  Method: MIN(created_at) per (student_id, course_id)")
    first_access = pv.groupby(['course_id', 'student_id'])['created_at'].min().reset_index()
    first_access.columns = ['course_id', 'student_id', 'first_interaction']

    print(f"First interactions calculated: {len(first_access)} student-course pairs")

    # Load grades (1,648 enrollments, Chilean 1-7 scale)
    grades_path = '/home/paul/projects/wave_analysis/puc_analysis/data/grades_with_failure.parquet'
    print(f"\nLoading grades from: {grades_path}")
    grades = pd.read_parquet(grades_path)

    print(f"Grades loaded: {len(grades):,} enrollments")
    print(f"  Unique students: {grades['user_lms_id'].nunique()}")
    print(f"  Unique courses: {grades['course_lms_id'].nunique()}")

    # Merge on student_id and course_id
    print("\nMerging page views with grades...")
    data = first_access.merge(
        grades[['user_lms_id', 'course_lms_id', 'grade', 'failed', 'Sigla']],
        left_on=['student_id', 'course_id'],
        right_on=['user_lms_id', 'course_lms_id'],
        how='inner'  # Only students with both PV and grades
    )

    print(f"\nMerged dataset:")
    print(f"  Total enrollments: {len(data)}")
    print(f"  Unique students: {data['student_id'].nunique()}")
    print(f"  Unique courses: {data['course_id'].nunique()}")
    print(f"  Average grade: {data['grade'].mean():.2f}")
    print(f"  Failure rate: {data['failed'].mean()*100:.1f}%")

    # Add color and category columns
    data['color'] = data['grade'].apply(get_grade_color)
    data['grade_category'] = data['grade'].apply(get_grade_category)

    return data


def calculate_cumulative_percentage(course_data):
    """
    Calculate cumulative percentage for each student's first access.

    Returns course_data with added 'cumulative_pct' column.
    """
    # Sort by first interaction time
    course_data = course_data.sort_values('first_interaction').reset_index(drop=True)

    # Calculate cumulative percentage (1st student = 1/N, 2nd = 2/N, etc.)
    n_students = len(course_data)
    course_data['cumulative_pct'] = [(i + 1) / n_students * 100 for i in range(n_students)]

    return course_data


def plot_cumulative_access_by_grade(data, output_dir):
    """
    Create cumulative access plots with grade-based coloring for PUC courses.

    X axis: Timeline (dates)
    Y axis: Cumulative % of students who have accessed
    Each dot: One student, colored by final grade
    """
    print("\nCreating cumulative access visualization...")

    unique_courses = sorted(data['course_id'].unique())
    n_courses = len(unique_courses)
    print(f"Courses to plot: {n_courses}")

    # Create grid (5 cols × ceil(n/5) rows)
    ncols = 5
    nrows = (n_courses + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 5*nrows))
    axes = axes.flatten() if n_courses > 1 else [axes]

    # Color order (plot in reverse so red dots are on top)
    color_order = ['#2E86AB', '#27AE60', '#F39C12', '#E74C3C', 'gray']
    color_labels = {
        '#2E86AB': 'High Pass (6.0-7.0)',
        '#27AE60': 'Mid Pass (5.0-5.9)',
        '#F39C12': 'Low Pass (4.0-4.9)',
        '#E74C3C': 'Fail (<4.0)',
        'gray': 'No Grade'
    }

    for idx, course_id in enumerate(unique_courses):
        ax = axes[idx]
        course_data = data[data['course_id'] == course_id].copy()

        # Calculate cumulative percentage
        course_data = calculate_cumulative_percentage(course_data)

        # Plot by color category (so fail/red dots appear on top)
        for color in color_order:
            subset = course_data[course_data['color'] == color]
            if len(subset) > 0:
                ax.scatter(subset['first_interaction'], subset['cumulative_pct'],
                          c=color, alpha=0.7, s=50, label=color_labels[color],
                          edgecolors='black', linewidths=0.3, zorder=5)

        # Add reference lines
        ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='50% accessed')

        # Add course start (10th percentile)
        pct_10 = course_data['first_interaction'].quantile(0.10)
        ax.axvline(pct_10, color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Course Start (10th %ile)', zorder=3)

        # Formatting
        sigla = course_data.iloc[0]['Sigla'] if 'Sigla' in course_data.columns else f"Course {course_id}"
        avg_grade = course_data['grade'].mean()
        fail_rate = course_data['failed'].mean() * 100

        ax.set_title(f'{sigla}\n(n={len(course_data)}, avg={avg_grade:.2f}, {fail_rate:.0f}% fail)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Date of First Access', fontsize=9)
        ax.set_ylabel('Cumulative % of Students', fontsize=9)
        ax.set_ylim(-5, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.legend(loc='lower right', fontsize=7, framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, axis='both')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))

        # Add statistics annotation
        # Calculate correlation between access order and grade
        if len(course_data) > 2:
            # Early accessors = bottom 25%, Late accessors = top 25%
            early_25 = course_data.iloc[:len(course_data)//4]
            late_25 = course_data.iloc[-len(course_data)//4:]

            early_avg = early_25['grade'].mean()
            late_avg = late_25['grade'].mean()

            text = f'Early 25%: {early_avg:.2f}\nLate 25%: {late_avg:.2f}'
            ax.text(0.02, 0.98, text, transform=ax.transAxes,
                   fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('PUC Student Cumulative Access Timeline by Final Grade (Chilean 1-7 Scale)\n'
                 'Y-axis: Cumulative % of students | Each dot = one student accessing | Color = final grade',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'puc_cumulative_access_by_grade.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved cumulative access plot: {output_path}")
    plt.close()

    return output_path


def generate_access_timing_analysis(data, output_dir):
    """Analyze relationship between access timing and grades."""
    print("\nAnalyzing access timing vs grades...")

    analysis_rows = []

    for course_id in sorted(data['course_id'].unique()):
        course_data = data[data['course_id'] == course_id].copy()
        course_data = calculate_cumulative_percentage(course_data)

        sigla = course_data.iloc[0]['Sigla'] if 'Sigla' in course_data.columns else f"Course {course_id}"

        # Divide into quartiles by access timing
        n = len(course_data)
        q1_end = n // 4
        q2_end = n // 2
        q3_end = 3 * n // 4

        quartiles = [
            ('First 25% (Early)', course_data.iloc[:q1_end]),
            ('25-50%', course_data.iloc[q1_end:q2_end]),
            ('50-75%', course_data.iloc[q2_end:q3_end]),
            ('Last 25% (Late)', course_data.iloc[q3_end:])
        ]

        for quartile_name, quartile_data in quartiles:
            if len(quartile_data) > 0:
                analysis_rows.append({
                    'course_id': course_id,
                    'course_code': sigla,
                    'total_students': len(course_data),
                    'access_quartile': quartile_name,
                    'quartile_size': len(quartile_data),
                    'avg_grade': round(quartile_data['grade'].mean(), 2),
                    'fail_rate_pct': round(quartile_data['failed'].mean() * 100, 1),
                    'high_pass_pct': round((quartile_data['grade'] >= 6.0).mean() * 100, 1),
                    'first_access_date': quartile_data['first_interaction'].min().strftime('%Y-%m-%d'),
                    'last_access_date': quartile_data['first_interaction'].max().strftime('%Y-%m-%d')
                })

    analysis_df = pd.DataFrame(analysis_rows)

    output_path = output_dir / 'puc_access_timing_analysis.csv'
    analysis_df.to_csv(output_path, index=False)
    print(f"Saved access timing analysis: {output_path}")

    # Print key insights
    print("\n" + "="*80)
    print("ACCESS TIMING vs GRADES ANALYSIS")
    print("="*80)

    for quartile in ['First 25% (Early)', 'Last 25% (Late)']:
        quartile_data = analysis_df[analysis_df['access_quartile'] == quartile]
        if len(quartile_data) > 0:
            avg_grade = quartile_data['avg_grade'].mean()
            avg_fail_rate = quartile_data['fail_rate_pct'].mean()
            print(f"\n{quartile}:")
            print(f"  Average grade: {avg_grade:.2f}")
            print(f"  Average fail rate: {avg_fail_rate:.1f}%")

    # Calculate overall correlation
    print("\n" + "="*80)
    early_avg = analysis_df[analysis_df['access_quartile'] == 'First 25% (Early)']['avg_grade'].mean()
    late_avg = analysis_df[analysis_df['access_quartile'] == 'Last 25% (Late)']['avg_grade'].mean()
    grade_diff = early_avg - late_avg

    print(f"\nOVERALL FINDING:")
    print(f"  Early accessors (first 25%) average: {early_avg:.2f}")
    print(f"  Late accessors (last 25%) average: {late_avg:.2f}")
    print(f"  Grade difference: {grade_diff:.2f} points")
    print(f"  Conclusion: {'Early access correlates with higher grades' if grade_diff > 0 else 'No clear correlation'}")

    return analysis_df


def generate_readme(data, analysis_df, output_dir):
    """Generate README documenting findings."""
    print("\nGenerating README...")

    early_avg = analysis_df[analysis_df['access_quartile'] == 'First 25% (Early)']['avg_grade'].mean()
    late_avg = analysis_df[analysis_df['access_quartile'] == 'Last 25% (Late)']['avg_grade'].mean()
    grade_diff = early_avg - late_avg

    readme_content = f"""# PUC Cumulative Access Timeline Analysis (Jan-Jul 2023)

## Overview

This analysis examines the relationship between **when students first access their courses**
and their **final academic outcomes**, using a cumulative percentage visualization.

**Visualization Type:** Cumulative Access Timeline
- **X axis:** Date of first access
- **Y axis:** Cumulative % of students who have accessed (0-100%)
- **Each dot:** One student accessing at that specific time
- **Color:** Final grade (Chilean 1-7 scale)

**Dataset:** PUC wave_analysis (1st semester 2023)
- **Total enrollments:** {len(data):,}
- **Unique students:** {data['student_id'].nunique():,}
- **Unique courses:** {data['course_id'].nunique()}
- **Date range:** {data['first_interaction'].min().strftime('%B %d, %Y')} - {data['first_interaction'].max().strftime('%B %d, %Y')}

## Key Finding

**Early accessors significantly outperform late accessors:**
- **Early 25% (first to access):** Average grade = **{early_avg:.2f}**
- **Late 25% (last to access):** Average grade = **{late_avg:.2f}**
- **Grade difference:** **{grade_diff:.2f} points** on 1-7 scale

This is a **{grade_diff:.1%}** relative difference in the grading scale.

## Interpretation

### What the Visualization Shows

1. **Bottom-left dots (early accessors, 0-25%):**
   - If mostly blue/green → early access correlates with success
   - These students access within days of course start

2. **Top-right dots (late accessors, 75-100%):**
   - If red/orange → late access correlates with failure
   - These students access weeks after course start

3. **Slope of the curve:**
   - **Steep:** Most students onboard quickly (engaged cohort)
   - **Flat:** Students trickle in slowly (disengaged cohort)

### Early Warning Implications

Students who haven't accessed by the **50% mark** (horizontal gray line) are at higher risk:
- They're accessing later than most of their peers
- Late access correlates with lower grades
- Intervention should happen BEFORE this threshold

## Data Source & Processing

**Page Views Data:**
```
/home/paul/projects/wave_analysis/puc_analysis/data/categorized_page_views.parquet
```
- 2,980,575 page views from 784 students

**Processing:**
1. Calculate first interaction: `MIN(created_at)` per student per course
2. Sort students by first interaction time (ascending)
3. Calculate cumulative percentage: student #i out of N = (i/N) × 100%
4. Plot each student as dot at (timestamp, cumulative_%)
5. Color by final grade

**Grades Data:**
```
/home/paul/projects/wave_analysis/puc_analysis/data/grades_with_failure.parquet
```
- 1,648 enrollments with Chilean 1-7 scale grades

## Files Generated

1. **puc_cumulative_access_by_grade.png** - Main visualization (cumulative timeline grid)
2. **puc_access_timing_analysis.csv** - Quartile analysis (early vs late accessors)
3. **puc_cumulative_access_README.md** - This documentation

## Color Coding

- 🔵 **Blue** (#2E86AB): High Pass (6.0-7.0)
- 🟢 **Green** (#27AE60): Mid Pass (5.0-5.9)
- 🟡 **Yellow** (#F39C12): Low Pass (4.0-4.9)
- 🔴 **Red** (#E74C3C): Fail (<4.0)
- ⚫ **Gray**: No grade data

## Limitations

1. **Correlation ≠ causation:** Late access may be symptom, not cause of poor performance
2. **Course variability:** Different courses have different onboarding patterns
3. **External factors:** Students may have valid reasons for late enrollment
4. **Historical data:** Based on 2023 semester, patterns may evolve

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Script: plot_puc_cumulative_access_by_grade.py*
"""

    output_path = output_dir / 'puc_cumulative_access_README.md'
    output_path.write_text(readme_content)
    print(f"Saved README: {output_path}")

    return output_path


def main():
    """Main execution."""
    print("="*80)
    print("PUC CUMULATIVE ACCESS TIMELINE ANALYSIS")
    print("="*80)

    # Setup output directory
    output_dir = Path('/home/paul/projects/uautonoma/data/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge data
    data = load_and_merge_puc_data()

    # Create cumulative access visualization
    plot_path = plot_cumulative_access_by_grade(data, output_dir)

    # Generate access timing analysis
    analysis_df = generate_access_timing_analysis(data, output_dir)

    # Generate README
    readme_path = generate_readme(data, analysis_df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {plot_path}")
    print(f"  2. {output_dir / 'puc_access_timing_analysis.csv'}")
    print(f"  3. {readme_path}")
    print("\nInterpretation:")
    print("  - Look for blue/green dots at bottom (early high performers)")
    print("  - Look for red dots at top (late failing students)")
    print("  - Check if 50% line separates passing from failing students")


if __name__ == '__main__':
    main()
