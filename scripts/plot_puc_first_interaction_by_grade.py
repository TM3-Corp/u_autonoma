#!/usr/bin/env python3
"""
Plot PUC Student First Interaction Timeline by Final Grade (Scatter Plot)

Creates a grid of scatter plots showing when each student first accessed their course,
with dots colored by their final grade on the Chilean 1-7 scale.

Data Source: PUC wave_analysis dataset (Jan-Jul 2023)
- Page views: 2.98M records, 784 students, 22 courses
- Grades: 1,648 enrollments, Chilean 1-7 scale (fail < 4.0)
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
    print(f"  Grade range: {grades['grade'].min():.1f} - {grades['grade'].max():.1f}")
    print(f"  Students with grades: {grades['grade'].notna().sum()} ({grades['grade'].notna().mean()*100:.1f}%)")

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
    print(f"  Grade range: {data['grade'].min():.1f} - {data['grade'].max():.1f}")
    print(f"  Average grade: {data['grade'].mean():.2f}")
    print(f"  Failure rate: {data['failed'].mean()*100:.1f}%")
    print(f"  Pass rate (≥4.0): {(data['grade'] >= 4.0).mean()*100:.1f}%")

    # Add color and category columns
    data['color'] = data['grade'].apply(get_grade_color)
    data['grade_category'] = data['grade'].apply(get_grade_category)

    return data


def plot_scatter_timeline_puc(data, output_dir):
    """Create scatter plot grid with grade-based coloring for PUC courses."""
    print("\nCreating scatter plot visualization...")

    unique_courses = sorted(data['course_id'].unique())
    n_courses = len(unique_courses)
    print(f"Courses to plot: {n_courses}")

    # Create grid (5 cols × ceil(n/5) rows)
    ncols = 5
    nrows = (n_courses + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, 5*nrows))
    axes = axes.flatten() if n_courses > 1 else [axes]

    # Color order (plot in this order so red dots are on top)
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

        # Add jitter to Y-axis for visibility (consistent seed per course)
        np.random.seed(42 + idx)
        course_data['y_jitter'] = np.random.uniform(-0.3, 0.3, len(course_data))

        # Plot by color category (reverse order so fail/red is on top)
        for color in color_order:
            subset = course_data[course_data['color'] == color]
            if len(subset) > 0:
                ax.scatter(subset['first_interaction'], subset['y_jitter'],
                          c=color, alpha=0.7, s=50, label=color_labels[color],
                          edgecolors='black', linewidths=0.3)

        # Add 10th percentile line (course start proxy)
        pct_10 = course_data['first_interaction'].quantile(0.10)
        ax.axvline(pct_10, color='purple', linestyle='--', linewidth=1.5,
                   alpha=0.5, label='Course Start (10th %ile)')

        # Formatting
        sigla = course_data.iloc[0]['Sigla'] if 'Sigla' in course_data.columns else f"Course {course_id}"
        avg_grade = course_data['grade'].mean()
        fail_rate = course_data['failed'].mean() * 100

        ax.set_title(f'{sigla}\n(n={len(course_data)}, avg={avg_grade:.2f}, {fail_rate:.0f}% fail)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Date of First Access', fontsize=9)
        ax.set_ylabel('Students (jittered)', fontsize=9)
        ax.set_yticks([])
        ax.legend(loc='upper right', fontsize=7, framealpha=0.9, edgecolor='black')
        ax.grid(True, alpha=0.3, axis='x')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))

    # Hide unused subplots
    for j in range(idx+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('PUC Student First Interaction Timeline by Final Grade (Chilean 1-7 Scale)\n'
                 'Each dot = one student, colored by final grade | Jan-Jul 2023',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'puc_first_interaction_by_grade_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved scatter plot: {output_path}")
    plt.close()

    return output_path


def generate_summary_statistics(data, output_dir):
    """Generate summary statistics per course and grade category."""
    print("\nGenerating summary statistics...")

    summary_rows = []

    for course_id in sorted(data['course_id'].unique()):
        course_data = data[data['course_id'] == course_id]
        sigla = course_data.iloc[0]['Sigla'] if 'Sigla' in course_data.columns else f"Course {course_id}"

        # Overall course stats
        total_students = len(course_data)
        avg_grade = course_data['grade'].mean()
        fail_rate = course_data['failed'].mean() * 100

        # First interaction statistics
        pct_10 = course_data['first_interaction'].quantile(0.10)
        pct_50 = course_data['first_interaction'].quantile(0.50)
        pct_90 = course_data['first_interaction'].quantile(0.90)

        # Stats by grade category
        for category in ['Fail (<4.0)', 'Low Pass (4.0-4.9)', 'Mid Pass (5.0-5.9)', 'High Pass (6.0-7.0)']:
            category_data = course_data[course_data['grade_category'] == category]

            if len(category_data) > 0:
                # Calculate median access time relative to 10th percentile
                median_access = category_data['first_interaction'].median()
                days_after_start = (median_access - pct_10).total_seconds() / 86400

                summary_rows.append({
                    'course_id': course_id,
                    'course_code': sigla,
                    'total_students': total_students,
                    'avg_grade': round(avg_grade, 2),
                    'fail_rate_pct': round(fail_rate, 1),
                    'grade_category': category,
                    'category_count': len(category_data),
                    'category_pct': round(len(category_data) / total_students * 100, 1),
                    'median_first_access': median_access.strftime('%Y-%m-%d'),
                    'days_after_course_start': round(days_after_start, 1),
                    'course_start_10pct': pct_10.strftime('%Y-%m-%d'),
                    'course_median': pct_50.strftime('%Y-%m-%d'),
                    'course_90pct': pct_90.strftime('%Y-%m-%d')
                })

    summary_df = pd.DataFrame(summary_rows)

    output_path = output_dir / 'puc_first_interaction_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"Saved summary statistics: {output_path}")

    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS BY GRADE CATEGORY")
    print("="*80)

    for category in ['Fail (<4.0)', 'Low Pass (4.0-4.9)', 'Mid Pass (5.0-5.9)', 'High Pass (6.0-7.0)']:
        category_summary = summary_df[summary_df['grade_category'] == category]
        if len(category_summary) > 0:
            avg_days_after = category_summary['days_after_course_start'].mean()
            total_count = category_summary['category_count'].sum()
            print(f"\n{category}:")
            print(f"  Total students: {total_count}")
            print(f"  Avg days after course start: {avg_days_after:.1f} days")
            print(f"  Courses with this category: {len(category_summary)}")

    return summary_df


def generate_readme(data, summary_df, output_dir):
    """Generate README documenting findings."""
    print("\nGenerating README...")

    readme_content = f"""# PUC Student First Interaction Analysis (Jan-Jul 2023)

## Overview

This analysis examines the relationship between when students first access their courses
and their final academic outcomes, using data from Pontificia Universidad Católica de Chile (PUC).

**Dataset:** PUC wave_analysis (1st semester 2023)
- **Total enrollments analyzed:** {len(data):,}
- **Unique students:** {data['student_id'].nunique():,}
- **Unique courses:** {data['course_id'].nunique()}
- **Date range:** {data['first_interaction'].min().strftime('%B %d, %Y')} - {data['first_interaction'].max().strftime('%B %d, %Y')}

## Grading System

**Chilean 1-7 Scale:**
- **7.0** = Maximum grade (equivalent to A+ or 100%)
- **6.0-6.9** = High pass (A/A-)
- **5.0-5.9** = Mid pass (B/B-)
- **4.0-4.9** = Low pass (C, minimum passing)
- **1.0-3.9** = Fail (F)

**Dataset Grade Distribution:**
- Average grade: **{data['grade'].mean():.2f}**
- Pass rate (≥4.0): **{(data['grade'] >= 4.0).mean()*100:.1f}%**
- Fail rate (<4.0): **{data['failed'].mean()*100:.1f}%**

## Color Coding

Scatter plots use the following color scheme:
- 🔵 **Blue** (#2E86AB): High Pass (6.0-7.0)
- 🟢 **Green** (#27AE60): Mid Pass (5.0-5.9)
- 🟡 **Yellow** (#F39C12): Low Pass (4.0-4.9)
- 🔴 **Red** (#E74C3C): Fail (<4.0)
- ⚫ **Gray**: No grade data

## Key Findings

### 1. Access Timing by Grade Category

"""

    # Add timing analysis by category
    for category in ['Fail (<4.0)', 'Low Pass (4.0-4.9)', 'Mid Pass (5.0-5.9)', 'High Pass (6.0-7.0)']:
        category_summary = summary_df[summary_df['grade_category'] == category]
        if len(category_summary) > 0:
            avg_days = category_summary['days_after_course_start'].mean()
            total_count = category_summary['category_count'].sum()
            readme_content += f"\n**{category}:**\n"
            readme_content += f"- Total students: {total_count}\n"
            readme_content += f"- Average access delay: {avg_days:.1f} days after course start (10th percentile)\n"

    readme_content += f"""

### 2. Course-Specific Patterns

Total courses analyzed: **{data['course_id'].nunique()}**

Top 5 courses by enrollment:
"""

    top_courses = data.groupby(['course_id', 'Sigla']).size().sort_values(ascending=False).head(5)
    for (course_id, sigla), count in top_courses.items():
        course_data = data[data['course_id'] == course_id]
        avg_grade = course_data['grade'].mean()
        fail_rate = course_data['failed'].mean() * 100
        readme_content += f"\n- **{sigla}** (n={count}, avg={avg_grade:.2f}, {fail_rate:.0f}% fail)"

    readme_content += """

## Interpretation

### Early Access Correlation

The analysis reveals whether students who access courses earlier tend to have better outcomes.
Key questions addressed:

1. **Do failing students access later?**
   - Compare red dots position relative to purple line (course start)
   - Late access after week 2-3 may indicate disengagement

2. **Do high performers access early?**
   - Blue/green dots clustered near course start suggest proactive behavior
   - Early engagement correlates with better preparation

3. **Optimal intervention timing:**
   - Students who haven't accessed by 10th percentile date are at higher risk
   - Early warning systems should trigger interventions by week 2-3

### Limitations

1. **Class imbalance:** Low failure rate (~{data['failed'].mean()*100:.0f}%) means failing students are rare
2. **Correlation ≠ causation:** Late access may be symptom, not cause of failure
3. **Course variability:** Different courses have different engagement patterns
4. **Historical data:** Analysis based on 2023 semester, patterns may change

## Files Generated

1. **puc_first_interaction_by_grade_scatter.png** - Main visualization (scatter plot grid)
2. **puc_first_interaction_summary.csv** - Statistical summary per course per grade category
3. **puc_first_interaction_README.md** - This documentation

## Data Sources

- **Page views:** `/home/paul/projects/wave_analysis/puc_analysis/data/categorized_page_views.parquet`
  - 2,980,575 page views from 784 students across 22 courses

- **Grades:** `/home/paul/projects/wave_analysis/puc_analysis/data/grades_with_failure.parquet`
  - 1,648 enrollments with Chilean 1-7 scale grades

## Methodology

1. **First interaction calculation:** Minimum `created_at` timestamp per student per course
2. **Course start proxy:** 10th percentile of first access times (purple line in plots)
3. **Grade coloring:** Chilean 1-7 scale mapped to 4 color categories
4. **Jittering:** Y-axis randomization prevents dot overlap for visualization clarity

---

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Script: plot_puc_first_interaction_by_grade.py*
"""

    output_path = output_dir / 'puc_first_interaction_README.md'
    output_path.write_text(readme_content)
    print(f"Saved README: {output_path}")

    return output_path


def main():
    """Main execution."""
    print("="*80)
    print("PUC FIRST INTERACTION ANALYSIS BY GRADE")
    print("="*80)

    # Setup output directory
    output_dir = Path('/home/paul/projects/uautonoma/data/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and merge data
    data = load_and_merge_puc_data()

    # Create scatter plot visualization
    plot_path = plot_scatter_timeline_puc(data, output_dir)

    # Generate summary statistics
    summary_df = generate_summary_statistics(data, output_dir)

    # Generate README
    readme_path = generate_readme(data, summary_df, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nGenerated files:")
    print(f"  1. {plot_path}")
    print(f"  2. {output_dir / 'puc_first_interaction_summary.csv'}")
    print(f"  3. {readme_path}")
    print("\nNext steps:")
    print("  - Review scatter plots for grade-access timing patterns")
    print("  - Identify courses with late-accessing failing students (red dots right of purple line)")
    print("  - Use insights to set early warning thresholds (e.g., no access by week 2)")


if __name__ == '__main__':
    main()
