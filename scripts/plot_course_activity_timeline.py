#!/usr/bin/env python3
"""
Plot course activity timelines showing daily student access patterns.

For each course, shows the percentage of enrolled students who accessed
the course on each date throughout the semester.

Outputs:
    - data/visualizations/course_activity_timelines.png (grid view)
    - data/visualizations/individual_timelines/*.png (individual plots)
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

def load_data() -> tuple[pd.DataFrame, dict]:
    """Load page views and enrollment counts."""
    print("Loading data...")

    # Load page views
    pv = pd.read_parquet('data/page_views/categorized_page_views.parquet')
    print(f"  Page views: {len(pv):,} records")
    print(f"  Unique users: {pv['user_id'].nunique():,}")
    print(f"  Unique courses: {pv['course_id'].nunique():,}")

    # Load course metadata
    import json
    with open('data/postgrado_courses_with_grades.json') as f:
        courses = json.load(f)

    # Create enrollment counts dictionary
    enrollment_counts = {}
    for course in courses:
        enrollment_counts[course['id']] = {
            'students': course['students'],
            'name': course['name'],
        }

    print(f"  Courses with metadata: {len(enrollment_counts)}")

    return pv, enrollment_counts


def calculate_daily_access_pct(pv: pd.DataFrame, enrollment_counts: dict) -> pd.DataFrame:
    """Calculate % of students accessing each course per date."""
    print("\nCalculating daily access percentages...")

    # Ensure date column is datetime (extract date from created_at)
    pv['date'] = pd.to_datetime(pv['created_at']).dt.date

    # Group by course and date, count unique students
    daily = pv.groupby(['course_id', 'date'])['user_id'].nunique().reset_index()
    daily.rename(columns={'user_id': 'n_students'}, inplace=True)

    # Add enrollment counts and calculate percentage
    daily['total_enrolled'] = daily['course_id'].map(
        lambda cid: enrollment_counts.get(cid, {}).get('students', 0)
    )

    # Filter out courses without enrollment data
    daily = daily[daily['total_enrolled'] > 0].copy()

    daily['pct_accessed'] = (daily['n_students'] / daily['total_enrolled']) * 100

    # Add course metadata
    daily['course_name'] = daily['course_id'].map(
        lambda cid: enrollment_counts.get(cid, {}).get('name', f'Course {cid}')
    )

    print(f"  Daily records: {len(daily):,}")
    print(f"  Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"  Courses: {daily['course_id'].nunique()}")

    return daily


def plot_course_timelines(daily: pd.DataFrame, enrollment_counts: dict, output_dir: Path):
    """Create timeline plots for all courses in a grid layout."""
    print("\nCreating grid timeline plot...")

    unique_courses = sorted(daily['course_id'].unique())
    n_courses = len(unique_courses)

    # Create figure with subplots (4 columns, ceil(n/4) rows)
    ncols = 4
    nrows = (n_courses + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))
    axes = axes.flatten() if n_courses > 1 else [axes]

    for i, course_id in enumerate(unique_courses):
        ax = axes[i]
        course_data = daily[daily['course_id'] == course_id].sort_values('date')

        # Get course metadata
        meta = enrollment_counts.get(course_id, {})
        course_name = meta.get('name', f'Course {course_id}')
        # Shorten name if too long
        if len(course_name) > 35:
            course_name = course_name[:32] + '...'
        total = meta.get('students', 0)

        # Plot
        ax.plot(course_data['date'], course_data['pct_accessed'],
                linewidth=1.5, color='#2E86AB', alpha=0.8)
        ax.set_title(f'{course_name}\n(N={total} students)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('% Students Accessed', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_ylim(0, max(100, course_data['pct_accessed'].max() * 1.1))

        # Add summary statistics as text
        max_pct = course_data['pct_accessed'].max()
        mean_pct = course_data['pct_accessed'].mean()
        ax.text(0.98, 0.98, f'Max: {max_pct:.1f}%\nAvg: {mean_pct:.1f}%',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Course Activity Timelines - Daily Student Access Patterns',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'course_activity_timelines.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_path}")
    plt.close()


def plot_individual_courses(daily: pd.DataFrame, enrollment_counts: dict, output_dir: Path):
    """Create separate plot for each course."""
    print("\nCreating individual course plots...")

    individual_dir = output_dir / 'individual_timelines'
    individual_dir.mkdir(exist_ok=True)

    for course_id in sorted(daily['course_id'].unique()):
        course_data = daily[daily['course_id'] == course_id].sort_values('date')
        meta = enrollment_counts.get(course_id, {})
        course_name = meta.get('name', f'Course {course_id}')
        total = meta.get('students', 0)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(course_data['date'], course_data['pct_accessed'],
                linewidth=2, color='#2E86AB', marker='o', markersize=3, alpha=0.7)

        # Add horizontal line at mean
        mean_pct = course_data['pct_accessed'].mean()
        ax.axhline(mean_pct, color='red', linestyle='--', linewidth=1,
                   alpha=0.5, label=f'Mean: {mean_pct:.1f}%')

        ax.set_title(f'{course_name} - Daily Student Access\n(Total Enrolled: {total} students)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('% of Students Accessed', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper left', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Clean filename
        clean_name = course_name.replace('/', '-').replace(' ', '_').replace('...', '')
        output_path = individual_dir / f'timeline_{course_id}_{clean_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(daily['course_id'].unique())} individual plots to {individual_dir}")


def generate_summary_statistics(daily: pd.DataFrame, enrollment_counts: dict) -> pd.DataFrame:
    """Generate summary statistics for each course."""
    print("\nGenerating summary statistics...")

    summary = []
    for course_id in sorted(daily['course_id'].unique()):
        course_data = daily[daily['course_id'] == course_id]
        meta = enrollment_counts.get(course_id, {})

        summary.append({
            'course_id': course_id,
            'course_name': meta.get('name', f'Course {course_id}'),
            'total_enrolled': meta.get('students', 0),
            'days_with_activity': len(course_data),
            'max_pct_accessed': course_data['pct_accessed'].max(),
            'mean_pct_accessed': course_data['pct_accessed'].mean(),
            'median_pct_accessed': course_data['pct_accessed'].median(),
            'min_pct_accessed': course_data['pct_accessed'].min(),
            'std_pct_accessed': course_data['pct_accessed'].std(),
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('mean_pct_accessed', ascending=False)

    output_path = Path('data/visualizations/course_activity_summary.csv')
    summary_df.to_csv(output_path, index=False)
    print(f"  Saved summary statistics to {output_path}")

    # Print top/bottom courses by average access
    print("\n  Top 5 courses by average % accessed:")
    for _, row in summary_df.head(5).iterrows():
        print(f"    {row['course_name']}: {row['mean_pct_accessed']:.1f}% avg, {row['max_pct_accessed']:.1f}% max")

    print("\n  Bottom 5 courses by average % accessed:")
    for _, row in summary_df.tail(5).iterrows():
        print(f"    {row['course_name']}: {row['mean_pct_accessed']:.1f}% avg, {row['max_pct_accessed']:.1f}% max")

    return summary_df


def main():
    """Main execution."""
    # Create output directory
    output_dir = Path('data/visualizations')
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    pv, enrollment_counts = load_data()

    # Calculate daily access percentages
    daily = calculate_daily_access_pct(pv, enrollment_counts)

    # Generate visualizations
    plot_course_timelines(daily, enrollment_counts, output_dir)
    plot_individual_courses(daily, enrollment_counts, output_dir)

    # Generate summary statistics
    summary = generate_summary_statistics(daily, enrollment_counts)

    print("\n✅ Course activity timeline plots completed successfully!")
    print(f"\nOutputs:")
    print(f"  - Grid view: {output_dir / 'course_activity_timelines.png'}")
    print(f"  - Individual plots: {output_dir / 'individual_timelines'}/*.png")
    print(f"  - Summary stats: {output_dir / 'course_activity_summary.csv'}")


if __name__ == '__main__':
    main()
