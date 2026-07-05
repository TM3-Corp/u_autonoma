#!/usr/bin/env python3
"""
Plot cumulative first interaction timeline for students in each course.

Shows the percentage of enrolled students who have accessed the course
AT LEAST ONCE by each date (cumulative, starting at 0% and increasing).

This helps identify:
- When students become active during the semester
- Activity distribution patterns across different courses
- Whether low early-cutoff AUC correlates with delayed engagement

Output:
    data/visualizations/cumulative_first_interaction_timeline.png
    data/visualizations/cumulative_first_interaction_summary.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
COURSES_FILE = BASE_DIR / "data/courses_raw.json"
OUTPUT_DIR = BASE_DIR / "data/visualizations"


def normalize_user_id(user_id):
    """Normalize Canvas shard format user_id (155100000000XXXXX) to short ID."""
    user_id_str = str(int(user_id))
    if len(user_id_str) > 12:
        # Extract everything after the 155100000000 prefix
        return int(user_id_str[12:])
    return int(user_id)


def load_data():
    """Load page views and enrollment data."""
    print("Loading data...")

    # Page views
    df_pv = pd.read_parquet(PAGE_VIEWS_FILE)
    df_pv['created_at'] = pd.to_datetime(df_pv['created_at'])
    # Remove timezone info for consistent comparison
    if df_pv['created_at'].dt.tz is not None:
        df_pv['created_at'] = df_pv['created_at'].dt.tz_localize(None)

    # Extract date only
    df_pv['date'] = df_pv['created_at'].dt.date

    print(f"  Page views: {len(df_pv):,} records")

    # Drop rows with missing course_id or user_id
    df_pv = df_pv.dropna(subset=['course_id', 'user_id'])

    # Normalize user_id (Canvas shard format to short ID)
    df_pv['user_id'] = df_pv['user_id'].apply(normalize_user_id)

    # Ensure course_id is integer
    df_pv['course_id'] = df_pv['course_id'].astype(int)

    print(f"  Unique users: {df_pv['user_id'].nunique():,}")
    print(f"  Unique courses: {df_pv['course_id'].nunique():,}")
    print(f"  Date range: {df_pv['date'].min()} to {df_pv['date'].max()}")

    # Enrollments
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    print(f"  Enrollments: {len(df_enroll):,} records")
    print(f"  Courses in enrollments: {df_enroll['course_id'].nunique()}")

    # Load course names
    import json
    with open(COURSES_FILE, 'r') as f:
        courses_data = json.load(f)

    course_names = {c['id']: c.get('name', f'Course {c["id"]}') for c in courses_data}

    return df_pv, df_enroll, course_names


def calculate_cumulative_first_interaction(df_pv, df_enroll, course_names):
    """
    Calculate cumulative % of students who have accessed each course by each date.

    Returns:
        DataFrame with columns: course_id, course_name, date, cumulative_pct, total_enrolled
    """
    print("\nCalculating cumulative first interactions...")

    # Get unique courses that have both page views AND enrollments
    courses_with_pv = set(df_pv['course_id'].unique())
    courses_with_enroll = set(df_enroll['course_id'].unique())
    matched_courses = sorted(courses_with_pv & courses_with_enroll)

    print(f"  Courses with page views: {len(courses_with_pv)}")
    print(f"  Courses with enrollments: {len(courses_with_enroll)}")
    print(f"  Matched courses: {len(matched_courses)}")

    results = []

    for course_id in matched_courses:
        # Get page views for this course
        course_pv = df_pv[df_pv['course_id'] == course_id]

        # Get enrollments for this course
        course_enroll = df_enroll[df_enroll['course_id'] == course_id]
        total_enrolled = len(course_enroll)

        if total_enrolled == 0:
            continue

        # Get first interaction date for each student
        first_interactions = course_pv.groupby('user_id')['date'].min().reset_index()
        first_interactions.columns = ['user_id', 'first_date']

        # Get all unique dates in course
        all_dates = sorted(course_pv['date'].unique())

        # Calculate cumulative count for each date
        for date in all_dates:
            # Count students who accessed by this date
            students_accessed_by_date = len(first_interactions[first_interactions['first_date'] <= date])
            pct_accessed = (students_accessed_by_date / total_enrolled) * 100

            results.append({
                'course_id': course_id,
                'course_name': course_names.get(course_id, f'Course {course_id}'),
                'date': date,
                'cumulative_pct': pct_accessed,
                'total_enrolled': total_enrolled,
                'students_accessed': students_accessed_by_date
            })

    df_cumulative = pd.DataFrame(results)
    print(f"  Generated {len(df_cumulative):,} date-course records for {len(matched_courses)} courses")

    return df_cumulative


def plot_cumulative_timelines(df_cumulative, output_dir):
    """Create timeline plots showing cumulative % of students accessing each course."""
    print("\nCreating cumulative timeline visualizations...")

    unique_courses = sorted(df_cumulative['course_id'].unique())
    n_courses = len(unique_courses)

    # Create figure with subplots (4 columns)
    ncols = 4
    nrows = (n_courses + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 5*nrows))

    # Handle single subplot case
    if n_courses == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, course_id in enumerate(unique_courses):
        ax = axes[i]
        course_data = df_cumulative[df_cumulative['course_id'] == course_id].copy()
        course_data = course_data.sort_values('date')

        # Convert date to datetime for plotting
        course_data['date'] = pd.to_datetime(course_data['date'])

        # Get course metadata
        course_name = course_data.iloc[0]['course_name']
        total_enrolled = course_data.iloc[0]['total_enrolled']

        # Shorten name if too long
        if len(course_name) > 35:
            course_name = course_name[:32] + '...'

        # Plot cumulative percentage
        ax.plot(course_data['date'], course_data['cumulative_pct'],
                linewidth=2, color='#2E86AB', alpha=0.8)

        # Add horizontal line at 100%
        ax.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.3)

        ax.set_title(f'{course_name}\n(N={total_enrolled} enrolled)',
                     fontsize=10, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('% Students Accessed\n(Cumulative)', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.set_ylim(0, min(105, course_data['cumulative_pct'].max() * 1.05))

        # Add final percentage as text
        final_pct = course_data.iloc[-1]['cumulative_pct']
        ax.text(0.98, 0.02, f'Final: {final_pct:.1f}%',
                transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Cumulative First Interaction Timelines\n'
                 '% of Enrolled Students Who Have Accessed Course By Each Date',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()

    output_path = output_dir / 'cumulative_first_interaction_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved grid view: {output_path}")
    plt.close()


def plot_individual_cumulative_timelines(df_cumulative, output_dir):
    """Create individual plots for each course."""
    print("\nCreating individual cumulative timeline plots...")

    individual_dir = output_dir / 'cumulative_timelines_individual'
    individual_dir.mkdir(exist_ok=True)

    for course_id in sorted(df_cumulative['course_id'].unique()):
        course_data = df_cumulative[df_cumulative['course_id'] == course_id].copy()
        course_data = course_data.sort_values('date')
        course_data['date'] = pd.to_datetime(course_data['date'])

        course_name = course_data.iloc[0]['course_name']
        total_enrolled = course_data.iloc[0]['total_enrolled']
        final_pct = course_data.iloc[-1]['cumulative_pct']

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot cumulative percentage
        ax.plot(course_data['date'], course_data['cumulative_pct'],
                linewidth=2.5, color='#2E86AB', marker='o', markersize=4, alpha=0.7)

        # Add horizontal lines
        ax.axhline(100, color='green', linestyle='--', linewidth=1.5,
                   alpha=0.4, label='100% Accessed')
        ax.axhline(50, color='orange', linestyle=':', linewidth=1,
                   alpha=0.3, label='50% Threshold')

        ax.set_title(f'{course_name}\nCumulative First Interaction Timeline\n'
                     f'(Total Enrolled: {total_enrolled} students)',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('% of Students Who Have Accessed At Least Once (Cumulative)', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0, min(105, final_pct * 1.05))

        plt.xticks(rotation=45)
        plt.tight_layout()

        # Clean filename
        clean_name = course_name.replace('/', '-').replace(' ', '_').replace('...', '')[:50]
        output_path = individual_dir / f'cumulative_{course_id}_{clean_name}.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved {len(df_cumulative['course_id'].unique())} individual plots to {individual_dir}")


def generate_summary_statistics(df_cumulative, output_dir):
    """Generate summary statistics for cumulative first interactions."""
    print("\nGenerating summary statistics...")

    summary = []

    for course_id in sorted(df_cumulative['course_id'].unique()):
        course_data = df_cumulative[df_cumulative['course_id'] == course_id].copy()
        course_data = course_data.sort_values('date')

        course_name = course_data.iloc[0]['course_name']
        total_enrolled = course_data.iloc[0]['total_enrolled']

        # Calculate statistics
        final_pct = course_data.iloc[-1]['cumulative_pct']

        # Days to reach key thresholds
        days_to_50 = None
        days_to_80 = None
        if course_data['cumulative_pct'].max() >= 50:
            idx_50 = course_data[course_data['cumulative_pct'] >= 50].index[0]
            days_to_50 = (course_data.loc[idx_50, 'date'] - course_data.iloc[0]['date']).days

        if course_data['cumulative_pct'].max() >= 80:
            idx_80 = course_data[course_data['cumulative_pct'] >= 80].index[0]
            days_to_80 = (course_data.loc[idx_80, 'date'] - course_data.iloc[0]['date']).days

        summary.append({
            'course_id': course_id,
            'course_name': course_name,
            'total_enrolled': total_enrolled,
            'final_pct_accessed': final_pct,
            'days_tracked': len(course_data),
            'days_to_50pct': days_to_50,
            'days_to_80pct': days_to_80,
            'date_first_access': str(course_data.iloc[0]['date']),
            'date_last_access': str(course_data.iloc[-1]['date']),
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values('final_pct_accessed', ascending=False)

    output_path = output_dir / 'cumulative_first_interaction_summary.csv'
    summary_df.to_csv(output_path, index=False)
    print(f"  Saved summary: {output_path}")

    # Print insights
    print("\n  Top 5 courses by final % accessed:")
    for _, row in summary_df.head(5).iterrows():
        print(f"    {row['course_name'][:40]}: {row['final_pct_accessed']:.1f}% "
              f"({row['total_enrolled']} enrolled)")

    print("\n  Bottom 5 courses by final % accessed:")
    for _, row in summary_df.tail(5).iterrows():
        print(f"    {row['course_name'][:40]}: {row['final_pct_accessed']:.1f}% "
              f"({row['total_enrolled']} enrolled)")

    return summary_df


def main():
    """Main execution."""
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load data
    df_pv, df_enroll, course_names = load_data()

    # Calculate cumulative first interactions
    df_cumulative = calculate_cumulative_first_interaction(df_pv, df_enroll, course_names)

    if len(df_cumulative) == 0:
        print("\n❌ No matched courses found between page views and enrollments!")
        return

    # Generate visualizations
    plot_cumulative_timelines(df_cumulative, OUTPUT_DIR)
    plot_individual_cumulative_timelines(df_cumulative, OUTPUT_DIR)

    # Generate summary statistics
    summary = generate_summary_statistics(df_cumulative, OUTPUT_DIR)

    print("\n✅ Cumulative first interaction timeline analysis completed!")
    print(f"\nOutputs:")
    print(f"  - Grid view: {OUTPUT_DIR / 'cumulative_first_interaction_timeline.png'}")
    print(f"  - Individual plots: {OUTPUT_DIR / 'cumulative_timelines_individual'}/*.png")
    print(f"  - Summary CSV: {OUTPUT_DIR / 'cumulative_first_interaction_summary.csv'}")


if __name__ == '__main__':
    main()
