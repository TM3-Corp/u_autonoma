#!/usr/bin/env python3
"""
Plot first interaction timeline for students in each course.

Visualizes when students actually started interacting with each course,
compared to official term dates. Shows:
- Scatter plot: Each point = first interaction of a student
- Color: Green (passed) / Red (failed)
- Vertical lines: Term start date and 5th percentile start

Output:
    data/report/analysis/first_interaction_timeline.png
    data/report/analysis/first_interaction_stats.json
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
COURSES_FILE = BASE_DIR / "data/courses_raw.json"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_DIR = BASE_DIR / "data/report/analysis"

# Model courses
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Style
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'pass': '#27ae60',      # Green
    'fail': '#e74c3c',      # Red
    'term_start': '#3498db', # Blue
    '5th_pct': '#9b59b6',   # Purple
}


def normalize_user_id(user_id):
    """Normalize Canvas shard format user_id (155100000000XXXXX) to short ID."""
    user_id_str = str(int(user_id))
    if len(user_id_str) > 12:
        # Extract everything after the 155100000000 prefix
        return int(user_id_str[12:])
    return int(user_id)


def load_data():
    """Load page views, courses, and enrollment data."""
    print("Loading data...")

    # Page views
    df_pv = pd.read_parquet(PAGE_VIEWS_FILE)
    df_pv['created_at'] = pd.to_datetime(df_pv['created_at'])
    # Remove timezone info for consistent comparison
    if df_pv['created_at'].dt.tz is not None:
        df_pv['created_at'] = df_pv['created_at'].dt.tz_localize(None)
    print(f"  Page views: {len(df_pv):,} records")

    # Drop rows with missing course_id
    df_pv = df_pv.dropna(subset=['course_id', 'user_id'])

    # Normalize user_id (Canvas shard format to short ID)
    df_pv['user_id'] = df_pv['user_id'].apply(normalize_user_id)

    # Ensure course_id is integer for consistent merging
    df_pv['course_id'] = df_pv['course_id'].astype(int)

    # Filter to model courses
    df_pv = df_pv[df_pv['course_id'].isin(MODEL_COURSES)].copy()
    print(f"  Filtered to model courses: {len(df_pv):,} records")

    # Courses
    with open(COURSES_FILE, 'r') as f:
        courses_data = json.load(f)
    print(f"  Courses: {len(courses_data)} records")

    # Enrollments
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    df_enroll['failed'] = (df_enroll['final_score'] < 60).astype(int)
    print(f"  Enrollments: {len(df_enroll):,} records")
    print(f"    Passed: {len(df_enroll[df_enroll['failed'] == 0])}")
    print(f"    Failed: {len(df_enroll[df_enroll['failed'] == 1])}")

    return df_pv, courses_data, df_enroll


def get_course_info(courses_data):
    """Extract course names and term info."""
    course_info = {}
    for course in courses_data:
        course_id = course.get('id')
        if course_id in MODEL_COURSES:
            term = course.get('term', {})
            # Parse and make timezone-naive
            term_start = term.get('start_at')
            term_end = term.get('end_at')
            if term_start:
                term_start_dt = pd.to_datetime(term_start)
                if term_start_dt.tz is not None:
                    term_start_dt = term_start_dt.tz_localize(None)
                term_start = term_start_dt
            if term_end:
                term_end_dt = pd.to_datetime(term_end)
                if term_end_dt.tz is not None:
                    term_end_dt = term_end_dt.tz_localize(None)
                term_end = term_end_dt

            course_info[course_id] = {
                'name': course.get('name', f'Course {course_id}'),
                'term_name': term.get('name', ''),
                'term_start': term_start,
                'term_end': term_end,
            }
    return course_info


def calculate_first_interactions(df_pv, df_enroll):
    """Calculate first interaction per student per course."""
    print("\nCalculating first interactions...")

    # Get first interaction per student per course
    first_interactions = df_pv.groupby(['course_id', 'user_id'])['created_at'].min().reset_index()
    first_interactions.columns = ['course_id', 'user_id', 'first_interaction']

    print(f"  Found {len(first_interactions)} student-course combinations")

    # Merge with enrollment data (pass/fail)
    first_interactions = first_interactions.merge(
        df_enroll[['user_id', 'course_id', 'failed', 'final_score']],
        on=['user_id', 'course_id'],
        how='left'
    )

    # Fill missing (students with page views but no enrollment record)
    first_interactions['failed'] = first_interactions['failed'].fillna(1).astype(int)

    print(f"  With pass/fail status: {len(first_interactions)} records")
    print(f"  Passed: {len(first_interactions[first_interactions['failed'] == 0])}")
    print(f"  Failed: {len(first_interactions[first_interactions['failed'] == 1])}")

    return first_interactions


def calculate_course_stats(first_interactions, df_pv, course_info):
    """Calculate statistics per course."""
    stats = {}

    for course_id in MODEL_COURSES:
        df_course = first_interactions[first_interactions['course_id'] == course_id]
        df_pv_course = df_pv[df_pv['course_id'] == course_id]

        if len(df_course) == 0:
            continue

        # First interactions
        first_dates = df_course['first_interaction']

        # 5th percentile (used as course start)
        pct_5 = first_dates.quantile(0.05)

        # Term start (already parsed as datetime in get_course_info)
        info = course_info.get(course_id, {})
        term_start = info.get('term_start')

        # Calculate days from term start to first interaction
        if term_start is not None:
            days_to_first = (first_dates - term_start).dt.days
            days_to_5th_pct = int((pct_5 - term_start).days)
            term_start_str = term_start.isoformat()
        else:
            days_to_first = None
            days_to_5th_pct = None
            term_start_str = None

        stats[course_id] = {
            'name': info.get('name', f'Course {course_id}'),
            'term_name': info.get('term_name', ''),
            'term_start': term_start_str,
            'n_students': len(df_course),
            'n_passed': len(df_course[df_course['failed'] == 0]),
            'n_failed': len(df_course[df_course['failed'] == 1]),
            'first_interaction_earliest': first_dates.min().isoformat(),
            'first_interaction_5th_pct': pct_5.isoformat(),
            'first_interaction_median': first_dates.median().isoformat(),
            'first_interaction_latest': first_dates.max().isoformat(),
            'days_from_term_to_earliest': int(days_to_first.min()) if days_to_first is not None else None,
            'days_from_term_to_5th_pct': days_to_5th_pct,
            'days_from_term_to_median': int(days_to_first.median()) if days_to_first is not None else None,
        }

    return stats


def plot_timeline(first_interactions, course_info, stats):
    """Create the timeline visualization."""
    print("\nCreating timeline visualization...")

    # Sort courses by 5th percentile start date
    course_order = sorted(
        stats.keys(),
        key=lambda x: pd.to_datetime(stats[x]['first_interaction_5th_pct'])
    )

    fig, ax = plt.subplots(figsize=(14, 10))

    # Y positions for courses
    y_positions = {course_id: i for i, course_id in enumerate(course_order)}

    # Track all dates for x-axis range
    all_dates = []

    for course_id in course_order:
        df_course = first_interactions[first_interactions['course_id'] == course_id]
        y_pos = y_positions[course_id]
        info = course_info.get(course_id, {})

        # Separate passed and failed
        passed = df_course[df_course['failed'] == 0]
        failed = df_course[df_course['failed'] == 1]

        # Add small jitter for y to avoid overlap
        jitter = 0.15

        # Plot failed students (bottom, red)
        if len(failed) > 0:
            ax.scatter(
                failed['first_interaction'],
                [y_pos - jitter] * len(failed),
                c=COLORS['fail'],
                alpha=0.6,
                s=30,
                label='Failed' if course_id == course_order[0] else '',
                marker='x'
            )
            all_dates.extend(failed['first_interaction'].tolist())

        # Plot passed students (top, green)
        if len(passed) > 0:
            ax.scatter(
                passed['first_interaction'],
                [y_pos + jitter] * len(passed),
                c=COLORS['pass'],
                alpha=0.6,
                s=30,
                label='Passed' if course_id == course_order[0] else '',
                marker='o'
            )
            all_dates.extend(passed['first_interaction'].tolist())

        # Add vertical line for 5th percentile
        pct_5 = pd.to_datetime(stats[course_id]['first_interaction_5th_pct'])
        ax.axvline(x=pct_5, ymin=(y_pos - 0.4 + 0.5) / len(course_order),
                   ymax=(y_pos + 0.4 + 0.5) / len(course_order),
                   color=COLORS['5th_pct'], linestyle='--', linewidth=1.5, alpha=0.7)

        # Add vertical line for term start if available
        term_start = info.get('term_start')
        if term_start is not None:
            ax.axvline(x=term_start, ymin=(y_pos - 0.4 + 0.5) / len(course_order),
                       ymax=(y_pos + 0.4 + 0.5) / len(course_order),
                       color=COLORS['term_start'], linestyle=':', linewidth=1.5, alpha=0.7)
            all_dates.append(term_start)

    # Set y-axis labels
    y_labels = []
    for course_id in course_order:
        name = stats[course_id]['name']
        # Truncate long names
        if len(name) > 40:
            name = name[:37] + '...'
        n_students = stats[course_id]['n_students']
        fail_rate = stats[course_id]['n_failed'] / n_students * 100
        y_labels.append(f"{name}\n(n={n_students}, {fail_rate:.0f}% fail)")

    ax.set_yticks(range(len(course_order)))
    ax.set_yticklabels(y_labels, fontsize=9)

    # X-axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45, ha='right')

    # Set x-axis range with padding
    if all_dates:
        min_date = min(all_dates)
        max_date = max(all_dates)
        date_range = (max_date - min_date).days
        padding = pd.Timedelta(days=max(7, date_range * 0.05))
        ax.set_xlim(min_date - padding, max_date + padding)

    # Labels and title
    ax.set_xlabel('Date of First Interaction', fontsize=12)
    ax.set_ylabel('Course', fontsize=12)
    ax.set_title('Student First Interaction Timeline by Course\n'
                 'Each point = when a student first accessed the course',
                 fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['pass'],
               markersize=8, label='Passed'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor=COLORS['fail'],
               markeredgewidth=2, markersize=8, label='Failed'),
        Line2D([0], [0], color=COLORS['5th_pct'], linestyle='--', linewidth=2,
               label='5th Percentile Start'),
        Line2D([0], [0], color=COLORS['term_start'], linestyle=':', linewidth=2,
               label='Term Start Date'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Grid
    ax.grid(True, axis='x', alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'first_interaction_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")

    return output_path


def plot_distribution_boxplot(first_interactions, stats):
    """Create boxplot showing distribution of days to first interaction."""
    print("\nCreating distribution boxplot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Calculate days from 5th percentile for each student
    data_for_plot = []
    labels = []

    for course_id in sorted(stats.keys(), key=lambda x: stats[x]['name']):
        df_course = first_interactions[first_interactions['course_id'] == course_id].copy()
        pct_5 = pd.to_datetime(stats[course_id]['first_interaction_5th_pct'])

        # Days from 5th percentile
        df_course['days_from_start'] = (df_course['first_interaction'] - pct_5).dt.days

        data_for_plot.append(df_course['days_from_start'].values)

        name = stats[course_id]['name']
        if len(name) > 25:
            name = name[:22] + '...'
        labels.append(name)

    # Create boxplot
    bp = ax.boxplot(data_for_plot, tick_labels=labels, patch_artist=True, vert=True)

    # Color the boxes
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.6)

    # Rotate labels
    plt.xticks(rotation=45, ha='right')

    # Add reference line at 0
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Course Start (5th pct)')

    # Labels
    ax.set_xlabel('Course', fontsize=12)
    ax.set_ylabel('Days from Course Start (5th percentile)', fontsize=12)
    ax.set_title('Distribution of Student First Interactions Relative to Course Start\n'
                 '(Negative = before 5th percentile, Positive = after)',
                 fontsize=14, fontweight='bold')

    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'first_interaction_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def plot_pass_fail_comparison(first_interactions, stats):
    """Compare first interaction timing between passed and failed students."""
    print("\nCreating pass/fail comparison...")

    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()

    sorted_courses = sorted(stats.keys(), key=lambda x: stats[x]['name'])

    for idx, course_id in enumerate(sorted_courses):
        ax = axes[idx]
        df_course = first_interactions[first_interactions['course_id'] == course_id].copy()
        pct_5 = pd.to_datetime(stats[course_id]['first_interaction_5th_pct'])

        # Days from 5th percentile
        df_course['days_from_start'] = (df_course['first_interaction'] - pct_5).dt.days

        passed = df_course[df_course['failed'] == 0]['days_from_start']
        failed = df_course[df_course['failed'] == 1]['days_from_start']

        # Create violin or box comparison
        data = [passed.values if len(passed) > 0 else [0],
                failed.values if len(failed) > 0 else [0]]

        bp = ax.boxplot(data, tick_labels=['Pass', 'Fail'], patch_artist=True)
        bp['boxes'][0].set_facecolor(COLORS['pass'])
        bp['boxes'][1].set_facecolor(COLORS['fail'])
        for box in bp['boxes']:
            box.set_alpha(0.7)

        # Title
        name = stats[course_id]['name']
        if len(name) > 20:
            name = name[:17] + '...'
        ax.set_title(name, fontsize=9)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)

        # Calculate and show medians
        if len(passed) > 0 and len(failed) > 0:
            median_diff = failed.median() - passed.median()
            ax.text(0.5, 0.02, f'Δmed: {median_diff:+.1f}d',
                   transform=ax.transAxes, ha='center', fontsize=8,
                   color='gray')

    fig.suptitle('First Interaction Timing: Passed vs Failed Students\n'
                 '(Days from course start)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'first_interaction_pass_fail.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary(stats):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("FIRST INTERACTION SUMMARY")
    print("="*70)

    print(f"\n{'Course':<45} {'N':>5} {'Days to 5th%':>12} {'Days to Med':>12}")
    print("-"*70)

    for course_id in sorted(stats.keys(), key=lambda x: stats[x]['name']):
        s = stats[course_id]
        name = s['name'][:42] + '...' if len(s['name']) > 45 else s['name']
        days_5th = s['days_from_term_to_5th_pct']
        days_med = s['days_from_term_to_median']

        days_5th_str = f"{days_5th:+d}" if days_5th is not None else "N/A"
        days_med_str = f"{days_med:+d}" if days_med is not None else "N/A"

        print(f"{name:<45} {s['n_students']:>5} {days_5th_str:>12} {days_med_str:>12}")

    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    # Calculate overall statistics
    total_students = sum(s['n_students'] for s in stats.values())
    total_passed = sum(s['n_passed'] for s in stats.values())
    total_failed = sum(s['n_failed'] for s in stats.values())

    print(f"\nTotal students with page views: {total_students}")
    print(f"  Passed: {total_passed} ({total_passed/total_students*100:.1f}%)")
    print(f"  Failed: {total_failed} ({total_failed/total_students*100:.1f}%)")

    # Days from term to 5th percentile
    valid_days = [s['days_from_term_to_5th_pct'] for s in stats.values()
                  if s['days_from_term_to_5th_pct'] is not None]
    if valid_days:
        print(f"\nDays from term start to course activity (5th percentile):")
        print(f"  Range: {min(valid_days)} to {max(valid_days)} days")
        print(f"  Mean: {np.mean(valid_days):.1f} days")


def main():
    """Main function."""
    print("="*60)
    print("First Interaction Timeline Analysis")
    print("="*60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df_pv, courses_data, df_enroll = load_data()

    # Get course info
    course_info = get_course_info(courses_data)

    # Calculate first interactions
    first_interactions = calculate_first_interactions(df_pv, df_enroll)

    # Calculate statistics
    stats = calculate_course_stats(first_interactions, df_pv, course_info)

    # Save statistics
    stats_path = OUTPUT_DIR / 'first_interaction_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"\nSaved statistics: {stats_path}")

    # Create visualizations
    plot_timeline(first_interactions, course_info, stats)
    plot_distribution_boxplot(first_interactions, stats)
    plot_pass_fail_comparison(first_interactions, stats)

    # Print summary
    print_summary(stats)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
