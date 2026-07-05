#!/usr/bin/env python3
"""
Analyze course time ranges for time-limited early warning model.

This script analyzes:
1. First and last interaction per course
2. Comparison with enrollment term dates
3. Activity distribution by week
4. Data availability at different cutoff points (2, 4, 6, 8 weeks)

Output: data/analysis/course_time_ranges.json
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
COURSES_FILE = BASE_DIR / "data/courses_raw.json"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/analysis/course_time_ranges.json"

# Cutoffs to analyze
CUTOFF_WEEKS = [2, 4, 6, 8]

# Courses used in the model (from normalized_features)
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]


def load_data():
    """Load page views and course data."""
    print("Loading data...")

    # Page views
    df_pv = pd.read_parquet(PAGE_VIEWS_FILE)
    df_pv['created_at'] = pd.to_datetime(df_pv['created_at'])
    print(f"  Page views: {len(df_pv):,} records")

    # Courses
    with open(COURSES_FILE, 'r') as f:
        courses_data = json.load(f)
    print(f"  Courses: {len(courses_data)} records")

    # Enrollments (for grade info)
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    print(f"  Enrollments: {len(df_enroll):,} records")

    return df_pv, courses_data, df_enroll


def get_term_dates(courses_data):
    """Extract term dates from courses data."""
    terms = {}
    for course in courses_data:
        if 'term' in course and course['term']:
            term = course['term']
            term_id = term.get('id')
            if term_id and term_id not in terms:
                terms[term_id] = {
                    'id': term_id,
                    'name': term.get('name', ''),
                    'start_at': term.get('start_at'),
                    'end_at': term.get('end_at')
                }
    return terms


def get_course_term_mapping(courses_data):
    """Map course_id to term_id."""
    mapping = {}
    for course in courses_data:
        course_id = course.get('id')
        term_id = course.get('enrollment_term_id')
        if course_id and term_id:
            mapping[course_id] = {
                'term_id': term_id,
                'name': course.get('name', '')
            }
    return mapping


def analyze_course_activity(df_pv, course_id, course_start):
    """Analyze activity distribution for a course."""
    df_course = df_pv[df_pv['course_id'] == course_id].copy()

    if len(df_course) == 0:
        return None

    # Calculate week number for each page view
    df_course['week'] = df_course['created_at'].apply(
        lambda x: max(1, ((x - course_start).days // 7) + 1)
    )

    # Activity by week
    weekly_activity = df_course.groupby('week').size().to_dict()

    # Activity by resource type
    resource_activity = df_course.groupby('resource_type').size().to_dict()

    # Cumulative activity at each cutoff
    cutoff_data = {}
    total_views = len(df_course)

    for cutoff in CUTOFF_WEEKS:
        views_at_cutoff = len(df_course[df_course['week'] <= cutoff])
        cutoff_data[cutoff] = {
            'views': views_at_cutoff,
            'pct_of_total': round(views_at_cutoff / total_views * 100, 1) if total_views > 0 else 0
        }

    return {
        'total_views': total_views,
        'weekly_activity': weekly_activity,
        'resource_activity': resource_activity,
        'cutoff_data': cutoff_data
    }


def analyze_courses(df_pv, courses_data, df_enroll):
    """Main analysis function."""
    terms = get_term_dates(courses_data)
    course_term_map = get_course_term_mapping(courses_data)

    results = {
        'analysis_date': datetime.now().isoformat(),
        'cutoffs_analyzed': CUTOFF_WEEKS,
        'terms': terms,
        'courses': {}
    }

    print("\nAnalyzing courses...")

    for course_id in MODEL_COURSES:
        print(f"\n  Course {course_id}:")

        df_course = df_pv[df_pv['course_id'] == course_id]

        if len(df_course) == 0:
            print(f"    No page views found!")
            continue

        # Basic stats
        first_interaction = df_course['created_at'].min()
        last_interaction = df_course['created_at'].max()
        n_students = df_course['user_id'].nunique()
        total_views = len(df_course)

        # Term info
        course_info = course_term_map.get(course_id, {})
        term_id = course_info.get('term_id')
        term_info = terms.get(term_id, {})

        # Use 5th percentile as course start (to avoid outliers)
        course_start = df_course['created_at'].quantile(0.05)

        # Duration in weeks
        duration_days = (last_interaction - course_start).days
        duration_weeks = duration_days / 7

        print(f"    Name: {course_info.get('name', 'Unknown')[:50]}")
        print(f"    Term: {term_info.get('name', 'Unknown')[:40]}")
        print(f"    First interaction: {first_interaction.strftime('%Y-%m-%d')}")
        print(f"    Last interaction: {last_interaction.strftime('%Y-%m-%d')}")
        print(f"    Course start (5th pct): {course_start.strftime('%Y-%m-%d')}")
        print(f"    Duration: {duration_weeks:.1f} weeks")
        print(f"    Students: {n_students}")
        print(f"    Total views: {total_views:,}")

        # Analyze activity distribution
        activity = analyze_course_activity(df_pv, course_id, course_start)

        if activity:
            print(f"    Activity at cutoffs:")
            for cutoff in CUTOFF_WEEKS:
                cd = activity['cutoff_data'][cutoff]
                print(f"      Week {cutoff}: {cd['views']:,} views ({cd['pct_of_total']}%)")

        # Get enrollment info
        df_course_enroll = df_enroll[df_enroll['course_id'] == course_id]
        n_enrolled = len(df_course_enroll)
        n_failed = len(df_course_enroll[df_course_enroll['final_score'] < 60])
        failure_rate = n_failed / n_enrolled * 100 if n_enrolled > 0 else 0

        print(f"    Enrolled: {n_enrolled}, Failed: {n_failed} ({failure_rate:.1f}%)")

        # Store results
        results['courses'][str(course_id)] = {
            'name': course_info.get('name', 'Unknown'),
            'term_id': term_id,
            'term_name': term_info.get('name', ''),
            'term_start': term_info.get('start_at'),
            'term_end': term_info.get('end_at'),
            'first_interaction': first_interaction.isoformat(),
            'last_interaction': last_interaction.isoformat(),
            'course_start_5pct': course_start.isoformat(),
            'duration_weeks': round(duration_weeks, 1),
            'n_students': n_students,
            'n_enrolled': n_enrolled,
            'n_failed': n_failed,
            'failure_rate': round(failure_rate, 1),
            'total_views': total_views,
            'activity': activity
        }

    return results


def print_summary(results):
    """Print summary of cutoff viability."""
    print("\n" + "="*60)
    print("SUMMARY: Activity at Each Cutoff")
    print("="*60)

    for cutoff in CUTOFF_WEEKS:
        print(f"\n--- Week {cutoff} Cutoff ---")
        total_views = 0
        total_views_at_cutoff = 0

        for course_id, data in results['courses'].items():
            if data.get('activity'):
                cd = data['activity']['cutoff_data'].get(cutoff, {})
                views = cd.get('views', 0)
                pct = cd.get('pct_of_total', 0)
                total_views += data['total_views']
                total_views_at_cutoff += views
                print(f"  Course {course_id}: {views:,} views ({pct}%)")

        overall_pct = total_views_at_cutoff / total_views * 100 if total_views > 0 else 0
        print(f"\n  TOTAL: {total_views_at_cutoff:,} / {total_views:,} views ({overall_pct:.1f}%)")

    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    # Calculate minimum viable cutoff
    for cutoff in CUTOFF_WEEKS:
        all_have_data = True
        min_pct = 100

        for course_id, data in results['courses'].items():
            if data.get('activity'):
                cd = data['activity']['cutoff_data'].get(cutoff, {})
                pct = cd.get('pct_of_total', 0)
                min_pct = min(min_pct, pct)
                if pct < 5:  # Less than 5% of data
                    all_have_data = False

        status = "OK" if all_have_data else "Low data"
        print(f"  Week {cutoff}: {status} (min {min_pct:.0f}% data in any course)")


def main():
    # Load data
    df_pv, courses_data, df_enroll = load_data()

    # Filter to model courses
    df_pv = df_pv[df_pv['course_id'].isin(MODEL_COURSES)].copy()
    print(f"\nFiltered to {len(df_pv):,} page views for {len(MODEL_COURSES)} model courses")

    # Analyze
    results = analyze_courses(df_pv, courses_data, df_enroll)

    # Summary
    print_summary(results)

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
