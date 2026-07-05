#!/usr/bin/env python3
"""
Calculate engagement ratios: course-specific vs total LMS activity.

This script computes the ratio of course-focused sessions/views/time
against total LMS activity, providing context for engagement intensity.

Features created:
- total_sessions: All sessions across entire LMS
- total_views: All page views
- total_time_min: Total time on LMS
- course_sessions: Sessions within specific course
- course_views: Page views within course
- course_time_min: Time within course
- course_session_ratio: course_sessions / total_sessions
- course_views_ratio: course_views / total_views
- course_time_ratio: course_time_min / total_time_min

Aggregation levels:
- Weekly: Per user, per course, per week
- Overall: Per user, per course (semester totals)
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

# Input/Output paths
INPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/all_page_views.parquet')
ENROLLMENTS_FILE = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/engagement_ratios.parquet')
OUTPUT_WEEKLY = Path('/home/paul/projects/uautonoma/data/enriched_features/engagement_ratios_weekly.parquet')

# Session threshold (30 minutes)
SESSION_GAP_MINUTES = 30

# Target courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]


def extract_course_id(url):
    """Extract course_id from Canvas URL using regex."""
    if not url or pd.isna(url):
        return None
    match = re.search(r'/courses/(\d+)', str(url))
    return int(match.group(1)) if match else None


def calculate_sessions(df, time_col='created_at', gap_minutes=30):
    """
    Calculate session count from page views.
    A new session starts after a gap of `gap_minutes` minutes.

    Returns: Number of sessions
    """
    if len(df) == 0:
        return 0

    df = df.sort_values(time_col)
    times = pd.to_datetime(df[time_col])

    if len(times) == 1:
        return 1

    gaps = times.diff()
    threshold = timedelta(minutes=gap_minutes)
    session_starts = (gaps > threshold) | gaps.isna()

    return session_starts.sum()


def calculate_total_engagement(user_df, time_col='created_at'):
    """
    Calculate total engagement metrics for a user across ALL LMS activity.

    Returns dict with:
    - total_sessions
    - total_views
    - total_time_min
    """
    total_views = len(user_df)
    total_sessions = calculate_sessions(user_df, time_col, SESSION_GAP_MINUTES)

    # Calculate total time (sum of interaction_seconds if available)
    if 'interaction_seconds' in user_df.columns:
        total_time_sec = user_df['interaction_seconds'].fillna(0).sum()
    else:
        total_time_sec = 0

    return {
        'total_sessions': total_sessions,
        'total_views': total_views,
        'total_time_min': total_time_sec / 60
    }


def calculate_course_engagement(user_course_df, time_col='created_at'):
    """
    Calculate course-specific engagement metrics.

    Returns dict with:
    - course_sessions
    - course_views
    - course_time_min
    """
    course_views = len(user_course_df)
    course_sessions = calculate_sessions(user_course_df, time_col, SESSION_GAP_MINUTES)

    if 'interaction_seconds' in user_course_df.columns:
        course_time_sec = user_course_df['interaction_seconds'].fillna(0).sum()
    else:
        course_time_sec = 0

    return {
        'course_sessions': course_sessions,
        'course_views': course_views,
        'course_time_min': course_time_sec / 60
    }


def main():
    print('=' * 60)
    print('Calculating Engagement Ratios (Course vs Total LMS)')
    print('=' * 60)
    print()

    # Load all page views (unfiltered)
    print('Loading page views...')
    df = pd.read_parquet(INPUT_FILE)
    print(f'  Loaded {len(df):,} page views')

    # Identify user column
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'
    print(f'  User column: {user_col}')

    # Identify URL column
    url_col = 'http_request' if 'http_request' in df.columns else 'url'
    print(f'  URL column: {url_col}')

    # Identify time column
    time_col = 'created_at'
    print(f'  Time column: {time_col}')

    # Parse timestamps
    print('  Parsing timestamps...')
    df[time_col] = pd.to_datetime(df[time_col], utc=True)

    # Extract course_id from URLs
    print('  Extracting course IDs from URLs...')
    df['course_id'] = df[url_col].apply(extract_course_id)

    # Get unique users
    unique_users = df[user_col].dropna().unique()
    print(f'  Unique users: {len(unique_users)}')
    print()

    # Load enrollments to know which users are in which courses
    print('Loading enrollments...')
    enrollments = pd.read_csv(ENROLLMENTS_FILE)
    print(f'  Loaded {len(enrollments)} enrollments')

    # Filter to target courses
    enrollments = enrollments[enrollments['course_id'].isin(COURSES)]
    print(f'  Enrollments in target courses: {len(enrollments)}')
    print()

    # Calculate engagement ratios
    print('Calculating engagement ratios...')
    results = []
    results_weekly = []

    # Group by user
    user_groups = df.groupby(user_col)
    processed = 0

    for user_id, user_df in user_groups:
        # Get total engagement for this user
        total_metrics = calculate_total_engagement(user_df, time_col)

        # Get courses this user is enrolled in
        user_enrollments = enrollments[enrollments['user_id'] == user_id]

        if len(user_enrollments) == 0:
            continue

        # For each course the user is enrolled in
        for _, enrollment in user_enrollments.iterrows():
            course_id = enrollment['course_id']

            # Filter to this course's page views
            course_df = user_df[user_df['course_id'] == course_id]

            # Calculate course-specific engagement
            course_metrics = calculate_course_engagement(course_df, time_col)

            # Calculate ratios (avoid division by zero)
            session_ratio = (course_metrics['course_sessions'] / total_metrics['total_sessions']
                           if total_metrics['total_sessions'] > 0 else 0)
            views_ratio = (course_metrics['course_views'] / total_metrics['total_views']
                         if total_metrics['total_views'] > 0 else 0)
            time_ratio = (course_metrics['course_time_min'] / total_metrics['total_time_min']
                        if total_metrics['total_time_min'] > 0 else 0)

            # Store overall results
            results.append({
                'user_id': user_id,
                'course_id': course_id,
                **total_metrics,
                **course_metrics,
                'course_session_ratio': session_ratio,
                'course_views_ratio': views_ratio,
                'course_time_ratio': time_ratio
            })

            # Calculate weekly aggregations
            if len(course_df) > 0:
                course_df = course_df.copy()
                course_df['week'] = course_df[time_col].dt.isocalendar().week

                # Also need weekly totals for this user
                user_df_copy = user_df.copy()
                user_df_copy['week'] = user_df_copy[time_col].dt.isocalendar().week

                for week in course_df['week'].unique():
                    week_user_df = user_df_copy[user_df_copy['week'] == week]
                    week_course_df = course_df[course_df['week'] == week]

                    week_total = calculate_total_engagement(week_user_df, time_col)
                    week_course = calculate_course_engagement(week_course_df, time_col)

                    week_session_ratio = (week_course['course_sessions'] / week_total['total_sessions']
                                         if week_total['total_sessions'] > 0 else 0)
                    week_views_ratio = (week_course['course_views'] / week_total['total_views']
                                       if week_total['total_views'] > 0 else 0)
                    week_time_ratio = (week_course['course_time_min'] / week_total['total_time_min']
                                      if week_total['total_time_min'] > 0 else 0)

                    results_weekly.append({
                        'user_id': user_id,
                        'course_id': course_id,
                        'week': week,
                        **{f'week_{k}': v for k, v in week_total.items()},
                        **{f'week_{k}': v for k, v in week_course.items()},
                        'week_course_session_ratio': week_session_ratio,
                        'week_course_views_ratio': week_views_ratio,
                        'week_course_time_ratio': week_time_ratio
                    })

        processed += 1
        if processed % 50 == 0:
            print(f'  Processed {processed} users...')

    print(f'  Processed {processed} users total')
    print()

    # Create DataFrames
    results_df = pd.DataFrame(results)
    results_weekly_df = pd.DataFrame(results_weekly)

    print('Results summary (Overall):')
    print(f'  Total rows: {len(results_df)}')
    print(f'  Unique users: {results_df["user_id"].nunique()}')
    print(f'  Unique courses: {results_df["course_id"].nunique()}')
    print()

    if len(results_df) > 0:
        print('  Engagement ratio statistics:')
        print(f'    course_session_ratio: mean={results_df["course_session_ratio"].mean():.3f}, '
              f'median={results_df["course_session_ratio"].median():.3f}')
        print(f'    course_views_ratio: mean={results_df["course_views_ratio"].mean():.3f}, '
              f'median={results_df["course_views_ratio"].median():.3f}')
        print(f'    course_time_ratio: mean={results_df["course_time_ratio"].mean():.3f}, '
              f'median={results_df["course_time_ratio"].median():.3f}')
    print()

    print('Results summary (Weekly):')
    print(f'  Total rows: {len(results_weekly_df)}')
    if len(results_weekly_df) > 0:
        print(f'  Weeks covered: {results_weekly_df["week"].nunique()}')
    print()

    # Save results
    print(f'Saving overall results to {OUTPUT_FILE}...')
    results_df.to_parquet(OUTPUT_FILE, index=False)

    print(f'Saving weekly results to {OUTPUT_WEEKLY}...')
    results_weekly_df.to_parquet(OUTPUT_WEEKLY, index=False)

    print()
    print('Done!')


if __name__ == '__main__':
    main()
