#!/usr/bin/env python3
"""
Calculate weekly temporal features from page views.

Features:
- sessions_week_X: Sessions in week X of the course
- views_week_X: Total views in week X
- early_vs_late_ratio: Activity in first half vs second half
- week_over_week_change: Average change in activity between weeks
- peak_week: Week with most activity
- activity_consistency: Coefficient of variation of weekly activity
- first_active_week: First week with activity
- last_active_week: Last week with activity
- active_weeks_count: Number of weeks with activity

Resource-specific temporal features:
- quiz_first_access_week: Week of first quiz access
- assignment_first_access_week: Week of first assignment access
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

INPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/categorized_page_views.parquet')
ENROLLMENTS_FILE = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/weekly_features.parquet')
OUTPUT_WEEKLY_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/weekly_activity_detailed.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
CATEGORIES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules', 'grades', 'announcements', 'home']


def calculate_sessions(df, time_col='created_at', gap_minutes=30):
    """Calculate session count from page views using 30-min gap threshold."""
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


def get_course_week(timestamp, course_start):
    """Get the week number relative to course start."""
    if pd.isna(timestamp) or pd.isna(course_start):
        return 0
    delta = (timestamp - course_start).days
    return max(1, (delta // 7) + 1)


def calculate_weekly_features(df_user, course_start, total_weeks=16):
    """Calculate weekly temporal features for a user's page views in a course."""
    if len(df_user) == 0:
        return None, None

    # Parse timestamps
    df_user = df_user.copy()
    df_user['timestamp'] = pd.to_datetime(df_user['created_at'])
    df_user['week'] = df_user['timestamp'].apply(lambda x: get_course_week(x, course_start))

    # Clamp weeks to reasonable range (1-20)
    df_user['week'] = df_user['week'].clip(1, 20)

    # Calculate weekly aggregates
    weekly_stats = []
    for week in range(1, total_weeks + 1):
        week_data = df_user[df_user['week'] == week]

        week_stat = {
            'week': week,
            'views': len(week_data),
            'sessions': calculate_sessions(week_data) if len(week_data) > 0 else 0
        }

        # Per-category views
        for cat in CATEGORIES:
            week_stat[f'{cat}_views'] = len(week_data[week_data['resource_type'] == cat])

        weekly_stats.append(week_stat)

    weekly_df = pd.DataFrame(weekly_stats)

    # Calculate aggregate temporal features
    features = {}

    # Views per week
    active_weeks = weekly_df[weekly_df['views'] > 0]

    if len(active_weeks) == 0:
        return None, None

    features['total_views'] = weekly_df['views'].sum()
    features['total_sessions'] = weekly_df['sessions'].sum()
    features['active_weeks_count'] = len(active_weeks)
    features['first_active_week'] = active_weeks['week'].min()
    features['last_active_week'] = active_weeks['week'].max()
    features['peak_week'] = weekly_df.loc[weekly_df['views'].idxmax(), 'week']

    # Views in first half vs second half
    mid_week = total_weeks // 2
    early_views = weekly_df[weekly_df['week'] <= mid_week]['views'].sum()
    late_views = weekly_df[weekly_df['week'] > mid_week]['views'].sum()
    features['early_semester_views'] = early_views
    features['late_semester_views'] = late_views
    features['early_vs_late_ratio'] = early_views / late_views if late_views > 0 else (early_views if early_views > 0 else 0)

    # Week-over-week change (average)
    weekly_views = weekly_df['views'].values
    if len(weekly_views) > 1:
        changes = []
        for i in range(1, len(weekly_views)):
            if weekly_views[i-1] > 0:
                change = (weekly_views[i] - weekly_views[i-1]) / weekly_views[i-1]
                changes.append(change)
        features['avg_week_over_week_change'] = np.mean(changes) if changes else 0
    else:
        features['avg_week_over_week_change'] = 0

    # Activity consistency (coefficient of variation)
    if len(active_weeks) > 1:
        mean_views = active_weeks['views'].mean()
        std_views = active_weeks['views'].std()
        features['activity_consistency'] = std_views / mean_views if mean_views > 0 else 0
    else:
        features['activity_consistency'] = 0

    # Resource-specific first access timing
    for cat in ['quizzes', 'assignments', 'discussions', 'grades']:
        cat_data = df_user[df_user['resource_type'] == cat]
        if len(cat_data) > 0:
            first_week = cat_data['week'].min()
            features[f'{cat}_first_access_week'] = first_week
        else:
            features[f'{cat}_first_access_week'] = 0

    # Weekly engagement pattern (1=early, 2=middle, 3=late, 4=consistent)
    if features['active_weeks_count'] >= 3:
        early_pct = early_views / (early_views + late_views) if (early_views + late_views) > 0 else 0.5
        if early_pct > 0.7:
            features['engagement_pattern'] = 1  # Early-focused
        elif early_pct < 0.3:
            features['engagement_pattern'] = 3  # Late-focused
        elif features['activity_consistency'] < 0.5:
            features['engagement_pattern'] = 4  # Consistent
        else:
            features['engagement_pattern'] = 2  # Middle/variable
    else:
        features['engagement_pattern'] = 0  # Not enough data

    return features, weekly_df


def main():
    print('=' * 60)
    print('Calculating Weekly Temporal Features')
    print('=' * 60)
    print()

    # Load data
    print('Loading categorized page views...')
    df = pd.read_parquet(INPUT_FILE)
    print(f'Loaded {len(df)} page views')

    # Load enrollments for course start dates
    print('Loading enrollments...')
    enrollments = pd.read_csv(ENROLLMENTS_FILE)
    print(f'Loaded {len(enrollments)} enrollments')

    # Get user_id column
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'

    # Filter to our courses
    df = df[df['course_id'].isin(COURSES)].copy()
    print(f'Filtered to {len(COURSES)} courses: {len(df)} page views')
    print()

    # Estimate course start dates from earliest activity
    print('Estimating course start dates...')
    course_starts = {}
    for course_id in COURSES:
        course_data = df[df['course_id'] == course_id]
        if len(course_data) > 0:
            timestamps = pd.to_datetime(course_data['created_at'])
            # Use 5th percentile as start (avoids outlier early access)
            course_starts[course_id] = timestamps.quantile(0.05)
            print(f'  Course {course_id}: starts ~{course_starts[course_id].strftime("%Y-%m-%d")}')
    print()

    # Calculate features per user per course
    print('Calculating weekly features...')
    all_features = []
    all_weekly = []

    for (user_id, course_id), group in df.groupby([user_col, 'course_id']):
        if course_id not in course_starts:
            continue

        course_start = course_starts[course_id]
        features, weekly_df = calculate_weekly_features(group, course_start)

        if features:
            features['user_id'] = user_id
            features['course_id'] = course_id
            all_features.append(features)

            # Store detailed weekly data
            if weekly_df is not None:
                weekly_df['user_id'] = user_id
                weekly_df['course_id'] = course_id
                all_weekly.append(weekly_df)

    features_df = pd.DataFrame(all_features)
    print(f'Generated features for {len(features_df)} user-course pairs')
    print()

    # Summary stats
    print('Feature summary:')
    summary_cols = ['active_weeks_count', 'early_vs_late_ratio', 'activity_consistency',
                    'quizzes_first_access_week', 'assignments_first_access_week']
    for col in summary_cols:
        if col in features_df.columns:
            print(f'  {col}: mean={features_df[col].mean():.2f}, std={features_df[col].std():.2f}')
    print()

    # Engagement pattern distribution
    if 'engagement_pattern' in features_df.columns:
        pattern_names = {0: 'Insufficient', 1: 'Early-focused', 2: 'Variable', 3: 'Late-focused', 4: 'Consistent'}
        print('Engagement pattern distribution:')
        for pattern, count in features_df['engagement_pattern'].value_counts().sort_index().items():
            pct = count / len(features_df) * 100
            print(f'  {pattern_names.get(pattern, "Unknown")}: {count} ({pct:.1f}%)')
    print()

    # Save aggregate features
    print(f'Saving aggregate features to {OUTPUT_FILE}...')
    features_df.to_parquet(OUTPUT_FILE, index=False)

    # Save detailed weekly data
    if all_weekly:
        weekly_detailed = pd.concat(all_weekly, ignore_index=True)
        print(f'Saving detailed weekly data to {OUTPUT_WEEKLY_FILE}...')
        print(f'  {len(weekly_detailed)} weekly records')
        weekly_detailed.to_parquet(OUTPUT_WEEKLY_FILE, index=False)

    print()
    print('Done!')

    # Preview
    print()
    print('Sample features:')
    print(features_df.head(3).to_string())


if __name__ == '__main__':
    main()
