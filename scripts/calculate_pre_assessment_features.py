#!/usr/bin/env python3
"""
Calculate Pre-Assessment Activity Features.

This script captures the "deadline-relative dimension" - how students behave
in relation to upcoming assessments. Based on findings from:
- Oviedo paper: QAT (Quiz Access Time), pre-deadline activity patterns
- Beyond Time on Task: File access patterns before assessments

Features generated per student per course:

1. Deadline-Relative Activity (for courses with due dates):
   - activity_24h_before: Page views in 24h window before any deadline
   - activity_48h_before: Page views in 48h window before any deadline
   - activity_72h_before: Page views in 72h window before any deadline
   - files_before_deadline: File accesses before deadlines
   - preparation_intensity: Ratio of pre-deadline to total activity

2. Quiz Access Time (QAT) Features:
   - first_quiz_access_days: Days from course start to first quiz access
   - quiz_access_pct: Percentile ranking among classmates (1.0 = first)
   - early_quiz_accessor: Flag if in top 25% of quiz accessors

3. Assessment Engagement Patterns:
   - quiz_sessions: Number of distinct quiz sessions
   - assignment_sessions: Number of distinct assignment sessions
   - assessment_diversity: How many different assessments accessed
   - assessment_revisits: Average revisits per assessment

4. Temporal Preparation Patterns:
   - early_half_activity_pct: % of activity in first half of course
   - late_surge_ratio: Activity in last 25% vs middle 50%
   - consistent_preparer: Low variance in weekly activity before deadlines
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import json

# Paths
INPUT_PAGE_VIEWS = Path('/home/paul/projects/uautonoma/data/page_views/categorized_page_views.parquet')
INPUT_ASSIGNMENTS = Path('/home/paul/projects/uautonoma/data/assignment_analytics.json')
INPUT_ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/pre_assessment_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Our courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Session threshold (30 minutes)
SESSION_GAP_MINUTES = 30


def load_data():
    """Load page views, assignments, and enrollments."""
    print("Loading data...")

    # Page views
    df = pd.read_parquet(INPUT_PAGE_VIEWS)
    df['created_at'] = pd.to_datetime(df['created_at']).dt.tz_localize(None)  # Remove timezone
    df = df[df['course_id'].isin(COURSES)].copy()
    print(f"  Page views: {len(df):,}")

    # Assignments with due dates
    with open(INPUT_ASSIGNMENTS) as f:
        assignments = json.load(f)

    # Filter to our courses and those with due dates
    assignments = [a for a in assignments
                   if a.get('course_id') in COURSES and a.get('due_at')]
    print(f"  Assignments with due dates: {len(assignments)}")

    # Convert to DataFrame
    assignments_df = pd.DataFrame(assignments)
    if len(assignments_df) > 0:
        assignments_df['due_at'] = pd.to_datetime(assignments_df['due_at']).dt.tz_localize(None)  # Remove timezone

    # Enrollments
    enrollments = pd.read_csv(INPUT_ENROLLMENTS)
    print(f"  Enrollments: {len(enrollments)}")

    return df, assignments_df, enrollments


def get_course_date_range(df_course):
    """Get the date range for a course based on activity."""
    timestamps = pd.to_datetime(df_course['created_at'])
    return timestamps.min(), timestamps.max()


def calculate_deadline_features(df_user, assignments_df, course_id):
    """Calculate activity relative to deadlines for a user."""
    features = {
        'activity_24h_before': 0,
        'activity_48h_before': 0,
        'activity_72h_before': 0,
        'files_24h_before': 0,
        'files_48h_before': 0,
        'files_72h_before': 0,
        'n_deadlines': 0,
        'preparation_intensity': 0.0,
    }

    # Get assignments for this course
    course_assignments = assignments_df[assignments_df['course_id'] == course_id]

    if len(course_assignments) == 0 or len(df_user) == 0:
        return features

    features['n_deadlines'] = len(course_assignments)

    user_timestamps = pd.to_datetime(df_user['created_at'])
    user_files = df_user[df_user['resource_type'] == 'files']
    file_timestamps = pd.to_datetime(user_files['created_at']) if len(user_files) > 0 else pd.Series(dtype='datetime64[ns]')

    total_pre_deadline = 0
    total_activity = len(df_user)

    for _, assignment in course_assignments.iterrows():
        due_at = assignment['due_at']

        # Calculate time windows before deadline
        window_24h = (due_at - timedelta(hours=24), due_at)
        window_48h = (due_at - timedelta(hours=48), due_at)
        window_72h = (due_at - timedelta(hours=72), due_at)

        # Count activity in each window
        in_24h = ((user_timestamps >= window_24h[0]) & (user_timestamps <= window_24h[1])).sum()
        in_48h = ((user_timestamps >= window_48h[0]) & (user_timestamps <= window_48h[1])).sum()
        in_72h = ((user_timestamps >= window_72h[0]) & (user_timestamps <= window_72h[1])).sum()

        features['activity_24h_before'] += in_24h
        features['activity_48h_before'] += in_48h
        features['activity_72h_before'] += in_72h

        # Count file accesses in windows
        if len(file_timestamps) > 0:
            files_24h = ((file_timestamps >= window_24h[0]) & (file_timestamps <= window_24h[1])).sum()
            files_48h = ((file_timestamps >= window_48h[0]) & (file_timestamps <= window_48h[1])).sum()
            files_72h = ((file_timestamps >= window_72h[0]) & (file_timestamps <= window_72h[1])).sum()

            features['files_24h_before'] += files_24h
            features['files_48h_before'] += files_48h
            features['files_72h_before'] += files_72h

        total_pre_deadline += in_72h

    # Preparation intensity: ratio of pre-deadline activity to total
    if total_activity > 0:
        features['preparation_intensity'] = total_pre_deadline / total_activity

    # Normalize by number of deadlines
    if features['n_deadlines'] > 0:
        features['activity_24h_per_deadline'] = features['activity_24h_before'] / features['n_deadlines']
        features['activity_48h_per_deadline'] = features['activity_48h_before'] / features['n_deadlines']
        features['activity_72h_per_deadline'] = features['activity_72h_before'] / features['n_deadlines']
    else:
        features['activity_24h_per_deadline'] = 0
        features['activity_48h_per_deadline'] = 0
        features['activity_72h_per_deadline'] = 0

    return features


def calculate_qat_features(df_course, user_id, course_start, user_col):
    """Calculate Quiz Access Time features for a user."""
    features = {
        'first_quiz_access_days': np.nan,
        'first_assignment_access_days': np.nan,
        'first_assessment_access_days': np.nan,
        'quiz_access_count': 0,
        'assignment_access_count': 0,
    }

    # Get user's quiz and assignment accesses
    user_quizzes = df_course[(df_course[user_col] == user_id) &
                              (df_course['resource_type'] == 'quizzes')]
    user_assignments = df_course[(df_course[user_col] == user_id) &
                                  (df_course['resource_type'] == 'assignments')]

    features['quiz_access_count'] = len(user_quizzes)
    features['assignment_access_count'] = len(user_assignments)

    if len(user_quizzes) > 0:
        first_quiz = pd.to_datetime(user_quizzes['created_at']).min()
        features['first_quiz_access_days'] = (first_quiz - course_start).days

    if len(user_assignments) > 0:
        first_assignment = pd.to_datetime(user_assignments['created_at']).min()
        features['first_assignment_access_days'] = (first_assignment - course_start).days

    # First assessment (quiz or assignment, whichever comes first)
    first_times = []
    if len(user_quizzes) > 0:
        first_times.append(pd.to_datetime(user_quizzes['created_at']).min())
    if len(user_assignments) > 0:
        first_times.append(pd.to_datetime(user_assignments['created_at']).min())

    if first_times:
        first_assessment = min(first_times)
        features['first_assessment_access_days'] = (first_assessment - course_start).days

    return features


def calculate_qat_percentiles(df_course, enrolled_users, course_start, user_col):
    """Calculate QAT percentile rankings for all users in a course."""
    # Get first quiz access for each user
    quiz_df = df_course[df_course['resource_type'] == 'quizzes']

    first_access = {}
    for user_id in enrolled_users:
        user_quizzes = quiz_df[quiz_df[user_col] == user_id]
        if len(user_quizzes) > 0:
            first = pd.to_datetime(user_quizzes['created_at']).min()
            first_access[user_id] = first

    if len(first_access) == 0:
        return {user_id: {'quiz_access_pct': 0.0, 'early_quiz_accessor': 0}
                for user_id in enrolled_users}

    # Rank users by first access time
    sorted_users = sorted(first_access.items(), key=lambda x: x[1])
    n_accessors = len(sorted_users)

    percentiles = {}
    for rank, (user_id, _) in enumerate(sorted_users, 1):
        # First accessor gets 1.0, last gets 1/N
        pct = (n_accessors - rank + 1) / n_accessors
        percentiles[user_id] = {
            'quiz_access_pct': pct,
            'early_quiz_accessor': 1 if pct >= 0.75 else 0
        }

    # Users who never accessed get 0
    for user_id in enrolled_users:
        if user_id not in percentiles:
            percentiles[user_id] = {
                'quiz_access_pct': 0.0,
                'early_quiz_accessor': 0
            }

    return percentiles


def calculate_assessment_patterns(df_user):
    """Calculate assessment engagement patterns for a user."""
    features = {
        'unique_quizzes_accessed': 0,
        'unique_assignments_accessed': 0,
        'assessment_diversity': 0,
        'quiz_revisits': 0.0,
        'assignment_revisits': 0.0,
        'quiz_sessions': 0,
        'assignment_sessions': 0,
    }

    if len(df_user) == 0:
        return features

    # Unique assessments
    quiz_df = df_user[df_user['resource_type'] == 'quizzes']
    assgn_df = df_user[df_user['resource_type'] == 'assignments']

    if len(quiz_df) > 0:
        unique_quizzes = quiz_df['resource_id'].nunique()
        features['unique_quizzes_accessed'] = unique_quizzes
        features['quiz_revisits'] = len(quiz_df) / unique_quizzes if unique_quizzes > 0 else 0

        # Count quiz sessions (30-min gap)
        quiz_times = pd.to_datetime(quiz_df['created_at']).sort_values()
        gaps = quiz_times.diff().dt.total_seconds() / 60
        features['quiz_sessions'] = (gaps >= SESSION_GAP_MINUTES).sum() + 1

    if len(assgn_df) > 0:
        unique_assgn = assgn_df['resource_id'].nunique()
        features['unique_assignments_accessed'] = unique_assgn
        features['assignment_revisits'] = len(assgn_df) / unique_assgn if unique_assgn > 0 else 0

        # Count assignment sessions
        assgn_times = pd.to_datetime(assgn_df['created_at']).sort_values()
        gaps = assgn_times.diff().dt.total_seconds() / 60
        features['assignment_sessions'] = (gaps >= SESSION_GAP_MINUTES).sum() + 1

    features['assessment_diversity'] = (features['unique_quizzes_accessed'] +
                                         features['unique_assignments_accessed'])

    return features


def calculate_temporal_patterns(df_user, course_start, course_end):
    """Calculate temporal preparation patterns."""
    features = {
        'early_half_activity_pct': 0.0,
        'late_surge_ratio': 0.0,
        'activity_acceleration': 0.0,
        'consistent_preparer': 0,
    }

    if len(df_user) == 0 or course_start >= course_end:
        return features

    user_timestamps = pd.to_datetime(df_user['created_at'])
    course_duration = (course_end - course_start).days
    midpoint = course_start + timedelta(days=course_duration / 2)

    # Early vs late activity
    early_activity = (user_timestamps <= midpoint).sum()
    total_activity = len(df_user)

    features['early_half_activity_pct'] = early_activity / total_activity if total_activity > 0 else 0

    # Late surge: activity in last 25% vs middle 50%
    q75_point = course_start + timedelta(days=course_duration * 0.75)
    q25_point = course_start + timedelta(days=course_duration * 0.25)

    late_activity = (user_timestamps >= q75_point).sum()
    middle_activity = ((user_timestamps >= q25_point) & (user_timestamps < q75_point)).sum()

    # Avoid division by zero - if no middle activity, check if there's late activity
    if middle_activity > 0:
        features['late_surge_ratio'] = late_activity / middle_activity
    elif late_activity > 0:
        features['late_surge_ratio'] = 2.0  # High late surge

    # Activity acceleration: comparing first and second half
    if early_activity > 0 and total_activity > early_activity:
        late_total = total_activity - early_activity
        features['activity_acceleration'] = (late_total - early_activity) / early_activity

    # Consistent preparer: calculate weekly variance
    df_user = df_user.copy()
    df_user['week'] = (user_timestamps - course_start).dt.days // 7
    weekly_counts = df_user.groupby('week').size()

    if len(weekly_counts) > 1:
        cv = weekly_counts.std() / weekly_counts.mean() if weekly_counts.mean() > 0 else 0
        features['consistent_preparer'] = 1 if cv < 0.5 else 0  # Low variance = consistent

    return features


def calculate_file_preparation(df_user, course_start, course_end):
    """Calculate file access patterns as preparation indicator."""
    features = {
        'files_early_pct': 0.0,
        'files_total': 0,
        'files_per_week': 0.0,
        'files_diversity': 0,
    }

    file_df = df_user[df_user['resource_type'] == 'files']

    if len(file_df) == 0:
        return features

    features['files_total'] = len(file_df)
    features['files_diversity'] = file_df['resource_id'].nunique()

    # Files accessed in first half of course
    if course_start < course_end:
        file_timestamps = pd.to_datetime(file_df['created_at'])
        course_duration = (course_end - course_start).days
        midpoint = course_start + timedelta(days=course_duration / 2)

        early_files = (file_timestamps <= midpoint).sum()
        features['files_early_pct'] = early_files / len(file_df)

        # Files per week
        weeks = max(course_duration / 7, 1)
        features['files_per_week'] = len(file_df) / weeks

    return features


def main():
    print("=" * 60)
    print("Calculating Pre-Assessment Activity Features")
    print("=" * 60)
    print()

    # Load data
    df, assignments_df, enrollments = load_data()

    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'

    # Calculate features per course
    all_features = []

    for course_id in COURSES:
        print(f"\nProcessing course {course_id}...")

        # Get course data
        df_course = df[df['course_id'] == course_id].copy()

        if len(df_course) == 0:
            print(f"  No page views, skipping")
            continue

        # Get course date range
        course_start, course_end = get_course_date_range(df_course)
        print(f"  Date range: {course_start.date()} to {course_end.date()}")

        # Get enrolled users
        course_enrollments = enrollments[enrollments['course_id'] == course_id]
        enrolled_users = set(course_enrollments['user_id'].unique())

        # Also include users with page views (in case enrollment data is incomplete)
        active_users = set(df_course[user_col].unique())
        all_users = enrolled_users | active_users

        print(f"  Users: {len(all_users)}")

        # Pre-calculate QAT percentiles for the course
        qat_percentiles = calculate_qat_percentiles(df_course, all_users, course_start, user_col)

        # Count assignments with due dates for this course
        course_assignments = assignments_df[assignments_df['course_id'] == course_id]
        print(f"  Assignments with due dates: {len(course_assignments)}")

        # Calculate features per user
        for user_id in all_users:
            df_user = df_course[df_course[user_col] == user_id]

            user_features = {
                'user_id': user_id,
                'course_id': course_id,
                'total_page_views': len(df_user),
            }

            # 1. Deadline-relative features
            deadline_features = calculate_deadline_features(df_user, assignments_df, course_id)
            user_features.update(deadline_features)

            # 2. QAT features
            qat_features = calculate_qat_features(df_course, user_id, course_start, user_col)
            user_features.update(qat_features)

            # Add percentile ranking
            user_features.update(qat_percentiles.get(user_id, {'quiz_access_pct': 0.0, 'early_quiz_accessor': 0}))

            # 3. Assessment patterns
            assessment_features = calculate_assessment_patterns(df_user)
            user_features.update(assessment_features)

            # 4. Temporal patterns
            temporal_features = calculate_temporal_patterns(df_user, course_start, course_end)
            user_features.update(temporal_features)

            # 5. File preparation patterns
            file_features = calculate_file_preparation(df_user, course_start, course_end)
            user_features.update(file_features)

            all_features.append(user_features)

    # Create DataFrame
    features_df = pd.DataFrame(all_features)
    print(f"\n{'=' * 60}")
    print(f"Generated features for {len(features_df)} user-course pairs")
    print(f"Total features: {len(features_df.columns)}")

    # Feature summary
    print("\n=== Feature Summary ===")

    key_features = [
        'activity_72h_before', 'preparation_intensity',
        'first_quiz_access_days', 'quiz_access_pct', 'early_quiz_accessor',
        'assessment_diversity', 'early_half_activity_pct', 'late_surge_ratio',
        'files_early_pct', 'consistent_preparer'
    ]

    for feat in key_features:
        if feat in features_df.columns:
            mean_val = features_df[feat].mean()
            std_val = features_df[feat].std()
            non_zero = (features_df[feat] != 0).sum()
            print(f"  {feat:30s}: mean={mean_val:8.3f}, std={std_val:8.3f}, non-zero={non_zero}")

    # Deadline features available for some courses
    print("\n=== Deadline Features by Course ===")
    for course_id in COURSES:
        course_df = features_df[features_df['course_id'] == course_id]
        if len(course_df) > 0:
            n_deadlines = course_df['n_deadlines'].iloc[0] if 'n_deadlines' in course_df.columns else 0
            avg_72h = course_df['activity_72h_before'].mean()
            print(f"  Course {course_id}: {n_deadlines} deadlines, avg 72h activity: {avg_72h:.1f}")

    # Save
    print(f"\nSaving to {OUTPUT_FILE}...")
    features_df.to_parquet(OUTPUT_FILE, index=False)
    print("Done!")

    # Preview
    print("\nSample features (first 5 rows):")
    display_cols = ['user_id', 'course_id', 'activity_72h_before', 'quiz_access_pct',
                    'assessment_diversity', 'early_half_activity_pct']
    display_cols = [c for c in display_cols if c in features_df.columns]
    print(features_df[display_cols].head().to_string())

    return features_df


if __name__ == '__main__':
    main()
