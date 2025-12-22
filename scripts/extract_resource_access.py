#!/usr/bin/env python3
"""
Extract Resource-Level Access Data from Canvas API

Implements Oviedo-style metrics:
- ModuleCompletionPct: % of modules completed by each student
- ModuleCompletionTiming: When each student completed each module
- EarlyAccessRank: Rank students by how early they access resources (1=first)
- ActivityTimingPattern: Hourly distribution of activity

Based on Canvas API research:
- Modules with student_id returns state + completed_at
- analytics/users/:id/activity returns hourly page_views + participations
"""

import requests
import json
import sys
import os
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np

sys.path.insert(0, '/home/paul/projects/uautonoma/scripts')
from config import API_URL, API_TOKEN, DATA_DIR
from utils.pagination import paginate_canvas

headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Completed courses with good grade data
COMPLETED_COURSES = [84936, 84941]  # FUNDAMENTOS DE MICROECONOMÍA P03, P01


def get_course_modules(course_id):
    """Get all modules for a course."""
    response = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/modules',
        headers=headers,
        params={'include[]': ['items'], 'per_page': 100},
        timeout=30
    )
    if response.status_code == 200:
        return response.json()
    return []


def get_student_module_progress(course_id, student_id):
    """Get module completion state for a specific student."""
    response = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/modules',
        headers=headers,
        params={'student_id': student_id, 'per_page': 100},
        timeout=30
    )
    if response.status_code == 200:
        return response.json()
    return []


def get_student_activity(course_id, student_id):
    """Get hourly activity breakdown for a student."""
    response = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/users/{student_id}/activity',
        headers=headers,
        timeout=30
    )
    if response.status_code == 200:
        return response.json()
    return {}


def get_enrollments(course_id):
    """Get all student enrollments for a course."""
    return paginate_canvas(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers,
        params={'type[]': 'StudentEnrollment', 'include[]': ['grades', 'total_scores']}
    )


def extract_module_completion_data(course_id, enrollments, modules):
    """
    Extract module completion data for all students.

    Returns DataFrame with:
    - user_id, module_id, module_name
    - state (locked/unlocked/started/completed)
    - completed_at (timestamp)
    - completion_rank (1=first, N=last among completers)
    """
    all_data = []
    module_ids = [m['id'] for m in modules]
    module_names = {m['id']: m['name'] for m in modules}

    print(f"  Extracting module progress for {len(enrollments)} students, {len(modules)} modules...")

    for i, enrollment in enumerate(enrollments):
        user_id = enrollment['user_id']
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(enrollments)} students")

        student_modules = get_student_module_progress(course_id, user_id)

        for module in student_modules:
            module_id = module.get('id')
            if module_id not in module_ids:
                continue

            all_data.append({
                'course_id': course_id,
                'user_id': user_id,
                'module_id': module_id,
                'module_name': module_names.get(module_id, 'Unknown'),
                'state': module.get('state', 'unknown'),
                'completed_at': module.get('completed_at'),
                'items_count': module.get('items_count', 0)
            })

    df = pd.DataFrame(all_data)

    if len(df) == 0:
        return df

    # Calculate completion rank per module (who completed first)
    df['completed_at_dt'] = pd.to_datetime(df['completed_at'], utc=True, errors='coerce')

    # Rank within each module (1 = first completer)
    df['completion_rank'] = df.groupby('module_id')['completed_at_dt'].rank(method='min')

    # Calculate total students who completed each module
    completed_per_module = df[df['state'] == 'completed'].groupby('module_id').size().to_dict()
    df['completers_in_module'] = df['module_id'].map(completed_per_module).fillna(0).astype(int)

    # Normalize rank: 0 = first (early), 1 = last (late)
    df['completion_rank_normalized'] = df.apply(
        lambda row: (row['completion_rank'] - 1) / (row['completers_in_module'] - 1)
        if row['completers_in_module'] > 1 else 0.5,
        axis=1
    )

    return df


def extract_activity_timing_data(course_id, enrollments):
    """
    Extract hourly activity patterns for all students.

    Returns DataFrame with:
    - user_id
    - first_activity_at (earliest page view)
    - last_activity_at (latest page view)
    - activity_span_days
    - morning/afternoon/evening/night activity counts
    - participation_count
    """
    all_data = []

    print(f"  Extracting activity timing for {len(enrollments)} students...")

    for i, enrollment in enumerate(enrollments):
        user_id = enrollment['user_id']
        if (i + 1) % 10 == 0:
            print(f"    Progress: {i + 1}/{len(enrollments)} students")

        activity = get_student_activity(course_id, user_id)

        page_views = activity.get('page_views', {})
        participations = activity.get('participations', [])

        if not page_views:
            all_data.append({
                'course_id': course_id,
                'user_id': user_id,
                'total_page_views': 0,
                'total_participations': len(participations),
                'first_activity_at': None,
                'last_activity_at': None,
                'activity_span_days': 0,
                'morning_views': 0,
                'afternoon_views': 0,
                'evening_views': 0,
                'night_views': 0,
                'unique_hours': 0
            })
            continue

        # Parse timestamps
        timestamps = []
        hour_counts = defaultdict(int)
        total_views = 0

        for ts_str, count in page_views.items():
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                timestamps.append(ts)
                hour = ts.hour
                total_views += count

                # Categorize by time of day
                if 6 <= hour < 12:
                    hour_counts['morning'] += count
                elif 12 <= hour < 18:
                    hour_counts['afternoon'] += count
                elif 18 <= hour < 22:
                    hour_counts['evening'] += count
                else:
                    hour_counts['night'] += count
            except:
                pass

        first_activity = min(timestamps) if timestamps else None
        last_activity = max(timestamps) if timestamps else None

        activity_span = 0
        if first_activity and last_activity:
            activity_span = (last_activity - first_activity).days

        all_data.append({
            'course_id': course_id,
            'user_id': user_id,
            'total_page_views': total_views,
            'total_participations': len(participations),
            'first_activity_at': first_activity.isoformat() if first_activity else None,
            'last_activity_at': last_activity.isoformat() if last_activity else None,
            'activity_span_days': activity_span,
            'morning_views': hour_counts['morning'],
            'afternoon_views': hour_counts['afternoon'],
            'evening_views': hour_counts['evening'],
            'night_views': hour_counts['night'],
            'unique_hours': len(page_views)
        })

    return pd.DataFrame(all_data)


def calculate_oviedo_features(module_df, activity_df, enrollments):
    """
    Calculate Oviedo-style features per student.

    Features:
    - ModuleCompletionPct: % of modules completed
    - AvgCompletionRank: Average normalized rank (0=early, 1=late)
    - EarlyAccessScore: Inverse of rank (higher = earlier)
    - ActivitySpanDays: How many days the student was active
    - TimeOfDayPreference: Dominant time of day for activity
    """
    # Get unique students
    user_ids = set(e['user_id'] for e in enrollments)

    features = []

    for user_id in user_ids:
        # Module completion features
        user_modules = module_df[module_df['user_id'] == user_id]
        total_modules = len(user_modules)
        completed_modules = len(user_modules[user_modules['state'] == 'completed'])

        module_completion_pct = completed_modules / total_modules if total_modules > 0 else 0

        # Average completion rank (only for completed modules)
        completed_only = user_modules[user_modules['state'] == 'completed']
        avg_rank = completed_only['completion_rank_normalized'].mean() if len(completed_only) > 0 else 0.5

        # Activity features
        user_activity = activity_df[activity_df['user_id'] == user_id]
        if len(user_activity) > 0:
            row = user_activity.iloc[0]
            activity_span = row['activity_span_days']
            total_views = row['total_page_views']

            # Determine dominant time of day
            time_counts = {
                'morning': row['morning_views'],
                'afternoon': row['afternoon_views'],
                'evening': row['evening_views'],
                'night': row['night_views']
            }
            dominant_time = max(time_counts, key=time_counts.get) if total_views > 0 else 'none'
        else:
            activity_span = 0
            total_views = 0
            dominant_time = 'none'

        # Get grade from enrollments
        enrollment = next((e for e in enrollments if e['user_id'] == user_id), {})
        grades = enrollment.get('grades', {})
        final_score = grades.get('final_score')

        features.append({
            'user_id': user_id,
            'module_completion_pct': module_completion_pct,
            'avg_completion_rank': avg_rank,
            'early_access_score': 1 - avg_rank,  # Higher = earlier
            'activity_span_days': activity_span,
            'total_page_views': total_views,
            'dominant_time_of_day': dominant_time,
            'final_score': final_score
        })

    return pd.DataFrame(features)


def main():
    print("=" * 60)
    print("RESOURCE ACCESS DATA EXTRACTION")
    print("Implementing Oviedo-style metrics from Canvas API")
    print("=" * 60)

    # Ensure output directory exists
    output_dir = os.path.join(DATA_DIR, 'resource_access')
    os.makedirs(output_dir, exist_ok=True)

    all_module_data = []
    all_activity_data = []
    all_features = []

    for course_id in COMPLETED_COURSES:
        print(f"\n{'=' * 60}")
        print(f"Processing Course: {course_id}")
        print("=" * 60)

        # Get enrollments
        print("\n1. Getting enrollments...")
        enrollments = get_enrollments(course_id)
        print(f"   Found {len(enrollments)} students")

        # Get modules
        print("\n2. Getting course modules...")
        modules = get_course_modules(course_id)
        print(f"   Found {len(modules)} modules")

        if modules:
            for m in modules:
                print(f"     - {m['name']} ({m.get('items_count', 0)} items)")

        # Extract module completion data
        print("\n3. Extracting module completion data...")
        module_df = extract_module_completion_data(course_id, enrollments, modules)
        print(f"   Extracted {len(module_df)} module-student records")
        all_module_data.append(module_df)

        # Module completion summary
        if len(module_df) > 0:
            completion_stats = module_df[module_df['state'] == 'completed'].groupby('module_name').agg({
                'user_id': 'count',
                'completion_rank': 'max'
            }).rename(columns={'user_id': 'completers', 'completion_rank': 'max_rank'})
            print("\n   Module completion summary:")
            for line in completion_stats.to_string().split('\n'):
                print(f"      {line}")

        # Extract activity timing data
        print("\n4. Extracting activity timing data...")
        activity_df = extract_activity_timing_data(course_id, enrollments)
        print(f"   Extracted {len(activity_df)} activity records")
        all_activity_data.append(activity_df)

        # Calculate Oviedo features
        print("\n5. Calculating Oviedo-style features...")
        features_df = calculate_oviedo_features(module_df, activity_df, enrollments)
        features_df['course_id'] = course_id
        print(f"   Calculated features for {len(features_df)} students")
        all_features.append(features_df)

    # Combine all data
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    # Save module data
    combined_modules = pd.concat(all_module_data, ignore_index=True)
    module_path = os.path.join(output_dir, 'module_completion_data.csv')
    combined_modules.to_csv(module_path, index=False)
    print(f"Saved: {module_path} ({len(combined_modules)} records)")

    # Save activity data
    combined_activity = pd.concat(all_activity_data, ignore_index=True)
    activity_path = os.path.join(output_dir, 'activity_timing_data.csv')
    combined_activity.to_csv(activity_path, index=False)
    print(f"Saved: {activity_path} ({len(combined_activity)} records)")

    # Save Oviedo features
    combined_features = pd.concat(all_features, ignore_index=True)
    features_path = os.path.join(output_dir, 'oviedo_features.csv')
    combined_features.to_csv(features_path, index=False)
    print(f"Saved: {features_path} ({len(combined_features)} records)")

    # Print feature correlations with grade
    print("\n" + "=" * 60)
    print("OVIEDO FEATURE CORRELATIONS WITH GRADE")
    print("=" * 60)

    numeric_cols = ['module_completion_pct', 'early_access_score', 'activity_span_days', 'total_page_views']

    # Filter to students with grades
    with_grades = combined_features.dropna(subset=['final_score'])
    print(f"\nStudents with grades: {len(with_grades)}")

    if len(with_grades) > 5:
        print("\nCorrelations with final_score:")
        for col in numeric_cols:
            if col in with_grades.columns:
                corr = with_grades[col].corr(with_grades['final_score'])
                print(f"  {col}: r = {corr:.3f}")

        # Quick regression check
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import cross_val_score

        X = with_grades[numeric_cols].fillna(0)
        y = with_grades['final_score']

        lr = LinearRegression()
        scores = cross_val_score(lr, X, y, cv=min(5, len(with_grades)), scoring='r2')
        print(f"\nLinear Regression R² (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
