#!/usr/bin/env python3
"""
FASE 0: Analyze 3 high-potential Pregrado courses for validation.

Courses:
- 81837: ÁLGEBRA-P01 (Ing. Civil Industrial) - 65% failure rate, StdDev 26.7
- 82198: NEUROCIENCIAS-P01 (Medicina) - highest engagement
- 83844: SALUD FAM. Y COMUNITARIA-P01 (Kinesiología) - 80% failure rate
"""

import requests
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Target courses
COURSES = [
    {'id': 81837, 'name': 'ÁLGEBRA-P01', 'career': 'Ing. Civil Industrial'},
    {'id': 82198, 'name': 'NEUROCIENCIAS-P01', 'career': 'Medicina'},
    {'id': 83844, 'name': 'SALUD FAM. Y COMUNITARIA-P01', 'career': 'Kinesiología'},
]

# Pure activity features (no data leakage)
PURE_ACTIVITY_FEATURES = [
    'page_views', 'page_views_level', 'total_activity_time',
    'morning_activity', 'afternoon_activity', 'evening_activity', 'night_activity',
    'time_concentration', 'activity_span_days', 'unique_active_hours',
    'avg_gap_hours', 'gap_std_hours'
]


def get_enrollments_with_grades(course_id):
    """Get student enrollments with grades."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 100, 'include[]': ['grades', 'total_activity_time']}
    )
    if r.status_code != 200:
        print(f"  Error getting enrollments: {r.status_code}")
        return []
    return r.json()


def get_student_summaries(course_id):
    """Get student activity summaries."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries',
        headers=headers,
        params={'per_page': 100}
    )
    if r.status_code != 200:
        print(f"  Error getting summaries: {r.status_code}")
        return []
    return r.json()


def get_user_activity(course_id, user_id):
    """Get hourly activity breakdown for a user."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/users/{user_id}/activity',
        headers=headers
    )
    if r.status_code != 200:
        return None
    return r.json()


def calculate_time_features(activity_data):
    """Calculate time-of-day features from activity data."""
    if not activity_data or 'page_views' not in activity_data:
        return {}

    page_views = activity_data.get('page_views', {})

    # Aggregate by hour of day
    hourly = {}
    for timestamp, count in page_views.items():
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            hour = dt.hour
            hourly[hour] = hourly.get(hour, 0) + count
        except:
            continue

    if not hourly:
        return {}

    total = sum(hourly.values())
    if total == 0:
        return {}

    # Time-of-day distribution
    morning = sum(hourly.get(h, 0) for h in range(6, 12))
    afternoon = sum(hourly.get(h, 0) for h in range(12, 18))
    evening = sum(hourly.get(h, 0) for h in range(18, 24))
    night = sum(hourly.get(h, 0) for h in range(0, 6))

    # Unique active hours
    unique_hours = len([h for h, c in hourly.items() if c > 0])

    # Time concentration (Gini-like)
    props = [hourly.get(h, 0) / total for h in range(24)]
    time_concentration = sum(p * p for p in props if p > 0)

    # Activity timestamps for gap analysis
    timestamps = []
    for ts_str, count in page_views.items():
        if count > 0:
            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                timestamps.append(dt)
            except:
                continue

    timestamps.sort()

    # Gap features
    avg_gap = 0
    gap_std = 0
    activity_span = 0

    if len(timestamps) >= 2:
        gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() / 3600
                for i in range(len(timestamps) - 1)]
        gaps = [g for g in gaps if g > 0.5]  # Filter very short gaps

        if gaps:
            avg_gap = np.mean(gaps)
            gap_std = np.std(gaps) if len(gaps) > 1 else 0

        activity_span = (timestamps[-1] - timestamps[0]).days

    return {
        'morning_activity': morning / total if total > 0 else 0,
        'afternoon_activity': afternoon / total if total > 0 else 0,
        'evening_activity': evening / total if total > 0 else 0,
        'night_activity': night / total if total > 0 else 0,
        'unique_active_hours': unique_hours,
        'time_concentration': time_concentration,
        'avg_gap_hours': avg_gap,
        'gap_std_hours': gap_std,
        'activity_span_days': activity_span,
    }


def extract_course_data(course):
    """Extract all data for a course."""
    course_id = course['id']
    print(f"\nExtracting data for {course['name']} ({course_id})...")

    # Get enrollments
    enrollments = get_enrollments_with_grades(course_id)
    print(f"  Enrollments: {len(enrollments)}")

    # Get activity summaries
    summaries = get_student_summaries(course_id)
    print(f"  Activity summaries: {len(summaries)}")

    # Build student features
    students = []

    for enrollment in enrollments:
        user_id = enrollment['user_id']
        grades = enrollment.get('grades', {})
        final_score = grades.get('final_score')

        # Skip students without valid grades
        if final_score is None or final_score <= 0 or final_score > 100:
            continue

        # Find matching summary
        summary = next((s for s in summaries if s['id'] == user_id), None)

        student = {
            'course_id': course_id,
            'course_name': course['name'],
            'career': course['career'],
            'user_id': user_id,
            'final_score': final_score,
            'current_score': grades.get('current_score'),
            'failed': 1 if final_score < 57 else 0,
            'total_activity_time': enrollment.get('total_activity_time', 0) or 0,
        }

        if summary:
            student['page_views'] = summary.get('page_views', 0)
            student['page_views_level'] = summary.get('page_views_level', 0)
            student['participations'] = summary.get('participations', 0)
            student['participations_level'] = summary.get('participations_level', 0)

            tardiness = summary.get('tardiness_breakdown', {})
            student['on_time'] = tardiness.get('on_time', 0)
            student['late'] = tardiness.get('late', 0)
            student['missing'] = tardiness.get('missing', 0)

        students.append(student)

    print(f"  Students with valid grades: {len(students)}")

    # Get detailed activity for sample (to calculate time features)
    print(f"  Fetching hourly activity...")
    for i, student in enumerate(students):
        activity = get_user_activity(course_id, student['user_id'])
        if activity:
            time_features = calculate_time_features(activity)
            student.update(time_features)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(students)} students")
        time.sleep(0.3)  # Rate limiting

    return students


def calculate_correlations(df, features):
    """Calculate correlations between features and final_score."""
    correlations = {}

    for feature in features:
        if feature not in df.columns:
            continue

        valid = df[[feature, 'final_score']].dropna()
        if len(valid) < 10:
            continue

        corr = valid[feature].corr(valid['final_score'])
        if not np.isnan(corr):
            correlations[feature] = round(corr, 3)

    return correlations


def main():
    print("=" * 70)
    print("FASE 0: Analyzing Pregrado High-Potential Courses")
    print("=" * 70)

    all_students = []
    course_results = []

    for course in COURSES:
        students = extract_course_data(course)
        all_students.extend(students)

        if len(students) >= 10:
            df = pd.DataFrame(students)

            # Calculate correlations
            correlations = calculate_correlations(df, PURE_ACTIVITY_FEATURES)

            # Course summary
            result = {
                'course_id': course['id'],
                'course_name': course['name'],
                'career': course['career'],
                'n_students': len(students),
                'grade_mean': round(df['final_score'].mean(), 1),
                'grade_std': round(df['final_score'].std(), 1),
                'fail_rate': round(df['failed'].mean(), 3),
                'correlations': correlations
            }
            course_results.append(result)

            print(f"\n  CORRELATIONS for {course['name']}:")
            for feat, corr in sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
                print(f"    {feat}: r = {corr:+.3f}")

    # Save results
    output_dir = 'data/pregrado_validation'
    os.makedirs(output_dir, exist_ok=True)

    # Save student features
    df_all = pd.DataFrame(all_students)
    df_all.to_csv(f'{output_dir}/students_features.csv', index=False)
    print(f"\nSaved {len(all_students)} students to {output_dir}/students_features.csv")

    # Save course results
    with open(f'{output_dir}/course_correlations.json', 'w') as f:
        json.dump(course_results, f, indent=2, ensure_ascii=False)
    print(f"Saved course correlations to {output_dir}/course_correlations.json")

    # Print comparison summary
    print("\n" + "=" * 70)
    print("COMPARISON WITH CONTROL DE GESTIÓN FINDINGS")
    print("=" * 70)

    # Load CdG average correlations
    try:
        with open('data/correlation_analysis/average_correlations.json', 'r') as f:
            cdg_corrs = json.load(f)
    except:
        cdg_corrs = {}

    print("\n| Feature | CdG Avg | ÁLGEBRA | NEUROCIENCIAS | SALUD FAM. |")
    print("|---------|---------|---------|---------------|------------|")

    key_features = ['unique_active_hours', 'total_activity_time', 'avg_gap_hours', 'gap_std_hours']

    for feat in key_features:
        cdg_val = cdg_corrs.get(feat, {}).get('mean', '-')
        if isinstance(cdg_val, float):
            cdg_val = f"{cdg_val:+.2f}"

        vals = []
        for result in course_results:
            corr = result['correlations'].get(feat, '-')
            if isinstance(corr, float):
                vals.append(f"{corr:+.2f}")
            else:
                vals.append('-')

        while len(vals) < 3:
            vals.append('-')

        print(f"| {feat[:20]:<20} | {cdg_val:>7} | {vals[0]:>7} | {vals[1]:>13} | {vals[2]:>10} |")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Check consistency
    consistent_features = []
    for feat in key_features:
        signs_match = True
        cdg_mean = cdg_corrs.get(feat, {}).get('mean', 0)

        for result in course_results:
            corr = result['correlations'].get(feat, 0)
            if corr != 0 and cdg_mean != 0:
                if (corr > 0) != (cdg_mean > 0):
                    signs_match = False

        if signs_match and cdg_mean != 0:
            consistent_features.append(feat)

    if consistent_features:
        print(f"\n✓ Features with CONSISTENT direction across all courses:")
        for feat in consistent_features:
            print(f"  - {feat}")

    print("\nValidation complete. Results saved to data/pregrado_validation/")


if __name__ == '__main__':
    main()
