#!/usr/bin/env python3
"""
Providencia Sub-Account Course Discovery
Finds high-potential courses for early failure prediction in Providencia (Account 176)
"""

import requests
import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Configuration
PROVIDENCIA_ID = 176
EXCLUDE_ACCOUNTS = [719, 718]  # Already analyzed: Ing. en Control de Gestión
TARGET_TERMS = [336, 322]  # 2nd Sem 2025 (current) and 1st Sem 2025 (recent)
MIN_STUDENTS = 20
MAX_COURSES_TO_ANALYZE = 50  # Limit to top 50 by student count

def safe_request(url, params=None, max_retries=3):
    """Make a request with rate limit handling."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            remaining = int(float(r.headers.get('X-Rate-Limit-Remaining', 700)))

            if r.status_code == 403:
                print(f"Rate limited! Waiting 60s... (attempt {attempt + 1})")
                time.sleep(60)
                continue

            if r.status_code == 200:
                # Adaptive delay based on quota
                if remaining < 100:
                    time.sleep(10)
                elif remaining < 300:
                    time.sleep(2)
                elif remaining < 500:
                    time.sleep(1)
                else:
                    time.sleep(0.3)
                return r.json(), remaining
            else:
                print(f"Error {r.status_code}: {r.text[:100]}")
                return None, remaining

        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    return None, 0

def get_sub_accounts(account_id):
    """Get all sub-accounts under an account."""
    all_sub_accounts = []
    url = f'{API_URL}/api/v1/accounts/{account_id}/sub_accounts'
    params = {'per_page': 100, 'recursive': True}

    data, _ = safe_request(url, params)
    if data:
        all_sub_accounts.extend(data)

    return all_sub_accounts

def get_courses_from_account(account_id, term_ids=None, min_students=20):
    """Get courses from an account with minimum student count."""
    all_courses = []

    for term_id in (term_ids or [None]):
        params = {
            'per_page': 100,
            'include[]': ['total_students', 'term'],
            'with_enrollments': True
        }
        if term_id:
            params['enrollment_term_id'] = term_id

        url = f'{API_URL}/api/v1/accounts/{account_id}/courses'

        data, remaining = safe_request(url, params)
        if not data:
            continue

        for course in data:
            if course.get('total_students', 0) >= min_students:
                all_courses.append({
                    'course_id': course['id'],
                    'name': course['name'],
                    'account_id': course.get('account_id', account_id),
                    'students': course.get('total_students', 0),
                    'term_name': course.get('term', {}).get('name', 'Unknown'),
                    'term_id': course.get('enrollment_term_id')
                })

        print(f"  Term {term_id}: {len(data)} courses fetched, {len([c for c in data if c.get('total_students', 0) >= min_students])} with {min_students}+ students")

    return all_courses

def analyze_course_potential(course_id):
    """Analyze a course for analytical potential."""
    result = {
        'course_id': course_id,
        'has_grades': False,
        'n_students_with_grades': 0,
        'grade_variance': 0.0,
        'grade_mean': 0.0,
        'pass_rate': None,
        'n_assignments': 0,
        'n_modules': 0,
        'recommendation': 'SKIP'
    }

    # 1. Get enrollments with grades
    enrollments, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        params={
            'type[]': 'StudentEnrollment',
            'per_page': 100,
            'include[]': 'grades'
        }
    )

    if enrollments:
        grades = [
            e['grades'].get('final_score')
            for e in enrollments
            if e.get('grades', {}).get('final_score') is not None
        ]

        if len(grades) >= 10:
            result['has_grades'] = True
            result['n_students_with_grades'] = len(grades)
            result['grade_variance'] = float(np.std(grades))
            result['grade_mean'] = float(np.mean(grades))
            result['pass_rate'] = sum(1 for g in grades if g >= 57) / len(grades)

    # 2. Count assignments
    assignments, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/assignments',
        params={'per_page': 100}
    )
    if assignments:
        result['n_assignments'] = len(assignments)

    # 3. Count modules
    modules, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/modules',
        params={'per_page': 100}
    )
    if modules:
        result['n_modules'] = len(modules)

    # 4. Determine recommendation
    if result['has_grades'] and result['grade_variance'] > 10:
        if result['n_assignments'] >= 5 and 0.2 <= (result['pass_rate'] or 0) <= 0.8:
            result['recommendation'] = 'HIGH POTENTIAL'
        elif result['n_assignments'] >= 3:
            result['recommendation'] = 'MEDIUM POTENTIAL'
        else:
            result['recommendation'] = 'LOW - Few assignments'
    elif result['has_grades']:
        result['recommendation'] = 'LOW - Low grade variance'
    else:
        result['recommendation'] = 'SKIP - No grades'

    return result

def main():
    print("=" * 60)
    print("PROVIDENCIA COURSE DISCOVERY")
    print("=" * 60)

    # Test connection
    r = requests.get(f'{API_URL}/api/v1/users/self', headers=headers)
    if r.status_code == 200:
        user = r.json()
        print(f"Connected as: {user.get('name', 'Unknown')}")
        print(f"Rate Limit: {r.headers.get('X-Rate-Limit-Remaining', 'N/A')}")
    else:
        print(f"Connection failed: {r.status_code}")
        return

    # Step 1: Get Providencia sub-accounts
    print(f"\n--- Step 1: Discovering sub-accounts under Providencia ({PROVIDENCIA_ID}) ---")
    sub_accounts = get_sub_accounts(PROVIDENCIA_ID)
    sub_accounts = [a for a in sub_accounts if a['id'] not in EXCLUDE_ACCOUNTS]
    print(f"Found {len(sub_accounts)} sub-accounts (excluding already analyzed)")

    for acc in sub_accounts[:10]:
        print(f"  {acc['id']}: {acc['name']}")
    if len(sub_accounts) > 10:
        print(f"  ... and {len(sub_accounts) - 10} more")

    # Step 2: Get courses from each sub-account
    print(f"\n--- Step 2: Fetching courses with {MIN_STUDENTS}+ students ---")
    all_courses = []

    for acc in sub_accounts:
        acc_id = acc['id']
        acc_name = acc['name']
        print(f"\nScanning: {acc_name} (ID: {acc_id})")

        courses = get_courses_from_account(acc_id, term_ids=TARGET_TERMS, min_students=MIN_STUDENTS)
        print(f"  Total: {len(courses)} courses with {MIN_STUDENTS}+ students")

        all_courses.extend(courses)

    print(f"\n{'=' * 40}")
    print(f"Total candidate courses: {len(all_courses)}")

    if not all_courses:
        print("No courses found. Try lowering MIN_STUDENTS threshold.")
        return

    # Step 3: Sort by student count and limit
    courses_df = pd.DataFrame(all_courses)
    courses_df = courses_df.sort_values('students', ascending=False)
    courses_df = courses_df.head(MAX_COURSES_TO_ANALYZE)

    print(f"Analyzing top {len(courses_df)} courses by student count...")

    # Step 4: Analyze each course
    print(f"\n--- Step 3: Analyzing course potential ---")
    analysis_results = []

    for i, (idx, row) in enumerate(courses_df.iterrows()):
        course_id = row['course_id']
        print(f"[{i+1}/{len(courses_df)}] Analyzing {course_id}: {row['name'][:40]}...")

        analysis = analyze_course_potential(course_id)
        analysis['name'] = row['name']
        analysis['account_id'] = row['account_id']
        analysis['total_students'] = row['students']
        analysis['term_id'] = row['term_id']
        analysis['term_name'] = row['term_name']

        analysis_results.append(analysis)

        if 'HIGH' in analysis['recommendation']:
            print(f"  *** HIGH POTENTIAL: Variance={analysis['grade_variance']:.1f}, Pass Rate={analysis['pass_rate']:.0%}")
        elif 'MEDIUM' in analysis['recommendation']:
            print(f"  ** MEDIUM: Variance={analysis['grade_variance']:.1f}, Pass Rate={analysis['pass_rate']:.0%}")

    # Step 5: Create results DataFrame
    results_df = pd.DataFrame(analysis_results)
    results_df['rec_order'] = results_df['recommendation'].map({
        'HIGH POTENTIAL': 1,
        'MEDIUM POTENTIAL': 2,
        'LOW - Few assignments': 3,
        'LOW - Low grade variance': 4,
        'SKIP - No grades': 5
    })
    results_df = results_df.sort_values(['rec_order', 'grade_variance'], ascending=[True, False])

    # Step 6: Save results
    output_path = 'exploratory/data/providencia_discovery_results.csv'
    output_cols = [
        'course_id', 'name', 'account_id', 'term_id', 'term_name',
        'total_students', 'n_students_with_grades',
        'grade_mean', 'grade_variance', 'pass_rate',
        'n_assignments', 'n_modules', 'recommendation'
    ]
    results_df[output_cols].to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("DISCOVERY SUMMARY")
    print("=" * 60)

    print(f"\nResults saved to: {output_path}")
    print(f"Total courses analyzed: {len(results_df)}")

    print("\nRecommendation breakdown:")
    print(results_df['recommendation'].value_counts().to_string())

    high_potential = results_df[results_df['recommendation'].str.contains('HIGH|MEDIUM')]

    if len(high_potential) > 0:
        print("\n" + "-" * 60)
        print("HIGH/MEDIUM POTENTIAL COURSES:")
        print("-" * 60)

        for idx, row in high_potential.iterrows():
            print(f"\n{row['recommendation']}")
            print(f"  Course ID: {row['course_id']}")
            print(f"  Name: {row['name']}")
            print(f"  Students: {row['total_students']} (with grades: {row['n_students_with_grades']})")
            print(f"  Grade Variance: {row['grade_variance']:.1f}")
            print(f"  Pass Rate: {row['pass_rate']:.1%}" if row['pass_rate'] else "  Pass Rate: N/A")
            print(f"  Assignments: {row['n_assignments']}, Modules: {row['n_modules']}")
            print(f"  Term: {row['term_name']}")
    else:
        print("\nNo HIGH or MEDIUM potential courses found.")

if __name__ == '__main__':
    main()
