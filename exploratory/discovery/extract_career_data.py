#!/usr/bin/env python3
"""
Canvas Career Data Extractor
Fetches RAW course data from Canvas API and saves to Parquet files.
NO analysis is performed - only data extraction and storage.

Usage:
    python extract_career_data.py --career-id 248
    python extract_career_data.py --career-id 248 --terms 336 322 --min-students 20
"""

import argparse
import os
import time
import requests
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Default configuration
DEFAULT_TERMS = [336, 322]  # 2nd Sem 2025 (current) and 1st Sem 2025 (recent)
DEFAULT_MIN_STUDENTS = 16
DEFAULT_OUTPUT_DIR = 'exploratory/data/careers'
PASS_THRESHOLD = 57  # Chilean grading: 57% = 4.0 = passing


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


def get_career_name(account_id: int) -> str:
    """Get career name from account ID."""
    data, _ = safe_request(f'{API_URL}/api/v1/accounts/{account_id}')
    if data:
        return data.get('name', f'Account {account_id}')
    return f'Account {account_id}'


def get_all_career_courses(account_id: int, term_ids: List[int], min_students: int) -> List[dict]:
    """Get ALL courses from a career (sub-account) with minimum student count."""
    all_courses = []

    for term_id in term_ids:
        page = 1
        while True:
            params = {
                'per_page': 100,
                'page': page,
                'include[]': ['total_students', 'term'],
                'with_enrollments': True,
                'enrollment_term_id': term_id
            }

            url = f'{API_URL}/api/v1/accounts/{account_id}/courses'
            data, remaining = safe_request(url, params)

            if not data:
                break

            for course in data:
                if course.get('total_students', 0) >= min_students:
                    all_courses.append({
                        'course_id': course['id'],
                        'name': course['name'],
                        'account_id': course.get('account_id', account_id),
                        'total_students': course.get('total_students', 0),
                        'term_name': course.get('term', {}).get('name', 'Unknown'),
                        'term_id': course.get('enrollment_term_id')
                    })

            print(f"  Term {term_id}, Page {page}: {len(data)} courses fetched")

            if len(data) < 100:
                break
            page += 1

    return all_courses


def fetch_course_data(course_id: int) -> Dict:
    """
    Fetch RAW data for a single course from Canvas API.
    Returns raw metrics without any analysis or recommendations.
    """
    data = {
        'course_id': course_id,
        'n_students_with_grades': 0,
        'grade_mean': 0.0,
        'grade_variance': 0.0,
        'pass_rate': None,
        'n_assignments': 0,
        'n_modules': 0,
        'has_activity': False
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

        if grades:
            data['n_students_with_grades'] = len(grades)
            data['grade_variance'] = float(np.std(grades))
            data['grade_mean'] = float(np.mean(grades))
            data['pass_rate'] = sum(1 for g in grades if g >= PASS_THRESHOLD) / len(grades)

    # 2. Count assignments
    assignments, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/assignments',
        params={'per_page': 100}
    )
    if assignments:
        data['n_assignments'] = len(assignments)

    # 3. Count modules
    modules, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/modules',
        params={'per_page': 100}
    )
    if modules:
        data['n_modules'] = len(modules)

    # 4. Check activity data exists
    summaries, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries',
        params={'per_page': 10}
    )
    if summaries and len(summaries) > 0:
        data['has_activity'] = True

    return data


def save_to_parquet(courses: List[Dict], career_id: int, career_name: str, output_dir: str) -> pd.DataFrame:
    """Save raw course data to Parquet files with hierarchical structure."""
    if not courses:
        print(f"\nNo courses to save - skipping Parquet generation")
        return None

    # Create directory structure
    career_dir = f'{output_dir}/career_{career_id}'
    os.makedirs(career_dir, exist_ok=True)

    # Save individual course parquets
    for course in courses:
        course_df = pd.DataFrame([course])
        course_path = f'{career_dir}/course_{course["course_id"]}.parquet'
        course_df.to_parquet(course_path, index=False)

    # Create combined DataFrame
    df = pd.DataFrame(courses)

    # Add metadata columns
    df['career_id'] = career_id
    df['career_name'] = career_name

    # Save combined parquet
    combined_path = f'{output_dir}/career_{career_id}_combined.parquet'
    df.to_parquet(combined_path, index=False)

    print(f"\nParquet files saved:")
    print(f"  - Individual: {career_dir}/ ({len(courses)} files)")
    print(f"  - Combined: {combined_path}")

    return df


def extract_career_data(career_id: int, career_name: Optional[str], terms: List[int],
                         min_students: int, output_dir: str) -> List[Dict]:
    """Main extraction function - fetches and saves raw data only."""
    print("=" * 60)
    print("CANVAS CAREER DATA EXTRACTOR")
    print("=" * 60)

    # Test connection
    r = requests.get(f'{API_URL}/api/v1/users/self', headers=headers)
    if r.status_code == 200:
        user = r.json()
        print(f"Connected as: {user.get('name', 'Unknown')}")
        print(f"Rate Limit: {r.headers.get('X-Rate-Limit-Remaining', 'N/A')}")
    else:
        print(f"Connection failed: {r.status_code}")
        return []

    # Get career name if not provided
    if career_name is None:
        career_name = get_career_name(career_id)

    print(f"\n{'=' * 60}")
    print(f"EXTRACTING: {career_name} (Account {career_id})")
    print(f"{'=' * 60}")
    print(f"Terms: {terms}")
    print(f"Min students: {min_students}")

    # Fetch all courses
    print(f"\n--- Fetching courses ---")
    courses_raw = get_all_career_courses(career_id, terms, min_students)
    print(f"\nFound {len(courses_raw)} candidate courses")

    if not courses_raw:
        print("No courses found matching criteria")
        return []

    # Fetch data for each course
    print(f"\n--- Fetching course data ---")
    courses_data = []
    for i, course in enumerate(courses_raw):
        course_id = course['course_id']
        print(f"[{i+1}/{len(courses_raw)}] Fetching {course_id}: {course['name'][:40]}...")

        # Fetch raw data
        raw_data = fetch_course_data(course_id)

        # Combine course info with fetched data
        course_data = {
            'course_id': course['course_id'],
            'name': course['name'],
            'account_id': course['account_id'],
            'term_id': course['term_id'],
            'term_name': course['term_name'],
            'total_students': course['total_students'],
            'n_students_with_grades': raw_data['n_students_with_grades'],
            'grade_mean': raw_data['grade_mean'],
            'grade_variance': raw_data['grade_variance'],
            'pass_rate': raw_data['pass_rate'],
            'n_assignments': raw_data['n_assignments'],
            'n_modules': raw_data['n_modules'],
            'has_activity': raw_data['has_activity']
        }
        courses_data.append(course_data)

    # Save to Parquet
    save_to_parquet(courses_data, career_id, career_name, output_dir)

    # Print summary (raw counts only, no analysis)
    n_with_grades = sum(1 for c in courses_data if c['n_students_with_grades'] > 0)
    n_with_activity = sum(1 for c in courses_data if c['has_activity'])

    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Total courses extracted: {len(courses_data)}")
    print(f"  With grades: {n_with_grades}")
    print(f"  With activity data: {n_with_activity}")

    return courses_data


def main():
    parser = argparse.ArgumentParser(
        description='Extract RAW course data from Canvas API and save to Parquet files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_career_data.py --career-id 248
  python extract_career_data.py --career-id 248 --terms 336 322
  python extract_career_data.py --career-id 248 --min-students 20
  python extract_career_data.py --career-id 248 --output-dir data/careers
        """
    )

    parser.add_argument('--career-id', type=int, required=True,
                        help='Canvas account ID for the career')
    parser.add_argument('--career-name', type=str, default=None,
                        help='Career name (optional, fetched from API if not provided)')
    parser.add_argument('--terms', type=int, nargs='+', default=DEFAULT_TERMS,
                        help=f'Term IDs to fetch (default: {DEFAULT_TERMS})')
    parser.add_argument('--min-students', type=int, default=DEFAULT_MIN_STUDENTS,
                        help=f'Minimum students per course (default: {DEFAULT_MIN_STUDENTS})')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for Parquet files (default: {DEFAULT_OUTPUT_DIR})')

    args = parser.parse_args()

    extract_career_data(
        career_id=args.career_id,
        career_name=args.career_name,
        terms=args.terms,
        min_students=args.min_students,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()
