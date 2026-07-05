#!/usr/bin/env python3
"""
Extract page views for students in the analyzed courses.
Uses Canvas API with adaptive rate limiting.

Output: data/page_views/course_{id}_page_views.parquet
"""

import os
import re
import sys
import json
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# Load environment variables
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

# Course configurations with term dates
COURSES = {
    # Term 336: Segundo Semestre 2025 - Periodo Completo
    79875: {'name': 'TALLER DE COMP DIGITALES-P01', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    79913: {'name': 'FUND. DE BUSINESS ANALYTICS-P01', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    86020: {'name': 'TALL DE COMPETENCIAS DIGITALES-P02', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    86676: {'name': 'FUND DE BUSINESS ANALYTICS-P01', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    88381: {'name': 'MATEMATICAS PARA LOS NEGOCIOS-P01', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    89099: {'name': 'TALLER DE COMP DIGITALES-P01', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    89390: {'name': 'GESTION DEL TALENTO-P01', 'start': '2025-03-04T00:00:00Z', 'end': '2025-12-31T23:59:59Z'},
    # Term 352: Segundo Semestre 2025 - Bimestral
    84936: {'name': 'FUNDAMENTOS DE MICROECONOMIA-P03', 'start': '2025-07-28T00:00:00Z', 'end': '2025-12-05T23:59:59Z'},
    84941: {'name': 'FUNDAMENTOS DE MICROECONOMIA-P01', 'start': '2025-07-28T00:00:00Z', 'end': '2025-12-05T23:59:59Z'},
    84944: {'name': 'FUNDAMENTOS DE MACROECONOMIA-P03', 'start': '2025-07-28T00:00:00Z', 'end': '2025-12-05T23:59:59Z'},
}

# Rate limiting configuration
MAX_RETRIES = 3
RETRY_DELAY = 2

def calculate_delay(remaining_quota):
    """Calculate adaptive delay based on remaining API quota."""
    if remaining_quota < 10:
        return 30
    elif remaining_quota < 50:
        return 10
    elif remaining_quota < 100:
        return 5
    elif remaining_quota < 200:
        return 2
    elif remaining_quota < 300:
        return 1
    else:
        return 0

def extract_course_id(url):
    """Extract course_id from Canvas URL."""
    if not url:
        return -1
    match = re.search(r"/courses/(\d+)", url)
    return int(match.group(1)) if match else -1

def fetch_page_views(user_id, start_time, end_time, target_course_id):
    """
    Fetch all page views for a user within a time range,
    filtered to a specific course.
    """
    all_views = []
    uri = f"{API_URL}/api/v1/users/{user_id}/page_views"
    params = {
        'start_time': start_time,
        'end_time': end_time,
        'per_page': 100
    }

    page = 0
    retry_count = 0
    current_delay = RETRY_DELAY

    while uri:
        try:
            response = requests.get(uri, headers=HEADERS, params=params if page == 0 else None)

            # Handle rate limiting
            remaining_quota = float(response.headers.get('X-Rate-Limit-Remaining', 700))
            delay = calculate_delay(remaining_quota)
            if delay > 0:
                time.sleep(delay)

            if response.status_code == 200:
                views = response.json()
                if not views:
                    break

                # Filter for target course
                for view in views:
                    url = view.get('url', '')
                    course_id = extract_course_id(url)
                    if course_id == target_course_id:
                        all_views.append({
                            'user_id': user_id,
                            'url': url,
                            'created_at': view.get('created_at'),
                            'updated_at': view.get('updated_at'),
                            'context_type': view.get('context_type'),
                            'controller': view.get('controller'),
                            'action': view.get('action'),
                            'interaction_seconds': view.get('interaction_seconds', 0),
                            'participated': view.get('participated', False),
                            'http_method': view.get('http_method'),
                            'render_time': view.get('render_time'),
                            'user_agent': view.get('user_agent', '')[:100] if view.get('user_agent') else '',
                            'contributed': view.get('contributed', False)
                        })

                # Handle pagination
                if 'next' in response.links:
                    uri = response.links['next']['url']
                    page += 1
                else:
                    break

                retry_count = 0
                current_delay = RETRY_DELAY

            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = int(response.headers.get('Retry-After', 60))
                print(f"  Rate limited. Waiting {wait_time}s...")
                time.sleep(wait_time)

            else:
                print(f"  Error {response.status_code} for user {user_id}")
                if retry_count < MAX_RETRIES:
                    retry_count += 1
                    time.sleep(current_delay)
                    current_delay *= 2
                else:
                    break

        except requests.exceptions.RequestException as e:
            if retry_count < MAX_RETRIES:
                retry_count += 1
                print(f"  Request error: {e}. Retry {retry_count}/{MAX_RETRIES}")
                time.sleep(current_delay)
                current_delay *= 2
            else:
                print(f"  Max retries exceeded for user {user_id}")
                break

    return all_views

def get_students_by_course(features_path):
    """Load students grouped by course from features CSV."""
    df = pd.read_csv(features_path)

    # Filter to students only and relevant courses
    students = df[
        (df['user_role'] == 'student') &
        (df['course_id'].isin(COURSES.keys()))
    ][['course_id', 'user_id']].drop_duplicates()

    return students.groupby('course_id')['user_id'].apply(list).to_dict()

def main():
    """Main extraction function."""
    base_path = Path(__file__).parent.parent
    features_path = base_path / 'data' / 'engagement_dynamics' / 'student_features.csv'
    output_dir = base_path / 'data' / 'page_views'

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get students by course
    print("Loading students from features CSV...")
    students_by_course = get_students_by_course(features_path)

    total_students = sum(len(students) for students in students_by_course.values())
    print(f"Found {total_students} students across {len(students_by_course)} courses")

    # Progress tracking
    extraction_stats = {
        'start_time': datetime.now().isoformat(),
        'courses': {},
        'total_views': 0,
        'total_students': 0
    }

    # Process each course
    for course_id, student_ids in students_by_course.items():
        course_config = COURSES.get(course_id)
        if not course_config:
            print(f"Skipping course {course_id} - not in configuration")
            continue

        print(f"\n{'='*60}")
        print(f"Course: {course_config['name']} (ID: {course_id})")
        print(f"Students: {len(student_ids)}")
        print(f"Period: {course_config['start']} to {course_config['end']}")
        print(f"{'='*60}")

        all_course_views = []

        for user_id in tqdm(student_ids, desc=f"Extracting course {course_id}"):
            views = fetch_page_views(
                user_id=user_id,
                start_time=course_config['start'],
                end_time=course_config['end'],
                target_course_id=course_id
            )
            all_course_views.extend(views)

        # Save course data
        if all_course_views:
            df = pd.DataFrame(all_course_views)
            output_file = output_dir / f"course_{course_id}_page_views.parquet"
            df.to_parquet(output_file, index=False)

            print(f"Saved {len(df)} page views to {output_file.name}")

            extraction_stats['courses'][str(course_id)] = {
                'name': course_config['name'],
                'students': len(student_ids),
                'views': len(df),
                'students_with_views': df['user_id'].nunique()
            }
            extraction_stats['total_views'] += len(df)
            extraction_stats['total_students'] += len(student_ids)
        else:
            print(f"No page views found for course {course_id}")
            extraction_stats['courses'][str(course_id)] = {
                'name': course_config['name'],
                'students': len(student_ids),
                'views': 0,
                'students_with_views': 0
            }

    # Save extraction stats
    extraction_stats['end_time'] = datetime.now().isoformat()
    stats_file = output_dir / 'extraction_stats.json'
    with open(stats_file, 'w') as f:
        json.dump(extraction_stats, f, indent=2)

    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total students processed: {extraction_stats['total_students']}")
    print(f"Total page views extracted: {extraction_stats['total_views']}")
    print(f"Stats saved to: {stats_file}")

    return extraction_stats

if __name__ == '__main__':
    main()
