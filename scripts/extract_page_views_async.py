#!/usr/bin/env python3
"""
Extract page views using Canvas async query API.
This endpoint allows access to 1 year of historical data.

Critical: Dates MUST be first day of month (YYYY-MM-01)
"""

import os
import io
import time
import json
import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}
JSON_HEADERS = {**HEADERS, 'Content-Type': 'application/json'}

# Rate limit handling
def check_rate_limit(response, min_remaining=50):
    """Check rate limit headers and sleep if needed."""
    remaining = response.headers.get('X-Rate-Limit-Remaining', '100')
    try:
        remaining = float(remaining)
        if remaining < min_remaining:
            wait_time = max(5, (min_remaining - remaining) * 0.5)
            print(f'\n  Rate limit low ({remaining}), waiting {wait_time:.0f}s...', end='', flush=True)
            time.sleep(wait_time)
            print(' done')
    except (ValueError, TypeError):
        pass

def rate_limited_request(method, url, **kwargs):
    """Make a request with rate limit handling and retries."""
    max_retries = 5
    for attempt in range(max_retries):
        if method == 'get':
            r = requests.get(url, **kwargs)
        elif method == 'post':
            r = requests.post(url, **kwargs)
        else:
            raise ValueError(f'Unknown method: {method}')

        if r.status_code == 429:
            wait_time = 60 * (attempt + 1)  # Exponential backoff
            print(f'\n  Rate limited (429), waiting {wait_time}s...', end='', flush=True)
            time.sleep(wait_time)
            print(' retrying')
            continue

        check_rate_limit(r)
        return r

    return r  # Return last response even if all retries failed

# Output directory
OUTPUT_DIR = '/home/paul/projects/uautonoma/data/page_views'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Courses to extract (from plan)
COURSES = {
    79875: {'name': 'TALLER DE COMP DIGITALES-P01', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
    79913: {'name': 'FUND. DE BUSINESS ANALYTICS-P01', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
    84936: {'name': 'FUNDAMENTOS DE MICROECONOMÍA-P03', 'term': 352, 'start': '2025-07-01', 'end': '2025-12-01'},
    84941: {'name': 'FUNDAMENTOS DE MICROECONOMÍA-P01', 'term': 352, 'start': '2025-07-01', 'end': '2025-12-01'},
    84944: {'name': 'FUNDAMENTOS DE MACROECONOMÍA-P03', 'term': 352, 'start': '2025-07-01', 'end': '2025-12-01'},
    86020: {'name': 'TALL DE COMPETENCIAS DIGITALES-P02', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
    86676: {'name': 'FUND DE BUSINESS ANALYTICS-P01', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
    88381: {'name': 'MATEMÁTICAS PARA LOS NEGOCIOS-P01', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
    89099: {'name': 'TALLER DE COMP DIGITALES-P01', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
    89390: {'name': 'GESTIÓN DEL TALENTO-P01', 'term': 336, 'start': '2025-03-01', 'end': '2026-01-01'},
}


def get_course_students(course_id):
    """Get all students enrolled in a course with rate limit handling."""
    students = []
    url = f'{API_URL}/api/v1/courses/{course_id}/enrollments'
    params = {'type[]': 'StudentEnrollment', 'per_page': 100, 'state[]': ['active', 'completed']}

    while url:
        r = rate_limited_request('get', url, headers=HEADERS, params=params)
        if r.status_code != 200:
            print(f'Error getting enrollments for course {course_id}: {r.status_code}')
            break

        data = r.json()
        for enrollment in data:
            students.append({
                'user_id': enrollment['user_id'],
                'course_id': course_id,
                'enrollment_state': enrollment.get('enrollment_state'),
                'current_score': enrollment.get('grades', {}).get('current_score'),
                'final_score': enrollment.get('grades', {}).get('final_score'),
            })

        # Check for next page
        params = None  # Only use params on first request
        link_header = r.headers.get('Link', '')
        url = None
        for part in link_header.split(','):
            if 'rel="next"' in part:
                import re
                match = re.search(r'<([^>]+)>', part)
                if match:
                    url = match.group(1)

        time.sleep(0.2)  # Small delay between pagination

    return students


def start_page_views_query(user_id, start_date, end_date):
    """Start an async page views query. Dates must be first of month."""
    r = rate_limited_request(
        'post',
        f'{API_URL}/api/v1/users/{user_id}/page_views/query',
        headers=JSON_HEADERS,
        json={
            'start_date': start_date,
            'end_date': end_date,
            'results_format': 'csv'
        }
    )

    if r.status_code == 201:
        return r.json().get('poll_url')
    else:
        return None


def poll_query_status(poll_url, max_attempts=120, delay=1):
    """Poll until query is finished or failed."""
    for attempt in range(max_attempts):
        try:
            r = rate_limited_request('get', f'{API_URL}{poll_url}', headers=HEADERS)
            if r.status_code != 200:
                return None, f'Poll error: {r.status_code}'

            data = r.json()
            status = data.get('status')

            if status == 'finished':
                return poll_url + '/results', None
            elif status == 'failed':
                return None, data.get('error', 'Query failed')

            time.sleep(delay)
        except Exception as e:
            time.sleep(2)
            continue

    return None, 'Timeout waiting for query'


def get_query_results(results_url):
    """Download CSV results."""
    full_url = f'{API_URL}{results_url}' if not results_url.startswith('http') else results_url
    r = rate_limited_request('get', full_url, headers=HEADERS)
    if r.status_code == 200:
        return r.content.decode('utf-8')
    return None


def extract_user_page_views(user_id, start_date, end_date):
    """Extract page views for a user using async API."""
    # Start query
    poll_url = start_page_views_query(user_id, start_date, end_date)
    if not poll_url:
        return None, 'Failed to start query'

    # Poll for results
    results_url, error = poll_query_status(poll_url)
    if error:
        return None, error

    # Get results
    csv_content = get_query_results(results_url)
    if csv_content:
        try:
            df = pd.read_csv(io.StringIO(csv_content))
            return df, None
        except Exception as e:
            return None, str(e)

    return None, 'Failed to get results'


def main():
    print('=' * 60)
    print('Page Views Extraction (Async API)')
    print('=' * 60)
    print()

    # Check for complete student list backup first
    complete_list_file = f'{OUTPUT_DIR}/student_enrollments_complete.csv'
    regular_list_file = f'{OUTPUT_DIR}/student_enrollments.csv'

    if os.path.exists(complete_list_file):
        # Use the complete backup
        print(f'Loading complete student list from backup...')
        students_df = pd.read_csv(complete_list_file)
        all_students = students_df.to_dict('records')
        print(f'Loaded {len(all_students)} enrollments')
    else:
        # Fetch from API
        all_students = []
        print('Fetching students from courses...')

        for course_id, info in COURSES.items():
            print(f'  Course {course_id} ({info["name"]})...', end=' ', flush=True)
            students = get_course_students(course_id)
            print(f'{len(students)} students')
            all_students.extend(students)

        # Save as complete backup if we got a reasonable number
        if len(all_students) >= 350:  # We expect ~373 enrollments
            students_df = pd.DataFrame(all_students)
            students_df.to_csv(complete_list_file, index=False)
            print(f'Saved COMPLETE student list to {complete_list_file}')

    # Deduplicate by user_id (some students may be in multiple courses)
    unique_users = {}
    for s in all_students:
        uid = s['user_id']
        if uid not in unique_users:
            unique_users[uid] = []
        unique_users[uid].append(s['course_id'])

    print()
    print(f'Total enrollments: {len(all_students)}')
    print(f'Unique students: {len(unique_users)}')
    print()

    # Save current student list
    students_df = pd.DataFrame(all_students)
    students_df.to_csv(regular_list_file, index=False)
    print(f'Saved student list to {regular_list_file}')
    print()

    # Create subdirectory for individual user files
    users_dir = f'{OUTPUT_DIR}/users'
    os.makedirs(users_dir, exist_ok=True)

    # Check which users have already been processed (for resume capability)
    processed_users = set()
    for f in os.listdir(users_dir):
        if f.startswith('user_') and f.endswith('.parquet'):
            try:
                uid = int(f.replace('user_', '').replace('.parquet', ''))
                processed_users.add(uid)
            except:
                pass

    # Load existing errors if resuming
    errors_file = f'{OUTPUT_DIR}/extraction_errors.csv'
    if os.path.exists(errors_file):
        existing_errors = pd.read_csv(errors_file).to_dict('records')
        processed_users.update([e['user_id'] for e in existing_errors])
    else:
        existing_errors = []

    users_to_process = [u for u in unique_users.keys() if u not in processed_users]

    print(f'Already processed: {len(processed_users)} users')
    print(f'Remaining to process: {len(users_to_process)} users')
    print()

    # Extract page views for each unique user (with incremental saving)
    print('Extracting page views...')
    errors = list(existing_errors)
    success_count = len([f for f in os.listdir(users_dir) if f.endswith('.parquet')])

    for user_id in tqdm(users_to_process, desc='Extracting'):
        # Use date range that covers all courses (Mar 2025 - Jan 2026)
        df, error = extract_user_page_views(user_id, '2025-03-01', '2026-01-01')

        if df is not None and len(df) > 0:
            df['source_user_id'] = user_id
            # Save immediately to individual file
            user_file = f'{users_dir}/user_{user_id}.parquet'
            df.to_parquet(user_file, index=False)
            success_count += 1
        elif error:
            errors.append({'user_id': user_id, 'error': error})
            # Save errors incrementally too
            pd.DataFrame(errors).to_csv(errors_file, index=False)

        # Rate limiting
        time.sleep(0.5)

    print()
    print(f'Successfully extracted: {success_count} users')
    print(f'Errors: {len(errors)}')

    if errors:
        print('Sample errors:')
        for e in errors[:5]:
            print(f'  User {e["user_id"]}: {e["error"]}')
        pd.DataFrame(errors).to_csv(errors_file, index=False)

    # Combine all individual user files into one
    print()
    print('Combining all user files...')
    all_files = [f'{users_dir}/{f}' for f in os.listdir(users_dir) if f.endswith('.parquet')]

    if all_files:
        all_dfs = []
        for f in tqdm(all_files, desc='Loading'):
            all_dfs.append(pd.read_parquet(f))

        combined_df = pd.concat(all_dfs, ignore_index=True)
        print()
        print(f'Total page views: {len(combined_df)}')
        print(f'Date range: {combined_df["created_at"].min()} to {combined_df["created_at"].max()}')

        # Save combined data
        output_file = f'{OUTPUT_DIR}/all_page_views.parquet'
        combined_df.to_parquet(output_file, index=False)
        print(f'Saved to {output_file}')

    print()
    print('Done!')


if __name__ == '__main__':
    main()
