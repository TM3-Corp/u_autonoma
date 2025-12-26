#!/usr/bin/env python3
"""
Canvas Course Data Extractor
Fetches ALL available data for a specific course from Canvas API and saves to Parquet files.
Includes: enrollments, assignments, submissions, activity, modules, and more.

Usage:
    python extract_course_data.py --course-id 86676
    python extract_course_data.py --course-id 86676 --start-date 2025-08-01 --end-date 2025-12-31
    python extract_course_data.py --course-id 86676 --include-page-views
"""

import argparse
import os
import re
import time
import requests
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Default configuration
DEFAULT_OUTPUT_DIR = 'exploratory/data/courses'
DEFAULT_START_DATE = '2025-08-01'
DEFAULT_END_DATE = '2025-12-31'
PASS_THRESHOLD = 57  # Chilean grading: 57% = 4.0 = passing


def safe_request(url: str, params: Optional[Dict] = None, max_retries: int = 3) -> tuple:
    """Make a request with rate limit handling and exponential backoff."""
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
                if remaining < 50:
                    time.sleep(30)
                elif remaining < 100:
                    time.sleep(10)
                elif remaining < 200:
                    time.sleep(5)
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


def paginate_request(base_url: str, params: Optional[Dict] = None, desc: str = "") -> List[Dict]:
    """Fetch all pages from a paginated Canvas API endpoint."""
    all_data = []
    url = base_url
    page = 1

    while url:
        data, remaining = safe_request(url, params)

        if data is None:
            break

        if isinstance(data, list):
            all_data.extend(data)
        else:
            # Some endpoints return dict with data inside
            break

        if desc:
            print(f"  {desc}: Page {page}, fetched {len(data)} records (quota: {remaining})")

        if len(data) < params.get('per_page', 100) if params else True:
            break

        page += 1
        # For subsequent pages, params are in the URL
        params = None
        # Check for next page in Link header - simulated by incrementing page
        url = f"{base_url}?page={page}&per_page=100"

    return all_data


def get_course_info(course_id: int) -> Optional[Dict]:
    """Get course metadata."""
    url = f'{API_URL}/api/v1/courses/{course_id}'
    params = {'include[]': ['total_students', 'term', 'teachers']}
    data, _ = safe_request(url, params)
    return data


def get_enrollments(course_id: int) -> List[Dict]:
    """Get all student enrollments with grades."""
    print("  Fetching enrollments...")
    all_enrollments = []
    page = 1

    while True:
        url = f'{API_URL}/api/v1/courses/{course_id}/enrollments'
        params = {
            'type[]': 'StudentEnrollment',
            'per_page': 100,
            'page': page,
            'include[]': ['grades', 'total_scores']
        }

        data, remaining = safe_request(url, params)

        if not data:
            break

        for e in data:
            grades = e.get('grades', {})
            all_enrollments.append({
                'user_id': e.get('user_id'),
                'course_id': course_id,
                'enrollment_state': e.get('enrollment_state'),
                'created_at': e.get('created_at'),
                'updated_at': e.get('updated_at'),
                'current_score': grades.get('current_score'),
                'final_score': grades.get('final_score'),
                'current_grade': grades.get('current_grade'),
                'final_grade': grades.get('final_grade'),
                'unposted_current_score': grades.get('unposted_current_score'),
                'unposted_final_score': grades.get('unposted_final_score')
            })

        print(f"    Page {page}: {len(data)} enrollments (quota: {remaining})")

        if len(data) < 100:
            break
        page += 1

    return all_enrollments


def get_assignments(course_id: int) -> List[Dict]:
    """Get all assignments with metadata."""
    print("  Fetching assignments...")
    all_assignments = []
    page = 1

    while True:
        url = f'{API_URL}/api/v1/courses/{course_id}/assignments'
        params = {
            'per_page': 100,
            'page': page,
            'order_by': 'due_at'
        }

        data, remaining = safe_request(url, params)

        if not data:
            break

        for a in data:
            all_assignments.append({
                'assignment_id': a.get('id'),
                'course_id': course_id,
                'name': a.get('name'),
                'description': a.get('description', '')[:500] if a.get('description') else None,
                'points_possible': a.get('points_possible'),
                'due_at': a.get('due_at'),
                'unlock_at': a.get('unlock_at'),
                'lock_at': a.get('lock_at'),
                'grading_type': a.get('grading_type'),
                'submission_types': ','.join(a.get('submission_types', [])),
                'assignment_group_id': a.get('assignment_group_id'),
                'position': a.get('position'),
                'published': a.get('published'),
                'has_submitted_submissions': a.get('has_submitted_submissions'),
                'created_at': a.get('created_at'),
                'updated_at': a.get('updated_at')
            })

        print(f"    Page {page}: {len(data)} assignments (quota: {remaining})")

        if len(data) < 100:
            break
        page += 1

    return all_assignments


def get_assignment_groups(course_id: int) -> List[Dict]:
    """Get assignment groups with weights."""
    print("  Fetching assignment groups...")
    url = f'{API_URL}/api/v1/courses/{course_id}/assignment_groups'
    params = {'per_page': 100}

    data, _ = safe_request(url, params)

    if not data:
        return []

    groups = []
    for g in data:
        groups.append({
            'group_id': g.get('id'),
            'course_id': course_id,
            'name': g.get('name'),
            'position': g.get('position'),
            'group_weight': g.get('group_weight'),
            'rules': str(g.get('rules', {}))
        })

    print(f"    Found {len(groups)} assignment groups")
    return groups


def get_submissions(course_id: int) -> List[Dict]:
    """Get all student submissions using bookmark-based pagination."""
    print("  Fetching submissions...")
    all_submissions = []
    page_num = 1

    url = f'{API_URL}/api/v1/courses/{course_id}/students/submissions'
    params = {
        'student_ids[]': 'all',
        'per_page': 100
    }

    while url:
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            remaining = int(float(r.headers.get('X-Rate-Limit-Remaining', 700)))

            # Adaptive delay
            if remaining < 100:
                time.sleep(10)
            elif remaining < 300:
                time.sleep(2)
            else:
                time.sleep(0.5)

            if r.status_code != 200:
                print(f"    Error {r.status_code} on page {page_num}")
                break

            data = r.json()

            if not data:
                break

            for s in data:
                all_submissions.append({
                    'submission_id': s.get('id'),
                    'user_id': s.get('user_id'),
                    'assignment_id': s.get('assignment_id'),
                    'course_id': course_id,
                    'score': s.get('score'),
                    'grade': s.get('grade'),
                    'submitted_at': s.get('submitted_at'),
                    'graded_at': s.get('graded_at'),
                    'workflow_state': s.get('workflow_state'),
                    'late': s.get('late'),
                    'missing': s.get('missing'),
                    'excused': s.get('excused'),
                    'attempt': s.get('attempt'),
                    'seconds_late': s.get('seconds_late'),
                    'grade_matches_current_submission': s.get('grade_matches_current_submission')
                })

            print(f"    Page {page_num}: {len(data)} submissions (quota: {remaining})")

            # Get next URL from Link header (bookmark pagination)
            link_header = r.headers.get('Link', '')
            match = re.search(r'<([^>]+)>; rel="next"', link_header)
            if match:
                url = match.group(1)
                params = None  # URL already contains params
                page_num += 1
            else:
                break

        except Exception as e:
            print(f"    Exception on page {page_num}: {e}")
            break

    return all_submissions


def get_student_summaries(course_id: int) -> List[Dict]:
    """Get student activity summaries (page views, participations, tardiness)."""
    print("  Fetching student summaries...")
    all_summaries = []
    page = 1

    while True:
        url = f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries'
        params = {
            'per_page': 100,
            'page': page
        }

        data, remaining = safe_request(url, params)

        if not data:
            break

        for s in data:
            tardiness = s.get('tardiness_breakdown', {})
            all_summaries.append({
                'user_id': s.get('id'),
                'course_id': course_id,
                'page_views': s.get('page_views'),
                'page_views_level': s.get('page_views_level'),
                'participations': s.get('participations'),
                'participations_level': s.get('participations_level'),
                'on_time': tardiness.get('on_time', 0),
                'late': tardiness.get('late', 0),
                'missing': tardiness.get('missing', 0),
                'floating': tardiness.get('floating', 0)
            })

        print(f"    Page {page}: {len(data)} summaries (quota: {remaining})")

        if len(data) < 100:
            break
        page += 1

    return all_summaries


def get_modules(course_id: int) -> List[Dict]:
    """Get course modules."""
    print("  Fetching modules...")
    url = f'{API_URL}/api/v1/courses/{course_id}/modules'
    params = {'per_page': 100}

    data, _ = safe_request(url, params)

    if not data:
        return []

    modules = []
    for m in data:
        modules.append({
            'module_id': m.get('id'),
            'course_id': course_id,
            'name': m.get('name'),
            'position': m.get('position'),
            'items_count': m.get('items_count'),
            'state': m.get('state'),
            'unlock_at': m.get('unlock_at'),
            'published': m.get('published')
        })

    print(f"    Found {len(modules)} modules")
    return modules


def get_quizzes(course_id: int) -> List[Dict]:
    """Get quizzes metadata."""
    print("  Fetching quizzes...")
    url = f'{API_URL}/api/v1/courses/{course_id}/quizzes'
    params = {'per_page': 100}

    data, _ = safe_request(url, params)

    if not data:
        return []

    quizzes = []
    for q in data:
        quizzes.append({
            'quiz_id': q.get('id'),
            'course_id': course_id,
            'title': q.get('title'),
            'quiz_type': q.get('quiz_type'),
            'points_possible': q.get('points_possible'),
            'time_limit': q.get('time_limit'),
            'question_count': q.get('question_count'),
            'due_at': q.get('due_at'),
            'unlock_at': q.get('unlock_at'),
            'lock_at': q.get('lock_at'),
            'published': q.get('published'),
            'allowed_attempts': q.get('allowed_attempts')
        })

    print(f"    Found {len(quizzes)} quizzes")
    return quizzes


def get_pages(course_id: int) -> List[Dict]:
    """Get course pages (content)."""
    print("  Fetching pages...")
    all_pages = []
    page_num = 1

    while True:
        url = f'{API_URL}/api/v1/courses/{course_id}/pages'
        params = {
            'per_page': 100,
            'page': page_num
        }

        data, remaining = safe_request(url, params)

        if not data:
            break

        for p in data:
            all_pages.append({
                'page_id': p.get('page_id'),
                'course_id': course_id,
                'url': p.get('url'),
                'title': p.get('title'),
                'created_at': p.get('created_at'),
                'updated_at': p.get('updated_at'),
                'published': p.get('published'),
                'front_page': p.get('front_page'),
                'locked_for_user': p.get('locked_for_user')
            })

        if len(data) < 100:
            break
        page_num += 1

    print(f"    Found {len(all_pages)} pages")
    return all_pages


def get_files(course_id: int) -> List[Dict]:
    """Get course files (materials)."""
    print("  Fetching files...")
    all_files = []
    page_num = 1

    while True:
        url = f'{API_URL}/api/v1/courses/{course_id}/files'
        params = {
            'per_page': 100,
            'page': page_num
        }

        data, remaining = safe_request(url, params)

        if not data:
            break

        for f in data:
            all_files.append({
                'file_id': f.get('id'),
                'course_id': course_id,
                'folder_id': f.get('folder_id'),
                'display_name': f.get('display_name'),
                'filename': f.get('filename'),
                'content_type': f.get('content-type'),
                'size': f.get('size'),
                'created_at': f.get('created_at'),
                'updated_at': f.get('updated_at'),
                'locked': f.get('locked'),
                'hidden': f.get('hidden')
            })

        if len(data) < 100:
            break
        page_num += 1

    print(f"    Found {len(all_files)} files")
    return all_files


def get_discussion_topics(course_id: int) -> List[Dict]:
    """Get discussion topics (interaction)."""
    print("  Fetching discussion topics...")
    all_topics = []
    page_num = 1

    while True:
        url = f'{API_URL}/api/v1/courses/{course_id}/discussion_topics'
        params = {
            'per_page': 100,
            'page': page_num
        }

        data, remaining = safe_request(url, params)

        if not data:
            break

        for t in data:
            all_topics.append({
                'topic_id': t.get('id'),
                'course_id': course_id,
                'title': t.get('title'),
                'message': t.get('message', '')[:500] if t.get('message') else None,
                'discussion_type': t.get('discussion_type'),
                'posted_at': t.get('posted_at'),
                'delayed_post_at': t.get('delayed_post_at'),
                'last_reply_at': t.get('last_reply_at'),
                'discussion_subentry_count': t.get('discussion_subentry_count'),
                'published': t.get('published'),
                'locked': t.get('locked'),
                'pinned': t.get('pinned'),
                'assignment_id': t.get('assignment_id')
            })

        if len(data) < 100:
            break
        page_num += 1

    print(f"    Found {len(all_topics)} discussion topics")
    return all_topics


def extract_course_id_from_url(url: str) -> Optional[int]:
    """Extract course_id from Canvas URL."""
    match = re.search(r"/courses/(\d+)", url)
    return int(match.group(1)) if match else None


def get_page_views(course_id: int, user_ids: List[int], start_date: str, end_date: str) -> List[Dict]:
    """Get page views for all students in a course."""
    print(f"  Fetching page views for {len(user_ids)} students...")
    all_views = []

    start_time = f"{start_date}T00:00:00Z"
    end_time = f"{end_date}T23:59:59Z"

    for i, user_id in enumerate(user_ids):
        user_views = []
        page = 1

        while True:
            url = f'{API_URL}/api/v1/users/{user_id}/page_views'
            params = {
                'start_time': start_time,
                'end_time': end_time,
                'per_page': 100,
                'page': page
            }

            data, remaining = safe_request(url, params)

            if not data:
                break

            # Filter to our course only
            for pv in data:
                pv_course = extract_course_id_from_url(pv.get('url', ''))
                if pv_course == course_id:
                    user_views.append({
                        'user_id': user_id,
                        'course_id': course_id,
                        'url': pv.get('url'),
                        'context_type': pv.get('context_type'),
                        'asset_type': pv.get('asset_type'),
                        'controller': pv.get('controller'),
                        'action': pv.get('action'),
                        'interaction_seconds': pv.get('interaction_seconds'),
                        'created_at': pv.get('created_at'),
                        'participated': pv.get('participated', False),
                        'user_agent': pv.get('user_agent', '')[:200] if pv.get('user_agent') else None
                    })

            if len(data) < 100:
                break
            page += 1

        all_views.extend(user_views)

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(user_ids)} students, {len(all_views)} views total")

    print(f"    Total page views: {len(all_views)}")
    return all_views


def save_to_parquet(data: List[Dict], filepath: str, name: str) -> bool:
    """Save data to Parquet file."""
    if not data:
        print(f"    No {name} data to save")
        return False

    df = pd.DataFrame(data)
    df.to_parquet(filepath, index=False)
    print(f"    Saved {len(data)} {name} records to {filepath}")
    return True


def extract_course_data(course_id: int, output_dir: str, start_date: str, end_date: str,
                         include_page_views: bool = False) -> Dict[str, int]:
    """Main extraction function - fetches and saves all course data."""
    print("=" * 60)
    print("CANVAS COURSE DATA EXTRACTOR")
    print("=" * 60)

    # Test connection
    r = requests.get(f'{API_URL}/api/v1/users/self', headers=headers)
    if r.status_code == 200:
        user = r.json()
        print(f"Connected as: {user.get('name', 'Unknown')}")
        print(f"Rate Limit: {r.headers.get('X-Rate-Limit-Remaining', 'N/A')}")
    else:
        print(f"Connection failed: {r.status_code}")
        return {}

    # Get course info
    print(f"\n{'=' * 60}")
    print(f"EXTRACTING COURSE: {course_id}")
    print("=" * 60)

    course_info = get_course_info(course_id)
    if not course_info:
        print(f"ERROR: Could not fetch course {course_id}")
        return {}

    course_name = course_info.get('name', f'Course {course_id}')
    print(f"Course: {course_name}")
    print(f"Term: {course_info.get('term', {}).get('name', 'Unknown')}")
    print(f"Total Students: {course_info.get('total_students', 'N/A')}")

    # Create output directory
    course_dir = f'{output_dir}/course_{course_id}'
    os.makedirs(course_dir, exist_ok=True)

    # Track extraction stats
    stats = {}

    # 1. Enrollments (grades)
    print("\n--- Extracting Enrollments ---")
    enrollments = get_enrollments(course_id)
    save_to_parquet(enrollments, f'{course_dir}/enrollments.parquet', 'enrollments')
    stats['enrollments'] = len(enrollments)

    # 2. Assignments
    print("\n--- Extracting Assignments ---")
    assignments = get_assignments(course_id)
    save_to_parquet(assignments, f'{course_dir}/assignments.parquet', 'assignments')
    stats['assignments'] = len(assignments)

    # 3. Assignment Groups
    print("\n--- Extracting Assignment Groups ---")
    groups = get_assignment_groups(course_id)
    save_to_parquet(groups, f'{course_dir}/assignment_groups.parquet', 'assignment_groups')
    stats['assignment_groups'] = len(groups)

    # 4. Submissions
    print("\n--- Extracting Submissions ---")
    submissions = get_submissions(course_id)
    save_to_parquet(submissions, f'{course_dir}/submissions.parquet', 'submissions')
    stats['submissions'] = len(submissions)

    # 5. Student Summaries (activity)
    print("\n--- Extracting Student Summaries ---")
    summaries = get_student_summaries(course_id)
    save_to_parquet(summaries, f'{course_dir}/student_summaries.parquet', 'student_summaries')
    stats['student_summaries'] = len(summaries)

    # 6. Modules
    print("\n--- Extracting Modules ---")
    modules = get_modules(course_id)
    save_to_parquet(modules, f'{course_dir}/modules.parquet', 'modules')
    stats['modules'] = len(modules)

    # 7. Quizzes
    print("\n--- Extracting Quizzes ---")
    quizzes = get_quizzes(course_id)
    save_to_parquet(quizzes, f'{course_dir}/quizzes.parquet', 'quizzes')
    stats['quizzes'] = len(quizzes)

    # 8. Pages (content)
    print("\n--- Extracting Pages ---")
    pages = get_pages(course_id)
    save_to_parquet(pages, f'{course_dir}/pages.parquet', 'pages')
    stats['pages'] = len(pages)

    # 9. Files (materials)
    print("\n--- Extracting Files ---")
    files = get_files(course_id)
    save_to_parquet(files, f'{course_dir}/files.parquet', 'files')
    stats['files'] = len(files)

    # 10. Discussion Topics (interaction)
    print("\n--- Extracting Discussion Topics ---")
    discussions = get_discussion_topics(course_id)
    save_to_parquet(discussions, f'{course_dir}/discussion_topics.parquet', 'discussion_topics')
    stats['discussion_topics'] = len(discussions)

    # 11. Page Views (optional - can be slow)
    if include_page_views and enrollments:
        print("\n--- Extracting Page Views ---")
        user_ids = [e['user_id'] for e in enrollments if e['user_id']]
        page_views = get_page_views(course_id, user_ids, start_date, end_date)
        save_to_parquet(page_views, f'{course_dir}/page_views.parquet', 'page_views')
        stats['page_views'] = len(page_views)

    # 9. Save course metadata
    course_meta = {
        'course_id': course_id,
        'course_name': course_name,
        'account_id': course_info.get('account_id'),
        'term_id': course_info.get('enrollment_term_id'),
        'term_name': course_info.get('term', {}).get('name'),
        'total_students': course_info.get('total_students'),
        'extraction_date': datetime.now().isoformat(),
        'start_date': start_date,
        'end_date': end_date,
        **{f'n_{k}': v for k, v in stats.items()}
    }
    save_to_parquet([course_meta], f'{course_dir}/course_info.parquet', 'course_info')

    # Print summary
    print(f"\n{'=' * 60}")
    print("EXTRACTION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {course_dir}/")
    print(f"\nRecords extracted:")
    for name, count in stats.items():
        print(f"  {name}: {count}")

    # Compute some quick stats
    if enrollments:
        grades = [e['final_score'] for e in enrollments if e['final_score'] is not None]
        if grades:
            passing = sum(1 for g in grades if g >= PASS_THRESHOLD)
            print(f"\nGrade Summary:")
            print(f"  Students with grades: {len(grades)}")
            print(f"  Pass rate: {passing}/{len(grades)} ({100*passing/len(grades):.1f}%)")
            print(f"  Grade range: {min(grades):.1f}% - {max(grades):.1f}%")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Extract ALL data for a specific Canvas course and save to Parquet files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_course_data.py --course-id 86676
  python extract_course_data.py --course-id 86676 --include-page-views
  python extract_course_data.py --course-id 86676 --start-date 2025-08-01 --end-date 2025-12-31
  python extract_course_data.py --course-id 86676 --output-dir data/courses

Output Structure:
  {output_dir}/course_{id}/
  ├── enrollments.parquet       # Student grades (current_score, final_score)
  ├── assignments.parquet       # Assignment metadata
  ├── assignment_groups.parquet # Grade weights
  ├── submissions.parquet       # Per-student submission data
  ├── student_summaries.parquet # Activity metrics (page_views, participations)
  ├── modules.parquet           # Course structure
  ├── quizzes.parquet           # Quiz metadata
  ├── pages.parquet             # Content pages
  ├── files.parquet             # Course materials/files
  ├── discussion_topics.parquet # Discussion forums
  ├── page_views.parquet        # Clickstream data (if --include-page-views)
  └── course_info.parquet       # Course metadata
        """
    )

    parser.add_argument('--course-id', type=int, required=True,
                        help='Canvas course ID to extract')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory for Parquet files (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE,
                        help=f'Start date for page views (YYYY-MM-DD, default: {DEFAULT_START_DATE})')
    parser.add_argument('--end-date', type=str, default=DEFAULT_END_DATE,
                        help=f'End date for page views (YYYY-MM-DD, default: {DEFAULT_END_DATE})')
    parser.add_argument('--include-page-views', action='store_true',
                        help='Include page views extraction (can be slow for large courses)')

    args = parser.parse_args()

    extract_course_data(
        course_id=args.course_id,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        include_page_views=args.include_page_views
    )


if __name__ == '__main__':
    main()
