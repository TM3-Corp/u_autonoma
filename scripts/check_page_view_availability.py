#!/usr/bin/env python3
"""Check which courses have recent page view data available."""

import os
import re
import time
import requests
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

OUR_COURSES = set([79875, 79913, 84936, 84941, 84944, 86005, 86020, 86676, 88381, 89099, 89390, 89736])


def parse_link_header(link_header):
    """Parse Link header to extract rel=next URL."""
    if not link_header:
        return None

    for part in link_header.split(','):
        if 'rel="next"' in part:
            # Extract URL from <URL>; rel="next"
            match = re.search(r'<([^>]+)>', part)
            if match:
                return match.group(1)
    return None


def get_all_page_views(user_id, max_pages=100):
    """Get ALL page views for a user using bookmark pagination."""
    all_views = []
    url = f'{API_URL}/api/v1/users/{user_id}/page_views'
    params = {'per_page': 100}
    pages_fetched = 0

    while url and pages_fetched < max_pages:
        r = requests.get(url, headers=headers, params=params if pages_fetched == 0 else None)

        if r.status_code != 200:
            print(f'Error: {r.status_code}')
            break

        views = r.json()
        if not views:
            break

        all_views.extend(views)
        pages_fetched += 1

        # Get next page URL from Link header
        link_header = r.headers.get('Link', '')
        url = parse_link_header(link_header)

        if not url:
            break

        time.sleep(0.1)  # Rate limiting

    return all_views


def extract_course_id(url):
    """Extract course_id from Canvas URL."""
    if not url:
        return None
    match = re.search(r'/courses/(\d+)', url)
    return int(match.group(1)) if match else None


print('=== Checking Page View Availability (Full Pagination) ===')
print()

# Pick 5 students to test (from different courses)
test_students = []

for course_id in [86005, 86676, 89099]:  # Sample courses
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 3}
    )
    if r.status_code == 200:
        for e in r.json()[:2]:
            test_students.append((e['user_id'], course_id))

print(f'Testing {len(test_students)} students...')
print()

# Aggregate page views by course
course_views = Counter()
total_views_all = 0

for user_id, source_course in test_students:
    print(f'  User {user_id} (from course {source_course})...', end=' ', flush=True)

    all_views = get_all_page_views(user_id)
    total_views_all += len(all_views)

    # Count by course
    for v in all_views:
        cid = extract_course_id(v.get('url', ''))
        if cid:
            course_views[cid] += 1

    # Get dates
    if all_views:
        dates = sorted([v.get('created_at', '')[:10] for v in all_views if v.get('created_at')])
        print(f'{len(all_views)} views ({dates[0]} to {dates[-1]})')
    else:
        print('0 views')

print()
print(f'Total page views collected: {total_views_all}')
print()
print('=== Page Views by Course ===')
print()

for cid, count in course_views.most_common(20):
    in_our_list = '✓' if cid in OUR_COURSES else ''
    print(f'  Course {cid}: {count:>5} views {in_our_list}')

print()
print('=== Our Courses Summary ===')
our_course_views = {cid: course_views.get(cid, 0) for cid in OUR_COURSES}
for cid in sorted(OUR_COURSES):
    count = our_course_views.get(cid, 0)
    status = 'YES' if count > 0 else 'NO '
    print(f'  {status} | Course {cid}: {count:>5} views')
