#!/usr/bin/env python3
"""Fetch complete student list with aggressive rate limiting."""

import requests
import time
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

COURSES = {
    79875: 'TALLER DE COMP DIGITALES-P01',
    79913: 'FUND. DE BUSINESS ANALYTICS-P01',
    84936: 'FUNDAMENTOS DE MICROECONOMÍA-P03',
    84941: 'FUNDAMENTOS DE MICROECONOMÍA-P01',
    84944: 'FUNDAMENTOS DE MACROECONOMÍA-P03',
    86020: 'TALL DE COMPETENCIAS DIGITALES-P02',
    86676: 'FUND DE BUSINESS ANALYTICS-P01',
    88381: 'MATEMÁTICAS PARA LOS NEGOCIOS-P01',
    89099: 'TALLER DE COMP DIGITALES-P01',
    89390: 'GESTIÓN DEL TALENTO-P01',
}

OUTPUT_DIR = '/home/paul/projects/uautonoma/data/page_views'

print("Fetching complete student list with aggressive rate limiting...")
print()

all_students = []

for course_id, name in COURSES.items():
    print(f"  Course {course_id} ({name})...", end=' ', flush=True)

    # Wait and check rate limit with retries
    for attempt in range(10):
        r = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/enrollments',
            headers=HEADERS,
            params={'type[]': 'StudentEnrollment', 'per_page': 100, 'state[]': ['active', 'completed']}
        )

        remaining = float(r.headers.get('X-Rate-Limit-Remaining', 100))

        if r.status_code == 429:
            wait_time = 60 * (attempt + 1)
            print(f"Rate limited, waiting {wait_time}s...", end=' ', flush=True)
            time.sleep(wait_time)
            continue
        elif r.status_code == 200:
            data = r.json()
            for e in data:
                all_students.append({
                    'user_id': e['user_id'],
                    'course_id': course_id,
                    'enrollment_state': e.get('enrollment_state'),
                    'current_score': e.get('grades', {}).get('current_score'),
                    'final_score': e.get('grades', {}).get('final_score'),
                })
            print(f"{len(data)} students (quota: {remaining:.0f})")

            # Wait based on remaining quota
            if remaining < 100:
                wait = max(10, (100 - remaining) * 0.5)
                print(f"    Low quota ({remaining:.0f}), waiting {wait:.0f}s...")
                time.sleep(wait)
            else:
                time.sleep(3)  # Small delay between courses
            break
        else:
            print(f"Error {r.status_code}, retrying...")
            time.sleep(30)
            continue

# Deduplicate
unique = {}
for s in all_students:
    uid = s['user_id']
    if uid not in unique:
        unique[uid] = []
    unique[uid].append(s['course_id'])

print()
print(f"Total enrollments: {len(all_students)}")
print(f"Unique students: {len(unique)}")

# Save complete list
if len(all_students) >= 350:
    df = pd.DataFrame(all_students)
    df.to_csv(f'{OUTPUT_DIR}/student_enrollments_complete.csv', index=False)
    print(f"Saved COMPLETE list to {OUTPUT_DIR}/student_enrollments_complete.csv")
elif len(all_students) > 0:
    df = pd.DataFrame(all_students)
    df.to_csv(f'{OUTPUT_DIR}/student_enrollments_partial.csv', index=False)
    print(f"WARNING: Only got {len(all_students)} enrollments, saved as partial")
else:
    print("ERROR: No students fetched!")
