#!/usr/bin/env python3
"""
Fetch enrollment grades for all model courses.

This script fetches enrollment data with grades for the 10 courses used in our model.
"""

import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from time import sleep

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

DATA_DIR = Path('/home/paul/projects/uautonoma/data')

# Model courses
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]


def fetch_enrollments(course_id):
    """Fetch all student enrollments with grades for a course."""
    all_enrollments = []
    url = f'{API_URL}/api/v1/courses/{course_id}/enrollments'
    params = {
        'type[]': 'StudentEnrollment',
        'per_page': 100,
        'include[]': ['grades', 'total_scores']
    }

    while url:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code != 200:
            print(f"  Error {response.status_code}: {response.text[:200]}")
            break

        data = response.json()
        all_enrollments.extend(data)

        # Get next page from Link header
        links = response.headers.get('Link', '')
        url = None
        for link in links.split(','):
            if 'rel="next"' in link:
                url = link.split('<')[1].split('>')[0]
                params = {}  # Clear params for next page URL
                break

        sleep(0.1)  # Rate limiting

    return all_enrollments


def main():
    print("=" * 60)
    print("FETCHING ENROLLMENT GRADES FOR MODEL COURSES")
    print("=" * 60)

    all_enrollments = []

    for course_id in MODEL_COURSES:
        print(f"\nFetching course {course_id}...")
        enrollments = fetch_enrollments(course_id)
        print(f"  Found {len(enrollments)} enrollments")

        # Count with grades
        with_grades = sum(
            1 for e in enrollments
            if e.get('grades', {}).get('final_score') or e.get('grades', {}).get('current_score')
        )
        print(f"  With grades: {with_grades}")

        all_enrollments.extend(enrollments)

    # Save
    output_path = DATA_DIR / 'model_courses_enrollments.json'
    with open(output_path, 'w') as f:
        json.dump(all_enrollments, f, indent=2)

    print(f"\nTotal enrollments: {len(all_enrollments)}")
    print(f"Saved to: {output_path}")

    # Summary by course
    print("\nSummary by course:")
    for course_id in MODEL_COURSES:
        course_enrollments = [e for e in all_enrollments if e.get('course_id') == course_id]
        with_grades = sum(
            1 for e in course_enrollments
            if e.get('grades', {}).get('final_score') or e.get('grades', {}).get('current_score')
        )
        print(f"  Course {course_id}: {len(course_enrollments)} students, {with_grades} with grades")


if __name__ == '__main__':
    main()
