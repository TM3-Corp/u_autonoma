"""
Extract all Canvas data for grade prediction analysis
Uses efficient aggregated APIs (student_summaries, enrollments)
"""

import requests
import json
import os
from datetime import datetime
from config import API_URL, API_TOKEN, HIGH_POTENTIAL_COURSES, DATA_DIR, ACCOUNT_ID_CARRERA

headers = {'Authorization': f'Bearer {API_TOKEN}'}

def paginate(url, params=None):
    """Helper to paginate through Canvas API results using Link header"""
    if params is None:
        params = {}
    params['per_page'] = 100

    all_results = []

    # First request
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        print(f"  Error {r.status_code}: {r.text[:100]}")
        return all_results

    data = r.json()
    if not data:
        return all_results

    all_results.extend(data)

    # Follow Link header pagination (Canvas-proper method)
    while 'next' in r.links:
        next_url = r.links['next']['url']
        r = requests.get(next_url, headers=headers)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        all_results.extend(data)

    return all_results


def extract_courses():
    """Extract all courses with metadata"""
    print("=" * 80)
    print("EXTRACTING COURSES")
    print("=" * 80)

    courses = paginate(f'{API_URL}/api/v1/accounts/{ACCOUNT_ID_CARRERA}/courses',
                       {'include[]': ['total_students', 'term']})

    print(f"Found {len(courses)} courses")

    # Save raw courses
    with open(os.path.join(DATA_DIR, 'courses_raw.json'), 'w') as f:
        json.dump(courses, f, indent=2)

    return courses


def extract_student_summaries(course_ids):
    """Extract aggregated student summaries for each course"""
    print("\n" + "=" * 80)
    print("EXTRACTING STUDENT SUMMARIES (Aggregated)")
    print("=" * 80)

    all_summaries = []

    for i, course_id in enumerate(course_ids, 1):
        print(f"  [{i}/{len(course_ids)}] Course {course_id}...", end=" ")

        try:
            summaries = paginate(f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries')

            # Add course_id to each summary
            for s in summaries:
                s['course_id'] = course_id

            all_summaries.extend(summaries)
            print(f"{len(summaries)} students")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal student summaries: {len(all_summaries)}")

    with open(os.path.join(DATA_DIR, 'student_summaries.json'), 'w') as f:
        json.dump(all_summaries, f, indent=2)

    return all_summaries


def extract_enrollments(course_ids):
    """Extract enrollments with activity metrics"""
    print("\n" + "=" * 80)
    print("EXTRACTING ENROLLMENTS (Activity Metrics)")
    print("=" * 80)

    all_enrollments = []

    for i, course_id in enumerate(course_ids, 1):
        print(f"  [{i}/{len(course_ids)}] Course {course_id}...", end=" ")

        try:
            enrollments = paginate(f'{API_URL}/api/v1/courses/{course_id}/enrollments',
                                   {'type[]': 'StudentEnrollment'})

            # Add course_id to each enrollment
            for e in enrollments:
                e['course_id'] = course_id

            all_enrollments.extend(enrollments)
            print(f"{len(enrollments)} enrollments")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal enrollments: {len(all_enrollments)}")

    with open(os.path.join(DATA_DIR, 'enrollments.json'), 'w') as f:
        json.dump(all_enrollments, f, indent=2)

    return all_enrollments


def extract_assignments(course_ids):
    """Extract assignments with analytics"""
    print("\n" + "=" * 80)
    print("EXTRACTING ASSIGNMENTS")
    print("=" * 80)

    all_assignments = []
    all_assignment_analytics = []

    for i, course_id in enumerate(course_ids, 1):
        print(f"  [{i}/{len(course_ids)}] Course {course_id}...", end=" ")

        try:
            # Get assignments
            assignments = paginate(f'{API_URL}/api/v1/courses/{course_id}/assignments')
            for a in assignments:
                a['course_id'] = course_id
            all_assignments.extend(assignments)

            # Get assignment analytics
            r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/analytics/assignments', headers=headers)
            if r.status_code == 200:
                analytics = r.json()
                for a in analytics:
                    a['course_id'] = course_id
                all_assignment_analytics.extend(analytics)

            print(f"{len(assignments)} assignments")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal assignments: {len(all_assignments)}")
    print(f"Total assignment analytics: {len(all_assignment_analytics)}")

    with open(os.path.join(DATA_DIR, 'assignments.json'), 'w') as f:
        json.dump(all_assignments, f, indent=2)

    with open(os.path.join(DATA_DIR, 'assignment_analytics.json'), 'w') as f:
        json.dump(all_assignment_analytics, f, indent=2)

    return all_assignments, all_assignment_analytics


def extract_submissions(course_ids):
    """Extract all student submissions with grades"""
    print("\n" + "=" * 80)
    print("EXTRACTING SUBMISSIONS (Grades)")
    print("=" * 80)

    all_submissions = []

    for i, course_id in enumerate(course_ids, 1):
        print(f"  [{i}/{len(course_ids)}] Course {course_id}...", end=" ")

        try:
            submissions = paginate(f'{API_URL}/api/v1/courses/{course_id}/students/submissions',
                                   {'student_ids[]': 'all'})

            for s in submissions:
                s['course_id'] = course_id

            all_submissions.extend(submissions)
            print(f"{len(submissions)} submissions")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal submissions: {len(all_submissions)}")

    with open(os.path.join(DATA_DIR, 'submissions.json'), 'w') as f:
        json.dump(all_submissions, f, indent=2)

    return all_submissions


def extract_course_activity(course_ids):
    """Extract daily course activity"""
    print("\n" + "=" * 80)
    print("EXTRACTING COURSE ACTIVITY (Daily)")
    print("=" * 80)

    all_activity = []

    for i, course_id in enumerate(course_ids, 1):
        print(f"  [{i}/{len(course_ids)}] Course {course_id}...", end=" ")

        try:
            r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/analytics/activity', headers=headers)
            if r.status_code == 200:
                activity = r.json()
                for a in activity:
                    a['course_id'] = course_id
                all_activity.extend(activity)
                print(f"{len(activity)} days")
            else:
                print(f"Error {r.status_code}")

        except Exception as e:
            print(f"ERROR: {e}")

    print(f"\nTotal activity records: {len(all_activity)}")

    with open(os.path.join(DATA_DIR, 'course_activity.json'), 'w') as f:
        json.dump(all_activity, f, indent=2)

    return all_activity


def main():
    """Main extraction function"""
    print("=" * 80)
    print("CANVAS DATA EXTRACTION - UNIVERSIDAD AUTÓNOMA")
    print(f"Started: {datetime.now().isoformat()}")
    print("=" * 80)

    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Extract courses
    courses = extract_courses()

    # Use high-potential courses for detailed extraction
    course_ids = HIGH_POTENTIAL_COURSES
    print(f"\nExtracting detailed data for {len(course_ids)} high-potential courses")

    # Extract all data
    student_summaries = extract_student_summaries(course_ids)
    enrollments = extract_enrollments(course_ids)
    assignments, assignment_analytics = extract_assignments(course_ids)
    submissions = extract_submissions(course_ids)
    course_activity = extract_course_activity(course_ids)

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Courses: {len(courses)}")
    print(f"Student summaries: {len(student_summaries)}")
    print(f"Enrollments: {len(enrollments)}")
    print(f"Assignments: {len(assignments)}")
    print(f"Submissions: {len(submissions)}")
    print(f"Activity records: {len(course_activity)}")
    print(f"\nData saved to: {DATA_DIR}")
    print(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
