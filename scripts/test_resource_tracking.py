#!/usr/bin/env python3
"""
Test Canvas API endpoints for resource-level access tracking.

Exploring endpoints to implement Oviedo-style metrics:
- ResourceViewUniquePct: % of course resources viewed by each student
- ResourceViewTime: Timing/ranking of when students accessed resources
"""

import requests
import json
import sys
sys.path.insert(0, '/home/paul/projects/uautonoma/scripts')
from config import API_URL, API_TOKEN

headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Test with a completed course that has good data
TEST_COURSE_ID = 84936  # FUNDAMENTOS DE MICROECONOMÍA-P03 (41 students, completed)


def test_endpoint(name, url, params=None):
    """Test an endpoint and return response."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    print("-" * 60)

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        print(f"Status: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"Records: {len(data)}")
                if data:
                    print(f"Sample keys: {list(data[0].keys())[:10]}")
                    if len(data) > 0:
                        print(f"\nFirst record sample:")
                        print(json.dumps(data[0], indent=2, default=str)[:1000])
            elif isinstance(data, dict):
                print(f"Keys: {list(data.keys())[:15]}")
                print(f"\nResponse sample:")
                print(json.dumps(data, indent=2, default=str)[:1500])
            return data
        else:
            print(f"Error: {response.text[:500]}")
            return None
    except Exception as e:
        print(f"Exception: {e}")
        return None


def main():
    print("=" * 60)
    print("CANVAS RESOURCE TRACKING API EXPLORATION")
    print(f"Test Course: {TEST_COURSE_ID}")
    print("=" * 60)

    # 1. Test bulk_user_progress endpoint
    progress_data = test_endpoint(
        "1. Bulk User Progress (module completion per student)",
        f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/bulk_user_progress"
    )

    # 2. Test modules list
    modules_data = test_endpoint(
        "2. List Modules",
        f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/modules",
        params={'include[]': ['items', 'content_details']}
    )

    # 3. Get first student ID from enrollments to test student-specific endpoints
    enrollments = test_endpoint(
        "3. Get Enrollments (to find student IDs)",
        f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/enrollments",
        params={'type[]': 'StudentEnrollment', 'per_page': 5}
    )

    student_id = None
    if enrollments and len(enrollments) > 0:
        student_id = enrollments[0].get('user_id')
        print(f"\nUsing student_id: {student_id} for testing")

    # 4. Test modules with student_id parameter (shows completion state)
    if student_id:
        test_endpoint(
            f"4. Modules with student_id={student_id} (completion state)",
            f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/modules",
            params={'student_id': student_id, 'include[]': ['items']}
        )

    # 5. If we have modules, get module items with student completion
    if modules_data and len(modules_data) > 0:
        module_id = modules_data[0].get('id')
        print(f"\nUsing module_id: {module_id} for testing")

        # Module items (shows completion_requirement)
        items_data = test_endpoint(
            f"5. Module Items (module {module_id})",
            f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/modules/{module_id}/items",
            params={'include[]': 'content_details'}
        )

        # Module items with student_id (shows if student completed)
        if student_id:
            test_endpoint(
                f"6. Module Items with student_id={student_id}",
                f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/modules/{module_id}/items",
                params={'student_id': student_id, 'include[]': 'content_details'}
            )

    # 7. Test individual user progress
    if student_id:
        test_endpoint(
            f"7. Individual User Progress (student {student_id})",
            f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/users/{student_id}/progress"
        )

    # 8. Test per-user activity (detailed page views by hour)
    if student_id:
        test_endpoint(
            f"8. User Activity in Course (hourly breakdown)",
            f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/analytics/users/{student_id}/activity"
        )

    # 9. Test per-user assignments (shows submission status per assignment)
    if student_id:
        test_endpoint(
            f"9. User Assignments Analytics",
            f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/analytics/users/{student_id}/assignments"
        )

    # 10. Test pages endpoint (for wiki/content pages)
    test_endpoint(
        "10. Course Pages (wiki pages)",
        f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/pages",
        params={'per_page': 10}
    )

    # 11. Test files endpoint
    test_endpoint(
        "11. Course Files",
        f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/files",
        params={'per_page': 10}
    )

    # 12. Test usage rights/access data for files
    files_data = None
    resp = requests.get(
        f"{API_URL}/api/v1/courses/{TEST_COURSE_ID}/files",
        headers=headers,
        params={'per_page': 5}
    )
    if resp.status_code == 200:
        files_data = resp.json()

    if files_data and len(files_data) > 0:
        file_id = files_data[0].get('id')
        test_endpoint(
            f"12. File Details (file {file_id})",
            f"{API_URL}/api/v1/files/{file_id}"
        )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("""
Key Findings for Resource Tracking:

1. bulk_user_progress: Shows requirement_completed_count / requirement_count
   → Use this for ResourceViewUniquePct (if modules have requirements)

2. Modules with student_id: Shows 'state' (locked/unlocked/started/completed)
   → Shows progression through course structure

3. Module Items: Show completion_requirement and 'completed' status per student
   → Can track which specific resources each student viewed

4. User Activity: Hourly page views and participations
   → Can derive timing/recency of access

5. Pages/Files: List available content
   → Need to cross-reference with page views to see who accessed what

LIMITATION: No direct "views per resource per student" endpoint.
Must either:
- Use module requirements (if configured by instructors)
- Use page_views API (per user, filter by URL, expensive)
- Use Live Events (requires AWS SQS setup - enterprise feature)
""")


if __name__ == '__main__':
    main()
