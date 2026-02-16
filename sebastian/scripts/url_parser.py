"""
URL Parser Module

This module provides utilities for parsing Canvas URLs to extract:
- Course IDs
- Resource types (assignments, modules, files, etc.)
- User IDs
- Other identifiers

Designed to support page views analysis later.

Usage:
    from scripts.discovery.url_parser import parse_canvas_url, extract_course_id

    url = "https://canvas.example.com/courses/12345/assignments/67890"
    info = parse_canvas_url(url)
    # {'course_id': 12345, 'resource_type': 'assignments', 'resource_id': 67890}
"""

import re
from typing import Optional, Dict, Any
from urllib.parse import urlparse, parse_qs


# Resource type patterns
RESOURCE_PATTERNS = {
    'assignments': r'/courses/(\d+)/assignments(?:/(\d+))?',
    'modules': r'/courses/(\d+)/modules(?:/(\d+))?',
    'files': r'/courses/(\d+)/files(?:/(\d+))?',
    'quizzes': r'/courses/(\d+)/quizzes(?:/(\d+))?',
    'discussions': r'/courses/(\d+)/discussion_topics(?:/(\d+))?',
    'pages': r'/courses/(\d+)/pages(?:/([^/]+))?',
    'announcements': r'/courses/(\d+)/announcements(?:/(\d+))?',
    'grades': r'/courses/(\d+)/grades(?:/(\d+))?',
    'gradebook': r'/courses/(\d+)/gradebook',
    'syllabus': r'/courses/(\d+)/assignments/syllabus',
    'users': r'/courses/(\d+)/users(?:/(\d+))?',
    'groups': r'/courses/(\d+)/groups(?:/(\d+))?',
    'external_tools': r'/courses/(\d+)/external_tools(?:/(\d+))?',
    'conferences': r'/courses/(\d+)/conferences(?:/(\d+))?',
    'collaborations': r'/courses/(\d+)/collaborations(?:/(\d+))?',
    'outcomes': r'/courses/(\d+)/outcomes(?:/(\d+))?',
    'rubrics': r'/courses/(\d+)/rubrics(?:/(\d+))?',
    'settings': r'/courses/(\d+)/settings',
    'home': r'/courses/(\d+)/?$',
}

# Action patterns (appended to resource URLs)
ACTION_PATTERNS = {
    'edit': r'/edit$',
    'new': r'/new$',
    'submit': r'/submissions',
    'preview': r'/preview',
    'download': r'/download',
}


def extract_course_id(url: str) -> Optional[int]:
    """
    Extract course ID from a Canvas URL.

    Args:
        url: Canvas URL string

    Returns:
        Course ID as integer or None if not found
    """
    match = re.search(r'/courses/(\d+)', url)
    if match:
        return int(match.group(1))
    return None


def extract_user_id(url: str) -> Optional[int]:
    """
    Extract user ID from a Canvas URL.

    Args:
        url: Canvas URL string

    Returns:
        User ID as integer or None if not found
    """
    # User profile URLs
    match = re.search(r'/users/(\d+)', url)
    if match:
        return int(match.group(1))

    # Submission URLs with user
    match = re.search(r'/submissions/(\d+)', url)
    if match:
        return int(match.group(1))

    return None


def parse_canvas_url(url: str) -> Dict[str, Any]:
    """
    Parse a Canvas URL and extract structured information.

    Args:
        url: Canvas URL string

    Returns:
        Dictionary with parsed components:
        - course_id: Course ID (int or None)
        - resource_type: Type of resource (str or None)
        - resource_id: Resource ID (int/str or None)
        - action: Action being performed (str or None)
        - is_api: Whether this is an API URL
        - path: URL path component
        - query_params: Query string parameters
    """
    result = {
        'course_id': None,
        'resource_type': None,
        'resource_id': None,
        'action': None,
        'is_api': False,
        'path': '',
        'query_params': {}
    }

    try:
        parsed = urlparse(url)
        result['path'] = parsed.path
        result['query_params'] = parse_qs(parsed.query)

        # Check if API URL
        result['is_api'] = '/api/v1/' in parsed.path

        # Extract course ID
        result['course_id'] = extract_course_id(url)

        # Try to match resource patterns
        for resource_type, pattern in RESOURCE_PATTERNS.items():
            match = re.search(pattern, parsed.path)
            if match:
                result['resource_type'] = resource_type
                if match.lastindex and match.lastindex >= 2 and match.group(2):
                    # Try to convert to int, keep as string if not possible
                    try:
                        result['resource_id'] = int(match.group(2))
                    except ValueError:
                        result['resource_id'] = match.group(2)
                break

        # Check for actions
        for action, pattern in ACTION_PATTERNS.items():
            if re.search(pattern, parsed.path):
                result['action'] = action
                break

    except Exception:
        pass

    return result


def categorize_page_view(url: str) -> str:
    """
    Categorize a page view URL into a high-level activity type.

    Args:
        url: Canvas URL from page view data

    Returns:
        Activity category string
    """
    info = parse_canvas_url(url)

    if info['is_api']:
        return 'api_call'

    resource = info['resource_type']
    action = info['action']

    # Map to activity categories
    if resource == 'assignments':
        if action == 'submit':
            return 'assignment_submission'
        elif action in ['edit', 'new']:
            return 'assignment_edit'
        return 'assignment_view'

    if resource == 'quizzes':
        return 'quiz_activity'

    if resource == 'modules':
        return 'module_navigation'

    if resource == 'files':
        if action == 'download':
            return 'file_download'
        return 'file_view'

    if resource == 'discussions':
        return 'discussion_activity'

    if resource == 'pages':
        return 'page_view'

    if resource == 'grades' or resource == 'gradebook':
        return 'grade_check'

    if resource == 'home':
        return 'course_home'

    if resource == 'syllabus':
        return 'syllabus_view'

    if info['course_id']:
        return 'other_course_activity'

    return 'other'


def filter_page_views_by_course(
    page_views: list,
    course_id: int
) -> list:
    """
    Filter page views to only include those for a specific course.

    Canvas Page Views API doesn't have a course filter, so we must
    filter post-fetch by parsing URLs.

    Args:
        page_views: List of page view records
        course_id: Course ID to filter by

    Returns:
        Filtered list of page views
    """
    filtered = []
    for pv in page_views:
        url = pv.get('url', '')
        pv_course_id = extract_course_id(url)
        if pv_course_id == course_id:
            filtered.append(pv)
    return filtered


def aggregate_page_views(page_views: list) -> Dict[str, Any]:
    """
    Aggregate page views into summary statistics.

    Args:
        page_views: List of page view records

    Returns:
        Dictionary with aggregated statistics
    """
    stats = {
        'total_views': len(page_views),
        'total_interaction_time': 0,
        'by_category': {},
        'by_course': {},
        'by_hour': {str(h): 0 for h in range(24)},
    }

    for pv in page_views:
        # Interaction time
        stats['total_interaction_time'] += pv.get('interaction_seconds', 0) or 0

        # By category
        url = pv.get('url', '')
        category = categorize_page_view(url)
        stats['by_category'][category] = stats['by_category'].get(category, 0) + 1

        # By course
        course_id = extract_course_id(url)
        if course_id:
            stats['by_course'][course_id] = stats['by_course'].get(course_id, 0) + 1

        # By hour
        created_at = pv.get('created_at', '')
        if created_at and len(created_at) >= 13:
            try:
                hour = created_at[11:13]
                stats['by_hour'][hour] = stats['by_hour'].get(hour, 0) + 1
            except (IndexError, KeyError):
                pass

    return stats


if __name__ == '__main__':
    # Test the module
    print("Testing url_parser module...")
    print("=" * 60)

    test_urls = [
        "https://uautonoma.test.instructure.com/courses/86005/assignments/465607",
        "https://uautonoma.test.instructure.com/courses/86005/modules",
        "https://uautonoma.test.instructure.com/courses/86005/quizzes/12345/edit",
        "https://uautonoma.test.instructure.com/courses/86005/files/67890/download",
        "https://uautonoma.test.instructure.com/courses/86005/grades",
        "https://uautonoma.test.instructure.com/courses/86005",
        "https://uautonoma.test.instructure.com/api/v1/courses/86005/enrollments",
    ]

    for url in test_urls:
        info = parse_canvas_url(url)
        category = categorize_page_view(url)
        print(f"\nURL: {url}")
        print(f"  Course ID: {info['course_id']}")
        print(f"  Resource: {info['resource_type']}")
        print(f"  Resource ID: {info['resource_id']}")
        print(f"  Action: {info['action']}")
        print(f"  Is API: {info['is_api']}")
        print(f"  Category: {category}")
