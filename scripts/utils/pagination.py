"""
Canvas API Pagination Utility

IMPORTANT: Canvas uses BOOKMARK-based pagination, NOT page numbers.
The 'next' URL in the Link header contains the bookmark token.
Using page=2, page=3 etc. does NOT work reliably.

This module provides a robust pagination function that:
1. Follows the Link header to get next pages
2. Logs progress for visibility
3. Handles errors gracefully with retries
4. Validates response data
"""

import re
import time
import logging
import requests
from typing import Optional, Dict, List, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PaginationError(Exception):
    """Raised when pagination fails after retries."""
    pass


def paginate_canvas(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    max_pages: int = 100,
    per_page: int = 100,
    delay: float = 0.1,
    max_retries: int = 3,
    retry_delay: float = 2.0,
    log_progress: bool = True,
    log_every: int = 5
) -> List[Dict]:
    """
    Paginate through Canvas API results using Link header (bookmark-based).

    IMPORTANT: Canvas uses bookmark pagination. The 'next' URL in the Link
    header contains an encoded bookmark token. Do NOT use page=N parameters.

    Args:
        url: Base URL for the API endpoint
        headers: Request headers (must include Authorization)
        params: Initial query parameters (applied only to first request)
        max_pages: Maximum number of pages to fetch (safety limit)
        per_page: Number of records per page (max 100 for Canvas)
        delay: Delay between requests in seconds (rate limiting)
        max_retries: Number of retries on failure
        retry_delay: Delay between retries in seconds
        log_progress: Whether to log progress
        log_every: Log progress every N pages

    Returns:
        List of all records from all pages

    Raises:
        PaginationError: If pagination fails after max_retries

    Example:
        >>> results = paginate_canvas(
        ...     url=f'{API_URL}/api/v1/courses/86005/enrollments',
        ...     headers={'Authorization': f'Bearer {TOKEN}'},
        ...     params={'type[]': 'StudentEnrollment', 'include[]': 'grades'}
        ... )
        >>> print(f"Fetched {len(results)} enrollments")
    """
    all_results: List[Dict] = []
    params = params or {}
    params['per_page'] = min(per_page, 100)  # Canvas max is 100

    current_url = url
    page_count = 0
    first_request = True

    while current_url and page_count < max_pages:
        # Retry logic
        response = None
        last_error = None

        for attempt in range(max_retries):
            try:
                # Apply params only on first request
                # Subsequent requests use the full URL from Link header
                if first_request:
                    response = requests.get(
                        current_url,
                        headers=headers,
                        params=params,
                        timeout=30
                    )
                    first_request = False
                else:
                    response = requests.get(
                        current_url,
                        headers=headers,
                        timeout=30
                    )

                # Check for rate limiting
                if response.status_code == 403:
                    rate_limit_remaining = response.headers.get('X-Rate-Limit-Remaining', '?')
                    logger.warning(f"Rate limited. Remaining: {rate_limit_remaining}")
                    time.sleep(retry_delay * (attempt + 1))
                    continue

                # Check for success
                if response.status_code == 200:
                    break
                else:
                    last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {last_error}")
                    time.sleep(retry_delay)

            except requests.exceptions.Timeout:
                last_error = "Request timeout"
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: Timeout")
                time.sleep(retry_delay)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1}/{max_retries}: {e}")
                time.sleep(retry_delay)

        # Check if we got a successful response
        if response is None or response.status_code != 200:
            error_msg = f"Pagination failed after {max_retries} retries: {last_error}"
            logger.error(error_msg)
            raise PaginationError(error_msg)

        # Parse response
        try:
            data = response.json()
        except ValueError as e:
            raise PaginationError(f"Invalid JSON response: {e}")

        # Handle empty response
        if not data:
            if log_progress:
                logger.info(f"Empty response on page {page_count + 1}, stopping")
            break

        # Handle dict response (some endpoints return {key: [...]})
        if isinstance(data, dict):
            # Find the list value in the dict
            for key, value in data.items():
                if isinstance(value, list):
                    data = value
                    break
            else:
                # No list found, wrap in list
                data = [data]

        all_results.extend(data)
        page_count += 1

        # Log progress
        if log_progress and page_count % log_every == 0:
            logger.info(f"Page {page_count}: fetched {len(all_results)} records so far")

        # Extract next URL from Link header
        # Format: <url>; rel="current", <url>; rel="next", <url>; rel="first"
        link_header = response.headers.get('Link', '')
        next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
        current_url = next_match.group(1) if next_match else None

        # Rate limiting delay
        if current_url and delay > 0:
            time.sleep(delay)

    # Final log
    if log_progress:
        logger.info(f"Pagination complete: {len(all_results)} total records in {page_count} pages")

    return all_results


def paginate_canvas_with_stats(
    url: str,
    headers: Dict[str, str],
    params: Optional[Dict[str, Any]] = None,
    **kwargs
) -> tuple[List[Dict], Dict[str, Any]]:
    """
    Same as paginate_canvas but returns pagination statistics.

    Returns:
        Tuple of (results, stats) where stats contains:
        - total_records: Number of records fetched
        - pages_fetched: Number of pages fetched
        - url: Original URL
    """
    results = paginate_canvas(url, headers, params, **kwargs)

    stats = {
        'total_records': len(results),
        'url': url,
    }

    return results, stats


# Convenience functions for common Canvas endpoints

def get_enrollments(api_url: str, headers: Dict, course_id: int, include_grades: bool = True) -> List[Dict]:
    """Get all student enrollments for a course."""
    params = {
        'type[]': 'StudentEnrollment',
        'per_page': 100
    }
    if include_grades:
        params['include[]'] = ['grades', 'total_scores']

    return paginate_canvas(
        url=f'{api_url}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params=params
    )


def get_student_summaries(api_url: str, headers: Dict, course_id: int) -> List[Dict]:
    """Get activity summaries for all students in a course."""
    return paginate_canvas(
        url=f'{api_url}/api/v1/courses/{course_id}/analytics/student_summaries',
        headers=headers
    )


def get_submissions(api_url: str, headers: Dict, course_id: int) -> List[Dict]:
    """Get all submissions for a course."""
    return paginate_canvas(
        url=f'{api_url}/api/v1/courses/{course_id}/students/submissions',
        headers=headers,
        params={'student_ids[]': 'all'}
    )


def get_assignments(api_url: str, headers: Dict, course_id: int) -> List[Dict]:
    """Get all assignments for a course."""
    return paginate_canvas(
        url=f'{api_url}/api/v1/courses/{course_id}/assignments',
        headers=headers
    )


if __name__ == '__main__':
    # Test the pagination utility
    import sys
    sys.path.insert(0, '/home/paul/projects/uautonoma/scripts')
    from config import API_URL, API_TOKEN

    headers = {'Authorization': f'Bearer {API_TOKEN}'}

    print("Testing pagination utility...")
    print("=" * 50)

    # Test with a known course
    test_course_id = 86005

    print(f"\n1. Testing get_enrollments for course {test_course_id}")
    enrollments = get_enrollments(API_URL, headers, test_course_id)
    print(f"   Result: {len(enrollments)} enrollments")

    print(f"\n2. Testing get_student_summaries for course {test_course_id}")
    summaries = get_student_summaries(API_URL, headers, test_course_id)
    print(f"   Result: {len(summaries)} student summaries")

    print(f"\n3. Testing get_submissions for course {test_course_id}")
    submissions = get_submissions(API_URL, headers, test_course_id)
    print(f"   Result: {len(submissions)} submissions")

    print("\n" + "=" * 50)
    print("Pagination utility tests complete!")
