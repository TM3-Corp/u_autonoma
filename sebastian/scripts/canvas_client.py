"""
Canvas API Client with Rate Limiting and Thread Safety

This module provides a robust, thread-safe API client for Canvas LMS that:
- Monitors rate limits via X-Rate-Limit-Remaining header
- Implements adaptive delays based on remaining quota
- Supports exponential backoff on failures
- Provides thread-safe shared state management
- Uses bookmark-based pagination (Canvas standard)

Usage:
    from scripts.discovery.canvas_client import CanvasClient

    client = CanvasClient()
    enrollments = client.get_enrollments(course_id=86005)
"""

import os
import re
import time
import logging
import threading
from typing import Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when rate limit is critically low."""
    pass


class APIError(Exception):
    """Raised when API request fails after retries."""
    pass


@dataclass
class RateLimitState:
    """Thread-safe rate limit state tracker."""
    remaining: int = 700  # Canvas default bucket size
    last_updated: datetime = field(default_factory=datetime.now)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, remaining: int):
        """Thread-safe update of remaining quota."""
        with self.lock:
            self.remaining = remaining
            self.last_updated = datetime.now()

    def get(self) -> int:
        """Thread-safe read of remaining quota."""
        with self.lock:
            return self.remaining


class CanvasClient:
    """
    Thread-safe Canvas API client with adaptive rate limiting.

    Attributes:
        api_url: Canvas API base URL
        headers: Authorization headers
        rate_state: Shared rate limit state
        min_quota: Minimum quota before stopping (safety threshold)
        max_workers: Maximum concurrent threads allowed
    """

    # Rate limit thresholds (Canvas has 700 bucket capacity)
    QUOTA_CRITICAL = 50      # Stop all requests
    QUOTA_LOW = 150          # Heavy throttling
    QUOTA_MODERATE = 300     # Moderate throttling
    QUOTA_COMFORTABLE = 500  # Light throttling

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        min_quota: int = 100,
        max_workers: int = 5
    ):
        """
        Initialize Canvas client.

        Args:
            api_url: Canvas API URL (defaults to env CANVAS_API_URL)
            api_token: API token (defaults to env CANVAS_API_TOKEN)
            min_quota: Minimum quota before stopping requests
            max_workers: Maximum concurrent workers for threading
        """
        self.api_url = api_url or os.getenv('CANVAS_API_URL')
        api_token = api_token or os.getenv('CANVAS_API_TOKEN')

        if not self.api_url:
            raise ValueError("CANVAS_API_URL not found in environment")
        if not api_token:
            raise ValueError("CANVAS_API_TOKEN not found in environment")

        self.headers = {'Authorization': f'Bearer {api_token}'}
        self.rate_state = RateLimitState()
        self.min_quota = min_quota
        self.max_workers = max_workers

        # Request statistics
        self._stats_lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0

        logger.info(f"CanvasClient initialized for {self.api_url}")

    def calculate_delay(self, remaining: Optional[int] = None) -> float:
        """
        Calculate adaptive delay based on remaining quota.

        Uses conservative thresholds to prevent quota exhaustion.

        Args:
            remaining: Explicit remaining quota (uses cached if None)

        Returns:
            Delay in seconds before next request
        """
        quota = remaining if remaining is not None else self.rate_state.get()

        if quota <= self.QUOTA_CRITICAL:
            return 10.0  # Emergency slowdown
        elif quota <= self.QUOTA_LOW:
            return 2.0   # Heavy throttling
        elif quota <= self.QUOTA_MODERATE:
            return 0.5   # Moderate throttling
        elif quota <= self.QUOTA_COMFORTABLE:
            return 0.2   # Light throttling
        else:
            return 0.1   # Normal operation

    def safe_get(
        self,
        url: str,
        params: Optional[Dict] = None,
        max_retries: int = 3,
        timeout: int = 30
    ) -> Optional[requests.Response]:
        """
        Thread-safe GET request with rate limit handling and retries.

        Args:
            url: Full URL to request
            params: Query parameters
            max_retries: Number of retry attempts
            timeout: Request timeout in seconds

        Returns:
            Response object or None on failure

        Raises:
            RateLimitError: If quota falls below critical threshold
        """
        # Check quota before requesting
        current_quota = self.rate_state.get()
        if current_quota < self.min_quota:
            raise RateLimitError(
                f"Rate limit too low: {current_quota} < {self.min_quota}. "
                "Stopping to preserve quota."
            )

        # Calculate delay based on current quota
        delay = self.calculate_delay(current_quota)
        if delay > 0:
            time.sleep(delay)

        last_error = None

        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=timeout
                )

                # Update rate limit state from headers
                remaining = response.headers.get('X-Rate-Limit-Remaining')
                if remaining:
                    self.rate_state.update(int(float(remaining)))

                # Track statistics
                with self._stats_lock:
                    self._request_count += 1

                # Handle rate limiting (403)
                if response.status_code == 403:
                    logger.warning(
                        f"Rate limited (403). Remaining: {remaining}. "
                        f"Attempt {attempt + 1}/{max_retries}"
                    )
                    time.sleep(2 ** (attempt + 1))  # Exponential backoff
                    continue

                # Success
                if response.status_code == 200:
                    return response

                # Client error (don't retry)
                if 400 <= response.status_code < 500:
                    logger.warning(f"Client error {response.status_code}: {url}")
                    return None

                # Server error (retry)
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    f"Server error on attempt {attempt + 1}/{max_retries}: "
                    f"{response.status_code}"
                )
                time.sleep(2 ** attempt)

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                logger.warning(f"Timeout on attempt {attempt + 1}/{max_retries}")
                time.sleep(2 ** attempt)

            except requests.exceptions.RequestException as e:
                last_error = str(e)
                logger.warning(f"Request error: {e}")
                time.sleep(2 ** attempt)

        # All retries failed
        with self._stats_lock:
            self._error_count += 1

        logger.error(f"Request failed after {max_retries} retries: {last_error}")
        return None

    def paginate(
        self,
        url: str,
        params: Optional[Dict] = None,
        max_pages: int = 50,
        per_page: int = 100
    ) -> List[Dict]:
        """
        Paginate through Canvas API results using Link header.

        Canvas uses bookmark-based pagination - the 'next' URL in the
        Link header contains the bookmark token.

        Args:
            url: Base API URL
            params: Initial query parameters
            max_pages: Maximum pages to fetch (safety limit)
            per_page: Records per page (max 100)

        Returns:
            List of all records from all pages
        """
        all_results = []
        params = params or {}
        params['per_page'] = min(per_page, 100)

        current_url = url
        page_count = 0
        first_request = True

        while current_url and page_count < max_pages:
            # Make request
            if first_request:
                response = self.safe_get(current_url, params)
                first_request = False
            else:
                response = self.safe_get(current_url)

            if not response:
                break

            # Parse response
            try:
                data = response.json()
            except ValueError:
                logger.error("Invalid JSON response")
                break

            # Handle empty response
            if not data:
                break

            # Handle dict response (some endpoints return {key: [...]})
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        data = value
                        break
                else:
                    data = [data]

            all_results.extend(data)
            page_count += 1

            # Extract next URL from Link header
            link_header = response.headers.get('Link', '')
            next_match = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
            current_url = next_match.group(1) if next_match else None

        return all_results

    # ==========================================================================
    # Convenience methods for common endpoints
    # ==========================================================================

    def get_enrollments(
        self,
        course_id: int,
        include_grades: bool = True
    ) -> List[Dict]:
        """Get all student enrollments for a course."""
        params = {
            'type[]': 'StudentEnrollment',
            'per_page': 100
        }
        if include_grades:
            params['include[]'] = ['grades', 'total_scores']

        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/enrollments",
            params
        )

    def get_assignments(self, course_id: int) -> List[Dict]:
        """Get all assignments for a course."""
        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/assignments"
        )

    def get_modules(self, course_id: int) -> List[Dict]:
        """Get all modules for a course."""
        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/modules"
        )

    def get_quizzes(self, course_id: int) -> List[Dict]:
        """Get all quizzes for a course."""
        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/quizzes"
        )

    def get_files(self, course_id: int) -> List[Dict]:
        """Get all files for a course."""
        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/files"
        )

    def get_discussions(self, course_id: int) -> List[Dict]:
        """Get all discussion topics for a course."""
        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/discussion_topics"
        )

    def get_student_summaries(self, course_id: int) -> List[Dict]:
        """Get activity summaries for all students in a course."""
        return self.paginate(
            f"{self.api_url}/api/v1/courses/{course_id}/analytics/student_summaries"
        )

    def get_course(self, course_id: int) -> Optional[Dict]:
        """Get course details."""
        response = self.safe_get(
            f"{self.api_url}/api/v1/courses/{course_id}",
            params={'include[]': ['total_students', 'term']}
        )
        return response.json() if response else None

    def get_account_courses(
        self,
        account_id: int,
        term_id: Optional[int] = None,
        with_enrollments: bool = True,
        include_students: bool = True
    ) -> List[Dict]:
        """Get all courses for an account with student counts."""
        params = {'per_page': 100}
        if term_id:
            params['enrollment_term_id'] = term_id
        if with_enrollments:
            params['with_enrollments'] = 'true'
        if include_students:
            params['include[]'] = ['total_students', 'term']

        return self.paginate(
            f"{self.api_url}/api/v1/accounts/{account_id}/courses",
            params
        )

    def get_sub_accounts(self, account_id: int) -> List[Dict]:
        """Get all sub-accounts for an account."""
        return self.paginate(
            f"{self.api_url}/api/v1/accounts/{account_id}/sub_accounts",
            {'recursive': 'false'}
        )

    # ==========================================================================
    # Statistics and diagnostics
    # ==========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        with self._stats_lock:
            return {
                'request_count': self._request_count,
                'error_count': self._error_count,
                'current_quota': self.rate_state.get(),
                'last_quota_update': self.rate_state.last_updated.isoformat()
            }

    def check_quota(self) -> Tuple[int, str]:
        """
        Check current quota and return status.

        Returns:
            Tuple of (remaining_quota, status_message)
        """
        quota = self.rate_state.get()

        if quota <= self.QUOTA_CRITICAL:
            status = "CRITICAL - Stopping requests"
        elif quota <= self.QUOTA_LOW:
            status = "LOW - Heavy throttling active"
        elif quota <= self.QUOTA_MODERATE:
            status = "MODERATE - Throttling active"
        elif quota <= self.QUOTA_COMFORTABLE:
            status = "COMFORTABLE - Light throttling"
        else:
            status = "OK - Normal operation"

        return quota, status

    def log_quota(self):
        """Log current quota status."""
        quota, status = self.check_quota()
        logger.info(f"Rate limit quota: {quota} - {status}")


# Module-level singleton for convenience
_client: Optional[CanvasClient] = None


def get_client() -> CanvasClient:
    """Get or create singleton client instance."""
    global _client
    if _client is None:
        _client = CanvasClient()
    return _client


if __name__ == '__main__':
    # Test the client
    client = CanvasClient()

    print("Testing CanvasClient...")
    print("=" * 50)

    # Check quota
    quota, status = client.check_quota()
    print(f"Initial quota: {quota} - {status}")

    # Test with a known course
    test_course_id = 86005

    print(f"\nFetching enrollments for course {test_course_id}...")
    enrollments = client.get_enrollments(test_course_id)
    print(f"Result: {len(enrollments)} enrollments")

    # Check quota after requests
    quota, status = client.check_quota()
    print(f"\nQuota after requests: {quota} - {status}")

    # Show stats
    stats = client.get_stats()
    print(f"\nClient stats: {stats}")
