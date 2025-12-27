#!/usr/bin/env python3
"""
Section 7 Refactor: Fast Multi-Threaded Course Analysis with Prediction Scoring

This script performs comprehensive analysis of Canvas LMS courses to identify
those with the best potential for building predictive models (early warning systems).

Features:
- Self-contained (no notebook dependencies)
- Multi-threaded I/O with ThreadPoolExecutor
- Adaptive rate-limit monitoring with automatic throttling
- Composite "Prediction Potential Score" ranking
- Efficient batch processing of ~3000+ courses

Usage:
    python section7_refactor.py [--account-id 176] [--max-courses 500] [--workers 5]

Output:
    - Console summary with top courses
    - data/discovery/course_analysis_results.parquet (if pandas available)
    - data/discovery/course_analysis_results.csv (always)
"""

import os
import sys
import time
import logging
import argparse
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

# Load environment
load_dotenv()

# API Configuration
API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')

if not API_URL or not API_TOKEN:
    print("ERROR: CANVAS_API_URL and CANVAS_API_TOKEN must be set in .env")
    sys.exit(1)

HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

# Prediction Model Thresholds
PASS_THRESHOLD = 57.0           # Chilean 4.0/7.0 scale
MIN_STUDENTS = 15               # Minimum for statistical validity
MIN_GRADE_VARIANCE = 10.0       # Minimum std dev for useful prediction
IDEAL_FAIL_RATE_MIN = 0.15      # 15% minimum failure for class balance
IDEAL_FAIL_RATE_MAX = 0.85      # 85% maximum (need some passes too)
MIN_GRADED_ASSIGNMENTS = 2      # Minimum assignments with grades

# Rate Limit Thresholds (Canvas has 700 bucket capacity)
QUOTA_CRITICAL = 50             # Stop all requests
QUOTA_LOW = 100                 # Heavy throttling (2s delay)
QUOTA_MODERATE = 200            # Moderate throttling (0.5s delay)
QUOTA_COMFORTABLE = 350         # Light throttling (0.1s delay)

# Threading Configuration
DEFAULT_MAX_WORKERS = 5
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RateLimitState:
    """Thread-safe rate limit tracker."""
    remaining: int = 700
    last_updated: datetime = field(default_factory=datetime.now)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update(self, remaining: int):
        with self._lock:
            self.remaining = remaining
            self.last_updated = datetime.now()

    def get(self) -> int:
        with self._lock:
            return self.remaining


@dataclass
class CourseMetrics:
    """Comprehensive metrics for a single course."""
    # Identifiers
    course_id: int = 0
    course_name: str = ""
    account_id: int = 0
    term_id: int = 0
    term_name: str = ""

    # Enrollment Metrics
    total_students: int = 0
    active_students: int = 0
    completed_students: int = 0
    inactive_students: int = 0

    # Grade Availability
    students_with_current_score: int = 0
    students_with_final_score: int = 0
    current_score_coverage: float = 0.0
    final_score_coverage: float = 0.0

    # Grade Statistics
    grade_mean: float = 0.0
    grade_std: float = 0.0
    grade_min: float = 0.0
    grade_max: float = 0.0
    grade_median: float = 0.0

    # Failure Analysis
    failing_count: int = 0
    passing_count: int = 0
    failure_rate: float = 0.0

    # Instructional Design Counts
    assignment_count: int = 0
    graded_assignment_count: int = 0
    quiz_count: int = 0
    module_count: int = 0
    file_count: int = 0
    discussion_count: int = 0
    page_count: int = 0

    # Activity Metrics
    total_page_views: int = 0
    total_participations: int = 0
    students_with_activity: int = 0
    avg_page_views: float = 0.0
    avg_participations: float = 0.0

    # Submission Metrics
    total_submissions: int = 0
    graded_submissions: int = 0
    submission_rate: float = 0.0

    # Computed Scores (0-100 scale)
    grade_availability_score: float = 0.0
    grade_variance_score: float = 0.0
    class_balance_score: float = 0.0
    design_richness_score: float = 0.0
    activity_score: float = 0.0
    prediction_potential_score: float = 0.0

    # Metadata
    analysis_timestamp: str = ""
    fetch_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# RATE-LIMITED API CLIENT
# =============================================================================

class RateLimitedClient:
    """Thread-safe API client with adaptive rate limiting."""

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS):
        self.api_url = API_URL
        self.headers = HEADERS
        self.rate_state = RateLimitState()
        self.max_workers = max_workers

        # Statistics
        self._stats_lock = threading.Lock()
        self._request_count = 0
        self._error_count = 0
        self._throttle_count = 0

    def _calculate_delay(self) -> float:
        """Calculate adaptive delay based on remaining quota."""
        quota = self.rate_state.get()

        if quota <= QUOTA_CRITICAL:
            return 10.0  # Emergency stop
        elif quota <= QUOTA_LOW:
            return 2.0   # Heavy throttle
        elif quota <= QUOTA_MODERATE:
            return 0.5   # Moderate throttle
        elif quota <= QUOTA_COMFORTABLE:
            return 0.15  # Light throttle
        else:
            return 0.05  # Minimal delay

    def _check_quota(self) -> bool:
        """Check if we should proceed with requests. Auto-recovers on low quota."""
        quota = self.rate_state.get()

        if quota <= QUOTA_CRITICAL:
            logger.warning(f"CRITICAL: Rate limit at {quota}. Pausing 60s for recovery...")
            time.sleep(60)
            # Reset quota estimate after waiting
            self.rate_state.update(200)
            return True  # Try again after recovery

        if quota <= QUOTA_LOW:
            logger.info(f"LOW quota ({quota}). Pausing 30s for recovery...")
            time.sleep(30)
            self.rate_state.update(300)
            return True

        return True

    def wait_for_recovery(self, target_quota: int = 400, max_wait: int = 180):
        """Wait until quota recovers to target level."""
        logger.info(f"Waiting for quota recovery to {target_quota}...")
        waited = 0
        while waited < max_wait:
            time.sleep(10)
            waited += 10
            # Canvas refills ~10 req/sec, so after 30s we should have ~300 more
            estimated = min(700, self.rate_state.get() + 100)
            self.rate_state.update(estimated)
            if estimated >= target_quota:
                logger.info(f"Quota recovered to ~{estimated}")
                return True
        logger.warning(f"Max wait reached, continuing with estimated quota")
        return True

    def safe_get(
        self,
        url: str,
        params: Optional[Dict] = None,
        timeout: int = REQUEST_TIMEOUT
    ) -> Optional[requests.Response]:
        """Thread-safe GET with rate limiting and retries."""

        # Check quota before proceeding
        if not self._check_quota():
            return None

        # Apply adaptive delay
        delay = self._calculate_delay()
        if delay > 0.1:
            with self._stats_lock:
                self._throttle_count += 1
        time.sleep(delay)

        last_error = None

        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(
                    url,
                    headers=self.headers,
                    params=params,
                    timeout=timeout
                )

                # Update rate limit from headers
                remaining = response.headers.get('X-Rate-Limit-Remaining')
                if remaining:
                    self.rate_state.update(int(float(remaining)))

                with self._stats_lock:
                    self._request_count += 1

                # Handle rate limit response
                if response.status_code == 403:
                    logger.warning(f"Rate limited (403). Backing off...")
                    time.sleep(2 ** (attempt + 2))
                    continue

                if response.status_code == 200:
                    return response

                # Client errors (don't retry)
                if 400 <= response.status_code < 500:
                    return None

                # Server errors (retry)
                last_error = f"HTTP {response.status_code}"
                time.sleep(2 ** attempt)

            except requests.exceptions.Timeout:
                last_error = "Timeout"
                time.sleep(2 ** attempt)
            except requests.exceptions.RequestException as e:
                last_error = str(e)
                time.sleep(2 ** attempt)

        with self._stats_lock:
            self._error_count += 1

        return None

    def paginate(
        self,
        url: str,
        params: Optional[Dict] = None,
        max_pages: int = 20,
        per_page: int = 100
    ) -> List[Dict]:
        """Paginate through Canvas API results."""
        import re

        all_results = []
        params = params or {}
        params['per_page'] = min(per_page, 100)

        current_url = url
        page_count = 0
        first_request = True

        while current_url and page_count < max_pages:
            if first_request:
                response = self.safe_get(current_url, params)
                first_request = False
            else:
                response = self.safe_get(current_url)

            if not response:
                break

            try:
                data = response.json()
            except ValueError:
                break

            if not data:
                break

            # Handle dict response
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

    def get_stats(self) -> Dict:
        """Get client statistics."""
        with self._stats_lock:
            return {
                'requests': self._request_count,
                'errors': self._error_count,
                'throttles': self._throttle_count,
                'quota': self.rate_state.get()
            }


# =============================================================================
# COURSE ANALYZER
# =============================================================================

class CourseAnalyzer:
    """Analyzes individual courses for prediction potential."""

    def __init__(self, client: RateLimitedClient):
        self.client = client

    def analyze_course(self, course: Dict) -> CourseMetrics:
        """Perform comprehensive analysis of a single course."""
        metrics = CourseMetrics(
            course_id=course.get('id', 0),
            course_name=course.get('name', '')[:80],
            account_id=course.get('account_id', 0),
            term_id=course.get('enrollment_term_id', 0),
            term_name=course.get('term', {}).get('name', '') if course.get('term') else '',
            total_students=course.get('total_students', 0),
            analysis_timestamp=datetime.now().isoformat()
        )

        course_id = metrics.course_id

        # Skip courses with too few students
        if metrics.total_students < 5:
            return metrics

        # 1. Fetch enrollments with grades
        self._analyze_enrollments(course_id, metrics)

        # 2. Fetch instructional design elements (parallel-safe counts only)
        self._analyze_design_elements(course_id, metrics)

        # 3. Fetch activity summaries if we have enrolled students
        if metrics.active_students > 0:
            self._analyze_activity(course_id, metrics)

        # 4. Calculate composite scores
        self._calculate_scores(metrics)

        return metrics

    def _analyze_enrollments(self, course_id: int, metrics: CourseMetrics):
        """Analyze enrollment and grade data."""
        enrollments = self.client.paginate(
            f"{self.client.api_url}/api/v1/courses/{course_id}/enrollments",
            params={
                'type[]': 'StudentEnrollment',
                'include[]': ['grades', 'total_scores'],
                'per_page': 100
            },
            max_pages=10
        )

        if not enrollments:
            metrics.fetch_errors.append("enrollments")
            return

        # Enrollment state counts
        current_scores = []
        final_scores = []

        for e in enrollments:
            state = e.get('enrollment_state', '')
            if state == 'active':
                metrics.active_students += 1
            elif state == 'completed':
                metrics.completed_students += 1
            elif state == 'inactive':
                metrics.inactive_students += 1

            grades = e.get('grades', {})
            if grades:
                cs = grades.get('current_score')
                fs = grades.get('final_score')

                if cs is not None:
                    current_scores.append(float(cs))
                if fs is not None:
                    final_scores.append(float(fs))

        # Grade availability
        total = len(enrollments)
        metrics.students_with_current_score = len(current_scores)
        metrics.students_with_final_score = len(final_scores)
        metrics.current_score_coverage = (len(current_scores) / total * 100) if total > 0 else 0
        metrics.final_score_coverage = (len(final_scores) / total * 100) if total > 0 else 0

        # Grade statistics (use current_score as primary)
        scores = current_scores if current_scores else final_scores
        if len(scores) >= 3:
            import statistics
            metrics.grade_mean = statistics.mean(scores)
            metrics.grade_std = statistics.stdev(scores) if len(scores) > 1 else 0
            metrics.grade_min = min(scores)
            metrics.grade_max = max(scores)
            metrics.grade_median = statistics.median(scores)

            # Failure analysis
            metrics.failing_count = sum(1 for s in scores if s < PASS_THRESHOLD)
            metrics.passing_count = sum(1 for s in scores if s >= PASS_THRESHOLD)
            metrics.failure_rate = metrics.failing_count / len(scores) if scores else 0

    def _analyze_design_elements(self, course_id: int, metrics: CourseMetrics):
        """Analyze instructional design elements (counts only for speed)."""
        base_url = f"{self.client.api_url}/api/v1/courses/{course_id}"

        # Assignments
        assignments = self.client.paginate(f"{base_url}/assignments", max_pages=5)
        if assignments:
            metrics.assignment_count = len(assignments)
            metrics.graded_assignment_count = sum(
                1 for a in assignments
                if a.get('grading_type') != 'not_graded' and (a.get('points_possible') or 0) > 0
            )

        # Quick counts via single-page requests
        endpoints = [
            ('quizzes', 'quiz_count'),
            ('modules', 'module_count'),
            ('files', 'file_count'),
            ('discussion_topics', 'discussion_count'),
            ('pages', 'page_count'),
        ]

        for endpoint, attr in endpoints:
            response = self.client.safe_get(
                f"{base_url}/{endpoint}",
                params={'per_page': 100}
            )
            if response:
                try:
                    data = response.json()
                    if isinstance(data, list):
                        setattr(metrics, attr, len(data))
                except ValueError:
                    pass

    def _analyze_activity(self, course_id: int, metrics: CourseMetrics):
        """Analyze student activity summaries."""
        summaries = self.client.paginate(
            f"{self.client.api_url}/api/v1/courses/{course_id}/analytics/student_summaries",
            max_pages=5
        )

        if not summaries:
            return

        total_views = 0
        total_participations = 0
        with_activity = 0

        for s in summaries:
            views = s.get('page_views', 0) or 0
            parts = s.get('participations', 0) or 0

            total_views += views
            total_participations += parts

            if views > 0 or parts > 0:
                with_activity += 1

        metrics.total_page_views = total_views
        metrics.total_participations = total_participations
        metrics.students_with_activity = with_activity

        n = len(summaries)
        if n > 0:
            metrics.avg_page_views = total_views / n
            metrics.avg_participations = total_participations / n

    def _calculate_scores(self, metrics: CourseMetrics):
        """Calculate composite prediction potential scores."""

        # 1. Grade Availability Score (0-100)
        # Weight current_score higher as it's more useful for early warning
        grade_avail = (metrics.current_score_coverage * 0.7 +
                       metrics.final_score_coverage * 0.3)
        metrics.grade_availability_score = min(100, grade_avail)

        # 2. Grade Variance Score (0-100)
        # Higher variance = better for prediction, cap at 30% std dev
        if metrics.grade_std > 0:
            variance_score = min(100, (metrics.grade_std / 30) * 100)
        else:
            variance_score = 0
        metrics.grade_variance_score = variance_score

        # 3. Class Balance Score (0-100)
        # Ideal failure rate between 15-85%, peak at 50%
        if metrics.failure_rate > 0:
            if IDEAL_FAIL_RATE_MIN <= metrics.failure_rate <= IDEAL_FAIL_RATE_MAX:
                # Score based on distance from 50%
                distance_from_ideal = abs(metrics.failure_rate - 0.5)
                balance_score = max(0, 100 - (distance_from_ideal * 200))
            else:
                balance_score = 10  # Some points for having any variance
        else:
            balance_score = 0
        metrics.class_balance_score = balance_score

        # 4. Design Richness Score (0-100)
        # Based on variety of instructional elements
        design_elements = [
            min(20, metrics.assignment_count * 2),           # Up to 20 points
            min(15, metrics.graded_assignment_count * 3),    # Up to 15 points
            min(15, metrics.quiz_count * 3),                 # Up to 15 points
            min(15, metrics.module_count * 5),               # Up to 15 points
            min(15, metrics.file_count * 0.5),               # Up to 15 points
            min(10, metrics.discussion_count * 2),           # Up to 10 points
            min(10, metrics.page_count * 1),                 # Up to 10 points
        ]
        metrics.design_richness_score = min(100, sum(design_elements))

        # 5. Activity Score (0-100)
        # Based on engagement levels
        if metrics.students_with_activity > 0 and metrics.active_students > 0:
            activity_coverage = metrics.students_with_activity / metrics.active_students
            avg_engagement = min(1, metrics.avg_page_views / 100)  # Normalize to 100 views
            metrics.activity_score = min(100, (activity_coverage * 50 + avg_engagement * 50))
        else:
            metrics.activity_score = 0

        # 6. COMPOSITE: Prediction Potential Score (0-100)
        # Weighted combination of all factors
        weights = {
            'grade_availability': 0.30,    # Most important - need grades to predict
            'grade_variance': 0.25,        # Need variance to build models
            'class_balance': 0.20,         # Need both passes and failures
            'design_richness': 0.15,       # Good design = more features
            'activity': 0.10,              # Activity helps but not required
        }

        # Minimum thresholds - zero out if basic requirements not met
        if metrics.students_with_current_score < MIN_STUDENTS:
            metrics.prediction_potential_score = 0
            return

        if metrics.grade_std < MIN_GRADE_VARIANCE:
            metrics.prediction_potential_score = max(10, metrics.grade_availability_score * 0.2)
            return

        composite = (
            metrics.grade_availability_score * weights['grade_availability'] +
            metrics.grade_variance_score * weights['grade_variance'] +
            metrics.class_balance_score * weights['class_balance'] +
            metrics.design_richness_score * weights['design_richness'] +
            metrics.activity_score * weights['activity']
        )

        # Bonus for meeting all key criteria
        if (metrics.graded_assignment_count >= MIN_GRADED_ASSIGNMENTS and
            metrics.students_with_activity >= MIN_STUDENTS):
            composite = min(100, composite * 1.1)

        metrics.prediction_potential_score = round(composite, 1)


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class BatchProcessor:
    """Processes courses in parallel with progress tracking."""

    def __init__(self, client: RateLimitedClient, max_workers: int = DEFAULT_MAX_WORKERS):
        self.client = client
        self.analyzer = CourseAnalyzer(client)
        self.max_workers = max_workers

        # Progress tracking
        self._progress_lock = threading.Lock()
        self._processed = 0
        self._total = 0
        self._start_time = None

    def discover_courses(
        self,
        account_id: int,
        term_id: Optional[int] = None,
        max_courses: Optional[int] = None,
        auto_subaccounts: bool = True
    ) -> List[Dict]:
        """Discover all courses under an account (with automatic sub-account fallback)."""
        logger.info(f"Discovering courses under account {account_id}...")

        params = {
            'with_enrollments': 'true',
            'include[]': ['total_students', 'term'],
            'per_page': 100
        }
        if term_id:
            params['enrollment_term_id'] = term_id

        # Get courses directly from account
        courses = self.client.paginate(
            f"{self.client.api_url}/api/v1/accounts/{account_id}/courses",
            params=params,
            max_pages=50
        )

        # Filter to courses with students
        courses = [c for c in courses if c.get('total_students', 0) >= 5]

        # If no courses found, try sub-accounts automatically
        if len(courses) == 0 and auto_subaccounts:
            logger.info(f"No direct courses found. Checking sub-accounts...")
            sub_accounts = self.discover_all_subaccounts(account_id)

            for sub_id in sub_accounts[1:]:  # Skip root (already checked)
                sub_courses = self.client.paginate(
                    f"{self.client.api_url}/api/v1/accounts/{sub_id}/courses",
                    params=params,
                    max_pages=30
                )
                sub_courses = [c for c in sub_courses if c.get('total_students', 0) >= 5]
                courses.extend(sub_courses)
                logger.info(f"  Account {sub_id}: {len(sub_courses)} courses")

                # Check quota
                if self.client.rate_state.get() < QUOTA_LOW:
                    logger.warning("Rate limit low, stopping sub-account discovery")
                    break

                if max_courses and len(courses) >= max_courses:
                    break

        if max_courses:
            courses = courses[:max_courses]

        logger.info(f"Found {len(courses)} courses with >= 5 students")
        return courses

    def discover_all_subaccounts(self, root_account_id: int) -> List[int]:
        """Recursively discover all sub-account IDs."""
        logger.info(f"Discovering sub-accounts under {root_account_id}...")

        all_accounts = [root_account_id]
        to_process = [root_account_id]

        while to_process:
            account_id = to_process.pop(0)
            subs = self.client.paginate(
                f"{self.client.api_url}/api/v1/accounts/{account_id}/sub_accounts",
                params={'recursive': 'false'},
                max_pages=10
            )

            for sub in subs:
                sub_id = sub.get('id')
                if sub_id and sub_id not in all_accounts:
                    all_accounts.append(sub_id)
                    to_process.append(sub_id)

        logger.info(f"Found {len(all_accounts)} total accounts")
        return all_accounts

    def process_courses(self, courses: List[Dict]) -> List[CourseMetrics]:
        """Process courses in parallel with progress tracking."""
        self._total = len(courses)
        self._processed = 0
        self._start_time = time.time()

        results = []

        logger.info(f"Analyzing {self._total} courses with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_course = {
                executor.submit(self._analyze_with_progress, course): course
                for course in courses
            }

            # Collect results as they complete
            for future in as_completed(future_to_course):
                try:
                    metrics = future.result()
                    if metrics:
                        results.append(metrics)
                except Exception as e:
                    course = future_to_course[future]
                    logger.error(f"Error analyzing course {course.get('id')}: {e}")

        elapsed = time.time() - self._start_time
        logger.info(f"Completed {len(results)} courses in {elapsed:.1f}s "
                   f"({len(results)/elapsed:.1f} courses/sec)")

        return results

    def _analyze_with_progress(self, course: Dict) -> Optional[CourseMetrics]:
        """Analyze course with progress tracking."""
        try:
            metrics = self.analyzer.analyze_course(course)

            with self._progress_lock:
                self._processed += 1
                if self._processed % 25 == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._processed / elapsed if elapsed > 0 else 0
                    quota = self.client.rate_state.get()
                    logger.info(
                        f"Progress: {self._processed}/{self._total} "
                        f"({rate:.1f}/sec) | Quota: {quota}"
                    )

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing course {course.get('id')}: {e}")
            return None


# =============================================================================
# RESULTS FORMATTER
# =============================================================================

class ResultsFormatter:
    """Formats and saves analysis results."""

    @staticmethod
    def to_dataframe(results: List[CourseMetrics]):
        """Convert results to pandas DataFrame."""
        try:
            import pandas as pd

            data = [m.to_dict() for m in results]
            df = pd.DataFrame(data)

            # Sort by prediction potential
            df = df.sort_values('prediction_potential_score', ascending=False)

            return df
        except ImportError:
            logger.warning("pandas not available, returning raw results")
            return results

    @staticmethod
    def print_summary(results: List[CourseMetrics], top_n: int = 20):
        """Print formatted summary to console."""

        # Sort by prediction potential
        sorted_results = sorted(
            results,
            key=lambda x: x.prediction_potential_score,
            reverse=True
        )

        print("\n" + "=" * 100)
        print("COURSE ANALYSIS SUMMARY")
        print("=" * 100)

        # Overall statistics
        total = len(results)
        with_grades = sum(1 for r in results if r.students_with_current_score >= MIN_STUDENTS)
        high_potential = sum(1 for r in results if r.prediction_potential_score >= 50)

        print(f"""
OVERALL STATISTICS
──────────────────
  Total Courses Analyzed:     {total}
  Courses with Valid Grades:  {with_grades} ({with_grades/total*100:.1f}%)
  High Potential (score≥50):  {high_potential} ({high_potential/total*100:.1f}%)

THRESHOLDS USED
───────────────
  Pass Threshold:             {PASS_THRESHOLD}%
  Min Students:               {MIN_STUDENTS}
  Min Grade Variance:         {MIN_GRADE_VARIANCE}%
  Ideal Failure Rate:         {IDEAL_FAIL_RATE_MIN*100:.0f}%-{IDEAL_FAIL_RATE_MAX*100:.0f}%
""")

        # Top courses table
        print(f"\nTOP {top_n} COURSES BY PREDICTION POTENTIAL")
        print("─" * 100)
        print(f"{'Rank':<5} {'ID':<8} {'Course Name':<35} {'Students':<10} "
              f"{'Grade%':<8} {'Var%':<7} {'Fail%':<7} {'Score':<7}")
        print("─" * 100)

        for i, m in enumerate(sorted_results[:top_n], 1):
            name = m.course_name[:33] + ".." if len(m.course_name) > 35 else m.course_name
            print(f"{i:<5} {m.course_id:<8} {name:<35} "
                  f"{m.students_with_current_score:<10} "
                  f"{m.current_score_coverage:<8.1f} "
                  f"{m.grade_std:<7.1f} "
                  f"{m.failure_rate*100:<7.1f} "
                  f"{m.prediction_potential_score:<7.1f}")

        print("─" * 100)

        # Score distribution
        print("\nSCORE DISTRIBUTION")
        print("──────────────────")
        ranges = [(80, 100), (60, 80), (40, 60), (20, 40), (0, 20)]
        for low, high in ranges:
            count = sum(1 for r in results if low <= r.prediction_potential_score < high)
            bar = "█" * (count // 5) if count > 0 else ""
            print(f"  {low:3d}-{high:3d}: {count:4d} {bar}")

        print("\n" + "=" * 100)

    @staticmethod
    def save_results(results: List[CourseMetrics], output_dir: str = "data/discovery"):
        """Save results to files (Parquet if available, always CSV)."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = []

        try:
            import pandas as pd

            df = ResultsFormatter.to_dataframe(results)

            # Always save CSV (human readable)
            csv_path = os.path.join(output_dir, f"course_analysis_{timestamp}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved: {csv_path}")
            saved_files.append(csv_path)

            # Also save latest symlink-style copy
            latest_csv = os.path.join(output_dir, "course_analysis_latest.csv")
            df.to_csv(latest_csv, index=False)

            # Try Parquet (more efficient for large datasets)
            try:
                parquet_path = os.path.join(output_dir, f"course_analysis_{timestamp}.parquet")
                df.to_parquet(parquet_path, index=False)
                logger.info(f"Saved: {parquet_path}")
                saved_files.append(parquet_path)
            except Exception as e:
                logger.warning(f"Parquet save skipped (install pyarrow): {e}")

            return csv_path

        except ImportError:
            # Fallback to JSON if pandas not available
            import json
            json_path = os.path.join(output_dir, f"course_analysis_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump([m.to_dict() for m in results], f, indent=2, default=str)
            logger.info(f"Saved: {json_path}")
            return json_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for course analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze Canvas courses for prediction potential"
    )
    parser.add_argument(
        '--account-id', type=int, default=176,
        help='Root account ID to analyze (default: 176 = Providencia)'
    )
    parser.add_argument(
        '--campus-ids', type=str, default=None,
        help='Comma-separated list of campus IDs to analyze (e.g., "173,174,175,176")'
    )
    parser.add_argument(
        '--term-id', type=int, default=None,
        help='Filter to specific term ID'
    )
    parser.add_argument(
        '--max-courses', type=int, default=None,
        help='Maximum courses to analyze per campus'
    )
    parser.add_argument(
        '--workers', type=int, default=DEFAULT_MAX_WORKERS,
        help=f'Number of parallel workers (default: {DEFAULT_MAX_WORKERS})'
    )
    parser.add_argument(
        '--all-subaccounts', action='store_true',
        help='Discover and analyze all sub-accounts recursively'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/discovery',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Initialize client
    client = RateLimitedClient(max_workers=args.workers)
    processor = BatchProcessor(client, max_workers=args.workers)

    # Determine which accounts to process
    if args.campus_ids:
        account_ids = [int(x.strip()) for x in args.campus_ids.split(',')]
        mode = f"Multi-Campus ({len(account_ids)} campuses)"
    else:
        account_ids = [args.account_id]
        mode = f"Single Account ({args.account_id})"

    print("\n" + "=" * 70)
    print("CANVAS COURSE ANALYSIS - PREDICTION POTENTIAL SCORING")
    print("=" * 70)
    print(f"Mode:           {mode}")
    print(f"Workers:        {args.workers}")
    print(f"Max Courses:    {args.max_courses or 'Unlimited'} per campus")
    print(f"Initial Quota:  {client.rate_state.get()}")
    print("=" * 70 + "\n")

    # Collect all courses from all accounts
    all_courses = []

    for i, acc_id in enumerate(account_ids):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing account {acc_id} ({i+1}/{len(account_ids)})")
        logger.info(f"{'='*50}")

        # Check quota before processing each campus
        quota = client.rate_state.get()
        if quota < QUOTA_MODERATE:
            logger.info(f"Quota low ({quota}). Waiting for recovery before next campus...")
            client.wait_for_recovery(target_quota=400, max_wait=120)

        if args.all_subaccounts:
            sub_accounts = processor.discover_all_subaccounts(acc_id)
            for sub_id in sub_accounts[:10]:
                courses = processor.discover_courses(
                    sub_id, args.term_id,
                    max_courses=100,
                    auto_subaccounts=False
                )
                all_courses.extend(courses)
        else:
            courses = processor.discover_courses(
                acc_id,
                args.term_id,
                args.max_courses
            )
            all_courses.extend(courses)

        logger.info(f"Total courses collected so far: {len(all_courses)}")

    if not all_courses:
        logger.error("No courses found to analyze")
        return

    # Remove duplicates by course ID
    seen = set()
    unique_courses = []
    for c in all_courses:
        cid = c.get('id')
        if cid not in seen:
            seen.add(cid)
            unique_courses.append(c)

    logger.info(f"\n{'='*50}")
    logger.info(f"ANALYSIS PHASE: {len(unique_courses)} unique courses")
    logger.info(f"{'='*50}")

    # Check quota before analysis phase
    quota = client.rate_state.get()
    if quota < QUOTA_MODERATE:
        logger.info(f"Quota low ({quota}). Waiting for recovery before analysis...")
        client.wait_for_recovery(target_quota=500, max_wait=180)

    # Process courses
    results = processor.process_courses(unique_courses)

    # Print summary
    ResultsFormatter.print_summary(results)

    # Save results
    output_path = ResultsFormatter.save_results(results, args.output_dir)

    # Final statistics
    stats = client.get_stats()
    print(f"\nAPI Statistics:")
    print(f"  Total Requests: {stats['requests']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Throttle Events: {stats['throttles']}")
    print(f"  Final Quota: {stats['quota']}")

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == '__main__':
    main()
