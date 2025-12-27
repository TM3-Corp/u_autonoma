#!/usr/bin/env python3
"""
Activity-Based Course Analysis Script

Based on CANVAS_API_REFERENCE.md - Uses activity-focused endpoints to analyze courses
and identify candidates for predictive modeling.

Endpoints Used (from documentation):
1. Enrollments API - grades (current_score, final_score)
2. Student Summaries API - activity metrics (page_views, participations, tardiness_breakdown)
3. Course Activity API - daily aggregates
4. Assignment Analytics API - statistics with quartiles and tardiness
5. Recent Students API - last login times
6. Assignments API - metadata

Key Metrics:
- tardiness_breakdown (on_time, late, missing rates)
- page_views_level, participations_level (Canvas computed 1-3)
- Assignment statistics (min, max, median, quartiles)
- Student engagement patterns

Usage:
    python activity_analysis.py --campus-ids "173,174,175,176" --max-courses 100

Output:
    - data/discovery/activity_analysis_latest.csv
    - Comparison with previous course analysis
"""

import os
import sys
import time
import logging
import argparse
import threading
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION (from CANVAS_API_REFERENCE.md)
# =============================================================================

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')

if not API_URL or not API_TOKEN:
    print("ERROR: CANVAS_API_URL and CANVAS_API_TOKEN must be set in .env")
    sys.exit(1)

HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

# Chilean grading threshold (from documentation)
PASS_THRESHOLD = 57.0  # 4.0 on 1-7 scale

# Rate limiting (from documentation: 700 bucket capacity)
RATE_LIMIT_DELAY = 0.3  # Recommended delay between calls
QUOTA_CRITICAL = 50
QUOTA_LOW = 100
QUOTA_MODERATE = 200

# Minimum requirements for analysis
MIN_STUDENTS = 15
MIN_ACTIVITY_COVERAGE = 0.3  # 30% students with activity

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
class ActivityMetrics:
    """Comprehensive activity-based metrics for a course."""

    # Course identifiers
    course_id: int = 0
    course_name: str = ""
    account_id: int = 0
    term_id: int = 0
    term_name: str = ""

    # Enrollment counts
    total_students: int = 0
    active_students: int = 0

    # Grade metrics (from Enrollments API)
    students_with_grades: int = 0
    grade_coverage: float = 0.0
    grade_mean: float = 0.0
    grade_std: float = 0.0
    grade_min: float = 0.0
    grade_max: float = 0.0
    grade_median: float = 0.0
    failing_count: int = 0
    passing_count: int = 0
    failure_rate: float = 0.0

    # Activity metrics (from Student Summaries API)
    students_with_activity: int = 0
    activity_coverage: float = 0.0
    total_page_views: int = 0
    total_participations: int = 0
    avg_page_views: float = 0.0
    avg_participations: float = 0.0
    median_page_views: float = 0.0
    median_participations: float = 0.0

    # Canvas-computed levels (1-3 scale)
    avg_page_views_level: float = 0.0
    avg_participations_level: float = 0.0

    # Tardiness breakdown (from Student Summaries API)
    avg_on_time_rate: float = 0.0
    avg_late_rate: float = 0.0
    avg_missing_rate: float = 0.0
    total_on_time: int = 0
    total_late: int = 0
    total_missing: int = 0

    # Assignment Analytics (from Assignment Analytics API)
    assignment_count: int = 0
    graded_assignment_count: int = 0
    avg_assignment_median: float = 0.0
    avg_assignment_missing_rate: float = 0.0
    assignments_with_high_missing: int = 0  # >40% missing

    # Recent activity (from Recent Students API)
    students_active_last_7_days: int = 0
    students_active_last_30_days: int = 0
    avg_days_since_last_login: float = 0.0

    # Course activity patterns (from Course Activity API)
    total_activity_days: int = 0
    avg_daily_views: float = 0.0
    avg_daily_participations: float = 0.0
    peak_activity_day: str = ""

    # Computed scores (0-100)
    activity_engagement_score: float = 0.0
    tardiness_score: float = 0.0
    recency_score: float = 0.0
    grade_quality_score: float = 0.0
    activity_prediction_score: float = 0.0

    # Metadata
    analysis_timestamp: str = ""
    api_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# RATE-LIMITED API CLIENT
# =============================================================================

class ActivityAnalysisClient:
    """API client following CANVAS_API_REFERENCE.md specifications."""

    def __init__(self):
        self.api_url = API_URL
        self.headers = HEADERS
        self._lock = threading.Lock()
        self._quota = 700
        self._request_count = 0
        self._error_count = 0

    def _update_quota(self, response: requests.Response):
        """Update quota from response headers."""
        remaining = response.headers.get('X-Rate-Limit-Remaining')
        if remaining:
            with self._lock:
                self._quota = int(float(remaining))

    def _check_and_wait(self):
        """Check quota and wait if necessary."""
        with self._lock:
            quota = self._quota

        if quota <= QUOTA_CRITICAL:
            logger.warning(f"CRITICAL quota ({quota}). Waiting 60s...")
            time.sleep(60)
        elif quota <= QUOTA_LOW:
            logger.info(f"Low quota ({quota}). Waiting 30s...")
            time.sleep(30)
        elif quota <= QUOTA_MODERATE:
            time.sleep(1.0)
        else:
            time.sleep(RATE_LIMIT_DELAY)

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Optional[requests.Response]:
        """Make a GET request with rate limiting."""
        self._check_and_wait()

        url = f"{self.api_url}{endpoint}"

        try:
            response = requests.get(url, headers=self.headers, params=params, timeout=30)
            self._update_quota(response)

            with self._lock:
                self._request_count += 1

            if response.status_code == 200:
                return response
            elif response.status_code == 403:
                logger.warning(f"Rate limited on {endpoint}")
                time.sleep(5)
                return None
            else:
                return None

        except requests.exceptions.RequestException as e:
            with self._lock:
                self._error_count += 1
            logger.error(f"Request error: {e}")
            return None

    def paginate(self, endpoint: str, params: Optional[Dict] = None, max_pages: int = 10) -> List[Dict]:
        """Paginate through results (max 100 per page as per documentation)."""
        import re

        all_results = []
        params = params or {}
        params['per_page'] = 100

        current_url = f"{self.api_url}{endpoint}"
        page_count = 0
        first_request = True

        while current_url and page_count < max_pages:
            if first_request:
                response = self.get(endpoint, params)
                first_request = False
            else:
                # For subsequent pages, use the full URL from Link header
                self._check_and_wait()
                try:
                    response = requests.get(current_url, headers=self.headers, timeout=30)
                    self._update_quota(response)
                    if response.status_code != 200:
                        break
                except:
                    break

            if not response:
                break

            try:
                data = response.json()
            except ValueError:
                break

            if not data:
                break

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
        with self._lock:
            return {
                'requests': self._request_count,
                'errors': self._error_count,
                'quota': self._quota
            }


# =============================================================================
# COURSE ACTIVITY ANALYZER
# =============================================================================

class CourseActivityAnalyzer:
    """Analyzes courses using activity-focused endpoints from documentation."""

    def __init__(self, client: ActivityAnalysisClient):
        self.client = client

    def analyze_course(self, course: Dict) -> ActivityMetrics:
        """Perform comprehensive activity-based analysis of a course."""
        metrics = ActivityMetrics(
            course_id=course.get('id', 0),
            course_name=course.get('name', '')[:80],
            account_id=course.get('account_id', 0),
            term_id=course.get('enrollment_term_id', 0),
            term_name=course.get('term', {}).get('name', '') if course.get('term') else '',
            total_students=course.get('total_students', 0),
            analysis_timestamp=datetime.now().isoformat()
        )

        course_id = metrics.course_id

        if metrics.total_students < 5:
            return metrics

        # 1. Enrollments API - Get grades
        self._analyze_enrollments(course_id, metrics)

        # 2. Student Summaries API - Activity metrics & tardiness
        self._analyze_student_summaries(course_id, metrics)

        # 3. Assignment Analytics API - Assignment statistics
        self._analyze_assignment_analytics(course_id, metrics)

        # 4. Recent Students API - Login recency
        self._analyze_recent_students(course_id, metrics)

        # 5. Course Activity API - Daily patterns
        self._analyze_course_activity(course_id, metrics)

        # 6. Compute composite scores
        self._compute_scores(metrics)

        return metrics

    def _analyze_enrollments(self, course_id: int, metrics: ActivityMetrics):
        """
        Enrollments API - Get student grades
        Endpoint: GET /api/v1/courses/{course_id}/enrollments
        Params: type[]=StudentEnrollment, include[]=grades, include[]=total_scores
        """
        enrollments = self.client.paginate(
            f"/api/v1/courses/{course_id}/enrollments",
            params={
                'type[]': 'StudentEnrollment',
                'include[]': ['grades', 'total_scores'],
                'per_page': 100
            }
        )

        if not enrollments:
            metrics.api_errors.append("enrollments")
            return

        scores = []
        active_count = 0

        for e in enrollments:
            state = e.get('enrollment_state', '')
            if state == 'active':
                active_count += 1

            grades = e.get('grades', {})
            if grades:
                # Use current_score (completed work only) as primary
                score = grades.get('current_score')
                if score is not None:
                    scores.append(float(score))

        metrics.active_students = active_count
        metrics.students_with_grades = len(scores)
        metrics.grade_coverage = (len(scores) / len(enrollments) * 100) if enrollments else 0

        if len(scores) >= 3:
            metrics.grade_mean = statistics.mean(scores)
            metrics.grade_std = statistics.stdev(scores) if len(scores) > 1 else 0
            metrics.grade_min = min(scores)
            metrics.grade_max = max(scores)
            metrics.grade_median = statistics.median(scores)

            metrics.failing_count = sum(1 for s in scores if s < PASS_THRESHOLD)
            metrics.passing_count = sum(1 for s in scores if s >= PASS_THRESHOLD)
            metrics.failure_rate = metrics.failing_count / len(scores) if scores else 0

    def _analyze_student_summaries(self, course_id: int, metrics: ActivityMetrics):
        """
        Student Summaries API - Activity metrics
        Endpoint: GET /api/v1/courses/{course_id}/analytics/student_summaries

        Key fields:
        - page_views, page_views_level (1-3)
        - participations, participations_level (1-3)
        - tardiness_breakdown: {on_time, late, missing, floating}
        """
        summaries = self.client.paginate(
            f"/api/v1/courses/{course_id}/analytics/student_summaries"
        )

        if not summaries:
            metrics.api_errors.append("student_summaries")
            return

        page_views_list = []
        participations_list = []
        pv_levels = []
        part_levels = []

        on_time_rates = []
        late_rates = []
        missing_rates = []

        total_on_time = 0
        total_late = 0
        total_missing = 0

        with_activity = 0

        for s in summaries:
            pv = s.get('page_views') or 0
            part = s.get('participations') or 0

            page_views_list.append(pv)
            participations_list.append(part)

            if pv > 0 or part > 0:
                with_activity += 1

            # Canvas-computed levels
            pv_level = s.get('page_views_level') or 0
            part_level = s.get('participations_level') or 0
            if pv_level > 0:
                pv_levels.append(pv_level)
            if part_level > 0:
                part_levels.append(part_level)

            # Tardiness breakdown
            tb = s.get('tardiness_breakdown', {})
            on_time = tb.get('on_time') or 0
            late = tb.get('late') or 0
            missing = tb.get('missing') or 0

            total_on_time += on_time
            total_late += late
            total_missing += missing

            total_assignments = on_time + late + missing
            if total_assignments > 0:
                on_time_rates.append(on_time / total_assignments)
                late_rates.append(late / total_assignments)
                missing_rates.append(missing / total_assignments)

        # Activity metrics
        metrics.students_with_activity = with_activity
        metrics.activity_coverage = (with_activity / len(summaries) * 100) if summaries else 0
        metrics.total_page_views = sum(page_views_list)
        metrics.total_participations = sum(participations_list)

        if page_views_list:
            metrics.avg_page_views = statistics.mean(page_views_list)
            metrics.median_page_views = statistics.median(page_views_list)

        if participations_list:
            metrics.avg_participations = statistics.mean(participations_list)
            metrics.median_participations = statistics.median(participations_list)

        if pv_levels:
            metrics.avg_page_views_level = statistics.mean(pv_levels)
        if part_levels:
            metrics.avg_participations_level = statistics.mean(part_levels)

        # Tardiness metrics
        metrics.total_on_time = total_on_time
        metrics.total_late = total_late
        metrics.total_missing = total_missing

        if on_time_rates:
            metrics.avg_on_time_rate = statistics.mean(on_time_rates)
        if late_rates:
            metrics.avg_late_rate = statistics.mean(late_rates)
        if missing_rates:
            metrics.avg_missing_rate = statistics.mean(missing_rates)

    def _analyze_assignment_analytics(self, course_id: int, metrics: ActivityMetrics):
        """
        Assignment Analytics API - Per-assignment statistics
        Endpoint: GET /api/v1/courses/{course_id}/analytics/assignments

        Key fields:
        - min_score, max_score, median, first_quartile, third_quartile
        - tardiness_breakdown: {missing, late, on_time, total}
        """
        response = self.client.get(f"/api/v1/courses/{course_id}/analytics/assignments")

        if not response:
            metrics.api_errors.append("assignment_analytics")
            return

        try:
            assignments = response.json()
        except:
            return

        if not assignments or not isinstance(assignments, list):
            return

        metrics.assignment_count = len(assignments)

        medians = []
        missing_rates = []
        graded_count = 0
        high_missing_count = 0

        for a in assignments:
            # Check if graded (has scores)
            if a.get('max_score') is not None:
                graded_count += 1

            median = a.get('median')
            if median is not None:
                medians.append(median)

            tb = a.get('tardiness_breakdown', {})
            missing_rate = tb.get('missing')
            if missing_rate is not None:
                missing_rates.append(missing_rate)
                if missing_rate > 0.4:  # >40% missing
                    high_missing_count += 1

        metrics.graded_assignment_count = graded_count
        metrics.assignments_with_high_missing = high_missing_count

        if medians:
            metrics.avg_assignment_median = statistics.mean(medians)
        if missing_rates:
            metrics.avg_assignment_missing_rate = statistics.mean(missing_rates)

    def _analyze_recent_students(self, course_id: int, metrics: ActivityMetrics):
        """
        Recent Students API - Activity recency
        Endpoint: GET /api/v1/courses/{course_id}/recent_students

        Key fields:
        - last_login (timestamp)
        """
        response = self.client.get(f"/api/v1/courses/{course_id}/recent_students")

        if not response:
            metrics.api_errors.append("recent_students")
            return

        try:
            students = response.json()
        except:
            return

        if not students or not isinstance(students, list):
            return

        now = datetime.now()
        days_since_login = []
        active_7_days = 0
        active_30_days = 0

        for s in students:
            last_login = s.get('last_login')
            if last_login:
                try:
                    login_dt = datetime.fromisoformat(last_login.replace('Z', '+00:00'))
                    days_ago = (now - login_dt.replace(tzinfo=None)).days
                    days_since_login.append(days_ago)

                    if days_ago <= 7:
                        active_7_days += 1
                    if days_ago <= 30:
                        active_30_days += 1
                except:
                    pass

        metrics.students_active_last_7_days = active_7_days
        metrics.students_active_last_30_days = active_30_days

        if days_since_login:
            metrics.avg_days_since_last_login = statistics.mean(days_since_login)

    def _analyze_course_activity(self, course_id: int, metrics: ActivityMetrics):
        """
        Course Activity API - Daily aggregates
        Endpoint: GET /api/v1/courses/{course_id}/analytics/activity

        Key fields:
        - date, views, participations (per day)
        """
        response = self.client.get(f"/api/v1/courses/{course_id}/analytics/activity")

        if not response:
            metrics.api_errors.append("course_activity")
            return

        try:
            activity = response.json()
        except:
            return

        if not activity or not isinstance(activity, list):
            return

        metrics.total_activity_days = len(activity)

        daily_views = []
        daily_participations = []
        max_views = 0
        peak_day = ""

        for day in activity:
            views = day.get('views') or 0
            parts = day.get('participations') or 0

            daily_views.append(views)
            daily_participations.append(parts)

            if views > max_views:
                max_views = views
                peak_day = day.get('date', '')

        if daily_views:
            metrics.avg_daily_views = statistics.mean(daily_views)
        if daily_participations:
            metrics.avg_daily_participations = statistics.mean(daily_participations)

        metrics.peak_activity_day = peak_day

    def _compute_scores(self, metrics: ActivityMetrics):
        """Compute composite activity-based prediction scores."""

        # 1. Activity Engagement Score (0-100)
        # Based on page views, participations, and Canvas-computed levels
        engagement_factors = []

        if metrics.activity_coverage > 0:
            engagement_factors.append(min(100, metrics.activity_coverage))

        if metrics.avg_page_views_level > 0:
            engagement_factors.append(metrics.avg_page_views_level / 3 * 100)

        if metrics.avg_participations_level > 0:
            engagement_factors.append(metrics.avg_participations_level / 3 * 100)

        # Normalize page views (assume 500 is good)
        if metrics.avg_page_views > 0:
            pv_score = min(100, metrics.avg_page_views / 500 * 100)
            engagement_factors.append(pv_score)

        metrics.activity_engagement_score = statistics.mean(engagement_factors) if engagement_factors else 0

        # 2. Tardiness Score (0-100)
        # Higher on_time rate = better, but we want some late/missing for prediction signal
        if metrics.avg_on_time_rate > 0 or metrics.avg_missing_rate > 0:
            # Ideal: some missing (20-40%) for prediction, but not too much
            if 0.15 <= metrics.avg_missing_rate <= 0.50:
                tardiness_score = 100
            elif metrics.avg_missing_rate < 0.15:
                tardiness_score = 50 + (metrics.avg_missing_rate / 0.15 * 50)
            else:
                tardiness_score = max(0, 100 - (metrics.avg_missing_rate - 0.50) * 200)

            metrics.tardiness_score = tardiness_score

        # 3. Recency Score (0-100)
        # Based on recent student activity
        if metrics.total_students > 0:
            active_rate_7d = metrics.students_active_last_7_days / metrics.total_students
            active_rate_30d = metrics.students_active_last_30_days / metrics.total_students

            recency_score = (active_rate_7d * 60 + active_rate_30d * 40) * 100
            metrics.recency_score = min(100, recency_score)

        # 4. Grade Quality Score (0-100)
        # Based on grade coverage and variance
        if metrics.students_with_grades >= MIN_STUDENTS:
            coverage_score = min(100, metrics.grade_coverage)
            variance_score = min(100, metrics.grade_std / 30 * 100) if metrics.grade_std > 0 else 0

            # Class balance bonus
            if 0.15 <= metrics.failure_rate <= 0.85:
                balance_bonus = 20
            else:
                balance_bonus = 0

            metrics.grade_quality_score = (coverage_score * 0.4 + variance_score * 0.4 + balance_bonus)

        # 5. Activity Prediction Score (0-100)
        # Weighted composite of all activity-based factors
        weights = {
            'activity_engagement': 0.30,
            'tardiness': 0.25,
            'recency': 0.15,
            'grade_quality': 0.30,
        }

        composite = (
            metrics.activity_engagement_score * weights['activity_engagement'] +
            metrics.tardiness_score * weights['tardiness'] +
            metrics.recency_score * weights['recency'] +
            metrics.grade_quality_score * weights['grade_quality']
        )

        # Minimum requirements check
        if metrics.students_with_activity < MIN_STUDENTS:
            composite = composite * 0.5  # Penalize low activity coverage

        if metrics.students_with_grades < MIN_STUDENTS:
            composite = composite * 0.5  # Penalize low grade coverage

        metrics.activity_prediction_score = round(composite, 1)


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

class ActivityBatchProcessor:
    """Process multiple courses with progress tracking."""

    def __init__(self, client: ActivityAnalysisClient, max_workers: int = 4):
        self.client = client
        self.analyzer = CourseActivityAnalyzer(client)
        self.max_workers = max_workers
        self._lock = threading.Lock()
        self._processed = 0
        self._total = 0
        self._start_time = None

    def discover_courses(self, account_id: int, max_courses: Optional[int] = None) -> List[Dict]:
        """Discover courses from an account."""
        logger.info(f"Discovering courses from account {account_id}...")

        courses = self.client.paginate(
            f"/api/v1/accounts/{account_id}/courses",
            params={
                'with_enrollments': 'true',
                'include[]': ['total_students', 'term'],
                'per_page': 100
            },
            max_pages=30
        )

        # Filter to courses with students
        courses = [c for c in courses if c.get('total_students', 0) >= 5]

        # If no courses, try sub-accounts
        if len(courses) == 0:
            logger.info("No direct courses. Checking sub-accounts...")
            sub_accounts = self.client.paginate(
                f"/api/v1/accounts/{account_id}/sub_accounts",
                params={'recursive': 'false'}
            )

            for sub in sub_accounts[:20]:
                sub_id = sub.get('id')
                if sub_id:
                    sub_courses = self.client.paginate(
                        f"/api/v1/accounts/{sub_id}/courses",
                        params={
                            'with_enrollments': 'true',
                            'include[]': ['total_students', 'term'],
                            'per_page': 100
                        },
                        max_pages=20
                    )
                    sub_courses = [c for c in sub_courses if c.get('total_students', 0) >= 5]
                    courses.extend(sub_courses)
                    logger.info(f"  Account {sub_id}: {len(sub_courses)} courses")

                    if max_courses and len(courses) >= max_courses:
                        break

        if max_courses:
            courses = courses[:max_courses]

        logger.info(f"Found {len(courses)} courses")
        return courses

    def process_courses(self, courses: List[Dict]) -> List[ActivityMetrics]:
        """Process courses with threading."""
        self._total = len(courses)
        self._processed = 0
        self._start_time = time.time()

        results = []

        logger.info(f"Analyzing {self._total} courses with {self.max_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_one, course): course
                for course in courses
            }

            for future in as_completed(futures):
                try:
                    metrics = future.result()
                    if metrics:
                        results.append(metrics)
                except Exception as e:
                    logger.error(f"Error: {e}")

        elapsed = time.time() - self._start_time
        logger.info(f"Completed {len(results)} courses in {elapsed:.1f}s")

        return results

    def _process_one(self, course: Dict) -> Optional[ActivityMetrics]:
        """Process single course with progress tracking."""
        try:
            metrics = self.analyzer.analyze_course(course)

            with self._lock:
                self._processed += 1
                if self._processed % 20 == 0:
                    elapsed = time.time() - self._start_time
                    rate = self._processed / elapsed if elapsed > 0 else 0
                    stats = self.client.get_stats()
                    logger.info(
                        f"Progress: {self._processed}/{self._total} "
                        f"({rate:.1f}/sec) | Quota: {stats['quota']}"
                    )

            return metrics
        except Exception as e:
            logger.error(f"Error on course {course.get('id')}: {e}")
            return None


# =============================================================================
# RESULTS & COMPARISON
# =============================================================================

def save_results(results: List[ActivityMetrics], output_dir: str) -> str:
    """Save results to CSV."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.DataFrame([m.to_dict() for m in results])
    df = df.sort_values('activity_prediction_score', ascending=False)

    # Save timestamped and latest versions
    csv_path = os.path.join(output_dir, f"activity_analysis_{timestamp}.csv")
    df.to_csv(csv_path, index=False)

    latest_path = os.path.join(output_dir, "activity_analysis_latest.csv")
    df.to_csv(latest_path, index=False)

    logger.info(f"Saved: {csv_path}")
    return csv_path


def compare_with_previous(results: List[ActivityMetrics], previous_csv: str):
    """Compare activity-based results with previous course analysis."""

    if not os.path.exists(previous_csv):
        logger.warning(f"Previous analysis not found: {previous_csv}")
        return

    print("\n" + "=" * 80)
    print("COMPARISON: Activity Analysis vs Previous Course Analysis")
    print("=" * 80)

    # Load previous results
    df_prev = pd.read_csv(previous_csv)

    # Create current dataframe
    df_curr = pd.DataFrame([m.to_dict() for m in results])

    # Get top 20 from each
    top_prev = set(df_prev.nlargest(20, 'prediction_potential_score')['course_id'].values)
    top_curr = set(df_curr.nlargest(20, 'activity_prediction_score')['course_id'].values)

    overlap = top_prev & top_curr

    print(f"\nTop 20 Overlap: {len(overlap)} courses ({len(overlap)/20*100:.0f}%)")

    if overlap:
        print("\nCourses in BOTH top 20:")
        for cid in overlap:
            prev_row = df_prev[df_prev['course_id'] == cid].iloc[0]
            curr_row = df_curr[df_curr['course_id'] == cid].iloc[0]
            print(f"  {cid}: {prev_row['course_name'][:40]}")
            print(f"         Previous: {prev_row['prediction_potential_score']:.1f} | Activity: {curr_row['activity_prediction_score']:.1f}")

    # Correlation between scores
    merged = pd.merge(
        df_prev[['course_id', 'prediction_potential_score']],
        df_curr[['course_id', 'activity_prediction_score']],
        on='course_id'
    )

    if len(merged) > 10:
        correlation = merged['prediction_potential_score'].corr(merged['activity_prediction_score'])
        print(f"\nScore Correlation: {correlation:.3f}")

    print("=" * 80)


def print_summary(results: List[ActivityMetrics]):
    """Print analysis summary."""

    print("\n" + "=" * 80)
    print("ACTIVITY-BASED COURSE ANALYSIS SUMMARY")
    print("=" * 80)

    total = len(results)
    with_activity = sum(1 for r in results if r.students_with_activity >= MIN_STUDENTS)
    with_grades = sum(1 for r in results if r.students_with_grades >= MIN_STUDENTS)
    high_potential = sum(1 for r in results if r.activity_prediction_score >= 50)

    print(f"""
OVERVIEW
────────
  Total Courses Analyzed:       {total}
  With Activity Data (≥15):     {with_activity} ({with_activity/total*100:.1f}%)
  With Grade Data (≥15):        {with_grades} ({with_grades/total*100:.1f}%)
  High Activity Potential (≥50): {high_potential} ({high_potential/total*100:.1f}%)

ACTIVITY METRICS (courses with activity)
────────────────────────────────────────""")

    active_courses = [r for r in results if r.students_with_activity >= MIN_STUDENTS]
    if active_courses:
        avg_pv = statistics.mean([r.avg_page_views for r in active_courses])
        avg_part = statistics.mean([r.avg_participations for r in active_courses])
        avg_missing = statistics.mean([r.avg_missing_rate for r in active_courses if r.avg_missing_rate > 0])

        print(f"  Avg Page Views/Student:      {avg_pv:.1f}")
        print(f"  Avg Participations/Student:  {avg_part:.1f}")
        print(f"  Avg Missing Rate:            {avg_missing*100:.1f}%")

    # Top courses
    sorted_results = sorted(results, key=lambda x: x.activity_prediction_score, reverse=True)

    print(f"""
TOP 15 COURSES BY ACTIVITY PREDICTION SCORE
───────────────────────────────────────────""")
    print(f"{'Rank':<5} {'ID':<8} {'Course':<35} {'Activity':<10} {'Tardiness':<10} {'Score':<8}")
    print("-" * 80)

    for i, m in enumerate(sorted_results[:15], 1):
        name = m.course_name[:33] + ".." if len(m.course_name) > 35 else m.course_name
        print(f"{i:<5} {m.course_id:<8} {name:<35} "
              f"{m.activity_engagement_score:<10.1f} {m.tardiness_score:<10.1f} "
              f"{m.activity_prediction_score:<8.1f}")

    print("=" * 80)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Activity-based course analysis using Canvas API endpoints"
    )
    parser.add_argument(
        '--campus-ids', type=str, default="176",
        help='Comma-separated campus IDs (default: 176 = Providencia)'
    )
    parser.add_argument(
        '--max-courses', type=int, default=100,
        help='Max courses per campus (default: 100)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Parallel workers (default: 4)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/discovery',
        help='Output directory'
    )
    parser.add_argument(
        '--compare', type=str, default='data/discovery/course_analysis_latest.csv',
        help='Previous analysis CSV to compare with'
    )

    args = parser.parse_args()

    # Parse campus IDs
    campus_ids = [int(x.strip()) for x in args.campus_ids.split(',')]

    print("\n" + "=" * 70)
    print("ACTIVITY-BASED COURSE ANALYSIS")
    print("Based on CANVAS_API_REFERENCE.md endpoints")
    print("=" * 70)
    print(f"Campuses:       {campus_ids}")
    print(f"Max Courses:    {args.max_courses} per campus")
    print(f"Workers:        {args.workers}")
    print("=" * 70 + "\n")

    # Initialize
    client = ActivityAnalysisClient()
    processor = ActivityBatchProcessor(client, max_workers=args.workers)

    # Collect courses from all campuses
    all_courses = []
    for campus_id in campus_ids:
        courses = processor.discover_courses(campus_id, args.max_courses)
        all_courses.extend(courses)

    # Deduplicate
    seen = set()
    unique_courses = []
    for c in all_courses:
        cid = c.get('id')
        if cid not in seen:
            seen.add(cid)
            unique_courses.append(c)

    logger.info(f"Total unique courses: {len(unique_courses)}")

    if not unique_courses:
        logger.error("No courses found")
        return

    # Process
    results = processor.process_courses(unique_courses)

    # Save
    output_path = save_results(results, args.output_dir)

    # Summary
    print_summary(results)

    # Compare with previous
    if args.compare:
        compare_with_previous(results, args.compare)

    # Stats
    stats = client.get_stats()
    print(f"\nAPI Statistics:")
    print(f"  Requests: {stats['requests']}")
    print(f"  Errors: {stats['errors']}")
    print(f"  Final Quota: {stats['quota']}")
    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
