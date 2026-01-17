"""
Batch Scanner Module

This module provides batch scanning capabilities with:
- Conservative multi-threading (max 5 workers)
- Dynamic rate limit monitoring
- Automatic pausing when quota is low
- Progress tracking and logging
- Intermediate results saving

Usage:
    from scripts.discovery.batch_scanner import BatchScanner

    scanner = BatchScanner(client)
    results = scanner.scan_courses(course_ids, progress_callback=callback)
"""

import os
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .canvas_client import CanvasClient, RateLimitError
from .course_analysis import (
    CourseMetrics, analyze_course, course_metrics_to_dict,
    extract_grade_metrics, GradeMetrics
)
from .career_analysis import (
    CareerMetrics, aggregate_career_metrics, career_metrics_to_dict
)

logger = logging.getLogger(__name__)


@dataclass
class ScanProgress:
    """Track scanning progress."""
    total: int = 0
    completed: int = 0
    errors: int = 0
    skipped: int = 0
    high_potential: int = 0
    medium_potential: int = 0
    low_potential: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    current_quota: int = 700

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.start_time).total_seconds()

    @property
    def rate_per_minute(self) -> float:
        if self.elapsed_seconds > 0:
            return (self.completed / self.elapsed_seconds) * 60
        return 0.0


class BatchScanner:
    """
    Batch scanner for courses and careers with rate limit safety.

    Implements conservative threading and automatic throttling
    based on Canvas API rate limits.
    """

    # Conservative defaults
    DEFAULT_MAX_WORKERS = 3
    MAX_WORKERS_LIMIT = 5

    # Quota thresholds
    QUOTA_STOP = 100       # Stop scanning completely
    QUOTA_PAUSE = 200      # Pause and wait for recovery
    QUOTA_SLOW = 350       # Reduce workers to 1

    def __init__(
        self,
        client: Optional[CanvasClient] = None,
        max_workers: int = DEFAULT_MAX_WORKERS,
        save_intermediate: bool = True,
        output_dir: Optional[str] = None
    ):
        """
        Initialize batch scanner.

        Args:
            client: CanvasClient instance (creates new if None)
            max_workers: Maximum concurrent workers (capped at 5)
            save_intermediate: Whether to save intermediate results
            output_dir: Directory for saving results
        """
        self.client = client or CanvasClient()
        self.max_workers = min(max_workers, self.MAX_WORKERS_LIMIT)
        self.save_intermediate = save_intermediate

        # Output directory
        if output_dir:
            self.output_dir = output_dir
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.output_dir = os.path.join(base_dir, 'data', 'discovery')

        os.makedirs(self.output_dir, exist_ok=True)

        # Thread-safe state
        self._lock = threading.Lock()
        self._results: List[Dict] = []
        self._stop_flag = False
        self._pause_flag = False

        logger.info(
            f"BatchScanner initialized: max_workers={self.max_workers}, "
            f"output_dir={self.output_dir}"
        )

    def _check_quota_and_adjust(self, progress: ScanProgress) -> int:
        """
        Check quota and adjust worker count dynamically.

        Returns:
            Adjusted number of workers to use
        """
        quota = self.client.rate_state.get()
        progress.current_quota = quota

        if quota <= self.QUOTA_STOP:
            logger.warning(f"Quota critically low ({quota}). Stopping scan.")
            self._stop_flag = True
            return 0

        if quota <= self.QUOTA_PAUSE:
            logger.warning(f"Quota low ({quota}). Pausing for recovery...")
            self._pause_flag = True
            time.sleep(30)  # Wait 30 seconds for recovery
            self._pause_flag = False
            return 1

        if quota <= self.QUOTA_SLOW:
            return 1

        return self.max_workers

    def _analyze_course_safe(
        self,
        course_id: int,
        include_lms_details: bool = False
    ) -> Optional[CourseMetrics]:
        """
        Safely analyze a course with error handling.

        Args:
            course_id: Course ID
            include_lms_details: Whether to fetch full LMS metrics

        Returns:
            CourseMetrics or None on error
        """
        if self._stop_flag:
            return None

        try:
            return analyze_course(
                self.client,
                course_id,
                include_lms_details=include_lms_details
            )
        except RateLimitError as e:
            logger.error(f"Rate limit error for course {course_id}: {e}")
            self._stop_flag = True
            return None
        except Exception as e:
            logger.warning(f"Error analyzing course {course_id}: {e}")
            return None

    def _quick_grade_check(self, course_id: int) -> Optional[GradeMetrics]:
        """
        Quick grade check without full analysis.

        Only fetches enrollments to check if course has Canvas grades.
        Much faster than full analysis.

        Args:
            course_id: Course ID

        Returns:
            GradeMetrics or None
        """
        if self._stop_flag:
            return None

        try:
            return extract_grade_metrics(self.client, course_id)
        except Exception as e:
            logger.warning(f"Error checking grades for course {course_id}: {e}")
            return None

    def scan_courses(
        self,
        course_ids: List[int],
        full_analysis: bool = False,
        progress_callback: Optional[Callable[[ScanProgress], None]] = None,
        batch_size: int = 30
    ) -> List[Dict]:
        """
        Scan multiple courses for prediction potential.

        Args:
            course_ids: List of course IDs to scan
            full_analysis: If True, fetches full LMS details (slower)
            progress_callback: Optional callback for progress updates
            batch_size: Number of courses per batch

        Returns:
            List of course metrics dictionaries
        """
        progress = ScanProgress(total=len(course_ids))
        self._results = []
        self._stop_flag = False

        logger.info(f"Starting batch scan of {len(course_ids)} courses")
        logger.info(f"Mode: {'Full analysis' if full_analysis else 'Quick grade check'}")

        # Process in batches
        for batch_start in range(0, len(course_ids), batch_size):
            if self._stop_flag:
                logger.warning("Scan stopped due to quota limits")
                break

            batch_end = min(batch_start + batch_size, len(course_ids))
            batch = course_ids[batch_start:batch_end]

            logger.info(
                f"Processing batch {batch_start//batch_size + 1}: "
                f"courses {batch_start + 1}-{batch_end} of {len(course_ids)}"
            )

            # Check quota and adjust workers
            effective_workers = self._check_quota_and_adjust(progress)
            if effective_workers == 0:
                break

            # Process batch with thread pool
            with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                futures = {}

                for course_id in batch:
                    if self._stop_flag:
                        break

                    if full_analysis:
                        future = executor.submit(
                            self._analyze_course_safe, course_id, True
                        )
                    else:
                        future = executor.submit(
                            self._analyze_course_safe, course_id, False
                        )
                    futures[future] = course_id

                # Collect results
                for future in as_completed(futures):
                    if self._stop_flag:
                        break

                    course_id = futures[future]
                    try:
                        metrics = future.result()
                        if metrics:
                            with self._lock:
                                self._results.append(course_metrics_to_dict(metrics))

                                # Update progress counts
                                if metrics.recommendation == "HIGH":
                                    progress.high_potential += 1
                                elif metrics.recommendation == "MEDIUM":
                                    progress.medium_potential += 1
                                elif metrics.recommendation == "LOW":
                                    progress.low_potential += 1
                                else:
                                    progress.skipped += 1

                            progress.completed += 1
                        else:
                            progress.errors += 1

                    except Exception as e:
                        logger.warning(f"Error processing course {course_id}: {e}")
                        progress.errors += 1

                    progress.last_update = datetime.now()
                    if progress_callback:
                        progress_callback(progress)

            # Save intermediate results
            if self.save_intermediate and self._results:
                self._save_intermediate_results(progress)

            # Log batch summary
            self.client.log_quota()
            logger.info(
                f"Batch complete: {progress.completed}/{progress.total} courses, "
                f"HIGH={progress.high_potential}, MEDIUM={progress.medium_potential}"
            )

        # Final summary
        logger.info(
            f"Scan complete: {progress.completed} courses in "
            f"{progress.elapsed_seconds:.1f}s ({progress.rate_per_minute:.1f}/min)"
        )

        return self._results

    def scan_career(
        self,
        career_id: int,
        career_name: str,
        course_ids: Optional[List[int]] = None,
        term_id: Optional[int] = None,
        campus_name: str = "",
        progress_callback: Optional[Callable[[ScanProgress], None]] = None
    ) -> CareerMetrics:
        """
        Scan all courses in a career and aggregate metrics.

        Args:
            career_id: Career (sub-account) ID
            career_name: Career name
            course_ids: Optional pre-fetched course IDs
            term_id: Optional term filter
            campus_name: Campus name for reference
            progress_callback: Progress callback

        Returns:
            CareerMetrics with aggregated data
        """
        logger.info(f"Scanning career: {career_name} (ID: {career_id})")

        # Get course IDs if not provided
        if course_ids is None:
            courses = self.client.get_account_courses(career_id, term_id)
            course_ids = [c.get('id') for c in courses if c.get('id')]

        if not course_ids:
            return CareerMetrics(
                career_id=career_id,
                career_name=career_name,
                campus_name=campus_name,
                recommendation="SKIP",
                reasons=["No courses found"]
            )

        # Scan courses
        course_results = self.scan_courses(
            course_ids,
            full_analysis=False,
            progress_callback=progress_callback
        )

        # Convert to CourseMetrics objects for aggregation
        courses_metrics = []
        for result in course_results:
            metrics = CourseMetrics(
                course_id=result.get('course_id', 0),
                course_name=result.get('course_name', ''),
                account_id=result.get('account_id', 0)
            )
            metrics.grades.total_enrolled = result.get('total_enrolled', 0)
            metrics.grades.students_with_grades = result.get('students_with_grades', 0)
            metrics.grades.has_canvas_grades = result.get('has_canvas_grades', False)
            metrics.grades.grade_mean = result.get('grade_mean')
            metrics.grades.grade_std = result.get('grade_std')
            metrics.grades.pass_rate = result.get('pass_rate')
            metrics.grades.fail_rate = result.get('fail_rate')
            metrics.lms.lms_design_score = result.get('lms_design_score', 0)
            metrics.lms.assignments_count = result.get('assignments_count', 0)
            metrics.prediction_learning_score = result.get('prediction_learning_score', 0)
            metrics.recommendation = result.get('recommendation', 'SKIP')
            courses_metrics.append(metrics)

        # Aggregate
        career_metrics = aggregate_career_metrics(
            career_id=career_id,
            career_name=career_name,
            courses_metrics=courses_metrics,
            campus_name=campus_name
        )

        return career_metrics

    def _save_intermediate_results(self, progress: ScanProgress):
        """Save intermediate results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"scan_intermediate_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        data = {
            'timestamp': timestamp,
            'progress': {
                'total': progress.total,
                'completed': progress.completed,
                'errors': progress.errors,
                'high_potential': progress.high_potential,
                'medium_potential': progress.medium_potential,
                'elapsed_seconds': progress.elapsed_seconds
            },
            'results': self._results
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"Intermediate results saved to {filepath}")

    def save_results(self, filename: Optional[str] = None) -> str:
        """
        Save final results to JSON file.

        Args:
            filename: Optional filename (auto-generated if None)

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scan_results_{timestamp}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self._results, f, indent=2, default=str)

        logger.info(f"Results saved to {filepath}")
        return filepath


def create_progress_printer() -> Callable[[ScanProgress], None]:
    """Create a simple progress printer callback."""
    def print_progress(progress: ScanProgress):
        pct = (progress.completed / progress.total * 100) if progress.total > 0 else 0
        print(
            f"\rProgress: {progress.completed}/{progress.total} ({pct:.1f}%) | "
            f"HIGH={progress.high_potential} MEDIUM={progress.medium_potential} | "
            f"Quota={progress.current_quota} | "
            f"{progress.rate_per_minute:.1f}/min",
            end='', flush=True
        )
    return print_progress


if __name__ == '__main__':
    # Test the scanner
    client = CanvasClient()
    scanner = BatchScanner(client, max_workers=2)

    print("Testing BatchScanner...")
    print("=" * 60)

    # Test with a few courses
    test_courses = [86005, 86676, 84936]

    print(f"\nScanning {len(test_courses)} test courses...")
    results = scanner.scan_courses(
        test_courses,
        full_analysis=False,
        progress_callback=create_progress_printer()
    )

    print(f"\n\nResults: {len(results)} courses analyzed")

    for r in results:
        print(f"  - {r['course_id']}: {r['course_name'][:40]} -> {r['recommendation']}")
