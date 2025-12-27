"""
Discovery Module

This module provides tools for discovering and analyzing Canvas LMS courses
for prediction potential. It includes:

- canvas_client: Thread-safe API client with rate limiting
- course_analysis: Course-level metrics and scoring
- career_analysis: Career (sub-account) level aggregation
- batch_scanner: Multi-threaded batch scanning
- url_parser: URL parsing utilities for page views

Usage:
    from scripts.discovery import CanvasClient, BatchScanner, analyze_course

    client = CanvasClient()
    scanner = BatchScanner(client)
    results = scanner.scan_courses([86005, 86676, 84936])
"""

from .canvas_client import CanvasClient, RateLimitError, APIError, get_client
from .course_analysis import (
    CourseMetrics,
    LMSDesignMetrics,
    GradeMetrics,
    analyze_course,
    extract_lms_metrics,
    extract_grade_metrics,
    course_metrics_to_dict,
    PASS_THRESHOLD,
    MIN_STUDENTS_FOR_PREDICTION,
    MIN_GRADE_STD,
)
from .career_analysis import (
    CareerMetrics,
    aggregate_career_metrics,
    discover_careers,
    discover_career_courses,
    build_account_hierarchy,
    career_metrics_to_dict,
)
from .batch_scanner import (
    BatchScanner,
    ScanProgress,
    create_progress_printer,
)
from .url_parser import (
    parse_canvas_url,
    extract_course_id,
    extract_user_id,
    categorize_page_view,
    filter_page_views_by_course,
    aggregate_page_views,
)

__all__ = [
    # Client
    'CanvasClient',
    'RateLimitError',
    'APIError',
    'get_client',

    # Course Analysis
    'CourseMetrics',
    'LMSDesignMetrics',
    'GradeMetrics',
    'analyze_course',
    'extract_lms_metrics',
    'extract_grade_metrics',
    'course_metrics_to_dict',
    'PASS_THRESHOLD',
    'MIN_STUDENTS_FOR_PREDICTION',
    'MIN_GRADE_STD',

    # Career Analysis
    'CareerMetrics',
    'aggregate_career_metrics',
    'discover_careers',
    'discover_career_courses',
    'build_account_hierarchy',
    'career_metrics_to_dict',

    # Batch Scanner
    'BatchScanner',
    'ScanProgress',
    'create_progress_printer',

    # URL Parser
    'parse_canvas_url',
    'extract_course_id',
    'extract_user_id',
    'categorize_page_view',
    'filter_page_views_by_course',
    'aggregate_page_views',
]

__version__ = '1.0.0'
