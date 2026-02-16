"""
Career (Account) Analysis Module

This module provides functions to analyze careers (Canvas sub-accounts) for:
- Career-level metrics aggregation
- Course discovery within careers
- Career ranking and recommendation

Career = Canvas sub-account representing an academic program.

Usage:
    from scripts.discovery.career_analysis import analyze_career, CareerMetrics

    client = CanvasClient()
    metrics = analyze_career(client, career_id=719, courses_data=courses_list)
"""

import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import numpy as np

from .canvas_client import CanvasClient
from .course_analysis import CourseMetrics, course_metrics_to_dict

logger = logging.getLogger(__name__)


@dataclass
class CareerMetrics:
    """Career-level aggregated metrics."""
    # Career info
    career_id: int = 0
    career_name: str = ""
    parent_account_id: Optional[int] = None
    campus_name: str = ""

    # Counts
    total_courses: int = 0
    courses_with_students: int = 0
    courses_with_canvas_grades: int = 0

    # Aggregated student counts
    total_students: int = 0
    total_students_with_grades: int = 0

    # Aggregated scores
    avg_lms_design_score: float = 0.0
    avg_prediction_learning_score: float = 0.0
    career_composite_score: float = 0.0

    # Course breakdown by recommendation
    high_potential_count: int = 0
    medium_potential_count: int = 0
    low_potential_count: int = 0
    skip_count: int = 0

    # Recommendation
    recommendation: str = "SKIP"
    reasons: List[str] = field(default_factory=list)

    # Course details
    courses: List[Dict] = field(default_factory=list)


def aggregate_career_metrics(
    career_id: int,
    career_name: str,
    courses_metrics: List[CourseMetrics],
    parent_account_id: Optional[int] = None,
    campus_name: str = ""
) -> CareerMetrics:
    """
    Aggregate course metrics into career-level metrics.

    Args:
        career_id: Career (sub-account) ID
        career_name: Career name
        courses_metrics: List of CourseMetrics for courses in this career
        parent_account_id: Parent account ID (campus)
        campus_name: Campus name

    Returns:
        CareerMetrics with aggregated data
    """
    metrics = CareerMetrics(
        career_id=career_id,
        career_name=career_name,
        parent_account_id=parent_account_id,
        campus_name=campus_name
    )

    if not courses_metrics:
        metrics.recommendation = "SKIP"
        metrics.reasons = ["No courses found"]
        return metrics

    metrics.total_courses = len(courses_metrics)

    # Aggregate metrics
    lms_scores = []
    prediction_scores = []

    for course in courses_metrics:
        # Count courses with students
        if course.grades.total_enrolled > 0:
            metrics.courses_with_students += 1
            metrics.total_students += course.grades.total_enrolled

        # Count courses with Canvas grades
        if course.grades.has_canvas_grades:
            metrics.courses_with_canvas_grades += 1
            metrics.total_students_with_grades += course.grades.students_with_grades

        # Collect scores
        if course.lms.lms_design_score > 0:
            lms_scores.append(course.lms.lms_design_score)
        if course.prediction_learning_score > 0:
            prediction_scores.append(course.prediction_learning_score)

        # Count by recommendation
        if course.recommendation == "HIGH":
            metrics.high_potential_count += 1
        elif course.recommendation == "MEDIUM":
            metrics.medium_potential_count += 1
        elif course.recommendation == "LOW":
            metrics.low_potential_count += 1
        else:
            metrics.skip_count += 1

        # Store course summary
        metrics.courses.append(course_metrics_to_dict(course))

    # Calculate averages
    if lms_scores:
        metrics.avg_lms_design_score = np.mean(lms_scores)
    if prediction_scores:
        metrics.avg_prediction_learning_score = np.mean(prediction_scores)

    # Calculate career composite score
    # Weighted by:
    # - Courses with Canvas grades (40%)
    # - High/Medium potential courses (30%)
    # - Average prediction score (30%)
    if metrics.total_courses > 0:
        grades_ratio = metrics.courses_with_canvas_grades / metrics.total_courses
        potential_ratio = (
            (metrics.high_potential_count + metrics.medium_potential_count) /
            metrics.total_courses
        )
        metrics.career_composite_score = (
            0.4 * grades_ratio +
            0.3 * potential_ratio +
            0.3 * metrics.avg_prediction_learning_score
        )

    # Generate recommendation
    metrics.recommendation, metrics.reasons = generate_career_recommendation(metrics)

    return metrics


def generate_career_recommendation(metrics: CareerMetrics) -> tuple:
    """
    Generate career recommendation based on aggregated metrics.

    Returns:
        Tuple of (recommendation, reasons_list)
    """
    reasons = []

    # Check for disqualifying conditions
    if metrics.total_courses == 0:
        reasons.append("No courses found")
        return "SKIP", reasons

    if metrics.courses_with_students == 0:
        reasons.append("No courses with enrolled students")
        return "SKIP", reasons

    if metrics.courses_with_canvas_grades == 0:
        reasons.append("No courses with Canvas grades (all use external gradebook)")
        return "SKIP", reasons

    # Analyze potential
    high_medium_count = metrics.high_potential_count + metrics.medium_potential_count
    analyzable_ratio = high_medium_count / metrics.total_courses if metrics.total_courses > 0 else 0

    # Add reasons
    reasons.append(f"{metrics.total_courses} total courses")
    reasons.append(f"{metrics.courses_with_canvas_grades} with Canvas grades")
    reasons.append(
        f"{metrics.high_potential_count} HIGH + {metrics.medium_potential_count} MEDIUM potential"
    )

    if metrics.total_students_with_grades > 100:
        reasons.append(f"{metrics.total_students_with_grades} students with grades (good sample)")

    # Determine recommendation
    if metrics.high_potential_count >= 5 or (
        metrics.high_potential_count >= 2 and metrics.medium_potential_count >= 3
    ):
        return "PRIORITIZE", reasons

    if high_medium_count >= 3 or analyzable_ratio >= 0.3:
        return "MONITOR", reasons

    if metrics.courses_with_canvas_grades >= 1:
        reasons.append("Few analyzable courses but some Canvas grades present")
        return "MONITOR", reasons

    return "SKIP", reasons


def discover_career_courses(
    client: CanvasClient,
    career_id: int,
    term_id: Optional[int] = None
) -> List[Dict]:
    """
    Discover all courses in a career (sub-account).

    Args:
        client: CanvasClient instance
        career_id: Career (sub-account) ID
        term_id: Optional term ID filter

    Returns:
        List of course dictionaries
    """
    try:
        courses = client.get_account_courses(
            account_id=career_id,
            term_id=term_id,
            with_enrollments=True
        )
        return courses
    except Exception as e:
        logger.error(f"Error discovering courses for career {career_id}: {e}")
        return []


def discover_careers(
    client: CanvasClient,
    parent_account_id: int
) -> List[Dict]:
    """
    Discover all careers (sub-accounts) under a parent account.

    Args:
        client: CanvasClient instance
        parent_account_id: Parent account ID (e.g., campus)

    Returns:
        List of career dictionaries with id, name, parent_account_id
    """
    try:
        sub_accounts = client.get_sub_accounts(parent_account_id)
        return sub_accounts
    except Exception as e:
        logger.error(f"Error discovering careers for account {parent_account_id}: {e}")
        return []


def build_account_hierarchy(
    client: CanvasClient,
    root_account_id: int,
    max_depth: int = 3
) -> Dict[str, Any]:
    """
    Build account hierarchy tree from root account.

    Args:
        client: CanvasClient instance
        root_account_id: Root account ID to start from
        max_depth: Maximum depth to traverse

    Returns:
        Dictionary representing the account tree
    """
    def traverse(account_id: int, depth: int) -> Dict:
        if depth >= max_depth:
            return {'id': account_id, 'children': []}

        sub_accounts = client.get_sub_accounts(account_id)
        children = []

        for sub in sub_accounts:
            child = {
                'id': sub.get('id'),
                'name': sub.get('name'),
                'parent_account_id': sub.get('parent_account_id'),
                'children': []
            }
            if depth < max_depth - 1:
                child['children'] = traverse(sub.get('id'), depth + 1).get('children', [])
            children.append(child)

        return {'id': account_id, 'children': children}

    return traverse(root_account_id, 0)


def career_metrics_to_dict(metrics: CareerMetrics) -> Dict[str, Any]:
    """Convert CareerMetrics to a flat dictionary for DataFrame creation."""
    return {
        # Career info
        'career_id': metrics.career_id,
        'career_name': metrics.career_name,
        'parent_account_id': metrics.parent_account_id,
        'campus_name': metrics.campus_name,

        # Counts
        'total_courses': metrics.total_courses,
        'courses_with_students': metrics.courses_with_students,
        'courses_with_canvas_grades': metrics.courses_with_canvas_grades,
        'total_students': metrics.total_students,
        'total_students_with_grades': metrics.total_students_with_grades,

        # Scores
        'avg_lms_design_score': metrics.avg_lms_design_score,
        'avg_prediction_learning_score': metrics.avg_prediction_learning_score,
        'career_composite_score': metrics.career_composite_score,

        # Potential breakdown
        'high_potential_count': metrics.high_potential_count,
        'medium_potential_count': metrics.medium_potential_count,
        'low_potential_count': metrics.low_potential_count,
        'skip_count': metrics.skip_count,

        # Recommendation
        'recommendation': metrics.recommendation,
        'reasons': '; '.join(metrics.reasons) if metrics.reasons else ''
    }


if __name__ == '__main__':
    # Test the module
    from .canvas_client import CanvasClient

    client = CanvasClient()

    print("Testing career_analysis module...")
    print("=" * 60)

    # Test with PREGRADO account
    pregrado_id = 46

    print(f"\nDiscovering sub-accounts under PREGRADO (ID: {pregrado_id})...")
    careers = discover_careers(client, pregrado_id)
    print(f"Found {len(careers)} sub-accounts (campuses)")

    for career in careers[:5]:
        print(f"  - {career.get('id')}: {career.get('name')}")
