"""
Course Analysis Module

This module provides functions to analyze individual courses for:
- LMS instructional design completeness
- Grade signal extraction and validation
- Prediction potential scoring
- Composite metrics generation

Usage:
    from scripts.discovery.course_analysis import analyze_course, CourseMetrics

    client = CanvasClient()
    metrics = analyze_course(client, course_id=86005)
"""

import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
import numpy as np

from .canvas_client import CanvasClient

logger = logging.getLogger(__name__)


# =============================================================================
# Constants and Thresholds
# =============================================================================

# Pass/Fail threshold (Chilean scale: 4.0/7.0 = 57%)
PASS_THRESHOLD = 57.0

# Minimum requirements for analyzable course
MIN_STUDENTS_WITH_GRADES = 5      # Minimum to consider "has Canvas grades"
MIN_STUDENTS_FOR_PREDICTION = 15  # Minimum for reliable prediction
MIN_GRADE_STD = 15.0              # Minimum grade variance (%)
MIN_FAIL_RATE = 0.10              # Minimum failure rate
MAX_FAIL_RATE = 0.90              # Maximum failure rate

# LMS Design Score weights
LMS_WEIGHTS = {
    'assignments': 0.30,    # Assignment count
    'quizzes': 0.20,        # Quiz count
    'modules': 0.15,        # Module count
    'files': 0.15,          # File count
    'discussions': 0.10,    # Discussion count
    'submissions': 0.10     # Assignments with online submission
}

# Prediction Score weights
PREDICTION_WEIGHTS = {
    'grade_variance': 0.35,   # StdDev contribution
    'fail_balance': 0.30,     # Balanced fail rate
    'sample_size': 0.20,      # Number of students
    'lms_design': 0.15        # LMS completeness
}


@dataclass
class LMSDesignMetrics:
    """LMS instructional design metrics for a course."""
    assignments_count: int = 0
    assignments_with_submissions: int = 0
    quizzes_count: int = 0
    modules_count: int = 0
    files_count: int = 0
    discussions_count: int = 0
    lms_design_score: float = 0.0


@dataclass
class GradeMetrics:
    """Grade-related metrics for a course."""
    total_enrolled: int = 0
    students_with_grades: int = 0
    has_canvas_grades: bool = False
    grade_mean: Optional[float] = None
    grade_std: Optional[float] = None
    grade_min: Optional[float] = None
    grade_max: Optional[float] = None
    pass_count: int = 0
    fail_count: int = 0
    pass_rate: Optional[float] = None
    fail_rate: Optional[float] = None
    grade_coverage: float = 0.0


@dataclass
class CourseMetrics:
    """Complete course analysis metrics."""
    # Course info
    course_id: int = 0
    course_name: str = ""
    account_id: int = 0
    term_id: Optional[int] = None
    term_name: str = ""

    # LMS Design
    lms: LMSDesignMetrics = field(default_factory=LMSDesignMetrics)

    # Grades
    grades: GradeMetrics = field(default_factory=GradeMetrics)

    # Prediction Potential
    prediction_learning_score: float = 0.0
    composite_potential_score: float = 0.0

    # Criteria checks
    meets_student_threshold: bool = False
    meets_grade_threshold: bool = False
    meets_variance_threshold: bool = False
    meets_fail_range: bool = False
    meets_lms_design_threshold: bool = False

    # Recommendation
    recommendation: str = "SKIP"
    reasons: List[str] = field(default_factory=list)

    # Error tracking
    error: Optional[str] = None


def extract_lms_metrics(client: CanvasClient, course_id: int) -> LMSDesignMetrics:
    """
    Extract LMS instructional design metrics from a course.

    Measures presence and volume of:
    - Assignments (and those with online submissions)
    - Quizzes (published)
    - Modules
    - Files
    - Discussion topics

    Args:
        client: CanvasClient instance
        course_id: Course ID to analyze

    Returns:
        LMSDesignMetrics dataclass with counts and normalized score
    """
    metrics = LMSDesignMetrics()

    try:
        # Get assignments
        assignments = client.get_assignments(course_id)
        metrics.assignments_count = len(assignments)

        # Count assignments with online submissions
        for a in assignments:
            sub_types = a.get('submission_types', [])
            if sub_types and sub_types != ['none'] and sub_types != ['not_graded']:
                metrics.assignments_with_submissions += 1

        # Get quizzes
        quizzes = client.get_quizzes(course_id)
        metrics.quizzes_count = len([q for q in quizzes if q.get('published', False)])

        # Get modules
        modules = client.get_modules(course_id)
        metrics.modules_count = len(modules)

        # Get files
        files = client.get_files(course_id)
        metrics.files_count = len(files)

        # Get discussions
        discussions = client.get_discussions(course_id)
        metrics.discussions_count = len(discussions)

        # Calculate normalized LMS Design Score
        # Each component normalized to [0, 1] then weighted
        score = 0.0

        # Assignments: 10+ is excellent
        score += LMS_WEIGHTS['assignments'] * min(metrics.assignments_count / 10, 1.0)

        # Quizzes: 5+ is excellent
        score += LMS_WEIGHTS['quizzes'] * min(metrics.quizzes_count / 5, 1.0)

        # Modules: 4+ is excellent
        score += LMS_WEIGHTS['modules'] * min(metrics.modules_count / 4, 1.0)

        # Files: 10+ is excellent
        score += LMS_WEIGHTS['files'] * min(metrics.files_count / 10, 1.0)

        # Discussions: 3+ is excellent
        score += LMS_WEIGHTS['discussions'] * min(metrics.discussions_count / 3, 1.0)

        # Submission-enabled assignments: 5+ is excellent
        score += LMS_WEIGHTS['submissions'] * min(
            metrics.assignments_with_submissions / 5, 1.0
        )

        metrics.lms_design_score = score

    except Exception as e:
        logger.warning(f"Error extracting LMS metrics for course {course_id}: {e}")

    return metrics


def extract_grade_metrics(
    client: CanvasClient,
    course_id: int,
    pass_threshold: float = PASS_THRESHOLD
) -> GradeMetrics:
    """
    Extract grade metrics from course enrollments.

    Uses current_score (actual performance on graded work) NOT final_score
    (which includes zeros for unsubmitted work).

    Args:
        client: CanvasClient instance
        course_id: Course ID to analyze
        pass_threshold: Grade threshold for pass/fail (default 57%)

    Returns:
        GradeMetrics dataclass with grade statistics
    """
    metrics = GradeMetrics()

    try:
        enrollments = client.get_enrollments(course_id, include_grades=True)
        metrics.total_enrolled = len(enrollments)

        # Extract grades using current_score
        grades = []
        for e in enrollments:
            grade_info = e.get('grades', {})
            # IMPORTANT: Use current_score, NOT final_score
            score = grade_info.get('current_score')
            if score is not None and score > 0:
                grades.append(float(score))

        metrics.students_with_grades = len(grades)
        metrics.has_canvas_grades = len(grades) >= MIN_STUDENTS_WITH_GRADES

        if metrics.students_with_grades > 0:
            metrics.grade_coverage = metrics.students_with_grades / metrics.total_enrolled

        if grades:
            metrics.grade_mean = np.mean(grades)
            metrics.grade_std = np.std(grades)
            metrics.grade_min = np.min(grades)
            metrics.grade_max = np.max(grades)

            # Calculate pass/fail
            metrics.pass_count = sum(1 for g in grades if g >= pass_threshold)
            metrics.fail_count = len(grades) - metrics.pass_count
            metrics.pass_rate = metrics.pass_count / len(grades)
            metrics.fail_rate = metrics.fail_count / len(grades)

    except Exception as e:
        logger.warning(f"Error extracting grade metrics for course {course_id}: {e}")

    return metrics


def calculate_prediction_score(
    lms_metrics: LMSDesignMetrics,
    grade_metrics: GradeMetrics
) -> float:
    """
    Calculate prediction learning score based on course characteristics.

    Higher scores indicate better potential for building prediction models.

    Args:
        lms_metrics: LMS design metrics
        grade_metrics: Grade metrics

    Returns:
        Prediction learning score in [0, 1]
    """
    if not grade_metrics.has_canvas_grades:
        return 0.0

    score = 0.0

    # Grade variance contribution (StdDev)
    if grade_metrics.grade_std:
        # StdDev of 30% is excellent, normalize
        variance_score = min(grade_metrics.grade_std / 30.0, 1.0)
        score += PREDICTION_WEIGHTS['grade_variance'] * variance_score

    # Fail balance contribution
    # Optimal fail rate around 30-40% for binary classification
    if grade_metrics.fail_rate is not None:
        if MIN_FAIL_RATE <= grade_metrics.fail_rate <= MAX_FAIL_RATE:
            # Closer to 35% is better
            balance = 1.0 - abs(grade_metrics.fail_rate - 0.35) * 2
            balance = max(0.0, balance)

            # Bonus for ideal range (20-80%)
            if 0.20 <= grade_metrics.fail_rate <= 0.80:
                balance = min(balance + 0.2, 1.0)

            score += PREDICTION_WEIGHTS['fail_balance'] * balance

    # Sample size contribution
    # 50 students is excellent, normalize
    sample_score = min(grade_metrics.students_with_grades / 50.0, 1.0)
    score += PREDICTION_WEIGHTS['sample_size'] * sample_score

    # LMS design contribution
    score += PREDICTION_WEIGHTS['lms_design'] * lms_metrics.lms_design_score

    return score


def analyze_course(
    client: CanvasClient,
    course_id: int,
    include_lms_details: bool = True
) -> CourseMetrics:
    """
    Perform complete analysis of a course's prediction potential.

    Args:
        client: CanvasClient instance
        course_id: Course ID to analyze
        include_lms_details: Whether to fetch LMS design metrics (slower)

    Returns:
        CourseMetrics dataclass with complete analysis
    """
    metrics = CourseMetrics(course_id=course_id)

    try:
        # Get course info
        course_data = client.get_course(course_id)
        if not course_data:
            metrics.error = "Course not accessible"
            metrics.recommendation = "SKIP"
            return metrics

        metrics.course_name = course_data.get('name', 'Unknown')
        metrics.account_id = course_data.get('account_id', 0)
        metrics.term_id = course_data.get('enrollment_term_id')
        metrics.term_name = course_data.get('term', {}).get('name', 'Unknown')

        # Extract grade metrics (always needed)
        metrics.grades = extract_grade_metrics(client, course_id)

        # Extract LMS metrics (optional, slower)
        if include_lms_details:
            metrics.lms = extract_lms_metrics(client, course_id)
        else:
            # Minimal LMS check - just assignments
            assignments = client.get_assignments(course_id)
            metrics.lms.assignments_count = len(assignments)

        # Check criteria
        metrics.meets_student_threshold = (
            metrics.grades.students_with_grades >= MIN_STUDENTS_FOR_PREDICTION
        )
        metrics.meets_grade_threshold = metrics.grades.has_canvas_grades
        metrics.meets_variance_threshold = (
            (metrics.grades.grade_std or 0) >= MIN_GRADE_STD
        )
        metrics.meets_fail_range = (
            metrics.grades.fail_rate is not None and
            MIN_FAIL_RATE <= metrics.grades.fail_rate <= MAX_FAIL_RATE
        )
        metrics.meets_lms_design_threshold = metrics.lms.assignments_count >= 3

        # Calculate scores
        metrics.prediction_learning_score = calculate_prediction_score(
            metrics.lms, metrics.grades
        )

        # Composite score (weighted average of LMS design + prediction potential)
        if metrics.grades.has_canvas_grades:
            metrics.composite_potential_score = (
                0.3 * metrics.lms.lms_design_score +
                0.7 * metrics.prediction_learning_score
            )
        else:
            metrics.composite_potential_score = 0.0

        # Generate recommendation
        metrics.recommendation, metrics.reasons = generate_recommendation(metrics)

    except Exception as e:
        logger.error(f"Error analyzing course {course_id}: {e}")
        metrics.error = str(e)
        metrics.recommendation = "SKIP"
        metrics.reasons = [f"Analysis error: {e}"]

    return metrics


def generate_recommendation(metrics: CourseMetrics) -> tuple:
    """
    Generate recommendation and reasons based on metrics.

    Returns:
        Tuple of (recommendation, reasons_list)
    """
    reasons = []

    # Check for disqualifying conditions
    if not metrics.grades.has_canvas_grades:
        reasons.append("No grades in Canvas (likely uses external gradebook)")
        return "SKIP", reasons

    if metrics.grades.total_enrolled == 0:
        reasons.append("No students enrolled")
        return "SKIP", reasons

    # Count met criteria
    criteria_met = sum([
        metrics.meets_student_threshold,
        metrics.meets_grade_threshold,
        metrics.meets_variance_threshold,
        metrics.meets_fail_range,
        metrics.meets_lms_design_threshold
    ])

    # Add specific reasons
    if not metrics.meets_student_threshold:
        reasons.append(
            f"Not enough students with grades "
            f"({metrics.grades.students_with_grades} < {MIN_STUDENTS_FOR_PREDICTION})"
        )

    if not metrics.meets_variance_threshold:
        std = metrics.grades.grade_std or 0
        reasons.append(f"Low grade variance (StdDev {std:.1f}% < {MIN_GRADE_STD}%)")

    if not metrics.meets_fail_range:
        if metrics.grades.fail_rate is not None:
            if metrics.grades.fail_rate < MIN_FAIL_RATE:
                reasons.append(
                    f"Too few failures ({metrics.grades.fail_rate:.0%} < {MIN_FAIL_RATE:.0%})"
                )
            elif metrics.grades.fail_rate > MAX_FAIL_RATE:
                reasons.append(
                    f"Too many failures ({metrics.grades.fail_rate:.0%} > {MAX_FAIL_RATE:.0%})"
                )
        else:
            reasons.append("No fail rate available")

    if not metrics.meets_lms_design_threshold:
        reasons.append(f"Few assignments ({metrics.lms.assignments_count} < 3)")

    # Positive reasons
    if metrics.meets_student_threshold and metrics.meets_variance_threshold:
        reasons.append("Good sample size and grade variance")

    if metrics.meets_fail_range:
        reasons.append(f"Balanced fail rate ({metrics.grades.fail_rate:.0%})")

    # Determine recommendation
    if criteria_met >= 4:
        return "HIGH", reasons
    elif criteria_met >= 3:
        return "MEDIUM", reasons
    elif criteria_met >= 2:
        return "LOW", reasons
    else:
        return "SKIP", reasons


def course_metrics_to_dict(metrics: CourseMetrics) -> Dict[str, Any]:
    """Convert CourseMetrics to a flat dictionary for DataFrame creation."""
    return {
        # Course info
        'course_id': metrics.course_id,
        'course_name': metrics.course_name,
        'account_id': metrics.account_id,
        'term_id': metrics.term_id,
        'term_name': metrics.term_name,

        # LMS Design
        'assignments_count': metrics.lms.assignments_count,
        'assignments_with_submissions': metrics.lms.assignments_with_submissions,
        'quizzes_count': metrics.lms.quizzes_count,
        'modules_count': metrics.lms.modules_count,
        'files_count': metrics.lms.files_count,
        'discussions_count': metrics.lms.discussions_count,
        'lms_design_score': metrics.lms.lms_design_score,

        # Grades
        'total_enrolled': metrics.grades.total_enrolled,
        'students_with_grades': metrics.grades.students_with_grades,
        'has_canvas_grades': metrics.grades.has_canvas_grades,
        'grade_mean': metrics.grades.grade_mean,
        'grade_std': metrics.grades.grade_std,
        'grade_min': metrics.grades.grade_min,
        'grade_max': metrics.grades.grade_max,
        'pass_count': metrics.grades.pass_count,
        'fail_count': metrics.grades.fail_count,
        'pass_rate': metrics.grades.pass_rate,
        'fail_rate': metrics.grades.fail_rate,
        'grade_coverage': metrics.grades.grade_coverage,

        # Scores
        'prediction_learning_score': metrics.prediction_learning_score,
        'composite_potential_score': metrics.composite_potential_score,

        # Criteria
        'meets_student_threshold': metrics.meets_student_threshold,
        'meets_grade_threshold': metrics.meets_grade_threshold,
        'meets_variance_threshold': metrics.meets_variance_threshold,
        'meets_fail_range': metrics.meets_fail_range,
        'meets_lms_design_threshold': metrics.meets_lms_design_threshold,

        # Recommendation
        'recommendation': metrics.recommendation,
        'reasons': '; '.join(metrics.reasons) if metrics.reasons else '',
        'error': metrics.error
    }


if __name__ == '__main__':
    # Test the module
    from .canvas_client import CanvasClient

    client = CanvasClient()

    print("Testing course_analysis module...")
    print("=" * 60)

    test_course_id = 86005

    print(f"\nAnalyzing course {test_course_id}...")
    metrics = analyze_course(client, test_course_id)

    print(f"\nCourse: {metrics.course_name}")
    print(f"Account: {metrics.account_id}")
    print(f"Term: {metrics.term_name}")

    print(f"\nLMS Design:")
    print(f"  Assignments: {metrics.lms.assignments_count}")
    print(f"  Quizzes: {metrics.lms.quizzes_count}")
    print(f"  Modules: {metrics.lms.modules_count}")
    print(f"  LMS Score: {metrics.lms.lms_design_score:.3f}")

    print(f"\nGrades:")
    print(f"  Enrolled: {metrics.grades.total_enrolled}")
    print(f"  With grades: {metrics.grades.students_with_grades}")
    print(f"  Mean: {metrics.grades.grade_mean:.1f}%" if metrics.grades.grade_mean else "  Mean: N/A")
    print(f"  StdDev: {metrics.grades.grade_std:.1f}%" if metrics.grades.grade_std else "  StdDev: N/A")
    print(f"  Pass rate: {metrics.grades.pass_rate:.0%}" if metrics.grades.pass_rate else "  Pass rate: N/A")

    print(f"\nScores:")
    print(f"  Prediction Score: {metrics.prediction_learning_score:.3f}")
    print(f"  Composite Score: {metrics.composite_potential_score:.3f}")

    print(f"\nRecommendation: {metrics.recommendation}")
    print(f"Reasons:")
    for r in metrics.reasons:
        print(f"  - {r}")
