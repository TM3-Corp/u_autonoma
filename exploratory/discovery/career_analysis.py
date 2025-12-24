#!/usr/bin/env python3
"""
Career Analysis & Scoring System
Analyzes all courses for a specific career and computes Career Potential Score (CPS)
"""

import requests
import os
import time
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Configuration
TARGET_TERMS = [336, 322]  # 2nd Sem 2025 (current) and 1st Sem 2025 (recent)
MIN_STUDENTS = 20
PASS_THRESHOLD = 57  # Chilean grading: 57% = 4.0 = passing


@dataclass
class CourseMetrics:
    """Metrics for a single course."""
    course_id: int
    name: str
    account_id: int
    term_id: Optional[int]
    term_name: str
    total_students: int
    n_students_with_grades: int = 0
    grade_mean: float = 0.0
    grade_variance: float = 0.0
    pass_rate: Optional[float] = None
    n_assignments: int = 0
    n_modules: int = 0
    recommendation: str = 'SKIP'
    has_grades: bool = False


@dataclass
class CareerMetrics:
    """Aggregated metrics for a career (sub-account)."""
    account_id: int
    career_name: str

    # Raw counts
    n_total_courses: int = 0
    n_high_potential: int = 0
    n_medium_potential: int = 0
    n_low_potential: int = 0
    n_skip: int = 0

    # Student counts
    total_students: int = 0
    analyzable_students: int = 0
    students_with_grades: int = 0

    # Quality metrics
    courses_with_grades: int = 0
    avg_grade_variance: float = 0.0
    avg_pass_rate: float = 0.0
    pass_rate_std: float = 0.0
    avg_assignments: float = 0.0
    avg_grade_completeness: float = 0.0

    # Computed scores (0-100)
    hp_score: float = 0.0
    quality_score: float = 0.0
    coverage_score: float = 0.0
    data_score: float = 0.0
    diversity_score: float = 0.0

    # Final score
    career_potential_score: float = 0.0

    # Flags
    flags: List[str] = field(default_factory=list)

    # Course details
    courses: List[CourseMetrics] = field(default_factory=list)


def safe_request(url, params=None, max_retries=3):
    """Make a request with rate limit handling."""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, params=params, timeout=30)
            remaining = int(float(r.headers.get('X-Rate-Limit-Remaining', 700)))

            if r.status_code == 403:
                print(f"Rate limited! Waiting 60s... (attempt {attempt + 1})")
                time.sleep(60)
                continue

            if r.status_code == 200:
                # Adaptive delay based on quota
                if remaining < 100:
                    time.sleep(10)
                elif remaining < 300:
                    time.sleep(2)
                elif remaining < 500:
                    time.sleep(1)
                else:
                    time.sleep(0.3)
                return r.json(), remaining
            else:
                print(f"Error {r.status_code}: {r.text[:100]}")
                return None, remaining

        except Exception as e:
            print(f"Request failed (attempt {attempt + 1}): {e}")
            time.sleep(2 ** attempt)

    return None, 0


def get_career_name(account_id):
    """Get career name from account ID."""
    data, _ = safe_request(f'{API_URL}/api/v1/accounts/{account_id}')
    if data:
        return data.get('name', f'Account {account_id}')
    return f'Account {account_id}'


def get_all_career_courses(account_id, term_ids=None, min_students=20):
    """Get ALL courses from a career (sub-account) with minimum student count."""
    all_courses = []

    for term_id in (term_ids or [None]):
        page = 1
        while True:
            params = {
                'per_page': 100,
                'page': page,
                'include[]': ['total_students', 'term'],
                'with_enrollments': True
            }
            if term_id:
                params['enrollment_term_id'] = term_id

            url = f'{API_URL}/api/v1/accounts/{account_id}/courses'
            data, remaining = safe_request(url, params)

            if not data:
                break

            for course in data:
                if course.get('total_students', 0) >= min_students:
                    all_courses.append({
                        'course_id': course['id'],
                        'name': course['name'],
                        'account_id': course.get('account_id', account_id),
                        'students': course.get('total_students', 0),
                        'term_name': course.get('term', {}).get('name', 'Unknown'),
                        'term_id': course.get('enrollment_term_id')
                    })

            print(f"  Term {term_id}, Page {page}: {len(data)} courses fetched")

            if len(data) < 100:
                break
            page += 1

    return all_courses


def analyze_course_potential(course_id) -> CourseMetrics:
    """Analyze a course for analytical potential."""
    result = CourseMetrics(
        course_id=course_id,
        name='',
        account_id=0,
        term_id=None,
        term_name='',
        total_students=0
    )

    # 1. Get enrollments with grades
    enrollments, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        params={
            'type[]': 'StudentEnrollment',
            'per_page': 100,
            'include[]': 'grades'
        }
    )

    if enrollments:
        grades = [
            e['grades'].get('final_score')
            for e in enrollments
            if e.get('grades', {}).get('final_score') is not None
        ]

        if len(grades) >= 10:
            result.has_grades = True
            result.n_students_with_grades = len(grades)
            result.grade_variance = float(np.std(grades))
            result.grade_mean = float(np.mean(grades))
            result.pass_rate = sum(1 for g in grades if g >= PASS_THRESHOLD) / len(grades)

    # 2. Count assignments
    assignments, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/assignments',
        params={'per_page': 100}
    )
    if assignments:
        result.n_assignments = len(assignments)

    # 3. Count modules
    modules, _ = safe_request(
        f'{API_URL}/api/v1/courses/{course_id}/modules',
        params={'per_page': 100}
    )
    if modules:
        result.n_modules = len(modules)

    # 4. Determine recommendation
    if result.has_grades and result.grade_variance > 10:
        if result.n_assignments >= 5 and 0.2 <= (result.pass_rate or 0) <= 0.8:
            result.recommendation = 'HIGH POTENTIAL'
        elif result.n_assignments >= 3:
            result.recommendation = 'MEDIUM POTENTIAL'
        else:
            result.recommendation = 'LOW - Few assignments'
    elif result.has_grades:
        result.recommendation = 'LOW - Low grade variance'
    else:
        result.recommendation = 'SKIP - No grades'

    return result


def compute_hp_score(metrics: CareerMetrics) -> float:
    """Compute High-Potential Score (30% weight)."""
    if metrics.n_total_courses == 0:
        return 0.0

    # Weighted count: HIGH = 1.0, MEDIUM = 0.4
    weighted_hp_count = metrics.n_high_potential + (0.4 * metrics.n_medium_potential)

    # Density ratio
    hp_density = weighted_hp_count / metrics.n_total_courses

    # Combined score with logarithmic scaling
    hp_raw = (math.log2(weighted_hp_count + 1) * 15) + (hp_density * 50)

    return min(100, hp_raw)


def compute_quality_score(metrics: CareerMetrics) -> float:
    """Compute Quality Score (25% weight)."""
    if metrics.n_total_courses == 0:
        return 0.0

    tier_weights = {
        'HIGH': 1.0,
        'MEDIUM': 0.5,
        'LOW': 0.1,
        'SKIP': 0.0
    }

    weighted_sum = (
        metrics.n_high_potential * tier_weights['HIGH'] +
        metrics.n_medium_potential * tier_weights['MEDIUM'] +
        metrics.n_low_potential * tier_weights['LOW'] +
        metrics.n_skip * tier_weights['SKIP']
    )

    quality_density = weighted_sum / metrics.n_total_courses

    # Tier presence bonus
    tiers_present = sum([
        1 if metrics.n_high_potential > 0 else 0,
        1 if metrics.n_medium_potential > 0 else 0
    ])
    tier_bonus = 10 * tiers_present

    return min(100, (quality_density * 70) + tier_bonus)


def compute_coverage_score(metrics: CareerMetrics) -> float:
    """Compute Coverage Score (20% weight)."""
    if metrics.total_students == 0:
        return 0.0

    coverage_ratio = metrics.analyzable_students / metrics.total_students

    # Size bonus (logarithmic)
    size_bonus = min(20, math.log2(metrics.analyzable_students + 1) * 3)

    return min(100, (coverage_ratio * 80) + size_bonus)


def compute_data_score(metrics: CareerMetrics) -> float:
    """Compute Data Score (15% weight)."""
    if metrics.n_total_courses == 0:
        return 0.0

    # Grade availability
    grade_availability = metrics.courses_with_grades / metrics.n_total_courses

    # Average completeness
    avg_completeness = metrics.avg_grade_completeness

    # Assignment factor (capped at 20 assignments)
    assignment_factor = min(1.0, metrics.avg_assignments / 20) if metrics.avg_assignments > 0 else 0

    return (grade_availability * 40) + (avg_completeness * 40) + (assignment_factor * 20)


def compute_diversity_score(metrics: CareerMetrics) -> float:
    """Compute Diversity Score (10% weight)."""
    # Pass rate balance (ideal = 50%)
    pass_balance = 100 - abs(metrics.avg_pass_rate - 0.5) * 200 if metrics.avg_pass_rate else 50
    pass_balance = max(0, pass_balance)

    # Variance quality (higher = better)
    variance_quality = min(100, metrics.avg_grade_variance * 3)

    # Cross-course diversity
    cross_diversity = min(30, metrics.pass_rate_std * 100) if metrics.pass_rate_std else 0

    return (pass_balance * 0.5) + (variance_quality * 0.3) + (cross_diversity * 0.2)


def compute_cps(metrics: CareerMetrics) -> float:
    """Compute Career Potential Score (CPS)."""
    metrics.hp_score = compute_hp_score(metrics)
    metrics.quality_score = compute_quality_score(metrics)
    metrics.coverage_score = compute_coverage_score(metrics)
    metrics.data_score = compute_data_score(metrics)
    metrics.diversity_score = compute_diversity_score(metrics)

    # Weighted sum
    cps = (
        metrics.hp_score * 0.30 +
        metrics.quality_score * 0.25 +
        metrics.coverage_score * 0.20 +
        metrics.data_score * 0.15 +
        metrics.diversity_score * 0.10
    )

    # Edge case handling
    if metrics.n_total_courses < 3:
        cps *= 0.5
        metrics.flags.append("Limited Data")

    if metrics.n_high_potential + metrics.n_medium_potential == 0:
        cps = min(cps, 20)
        metrics.flags.append("Needs Investigation")

    if metrics.courses_with_grades == 0:
        cps = 0
        metrics.flags.append("External Grading")

    if metrics.pass_rate_std < 0.05 and metrics.courses_with_grades > 0:
        metrics.diversity_score *= 0.5
        metrics.flags.append("Low Variance")

    metrics.career_potential_score = cps
    return cps


def analyze_career(account_id: int, career_name: str = None) -> CareerMetrics:
    """Analyze all courses in a career and compute CPS."""
    if career_name is None:
        career_name = get_career_name(account_id)

    print(f"\n{'=' * 60}")
    print(f"ANALYZING CAREER: {career_name} (Account {account_id})")
    print(f"{'=' * 60}")

    metrics = CareerMetrics(
        account_id=account_id,
        career_name=career_name
    )

    # Fetch all courses
    print(f"\n--- Fetching courses with {MIN_STUDENTS}+ students ---")
    courses = get_all_career_courses(account_id, term_ids=TARGET_TERMS, min_students=MIN_STUDENTS)

    print(f"\nFound {len(courses)} candidate courses")

    if not courses:
        metrics.flags.append("No Courses Found")
        return metrics

    # Analyze each course
    print(f"\n--- Analyzing course potential ---")
    for i, course in enumerate(courses):
        course_id = course['course_id']
        print(f"[{i+1}/{len(courses)}] Analyzing {course_id}: {course['name'][:40]}...")

        analysis = analyze_course_potential(course_id)
        analysis.name = course['name']
        analysis.account_id = course['account_id']
        analysis.total_students = course['students']
        analysis.term_id = course['term_id']
        analysis.term_name = course['term_name']

        metrics.courses.append(analysis)

        if 'HIGH' in analysis.recommendation:
            print(f"  *** HIGH POTENTIAL: Variance={analysis.grade_variance:.1f}, Pass Rate={analysis.pass_rate:.0%}")
        elif 'MEDIUM' in analysis.recommendation:
            print(f"  ** MEDIUM: Variance={analysis.grade_variance:.1f}, Pass Rate={analysis.pass_rate:.0%}")

    # Aggregate metrics
    metrics.n_total_courses = len(metrics.courses)
    metrics.n_high_potential = sum(1 for c in metrics.courses if 'HIGH' in c.recommendation)
    metrics.n_medium_potential = sum(1 for c in metrics.courses if 'MEDIUM' in c.recommendation)
    metrics.n_low_potential = sum(1 for c in metrics.courses if 'LOW' in c.recommendation)
    metrics.n_skip = sum(1 for c in metrics.courses if 'SKIP' in c.recommendation)

    metrics.total_students = sum(c.total_students for c in metrics.courses)
    metrics.analyzable_students = sum(
        c.total_students for c in metrics.courses
        if 'HIGH' in c.recommendation or 'MEDIUM' in c.recommendation
    )
    metrics.students_with_grades = sum(c.n_students_with_grades for c in metrics.courses)

    metrics.courses_with_grades = sum(1 for c in metrics.courses if c.has_grades)

    # Average metrics (only from courses with grades)
    graded_courses = [c for c in metrics.courses if c.has_grades]
    if graded_courses:
        metrics.avg_grade_variance = np.mean([c.grade_variance for c in graded_courses])
        pass_rates = [c.pass_rate for c in graded_courses if c.pass_rate is not None]
        if pass_rates:
            metrics.avg_pass_rate = np.mean(pass_rates)
            metrics.pass_rate_std = np.std(pass_rates) if len(pass_rates) > 1 else 0
        metrics.avg_assignments = np.mean([c.n_assignments for c in graded_courses])
        metrics.avg_grade_completeness = np.mean([
            c.n_students_with_grades / c.total_students
            for c in graded_courses if c.total_students > 0
        ])

    # Compute CPS
    compute_cps(metrics)

    return metrics


def generate_course_csv(metrics: CareerMetrics, output_path: str):
    """Generate CSV with all course details."""
    rows = []
    for c in metrics.courses:
        rows.append({
            'course_id': c.course_id,
            'name': c.name,
            'account_id': c.account_id,
            'term_id': c.term_id,
            'term_name': c.term_name,
            'total_students': c.total_students,
            'n_students_with_grades': c.n_students_with_grades,
            'grade_mean': c.grade_mean,
            'grade_variance': c.grade_variance,
            'pass_rate': c.pass_rate,
            'n_assignments': c.n_assignments,
            'n_modules': c.n_modules,
            'recommendation': c.recommendation
        })

    df = pd.DataFrame(rows)

    # Sort by recommendation priority
    rec_order = {
        'HIGH POTENTIAL': 1,
        'MEDIUM POTENTIAL': 2,
        'LOW - Few assignments': 3,
        'LOW - Low grade variance': 4,
        'SKIP - No grades': 5
    }
    df['rec_order'] = df['recommendation'].map(rec_order)
    df = df.sort_values(['rec_order', 'grade_variance'], ascending=[True, False])
    df = df.drop('rec_order', axis=1)

    df.to_csv(output_path, index=False)
    print(f"\nCSV saved to: {output_path}")
    return df


def generate_markdown_report(metrics: CareerMetrics, output_path: str):
    """Generate detailed markdown report."""
    lines = [
        f"# Career Analysis: {metrics.career_name}",
        f"",
        f"**Account ID:** {metrics.account_id}",
        f"**Career Potential Score (CPS):** {metrics.career_potential_score:.1f}/100",
        f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"",
        "---",
        "",
        "## Summary",
        "",
        "### Course Distribution",
        "",
        "| Tier | Count | % |",
        "|------|-------|---|",
        f"| HIGH POTENTIAL | {metrics.n_high_potential} | {metrics.n_high_potential/max(metrics.n_total_courses,1)*100:.0f}% |",
        f"| MEDIUM POTENTIAL | {metrics.n_medium_potential} | {metrics.n_medium_potential/max(metrics.n_total_courses,1)*100:.0f}% |",
        f"| LOW | {metrics.n_low_potential} | {metrics.n_low_potential/max(metrics.n_total_courses,1)*100:.0f}% |",
        f"| SKIP (No grades) | {metrics.n_skip} | {metrics.n_skip/max(metrics.n_total_courses,1)*100:.0f}% |",
        f"| **TOTAL** | **{metrics.n_total_courses}** | 100% |",
        "",
        "### Student Coverage",
        "",
        f"- **Total students:** {metrics.total_students}",
        f"- **Analyzable students (HIGH+MEDIUM):** {metrics.analyzable_students} ({metrics.analyzable_students/max(metrics.total_students,1)*100:.0f}%)",
        f"- **Students with grades:** {metrics.students_with_grades}",
        "",
        "---",
        "",
        "## CPS Component Scores",
        "",
        "| Component | Score | Weight | Contribution |",
        "|-----------|-------|--------|--------------|",
        f"| HP Score | {metrics.hp_score:.1f} | 30% | {metrics.hp_score*0.30:.1f} |",
        f"| Quality Score | {metrics.quality_score:.1f} | 25% | {metrics.quality_score*0.25:.1f} |",
        f"| Coverage Score | {metrics.coverage_score:.1f} | 20% | {metrics.coverage_score*0.20:.1f} |",
        f"| Data Score | {metrics.data_score:.1f} | 15% | {metrics.data_score*0.15:.1f} |",
        f"| Diversity Score | {metrics.diversity_score:.1f} | 10% | {metrics.diversity_score*0.10:.1f} |",
        f"| **CPS Total** | | | **{metrics.career_potential_score:.1f}** |",
        "",
    ]

    if metrics.flags:
        lines.extend([
            "### Flags",
            "",
            *[f"- {flag}" for flag in metrics.flags],
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Quality Metrics",
        "",
        f"- **Avg grade variance:** {metrics.avg_grade_variance:.1f}",
        f"- **Avg pass rate:** {metrics.avg_pass_rate:.0%}" if metrics.avg_pass_rate else "- **Avg pass rate:** N/A",
        f"- **Pass rate std dev:** {metrics.pass_rate_std:.2f}",
        f"- **Avg assignments:** {metrics.avg_assignments:.1f}",
        f"- **Avg grade completeness:** {metrics.avg_grade_completeness:.0%}",
        f"- **Courses with grades:** {metrics.courses_with_grades}/{metrics.n_total_courses}",
        "",
        "---",
        "",
        "## HIGH POTENTIAL Courses",
        "",
    ])

    high_courses = [c for c in metrics.courses if 'HIGH' in c.recommendation]
    if high_courses:
        lines.append("| Course ID | Name | Students | Variance | Pass Rate | Assignments |")
        lines.append("|-----------|------|----------|----------|-----------|-------------|")
        for c in sorted(high_courses, key=lambda x: x.grade_variance, reverse=True):
            pass_str = f"{c.pass_rate:.0%}" if c.pass_rate is not None else "N/A"
            lines.append(f"| {c.course_id} | {c.name[:40]} | {c.total_students} | {c.grade_variance:.1f} | {pass_str} | {c.n_assignments} |")
    else:
        lines.append("*No HIGH potential courses found.*")

    lines.extend([
        "",
        "## MEDIUM POTENTIAL Courses",
        "",
    ])

    medium_courses = [c for c in metrics.courses if 'MEDIUM' in c.recommendation]
    if medium_courses:
        lines.append("| Course ID | Name | Students | Variance | Pass Rate | Assignments |")
        lines.append("|-----------|------|----------|----------|-----------|-------------|")
        for c in sorted(medium_courses, key=lambda x: x.grade_variance, reverse=True):
            pass_str = f"{c.pass_rate:.0%}" if c.pass_rate is not None else "N/A"
            lines.append(f"| {c.course_id} | {c.name[:40]} | {c.total_students} | {c.grade_variance:.1f} | {pass_str} | {c.n_assignments} |")
    else:
        lines.append("*No MEDIUM potential courses found.*")

    lines.extend([
        "",
        "## LOW Potential Courses",
        "",
    ])

    low_courses = [c for c in metrics.courses if 'LOW' in c.recommendation]
    if low_courses:
        lines.append("| Course ID | Name | Students | Variance | Pass Rate | Issue |")
        lines.append("|-----------|------|----------|----------|-----------|-------|")
        for c in sorted(low_courses, key=lambda x: x.grade_variance, reverse=True):
            pass_str = f"{c.pass_rate:.0%}" if c.pass_rate is not None else "N/A"
            issue = "Few assignments" if "Few" in c.recommendation else "Low variance"
            lines.append(f"| {c.course_id} | {c.name[:40]} | {c.total_students} | {c.grade_variance:.1f} | {pass_str} | {issue} |")
    else:
        lines.append("*No LOW potential courses.*")

    lines.extend([
        "",
        "## SKIP Courses (No Grades)",
        "",
    ])

    skip_courses = [c for c in metrics.courses if 'SKIP' in c.recommendation]
    if skip_courses:
        lines.append("| Course ID | Name | Students | Assignments | Modules |")
        lines.append("|-----------|------|----------|-------------|---------|")
        for c in sorted(skip_courses, key=lambda x: x.total_students, reverse=True):
            lines.append(f"| {c.course_id} | {c.name[:40]} | {c.total_students} | {c.n_assignments} | {c.n_modules} |")
    else:
        lines.append("*No SKIP courses.*")

    lines.extend([
        "",
        "---",
        "",
        "## CPS Interpretation",
        "",
        "| Range | Interpretation |",
        "|-------|----------------|",
        "| 80-100 | Excellent - Prioritize for pilot |",
        "| 60-79 | Good - Strong candidate |",
        "| 40-59 | Moderate - Investigate specific courses |",
        "| 20-39 | Weak - May need curriculum changes |",
        "| 0-19 | Poor - Likely external grading |",
        "",
    ])

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Markdown report saved to: {output_path}")


def main():
    """Main entry point - analyze career 248."""
    print("=" * 60)
    print("CAREER ANALYSIS SYSTEM")
    print("=" * 60)

    # Test connection
    r = requests.get(f'{API_URL}/api/v1/users/self', headers=headers)
    if r.status_code == 200:
        user = r.json()
        print(f"Connected as: {user.get('name', 'Unknown')}")
        print(f"Rate Limit: {r.headers.get('X-Rate-Limit-Remaining', 'N/A')}")
    else:
        print(f"Connection failed: {r.status_code}")
        return

    # Analyze career 248 (Ingeniería Civil Informática)
    metrics = analyze_career(248, "Ingeniería Civil Informática")

    # Print summary
    print("\n" + "=" * 60)
    print("CAREER ANALYSIS SUMMARY")
    print("=" * 60)

    print(f"\nCareer: {metrics.career_name}")
    print(f"Career Potential Score (CPS): {metrics.career_potential_score:.1f}/100")
    print(f"\nCourse Distribution:")
    print(f"  HIGH POTENTIAL:   {metrics.n_high_potential}")
    print(f"  MEDIUM POTENTIAL: {metrics.n_medium_potential}")
    print(f"  LOW:              {metrics.n_low_potential}")
    print(f"  SKIP (no grades): {metrics.n_skip}")
    print(f"  TOTAL:            {metrics.n_total_courses}")

    print(f"\nStudent Coverage:")
    print(f"  Total students:     {metrics.total_students}")
    print(f"  Analyzable:         {metrics.analyzable_students} ({metrics.analyzable_students/max(metrics.total_students,1)*100:.0f}%)")

    print(f"\nComponent Scores:")
    print(f"  HP Score:        {metrics.hp_score:.1f}")
    print(f"  Quality Score:   {metrics.quality_score:.1f}")
    print(f"  Coverage Score:  {metrics.coverage_score:.1f}")
    print(f"  Data Score:      {metrics.data_score:.1f}")
    print(f"  Diversity Score: {metrics.diversity_score:.1f}")

    if metrics.flags:
        print(f"\nFlags: {', '.join(metrics.flags)}")

    # Generate outputs
    csv_path = 'exploratory/data/career_248_full_analysis.csv'
    md_path = 'exploratory/data/career_248_report.md'

    generate_course_csv(metrics, csv_path)
    generate_markdown_report(metrics, md_path)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
