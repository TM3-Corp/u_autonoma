#!/usr/bin/env python3
"""
Career Potential Score (CPS) Analyzer
Reads RAW Parquet data and performs ALL analysis:
- Determines course recommendations (HIGH/MEDIUM/LOW/SKIP)
- Computes CPS metrics for careers

Usage:
    python analyze_cps.py --career-id 248
    python analyze_cps.py --career-id 248 --update-report
    python analyze_cps.py --all --input-dir data/careers
"""

import argparse
import os
import re
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional
from glob import glob


# Default configuration
DEFAULT_INPUT_DIR = 'exploratory/data/careers'
DEFAULT_REPORT_PATH = 'exploratory/discovery/career_potential_score.md'

# Analysis thresholds (from course_analysis.md)
MIN_STUDENTS_WITH_GRADES = 10  # Minimum students with grades for analysis
MIN_GRADE_VARIANCE = 15        # Grade variance > 15%
MIN_PASS_RATE = 0.2            # Failure rate 20-80% means pass rate 20-80%
MAX_PASS_RATE = 0.8
MIN_ASSIGNMENTS_HIGH = 5       # Minimum assignments for HIGH potential
MIN_ASSIGNMENTS_MEDIUM = 3     # Minimum assignments for MEDIUM potential


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
    has_activity: bool = False


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


def determine_recommendation(n_students_with_grades: int, grade_variance: float,
                              pass_rate: Optional[float], n_assignments: int,
                              has_activity: bool) -> str:
    """
    Determine course recommendation based on prediction potential criteria.

    Criteria from course_analysis.md:
    - Grades IN Canvas (n_students_with_grades >= 10)
    - Grade variance > 15%
    - Failure rate 20-80% (pass rate 20-80%)
    - Activity data exists

    Returns: 'HIGH POTENTIAL', 'MEDIUM POTENTIAL', 'LOW - ...', or 'SKIP - ...'
    """
    has_grades = n_students_with_grades >= MIN_STUDENTS_WITH_GRADES

    if has_grades and grade_variance > MIN_GRADE_VARIANCE:
        # Has grades with good variance
        if (n_assignments >= MIN_ASSIGNMENTS_HIGH and
            pass_rate is not None and
            MIN_PASS_RATE <= pass_rate <= MAX_PASS_RATE and
            has_activity):
            return 'HIGH POTENTIAL'
        elif n_assignments >= MIN_ASSIGNMENTS_MEDIUM:
            return 'MEDIUM POTENTIAL'
        else:
            return 'LOW - Few assignments'
    elif has_grades:
        return 'LOW - Low grade variance'
    else:
        return 'SKIP - No grades'


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


def load_career_data(career_id: int, input_dir: str) -> Optional[pd.DataFrame]:
    """Load career data from Parquet file."""
    combined_path = f'{input_dir}/career_{career_id}_combined.parquet'

    if not os.path.exists(combined_path):
        print(f"Error: File not found: {combined_path}")
        return None

    return pd.read_parquet(combined_path)


def df_to_course_metrics(df: pd.DataFrame) -> List[CourseMetrics]:
    """
    Convert DataFrame rows to CourseMetrics objects.
    Determines recommendation from raw data (analysis happens here).
    """
    courses = []
    for _, row in df.iterrows():
        n_students_with_grades = int(row['n_students_with_grades'])
        grade_variance = float(row['grade_variance'])
        pass_rate = float(row['pass_rate']) if pd.notna(row['pass_rate']) else None
        n_assignments = int(row['n_assignments'])
        has_activity = bool(row['has_activity'])

        # Determine recommendation (ANALYSIS HAPPENS HERE)
        recommendation = determine_recommendation(
            n_students_with_grades=n_students_with_grades,
            grade_variance=grade_variance,
            pass_rate=pass_rate,
            n_assignments=n_assignments,
            has_activity=has_activity
        )

        course = CourseMetrics(
            course_id=int(row['course_id']),
            name=row['name'],
            account_id=int(row['account_id']),
            term_id=int(row['term_id']) if pd.notna(row['term_id']) else None,
            term_name=row['term_name'],
            total_students=int(row['total_students']),
            n_students_with_grades=n_students_with_grades,
            grade_mean=float(row['grade_mean']),
            grade_variance=grade_variance,
            pass_rate=pass_rate,
            n_assignments=n_assignments,
            n_modules=int(row['n_modules']),
            has_activity=has_activity,
            recommendation=recommendation,
            has_grades=n_students_with_grades >= MIN_STUDENTS_WITH_GRADES
        )
        courses.append(course)
    return courses


def analyze_career_from_parquet(career_id: int, input_dir: str) -> Optional[CareerMetrics]:
    """Analyze a career from Parquet data."""
    df = load_career_data(career_id, input_dir)
    if df is None:
        return None

    # Get career name from data if available
    career_name = df['career_name'].iloc[0] if 'career_name' in df.columns else f'Career {career_id}'

    # Convert to CourseMetrics (recommendations are determined here)
    courses = df_to_course_metrics(df)

    # Create CareerMetrics
    metrics = CareerMetrics(
        account_id=career_id,
        career_name=career_name,
        courses=courses
    )

    # Aggregate metrics
    metrics.n_total_courses = len(courses)
    metrics.n_high_potential = sum(1 for c in courses if 'HIGH' in c.recommendation)
    metrics.n_medium_potential = sum(1 for c in courses if 'MEDIUM' in c.recommendation)
    metrics.n_low_potential = sum(1 for c in courses if 'LOW' in c.recommendation)
    metrics.n_skip = sum(1 for c in courses if 'SKIP' in c.recommendation)

    metrics.total_students = sum(c.total_students for c in courses)
    metrics.analyzable_students = sum(
        c.total_students for c in courses
        if 'HIGH' in c.recommendation or 'MEDIUM' in c.recommendation
    )
    metrics.students_with_grades = sum(c.n_students_with_grades for c in courses)

    metrics.courses_with_grades = sum(1 for c in courses if c.has_grades)

    # Average metrics (only from courses with grades)
    graded_courses = [c for c in courses if c.has_grades]
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


def print_summary(metrics: CareerMetrics):
    """Print analysis summary to console."""
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

    # Show HIGH POTENTIAL courses
    high_courses = [c for c in metrics.courses if 'HIGH' in c.recommendation]
    if high_courses:
        print(f"\nHIGH POTENTIAL Courses:")
        for c in sorted(high_courses, key=lambda x: x.grade_variance, reverse=True)[:5]:
            pass_str = f"{c.pass_rate:.0%}" if c.pass_rate is not None else "N/A"
            print(f"  - {c.course_id}: {c.name[:40]} (Var={c.grade_variance:.1f}, Pass={pass_str})")


def update_centralized_report(metrics: CareerMetrics, report_path: str):
    """Update the centralized CPS report with new career data."""
    if metrics.n_total_courses == 0:
        print(f"\nNo courses found - skipping report update")
        return

    # Read existing report
    with open(report_path, 'r') as f:
        content = f.read()

    # Update last updated date
    content = re.sub(
        r'\*\*Last Updated:\*\* \d{4}-\d{2}-\d{2}',
        f'**Last Updated:** {pd.Timestamp.now().strftime("%Y-%m-%d")}',
        content
    )

    # Find the ranking table and parse existing entries
    ranking_pattern = r'(\| Rank \| Account ID \| Career \| CPS \| HIGH \| MEDIUM \| LOW \| SKIP \| Total \| Analyzable Students \|\n\|[-\s|]+\n)((?:\|[^\n]+\n)*)'
    ranking_match = re.search(ranking_pattern, content)

    if ranking_match:
        header = ranking_match.group(1)
        existing_rows = ranking_match.group(2)

        # Parse existing rows into list of tuples (account_id, row_data)
        careers = []
        for row in existing_rows.strip().split('\n'):
            if row.strip():
                parts = [p.strip() for p in row.split('|')[1:-1]]
                if len(parts) >= 10:
                    acc_id = int(parts[1])
                    cps = float(parts[3])
                    careers.append((acc_id, cps, row))

        # Check if this career already exists
        existing_ids = [c[0] for c in careers]
        analyzable_pct = metrics.analyzable_students / max(metrics.total_students, 1) * 100
        new_row = f"| - | {metrics.account_id} | {metrics.career_name} | {metrics.career_potential_score:.1f} | {metrics.n_high_potential} | {metrics.n_medium_potential} | {metrics.n_low_potential} | {metrics.n_skip} | {metrics.n_total_courses} | {metrics.analyzable_students}/{metrics.total_students} ({analyzable_pct:.0f}%) |"

        if metrics.account_id in existing_ids:
            # Update existing entry
            careers = [(acc_id, cps, row) if acc_id != metrics.account_id else (metrics.account_id, metrics.career_potential_score, new_row)
                       for acc_id, cps, row in careers]
        else:
            # Add new entry
            careers.append((metrics.account_id, metrics.career_potential_score, new_row))

        # Sort by CPS descending and assign ranks
        careers.sort(key=lambda x: x[1], reverse=True)
        ranked_rows = []
        for i, (acc_id, cps, row) in enumerate(careers):
            # Update rank in row
            parts = row.split('|')
            parts[1] = f' {i+1} '
            ranked_rows.append('|'.join(parts))

        new_table = header + '\n'.join(ranked_rows) + '\n'
        content = content[:ranking_match.start()] + new_table + content[ranking_match.end():]

    # Find and update/add the detailed breakdown section for this career
    breakdown_header = "### Detailed Breakdown"
    breakdown_pos = content.find(breakdown_header)

    if breakdown_pos != -1:
        # Find the section for this career or the next section
        career_section_pattern = rf'#### {metrics.account_id} - [^\n]+\n(.*?)(?=\n#### \d|---|$)'
        career_match = re.search(career_section_pattern, content, re.DOTALL)

        # Build new career section
        high_courses = [c for c in metrics.courses if 'HIGH' in c.recommendation]
        career_section = f"""#### {metrics.account_id} - {metrics.career_name}

**CPS: {metrics.career_potential_score:.1f}/100** ({'Excellent' if metrics.career_potential_score >= 80 else 'Good' if metrics.career_potential_score >= 60 else 'Moderate' if metrics.career_potential_score >= 40 else 'Weak' if metrics.career_potential_score >= 20 else 'Poor'})

| Component | Score | Contribution |
|-----------|-------|--------------|
| HP Score | {metrics.hp_score:.1f} | {metrics.hp_score*0.30:.1f} |
| Quality Score | {metrics.quality_score:.1f} | {metrics.quality_score*0.25:.1f} |
| Coverage Score | {metrics.coverage_score:.1f} | {metrics.coverage_score*0.20:.1f} |
| Data Score | {metrics.data_score:.1f} | {metrics.data_score*0.15:.1f} |
| Diversity Score | {metrics.diversity_score:.1f} | {metrics.diversity_score*0.10:.1f} |

**Quality Metrics:**
- Avg grade variance: {metrics.avg_grade_variance:.1f}
- Avg pass rate: {metrics.avg_pass_rate:.0%}
- Pass rate std: {metrics.pass_rate_std:.2f}
- Avg assignments: {metrics.avg_assignments:.1f}
- Courses with grades: {metrics.courses_with_grades}/{metrics.n_total_courses}

**Top HIGH POTENTIAL Courses:**

"""
        if high_courses:
            career_section += "| Course ID | Name | Students | Variance | Pass Rate |\n"
            career_section += "|-----------|------|----------|----------|----------|\n"
            for c in sorted(high_courses, key=lambda x: x.grade_variance, reverse=True)[:5]:
                pass_str = f"{c.pass_rate:.0%}" if c.pass_rate is not None else "N/A"
                career_section += f"| {c.course_id} | {c.name[:40]} | {c.total_students} | {c.grade_variance:.1f} | {pass_str} |\n"
        else:
            career_section += "*No HIGH potential courses found.*\n"

        if career_match:
            # Replace existing section
            content = content[:career_match.start()] + career_section + content[career_match.end():]
        else:
            # Add new section before "---\n\n## Analysis Configuration"
            config_pos = content.find("---\n\n## Analysis Configuration")
            if config_pos != -1:
                content = content[:config_pos] + career_section + "\n" + content[config_pos:]

    # Write updated report
    with open(report_path, 'w') as f:
        f.write(content)

    print(f"Centralized report updated: {report_path}")


def find_all_careers(input_dir: str) -> List[int]:
    """Find all career IDs in the input directory."""
    pattern = f'{input_dir}/career_*_combined.parquet'
    files = glob(pattern)
    career_ids = []
    for f in files:
        match = re.search(r'career_(\d+)_combined\.parquet', f)
        if match:
            career_ids.append(int(match.group(1)))
    return sorted(career_ids)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Career Potential Score (CPS) from Parquet data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_cps.py --career-id 248
  python analyze_cps.py --career-id 248 --update-report
  python analyze_cps.py --all --input-dir data/careers
  python analyze_cps.py --all --update-report
        """
    )

    parser.add_argument('--career-id', type=int, default=None,
                        help='Career account ID to analyze')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all careers in input directory')
    parser.add_argument('--input-dir', type=str, default=DEFAULT_INPUT_DIR,
                        help=f'Input directory with Parquet files (default: {DEFAULT_INPUT_DIR})')
    parser.add_argument('--update-report', action='store_true',
                        help='Update centralized markdown report')
    parser.add_argument('--report-path', type=str, default=DEFAULT_REPORT_PATH,
                        help=f'Report file path (default: {DEFAULT_REPORT_PATH})')

    args = parser.parse_args()

    if not args.career_id and not args.all:
        parser.error("Either --career-id or --all must be specified")

    print("=" * 60)
    print("CPS ANALYZER")
    print("=" * 60)

    if args.all:
        career_ids = find_all_careers(args.input_dir)
        print(f"Found {len(career_ids)} careers to analyze")

        for career_id in career_ids:
            print(f"\n--- Analyzing career {career_id} ---")
            metrics = analyze_career_from_parquet(career_id, args.input_dir)
            if metrics:
                print_summary(metrics)
                if args.update_report:
                    update_centralized_report(metrics, args.report_path)
    else:
        metrics = analyze_career_from_parquet(args.career_id, args.input_dir)
        if metrics:
            print_summary(metrics)
            if args.update_report:
                update_centralized_report(metrics, args.report_path)

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
