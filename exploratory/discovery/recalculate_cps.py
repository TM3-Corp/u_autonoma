#!/usr/bin/env python3
"""
Recalculate CPS with intervention-focused formula.
Uses only CSV data - no API calls.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from math import log2
from typing import List

DATA_DIR = Path(__file__).parent.parent / 'data'

# Career name mapping
CAREER_NAMES = {
    247: "Psicología",
    248: "Ingeniería Civil Informática",
    249: "Medicina",
    250: "Aud. e Ing En Control de Gest.",
    251: "Ingeniería Civil Química",
    253: "Derecho",
    254: "Kinesiología",
    260: "Nutrición y Dietética",
    262: "Química y Farmacia",
    263: "Ingeniería Civil Industrial",
    311: "Ingeniería Comercial",
    719: "Ingeniería en Control de Gestión",
}


@dataclass
class CareerMetrics:
    account_id: int
    career_name: str
    n_total_courses: int = 0
    n_high: int = 0
    n_medium: int = 0
    n_low: int = 0
    n_skip: int = 0
    total_students: int = 0
    analyzable_students: int = 0
    courses_with_grades: int = 0
    avg_variance: float = 0.0
    avg_pass_rate: float = 0.0
    pass_rate_std: float = 0.0
    avg_assignments: float = 0.0
    avg_completeness: float = 0.0
    # Component scores
    hp_score: float = 0.0
    quality_score: float = 0.0
    coverage_score: float = 0.0
    data_score: float = 0.0
    intervention_score: float = 0.0  # Replaces diversity_score
    cps: float = 0.0
    # New intervention metrics
    intervention_potential: int = 0  # Raw count of potential failures
    avg_failure_rate: float = 0.0


def load_career_data(account_id: int) -> pd.DataFrame:
    """Load CSV data for a career."""
    csv_path = DATA_DIR / f'career_{account_id}_courses.csv'
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_metrics_from_csv(account_id: int) -> CareerMetrics:
    """Compute all metrics from CSV data."""
    df = load_career_data(account_id)

    if df.empty:
        return CareerMetrics(
            account_id=account_id,
            career_name=CAREER_NAMES.get(account_id, f"Career {account_id}")
        )

    metrics = CareerMetrics(
        account_id=account_id,
        career_name=CAREER_NAMES.get(account_id, f"Career {account_id}")
    )

    # Count tiers
    metrics.n_total_courses = len(df)
    metrics.n_high = len(df[df['recommendation'].str.contains('HIGH POTENTIAL', na=False)])
    metrics.n_medium = len(df[df['recommendation'].str.contains('MEDIUM POTENTIAL', na=False)])
    metrics.n_low = len(df[df['recommendation'].str.startswith('LOW', na=False)])
    metrics.n_skip = len(df[df['recommendation'].str.startswith('SKIP', na=False)])

    # Student counts
    metrics.total_students = df['total_students'].sum()

    # Analyzable = students in HIGH or MEDIUM courses
    high_medium_mask = df['recommendation'].str.contains('HIGH POTENTIAL|MEDIUM POTENTIAL', na=False, regex=True)
    metrics.analyzable_students = df.loc[high_medium_mask, 'total_students'].sum()

    # Courses with grades (n_students_with_grades > 0)
    metrics.courses_with_grades = len(df[df['n_students_with_grades'] > 0])

    # Average metrics (only from courses with grades)
    graded = df[df['n_students_with_grades'] > 0]
    if len(graded) > 0:
        metrics.avg_variance = graded['grade_variance'].mean()
        metrics.avg_pass_rate = graded['pass_rate'].mean()
        metrics.pass_rate_std = graded['pass_rate'].std() if len(graded) > 1 else 0.0
        metrics.avg_assignments = graded['n_assignments'].mean()

        # Completeness: students_with_grades / total_students per course
        completeness_values = graded['n_students_with_grades'] / graded['total_students']
        metrics.avg_completeness = completeness_values.mean()

    # Failure rate and intervention potential
    metrics.avg_failure_rate = 1 - metrics.avg_pass_rate
    metrics.intervention_potential = int(metrics.analyzable_students * metrics.avg_failure_rate)

    return metrics


def compute_hp_score(m: CareerMetrics) -> float:
    """HP Score: High-potential course strength."""
    if m.n_total_courses == 0:
        return 0.0

    weighted_hp = m.n_high + (0.4 * m.n_medium)
    hp_density = weighted_hp / m.n_total_courses

    score = log2(weighted_hp + 1) * 15 + hp_density * 50
    return min(100, score)


def compute_quality_score(m: CareerMetrics) -> float:
    """Quality Score: Course tier distribution."""
    if m.n_total_courses == 0:
        return 0.0

    tier_weights = {'high': 1.0, 'medium': 0.5, 'low': 0.1, 'skip': 0.0}
    weighted_sum = (
        m.n_high * tier_weights['high'] +
        m.n_medium * tier_weights['medium'] +
        m.n_low * tier_weights['low'] +
        m.n_skip * tier_weights['skip']
    )

    quality_density = weighted_sum / m.n_total_courses

    # Tier bonus: 10 points per tier that has courses
    tiers_with_courses = sum([
        1 if m.n_high > 0 else 0,
        1 if m.n_medium > 0 else 0,
        1 if m.n_low > 0 else 0,
    ])
    tier_bonus = 10 * tiers_with_courses

    return min(100, quality_density * 70 + tier_bonus)


def compute_coverage_score(m: CareerMetrics) -> float:
    """Coverage Score: Student analyzability."""
    if m.total_students == 0:
        return 0.0

    coverage_ratio = m.analyzable_students / m.total_students
    size_bonus = min(20, log2(m.analyzable_students + 1) * 3)

    return min(100, coverage_ratio * 80 + size_bonus)


def compute_data_score(m: CareerMetrics) -> float:
    """Data Score: Grade data availability."""
    if m.n_total_courses == 0:
        return 0.0

    grade_availability = m.courses_with_grades / m.n_total_courses
    assignment_factor = min(1.0, m.avg_assignments / 20)

    return (
        grade_availability * 40 +
        m.avg_completeness * 40 +
        assignment_factor * 20
    )


def compute_intervention_score(m: CareerMetrics) -> float:
    """
    NEW: Intervention Score - replaces Diversity Score.

    Optimizes for:
    - High failure rate (more students to help)
    - But enough passes to train ML model (>15% minority class)

    Sweet spot: 20-40% pass rate (60-80% failure)
    """
    pass_rate = m.avg_pass_rate
    failure_rate = 1 - pass_rate

    # Handle edge cases
    if m.courses_with_grades == 0:
        return 0.0

    # Intervention factor based on pass rate
    if pass_rate < 0.15:
        # Too few passes - can't train properly
        intervention_factor = (pass_rate / 0.15) * 40
    elif pass_rate > 0.85:
        # Too few failures - little intervention value
        intervention_factor = (failure_rate / 0.15) * 40
    elif 0.20 <= pass_rate <= 0.40:
        # SWEET SPOT: 60-80% failure, still trainable
        intervention_factor = 100
    elif pass_rate < 0.20:
        # Very high failure (80-85%), slightly reduced trainability
        intervention_factor = 70 + ((pass_rate - 0.15) / 0.05) * 30
    else:
        # Moderate failure (15-60%), still valuable but decreasing
        # Linear decrease from 40% to 85% pass rate
        intervention_factor = 100 - ((pass_rate - 0.40) / 0.45) * 60

    # Variance quality (unchanged - need variance to predict)
    variance_quality = min(100, m.avg_variance * 3)

    # Cross-course diversity (helps model generalization)
    cross_diversity = min(30, m.pass_rate_std * 100) if m.pass_rate_std > 0 else 0

    # Weighted combination (intervention-focused)
    score = (
        intervention_factor * 0.50 +  # 50% weight on intervention potential
        variance_quality * 0.35 +      # 35% on variance (ML signal)
        cross_diversity * 0.15         # 15% on diversity
    )

    return min(100, score)


def compute_cps(m: CareerMetrics) -> float:
    """
    Compute intervention-focused CPS.

    Weights adjusted for intervention goals:
    - HP Score: 25% (was 30%)
    - Quality Score: 20% (was 25%)
    - Coverage Score: 20% (unchanged)
    - Data Score: 15% (unchanged)
    - Intervention Score: 20% (was Diversity 10%)
    """
    m.hp_score = compute_hp_score(m)
    m.quality_score = compute_quality_score(m)
    m.coverage_score = compute_coverage_score(m)
    m.data_score = compute_data_score(m)
    m.intervention_score = compute_intervention_score(m)

    cps = (
        m.hp_score * 0.25 +
        m.quality_score * 0.20 +
        m.coverage_score * 0.20 +
        m.data_score * 0.15 +
        m.intervention_score * 0.20
    )

    # Edge case handling
    if m.n_total_courses < 3:
        cps *= 0.5  # Limited data penalty

    if m.n_high + m.n_medium == 0:
        cps = min(cps, 20)  # Cap if no analyzable courses

    if m.courses_with_grades == 0:
        cps = 0  # No grades = can't do anything

    m.cps = round(cps, 1)
    return m.cps


def analyze_all_careers() -> List[CareerMetrics]:
    """Analyze all careers from CSV files."""
    results = []

    for account_id in CAREER_NAMES.keys():
        metrics = compute_metrics_from_csv(account_id)
        if metrics.n_total_courses > 0:
            compute_cps(metrics)
            results.append(metrics)

    # Sort by CPS descending
    results.sort(key=lambda x: x.cps, reverse=True)
    return results


def generate_report(results: List[CareerMetrics]) -> str:
    """Generate markdown report."""
    lines = [
        "# Career Potential Score (CPS) - Intervention Focused",
        "",
        "**Purpose:** Rank careers for early failure prediction and student intervention.",
        "",
        f"**Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        "",
        "---",
        "",
        "## CPS Formula (Intervention-Focused)",
        "",
        "```",
        "CPS = (HP_Score × 0.25) + (Quality_Score × 0.20) + (Coverage_Score × 0.20) +",
        "      (Data_Score × 0.15) + (Intervention_Score × 0.20)",
        "```",
        "",
        "### Key Change: Intervention Score",
        "",
        "Replaces Diversity Score. Optimizes for **high failure rates** (more students to help)",
        "while maintaining trainability (need some passes for ML model).",
        "",
        "| Pass Rate | Failure Rate | Intervention Factor | Reason |",
        "|:---------:|:------------:|:-------------------:|--------|",
        "| 20-40% | **60-80%** | 100 (max) | Sweet spot: high intervention, trainable |",
        "| 15-20% | 80-85% | 70-100 | Very high failure, slightly reduced trainability |",
        "| 40-85% | 15-60% | 40-100 | Moderate failure, decreasing intervention value |",
        "| >85% | <15% | 0-40 | Too few failures to help |",
        "| <15% | >85% | 0-40 | Too few passes to train model |",
        "",
        "---",
        "",
        "## Career Ranking",
        "",
        "| Rank | ID | Career | CPS | HIGH | MED | LOW | SKIP | Intervention Potential |",
        "|------|----|--------|-----|------|-----|-----|------|:----------------------:|",
    ]

    for i, m in enumerate(results, 1):
        lines.append(
            f"| {i} | {m.account_id} | {m.career_name} | {m.cps} | "
            f"{m.n_high} | {m.n_medium} | {m.n_low} | {m.n_skip} | "
            f"{m.intervention_potential} students |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## CPS Component Scores",
        "",
        "| ID | Career | HP | Quality | Coverage | Data | Intervention | CPS |",
        "|----|--------|---:|--------:|---------:|-----:|-------------:|----:|",
    ])

    for m in results:
        lines.append(
            f"| {m.account_id} | {m.career_name} | {m.hp_score:.1f} | {m.quality_score:.1f} | "
            f"{m.coverage_score:.1f} | {m.data_score:.1f} | {m.intervention_score:.1f} | {m.cps} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Quality & Intervention Metrics",
        "",
        "| ID | Career | Variance | Pass Rate | Failure Rate | Analyzable | Potential Failures |",
        "|----|--------|:--------:|:---------:|:------------:|:----------:|:------------------:|",
    ])

    for m in results:
        pass_pct = f"{m.avg_pass_rate:.0%}" if m.courses_with_grades > 0 else "N/A"
        fail_pct = f"{m.avg_failure_rate:.0%}" if m.courses_with_grades > 0 else "N/A"
        lines.append(
            f"| {m.account_id} | {m.career_name} | {m.avg_variance:.1f} | {pass_pct} | "
            f"**{fail_pct}** | {m.analyzable_students} | **{m.intervention_potential}** |"
        )

    # Add top courses section
    lines.extend([
        "",
        "---",
        "",
        "## Top Courses by Career",
        "",
    ])

    for m in results:
        df = load_career_data(m.account_id)
        if df.empty:
            continue

        high_courses = df[df['recommendation'].str.contains('HIGH POTENTIAL', na=False)]
        high_courses = high_courses.sort_values('grade_variance', ascending=False)

        lines.append(f"### {m.account_id} - {m.career_name} (CPS: {m.cps})")
        lines.append("")

        if len(high_courses) > 0:
            lines.append("| Course ID | Name | Students | Variance | Pass Rate | Failures |")
            lines.append("|-----------|------|:--------:|:--------:|:---------:|:--------:|")

            for _, row in high_courses.head(5).iterrows():
                failures = int(row['total_students'] * (1 - row['pass_rate']))
                lines.append(
                    f"| {row['course_id']} | {row['name'][:45]} | {row['total_students']} | "
                    f"{row['grade_variance']:.1f} | {row['pass_rate']:.0%} | **{failures}** |"
                )
            lines.append("")
        else:
            lines.append("*No HIGH potential courses found.*")
            lines.append("")

    # Add configuration section
    lines.extend([
        "---",
        "",
        "## Analysis Configuration",
        "",
        "- **Minimum students per course:** 20",
        "- **Pass threshold:** 57% (Chilean 4.0 scale)",
        "- **Sweet spot pass rate:** 20-40% (60-80% failure)",
        "- **Target terms:** 336 (2nd Sem 2025), 322 (1st Sem 2025)",
        "",
        "## Component Definitions",
        "",
        "| Component | Weight | What it Measures |",
        "|-----------|--------|------------------|",
        "| **HP Score** | 25% | High-potential course quantity + density |",
        "| **Quality Score** | 20% | Course tier distribution |",
        "| **Coverage Score** | 20% | % of students in analyzable courses |",
        "| **Data Score** | 15% | Grade availability and completeness |",
        "| **Intervention Score** | 20% | Failure rate optimization for intervention |",
        "",
        "### Course Tier Definitions",
        "",
        "| Tier | Criteria | ML Value |",
        "|------|----------|----------|",
        "| **HIGH POTENTIAL** | Variance > 10, Pass rate 20-80%, Assignments >= 5 | Ideal for training |",
        "| **MEDIUM POTENTIAL** | Variance > 10, Assignments >= 3 | Usable with caveats |",
        "| **LOW** | Variance <= 10 OR few assignments | Limited predictive value |",
        "| **SKIP** | No grade data available | Cannot use |",
    ])

    return "\n".join(lines)


def main():
    """Main entry point."""
    print("Recalculating CPS with intervention-focused formula...")
    print(f"Reading CSVs from: {DATA_DIR}")
    print()

    results = analyze_all_careers()

    print("=" * 70)
    print("INTERVENTION-FOCUSED CPS RANKING")
    print("=" * 70)
    print()
    print(f"{'Rank':<5} {'ID':<5} {'Career':<35} {'CPS':<6} {'Failures':<10}")
    print("-" * 70)

    for i, m in enumerate(results, 1):
        print(f"{i:<5} {m.account_id:<5} {m.career_name:<35} {m.cps:<6} {m.intervention_potential:<10}")

    print()
    print("=" * 70)
    print("COMPONENT SCORES")
    print("=" * 70)
    print()
    print(f"{'ID':<5} {'HP':<8} {'Quality':<8} {'Cover':<8} {'Data':<8} {'Interv':<8} {'CPS':<6}")
    print("-" * 70)

    for m in results:
        print(f"{m.account_id:<5} {m.hp_score:<8.1f} {m.quality_score:<8.1f} "
              f"{m.coverage_score:<8.1f} {m.data_score:<8.1f} {m.intervention_score:<8.1f} {m.cps:<6}")

    # Generate and save report
    report = generate_report(results)
    output_path = Path(__file__).parent / 'career_potential_score.md'
    output_path.write_text(report)
    print()
    print(f"Report saved to: {output_path}")


if __name__ == '__main__':
    main()
