"""
Dataset Comparison: PUC vs UA

Generates descriptive statistics and visualizations comparing
student grade data from two Chilean universities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
PUC_DATA_PATH = Path('/home/paul/projects/wave_analysis/puc_analysis/data/grades_with_failure.parquet')
UA_DATA_PATH = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_DIR = Path('/home/paul/projects/uautonoma/data/report/visualizations')
REPORT_DIR = Path('/home/paul/projects/uautonoma/data/report')

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Thresholds
PUC_PASS_THRESHOLD = 4.0  # Chilean 1-7 scale
UA_PASS_THRESHOLD = 57.0  # Percentage scale


def load_data():
    """Load both datasets"""
    puc = pd.read_parquet(PUC_DATA_PATH)
    ua = pd.read_csv(UA_DATA_PATH)

    # Add failure flag to UA
    ua['failed'] = ua['final_score'] < UA_PASS_THRESHOLD

    # Normalize grades to 0-100 scale for comparison
    puc['grade_normalized'] = (puc['grade'] - 1) / 6 * 100  # 1-7 → 0-100
    ua['grade_normalized'] = ua['final_score']

    # Add university label
    puc['university'] = 'PUC'
    ua['university'] = 'UA'

    return puc, ua


def compute_summary_stats(puc: pd.DataFrame, ua: pd.DataFrame) -> dict:
    """Compute summary statistics for both datasets"""

    puc_with_grades = puc[puc['grade'].notna()]
    ua_with_grades = ua[ua['final_score'].notna()]

    stats = {
        'puc': {
            'total_enrollments': len(puc),
            'with_grades': len(puc_with_grades),
            'unique_students': puc['user_lms_id'].nunique(),
            'unique_courses': puc['Sigla'].nunique(),
            'grade_scale': '1-7',
            'pass_threshold': f'>= {PUC_PASS_THRESHOLD}',
            'mean_grade': puc_with_grades['grade'].mean(),
            'std_grade': puc_with_grades['grade'].std(),
            'median_grade': puc_with_grades['grade'].median(),
            'min_grade': puc_with_grades['grade'].min(),
            'max_grade': puc_with_grades['grade'].max(),
            'failure_rate': puc_with_grades['failed'].mean() * 100,
            'failures': puc_with_grades['failed'].sum(),
        },
        'ua': {
            'total_enrollments': len(ua),
            'with_grades': len(ua_with_grades),
            'unique_students': ua['user_id'].nunique(),
            'unique_courses': ua['course_id'].nunique(),
            'grade_scale': '0-100%',
            'pass_threshold': f'>= {UA_PASS_THRESHOLD}%',
            'mean_grade': ua_with_grades['final_score'].mean(),
            'std_grade': ua_with_grades['final_score'].std(),
            'median_grade': ua_with_grades['final_score'].median(),
            'min_grade': ua_with_grades['final_score'].min(),
            'max_grade': ua_with_grades['final_score'].max(),
            'failure_rate': ua_with_grades['failed'].mean() * 100,
            'failures': ua_with_grades['failed'].sum(),
        }
    }

    return stats


def compute_course_stats(puc: pd.DataFrame, ua: pd.DataFrame) -> tuple:
    """Compute per-course statistics"""

    puc_course = puc[puc['grade'].notna()].groupby('Sigla').agg({
        'user_lms_id': 'count',
        'grade': ['mean', 'std', 'min', 'max'],
        'failed': ['sum', 'mean']
    }).round(2)
    puc_course.columns = ['students', 'mean_grade', 'std_grade', 'min_grade', 'max_grade', 'failures', 'failure_rate']
    puc_course['failure_rate'] = (puc_course['failure_rate'] * 100).round(1)
    puc_course = puc_course.sort_values('failure_rate', ascending=False)

    ua_course = ua[ua['final_score'].notna()].groupby('course_id').agg({
        'user_id': 'count',
        'final_score': ['mean', 'std', 'min', 'max'],
        'failed': ['sum', 'mean']
    }).round(2)
    ua_course.columns = ['students', 'mean_grade', 'std_grade', 'min_grade', 'max_grade', 'failures', 'failure_rate']
    ua_course['failure_rate'] = (ua_course['failure_rate'] * 100).round(1)
    ua_course = ua_course.sort_values('failure_rate', ascending=False)

    return puc_course, ua_course


def plot_grade_distributions(puc: pd.DataFrame, ua: pd.DataFrame):
    """Plot grade distributions for both universities"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # PUC histogram
    ax1 = axes[0]
    puc_grades = puc[puc['grade'].notna()]['grade']
    ax1.hist(puc_grades, bins=30, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax1.axvline(PUC_PASS_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Pass threshold ({PUC_PASS_THRESHOLD})')
    ax1.set_xlabel('Grade (1-7 scale)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('PUC - Grade Distribution\n(n=1,607)', fontsize=14, fontweight='bold')
    ax1.legend()

    # UA histogram
    ax2 = axes[1]
    ua_grades = ua[ua['final_score'].notna()]['final_score']
    ax2.hist(ua_grades, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    ax2.axvline(UA_PASS_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Pass threshold ({UA_PASS_THRESHOLD}%)')
    ax2.set_xlabel('Grade (0-100%)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('UA - Grade Distribution\n(n=373)', fontsize=14, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_grade_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_grade_distributions.png'}")


def plot_normalized_comparison(puc: pd.DataFrame, ua: pd.DataFrame):
    """Plot normalized grade comparison (both on 0-100 scale)"""

    fig, ax = plt.subplots(figsize=(10, 6))

    puc_norm = puc[puc['grade'].notna()]['grade_normalized']
    ua_norm = ua[ua['final_score'].notna()]['grade_normalized']

    # Overlapping histograms
    ax.hist(puc_norm, bins=30, alpha=0.6, label=f'PUC (n={len(puc_norm)})', color='#2ecc71', edgecolor='black')
    ax.hist(ua_norm, bins=30, alpha=0.6, label=f'UA (n={len(ua_norm)})', color='#3498db', edgecolor='black')

    # Normalized thresholds
    puc_threshold_norm = (PUC_PASS_THRESHOLD - 1) / 6 * 100  # 4.0 → 50%
    ua_threshold_norm = UA_PASS_THRESHOLD  # 57%

    ax.axvline(puc_threshold_norm, color='#27ae60', linestyle='--', linewidth=2, label=f'PUC pass (≈{puc_threshold_norm:.0f}%)')
    ax.axvline(ua_threshold_norm, color='#2980b9', linestyle='--', linewidth=2, label=f'UA pass ({ua_threshold_norm:.0f}%)')

    ax.set_xlabel('Normalized Grade (0-100%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Grade Distributions Comparison\n(Normalized to 0-100 scale)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_normalized_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_normalized_comparison.png'}")


def plot_failure_rates(puc_course: pd.DataFrame, ua_course: pd.DataFrame):
    """Plot failure rates by course for both universities"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PUC failure rates
    ax1 = axes[0]
    puc_sorted = puc_course.sort_values('failure_rate', ascending=True)
    colors = ['#e74c3c' if r > 10 else '#f39c12' if r > 5 else '#2ecc71' for r in puc_sorted['failure_rate']]
    bars1 = ax1.barh(range(len(puc_sorted)), puc_sorted['failure_rate'], color=colors, edgecolor='black')
    ax1.set_yticks(range(len(puc_sorted)))
    ax1.set_yticklabels(puc_sorted.index, fontsize=8)
    ax1.set_xlabel('Failure Rate (%)', fontsize=12)
    ax1.set_title(f'PUC - Failure Rate by Course\n(24 courses, avg={puc_course["failure_rate"].mean():.1f}%)', fontsize=12, fontweight='bold')
    ax1.axvline(puc_course['failure_rate'].mean(), color='black', linestyle='--', alpha=0.7)

    # UA failure rates
    ax2 = axes[1]
    ua_sorted = ua_course.sort_values('failure_rate', ascending=True)
    colors = ['#e74c3c' if r > 50 else '#f39c12' if r > 30 else '#2ecc71' for r in ua_sorted['failure_rate']]
    bars2 = ax2.barh(range(len(ua_sorted)), ua_sorted['failure_rate'], color=colors, edgecolor='black')
    ax2.set_yticks(range(len(ua_sorted)))
    ax2.set_yticklabels([f'Course {c}' for c in ua_sorted.index], fontsize=10)
    ax2.set_xlabel('Failure Rate (%)', fontsize=12)
    ax2.set_title(f'UA - Failure Rate by Course\n(10 courses, avg={ua_course["failure_rate"].mean():.1f}%)', fontsize=12, fontweight='bold')
    ax2.axvline(ua_course['failure_rate'].mean(), color='black', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_failure_rates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_failure_rates.png'}")


def plot_course_size_distribution(puc: pd.DataFrame, ua: pd.DataFrame):
    """Plot students per course distribution"""

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PUC
    puc_sizes = puc.groupby('Sigla')['user_lms_id'].count()
    ax1 = axes[0]
    ax1.hist(puc_sizes, bins=15, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax1.axvline(puc_sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {puc_sizes.mean():.0f}')
    ax1.set_xlabel('Students per Course', fontsize=12)
    ax1.set_ylabel('Number of Courses', fontsize=12)
    ax1.set_title(f'PUC - Course Size Distribution\n(min={puc_sizes.min()}, max={puc_sizes.max()})', fontsize=12, fontweight='bold')
    ax1.legend()

    # UA
    ua_sizes = ua.groupby('course_id')['user_id'].count()
    ax2 = axes[1]
    ax2.hist(ua_sizes, bins=10, edgecolor='black', alpha=0.7, color='#3498db')
    ax2.axvline(ua_sizes.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {ua_sizes.mean():.0f}')
    ax2.set_xlabel('Students per Course', fontsize=12)
    ax2.set_ylabel('Number of Courses', fontsize=12)
    ax2.set_title(f'UA - Course Size Distribution\n(min={ua_sizes.min()}, max={ua_sizes.max()})', fontsize=12, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_course_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_course_sizes.png'}")


def plot_boxplots(puc: pd.DataFrame, ua: pd.DataFrame):
    """Plot grade boxplots by course"""

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    # PUC boxplot
    puc_with_grades = puc[puc['grade'].notna()].copy()
    course_order = puc_with_grades.groupby('Sigla')['grade'].median().sort_values().index

    ax1 = axes[0]
    sns.boxplot(data=puc_with_grades, x='Sigla', y='grade', order=course_order, ax=ax1, palette='Greens')
    ax1.axhline(PUC_PASS_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Pass threshold ({PUC_PASS_THRESHOLD})')
    ax1.set_xlabel('Course', fontsize=12)
    ax1.set_ylabel('Grade (1-7)', fontsize=12)
    ax1.set_title('PUC - Grade Distribution by Course', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend()

    # UA boxplot
    ua_with_grades = ua[ua['final_score'].notna()].copy()
    ua_with_grades['course_label'] = 'Course ' + ua_with_grades['course_id'].astype(str)
    course_order_ua = ua_with_grades.groupby('course_label')['final_score'].median().sort_values().index

    ax2 = axes[1]
    sns.boxplot(data=ua_with_grades, x='course_label', y='final_score', order=course_order_ua, ax=ax2, palette='Blues')
    ax2.axhline(UA_PASS_THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Pass threshold ({UA_PASS_THRESHOLD}%)')
    ax2.set_xlabel('Course', fontsize=12)
    ax2.set_ylabel('Grade (%)', fontsize=12)
    ax2.set_title('UA - Grade Distribution by Course', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_grade_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_grade_boxplots.png'}")


def plot_summary_table(stats: dict):
    """Create a visual summary table"""

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    metrics = [
        ('Total Enrollments', f"{stats['puc']['total_enrollments']:,}", f"{stats['ua']['total_enrollments']:,}"),
        ('With Grades', f"{stats['puc']['with_grades']:,}", f"{stats['ua']['with_grades']:,}"),
        ('Unique Students', f"{stats['puc']['unique_students']:,}", f"{stats['ua']['unique_students']:,}"),
        ('Unique Courses', f"{stats['puc']['unique_courses']}", f"{stats['ua']['unique_courses']}"),
        ('Grade Scale', stats['puc']['grade_scale'], stats['ua']['grade_scale']),
        ('Pass Threshold', stats['puc']['pass_threshold'], stats['ua']['pass_threshold']),
        ('Mean Grade', f"{stats['puc']['mean_grade']:.2f}", f"{stats['ua']['mean_grade']:.2f}%"),
        ('Std Dev', f"{stats['puc']['std_grade']:.2f}", f"{stats['ua']['std_grade']:.2f}%"),
        ('Median Grade', f"{stats['puc']['median_grade']:.2f}", f"{stats['ua']['median_grade']:.2f}%"),
        ('Min Grade', f"{stats['puc']['min_grade']:.1f}", f"{stats['ua']['min_grade']:.1f}%"),
        ('Max Grade', f"{stats['puc']['max_grade']:.1f}", f"{stats['ua']['max_grade']:.1f}%"),
        ('Total Failures', f"{stats['puc']['failures']}", f"{stats['ua']['failures']}"),
        ('Failure Rate', f"{stats['puc']['failure_rate']:.1f}%", f"{stats['ua']['failure_rate']:.1f}%"),
    ]

    table_data = [[m[0], m[1], m[2]] for m in metrics]

    table = ax.table(
        cellText=table_data,
        colLabels=['Metric', 'PUC', 'UA'],
        cellLoc='center',
        loc='center',
        colWidths=[0.35, 0.25, 0.25]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j in range(3):
        table[(0, j)].set_facecolor('#34495e')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Style metric names
    for i in range(1, len(metrics) + 1):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 0)].set_text_props(fontweight='bold')

    # Highlight failure rate row
    failure_row = len(metrics)
    for j in range(3):
        table[(failure_row, j)].set_facecolor('#fadbd8')

    ax.set_title('Dataset Comparison: PUC vs UA\n', fontsize=16, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'dataset_summary_table.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'dataset_summary_table.png'}")


def generate_markdown_report(stats: dict, puc_course: pd.DataFrame, ua_course: pd.DataFrame):
    """Generate markdown comparison report"""

    report = f"""# Dataset Comparison: PUC vs UA

## Overview

This report compares student grade data from two Chilean universities:
- **PUC** (Pontificia Universidad Católica de Chile)
- **UA** (Universidad Autónoma de Chile)

## Summary Statistics

| Metric | PUC | UA |
|--------|-----|-----|
| Total Enrollments | {stats['puc']['total_enrollments']:,} | {stats['ua']['total_enrollments']:,} |
| With Grades | {stats['puc']['with_grades']:,} | {stats['ua']['with_grades']:,} |
| Unique Students | {stats['puc']['unique_students']:,} | {stats['ua']['unique_students']:,} |
| Unique Courses | {stats['puc']['unique_courses']} | {stats['ua']['unique_courses']} |
| Grade Scale | {stats['puc']['grade_scale']} | {stats['ua']['grade_scale']} |
| Pass Threshold | {stats['puc']['pass_threshold']} | {stats['ua']['pass_threshold']} |
| Mean Grade | {stats['puc']['mean_grade']:.2f} | {stats['ua']['mean_grade']:.2f}% |
| Std Dev | {stats['puc']['std_grade']:.2f} | {stats['ua']['std_grade']:.2f}% |
| Median Grade | {stats['puc']['median_grade']:.2f} | {stats['ua']['median_grade']:.2f}% |
| **Failure Rate** | **{stats['puc']['failure_rate']:.1f}%** | **{stats['ua']['failure_rate']:.1f}%** |

## Key Differences

1. **Sample Size**: PUC has 4.4x more enrollments than UA ({stats['puc']['total_enrollments']:,} vs {stats['ua']['total_enrollments']:,})
2. **Failure Rate**: UA has significantly higher failure rate ({stats['ua']['failure_rate']:.1f}% vs {stats['puc']['failure_rate']:.1f}%)
3. **Course Coverage**: PUC covers 24 courses, UA covers 10 courses
4. **Grade Variability**: UA shows higher variance (σ={stats['ua']['std_grade']:.2f}%) indicating more diverse outcomes

## Grade Distribution Comparison

When normalized to a 0-100 scale:
- PUC pass threshold (4.0 on 1-7 scale) ≈ 50%
- UA pass threshold is 57%

This means UA applies a **stricter passing standard** relative to their grade scale.

## PUC Course-Level Statistics

| Course | Students | Mean Grade | Failure Rate |
|--------|----------|------------|--------------|
"""

    for course, row in puc_course.head(10).iterrows():
        report += f"| {course} | {int(row['students'])} | {row['mean_grade']:.2f} | {row['failure_rate']:.1f}% |\n"

    if len(puc_course) > 10:
        report += f"| ... | ... | ... | ... |\n"
        for course, row in puc_course.tail(3).iterrows():
            report += f"| {course} | {int(row['students'])} | {row['mean_grade']:.2f} | {row['failure_rate']:.1f}% |\n"

    report += f"""
*Showing top 10 and bottom 3 courses by failure rate (24 total)*

## UA Course-Level Statistics

| Course ID | Students | Mean Grade | Failure Rate |
|-----------|----------|------------|--------------|
"""

    for course, row in ua_course.iterrows():
        report += f"| {course} | {int(row['students'])} | {row['mean_grade']:.2f}% | {row['failure_rate']:.1f}% |\n"

    report += f"""
## Visualizations

![Summary Table](visualizations/dataset_summary_table.png)

![Grade Distributions](visualizations/dataset_grade_distributions.png)

![Normalized Comparison](visualizations/dataset_normalized_comparison.png)

![Failure Rates by Course](visualizations/dataset_failure_rates.png)

![Course Size Distribution](visualizations/dataset_course_sizes.png)

![Grade Boxplots by Course](visualizations/dataset_grade_boxplots.png)

## Implications for Model Development

### Class Imbalance
- **PUC**: Highly imbalanced (6.3% failures) - may require oversampling or class weights
- **UA**: More balanced (39.9% failures) - better for binary classification

### Generalization
- Models trained on one university may not transfer well due to:
  - Different pass thresholds
  - Different failure rates
  - Different course types and student populations

### Recommended Approach
1. Train separate models per university initially
2. Test cross-university transfer with domain adaptation
3. Consider normalizing features within each course to reduce institutional bias

---

*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*
"""

    report_path = REPORT_DIR / 'DATASET_COMPARISON.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Saved: {report_path}")


def main():
    print("=" * 60)
    print("DATASET COMPARISON: PUC vs UA")
    print("=" * 60)

    # Load data
    print("\n1. Loading datasets...")
    puc, ua = load_data()
    print(f"   PUC: {len(puc)} enrollments")
    print(f"   UA: {len(ua)} enrollments")

    # Compute statistics
    print("\n2. Computing summary statistics...")
    stats = compute_summary_stats(puc, ua)

    print(f"\n   PUC Summary:")
    print(f"   - Students: {stats['puc']['unique_students']}")
    print(f"   - Courses: {stats['puc']['unique_courses']}")
    print(f"   - Mean grade: {stats['puc']['mean_grade']:.2f}")
    print(f"   - Failure rate: {stats['puc']['failure_rate']:.1f}%")

    print(f"\n   UA Summary:")
    print(f"   - Students: {stats['ua']['unique_students']}")
    print(f"   - Courses: {stats['ua']['unique_courses']}")
    print(f"   - Mean grade: {stats['ua']['mean_grade']:.2f}%")
    print(f"   - Failure rate: {stats['ua']['failure_rate']:.1f}%")

    # Course-level statistics
    print("\n3. Computing course-level statistics...")
    puc_course, ua_course = compute_course_stats(puc, ua)

    # Generate visualizations
    print("\n4. Generating visualizations...")
    plot_summary_table(stats)
    plot_grade_distributions(puc, ua)
    plot_normalized_comparison(puc, ua)
    plot_failure_rates(puc_course, ua_course)
    plot_course_size_distribution(puc, ua)
    plot_boxplots(puc, ua)

    # Generate markdown report
    print("\n5. Generating markdown report...")
    generate_markdown_report(stats, puc_course, ua_course)

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - Report: {REPORT_DIR / 'DATASET_COMPARISON.md'}")
    print(f"  - Visualizations: {OUTPUT_DIR}/dataset_*.png")


if __name__ == '__main__':
    main()
