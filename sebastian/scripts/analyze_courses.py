#!/usr/bin/env python3
"""
Course Analysis Script - Statistical Analysis & Correlation Study

Analyzes the course discovery CSV to identify best courses for predictive modeling.

Features:
- Correlation matrix between all metrics
- Feature importance for prediction potential
- Cluster analysis of course types
- Campus/career comparisons
- Exportable rankings and recommendations

Usage:
    python analyze_courses.py [--input FILE] [--output-dir DIR] [--top N]

Output:
    - Console summary with recommendations
    - Correlation heatmap (PNG)
    - Feature distributions (PNG)
    - Top courses ranked list (CSV)
"""

import os
import sys
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default paths
DEFAULT_INPUT = "data/discovery/course_analysis_latest.csv"
DEFAULT_OUTPUT_DIR = "data/discovery/analysis"

# Thresholds for "good" prediction candidates
THRESHOLDS = {
    'min_students': 15,
    'min_grade_coverage': 50.0,      # % students with grades
    'min_grade_variance': 10.0,       # std dev
    'ideal_fail_rate_min': 0.15,
    'ideal_fail_rate_max': 0.85,
    'min_activity_coverage': 30.0,    # % students with activity
    'min_assignments': 3,
}

# Key metrics for analysis
GRADE_METRICS = [
    'students_with_current_score',
    'current_score_coverage',
    'final_score_coverage',
    'grade_mean',
    'grade_std',
    'grade_min',
    'grade_max',
    'failure_rate',
]

DESIGN_METRICS = [
    'assignment_count',
    'graded_assignment_count',
    'quiz_count',
    'module_count',
    'file_count',
    'discussion_count',
    'page_count',
]

ACTIVITY_METRICS = [
    'total_page_views',
    'total_participations',
    'students_with_activity',
    'avg_page_views',
    'avg_participations',
]

SCORE_METRICS = [
    'grade_availability_score',
    'grade_variance_score',
    'class_balance_score',
    'design_richness_score',
    'activity_score',
    'prediction_potential_score',
]

ALL_NUMERIC_METRICS = GRADE_METRICS + DESIGN_METRICS + ACTIVITY_METRICS + SCORE_METRICS


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load and prepare the course analysis CSV."""
    print(f"Loading data from: {filepath}")

    df = pd.read_csv(filepath)

    # Basic cleaning
    df = df.fillna(0)

    # Ensure numeric columns are numeric
    for col in ALL_NUMERIC_METRICS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    print(f"Loaded {len(df)} courses")
    return df


def filter_viable_courses(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to courses that meet minimum viability thresholds."""
    viable = df[
        (df['total_students'] >= THRESHOLDS['min_students']) &
        (df['current_score_coverage'] >= THRESHOLDS['min_grade_coverage']) &
        (df['grade_std'] >= THRESHOLDS['min_grade_variance'])
    ].copy()

    print(f"Viable courses (meeting thresholds): {len(viable)} / {len(df)}")
    return viable


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for all numeric metrics."""
    # Select only columns that exist and are numeric
    available_cols = [c for c in ALL_NUMERIC_METRICS if c in df.columns]

    corr_matrix = df[available_cols].corr()
    return corr_matrix


def analyze_prediction_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze which metrics most strongly predict the prediction_potential_score."""
    target = 'prediction_potential_score'

    if target not in df.columns:
        print("Warning: prediction_potential_score not found")
        return pd.DataFrame()

    # Only analyze courses with some prediction potential
    df_valid = df[df[target] > 0].copy()

    if len(df_valid) < 10:
        print("Warning: Not enough courses with prediction potential for analysis")
        return pd.DataFrame()

    results = []

    # Compute correlation with target for each metric
    feature_cols = [c for c in GRADE_METRICS + DESIGN_METRICS + ACTIVITY_METRICS
                    if c in df.columns and c != target]

    for col in feature_cols:
        try:
            # Pearson correlation
            corr, p_value = stats.pearsonr(df_valid[col], df_valid[target])

            # Spearman correlation (rank-based, better for non-linear)
            spearman_corr, spearman_p = stats.spearmanr(df_valid[col], df_valid[target])

            results.append({
                'metric': col,
                'pearson_corr': corr,
                'pearson_p': p_value,
                'spearman_corr': spearman_corr,
                'spearman_p': spearman_p,
                'abs_correlation': abs(corr),
                'significant': p_value < 0.05,
            })
        except Exception as e:
            print(f"  Warning: Could not compute correlation for {col}: {e}")

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('abs_correlation', ascending=False)

    return df_results


def analyze_by_category(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analyze metrics grouped by different categories."""
    results = {}

    # By account (career/program)
    if 'account_id' in df.columns:
        by_account = df.groupby('account_id').agg({
            'course_id': 'count',
            'total_students': 'sum',
            'students_with_current_score': 'sum',
            'prediction_potential_score': 'mean',
            'grade_std': 'mean',
            'failure_rate': 'mean',
        }).rename(columns={'course_id': 'course_count'})
        by_account = by_account.sort_values('prediction_potential_score', ascending=False)
        results['by_account'] = by_account

    # By term
    if 'term_name' in df.columns:
        by_term = df.groupby('term_name').agg({
            'course_id': 'count',
            'total_students': 'sum',
            'prediction_potential_score': 'mean',
            'current_score_coverage': 'mean',
        }).rename(columns={'course_id': 'course_count'})
        by_term = by_term.sort_values('prediction_potential_score', ascending=False)
        results['by_term'] = by_term

    return results


def compute_composite_ranking(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a comprehensive ranking considering multiple factors."""
    df_ranked = df.copy()

    # Normalize key metrics to 0-100 scale
    def normalize(series):
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val == 0:
            return series * 0
        return (series - min_val) / (max_val - min_val) * 100

    # Create normalized scores
    if 'grade_std' in df.columns:
        df_ranked['norm_variance'] = normalize(df_ranked['grade_std'])

    if 'current_score_coverage' in df.columns:
        df_ranked['norm_coverage'] = df_ranked['current_score_coverage']  # Already 0-100

    if 'students_with_current_score' in df.columns:
        df_ranked['norm_sample_size'] = normalize(
            df_ranked['students_with_current_score'].clip(upper=100)
        )

    # Class balance score (0 = all pass or all fail, 100 = 50/50)
    if 'failure_rate' in df.columns:
        df_ranked['norm_balance'] = 100 - (abs(df_ranked['failure_rate'] - 0.5) * 200)
        df_ranked['norm_balance'] = df_ranked['norm_balance'].clip(lower=0)

    # Compute weighted composite score
    weights = {
        'norm_coverage': 0.25,
        'norm_variance': 0.25,
        'norm_balance': 0.20,
        'norm_sample_size': 0.15,
        'design_richness_score': 0.10,
        'activity_score': 0.05,
    }

    df_ranked['composite_score'] = 0
    for metric, weight in weights.items():
        if metric in df_ranked.columns:
            df_ranked['composite_score'] += df_ranked[metric] * weight

    df_ranked = df_ranked.sort_values('composite_score', ascending=False)
    df_ranked['rank'] = range(1, len(df_ranked) + 1)

    return df_ranked


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, output_path: str):
    """Create correlation heatmap visualization."""
    fig, ax = plt.subplots(figsize=(14, 12))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 8},
        cbar_kws={'shrink': 0.8}
    )

    ax.set_title('Correlation Matrix: Course Metrics', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_prediction_drivers(df_drivers: pd.DataFrame, output_path: str):
    """Plot the metrics that drive prediction potential."""
    if len(df_drivers) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Top 15 drivers
    df_top = df_drivers.head(15)

    colors = ['green' if x > 0 else 'red' for x in df_top['pearson_corr']]

    bars = ax.barh(
        df_top['metric'],
        df_top['pearson_corr'],
        color=colors,
        alpha=0.7,
        edgecolor='black'
    )

    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('Correlation with Prediction Potential Score')
    ax.set_title('Top Metrics Driving Prediction Potential', fontsize=14, fontweight='bold')

    # Add significance markers
    for i, (idx, row) in enumerate(df_top.iterrows()):
        if row['significant']:
            ax.annotate('*', xy=(row['pearson_corr'], i), fontsize=14, ha='left')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_score_distributions(df: pd.DataFrame, output_path: str):
    """Plot distributions of key scores."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    score_cols = [
        ('prediction_potential_score', 'Prediction Potential Score'),
        ('grade_availability_score', 'Grade Availability Score'),
        ('grade_variance_score', 'Grade Variance Score'),
        ('class_balance_score', 'Class Balance Score'),
        ('design_richness_score', 'Design Richness Score'),
        ('activity_score', 'Activity Score'),
    ]

    for ax, (col, title) in zip(axes, score_cols):
        if col in df.columns:
            data = df[col][df[col] > 0]  # Only non-zero values
            if len(data) > 0:
                ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
                ax.axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.1f}')
                ax.axvline(data.median(), color='orange', linestyle='--', label=f'Median: {data.median():.1f}')
                ax.legend(fontsize=8)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Score')
        ax.set_ylabel('Count')

    plt.suptitle('Distribution of Prediction Scores', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_grade_vs_activity(df: pd.DataFrame, output_path: str):
    """Scatter plot of grade metrics vs activity metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Filter to courses with both grades and activity
    df_valid = df[(df['current_score_coverage'] > 0) & (df['avg_page_views'] > 0)].copy()

    if len(df_valid) < 5:
        print("Not enough data for grade vs activity plot")
        return

    # Plot 1: Grade coverage vs Activity
    ax1 = axes[0]
    scatter = ax1.scatter(
        df_valid['current_score_coverage'],
        df_valid['avg_page_views'],
        c=df_valid['prediction_potential_score'],
        cmap='viridis',
        alpha=0.6,
        s=50
    )
    ax1.set_xlabel('Grade Coverage (%)')
    ax1.set_ylabel('Avg Page Views per Student')
    ax1.set_title('Grade Coverage vs Student Activity')
    plt.colorbar(scatter, ax=ax1, label='Prediction Score')

    # Plot 2: Grade variance vs Failure rate
    ax2 = axes[1]
    df_grades = df_valid[df_valid['grade_std'] > 0]
    if len(df_grades) > 0:
        scatter2 = ax2.scatter(
            df_grades['grade_std'],
            df_grades['failure_rate'] * 100,
            c=df_grades['prediction_potential_score'],
            cmap='viridis',
            alpha=0.6,
            s=50
        )
        ax2.set_xlabel('Grade Standard Deviation')
        ax2.set_ylabel('Failure Rate (%)')
        ax2.set_title('Grade Variance vs Failure Rate')
        plt.colorbar(scatter2, ax=ax2, label='Prediction Score')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_top_courses(df: pd.DataFrame, output_path: str, top_n: int = 25):
    """Bar chart of top courses by prediction potential."""
    fig, ax = plt.subplots(figsize=(14, 10))

    df_top = df.nlargest(top_n, 'prediction_potential_score')

    # Truncate course names
    names = [f"{row['course_name'][:30]}... ({row['course_id']})"
             for _, row in df_top.iterrows()]

    colors = plt.cm.viridis(df_top['prediction_potential_score'] / 100)

    bars = ax.barh(
        range(len(df_top)),
        df_top['prediction_potential_score'],
        color=colors,
        edgecolor='black',
        alpha=0.8
    )

    ax.set_yticks(range(len(df_top)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Prediction Potential Score')
    ax.set_title(f'Top {top_n} Courses for Predictive Modeling', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    # Add value labels
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.annotate(
            f"{row['prediction_potential_score']:.1f}",
            xy=(row['prediction_potential_score'] + 1, i),
            va='center',
            fontsize=8
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# =============================================================================
# REPORTING
# =============================================================================

def print_summary_report(
    df: pd.DataFrame,
    df_viable: pd.DataFrame,
    df_drivers: pd.DataFrame,
    df_ranked: pd.DataFrame
):
    """Print comprehensive summary report to console."""

    print("\n" + "=" * 80)
    print("COURSE ANALYSIS REPORT")
    print("=" * 80)

    # Overall statistics
    print(f"""
DATASET OVERVIEW
────────────────
  Total Courses:              {len(df)}
  Viable for Modeling:        {len(df_viable)} ({len(df_viable)/len(df)*100:.1f}%)
  With Grades (any):          {(df['students_with_current_score'] > 0).sum()}
  With High Potential (≥50):  {(df['prediction_potential_score'] >= 50).sum()}
  With Perfect Score (100):   {(df['prediction_potential_score'] == 100).sum()}

THRESHOLD CRITERIA
──────────────────
  Min Students:               {THRESHOLDS['min_students']}
  Min Grade Coverage:         {THRESHOLDS['min_grade_coverage']}%
  Min Grade Variance (std):   {THRESHOLDS['min_grade_variance']}
  Ideal Failure Rate:         {THRESHOLDS['ideal_fail_rate_min']*100:.0f}%-{THRESHOLDS['ideal_fail_rate_max']*100:.0f}%
""")

    # Grade statistics
    df_with_grades = df[df['students_with_current_score'] >= THRESHOLDS['min_students']]
    if len(df_with_grades) > 0:
        print(f"""
GRADE STATISTICS (courses with ≥{THRESHOLDS['min_students']} graded students)
────────────────────────────────────────────────────────
  Courses:                    {len(df_with_grades)}
  Avg Grade Coverage:         {df_with_grades['current_score_coverage'].mean():.1f}%
  Avg Grade Mean:             {df_with_grades['grade_mean'].mean():.1f}
  Avg Grade Std Dev:          {df_with_grades['grade_std'].mean():.1f}
  Avg Failure Rate:           {df_with_grades['failure_rate'].mean()*100:.1f}%
""")

    # Top prediction drivers
    if len(df_drivers) > 0:
        print("TOP METRICS DRIVING PREDICTION POTENTIAL")
        print("────────────────────────────────────────")
        for i, row in df_drivers.head(10).iterrows():
            sig = "*" if row['significant'] else ""
            print(f"  {row['metric']:<35} r={row['pearson_corr']:>6.3f} {sig}")
        print("  (* = statistically significant p<0.05)")

    # Top courses
    print(f"""
TOP 15 COURSES FOR PREDICTIVE MODELING
──────────────────────────────────────""")
    print(f"{'Rank':<5} {'ID':<8} {'Course Name':<40} {'Students':<10} {'Score':<8}")
    print("-" * 80)

    for i, row in df_ranked.head(15).iterrows():
        name = row['course_name'][:38] + ".." if len(row['course_name']) > 40 else row['course_name']
        print(f"{row['rank']:<5} {row['course_id']:<8} {name:<40} "
              f"{row['students_with_current_score']:<10} {row['prediction_potential_score']:<8.1f}")

    # Recommendations
    print(f"""

RECOMMENDATIONS
───────────────
1. BEST CANDIDATES: Focus on courses with score ≥ 80 ({(df['prediction_potential_score'] >= 80).sum()} courses)
   - High grade coverage AND variance
   - Balanced pass/fail distribution

2. QUICK WINS: Courses with score 60-80 ({((df['prediction_potential_score'] >= 60) & (df['prediction_potential_score'] < 80)).sum()} courses)
   - Good data quality, may need more features

3. DATA GAPS: {(df['current_score_coverage'] == 0).sum()} courses have NO Canvas grades
   - May use external gradebook (Libro de Calificaciones)
   - Consider excluding or finding alternative data

4. SAMPLE SIZE: {(df['students_with_current_score'] < THRESHOLDS['min_students']).sum()} courses have too few graded students
   - Need ≥{THRESHOLDS['min_students']} for statistical validity
""")

    print("=" * 80)


def save_rankings(df_ranked: pd.DataFrame, output_path: str, top_n: int = 100):
    """Save top courses ranking to CSV."""
    columns = [
        'rank', 'course_id', 'course_name', 'account_id', 'term_name',
        'total_students', 'students_with_current_score', 'current_score_coverage',
        'grade_mean', 'grade_std', 'failure_rate',
        'assignment_count', 'quiz_count', 'module_count',
        'avg_page_views', 'students_with_activity',
        'prediction_potential_score', 'composite_score'
    ]

    # Only include columns that exist
    columns = [c for c in columns if c in df_ranked.columns]

    df_export = df_ranked.head(top_n)[columns]
    df_export.to_csv(output_path, index=False)
    print(f"Saved top {top_n} rankings: {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze course discovery results for predictive modeling"
    )
    parser.add_argument(
        '--input', type=str, default=DEFAULT_INPUT,
        help=f'Input CSV file (default: {DEFAULT_INPUT})'
    )
    parser.add_argument(
        '--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
        help=f'Output directory for results (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--top', type=int, default=50,
        help='Number of top courses to include in ranking (default: 50)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='Skip generating plots'
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    df = load_data(args.input)

    if len(df) == 0:
        print("ERROR: No data loaded")
        return

    # Filter viable courses
    df_viable = filter_viable_courses(df)

    # Compute correlations
    print("\nComputing correlations...")
    corr_matrix = compute_correlations(df)

    # Analyze prediction drivers
    print("Analyzing prediction drivers...")
    df_drivers = analyze_prediction_drivers(df)

    # Compute rankings
    print("Computing course rankings...")
    df_ranked = compute_composite_ranking(df)

    # Category analysis
    print("Analyzing by category...")
    category_results = analyze_by_category(df)

    # Generate visualizations
    if not args.no_plots:
        print("\nGenerating visualizations...")

        plot_correlation_heatmap(
            corr_matrix,
            os.path.join(args.output_dir, "correlation_heatmap.png")
        )

        plot_prediction_drivers(
            df_drivers,
            os.path.join(args.output_dir, "prediction_drivers.png")
        )

        plot_score_distributions(
            df,
            os.path.join(args.output_dir, "score_distributions.png")
        )

        plot_grade_vs_activity(
            df,
            os.path.join(args.output_dir, "grade_vs_activity.png")
        )

        plot_top_courses(
            df_ranked,
            os.path.join(args.output_dir, "top_courses.png"),
            top_n=25
        )

    # Save rankings
    save_rankings(
        df_ranked,
        os.path.join(args.output_dir, "course_rankings.csv"),
        top_n=args.top
    )

    # Save correlation matrix
    corr_matrix.to_csv(os.path.join(args.output_dir, "correlation_matrix.csv"))
    print(f"Saved: {os.path.join(args.output_dir, 'correlation_matrix.csv')}")

    # Save drivers analysis
    if len(df_drivers) > 0:
        df_drivers.to_csv(os.path.join(args.output_dir, "prediction_drivers.csv"), index=False)
        print(f"Saved: {os.path.join(args.output_dir, 'prediction_drivers.csv')}")

    # Print summary report
    print_summary_report(df, df_viable, df_drivers, df_ranked)

    print(f"\nAll outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
