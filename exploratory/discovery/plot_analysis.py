#!/usr/bin/env python3
"""
Visualization script for course data analysis.

Generates various plots for exploratory data analysis.

Usage:
    python plot_analysis.py --course-dir exploratory/data/courses/course_86676
    python plot_analysis.py --course-dir exploratory/data/courses/course_86676 --plot scatter
    python plot_analysis.py --course-dir exploratory/data/courses/course_86676 --plot all
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

PASS_THRESHOLD = 57.0


def load_data(course_dir: str) -> pd.DataFrame:
    """Load consolidated student data."""
    csv_path = os.path.join(course_dir, 'student_consolidated.csv')
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"No consolidated data found at {csv_path}")


def plot_scatter_participation(df: pd.DataFrame, output_dir: str):
    """Scatter plot: final_score vs participations."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by pass/fail
    colors = df['final_score'].apply(lambda x: '#2ecc71' if x >= PASS_THRESHOLD else '#e74c3c')

    scatter = ax.scatter(
        df['participations'],
        df['final_score'],
        c=colors,
        s=100,
        alpha=0.7,
        edgecolors='white',
        linewidth=1
    )

    # Add pass threshold line
    ax.axhline(y=PASS_THRESHOLD, color='gray', linestyle='--', linewidth=2, label=f'Pass threshold ({PASS_THRESHOLD}%)')

    # Add correlation line
    z = np.polyfit(df['participations'], df['final_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['participations'].min(), df['participations'].max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, linewidth=2, label=f'Trend (r={df["participations"].corr(df["final_score"]):.2f})')

    # Labels
    ax.set_xlabel('Participations', fontsize=12)
    ax.set_ylabel('Final Score (%)', fontsize=12)
    ax.set_title('Final Score vs Participations\n(Each dot = 1 student)', fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label=f'PASS (n={len(df[df["final_score"] >= PASS_THRESHOLD])})'),
        Patch(facecolor='#e74c3c', label=f'FAIL (n={len(df[df["final_score"] < PASS_THRESHOLD])})'),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='lower right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_participation.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_missing(df: pd.DataFrame, output_dir: str):
    """Scatter plot: final_score vs tardiness_missing."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = df['final_score'].apply(lambda x: '#2ecc71' if x >= PASS_THRESHOLD else '#e74c3c')

    ax.scatter(
        df['tardiness_missing'],
        df['final_score'],
        c=colors,
        s=100,
        alpha=0.7,
        edgecolors='white',
        linewidth=1
    )

    ax.axhline(y=PASS_THRESHOLD, color='gray', linestyle='--', linewidth=2)

    # Correlation line
    z = np.polyfit(df['tardiness_missing'], df['final_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['tardiness_missing'].min(), df['tardiness_missing'].max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, linewidth=2,
            label=f'Trend (r={df["tardiness_missing"].corr(df["final_score"]):.2f})')

    ax.set_xlabel('Missing Assignments', fontsize=12)
    ax.set_ylabel('Final Score (%)', fontsize=12)
    ax.set_title('Final Score vs Missing Assignments\n(Each dot = 1 student)', fontsize=14, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='PASS'),
        Patch(facecolor='#e74c3c', label='FAIL'),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='upper right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_missing.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_scatter_pageviews(df: pd.DataFrame, output_dir: str):
    """Scatter plot: final_score vs page_views."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = df['final_score'].apply(lambda x: '#2ecc71' if x >= PASS_THRESHOLD else '#e74c3c')

    ax.scatter(
        df['page_views'],
        df['final_score'],
        c=colors,
        s=100,
        alpha=0.7,
        edgecolors='white',
        linewidth=1
    )

    ax.axhline(y=PASS_THRESHOLD, color='gray', linestyle='--', linewidth=2)

    # Correlation line
    z = np.polyfit(df['page_views'], df['final_score'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(df['page_views'].min(), df['page_views'].max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, linewidth=2,
            label=f'Trend (r={df["page_views"].corr(df["final_score"]):.2f})')

    ax.set_xlabel('Page Views', fontsize=12)
    ax.set_ylabel('Final Score (%)', fontsize=12)
    ax.set_title('Final Score vs Page Views\n(Each dot = 1 student)', fontsize=14, fontweight='bold')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='PASS'),
        Patch(facecolor='#e74c3c', label='FAIL'),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0], loc='lower right')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'scatter_pageviews.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Correlation heatmap for key features."""
    features = ['final_score', 'participations', 'page_views', 'tardiness_missing',
                'on_time', 'submitted_count', 'avg_score']

    # Filter to existing columns
    features = [f for f in features if f in df.columns]

    corr_matrix = df[features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                fmt='.2f', square=True, linewidths=0.5, ax=ax,
                vmin=-1, vmax=1)

    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_distribution_comparison(df: pd.DataFrame, output_dir: str):
    """Box plots comparing PASS vs FAIL distributions."""
    features = ['participations', 'page_views', 'tardiness_missing', 'on_time']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    df['Status'] = df['final_score'].apply(lambda x: 'PASS' if x >= PASS_THRESHOLD else 'FAIL')

    for i, feat in enumerate(features):
        sns.boxplot(data=df, x='Status', y=feat, ax=axes[i],
                    palette={'PASS': '#2ecc71', 'FAIL': '#e74c3c'})
        axes[i].set_title(f'{feat}', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')

    fig.suptitle('Feature Distribution: PASS vs FAIL', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'distribution_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_grade_distribution(df: pd.DataFrame, output_dir: str):
    """Histogram of final scores."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bins
    bins = np.arange(0, 105, 5)

    # Plot histogram
    n, bins_out, patches = ax.hist(df['final_score'], bins=bins, edgecolor='white', linewidth=1)

    # Color bars by pass/fail
    for i, patch in enumerate(patches):
        if bins_out[i] >= PASS_THRESHOLD:
            patch.set_facecolor('#2ecc71')
        else:
            patch.set_facecolor('#e74c3c')

    ax.axvline(x=PASS_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Pass threshold ({PASS_THRESHOLD}%)')

    ax.set_xlabel('Final Score (%)', fontsize=12)
    ax.set_ylabel('Number of Students', fontsize=12)
    ax.set_title('Grade Distribution', fontsize=14, fontweight='bold')
    ax.legend()

    # Add stats
    stats_text = f'n={len(df)}\nMean={df["final_score"].mean():.1f}%\nStd={df["final_score"].std():.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'grade_distribution.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_all(df: pd.DataFrame, output_dir: str):
    """Generate all plots."""
    plot_scatter_participation(df, output_dir)
    plot_scatter_missing(df, output_dir)
    plot_scatter_pageviews(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_distribution_comparison(df, output_dir)
    plot_grade_distribution(df, output_dir)


PLOT_FUNCTIONS = {
    'scatter': plot_scatter_participation,
    'scatter_participation': plot_scatter_participation,
    'scatter_missing': plot_scatter_missing,
    'scatter_pageviews': plot_scatter_pageviews,
    'heatmap': plot_correlation_heatmap,
    'distribution': plot_distribution_comparison,
    'grades': plot_grade_distribution,
    'all': plot_all,
}


def main():
    parser = argparse.ArgumentParser(description='Generate analysis plots')
    parser.add_argument('--course-dir', type=str, required=True,
                        help='Path to course data directory')
    parser.add_argument('--plot', type=str, default='all',
                        choices=list(PLOT_FUNCTIONS.keys()),
                        help='Type of plot to generate')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for plots (default: course-dir/plots)')

    args = parser.parse_args()

    # Load data
    df = load_data(args.course_dir)

    # Set output directory
    output_dir = args.output_dir or os.path.join(args.course_dir, 'plots')
    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating plots for {len(df)} students...")
    print(f"Output directory: {output_dir}\n")

    # Generate plot(s)
    PLOT_FUNCTIONS[args.plot](df, output_dir)

    print("\nDone!")


if __name__ == '__main__':
    main()
