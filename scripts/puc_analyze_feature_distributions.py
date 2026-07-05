#!/usr/bin/env python3
"""
Phase 1: Analyze feature distributions across 4 grade classes (EXCELLENT, GOOD, MARGINAL, FAIL).
Computes ANOVA, effect sizes, and variance explained to identify discriminative features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import json

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "puc"
FEATURES_FILE = DATA_DIR / "enriched_features" / "normalized_features.parquet"
OUTPUT_DIR = DATA_DIR / "analysis"
VIZ_DIR = DATA_DIR / "report" / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.facecolor'] = 'white'

# Grade class definitions
GRADE_CLASSES = {
    0: {'name': 'EXCELLENT', 'range': (6.0, 7.0), 'color': '#2ecc71'},
    1: {'name': 'GOOD', 'range': (5.0, 5.9), 'color': '#3498db'},
    2: {'name': 'MARGINAL', 'range': (4.0, 4.9), 'color': '#f39c12'},
    3: {'name': 'FAIL', 'range': (0.0, 3.9), 'color': '#e74c3c'}
}


def assign_grade_class(grade):
    """Assign grade to one of 4 classes."""
    if grade >= 6.0:
        return 0  # EXCELLENT
    elif grade >= 5.0:
        return 1  # GOOD
    elif grade >= 4.0:
        return 2  # MARGINAL
    else:
        return 3  # FAIL


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std


def calculate_eta_squared(groups):
    """Calculate eta-squared (variance explained) for ANOVA."""
    all_data = np.concatenate([g.values for g in groups])
    group_means = [g.mean() for g in groups]
    grand_mean = np.mean(all_data)

    # Between-group variance
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)

    # Total variance
    ss_total = np.sum((all_data - grand_mean)**2)

    if ss_total == 0:
        return 0

    return ss_between / ss_total


def analyze_feature(df, feature_name):
    """Analyze single feature across all grade classes."""
    # Skip if feature doesn't exist
    if feature_name not in df.columns:
        return None

    # Get data for each class
    groups = {}
    for class_id in range(4):
        class_name = GRADE_CLASSES[class_id]['name']
        groups[class_name] = df[df['grade_class'] == class_id][feature_name].dropna()

    # Check if all groups have sufficient data
    if any(len(g) < 5 for g in groups.values()):
        return None

    # ANOVA
    f_stat, p_anova = stats.f_oneway(*groups.values())

    # Eta-squared (variance explained)
    eta_squared = calculate_eta_squared(list(groups.values()))

    # Pairwise comparisons (Cohen's d)
    pairwise = {}
    class_names = ['EXCELLENT', 'GOOD', 'MARGINAL', 'FAIL']
    for i in range(len(class_names)):
        for j in range(i+1, len(class_names)):
            name1, name2 = class_names[i], class_names[j]
            d = calculate_cohens_d(groups[name1], groups[name2])
            _, p = stats.mannwhitneyu(groups[name1], groups[name2], alternative='two-sided')
            pairwise[f'{name1}_vs_{name2}'] = {
                'cohens_d': float(d),
                'p_value': float(p),
                'significant': bool(p < 0.001)
            }

    # Summary statistics per class
    class_stats = {}
    for class_name, data in groups.items():
        class_stats[class_name] = {
            'mean': float(data.mean()),
            'std': float(data.std()),
            'median': float(data.median()),
            'count': int(len(data))
        }

    return {
        'feature': feature_name,
        'anova_f': float(f_stat),
        'anova_p': float(p_anova),
        'eta_squared': float(eta_squared),
        'significant': bool(p_anova < 0.001),
        'class_stats': class_stats,
        'pairwise': pairwise
    }


def load_and_prepare_data():
    """Load features and assign grade classes."""
    print("Loading data...")
    df = pd.read_parquet(FEATURES_FILE)

    # Assign grade classes
    df['grade_class'] = df['grade'].apply(assign_grade_class)
    df['class_label'] = df['grade_class'].map(lambda x: GRADE_CLASSES[x]['name'])

    # Print class distribution
    print("\nClass Distribution:")
    for class_id in range(4):
        count = (df['grade_class'] == class_id).sum()
        pct = count / len(df) * 100
        class_info = GRADE_CLASSES[class_id]
        print(f"  Class {class_id} ({class_info['name']}): {count} students ({pct:.1f}%) - Grade {class_info['range'][0]}-{class_info['range'][1]}")

    return df


def analyze_all_features(df):
    """Analyze all features across grade classes."""
    # Exclude metadata columns
    exclude_cols = {'student_id', 'course_id', 'grade', 'failed', 'grade_class', 'class_label'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    print(f"\nAnalyzing {len(feature_cols)} features...")

    results = []
    for i, feature in enumerate(feature_cols):
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(feature_cols)} features...")

        result = analyze_feature(df, feature)
        if result:
            results.append(result)

    # Sort by variance explained (eta-squared)
    results.sort(key=lambda x: x['eta_squared'], reverse=True)

    print(f"\nCompleted analysis for {len(results)} features")
    print(f"  Significant features (p<0.001): {sum(r['significant'] for r in results)}")
    print(f"  Features with η² > 0.10: {sum(r['eta_squared'] > 0.10 for r in results)}")

    return results


def save_results(results):
    """Save analysis results to JSON and CSV."""
    # JSON (full results)
    json_path = OUTPUT_DIR / "feature_distributions_by_class.json"
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {json_path}")

    # CSV (summary)
    summary_data = []
    for r in results:
        row = {
            'feature': r['feature'],
            'anova_f': r['anova_f'],
            'anova_p': r['anova_p'],
            'eta_squared': r['eta_squared'],
            'significant': r['significant']
        }

        # Add mean per class
        for class_name in ['EXCELLENT', 'GOOD', 'MARGINAL', 'FAIL']:
            row[f'{class_name}_mean'] = r['class_stats'][class_name]['mean']
            row[f'{class_name}_std'] = r['class_stats'][class_name]['std']

        # Add key pairwise comparisons
        row['EXCELLENT_vs_FAIL_d'] = r['pairwise']['EXCELLENT_vs_FAIL']['cohens_d']
        row['GOOD_vs_MARGINAL_d'] = r['pairwise']['GOOD_vs_MARGINAL']['cohens_d']

        summary_data.append(row)

    csv_path = OUTPUT_DIR / "feature_distributions_by_class.csv"
    pd.DataFrame(summary_data).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")


def plot_variance_explained(results):
    """Create bar chart of variance explained (eta-squared) for top features."""
    # Top 20 features
    top_features = results[:20]

    fig, ax = plt.subplots(figsize=(12, 8))

    features = [r['feature'] for r in top_features]
    eta_squared = [r['eta_squared'] for r in top_features]
    colors = ['#e74c3c' if r['significant'] else '#95a5a6' for r in top_features]

    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, eta_squared, color=colors, alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Variance Explained (η²)', fontweight='bold')
    ax.set_title('Top 20 Features by Variance Explained Across Grade Classes',
                 fontsize=14, fontweight='bold', pad=20)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # Add values on bars
    for i, (bar, val) in enumerate(zip(bars, eta_squared)):
        ax.text(val + 0.005, i, f'{val:.3f}', va='center', fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#e74c3c', alpha=0.7, label='Significant (p<0.001)'),
        Patch(facecolor='#95a5a6', alpha=0.7, label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    output_path = VIZ_DIR / "variance_explained_bars.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")


def plot_feature_boxplots(df, results):
    """Create boxplots for top 20 discriminative features."""
    top_features = [r['feature'] for r in results[:20]]

    # Create 4x5 grid
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]

        # Prepare data for boxplot
        data = []
        labels = []
        colors_list = []
        for class_id in range(4):
            class_data = df[df['grade_class'] == class_id][feature].dropna()
            data.append(class_data)
            labels.append(GRADE_CLASSES[class_id]['name'][:4])  # Abbreviate
            colors_list.append(GRADE_CLASSES[class_id]['color'])

        # Create boxplot
        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xticklabels(labels, rotation=0, fontsize=9)
        ax.set_title(feature, fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add eta-squared annotation
        result = next(r for r in results if r['feature'] == feature)
        eta = result['eta_squared']
        ax.text(0.98, 0.98, f"η²={eta:.3f}", transform=ax.transAxes,
                ha='right', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    plt.suptitle('Top 20 Features by Discriminative Power Across Grade Classes',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = VIZ_DIR / "feature_boxplots_top20.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*60)
    print("Phase 1: Feature Distribution Analysis by Grade Class")
    print("="*60)

    # Load data
    df = load_and_prepare_data()

    # Analyze all features
    results = analyze_all_features(df)

    # Save results
    save_results(results)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_variance_explained(results)
    plot_feature_boxplots(df, results)

    # Print summary of top features
    print("\n" + "="*60)
    print("Top 10 Discriminative Features (by η²):")
    print("="*60)
    for i, r in enumerate(results[:10], 1):
        print(f"{i:2d}. {r['feature']:35s} η²={r['eta_squared']:.3f}  F={r['anova_f']:.1f}  p={r['anova_p']:.2e}")
        print(f"    EXCELLENT: {r['class_stats']['EXCELLENT']['mean']:.2f} ± {r['class_stats']['EXCELLENT']['std']:.2f}")
        print(f"    GOOD:      {r['class_stats']['GOOD']['mean']:.2f} ± {r['class_stats']['GOOD']['std']:.2f}")
        print(f"    MARGINAL:  {r['class_stats']['MARGINAL']['mean']:.2f} ± {r['class_stats']['MARGINAL']['std']:.2f}")
        print(f"    FAIL:      {r['class_stats']['FAIL']['mean']:.2f} ± {r['class_stats']['FAIL']['std']:.2f}")
        print()

    print("\n✓ Phase 1 complete!")


if __name__ == "__main__":
    main()
