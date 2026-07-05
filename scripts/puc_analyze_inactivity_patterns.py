#!/usr/bin/env python3
"""
Phase 2: Analyze inactivity/gap patterns across grade classes.
Focuses on session gaps, irregularity, and disengagement patterns.
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
PV_FILE = DATA_DIR / "puc_merged_data.parquet"
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


def identify_sessions(timestamps, gap_threshold_minutes=30):
    """Identify session boundaries using gap threshold."""
    timestamps = timestamps.sort_values()
    gaps = timestamps.diff().dt.total_seconds() / 60
    session_starts = gaps >= gap_threshold_minutes
    session_starts.iloc[0] = True
    return session_starts.cumsum()


def calculate_inactivity_features(df_user):
    """Calculate inactivity metrics for a single student."""
    if len(df_user) < 2:
        return {
            'max_inactivity_hours': np.nan,
            'avg_inactivity_hours': np.nan,
            'std_inactivity_hours': np.nan,
            'inactivity_cv': np.nan,
            'consecutive_inactive_weeks': np.nan
        }

    # Identify sessions
    sessions = identify_sessions(df_user['created_at'])
    session_ends = df_user.groupby(sessions)['created_at'].max()

    # Calculate gaps between sessions
    gaps_hours = session_ends.diff().dt.total_seconds() / 3600
    gaps_hours = gaps_hours.dropna()

    if len(gaps_hours) == 0:
        return {
            'max_inactivity_hours': np.nan,
            'avg_inactivity_hours': np.nan,
            'std_inactivity_hours': np.nan,
            'inactivity_cv': np.nan,
            'consecutive_inactive_weeks': np.nan
        }

    # Consecutive inactive weeks
    df_user_sorted = df_user.sort_values('created_at')
    weeks = df_user_sorted['week_number_from_start'].unique()
    if len(weeks) > 1:
        week_gaps = np.diff(sorted(weeks))
        consecutive_inactive = (week_gaps - 1).max() if len(week_gaps) > 0 else 0
    else:
        consecutive_inactive = 0

    return {
        'max_inactivity_hours': gaps_hours.max(),
        'avg_inactivity_hours': gaps_hours.mean(),
        'std_inactivity_hours': gaps_hours.std() if len(gaps_hours) > 1 else 0,
        'inactivity_cv': gaps_hours.std() / gaps_hours.mean() if gaps_hours.mean() > 0 and len(gaps_hours) > 1 else 0,
        'consecutive_inactive_weeks': consecutive_inactive
    }


def load_data():
    """Load page views and features with grade classes."""
    print("Loading data...")

    # Load features with grades
    features = pd.read_parquet(FEATURES_FILE)
    features['grade_class'] = features['grade'].apply(assign_grade_class)
    features['class_label'] = features['grade_class'].map(lambda x: GRADE_CLASSES[x]['name'])

    print(f"Loaded {len(features)} student enrollments")

    # Load page views
    print("Loading page views...")
    pv = pd.read_parquet(PV_FILE)
    pv['created_at'] = pd.to_datetime(pv['created_at'])

    print(f"Loaded {len(pv):,} page views")

    return features, pv


def calculate_all_inactivity_features(features, pv):
    """Calculate inactivity features for all students."""
    print("\nCalculating inactivity features...")

    inactivity_data = []

    for i, (_, row) in enumerate(features.iterrows()):
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(features)} students...")

        student_id = row['student_id']
        course_id = row['course_id']

        # Get page views for this enrollment
        df_user = pv[(pv['student_id'] == student_id) & (pv['course_id'] == course_id)]

        if len(df_user) == 0:
            continue

        # Calculate inactivity features
        inact_features = calculate_inactivity_features(df_user)

        # Add metadata
        inact_features['student_id'] = student_id
        inact_features['course_id'] = course_id
        inact_features['grade'] = row['grade']
        inact_features['grade_class'] = row['grade_class']
        inact_features['class_label'] = row['class_label']

        inactivity_data.append(inact_features)

    df_inact = pd.DataFrame(inactivity_data)
    print(f"\nCalculated inactivity features for {len(df_inact)} students")

    return df_inact


def analyze_by_class(df):
    """Analyze inactivity features by grade class."""
    print("\nAnalyzing by grade class...")

    inactivity_features = [
        'max_inactivity_hours',
        'avg_inactivity_hours',
        'std_inactivity_hours',
        'inactivity_cv',
        'consecutive_inactive_weeks'
    ]

    results = {}

    for feature in inactivity_features:
        # Get data for each class
        groups = {}
        for class_id in range(4):
            class_name = GRADE_CLASSES[class_id]['name']
            groups[class_name] = df[df['grade_class'] == class_id][feature].dropna()

        # Skip if insufficient data
        if any(len(g) < 5 for g in groups.values()):
            continue

        # ANOVA
        f_stat, p_anova = stats.f_oneway(*groups.values())

        # Effect size (Cohen's d for EXCELLENT vs FAIL)
        excellent = groups['EXCELLENT']
        fail = groups['FAIL']
        n1, n2 = len(excellent), len(fail)
        var1, var2 = excellent.var(), fail.var()
        pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        cohens_d = (excellent.mean() - fail.mean()) / pooled_std if pooled_std > 0 else 0

        # Mann-Whitney U test
        _, p_mw = stats.mannwhitneyu(excellent, fail, alternative='two-sided')

        # Summary stats
        class_stats = {}
        for class_name, data in groups.items():
            class_stats[class_name] = {
                'mean': float(data.mean()),
                'median': float(data.median()),
                'std': float(data.std()),
                'count': int(len(data))
            }

        results[feature] = {
            'anova_f': float(f_stat),
            'anova_p': float(p_anova),
            'cohens_d_excellent_vs_fail': float(cohens_d),
            'mannwhitney_p': float(p_mw),
            'significant': bool(p_anova < 0.001),
            'class_stats': class_stats
        }

        print(f"\n{feature}:")
        print(f"  ANOVA F={f_stat:.2f}, p={p_anova:.2e}")
        print(f"  Cohen's d (EXCELLENT vs FAIL) = {cohens_d:.2f}")
        print(f"  EXCELLENT: {excellent.mean():.2f} ± {excellent.std():.2f}")
        print(f"  FAIL:      {fail.mean():.2f} ± {fail.std():.2f}")

    return results


def save_results(results):
    """Save analysis results."""
    output_path = OUTPUT_DIR / "inactivity_by_class.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


def plot_violin_plots(df):
    """Create violin plots for inactivity features."""
    inactivity_features = [
        'max_inactivity_hours',
        'avg_inactivity_hours',
        'std_inactivity_hours',
        'inactivity_cv',
        'consecutive_inactive_weeks'
    ]

    feature_labels = {
        'max_inactivity_hours': 'Max Inactivity Gap (hours)',
        'avg_inactivity_hours': 'Avg Inactivity Gap (hours)',
        'std_inactivity_hours': 'Std Dev Inactivity (hours)',
        'inactivity_cv': 'Inactivity CV (regularity)',
        'consecutive_inactive_weeks': 'Max Consecutive Inactive Weeks'
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, feature in enumerate(inactivity_features):
        ax = axes[i]

        # Prepare data
        data = []
        labels = []
        for class_id in range(4):
            class_data = df[df['grade_class'] == class_id][feature].dropna()
            data.append(class_data)
            labels.append(GRADE_CLASSES[class_id]['name'])

        # Create violin plot
        parts = ax.violinplot(data, positions=range(4), widths=0.7,
                              showmeans=True, showmedians=True)

        # Color violins
        for pc, class_id in zip(parts['bodies'], range(4)):
            pc.set_facecolor(GRADE_CLASSES[class_id]['color'])
            pc.set_alpha(0.7)

        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(feature_labels[feature], fontweight='bold', fontsize=11)
        ax.set_ylabel('Value')
        ax.grid(axis='y', alpha=0.3)

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle('Inactivity Patterns Across Grade Classes', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = VIZ_DIR / "inactivity_violin_plots.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")


def plot_heatmap(results):
    """Create heatmap showing class means for each inactivity metric."""
    features = list(results.keys())
    classes = ['EXCELLENT', 'GOOD', 'MARGINAL', 'FAIL']

    # Build matrix
    matrix = []
    for feature in features:
        row = [results[feature]['class_stats'][cls]['mean'] for cls in classes]
        matrix.append(row)

    matrix = np.array(matrix)

    # Normalize by row for better visualization
    matrix_norm = (matrix - matrix.min(axis=1, keepdims=True)) / (matrix.max(axis=1, keepdims=True) - matrix.min(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix_norm, cmap='RdYlGn_r', aspect='auto')

    # Set ticks
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([f.replace('_', ' ').title() for f in features])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Value\n(0=min, 1=max)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(features)):
        for j in range(len(classes)):
            text = ax.text(j, i, f'{matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_title('Inactivity Metrics Heatmap by Grade Class', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    output_path = VIZ_DIR / "inactivity_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("="*60)
    print("Phase 2: Inactivity Pattern Analysis")
    print("="*60)

    # Load data
    features, pv = load_data()

    # Calculate inactivity features
    df_inact = calculate_all_inactivity_features(features, pv)

    # Analyze by class
    results = analyze_by_class(df_inact)

    # Save results
    save_results(results)

    # Create visualizations
    print("\nGenerating visualizations...")
    plot_violin_plots(df_inact)
    plot_heatmap(results)

    print("\n✓ Phase 2 complete!")


if __name__ == "__main__":
    main()
