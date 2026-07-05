#!/usr/bin/env python3
"""
Generate pass_fail_comparison.png showing key session metrics.

Displays boxplots comparing approved vs failed students for:
1. Session count (total number of sessions)
2. Total page views (clicks in LMS)
3. Session duration mean (average minutes per session)
4. Views per session (clicks per session)
5. Session density (clicks per minute - intensity)
6. Session spread days (unique active days - consistency)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "report" / "analysis" / "pass_fail_comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

COLORS = {
    'aprobados': '#6bcb77',
    'reprobados': '#ff6b6b'
}


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std


def main():
    print("="*60)
    print("Generating pass_fail_comparison.png")
    print("="*60)

    # Load session features
    print("\nLoading session features...")
    df_session = pd.read_parquet(DATA_DIR / "enriched_features" / "session_features.parquet")
    print(f"  Loaded {len(df_session)} student-course pairs")

    # Load grades
    print("\nLoading grades...")
    df_enroll = pd.read_csv(DATA_DIR / "page_views" / "student_enrollments.csv")
    df_enroll = df_enroll[df_enroll['final_score'].notna()]
    print(f"  Loaded {len(df_enroll)} enrollments with grades")

    # Merge
    df = df_session.merge(
        df_enroll[['user_id', 'course_id', 'final_score']],
        on=['user_id', 'course_id'],
        how='inner'
    )
    df['passed'] = df['final_score'] >= 57
    print(f"\n  Merged: {len(df)} students with both session data and grades")
    print(f"  Passed: {df['passed'].sum()} | Failed: {(~df['passed']).sum()}")

    # Features to plot
    features = [
        ('session_count', 'Total Sesiones'),
        ('total_views', 'Total Page Views'),
        ('session_duration_mean', 'Duración Media (min)'),
        ('views_per_session', 'Clicks/Sesión'),
        ('session_density', 'Densidad (clicks/min)'),
        ('session_spread_days', 'Días Activos'),
    ]

    # Filter out session_density outliers (cap at 99th percentile for visualization)
    df['session_density_capped'] = df['session_density'].clip(upper=df['session_density'].quantile(0.99))

    # Create figure
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(3.2 * n_features, 5.5))

    print("\nCalculating statistics:")
    print("-" * 60)

    results = []
    for i, (feat, label) in enumerate(features):
        ax = axes[i]

        # Use capped version for density visualization
        plot_feat = 'session_density_capped' if feat == 'session_density' else feat

        passed = df[df['passed']][plot_feat].dropna()
        failed = df[~df['passed']][plot_feat].dropna()

        # Calculate stats on original data
        orig_feat = feat
        d = calculate_cohens_d(df[df['passed']][orig_feat].dropna(),
                               df[~df['passed']][orig_feat].dropna())
        _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        print(f"  {label:25s}: d={d:+.2f} {sig:4s} (p={p:.4f})")

        # Create boxplot
        bp = ax.boxplot([passed, failed], patch_artist=True, widths=0.6)
        bp["boxes"][0].set_facecolor(COLORS['aprobados'])
        bp["boxes"][0].set_alpha(0.85)
        bp["boxes"][1].set_facecolor(COLORS['reprobados'])
        bp["boxes"][1].set_alpha(0.85)

        for median in bp['medians']:
            median.set_color('#d35400')
            median.set_linewidth(2)

        ax.set_xticklabels(["Aprobados", "Reprobados"])
        ax.set_title(label, fontweight='bold', fontsize=11)

        # Add effect size annotation
        effect_label = f'd={d:.2f}{sig}'
        ax.text(0.5, 0.95, effect_label, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, fontweight='bold',
                color='#2c3e50', bbox=dict(boxstyle='round,pad=0.2',
                facecolor='white', edgecolor='none', alpha=0.8))

        results.append({
            'feature': feat,
            'label': label,
            'cohens_d': d,
            'p_value': p,
            'passed_mean': df[df['passed']][orig_feat].mean(),
            'failed_mean': df[~df['passed']][orig_feat].mean()
        })

    # Add main title
    plt.suptitle('Comparación de Features: Aprobados vs Reprobados\nMétricas de Sesión del LMS',
                 fontsize=13, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'pass_fail_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("-" * 60)
    print(f"\nSaved: {output_path}")

    # Save stats to JSON
    import json
    stats_path = OUTPUT_DIR / 'pass_fail_comparison_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Stats: {stats_path}")

    # Summary
    print("\n" + "="*60)
    print("TOP DISCRIMINATIVE SESSION FEATURES:")
    print("="*60)
    for r in sorted(results, key=lambda x: abs(x['cohens_d']), reverse=True):
        print(f"  {r['label']:25s}: d={r['cohens_d']:+.2f}  (Pass: {r['passed_mean']:.1f}, Fail: {r['failed_mean']:.1f})")


if __name__ == '__main__':
    main()
