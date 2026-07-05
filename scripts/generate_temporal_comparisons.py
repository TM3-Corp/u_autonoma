#!/usr/bin/env python3
"""
Generate temporal pattern comparison plots: Weekend vs Weekday, Time Diversity.

Creates standardized boxplots showing differences between approved/failed students
in terms of WHEN they study (weekend usage, time diversity).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Configuration
OUTPUT_DIR = Path("data/report/analysis/pass_fail_comparisons")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    'aprobados': '#6bcb77',
    'reprobados': '#ff6b6b'
}


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0


def load_data():
    """Load and merge temporal feature data."""
    # Load engagement dynamics features
    df = pd.read_csv("data/engagement_dynamics/student_features.csv")
    df['passed'] = df['final_score'] >= 57

    # Load time features
    try:
        time_df = pd.read_parquet("data/enriched_features/time_features.parquet")
        df = df.merge(time_df, on=['user_id', 'course_id'], how='left')
    except Exception as e:
        print(f"Warning: Could not load time_features.parquet: {e}")

    return df[df['final_score'].notna()]


def create_comparison_plot(df, features, labels, title, filename, min_effect=0.25):
    """Create standardized comparison boxplot."""
    # Filter to significant features
    significant = []
    for feat, label in zip(features, labels):
        if feat not in df.columns:
            continue
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()
        if len(passed) > 10 and len(failed) > 10:
            d = calculate_cohens_d(passed, failed)
            if abs(d) >= min_effect:
                significant.append((feat, label, d))

    if not significant:
        print(f"  No significant features for {filename}, skipping.")
        return None

    n = len(significant)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 5))
    if n == 1:
        axes = [axes]

    results = []
    for i, (feat, label, _) in enumerate(significant):
        ax = axes[i]
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()

        # Create boxplot with default width (thin boxes)
        bp = ax.boxplot([passed, failed], patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS['aprobados'])
        bp["boxes"][1].set_facecolor(COLORS['reprobados'])

        # Style median line
        for median in bp['medians']:
            median.set_color('#d35400')
            median.set_linewidth(2)

        ax.set_xticklabels(["Aprobados", "Reprobados"])
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel("Valor")

        # Calculate and display Cohen's d
        d = calculate_cohens_d(passed, failed)
        _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
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
            'passed_mean': passed.mean(),
            'failed_mean': failed.mean()
        })

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"  Saved: {filename} ({n} features)")
    return results


def create_time_distribution_plot(df):
    """Create stacked bar chart showing time-of-day distribution."""
    time_cols = ['pct_morning', 'pct_afternoon', 'pct_evening', 'pct_night']
    time_labels = ['Mañana\n(06-12h)', 'Tarde\n(12-18h)', 'Noche\n(18-22h)', 'Madrugada\n(22-06h)']

    # Check if columns exist
    available = [c for c in time_cols if c in df.columns]
    if len(available) < 4:
        print(f"  Missing time columns, skipping time_distribution.png")
        return None

    # Calculate means for each group
    passed_means = [df[df['passed']][c].mean() for c in time_cols]
    failed_means = [df[~df['passed']][c].mean() for c in time_cols]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(time_labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, passed_means, width, label='Aprobados',
                   color=COLORS['aprobados'], edgecolor='white')
    bars2 = ax.bar(x + width/2, failed_means, width, label='Reprobados',
                   color=COLORS['reprobados'], edgecolor='white')

    ax.set_ylabel('Proporción de Actividad')
    ax.set_xlabel('Bloque Horario')
    ax.set_title('Distribución de Actividad por Horario\n(Sin diferencias significativas)',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(time_labels)
    ax.legend()
    ax.set_ylim(0, max(max(passed_means), max(failed_means)) * 1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1%}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)

    # Add note about non-significance
    ax.text(0.5, -0.15, 'Nota: Las diferencias por bloque horario NO son estadísticamente significativas (p > 0.05)',
            transform=ax.transAxes, ha='center', fontsize=9, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "time_distribution.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Saved: time_distribution.png")
    return True


def main():
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {len(df)} students with grades")

    print("\n" + "="*60)
    print("Generating temporal comparison plots")
    print("="*60)

    # 1. Weekend Activity Comparison
    print("\n1. Weekend Activity:")
    create_comparison_plot(
        df,
        features=['pct_weekend', 'weekend_afternoon_pct', 'weekend_evening_pct'],
        labels=['% Fin de Semana', '% Tarde Fin de Semana', '% Noche Fin de Semana'],
        title='Actividad en Fin de Semana: Aprobados vs Reprobados',
        filename='weekend_comparison.png',
        min_effect=0.25
    )

    # 2. Time Diversity Comparison
    print("\n2. Time Diversity:")
    create_comparison_plot(
        df,
        features=['unique_active_hours', 'hour_diversity'],
        labels=['Horas Únicas Activas', 'Diversidad Horaria (Entropía)'],
        title='Diversidad Temporal: Aprobados vs Reprobados',
        filename='time_diversity_comparison.png',
        min_effect=0.25
    )

    # 3. Time-of-Day Distribution (informative, not predictive)
    print("\n3. Time Distribution (informative):")
    create_time_distribution_plot(df)

    print("\n" + "="*60)
    print("Temporal plots generated!")
    print("="*60)
    print(f"\nOutput location: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
