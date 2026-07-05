#!/usr/bin/env python3
"""
Generate pass/fail comparison visualizations for the executive report.
Creates boxplots and bar charts comparing behavioral features between
approved and failed students.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_DIR = DATA_DIR / "enriched_features"
OUTPUT_DIR = DATA_DIR / "report" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Colors
COLORS = {
    'aprobados': '#2ecc71',  # Green
    'reprobados': '#e74c3c'  # Red
}

# Style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.facecolor'] = 'white'


def load_data():
    """Load and merge all feature datasets with enrollment grades."""
    # Load enrollments
    enroll = pd.read_csv(DATA_DIR / "page_views" / "student_enrollments.csv")
    enroll['failed'] = enroll['final_score'] < 57
    enroll = enroll[enroll['final_score'].notna()]

    # Load features
    norm_features = pd.read_parquet(FEATURES_DIR / "normalized_features.parquet")
    ngram_features = pd.read_parquet(FEATURES_DIR / "ngram_features.parquet")

    # Merge
    df = norm_features.merge(
        ngram_features[['user_id', 'course_id', 'total_transitions',
                        'transition_entropy', 'transition_diversity']],
        on=['user_id', 'course_id'],
        how='left'
    )

    df = df.merge(
        enroll[['user_id', 'course_id', 'final_score', 'failed']],
        on=['user_id', 'course_id'],
        how='inner'
    )

    print(f"Total students: {len(df)}")
    print(f"Passed: {(~df['failed']).sum()} ({(~df['failed']).mean()*100:.1f}%)")
    print(f"Failed: {df['failed'].sum()} ({df['failed'].mean()*100:.1f}%)")

    return df


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std


def create_comparison_boxplot(df, features, labels, title, output_name, figsize=(15, 5)):
    """Create side-by-side boxplots comparing aprobados vs reprobados."""
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=figsize)

    if n_features == 1:
        axes = [axes]

    for i, (feat, label) in enumerate(zip(features, labels)):
        if feat not in df.columns:
            print(f"Warning: {feat} not in dataframe")
            continue

        ax = axes[i]
        passed = df[~df['failed']][feat].dropna()
        failed = df[df['failed']][feat].dropna()

        bp = ax.boxplot([passed, failed], patch_artist=True, widths=0.6)
        bp['boxes'][0].set_facecolor(COLORS['aprobados'])
        bp['boxes'][1].set_facecolor(COLORS['reprobados'])

        # Style boxes
        for box in bp['boxes']:
            box.set_alpha(0.7)

        ax.set_xticklabels(['Aprobados', 'Reprobados'])
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel('Valor')

        # Add statistics
        d = calculate_cohens_d(passed, failed)
        _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        # Add annotation
        ax.text(0.5, 0.98, f'd={d:.2f} {sig}', transform=ax.transAxes,
                ha='center', va='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_name}")


def create_session_gap_comparison(df):
    """Create visualization for session gap features."""
    features = [
        'session_gap_max_days',
        'session_gap_median_hours',
        'session_gap_cv'
    ]
    labels = [
        'Gap Máximo (días)',
        'Gap Mediana (horas)',
        'Variabilidad del Gap (CV)'
    ]

    create_comparison_boxplot(
        df, features, labels,
        'Análisis de Regularidad: Gap entre Sesiones',
        'session_gap_comparison.png',
        figsize=(12, 5)
    )


def create_navigation_comparison(df):
    """Create visualization for navigation/transition features."""
    features = [
        'total_transitions',
        'transition_entropy',
        'transition_diversity',
        'content_vs_assessment_ratio'
    ]
    labels = [
        'Transiciones Totales',
        'Entropía de Navegación',
        'Diversidad de Transición',
        'Ratio Contenido/Evaluación'
    ]

    create_comparison_boxplot(
        df, features, labels,
        'Patrones de Navegación: N-gramas y Diversidad',
        'navigation_comparison.png',
        figsize=(16, 5)
    )


def create_module_pct_comparison(df):
    """Create visualization for module percentile features."""
    features = [
        'modu_mean_pct',
        'modu_access_rate',
        'modu_top25_rate',
        'modu_top50_rate'
    ]
    labels = [
        'Percentil Promedio',
        'Tasa de Acceso',
        'Top 25% Rate',
        'Top 50% Rate'
    ]

    create_comparison_boxplot(
        df, features, labels,
        'Posición Relativa en Módulos',
        'module_pct_comparison.png',
        figsize=(16, 5)
    )


def create_top_discriminating_features(df):
    """Find and visualize top features by effect size (excluding grade-based features)."""
    # Columns to analyze (exclude IDs, target, and grade-related features)
    exclude = ['user_id', 'course_id', 'final_score', 'failed']

    # Exclude assessment/grade-related features (data leakage)
    leaky_prefixes = ['assi_', 'quiz_', 'submission_', 'grade_']

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [c for c in numeric_cols
                    if c not in exclude
                    and not any(c.startswith(p) for p in leaky_prefixes)]

    print(f"Analyzing {len(feature_cols)} features (excluding grade-related)")

    # Calculate effect sizes
    effect_sizes = []
    for col in feature_cols:
        passed = df[~df['failed']][col].dropna()
        failed = df[df['failed']][col].dropna()

        if len(passed) > 5 and len(failed) > 5:
            d = calculate_cohens_d(passed, failed)
            _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
            effect_sizes.append({
                'feature': col,
                'cohens_d': d,
                'abs_d': abs(d),
                'p_value': p,
                'passed_mean': passed.mean(),
                'failed_mean': failed.mean()
            })

    effects_df = pd.DataFrame(effect_sizes)
    effects_df = effects_df.sort_values('abs_d', ascending=False)

    # Get top 6 most discriminating
    top_features = effects_df.head(6)
    print("\nTop 6 discriminating features:")
    print(top_features[['feature', 'cohens_d', 'p_value']])

    # Create visualization
    features = top_features['feature'].tolist()

    # Create nicer labels
    label_map = {
        'session_gap_max_days': 'Gap Máximo (días)',
        'session_gap_median_hours': 'Gap Mediana (horas)',
        'session_gap_cv': 'Variabilidad Gap',
        'session_count': 'Total Sesiones',
        'sessions_per_week': 'Sesiones/Semana',
        'total_page_views': 'Page Views',
        'unique_active_hours': 'Horas Activas',
        'session_regularity': 'Regularidad',
        'modu_mean_pct': 'Percentil Módulos',
        'total_transitions': 'Transiciones',
        'content_vs_assessment_ratio': 'Ratio Contenido/Eval',
        'session_duration_mean': 'Duración Media Sesión',
        'views_per_session': 'Views por Sesión',
    }

    labels = [label_map.get(f, f.replace('_', ' ').title()) for f in features]

    create_comparison_boxplot(
        df, features, labels,
        'Top 6 Features Discriminantes (Mayor Efecto)',
        'top_discriminating_features.png',
        figsize=(18, 5)
    )

    # Save effect sizes report
    effects_df.to_csv(OUTPUT_DIR / 'effect_sizes_all_features.csv', index=False)
    print(f"Saved: effect_sizes_all_features.csv")

    return effects_df


def create_gap_threshold_analysis(df):
    """Create detailed analysis of the gap threshold insight."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Histogram of gap_max_days by outcome
    ax1 = axes[0]
    passed = df[~df['failed']]['session_gap_max_days'].dropna()
    failed = df[df['failed']]['session_gap_max_days'].dropna()

    bins = np.linspace(0, max(passed.max(), failed.max()), 30)
    ax1.hist(passed, bins=bins, alpha=0.7, label='Aprobados', color=COLORS['aprobados'])
    ax1.hist(failed, bins=bins, alpha=0.7, label='Reprobados', color=COLORS['reprobados'])

    # Add threshold line at max of passed
    threshold = passed.max()
    ax1.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Umbral: {threshold:.0f} días')

    ax1.set_xlabel('Gap Máximo entre Sesiones (días)')
    ax1.set_ylabel('Frecuencia')
    ax1.set_title('Distribución del Gap Máximo')
    ax1.legend()

    # 2. Scatter plot: gap vs final score
    ax2 = axes[1]
    colors = df['failed'].map({True: COLORS['reprobados'], False: COLORS['aprobados']})
    ax2.scatter(df['session_gap_max_days'], df['final_score'],
                c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)

    ax2.axhline(57, color='orange', linestyle='--', label='Umbral aprobación (57%)')
    ax2.axvline(threshold, color='black', linestyle='--', label=f'Gap máximo aprobados: {threshold:.0f} días')

    ax2.set_xlabel('Gap Máximo entre Sesiones (días)')
    ax2.set_ylabel('Nota Final (%)')
    ax2.set_title('Relación Gap Máximo vs Nota Final')
    ax2.legend()

    plt.suptitle('Insight Clave: Umbral de Regularidad', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'gap_threshold_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: gap_threshold_analysis.png")

    # Report statistics
    failed_above_threshold = (df[df['failed']]['session_gap_max_days'] > threshold).sum()
    passed_above_threshold = (df[~df['failed']]['session_gap_max_days'] > threshold).sum()

    print(f"\nGap Threshold Analysis:")
    print(f"  Max gap for passed students: {threshold:.1f} days")
    print(f"  Failed students above threshold: {failed_above_threshold}")
    print(f"  Passed students above threshold: {passed_above_threshold}")


def main():
    print("=" * 60)
    print("Generating Pass/Fail Comparison Visualizations")
    print("=" * 60)

    # Load data
    df = load_data()

    # Generate visualizations
    print("\n--- Session Gap Analysis ---")
    create_session_gap_comparison(df)

    print("\n--- Navigation Patterns ---")
    create_navigation_comparison(df)

    print("\n--- Module Percentile ---")
    create_module_pct_comparison(df)

    print("\n--- Top Discriminating Features ---")
    effects_df = create_top_discriminating_features(df)

    print("\n--- Gap Threshold Deep Dive ---")
    create_gap_threshold_analysis(df)

    print("\n" + "=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
