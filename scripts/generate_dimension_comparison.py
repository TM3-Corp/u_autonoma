#!/usr/bin/env python3
"""
Generate a comprehensive comparison showing the TOP discriminating feature
from each behavioral dimension - ideal for presenting to authorities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

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
    """Load and merge all feature data."""
    df = pd.read_csv("data/engagement_dynamics/student_features.csv")
    df['passed'] = df['final_score'] >= 57

    try:
        time_df = pd.read_parquet("data/enriched_features/time_features.parquet")
        df = df.merge(time_df, on=['user_id', 'course_id'], how='left')
    except: pass

    try:
        ngram_df = pd.read_parquet("data/enriched_features/ngram_features.parquet")
        df = df.merge(ngram_df, on=['user_id', 'course_id'], how='left')
    except: pass

    return df[df['final_score'].notna()]


def create_dimension_summary_plot(df):
    """Create the main summary plot showing top feature per dimension."""

    # Define TOP feature per dimension (selected for impact and interpretability)
    dimension_features = [
        ('Volumen', 'session_count', 'Total de Sesiones', '1.85x más sesiones'),
        ('Regularidad', 'sessions_per_week', 'Sesiones por Semana', '1.69x más frecuente'),
        ('Brecha Temporal', 'session_gap_mean', 'Gap Medio (horas)', '39% menos espera'),
        ('Diversidad Horaria', 'hour_diversity', 'Entropía Horaria', '13% más diverso'),
        ('Fin de Semana', 'weekend_afternoon_pct', '% Tarde Fin de Semana', '2.4x más actividad'),
        ('Navegación', 'transition_entropy', 'Entropía de Transiciones', '19% más exploración'),
    ]

    n = len(dimension_features)
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    results = []

    for i, (dim, feat, label, insight) in enumerate(dimension_features):
        ax = axes[i]

        if feat not in df.columns:
            ax.text(0.5, 0.5, f'Feature {feat} not found', ha='center', va='center')
            continue

        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()

        # Create boxplot
        bp = ax.boxplot([passed, failed], patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS['aprobados'])
        bp["boxes"][1].set_facecolor(COLORS['reprobados'])

        # Style median
        for median in bp['medians']:
            median.set_color('#d35400')
            median.set_linewidth(2)

        ax.set_xticklabels(["Aprobados", "Reprobados"])
        ax.set_title(f'{dim}\n{label}', fontweight='bold', fontsize=11)
        ax.set_ylabel("Valor")

        # Calculate Cohen's d
        d = calculate_cohens_d(passed, failed)
        _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        # Add Cohen's d annotation
        effect_label = f'd={d:.2f}{sig}'
        ax.text(0.5, 0.95, effect_label, transform=ax.transAxes,
                ha='center', va='top', fontsize=10, fontweight='bold',
                color='#2c3e50', bbox=dict(boxstyle='round,pad=0.2',
                facecolor='white', edgecolor='none', alpha=0.8))

        # Add insight annotation at bottom
        ax.text(0.5, -0.12, insight, transform=ax.transAxes,
                ha='center', va='top', fontsize=9, style='italic',
                color='#555555')

        results.append({
            'dimension': dim,
            'feature': feat,
            'label': label,
            'cohens_d': d,
            'p_value': p,
            'passed_mean': passed.mean(),
            'failed_mean': failed.mean(),
            'insight': insight
        })

    plt.suptitle('Principales Diferencias Conductuales: Aprobados vs Reprobados\n(Top Feature por Dimensión)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_features_by_dimension.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: top_features_by_dimension.png")
    return results


def create_effect_size_summary(df):
    """Create horizontal bar chart showing effect sizes by dimension."""

    # All significant features by dimension
    all_features = [
        ('Volumen', 'session_count', 'Sesiones Totales'),
        ('Frecuencia', 'sessions_per_week', 'Sesiones/Semana'),
        ('Regularidad', 'session_gap_mean', 'Gap Medio'),
        ('Diversidad', 'hour_diversity', 'Diversidad Horaria'),
        ('Fin de Semana', 'weekend_afternoon_pct', '% Tarde Fds'),
        ('Navegación', 'transition_entropy', 'Entropía Nav.'),
        ('Actividad', 'total_page_views', 'Page Views'),
        ('Span', 'activity_span_days', 'Días Activo'),
        ('Trayectoria', 'weekly_cv', 'Variab. Semanal'),
    ]

    results = []
    for dim, feat, label in all_features:
        if feat not in df.columns:
            continue
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()
        if len(passed) > 10 and len(failed) > 10:
            d = calculate_cohens_d(passed, failed)
            _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
            results.append((f'{dim}: {label}', d, p))

    # Sort by absolute effect size
    results.sort(key=lambda x: abs(x[1]), reverse=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    labels = [r[0] for r in results]
    values = [r[1] for r in results]
    colors = [COLORS['aprobados'] if v > 0 else COLORS['reprobados'] for v in values]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor='white', height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (Tamaño del Efecto)")
    ax.set_title("Tamaño del Efecto por Dimensión Conductual\n(Verde = Mayor en Aprobados, Rojo = Mayor en Reprobados)",
                 fontweight='bold')

    # Add effect size thresholds
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Pequeño (0.2)')
    ax.axvline(x=0.5, color='gray', linestyle='-.', alpha=0.5, label='Mediano (0.5)')
    ax.axvline(x=0.8, color='gray', linestyle=':', alpha=0.5, label='Grande (0.8)')
    ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle='-.', alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle=':', alpha=0.5)

    # Add value labels
    for bar, val, (_, _, p) in zip(bars, values, results):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        ax.text(val + 0.02 if val > 0 else val - 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}{sig}', va='center', ha='left' if val > 0 else 'right',
                fontsize=9, fontweight='bold')

    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(-0.8, 1.0)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "effect_sizes_by_dimension.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: effect_sizes_by_dimension.png")


def create_ratio_comparison(df):
    """Create a simple ratio comparison showing 'X times more' for each dimension."""

    features = [
        ('session_count', 'Sesiones Totales', 'veces más'),
        ('sessions_per_week', 'Frecuencia Semanal', 'veces más'),
        ('weekend_afternoon_pct', 'Actividad Fin de Semana', 'veces más'),
        ('total_transitions', 'Transiciones de Navegación', 'veces más'),
        ('hour_diversity', 'Diversidad Horaria', 'veces más'),
        ('activity_span_days', 'Días de Actividad', 'veces más'),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    ratios = []

    for feat, label, suffix in features:
        if feat not in df.columns:
            continue
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()
        if failed.mean() > 0:
            ratio = passed.mean() / failed.mean()
            labels.append(label)
            ratios.append(ratio)

    y_pos = np.arange(len(labels))
    colors = [COLORS['aprobados'] if r > 1 else COLORS['reprobados'] for r in ratios]

    bars = ax.barh(y_pos, ratios, color=colors, edgecolor='white', height=0.6)

    ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, label='Sin diferencia')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Ratio (Aprobados / Reprobados)")
    ax.set_title("¿Cuánto Más Activos Son los Estudiantes Aprobados?\n(Ratio de medias)",
                 fontweight='bold')

    # Add value labels
    for bar, ratio in zip(bars, ratios):
        ax.text(ratio + 0.05, bar.get_y() + bar.get_height()/2,
                f'{ratio:.2f}x', va='center', ha='left',
                fontsize=10, fontweight='bold')

    ax.set_xlim(0, max(ratios) * 1.3)
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "ratio_comparison.png", dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: ratio_comparison.png")


def main():
    print("Loading data...")
    df = load_data()
    print(f"Data loaded: {len(df)} students")

    print("\n" + "="*60)
    print("Generating dimension comparison plots")
    print("="*60 + "\n")

    # 1. Main summary plot (6 dimensions, top feature each)
    results = create_dimension_summary_plot(df)

    # 2. Effect size bar chart
    create_effect_size_summary(df)

    # 3. Ratio comparison
    create_ratio_comparison(df)

    print("\n" + "="*60)
    print("Summary of Top Features by Dimension:")
    print("="*60)
    for r in results:
        print(f"  {r['dimension']:15} | d={r['cohens_d']:+.2f} | {r['insight']}")

    print(f"\nPlots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
