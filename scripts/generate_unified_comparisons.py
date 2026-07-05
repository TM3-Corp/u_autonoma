#!/usr/bin/env python3
"""
Generate unified pass/fail comparison plots with consistent styling.
Only includes features with significant differences (|d| >= 0.4 or p < 0.001).

Style based on original pass_fail_comparison.png:
- Colors: #6bcb77 (green), #ff6b6b (red)
- Narrower boxes
- Subtle Cohen's d annotation
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

# === UNIFIED STYLE (from original pass_fail_comparison.png) ===
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


def create_unified_comparison(df, features, labels, title, filename, min_effect_size=0.3):
    """
    Create comparison plot with unified style.
    Only includes features with |d| >= min_effect_size.
    """
    # Filter to significant features first
    significant_features = []
    significant_labels = []

    for feat, label in zip(features, labels):
        if feat not in df.columns:
            continue
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()
        if len(passed) < 5 or len(failed) < 5:
            continue
        d = calculate_cohens_d(passed, failed)
        _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')

        # Include if effect size is meaningful OR highly significant
        if abs(d) >= min_effect_size or p < 0.001:
            significant_features.append(feat)
            significant_labels.append(label)

    if not significant_features:
        print(f"  No significant features for {filename}, skipping.")
        return None

    n = len(significant_features)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 5))

    if n == 1:
        axes = [axes]

    results = []

    for i, (feat, label) in enumerate(zip(significant_features, significant_labels)):
        ax = axes[i]
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()

        # Create boxplot with default width (thin boxes like pass_fail_comparison.png)
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

        # Calculate stats
        d = calculate_cohens_d(passed, failed)
        _, p = stats.mannwhitneyu(passed, failed, alternative='two-sided')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''

        # Add subtle Cohen's d annotation inside plot (top center)
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


# === LOAD DATA ===
print("Loading data...")

# Engagement dynamics data (session-based features)
df_engagement = pd.read_csv(DATA_DIR / "engagement_dynamics" / "student_features.csv")
df_engagement['passed'] = df_engagement['final_score'] >= 57
df_engagement = df_engagement[df_engagement['final_score'].notna()]

# Normalized features (n-gram, module percentile)
df_normalized = pd.read_parquet(DATA_DIR / "enriched_features" / "normalized_features.parquet")
df_ngram = pd.read_parquet(DATA_DIR / "enriched_features" / "ngram_features.parquet")

# Merge normalized with grades
enroll = pd.read_csv(DATA_DIR / "page_views" / "student_enrollments.csv")
df_full = df_normalized.merge(
    df_ngram[['user_id', 'course_id', 'total_transitions', 'transition_entropy', 'transition_diversity']],
    on=['user_id', 'course_id'], how='left'
)
df_full = df_full.merge(
    enroll[['user_id', 'course_id', 'final_score']],
    on=['user_id', 'course_id'], how='inner'
)
df_full['passed'] = df_full['final_score'] >= 57

print(f"Engagement data: {len(df_engagement)} students")
print(f"Full features data: {len(df_full)} students")

# === GENERATE PLOTS ===
print("\n" + "="*60)
print("Generating unified comparison plots")
print("="*60)

# 1. Core Session Features (from engagement_dynamics)
print("\n1. Session Features:")
create_unified_comparison(
    df_engagement,
    features=['session_count', 'sessions_per_week', 'session_gap_mean', 'activity_span_days'],
    labels=['Total Sesiones', 'Sesiones/Semana', 'Gap Medio (horas)', 'Span Actividad (días)'],
    title='Características de Sesión: Aprobados vs Reprobados',
    filename='session_features_comparison.png'
)

# 2. Activity Volume (from engagement_dynamics)
print("\n2. Activity Volume:")
create_unified_comparison(
    df_engagement,
    features=['total_page_views', 'total_participations', 'unique_active_hours'],
    labels=['Page Views', 'Participaciones', 'Horas Activas'],
    title='Volumen de Actividad: Aprobados vs Reprobados',
    filename='activity_volume_comparison.png'
)

# 3. Navigation Patterns (n-grams from normalized features)
print("\n3. Navigation Patterns:")
create_unified_comparison(
    df_full,
    features=['total_transitions', 'transition_entropy', 'transition_diversity', 'content_vs_assessment_ratio'],
    labels=['Transiciones Totales', 'Entropía de Navegación', 'Diversidad de Transición', 'Ratio Contenido/Evaluación'],
    title='Patrones de Navegación: N-gramas y Diversidad',
    filename='navigation_comparison.png'
)

# 4. Module Percentile (from normalized features)
print("\n4. Module Percentile:")
create_unified_comparison(
    df_full,
    features=['modu_mean_pct', 'modu_access_rate', 'modu_top25_rate', 'modu_top50_rate'],
    labels=['Percentil Promedio', 'Tasa de Acceso', 'Top 25% Rate', 'Top 50% Rate'],
    title='Posición Relativa en Módulos',
    filename='module_pct_comparison.png',
    min_effect_size=0.2  # Lower threshold for these
)

# 5. Session Gap Deep Dive (from normalized features)
print("\n5. Session Gap:")
create_unified_comparison(
    df_full,
    features=['session_gap_max_days', 'session_gap_mean', 'session_gap_cv'],
    labels=['Gap Máximo (días)', 'Gap Medio (horas)', 'Variabilidad del Gap'],
    title='Regularidad: Gap entre Sesiones',
    filename='session_gap_comparison.png'
)

# === SUMMARY ===
print("\n" + "="*60)
print("Unified plots generated!")
print("="*60)
print("\nPlots with consistent style created in:")
print(f"  {OUTPUT_DIR}")
