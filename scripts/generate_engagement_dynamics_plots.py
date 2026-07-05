#!/usr/bin/env python3
"""
Generate pass/fail comparison plots using the engagement_dynamics dataset.
This dataset has stronger discriminative power due to session-based features.
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

# Load the engagement dynamics data (stronger signal)
df = pd.read_csv(DATA_DIR / "engagement_dynamics" / "student_features.csv")
df['passed'] = df['final_score'] >= 57
df = df[df['final_score'].notna()]

print(f"Total students: {len(df)}")
print(f"Passed: {df['passed'].sum()} ({df['passed'].mean()*100:.1f}%)")
print(f"Failed: {(~df['passed']).sum()} ({(~df['passed']).mean()*100:.1f}%)")

# Style matching the original pass_fail_comparison.png
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"

COLORS = {
    'aprobados': '#6bcb77',  # Green (original)
    'reprobados': '#ff6b6b'  # Red (original)
}


def calculate_stats(passed, failed):
    """Calculate Cohen's d and p-value."""
    n1, n2 = len(passed), len(failed)
    pooled_std = np.sqrt(((n1-1)*passed.var() + (n2-1)*failed.var()) / (n1+n2-2))
    cohens_d = (passed.mean() - failed.mean()) / pooled_std if pooled_std > 0 else 0
    _, p_val = stats.mannwhitneyu(passed, failed, alternative='two-sided')
    return cohens_d, p_val


def create_comparison_plot(features, labels, title, filename):
    """Create side-by-side boxplots matching original style."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3*n, 5))

    if n == 1:
        axes = [axes]

    stats_results = []

    for i, (feat, label) in enumerate(zip(features, labels)):
        if feat not in df.columns:
            print(f"  Warning: {feat} not found")
            continue

        ax = axes[i]
        passed = df[df['passed']][feat].dropna()
        failed = df[~df['passed']][feat].dropna()

        bp = ax.boxplot([passed, failed], patch_artist=True)
        bp["boxes"][0].set_facecolor(COLORS['aprobados'])
        bp["boxes"][1].set_facecolor(COLORS['reprobados'])

        ax.set_xticklabels(["Aprobados", "Reprobados"])
        ax.set_title(label, fontweight='bold')
        ax.set_ylabel("Valor")

        # Calculate stats
        d, p = calculate_stats(passed, failed)
        ratio = passed.mean() / failed.mean() if failed.mean() > 0 else float('inf')

        stats_results.append({
            'feature': feat,
            'label': label,
            'passed_mean': passed.mean(),
            'failed_mean': failed.mean(),
            'ratio': ratio,
            'cohens_d': d,
            'p_value': p
        })

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Saved: {filename}")
    return stats_results


# === PLOT 1: Session Features (Core insight) ===
print("\n=== Session Features ===")
stats1 = create_comparison_plot(
    features=['session_count', 'sessions_per_week', 'session_gap_mean', 'activity_span_days'],
    labels=['Total Sesiones', 'Sesiones/Semana', 'Gap Medio (horas)', 'Span Actividad (días)'],
    title='Características de Sesión: Aprobados vs Reprobados',
    filename='session_features_comparison.png'
)

# === PLOT 2: Engagement Trajectory ===
print("\n=== Engagement Trajectory ===")
stats2 = create_comparison_plot(
    features=['engagement_velocity', 'early_engagement_ratio', 'late_surge', 'weekly_cv'],
    labels=['Velocidad Engagement', 'Ratio Inicio', 'Surge Final', 'Variabilidad Semanal'],
    title='Trayectoria de Engagement: Aprobados vs Reprobados',
    filename='engagement_trajectory_comparison.png'
)

# === PLOT 3: Time-to-Access (Procrastination) ===
print("\n=== Time-to-Access ===")
stats3 = create_comparison_plot(
    features=['first_access_day', 'first_module_day', 'first_assignment_day', 'activity_span_days'],
    labels=['Primer Acceso (días)', 'Primer Módulo (días)', 'Primera Tarea (días)', 'Span Actividad (días)'],
    title='Indicadores de Procrastinación: Aprobados vs Reprobados',
    filename='procrastination_comparison.png'
)

# === PLOT 4: Activity Volume ===
print("\n=== Activity Volume ===")
stats4 = create_comparison_plot(
    features=['total_page_views', 'total_participations', 'unique_active_hours'],
    labels=['Page Views', 'Participaciones', 'Horas Únicas Activas'],
    title='Volumen de Actividad: Aprobados vs Reprobados',
    filename='activity_volume_comparison.png'
)

# === PLOT 5: Time Block Preferences ===
print("\n=== Time Block Preferences ===")
stats5 = create_comparison_plot(
    features=['weekday_evening_pct', 'weekend_total_sd', 'weekday_morning_pct', 'weekday_afternoon_pct'],
    labels=['% Vespertino', 'Variab. Fin Semana', '% Mañana', '% Tarde'],
    title='Preferencias Horarias: Aprobados vs Reprobados',
    filename='time_block_comparison.png'
)

# === Summary Statistics ===
print("\n" + "="*70)
print("RESUMEN DE ESTADÍSTICAS")
print("="*70)

all_stats = stats1 + stats2 + stats3 + stats4 + stats5
all_stats = [s for s in all_stats if s]  # Remove None

# Sort by absolute Cohen's d
all_stats.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

print(f"\n{'Feature':<30} {'Aprob':<10} {'Reprob':<10} {'Ratio':<8} {'Cohen d':<10} {'p-value':<12}")
print("-"*80)
for s in all_stats:
    sig = '***' if s['p_value'] < 0.001 else '**' if s['p_value'] < 0.01 else '*' if s['p_value'] < 0.05 else ''
    print(f"{s['label']:<30} {s['passed_mean']:<10.2f} {s['failed_mean']:<10.2f} {s['ratio']:<8.2f} {s['cohens_d']:<10.3f} {s['p_value']:<10.2e} {sig}")

# Save stats to JSON
import json
with open(OUTPUT_DIR / 'engagement_dynamics_stats.json', 'w') as f:
    json.dump(all_stats, f, indent=2)
print(f"\nSaved: engagement_dynamics_stats.json")

print("\n" + "="*70)
print("All plots generated successfully!")
print("="*70)
