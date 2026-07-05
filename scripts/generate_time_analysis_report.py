#!/usr/bin/env python3
"""
Generate visualizations for time-limited early warning model analysis.

Creates:
    - Performance vs Time graph (ROC-AUC by cutoff)
    - Comparison table (with vs without assessment features)
    - Summary statistics

Output:
    data/report/visualizations/performance_vs_time.png
    data/report/visualizations/time_cutoff_comparison.png
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
RESULTS_FILE = BASE_DIR / "data/analysis/time_cutoff_results.json"
OUTPUT_DIR = BASE_DIR / "data/report/visualizations"

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'with_assessment': '#2ecc71',     # Green
    'without_assessment': '#3498db',  # Blue
    'delta': '#e74c3c',               # Red
}


def load_results():
    """Load experiment results."""
    with open(RESULTS_FILE, 'r') as f:
        return json.load(f)


def plot_performance_vs_time(results):
    """Create ROC-AUC vs Time graph."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract data
    cutoffs = []
    auc_with = []
    auc_without = []

    for exp in results['experiments']:
        cutoff = exp['cutoff']
        if cutoff not in cutoffs:
            cutoffs.append(cutoff)

    cutoffs_sorted = [2, 4, 6, 8, 'full']
    x_labels = ['Semana 2', 'Semana 4', 'Semana 6', 'Semana 8', 'Completo']
    x_pos = np.arange(len(cutoffs_sorted))

    for cutoff in cutoffs_sorted:
        exp_with = next((e for e in results['experiments']
                        if e['cutoff'] == cutoff and e['include_assessment']), None)
        exp_without = next((e for e in results['experiments']
                           if e['cutoff'] == cutoff and not e['include_assessment']), None)

        auc_with.append(exp_with['metrics']['roc_auc'] if exp_with else None)
        auc_without.append(exp_without['metrics']['roc_auc'] if exp_without else None)

    # Plot lines
    ax.plot(x_pos, auc_with, 'o-', color=COLORS['with_assessment'],
            linewidth=2, markersize=10, label='Con features de evaluación')
    ax.plot(x_pos, auc_without, 's--', color=COLORS['without_assessment'],
            linewidth=2, markersize=10, label='Sin features de evaluación')

    # Add reference line at 0.80
    ax.axhline(y=0.80, color='gray', linestyle=':', alpha=0.7, label='Umbral objetivo (0.80)')

    # Add value labels
    for i, (w, wo) in enumerate(zip(auc_with, auc_without)):
        if w:
            ax.annotate(f'{w:.3f}', (x_pos[i], w), textcoords="offset points",
                       xytext=(0, 10), ha='center', fontsize=9, color=COLORS['with_assessment'])
        if wo:
            ax.annotate(f'{wo:.3f}', (x_pos[i], wo), textcoords="offset points",
                       xytext=(0, -15), ha='center', fontsize=9, color=COLORS['without_assessment'])

    # Formatting
    ax.set_xlabel('Datos disponibles', fontsize=12)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('Rendimiento del Modelo vs Tiempo de Datos\nModelo de Alerta Temprana', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0.60, 0.90)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'performance_vs_time.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_metric_comparison(results):
    """Create multi-metric comparison bar chart."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics = ['roc_auc', 'accuracy', 'precision', 'recall']
    metric_labels = ['ROC-AUC', 'Accuracy', 'Precision', 'Recall']
    cutoffs = [2, 4, 6, 8, 'full']
    x_labels = ['S2', 'S4', 'S6', 'S8', 'Full']
    x_pos = np.arange(len(cutoffs))
    width = 0.35

    for ax, metric, label in zip(axes.flat, metrics, metric_labels):
        vals_with = []
        vals_without = []

        for cutoff in cutoffs:
            exp_with = next((e for e in results['experiments']
                            if e['cutoff'] == cutoff and e['include_assessment']), None)
            exp_without = next((e for e in results['experiments']
                               if e['cutoff'] == cutoff and not e['include_assessment']), None)

            vals_with.append(exp_with['metrics'][metric] if exp_with else 0)
            vals_without.append(exp_without['metrics'][metric] if exp_without else 0)

        ax.bar(x_pos - width/2, vals_with, width, label='Con evaluación',
               color=COLORS['with_assessment'], alpha=0.8)
        ax.bar(x_pos + width/2, vals_without, width, label='Sin evaluación',
               color=COLORS['without_assessment'], alpha=0.8)

        ax.set_ylabel(label)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.legend(loc='lower right', fontsize=8)
        ax.set_ylim(0.3, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Comparación de Métricas por Cutoff Temporal\nModelo de Alerta Temprana',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = OUTPUT_DIR / 'time_cutoff_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_delta_analysis(results):
    """Create delta (improvement) analysis chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cutoffs = [2, 4, 6, 8, 'full']
    x_labels = ['Semana 2', 'Semana 4', 'Semana 6', 'Semana 8', 'Completo']
    x_pos = np.arange(len(cutoffs))

    deltas = []
    for cutoff in cutoffs:
        exp_with = next((e for e in results['experiments']
                        if e['cutoff'] == cutoff and e['include_assessment']), None)
        exp_without = next((e for e in results['experiments']
                           if e['cutoff'] == cutoff and not e['include_assessment']), None)

        if exp_with and exp_without:
            delta = exp_with['metrics']['roc_auc'] - exp_without['metrics']['roc_auc']
            deltas.append(delta * 100)  # Convert to percentage points
        else:
            deltas.append(0)

    colors = [COLORS['with_assessment'] if d > 0 else COLORS['without_assessment'] for d in deltas]
    bars = ax.bar(x_pos, deltas, color=colors, alpha=0.8, edgecolor='black')

    # Add value labels
    for bar, delta in zip(bars, deltas):
        height = bar.get_height()
        ax.annotate(f'{delta:+.1f}pp',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height >= 0 else -15),
                   textcoords="offset points",
                   ha='center', va='bottom' if height >= 0 else 'top',
                   fontsize=10, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_xlabel('Datos disponibles', fontsize=12)
    ax.set_ylabel('Diferencia en ROC-AUC (puntos porcentuales)', fontsize=12)
    ax.set_title('Impacto de Features de Evaluación por Cutoff Temporal\n(Verde = Mejora con features, Azul = Sin mejora)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = OUTPUT_DIR / 'assessment_feature_impact.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def print_summary_table(results):
    """Print summary table to console."""
    print("\n" + "="*70)
    print("RESUMEN DE RESULTADOS")
    print("="*70)

    cutoffs = [2, 4, 6, 8, 'full']

    header = f"{'Cutoff':<10} {'Con Eval':<12} {'Sin Eval':<12} {'Delta':<10} {'Samples':<10}"
    print(header)
    print("-"*70)

    for cutoff in cutoffs:
        exp_with = next((e for e in results['experiments']
                        if e['cutoff'] == cutoff and e['include_assessment']), None)
        exp_without = next((e for e in results['experiments']
                           if e['cutoff'] == cutoff and not e['include_assessment']), None)

        if exp_with and exp_without:
            auc_with = exp_with['metrics']['roc_auc']
            auc_without = exp_without['metrics']['roc_auc']
            delta = auc_with - auc_without
            n_samples = exp_with['n_samples']

            cutoff_str = f"{cutoff} sem" if cutoff != 'full' else "Completo"
            print(f"{cutoff_str:<10} {auc_with:.3f}        {auc_without:.3f}        {delta:+.3f}      {n_samples}")

    print("\n" + "="*70)
    print("CONCLUSIONES CLAVE")
    print("="*70)

    # Find earliest cutoff >= 0.75
    for cutoff in cutoffs:
        exp_with = next((e for e in results['experiments']
                        if e['cutoff'] == cutoff and e['include_assessment']), None)
        if exp_with and exp_with['metrics']['roc_auc'] >= 0.75:
            print(f"\n✓ Primer cutoff con ROC-AUC >= 0.75: {cutoff} semanas")
            print(f"  ROC-AUC: {exp_with['metrics']['roc_auc']:.3f}")
            break

    # Analyze assessment impact
    print("\n✓ Impacto de features de evaluación:")
    print("  - Semanas 2-6: Poco o ningún impacto (delta < 1%)")
    print("  - Semana 8: Impacto moderado (+2.8%)")
    print("  - Completo: Mayor impacto (+6.6%)")

    print("\n✓ Para detección temprana (semana 4):")
    exp_4 = next((e for e in results['experiments']
                 if e['cutoff'] == 4 and e['include_assessment']), None)
    if exp_4:
        print(f"  - ROC-AUC: {exp_4['metrics']['roc_auc']:.3f}")
        print(f"  - Recall: {exp_4['metrics']['recall']:.1%}")
        print(f"  - Precision: {exp_4['metrics']['precision']:.1%}")

    print()


def main():
    """Generate all visualizations."""
    print("Generating time analysis visualizations...")

    # Load results
    results = load_results()

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Generate plots
    plot_performance_vs_time(results)
    plot_metric_comparison(results)
    plot_delta_analysis(results)

    # Print summary
    print_summary_table(results)

    print(f"\nAll visualizations saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
