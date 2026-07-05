#!/usr/bin/env python3
"""
Generate Recall + False Positive Evolution Chart (Dual-Axis)

Creates a visualization showing how recall and false positive rate
evolve across weeks (2, 4, 6, Full) for the early warning system.

Data source: MODEL_RESULTS_REFERENCE.md (optimized models)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Configure output
OUTPUT_DIR = Path(__file__).parent.parent / "data/report/analysis/model_evolution"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data from MODEL_RESULTS_REFERENCE.md (optimized models)
# Strategic framing:
# - Semana 2: Best week 2 model
# - Semana 4: XGBoost V4 - Best model WITHOUT assessment features (available before evaluations exist)
# - Semana 6+: Best model WITH assessment features (Full model, available from week 6)
# FP Rate = 1 - Precision (% of alerts that are false positives)
DATA = {
    'weeks': ['Semana 2', 'Semana 4\n(sin assessment)', 'Semana 6+'],
    'weeks_numeric': [2, 4, 6],
    'recall': [81.7, 85.2, 85.9],
    'fp_rate': [45.3, 36.5, 24.3],  # 1 - Precision
    'roc_auc': [0.743, 0.849, 0.902],
    'accuracy': [65.7, 74.5, 83.4],
    'notes': [
        'Con assessment\nt=0.13, P5%',
        'SIN assessment\nXGBoost V4, t=0.20',
        'Con assessment\nt=0.33 (Full)'
    ]
}


def create_dual_axis_chart():
    """Create the main dual-axis chart showing Recall, Accuracy vs FP Rate evolution."""

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Colors
    color_recall = '#2E7D32'  # Dark green
    color_accuracy = '#1565C0'  # Blue
    color_fp = '#C62828'      # Dark red

    # X positions
    x = np.arange(len(DATA['weeks']))
    width = 0.25

    # Primary axis: Recall and Accuracy (bars)
    bars1 = ax1.bar(x - width, DATA['recall'], width,
                    label='Recall (% en riesgo detectados)',
                    color=color_recall, alpha=0.85, edgecolor='black', linewidth=1)
    bars_acc = ax1.bar(x, DATA['accuracy'], width,
                    label='Accuracy (% predicciones correctas)',
                    color=color_accuracy, alpha=0.85, edgecolor='black', linewidth=1)

    ax1.set_xlabel('Momento del Curso', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Porcentaje (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xticks(x)
    ax1.set_xticklabels(DATA['weeks'], fontsize=11)

    # Add value labels on recall bars
    for bar, val in zip(bars1, DATA['recall']):
        height = bar.get_height()
        ax1.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color_recall)

    # Add value labels on accuracy bars
    for bar, val in zip(bars_acc, DATA['accuracy']):
        height = bar.get_height()
        ax1.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color_accuracy)

    # Secondary axis: False Positive Rate (bars)
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width, DATA['fp_rate'], width,
                    label='Falsas Alarmas (%)',
                    color=color_fp, alpha=0.85, edgecolor='black', linewidth=1)

    ax2.set_ylabel('Falsas Alarmas (%)', color=color_fp, fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_fp)
    ax2.set_ylim(0, 100)

    # Add value labels on FP bars
    for bar, val in zip(bars2, DATA['fp_rate']):
        height = bar.get_height()
        ax2.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color=color_fp)

    # Title
    plt.title('Evolución del Sistema de Alerta Temprana por Semana\n' +
              'Recall + Accuracy vs Falsas Alarmas',
              fontsize=15, fontweight='bold', pad=20)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper center', bbox_to_anchor=(0.5, -0.12),
               ncol=3, fontsize=10, frameon=True)

    # Add horizontal reference lines
    ax1.axhline(y=80, color=color_recall, linestyle='--', alpha=0.3, linewidth=1)
    ax2.axhline(y=30, color=color_fp, linestyle='--', alpha=0.3, linewidth=1)

    # Grid
    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_axisbelow(True)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'recall_fp_evolution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close()


def create_combined_line_chart():
    """Create a line chart showing the trade-off over time with accuracy."""

    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Colors
    color_recall = '#2E7D32'
    color_fp = '#C62828'
    color_accuracy = '#1565C0'

    x = DATA['weeks_numeric']

    # Plot Recall
    ax1.plot(x, DATA['recall'], 'o-', color=color_recall,
             linewidth=3, markersize=12, label='Recall (%)')
    ax1.fill_between(x, DATA['recall'], alpha=0.15, color=color_recall)

    # Plot Accuracy
    ax1.plot(x, DATA['accuracy'], '^-', color=color_accuracy,
             linewidth=3, markersize=12, label='Accuracy (%)')
    ax1.fill_between(x, DATA['accuracy'], alpha=0.15, color=color_accuracy)

    # Plot FP Rate
    ax1.plot(x, DATA['fp_rate'], 's--', color=color_fp,
             linewidth=3, markersize=12, label='Falsas Alarmas (%)')
    ax1.fill_between(x, DATA['fp_rate'], alpha=0.15, color=color_fp)

    # Labels and styling
    ax1.set_xlabel('Semana del Curso', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Porcentaje (%)', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.set_xlim(1, 7)
    ax1.set_xticks([2, 4, 6])
    ax1.set_xticklabels(['Semana 2', 'Semana 4\n(sin assess.)', 'Semana 6+'], fontsize=11)

    # Add value annotations
    for i, (xi, recall, acc, fp) in enumerate(zip(x, DATA['recall'], DATA['accuracy'], DATA['fp_rate'])):
        ax1.annotate(f'{recall:.0f}%', (xi, recall),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=10, fontweight='bold', color=color_recall)
        ax1.annotate(f'{acc:.0f}%', (xi, acc),
                    textcoords="offset points", xytext=(15, 0),
                    ha='left', fontsize=10, fontweight='bold', color=color_accuracy)
        ax1.annotate(f'{fp:.0f}%', (xi, fp),
                    textcoords="offset points", xytext=(0, -15),
                    ha='center', fontsize=10, fontweight='bold', color=color_fp)

    # Title
    plt.title('Evolución de Recall, Accuracy y Falsas Alarmas\nSistema de Alerta Temprana',
              fontsize=14, fontweight='bold', pad=15)

    # Legend
    ax1.legend(loc='upper right', fontsize=11, frameon=True)

    # Grid
    ax1.grid(True, alpha=0.3)
    ax1.set_axisbelow(True)

    # Highlight zones
    ax1.axhspan(75, 90, alpha=0.1, color='green', label='_nolegend_')  # Good recall zone
    ax1.axhspan(20, 35, alpha=0.1, color='red', label='_nolegend_')    # Target FP zone

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'recall_fp_evolution_line.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close()


def create_summary_table_chart():
    """Create a visual table summarizing all metrics."""

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')

    # Table data - 3 strategic models
    table_data = [
        ['Semana 2', '0.74', '65.7%', '81.7%', '54.7%', '45.3%', 'Watch list inicial'],
        ['Semana 4\n(sin assessment)', '0.85', '74.5%', '85.2%', '63.5%', '36.5%', 'Intervencion temprana'],
        ['Semana 6+', '0.90', '83.4%', '85.9%', '75.7%', '24.3%', 'Maxima precision'],
    ]

    columns = ['Momento', 'ROC-AUC', 'Accuracy', 'Recall', 'Precision', 'Falsas Alarmas', 'Uso Recomendado']

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=columns,
                     cellLoc='center',
                     loc='center',
                     colColours=['#E3F2FD']*7)

    # Style table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Color cells based on values
    for i in range(len(table_data)):
        # Recall cells (column 3)
        recall_val = float(table_data[i][3].replace('%', ''))
        if recall_val >= 85:
            table[(i+1, 3)].set_facecolor('#C8E6C9')  # Light green
        elif recall_val >= 80:
            table[(i+1, 3)].set_facecolor('#DCEDC8')  # Lighter green

        # FP Rate cells (column 5)
        fp_val = float(table_data[i][5].replace('%', ''))
        if fp_val <= 25:
            table[(i+1, 5)].set_facecolor('#C8E6C9')  # Light green
        elif fp_val <= 35:
            table[(i+1, 5)].set_facecolor('#FFF9C4')  # Light yellow
        else:
            table[(i+1, 5)].set_facecolor('#FFCDD2')  # Light red

    # Bold header
    for j in range(len(columns)):
        table[(0, j)].set_text_props(fontweight='bold')

    plt.title('Resumen de Métricas del Sistema de Alerta Temprana\n',
              fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / 'metrics_summary_table.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"Saved: {output_path}")

    plt.close()


def main():
    print("=" * 60)
    print("Generating Recall + FP Evolution Charts")
    print("=" * 60)

    # Create all visualizations
    print("\n1. Creating dual-axis bar chart...")
    create_dual_axis_chart()

    print("\n2. Creating line chart...")
    create_combined_line_chart()

    print("\n3. Creating summary table...")
    create_summary_table_chart()

    print("\n" + "=" * 60)
    print("All charts generated successfully!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
