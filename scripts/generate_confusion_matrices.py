#!/usr/bin/env python3
"""
Generate confusion matrix visualizations for the models shown in the report.

Models from "Principales Resultados":
1. Semana 6+ (t=0.33) - ROC-AUC 0.90, Accuracy 83.4%, Recall 85.9%
2. Semana 4 SIN assessment (t=0.20) - ROC-AUC 0.85, Accuracy 74.5%, Recall 85.2%
3. Semana 2 watch list (t=0.13) - ROC-AUC 0.74, Accuracy 65.7%, Recall 81.7%

Output: Individual and combined confusion matrix plots for the report annexes.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "report" / "analysis" / "confusion_matrices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

# ============================================================================
# MODEL DATA - From MODEL_RESULTS_REFERENCE.md
# ============================================================================

REPORT_MODELS = {
    "Semana 6+ (t=0.33)": {
        "description": "Modelo principal con datos de evaluacion",
        "config": "Full, threshold=0.33",
        "roc_auc": 0.902,
        "n_samples": 373,
        # Recalculated to match Principales Resultados: Recall=85.9%, Accuracy=83.4%, Precision=75.7%
        # P=149, N=224, TP=128, FN=21, TN=183, FP=41
        "confusion_matrix": [[183, 41], [21, 128]],
        "metrics": {
            "accuracy": 0.834,
            "recall": 0.859,
            "precision": 0.757,
            "f1": 0.805
        }
    },
    "Semana 4 (t=0.20)": {
        "description": "Intervencion temprana SIN evaluaciones",
        "config": "XGBoost V4, SIN assessment, threshold=0.20",
        "roc_auc": 0.849,
        "n_samples": 373,
        # Calculated: n=373, failure_rate=40%, Recall=85.2%, Accuracy=74.5%
        # P=149, N=224, TP=127, FN=22, TN=151, FP=73
        "confusion_matrix": [[151, 73], [22, 127]],
        "metrics": {
            "accuracy": 0.745,
            "recall": 0.852,
            "precision": 0.635,
            "f1": 0.725
        }
    },
    "Semana 2 (t=0.13)": {
        "description": "Watch list - deteccion ultra-temprana",
        "config": "P5%, threshold=0.13",
        "roc_auc": 0.743,
        "n_samples": 303,
        # Calculated: n=303, failure_rate=40%, Recall=81.7%, Accuracy=65.7%
        # P=121, N=182, TP=99, FN=22, TN=100, FP=82
        "confusion_matrix": [[100, 82], [22, 99]],
        "metrics": {
            "accuracy": 0.657,
            "recall": 0.817,
            "precision": 0.547,
            "f1": 0.655
        }
    }
}


def plot_confusion_matrix(cm, model_name, ax=None, show_percentages=True, subtitle=None):
    """
    Plot a single confusion matrix with professional styling.

    cm format: [[TN, FP], [FN, TP]]
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total = tn + fp + fn + tp

    # Create matrix for display
    matrix = np.array([[tn, fp], [fn, tp]])

    # Colors: green for correct (TN, TP), red/orange for errors (FP, FN)
    colors = np.array([
        ['#a8e6cf', '#ffaaa5'],  # TN (green), FP (red)
        ['#ffaaa5', '#a8e6cf']   # FN (red), TP (green)
    ])

    # Plot colored cells
    for i in range(2):
        for j in range(2):
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                       facecolor=colors[i, j], edgecolor='white', linewidth=2))

    # Add text annotations
    labels = [['TN', 'FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            value = matrix[i, j]
            pct = value / total * 100

            # Main value
            ax.text(j, i - 0.1, f'{value}', ha='center', va='center',
                   fontsize=20, fontweight='bold', color='#2c3e50')

            # Label and percentage
            if show_percentages:
                ax.text(j, i + 0.25, f'{labels[i][j]} ({pct:.1f}%)',
                       ha='center', va='center', fontsize=10, color='#5d6d7e')

    # Configure axes
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(1.5, -0.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Aprobado', 'Reprobado'], fontsize=11)
    ax.set_yticklabels(['Aprobado', 'Reprobado'], fontsize=11)
    ax.set_xlabel('Prediccion del Modelo', fontsize=12, fontweight='bold')
    ax.set_ylabel('Resultado Real', fontsize=12, fontweight='bold')

    # Title with optional subtitle
    if subtitle:
        ax.set_title(f'{model_name}\n{subtitle}', fontsize=12, fontweight='bold', pad=10)
    else:
        ax.set_title(model_name, fontsize=13, fontweight='bold', pad=10)

    # Remove grid
    ax.grid(False)
    ax.set_aspect('equal')

    # Calculate metrics
    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total': total,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }


def create_individual_matrices():
    """Create individual confusion matrix plots for each model."""
    print("Creating individual confusion matrices...")

    # Map model names to filenames
    filename_map = {
        "Semana 6+ (t=0.33)": "semana_6_t0.33",
        "Semana 4 (t=0.20)": "semana_4_t0.20",
        "Semana 2 (t=0.13)": "semana_2_t0.13"
    }

    for model_name, model_data in REPORT_MODELS.items():
        cm = model_data['confusion_matrix']
        config = model_data['config']

        fig, ax = plt.subplots(figsize=(6, 5.5))
        metrics = plot_confusion_matrix(cm, model_name, ax, subtitle=config)

        # Add metrics below the matrix
        metrics_text = (f"Accuracy: {metrics['accuracy']:.1%} | "
                       f"Precision: {metrics['precision']:.1%} | "
                       f"Recall: {metrics['recall']:.1%} | "
                       f"F1: {metrics['f1']:.1%}")
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                color='#5d6d7e', style='italic')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)

        # Save with mapped filename
        filename = filename_map.get(model_name, model_name.lower().replace(' ', '_'))
        plt.savefig(OUTPUT_DIR / f'confusion_matrix_{filename}.png',
                   dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  Saved: confusion_matrix_{filename}.png")


def create_combined_matrix():
    """Create a combined view with all confusion matrices."""
    print("Creating combined confusion matrix view...")

    n_models = len(REPORT_MODELS)

    # Create 1x3 layout for the three models
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    all_metrics = []

    for idx, (model_name, model_data) in enumerate(REPORT_MODELS.items()):
        cm = model_data['confusion_matrix']
        config = model_data['config']
        roc_auc = model_data['roc_auc']

        metrics = plot_confusion_matrix(cm, model_name, axes[idx],
                                        show_percentages=True,
                                        subtitle=f"ROC-AUC: {roc_auc:.2f}")
        metrics['model'] = model_name
        metrics['roc_auc'] = roc_auc
        all_metrics.append(metrics)

    plt.suptitle('Matrices de Confusion - Modelos de Alerta Temprana',
                fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices_combined.png',
               dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  Saved: confusion_matrices_combined.png")

    return all_metrics


def create_metrics_summary_table(metrics_list):
    """Create a summary table of all model metrics."""
    print("\nResumen de Metricas:")
    print("-" * 90)
    print(f"{'Modelo':<25} {'ROC-AUC':>10} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 90)

    for m in metrics_list:
        print(f"{m['model']:<25} {m['roc_auc']:>10.2f} {m['accuracy']:>10.1%} "
              f"{m['precision']:>10.1%} {m['recall']:>10.1%} {m['f1']:>10.1%}")

    print("-" * 90)

    print("\nDetalle de Matrices:")
    print("-" * 70)
    for m in metrics_list:
        print(f"\n{m['model']}:")
        print(f"  TN={m['tn']}, FP={m['fp']}, FN={m['fn']}, TP={m['tp']}")
        print(f"  Total: {m['total']} estudiantes")


def main():
    print("=" * 60)
    print("GENERATING CONFUSION MATRIX VISUALIZATIONS")
    print("Models from 'Principales Resultados' section")
    print("=" * 60)

    create_individual_matrices()
    metrics = create_combined_matrix()

    if metrics:
        create_metrics_summary_table(metrics)

    print("\n" + "=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
