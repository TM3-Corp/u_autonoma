#!/usr/bin/env python3
"""
Generate ROC curves using REAL model data from early_warning_model_metrics.json
Output format matches the original roc_curves.png style (English, simple)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
METRICS_FILE = BASE_DIR / "data/report/early_warning_model_metrics.json"
OUTPUT_FILE = BASE_DIR / "data/report/visualizations/roc_curves.png"

def main():
    # Load real model metrics
    print(f"Loading metrics from: {METRICS_FILE}")
    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)

    # Model name mapping (Spanish -> English for original format)
    name_mapping = {
        'XGBoost Optimizado': 'XGBoost',
        'Random Forest': 'Random Forest',
        'Regresion Logistica': 'Logistic Regression',
        'Stacking Ensemble': 'Stacking Ensemble',
        'Baseline (Actividad)': 'Baseline',
    }

    # Colors matching original plot style
    colors = {
        'XGBoost': '#2ca02c',              # Green
        'Random Forest': '#ff7f0e',        # Orange
        'Logistic Regression': '#1f77b4',  # Blue
        'Stacking Ensemble': '#9467bd',    # Purple
        'Baseline': '#7f7f7f',             # Gray
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each model's ROC curve using REAL fpr/tpr data
    models_plotted = []

    for model_name_es, model_data in data['models'].items():
        if 'fpr' not in model_data or 'tpr' not in model_data:
            continue

        model_name_en = name_mapping.get(model_name_es, model_name_es)

        # Skip baseline (it has fake data)
        if 'Baseline' in model_name_en:
            continue

        fpr = np.array(model_data['fpr'])
        tpr = np.array(model_data['tpr'])
        auc = model_data['metrics']['roc_auc']

        color = colors.get(model_name_en, '#333333')

        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{model_name_en} (AUC = {auc:.3f})')

        models_plotted.append((model_name_en, auc))
        print(f"  {model_name_en}: AUC = {auc:.3f}")

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random (AUC = 0.5)')

    # Formatting (matching original style)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Binary Classification Models', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)

    # Save
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nSaved ROC curves to: {OUTPUT_FILE}")

    # Summary
    print("\n=== Model Comparison (REAL DATA) ===")
    print(f"{'Model':<25} {'AUC':>8}")
    print("-" * 35)
    for model, auc in sorted(models_plotted, key=lambda x: -x[1]):
        print(f"{model:<25} {auc:>8.3f}")
    print("-" * 35)
    print(f"{'Baseline (original)':<25} {0.787:>8.3f}")

    best_auc = max(auc for _, auc in models_plotted)
    print(f"\nImprovement over baseline: +{(best_auc - 0.787) / 0.787 * 100:.1f}%")

if __name__ == "__main__":
    main()
