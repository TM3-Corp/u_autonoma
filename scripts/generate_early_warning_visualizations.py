#!/usr/bin/env python3
"""
Generate visualizations for the Early Warning Model report.

Creates:
1. Feature importance bar chart (top 20 features)
2. Confusion matrix heatmap
3. Model comparison bar chart
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
METRICS_FILE = Path('/home/paul/projects/uautonoma/data/report/early_warning_model_metrics.json')
OUTPUT_DIR = Path('/home/paul/projects/uautonoma/data/report/visualizations')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Style configuration
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'


def load_metrics():
    """Load model metrics from JSON file."""
    with open(METRICS_FILE, 'r') as f:
        return json.load(f)


def create_feature_importance_chart(metrics, output_path):
    """Create horizontal bar chart of top 20 feature importances."""
    top_features = metrics['models']['XGBoost Optimizado']['top_features']

    # Get top 20 features
    features = list(top_features.keys())[:20]
    importances = [top_features[f] * 100 for f in features]  # Convert to percentage

    # Create human-readable labels
    label_map = {
        'modu_n_resources': 'N recursos (Modulos)',
        'pages_views_pct': '% vistas (Paginas)',
        'page_n_resources': 'N recursos (Paginas)',
        'content_vs_assessment_ratio': 'Ratio contenido/evaluacion',
        'modu_top50_rate': 'Tasa top 50% (Modulos)',
        'total_views': 'Total vistas',
        'files_pc1': 'PCA comp.1 (Archivos)',
        'session_regularity': 'Regularidad sesiones',
        'announcements_views': 'Vistas anuncios',
        'modu_hist_b4': 'Histograma bin4 (Modulos)',
        'home_views': 'Vistas inicio',
        'views_per_session': 'Vistas por sesion',
        'long_sessions_pct': '% sesiones largas',
        'pages_pc1': 'PCA comp.1 (Paginas)',
        'page_hist_b2': 'Histograma bin2 (Paginas)',
        'mods_var_explained': 'Varianza explicada (Modulos)',
        'last_active_week': 'Ultima semana activa',
        'discussions_views_pct': '% vistas (Discusiones)',
        'session_duration_median': 'Duracion mediana sesion',
        'disc_access_rate': 'Tasa acceso (Discusiones)',
    }

    labels = [label_map.get(f, f) for f in features]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Color gradient from dark blue to light blue
    colors = plt.cm.Blues(np.linspace(0.8, 0.4, len(features)))

    # Horizontal bar chart
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color=colors, edgecolor='darkblue', linewidth=0.5)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # Top features at top
    ax.set_xlabel('Importancia (%)')
    ax.set_title('Top 20 Features Predictivos - Modelo de Alerta Temprana\n(XGBoost Optimizado, ROC-AUC = 0.859)', pad=20)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, importances)):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=9)

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Saved: {output_path}')


def create_confusion_matrix_heatmap(metrics, output_path):
    """Create confusion matrix heatmap."""
    cm = metrics['models']['XGBoost Optimizado']['metrics']['confusion_matrix']

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Labels
    labels = np.array(cm)

    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Aprueba', 'Reprueba'],
                yticklabels=['Aprueba', 'Reprueba'],
                annot_kws={'size': 20, 'weight': 'bold'},
                ax=ax)

    ax.set_xlabel('Prediccion', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de Confusion - Modelo de Alerta Temprana\n(XGBoost Optimizado)', pad=20)

    # Add annotations with percentages
    total = np.sum(cm)
    annotations = [
        f'VN\n{cm[0][0]} ({cm[0][0]/total*100:.1f}%)',
        f'FP\n{cm[0][1]} ({cm[0][1]/total*100:.1f}%)',
        f'FN\n{cm[1][0]} ({cm[1][0]/total*100:.1f}%)',
        f'VP\n{cm[1][1]} ({cm[1][1]/total*100:.1f}%)'
    ]

    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Saved: {output_path}')


def create_model_comparison_chart(metrics, output_path):
    """Create bar chart comparing models."""
    models = {
        'Baseline\n(Actividad)': metrics['models']['Baseline (Actividad)']['metrics'],
        'Regresion\nLogistica': metrics['models']['Regresion Logistica']['metrics'],
        'Random\nForest': metrics['models']['Random Forest']['metrics'],
        'XGBoost\nOptimizado': metrics['models']['XGBoost Optimizado']['metrics']
    }

    model_names = list(models.keys())

    # Metrics to compare
    metric_names = ['roc_auc', 'accuracy', 'precision', 'recall']
    metric_labels = ['ROC-AUC', 'Exactitud', 'Precision', 'Sensibilidad']

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(model_names))
    width = 0.2
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [models[m][metric] for m in model_names]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=colors[i], alpha=0.8)

        # Add value labels on top of bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.2f}' if val < 1 else f'{val:.0f}',
                   ha='center', va='bottom', fontsize=8, rotation=0)

    ax.set_ylabel('Valor')
    ax.set_title('Comparacion de Modelos - Sistema de Alerta Temprana\n(Sin Features de Evaluaciones)', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)

    # Add horizontal line at baseline ROC-AUC
    ax.axhline(y=0.787, color='gray', linestyle='--', alpha=0.5, label='Baseline ROC-AUC')

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Saved: {output_path}')


def create_feature_categories_pie(metrics, output_path):
    """Create pie chart showing feature categories breakdown."""
    feature_list = metrics['dataset_info']['feature_list']

    # Categorize features
    categories = {
        'Sesiones': 0,
        'Categorias (vistas)': 0,
        'Proactividad (PCT)': 0,
        'PCA': 0,
        'Temporal': 0,
        'Otros': 0
    }

    for f in feature_list:
        f_lower = f.lower()
        if 'session' in f_lower:
            categories['Sesiones'] += 1
        elif '_views' in f_lower or 'ratio' in f_lower:
            categories['Categorias (vistas)'] += 1
        elif 'pct' in f_lower or 'hist_' in f_lower or 'top' in f_lower or 'access_rate' in f_lower or 'dct_' in f_lower or 'proactivity' in f_lower:
            categories['Proactividad (PCT)'] += 1
        elif '_pc' in f_lower or 'var_explained' in f_lower:
            categories['PCA'] += 1
        elif 'week' in f_lower or 'early' in f_lower or 'late' in f_lower or 'pattern' in f_lower:
            categories['Temporal'] += 1
        else:
            categories['Otros'] += 1

    # Create pie chart
    fig, ax = plt.subplots(figsize=(10, 8))

    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    explode = [0.05] * len(labels)

    wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.0f%%', shadow=False, startangle=90,
                                        textprops={'fontsize': 11})

    # Add legend with counts
    legend_labels = [f'{l} ({s})' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title='Categorias', loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title(f'Distribucion de {len(feature_list)} Features por Categoria\nModelo de Alerta Temprana', pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f'Saved: {output_path}')


def main():
    print('=' * 60)
    print('Generating Early Warning Model Visualizations')
    print('=' * 60)
    print()

    # Load metrics
    print('Loading metrics...')
    metrics = load_metrics()
    print(f'  Best model: {metrics["best_model"]}')
    print(f'  ROC-AUC: {metrics["models"]["XGBoost Optimizado"]["metrics"]["roc_auc"]:.3f}')
    print()

    # Create visualizations
    print('Creating visualizations...')

    # 1. Feature importance
    create_feature_importance_chart(
        metrics,
        OUTPUT_DIR / 'feature_importance_early_warning.png'
    )

    # 2. Confusion matrix
    create_confusion_matrix_heatmap(
        metrics,
        OUTPUT_DIR / 'confusion_matrix_early_warning.png'
    )

    # 3. Model comparison
    create_model_comparison_chart(
        metrics,
        OUTPUT_DIR / 'model_comparison_early_warning.png'
    )

    # 4. Feature categories pie chart
    create_feature_categories_pie(
        metrics,
        OUTPUT_DIR / 'feature_categories_early_warning.png'
    )

    print()
    print('Done!')
    print(f'Visualizations saved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
