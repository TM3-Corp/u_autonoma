#!/usr/bin/env python3
"""
Threshold Optimization Analysis for Early Warning Model

Finds the optimal threshold balancing:
- F2 score (prioritizes recall)
- Accuracy (overall correctness)
- Recall targets (75%, 80%)

Key insight: ROC-AUC is threshold-independent (model property)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/home/paul/projects/uautonoma/data')
ENRICHED_DIR = DATA_DIR / 'enriched_features'
OUTPUT_DIR = DATA_DIR / 'report/visualizations'


def load_data():
    """Load and merge all feature files."""
    session_df = pd.read_parquet(ENRICHED_DIR / 'session_features.parquet')
    category_df = pd.read_parquet(ENRICHED_DIR / 'category_features.parquet')
    proact_df = pd.read_parquet(ENRICHED_DIR / 'proactivity_features.parquet')
    pca_df = pd.read_parquet(ENRICHED_DIR / 'pca_features.parquet')

    merged = session_df.merge(category_df, on=['user_id', 'course_id'], how='outer')
    merged = merged.merge(proact_df, on=['user_id', 'course_id'], how='left')
    merged = merged.merge(pca_df, on=['user_id', 'course_id'], how='left')

    # Optional features
    for feat_file in ['ngram_features.parquet', 'graph_features.parquet', 'time_features.parquet']:
        feat_path = ENRICHED_DIR / feat_file
        if feat_path.exists():
            feat_df = pd.read_parquet(feat_path)
            feat_cols = [c for c in feat_df.columns if c not in ['user_id', 'course_id']]
            merged = merged.merge(feat_df[['user_id', 'course_id'] + feat_cols],
                                 on=['user_id', 'course_id'], how='left')

    # Enrollments with grades
    enrollments = pd.read_csv(DATA_DIR / 'page_views/student_enrollments.csv')
    enrollments['failed'] = enrollments['final_score'] < 57
    merged = merged.merge(
        enrollments[['user_id', 'course_id', 'failed']],
        on=['user_id', 'course_id'],
        how='inner'
    )

    return merged


def get_valid_features(df):
    """Get feature columns excluding grade-related and non-numeric."""
    exclude_patterns = ['quiz', 'quizzes', 'assi', 'assignment', 'grade', 'grad', 'score', 'submission']
    valid = []
    for col in df.columns:
        if col in ['user_id', 'course_id', 'failed']:
            continue
        col_lower = col.lower()
        if any(p in col_lower for p in exclude_patterns):
            continue
        # Skip non-numeric columns
        if df[col].dtype == 'object' or col == 'dominant_transition':
            continue
        valid.append(col)
    return valid


def calculate_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Calculate all metrics at a given threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

    balanced_acc = (recall + specificity) / 2

    return {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'specificity': specificity,
        'f1': f1,
        'f2': f2,
        'balanced_accuracy': balanced_acc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    print('=' * 70)
    print('THRESHOLD OPTIMIZATION ANALYSIS')
    print('Finding the Sweet Spot: F2, Recall, and Accuracy')
    print('=' * 70)
    print()

    # Load data
    print('Loading data...')
    df = load_data()
    feature_cols = get_valid_features(df)

    X = df[feature_cols].copy()
    y = df['failed'].astype(int)

    print(f'  Samples: {len(df)}')
    print(f'  Features: {len(feature_cols)}')
    print(f'  Class distribution: {(y==1).sum()} failed ({(y==1).mean()*100:.1f}%), {(y==0).sum()} passed')
    print()

    # Preprocess
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    X_imputed = imputer.fit_transform(X)
    X_scaled = scaler.fit_transform(X_imputed)

    # Train model and get cross-validated predictions
    print('Training XGBoost with 5-fold CV predictions...')
    n_neg, n_pos = (y == 0).sum(), (y == 1).sum()

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=n_neg / n_pos
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

    # ROC-AUC (threshold-independent)
    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f'\n  ROC-AUC: {roc_auc:.4f} (threshold-independent)')
    print()

    # Calculate metrics at all thresholds
    print('Analyzing thresholds from 0.20 to 0.70...')
    thresholds = np.arange(0.20, 0.71, 0.01)
    results = []

    for t in thresholds:
        metrics = calculate_metrics_at_threshold(y, y_pred_proba, t)
        results.append(metrics)

    df_results = pd.DataFrame(results)

    # Find optimal thresholds for different criteria
    print()
    print('=' * 70)
    print('OPTIMAL THRESHOLDS BY CRITERION')
    print('=' * 70)

    # 1. Maximum F2
    idx_f2 = df_results['f2'].idxmax()
    best_f2 = df_results.loc[idx_f2]

    # 2. Maximum F1
    idx_f1 = df_results['f1'].idxmax()
    best_f1 = df_results.loc[idx_f1]

    # 3. Maximum Accuracy
    idx_acc = df_results['accuracy'].idxmax()
    best_acc = df_results.loc[idx_acc]

    # 4. Recall >= 75% with max accuracy
    recall_75 = df_results[df_results['recall'] >= 0.75]
    if len(recall_75) > 0:
        idx_r75 = recall_75['accuracy'].idxmax()
        best_r75 = df_results.loc[idx_r75]
    else:
        best_r75 = None

    # 5. Recall >= 80% with max accuracy
    recall_80 = df_results[df_results['recall'] >= 0.80]
    if len(recall_80) > 0:
        idx_r80 = recall_80['accuracy'].idxmax()
        best_r80 = df_results.loc[idx_r80]
    else:
        best_r80 = None

    # 6. Balanced: F2 * Accuracy (custom metric)
    df_results['f2_x_acc'] = df_results['f2'] * df_results['accuracy']
    idx_balanced = df_results['f2_x_acc'].idxmax()
    best_balanced = df_results.loc[idx_balanced]

    # Print comparison table
    print()
    print(f'{"Criterion":<25} {"Thresh":<8} {"Recall":<10} {"Accuracy":<10} {"Precision":<10} {"F1":<8} {"F2":<8}')
    print('-' * 85)

    print(f'{"Max Accuracy":<25} {best_acc["threshold"]:<8.3f} {best_acc["recall"]*100:<10.1f} {best_acc["accuracy"]*100:<10.1f} {best_acc["precision"]*100:<10.1f} {best_acc["f1"]:<8.3f} {best_acc["f2"]:<8.3f}')
    print(f'{"Max F1":<25} {best_f1["threshold"]:<8.3f} {best_f1["recall"]*100:<10.1f} {best_f1["accuracy"]*100:<10.1f} {best_f1["precision"]*100:<10.1f} {best_f1["f1"]:<8.3f} {best_f1["f2"]:<8.3f}')
    print(f'{"Max F2":<25} {best_f2["threshold"]:<8.3f} {best_f2["recall"]*100:<10.1f} {best_f2["accuracy"]*100:<10.1f} {best_f2["precision"]*100:<10.1f} {best_f2["f1"]:<8.3f} {best_f2["f2"]:<8.3f}')
    print(f'{"Max F2 × Accuracy":<25} {best_balanced["threshold"]:<8.3f} {best_balanced["recall"]*100:<10.1f} {best_balanced["accuracy"]*100:<10.1f} {best_balanced["precision"]*100:<10.1f} {best_balanced["f1"]:<8.3f} {best_balanced["f2"]:<8.3f}')

    if best_r75 is not None:
        print(f'{"Recall≥75% + Max Acc":<25} {best_r75["threshold"]:<8.3f} {best_r75["recall"]*100:<10.1f} {best_r75["accuracy"]*100:<10.1f} {best_r75["precision"]*100:<10.1f} {best_r75["f1"]:<8.3f} {best_r75["f2"]:<8.3f}')

    if best_r80 is not None:
        print(f'{"Recall≥80% + Max Acc":<25} {best_r80["threshold"]:<8.3f} {best_r80["recall"]*100:<10.1f} {best_r80["accuracy"]*100:<10.1f} {best_r80["precision"]*100:<10.1f} {best_r80["f1"]:<8.3f} {best_r80["f2"]:<8.3f}')

    # Detailed analysis at key thresholds
    print()
    print('=' * 70)
    print('DETAILED ANALYSIS AT KEY THRESHOLDS')
    print('=' * 70)

    key_thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25]

    print()
    print(f'{"Threshold":<12} {"Recall":<12} {"Accuracy":<12} {"Precision":<12} {"F2":<10} {"TP":<6} {"FP":<6} {"FN":<6} {"TN":<6}')
    print('-' * 100)

    for t in key_thresholds:
        row = df_results[df_results['threshold'].round(2) == t].iloc[0]
        print(f'{row["threshold"]:<12.2f} {row["recall"]*100:<12.1f} {row["accuracy"]*100:<12.1f} {row["precision"]*100:<12.1f} {row["f2"]:<10.3f} {int(row["tp"]):<6} {int(row["fp"]):<6} {int(row["fn"]):<6} {int(row["tn"]):<6}')

    # Calculate trade-off from baseline
    print()
    print('=' * 70)
    print('TRADE-OFF ANALYSIS (vs Default Threshold 0.50)')
    print('=' * 70)

    baseline = df_results[df_results['threshold'].round(2) == 0.50].iloc[0]

    print()
    print(f'Baseline (t=0.50): Recall={baseline["recall"]*100:.1f}%, Accuracy={baseline["accuracy"]*100:.1f}%')
    print()

    for label, row in [('Max F2', best_f2), ('Recall≥75%', best_r75), ('Recall≥80%', best_r80)]:
        if row is not None:
            recall_gain = (row['recall'] - baseline['recall']) * 100
            acc_loss = (baseline['accuracy'] - row['accuracy']) * 100

            # Calculate students affected
            total_failed = int(row['tp'] + row['fn'])
            extra_caught = int(row['tp'] - baseline['tp'])
            extra_false_alarms = int(row['fp'] - baseline['fp'])

            print(f'{label} (t={row["threshold"]:.2f}):')
            print(f'  Recall:   {baseline["recall"]*100:.1f}% → {row["recall"]*100:.1f}% (+{recall_gain:.1f}pp)')
            print(f'  Accuracy: {baseline["accuracy"]*100:.1f}% → {row["accuracy"]*100:.1f}% ({-acc_loss:+.1f}pp)')
            print(f'  Students: +{extra_caught} at-risk caught, +{extra_false_alarms} extra false alarms')
            print()

    # Generate visualization
    print('=' * 70)
    print('GENERATING VISUALIZATION')
    print('=' * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Recall vs Accuracy trade-off
    ax1 = axes[0, 0]
    ax1.plot(df_results['threshold'], df_results['recall']*100, 'b-', linewidth=2, label='Recall (Sensibilidad)')
    ax1.plot(df_results['threshold'], df_results['accuracy']*100, 'g-', linewidth=2, label='Exactitud')
    ax1.plot(df_results['threshold'], df_results['precision']*100, 'r--', linewidth=1.5, label='Precision')
    ax1.axhline(y=75, color='orange', linestyle=':', alpha=0.7, label='Objetivo 75%')
    ax1.axhline(y=80, color='purple', linestyle=':', alpha=0.7, label='Objetivo 80%')
    ax1.axvline(x=best_f2['threshold'], color='blue', linestyle='--', alpha=0.5, label=f'Max F2 (t={best_f2["threshold"]:.2f})')
    ax1.set_xlabel('Umbral de Decision', fontsize=11)
    ax1.set_ylabel('Porcentaje (%)', fontsize=11)
    ax1.set_title('Trade-off: Recall vs Exactitud vs Precision', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0.20, 0.70)
    ax1.set_ylim(50, 100)

    # Plot 2: F1 and F2 scores
    ax2 = axes[0, 1]
    ax2.plot(df_results['threshold'], df_results['f1'], 'b-', linewidth=2, label='F1 (balanceado)')
    ax2.plot(df_results['threshold'], df_results['f2'], 'r-', linewidth=2, label='F2 (prioriza recall)')
    ax2.axvline(x=best_f1['threshold'], color='blue', linestyle='--', alpha=0.5, label=f'Max F1 (t={best_f1["threshold"]:.2f})')
    ax2.axvline(x=best_f2['threshold'], color='red', linestyle='--', alpha=0.5, label=f'Max F2 (t={best_f2["threshold"]:.2f})')
    ax2.set_xlabel('Umbral de Decision', fontsize=11)
    ax2.set_ylabel('Score', fontsize=11)
    ax2.set_title('F1 vs F2 Score por Umbral', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.20, 0.70)

    # Plot 3: Confusion matrix components
    ax3 = axes[1, 0]
    ax3.plot(df_results['threshold'], df_results['tp'], 'g-', linewidth=2, label='Verdaderos Positivos (TP)')
    ax3.plot(df_results['threshold'], df_results['fp'], 'r-', linewidth=2, label='Falsos Positivos (FP)')
    ax3.plot(df_results['threshold'], df_results['fn'], 'orange', linewidth=2, label='Falsos Negativos (FN)')
    ax3.axvline(x=best_f2['threshold'], color='blue', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Umbral de Decision', fontsize=11)
    ax3.set_ylabel('Numero de Estudiantes', fontsize=11)
    ax3.set_title('Componentes de la Matriz de Confusion', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.20, 0.70)

    # Plot 4: Summary recommendation
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Create summary text
    summary_text = f"""
    RESUMEN DE OPTIMIZACION
    {'='*50}

    ROC-AUC del Modelo: {roc_auc:.3f} (independiente del umbral)

    UMBRALES RECOMENDADOS:

    1. Maximo F2 (t={best_f2['threshold']:.2f}):
       • Recall: {best_f2['recall']*100:.1f}%  |  Exactitud: {best_f2['accuracy']*100:.1f}%
       • Detecta {int(best_f2['tp'])} de {int(best_f2['tp']+best_f2['fn'])} estudiantes en riesgo

    2. Recall ≥75% (t={best_r75['threshold']:.2f}):
       • Recall: {best_r75['recall']*100:.1f}%  |  Exactitud: {best_r75['accuracy']*100:.1f}%
       • Detecta {int(best_r75['tp'])} de {int(best_r75['tp']+best_r75['fn'])} estudiantes en riesgo
    """

    if best_r80 is not None:
        summary_text += f"""
    3. Recall ≥80% (t={best_r80['threshold']:.2f}):
       • Recall: {best_r80['recall']*100:.1f}%  |  Exactitud: {best_r80['accuracy']*100:.1f}%
       • Detecta {int(best_r80['tp'])} de {int(best_r80['tp']+best_r80['fn'])} estudiantes en riesgo
    """

    summary_text += f"""

    INTERPRETACION PARA U. AUTONOMA:

    Con umbral t={best_f2['threshold']:.2f} (Max F2):
    • De cada 10 estudiantes que reprueban, detectamos ~{int(best_f2['recall']*10)}
    • De cada 10 alertas, ~{int(best_f2['precision']*10)} son correctas
    • Exactitud general: {best_f2['accuracy']*100:.1f}%
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'threshold_optimization_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f'  Saved: {OUTPUT_DIR}/threshold_optimization_analysis.png')

    # Final recommendation
    print()
    print('=' * 70)
    print('RECOMENDACION FINAL PARA U. AUTONOMA')
    print('=' * 70)
    print()
    print(f'ROC-AUC del Modelo: {roc_auc:.3f} (no cambia con el umbral)')
    print()
    print('Opciones estrategicas:')
    print()
    print(f'  OPCION A - Maximo F2 (umbral = {best_f2["threshold"]:.2f}):')
    print(f'    Recall: {best_f2["recall"]*100:.1f}% | Exactitud: {best_f2["accuracy"]*100:.1f}% | Precision: {best_f2["precision"]*100:.1f}%')
    print(f'    → Detecta {int(best_f2["tp"])}/{int(best_f2["tp"]+best_f2["fn"])} estudiantes en riesgo')
    print()

    if best_r75 is not None:
        print(f'  OPCION B - 3 de 4 en riesgo (umbral = {best_r75["threshold"]:.2f}):')
        print(f'    Recall: {best_r75["recall"]*100:.1f}% | Exactitud: {best_r75["accuracy"]*100:.1f}% | Precision: {best_r75["precision"]*100:.1f}%')
        print(f'    → Detecta {int(best_r75["tp"])}/{int(best_r75["tp"]+best_r75["fn"])} estudiantes en riesgo')
        print()

    if best_r80 is not None:
        print(f'  OPCION C - 4 de 5 en riesgo (umbral = {best_r80["threshold"]:.2f}):')
        print(f'    Recall: {best_r80["recall"]*100:.1f}% | Exactitud: {best_r80["accuracy"]*100:.1f}% | Precision: {best_r80["precision"]*100:.1f}%')
        print(f'    → Detecta {int(best_r80["tp"])}/{int(best_r80["tp"]+best_r80["fn"])} estudiantes en riesgo')
        print()

    # Save results to JSON
    import json
    optimization_results = {
        'roc_auc': roc_auc,
        'baseline_threshold_0.50': {
            'recall': baseline['recall'],
            'accuracy': baseline['accuracy'],
            'precision': baseline['precision'],
            'f1': baseline['f1'],
            'f2': baseline['f2']
        },
        'max_f2': {
            'threshold': best_f2['threshold'],
            'recall': best_f2['recall'],
            'accuracy': best_f2['accuracy'],
            'precision': best_f2['precision'],
            'f1': best_f2['f1'],
            'f2': best_f2['f2']
        },
        'recall_75_percent': {
            'threshold': best_r75['threshold'] if best_r75 is not None else None,
            'recall': best_r75['recall'] if best_r75 is not None else None,
            'accuracy': best_r75['accuracy'] if best_r75 is not None else None,
            'precision': best_r75['precision'] if best_r75 is not None else None,
        },
        'recall_80_percent': {
            'threshold': best_r80['threshold'] if best_r80 is not None else None,
            'recall': best_r80['recall'] if best_r80 is not None else None,
            'accuracy': best_r80['accuracy'] if best_r80 is not None else None,
            'precision': best_r80['precision'] if best_r80 is not None else None,
        }
    }

    with open(OUTPUT_DIR / 'threshold_optimization_results.json', 'w') as f:
        json.dump(optimization_results, f, indent=2)

    print(f'Results saved to: {OUTPUT_DIR}/threshold_optimization_results.json')


if __name__ == '__main__':
    main()
