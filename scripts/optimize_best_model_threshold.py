#!/usr/bin/env python3
"""
Threshold Optimization for Best Model (ROC-AUC 0.90)

Optimizes the threshold for the Time-Limited Full model that includes:
- Assessment features (quiz, assignment access rates)
- Z-score normalized features per course

This model achieves ROC-AUC 0.9033, the best we've found.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from xgboost import XGBClassifier
import json
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path('/home/paul/projects/uautonoma')
DATA_DIR = BASE_DIR / 'data'
ENRICHED_DIR = DATA_DIR / 'enriched_features'
OUTPUT_DIR = DATA_DIR / 'report/models/best_model_optimized'

# XGBoost parameters (same as train_time_limited_model.py)
XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'min_child_weight': 1,
    'n_estimators': 100,
    'subsample': 0.8,
    'eval_metric': 'logloss',
    'verbosity': 0,
    'random_state': 42,
}

# Feature selection parameters
FEATURE_IMPORTANCE_THRESHOLD = 0.005
CORRELATION_THRESHOLD = 0.85


def load_features():
    """Load all features for full cutoff with z-norm (same as experiment 8)."""
    feature_files = [
        'session_features.parquet',
        'category_features.parquet',
        'proactivity_features.parquet',
        'pca_features.parquet',
        'weekly_features.parquet',
        'ngram_features.parquet',
        'graph_features.parquet',
        'time_features.parquet',
        'normalized_features.parquet',  # Z-norm features
    ]

    dfs = []
    for fname in feature_files:
        fpath = ENRICHED_DIR / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            # For normalized_features, only keep the _znorm columns
            if fname == 'normalized_features.parquet':
                znorm_cols = [c for c in df.columns if c.endswith('_znorm')]
                df = df[['user_id', 'course_id'] + znorm_cols]
                print(f"  Loaded {len(znorm_cols)} z-normalized features")
            dfs.append(df)

    # Merge on user_id, course_id
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
        df_merged = df_merged.drop(columns=dup_cols)

    return df_merged


def load_enrollments():
    """Load enrollment data with target variable."""
    df = pd.read_csv(DATA_DIR / 'page_views/student_enrollments.csv')
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


def select_features_by_importance(X, y, feature_cols, threshold=0.005):
    """Select features with importance above threshold."""
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_))
    selected = [f for f, imp in importances.items() if imp >= threshold]
    return selected, importances


def remove_correlated_features(X, feature_cols, threshold=0.85):
    """Remove highly correlated features."""
    if len(feature_cols) <= 1:
        return feature_cols, []

    corr_matrix = X[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [f for f in feature_cols if f not in to_drop]
    return selected, to_drop


def prepare_data(df_features, df_enroll):
    """Prepare features and target (WITH assessment features)."""
    df = df_features.merge(
        df_enroll[['user_id', 'course_id', 'failed', 'final_score']],
        on=['user_id', 'course_id'],
        how='inner'
    )
    df = df.dropna(subset=['failed'])

    exclude_cols = ['user_id', 'course_id', 'failed', 'final_score', 'enrollment_state']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['failed'].values

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_cols = numeric_cols

    # Feature selection (same as original model)
    if len(feature_cols) > 10:
        initial_count = len(feature_cols)
        selected_by_importance, _ = select_features_by_importance(X, y, feature_cols)
        selected_final, _ = remove_correlated_features(X, selected_by_importance)
        print(f"  Feature selection: {initial_count} -> {len(selected_final)} features")
        feature_cols = selected_final
        X = X[feature_cols]

    return X, y, feature_cols


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

    # NEW METRICS
    # F3 Score (more recall-weighted than F2)
    f3 = 10 * precision * recall / (9 * precision + recall) if (9 * precision + recall) > 0 else 0

    # Youden's J Statistic (Informedness) - standard clinical threshold selection
    youden_j = recall + specificity - 1

    # Matthews Correlation Coefficient - best single metric for imbalanced classes
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / denominator if denominator > 0 else 0

    # Geometric Mean - balances sensitivity and specificity
    g_mean = np.sqrt(recall * specificity)

    # Balanced Accuracy
    balanced_accuracy = (recall + specificity) / 2

    return {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'specificity': specificity,
        'f1': f1,
        'f2': f2,
        'f3': f3,
        'youden_j': youden_j,
        'mcc': mcc,
        'g_mean': g_mean,
        'balanced_accuracy': balanced_accuracy,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def calculate_cost(fn, fp, cost_fn=3.0, cost_fp=1.0):
    """
    Calculate weighted cost of misclassification.

    Default: FN costs 3x more than FP (missing at-risk student is worse than false alarm)
    """
    return fn * cost_fn + fp * cost_fp


def find_cost_optimal_threshold(df_results, cost_fn=3.0, cost_fp=1.0):
    """Find threshold that minimizes weighted cost."""
    df_copy = df_results.copy()
    df_copy['cost'] = df_copy.apply(
        lambda row: calculate_cost(row['fn'], row['fp'], cost_fn, cost_fp), axis=1
    )
    idx_min_cost = df_copy['cost'].idxmin()
    result = df_copy.loc[idx_min_cost].copy()
    return result


def main():
    print('=' * 70)
    print('THRESHOLD OPTIMIZATION - BEST MODEL (ROC-AUC 0.90)')
    print('Model: Time-Limited Full + Assessment Features + Z-Norm')
    print('=' * 70)
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print('Loading data...')
    df_features = load_features()
    df_enroll = load_enrollments()

    # Prepare data (WITH assessment features)
    X, y, feature_cols = prepare_data(df_features, df_enroll)

    print(f'\n  Samples: {len(X)}')
    print(f'  Features: {len(feature_cols)}')
    print(f'  Z-Norm features: {len([c for c in feature_cols if c.endswith("_znorm")])}')
    print(f'  Failure rate: {y.mean()*100:.1f}%')
    print()

    # Train model with CV predictions
    print('Training XGBoost with 5-fold CV...')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)

    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    # ROC-AUC
    roc_auc = roc_auc_score(y, y_pred_proba)
    print(f'\n  ROC-AUC: {roc_auc:.4f} (threshold-independent)')
    print()

    # Calculate metrics at all thresholds
    print('Analyzing thresholds from 0.10 to 0.70...')
    thresholds = np.arange(0.10, 0.71, 0.01)
    results = []

    for t in thresholds:
        metrics = calculate_metrics_at_threshold(y, y_pred_proba, t)
        results.append(metrics)

    df_results = pd.DataFrame(results)

    # Find optimal thresholds
    print()
    print('=' * 70)
    print('OPTIMAL THRESHOLDS BY CRITERION')
    print('=' * 70)

    # Baseline (0.50)
    baseline = df_results[df_results['threshold'].round(2) == 0.50].iloc[0]

    # Max F2
    idx_f2 = df_results['f2'].idxmax()
    best_f2 = df_results.loc[idx_f2]

    # Max F1
    idx_f1 = df_results['f1'].idxmax()
    best_f1 = df_results.loc[idx_f1]

    # Max Accuracy
    idx_acc = df_results['accuracy'].idxmax()
    best_acc = df_results.loc[idx_acc]

    # Recall >= 80% with max accuracy
    recall_80 = df_results[df_results['recall'] >= 0.80]
    best_r80 = recall_80.loc[recall_80['accuracy'].idxmax()] if len(recall_80) > 0 else None

    # Recall >= 85% with max accuracy
    recall_85 = df_results[df_results['recall'] >= 0.85]
    best_r85 = recall_85.loc[recall_85['accuracy'].idxmax()] if len(recall_85) > 0 else None

    # NEW OPTIMIZATION CRITERIA

    # Max Youden's J (optimal clinical threshold)
    idx_youden = df_results['youden_j'].idxmax()
    best_youden = df_results.loc[idx_youden]

    # Max MCC (best single metric for imbalanced data)
    idx_mcc = df_results['mcc'].idxmax()
    best_mcc = df_results.loc[idx_mcc]

    # Max G-Mean (balanced sensitivity/specificity)
    idx_gmean = df_results['g_mean'].idxmax()
    best_gmean = df_results.loc[idx_gmean]

    # Max F3 (aggressive recall-weighting)
    idx_f3 = df_results['f3'].idxmax()
    best_f3 = df_results.loc[idx_f3]

    # Cost-optimal (FN = 3x FP) - moderate intervention
    best_cost_3x = find_cost_optimal_threshold(df_results, cost_fn=3.0, cost_fp=1.0)

    # Cost-optimal (FN = 5x FP) - aggressive intervention
    best_cost_5x = find_cost_optimal_threshold(df_results, cost_fn=5.0, cost_fp=1.0)

    # Recall >= 90% with max accuracy (aggressive early warning)
    recall_90 = df_results[df_results['recall'] >= 0.90]
    best_r90 = recall_90.loc[recall_90['accuracy'].idxmax()] if len(recall_90) > 0 else None

    # Print comparison table
    print()
    print(f'{"Criterion":<25} {"Thresh":<8} {"Recall":<10} {"Accuracy":<10} {"Precision":<10} {"F1":<8} {"F2":<8}')
    print('-' * 85)

    print(f'{"Baseline (t=0.50)":<25} {baseline["threshold"]:<8.2f} {baseline["recall"]*100:<10.1f} {baseline["accuracy"]*100:<10.1f} {baseline["precision"]*100:<10.1f} {baseline["f1"]:<8.3f} {baseline["f2"]:<8.3f}')
    print(f'{"Max Accuracy":<25} {best_acc["threshold"]:<8.2f} {best_acc["recall"]*100:<10.1f} {best_acc["accuracy"]*100:<10.1f} {best_acc["precision"]*100:<10.1f} {best_acc["f1"]:<8.3f} {best_acc["f2"]:<8.3f}')
    print(f'{"Max F1":<25} {best_f1["threshold"]:<8.2f} {best_f1["recall"]*100:<10.1f} {best_f1["accuracy"]*100:<10.1f} {best_f1["precision"]*100:<10.1f} {best_f1["f1"]:<8.3f} {best_f1["f2"]:<8.3f}')
    print(f'{"Max F2":<25} {best_f2["threshold"]:<8.2f} {best_f2["recall"]*100:<10.1f} {best_f2["accuracy"]*100:<10.1f} {best_f2["precision"]*100:<10.1f} {best_f2["f1"]:<8.3f} {best_f2["f2"]:<8.3f}')

    if best_r80 is not None:
        print(f'{"Recall≥80% + Max Acc":<25} {best_r80["threshold"]:<8.2f} {best_r80["recall"]*100:<10.1f} {best_r80["accuracy"]*100:<10.1f} {best_r80["precision"]*100:<10.1f} {best_r80["f1"]:<8.3f} {best_r80["f2"]:<8.3f}')

    if best_r85 is not None:
        print(f'{"Recall≥85% + Max Acc":<25} {best_r85["threshold"]:<8.2f} {best_r85["recall"]*100:<10.1f} {best_r85["accuracy"]*100:<10.1f} {best_r85["precision"]*100:<10.1f} {best_r85["f1"]:<8.3f} {best_r85["f2"]:<8.3f}')

    # NEW OPTIMIZATION CRITERIA OUTPUT
    print()
    print('=' * 70)
    print('NEW OPTIMIZATION CRITERIA (Advanced Metrics)')
    print('=' * 70)
    print()
    print(f'{"Criterion":<30} {"Thresh":<8} {"Recall":<10} {"Accuracy":<10} {"Metric Value":<15}')
    print('-' * 75)

    print(f'{"Max Youden J (clinical)":<30} {best_youden["threshold"]:<8.2f} {best_youden["recall"]*100:<10.1f} {best_youden["accuracy"]*100:<10.1f} J={best_youden["youden_j"]:.3f}')
    print(f'{"Max MCC (imbalanced)":<30} {best_mcc["threshold"]:<8.2f} {best_mcc["recall"]*100:<10.1f} {best_mcc["accuracy"]*100:<10.1f} MCC={best_mcc["mcc"]:.3f}')
    print(f'{"Max G-Mean (balanced)":<30} {best_gmean["threshold"]:<8.2f} {best_gmean["recall"]*100:<10.1f} {best_gmean["accuracy"]*100:<10.1f} G={best_gmean["g_mean"]:.3f}')
    print(f'{"Max F3 (aggressive recall)":<30} {best_f3["threshold"]:<8.2f} {best_f3["recall"]*100:<10.1f} {best_f3["accuracy"]*100:<10.1f} F3={best_f3["f3"]:.3f}')
    print(f'{"Cost-Optimal (FN=3x FP)":<30} {best_cost_3x["threshold"]:<8.2f} {best_cost_3x["recall"]*100:<10.1f} {best_cost_3x["accuracy"]*100:<10.1f} Cost={best_cost_3x["cost"]:.0f}')
    print(f'{"Cost-Optimal (FN=5x FP)":<30} {best_cost_5x["threshold"]:<8.2f} {best_cost_5x["recall"]*100:<10.1f} {best_cost_5x["accuracy"]*100:<10.1f} Cost={best_cost_5x["cost"]:.0f}')

    if best_r90 is not None:
        print(f'{"Recall≥90% + Max Acc":<30} {best_r90["threshold"]:<8.2f} {best_r90["recall"]*100:<10.1f} {best_r90["accuracy"]*100:<10.1f}')

    # PRACTICAL DEPLOYMENT RECOMMENDATIONS
    print()
    print('=' * 70)
    print('PRACTICAL DEPLOYMENT RECOMMENDATIONS')
    print('=' * 70)
    print()
    print('Scenario-based threshold recommendations for early warning system:')
    print()

    # Aggressive: Use Cost-Optimal 5x or Max F3
    aggressive_t = best_cost_5x['threshold']
    aggressive_row = best_cost_5x

    # Balanced: Use Youden's J (clinical standard)
    balanced_t = best_youden['threshold']
    balanced_row = best_youden

    # Conservative: Use MCC (high confidence)
    conservative_t = best_mcc['threshold']
    conservative_row = best_mcc

    print(f'1. AGGRESSIVE (t={aggressive_t:.2f}) - Maximize early detection')
    print(f'   Recall: {aggressive_row["recall"]*100:.1f}% | Accuracy: {aggressive_row["accuracy"]*100:.1f}% | Precision: {aggressive_row["precision"]*100:.1f}%')
    print(f'   Use when: Resources abundant, cost of missing student is very high')
    print(f'   Trade-off: More false alarms, but catches most at-risk students')
    print()

    print(f'2. BALANCED (t={balanced_t:.2f}) - Optimal statistical trade-off (Youden J)')
    print(f'   Recall: {balanced_row["recall"]*100:.1f}% | Accuracy: {balanced_row["accuracy"]*100:.1f}% | Precision: {balanced_row["precision"]*100:.1f}%')
    print(f'   Use when: Need balance between detection and precision')
    print(f'   Trade-off: Clinically standard approach, good overall performance')
    print()

    print(f'3. CONSERVATIVE (t={conservative_t:.2f}) - Minimize false alarms (Max MCC)')
    print(f'   Recall: {conservative_row["recall"]*100:.1f}% | Accuracy: {conservative_row["accuracy"]*100:.1f}% | Precision: {conservative_row["precision"]*100:.1f}%')
    print(f'   Use when: Limited intervention resources, need high confidence')
    print(f'   Trade-off: Fewer false alarms, may miss some at-risk students')

    # Detailed analysis at key thresholds
    print()
    print('=' * 70)
    print('DETAILED ANALYSIS AT KEY THRESHOLDS')
    print('=' * 70)

    key_thresholds = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15]

    print()
    print(f'{"Threshold":<12} {"Recall":<12} {"Accuracy":<12} {"Precision":<12} {"F2":<10} {"TP":<6} {"FP":<6} {"FN":<6} {"TN":<6}')
    print('-' * 100)

    for t in key_thresholds:
        matches = df_results[df_results['threshold'].round(2) == t]
        if len(matches) > 0:
            row = matches.iloc[0]
            print(f'{row["threshold"]:<12.2f} {row["recall"]*100:<12.1f} {row["accuracy"]*100:<12.1f} {row["precision"]*100:<12.1f} {row["f2"]:<10.3f} {int(row["tp"]):<6} {int(row["fp"]):<6} {int(row["fn"]):<6} {int(row["tn"]):<6}')

    # Trade-off analysis
    print()
    print('=' * 70)
    print('TRADE-OFF ANALYSIS (vs Default Threshold 0.50)')
    print('=' * 70)
    print()
    print(f'Baseline (t=0.50): Recall={baseline["recall"]*100:.1f}%, Accuracy={baseline["accuracy"]*100:.1f}%')
    print()

    total_failed = int(baseline['tp'] + baseline['fn'])

    for label, row in [('Max F2', best_f2), ('Recall≥80%', best_r80), ('Recall≥85%', best_r85)]:
        if row is not None:
            recall_gain = (row['recall'] - baseline['recall']) * 100
            acc_loss = (baseline['accuracy'] - row['accuracy']) * 100

            extra_caught = int(row['tp'] - baseline['tp'])
            extra_false_alarms = int(row['fp'] - baseline['fp'])

            print(f'{label} (t={row["threshold"]:.2f}):')
            print(f'  Recall:   {baseline["recall"]*100:.1f}% → {row["recall"]*100:.1f}% (+{recall_gain:.1f}pp)')
            print(f'  Accuracy: {baseline["accuracy"]*100:.1f}% → {row["accuracy"]*100:.1f}% ({-acc_loss:+.1f}pp)')
            print(f'  Students: +{extra_caught} at-risk caught (of {total_failed}), +{extra_false_alarms} extra false alarms')
            print()

    # Final recommendation
    print('=' * 70)
    print('RECOMENDACION FINAL')
    print('=' * 70)
    print()
    print(f'ROC-AUC del Modelo: {roc_auc:.4f} (EXCELENTE)')
    print()

    if best_r85 is not None:
        print(f'OPCION RECOMENDADA - Recall ≥85% (umbral = {best_r85["threshold"]:.2f}):')
        print(f'  Recall: {best_r85["recall"]*100:.1f}%')
        print(f'  Accuracy: {best_r85["accuracy"]*100:.1f}%')
        print(f'  Precision: {best_r85["precision"]*100:.1f}%')
        print(f'  → Detecta {int(best_r85["tp"])} de {total_failed} estudiantes en riesgo')

    # Save results
    results_dict = {
        'model_info': {
            'name': 'Time-Limited Full + Assessment + Z-Norm',
            'roc_auc': float(roc_auc),
            'n_samples': int(len(X)),
            'n_features': len(feature_cols),
            'failure_rate': float(y.mean())
        },
        'baseline_threshold_0.50': {
            'recall': float(baseline['recall']),
            'accuracy': float(baseline['accuracy']),
            'precision': float(baseline['precision']),
            'f1': float(baseline['f1']),
            'f2': float(baseline['f2']),
            'confusion_matrix': {'tp': int(baseline['tp']), 'fp': int(baseline['fp']),
                                 'fn': int(baseline['fn']), 'tn': int(baseline['tn'])}
        },
        'max_f2': {
            'threshold': float(best_f2['threshold']),
            'recall': float(best_f2['recall']),
            'accuracy': float(best_f2['accuracy']),
            'precision': float(best_f2['precision']),
            'f1': float(best_f2['f1']),
            'f2': float(best_f2['f2']),
            'confusion_matrix': {'tp': int(best_f2['tp']), 'fp': int(best_f2['fp']),
                                 'fn': int(best_f2['fn']), 'tn': int(best_f2['tn'])}
        }
    }

    if best_r80 is not None:
        results_dict['recall_80_percent'] = {
            'threshold': float(best_r80['threshold']),
            'recall': float(best_r80['recall']),
            'accuracy': float(best_r80['accuracy']),
            'precision': float(best_r80['precision']),
            'confusion_matrix': {'tp': int(best_r80['tp']), 'fp': int(best_r80['fp']),
                                 'fn': int(best_r80['fn']), 'tn': int(best_r80['tn'])}
        }

    if best_r85 is not None:
        results_dict['recall_85_percent'] = {
            'threshold': float(best_r85['threshold']),
            'recall': float(best_r85['recall']),
            'accuracy': float(best_r85['accuracy']),
            'precision': float(best_r85['precision']),
            'confusion_matrix': {'tp': int(best_r85['tp']), 'fp': int(best_r85['fp']),
                                 'fn': int(best_r85['fn']), 'tn': int(best_r85['tn'])}
        }

    # NEW OPTIMIZATION CRITERIA IN JSON OUTPUT
    results_dict['max_youden_j'] = {
        'threshold': float(best_youden['threshold']),
        'youden_j': float(best_youden['youden_j']),
        'recall': float(best_youden['recall']),
        'accuracy': float(best_youden['accuracy']),
        'precision': float(best_youden['precision']),
        'specificity': float(best_youden['specificity']),
        'confusion_matrix': {'tp': int(best_youden['tp']), 'fp': int(best_youden['fp']),
                             'fn': int(best_youden['fn']), 'tn': int(best_youden['tn'])}
    }

    results_dict['max_mcc'] = {
        'threshold': float(best_mcc['threshold']),
        'mcc': float(best_mcc['mcc']),
        'recall': float(best_mcc['recall']),
        'accuracy': float(best_mcc['accuracy']),
        'precision': float(best_mcc['precision']),
        'specificity': float(best_mcc['specificity']),
        'confusion_matrix': {'tp': int(best_mcc['tp']), 'fp': int(best_mcc['fp']),
                             'fn': int(best_mcc['fn']), 'tn': int(best_mcc['tn'])}
    }

    results_dict['max_g_mean'] = {
        'threshold': float(best_gmean['threshold']),
        'g_mean': float(best_gmean['g_mean']),
        'recall': float(best_gmean['recall']),
        'accuracy': float(best_gmean['accuracy']),
        'precision': float(best_gmean['precision']),
        'specificity': float(best_gmean['specificity']),
        'confusion_matrix': {'tp': int(best_gmean['tp']), 'fp': int(best_gmean['fp']),
                             'fn': int(best_gmean['fn']), 'tn': int(best_gmean['tn'])}
    }

    results_dict['max_f3'] = {
        'threshold': float(best_f3['threshold']),
        'f3': float(best_f3['f3']),
        'recall': float(best_f3['recall']),
        'accuracy': float(best_f3['accuracy']),
        'precision': float(best_f3['precision']),
        'confusion_matrix': {'tp': int(best_f3['tp']), 'fp': int(best_f3['fp']),
                             'fn': int(best_f3['fn']), 'tn': int(best_f3['tn'])}
    }

    results_dict['cost_optimal_3x'] = {
        'threshold': float(best_cost_3x['threshold']),
        'cost': float(best_cost_3x['cost']),
        'cost_fn': 3.0,
        'cost_fp': 1.0,
        'recall': float(best_cost_3x['recall']),
        'accuracy': float(best_cost_3x['accuracy']),
        'precision': float(best_cost_3x['precision']),
        'confusion_matrix': {'tp': int(best_cost_3x['tp']), 'fp': int(best_cost_3x['fp']),
                             'fn': int(best_cost_3x['fn']), 'tn': int(best_cost_3x['tn'])}
    }

    results_dict['cost_optimal_5x'] = {
        'threshold': float(best_cost_5x['threshold']),
        'cost': float(best_cost_5x['cost']),
        'cost_fn': 5.0,
        'cost_fp': 1.0,
        'recall': float(best_cost_5x['recall']),
        'accuracy': float(best_cost_5x['accuracy']),
        'precision': float(best_cost_5x['precision']),
        'confusion_matrix': {'tp': int(best_cost_5x['tp']), 'fp': int(best_cost_5x['fp']),
                             'fn': int(best_cost_5x['fn']), 'tn': int(best_cost_5x['tn'])}
    }

    if best_r90 is not None:
        results_dict['recall_90_percent'] = {
            'threshold': float(best_r90['threshold']),
            'recall': float(best_r90['recall']),
            'accuracy': float(best_r90['accuracy']),
            'precision': float(best_r90['precision']),
            'confusion_matrix': {'tp': int(best_r90['tp']), 'fp': int(best_r90['fp']),
                                 'fn': int(best_r90['fn']), 'tn': int(best_r90['tn'])}
        }

    # Tiered recommendations for deployment
    results_dict['recommendations'] = {
        'aggressive': {
            'threshold': float(aggressive_t),
            'scenario': 'Maximize early detection',
            'description': 'Use when resources abundant and cost of missing student is very high'
        },
        'balanced': {
            'threshold': float(balanced_t),
            'scenario': 'Optimal statistical trade-off (Youden J)',
            'description': 'Clinically standard approach, good overall performance'
        },
        'conservative': {
            'threshold': float(conservative_t),
            'scenario': 'Minimize false alarms (Max MCC)',
            'description': 'Use when limited intervention resources, need high confidence'
        }
    }

    # Save all threshold results for plotting
    results_dict['all_thresholds'] = df_results.to_dict('records')

    output_file = OUTPUT_DIR / 'threshold_optimization_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print()
    print(f'Results saved to: {output_file}')


if __name__ == '__main__':
    main()
