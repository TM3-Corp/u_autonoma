#!/usr/bin/env python3
"""
Optimize thresholds for ALL time-limited models.

This script trains each time-limited model (2, 4, 6, 8 weeks, full)
with and without assessment features, then finds the optimal threshold
for each to maximize F2 score (prioritizing recall over precision).

Output:
    data/analysis/all_thresholds_optimized.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/analysis/all_thresholds_optimized.json"

# Cutoffs to evaluate
CUTOFFS = [2, 4, 6, 8, 'full']

# Assessment-related feature patterns (to exclude when testing without)
EXCLUDE_PATTERNS = [
    'quiz', 'quizzes',
    'assi', 'assignment',
    'grade', 'grad',
    'score',
    'submission',
]

# XGBoost hyperparameters
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
FEATURE_SELECTION_ENABLED = True
FEATURE_IMPORTANCE_THRESHOLD = 0.005
CORRELATION_THRESHOLD = 0.85


def calculate_znorm_features(df):
    """Calculate z-score normalized features per course (on-the-fly)."""
    exclude_cols = ['user_id', 'course_id', 'enrollment_state', 'failed', 'final_score']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.endswith('_znorm')]

    znorm_data = []
    for course_id in df['course_id'].unique():
        course_df = df[df['course_id'] == course_id].copy()
        for col in feature_cols:
            col_data = course_df[col]
            mean_val = col_data.mean()
            std_val = col_data.std()
            if std_val > 0:
                course_df[f'{col}_znorm'] = (col_data - mean_val) / std_val
            else:
                course_df[f'{col}_znorm'] = 0
        znorm_data.append(course_df)

    return pd.concat(znorm_data, ignore_index=True)


def load_features(cutoff, include_znorm=True):
    """Load all features for a given cutoff."""
    if cutoff == 'full':
        feature_dir = BASE_DIR / "data/enriched_features"
        feature_files = [
            'session_features.parquet',
            'category_features.parquet',
            'proactivity_features.parquet',
            'pca_features.parquet',
            'weekly_features.parquet',
            'ngram_features.parquet',
            'graph_features.parquet',
            'time_features.parquet',
        ]
        if include_znorm:
            feature_files.append('normalized_features.parquet')
    else:
        feature_dir = BASE_DIR / f"data/enriched_features/cutoff_week_{cutoff}"
        feature_files = list(feature_dir.glob("*_features.parquet"))
        feature_files = [f.name for f in feature_files]

    dfs = []
    for fname in feature_files:
        fpath = feature_dir / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            if fname == 'normalized_features.parquet':
                znorm_cols = [c for c in df.columns if c.endswith('_znorm')]
                df = df[['user_id', 'course_id'] + znorm_cols]
            dfs.append(df)

    if not dfs:
        return None

    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
        df_merged = df_merged.drop(columns=dup_cols)

    if cutoff != 'full' and include_znorm:
        df_merged = calculate_znorm_features(df_merged)

    return df_merged


def load_enrollments():
    """Load enrollment data with target variable."""
    df = pd.read_csv(ENROLLMENTS_FILE)
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


def filter_assessment_features(columns):
    """Filter out assessment-related features."""
    filtered = []
    for col in columns:
        col_lower = col.lower()
        is_excluded = any(pattern in col_lower for pattern in EXCLUDE_PATTERNS)
        if not is_excluded:
            filtered.append(col)
    return filtered


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


def prepare_data(df_features, df_enroll, include_assessment=True):
    """Prepare features and target for training."""
    df = df_features.merge(df_enroll[['user_id', 'course_id', 'failed', 'final_score']],
                           on=['user_id', 'course_id'], how='inner')
    df = df.dropna(subset=['failed'])

    exclude_cols = ['user_id', 'course_id', 'failed', 'final_score', 'enrollment_state']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not include_assessment:
        feature_cols = filter_assessment_features(feature_cols)

    X = df[feature_cols].copy()
    y = df['failed'].values

    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_cols = numeric_cols

    if FEATURE_SELECTION_ENABLED and len(feature_cols) > 10:
        selected_by_importance, _ = select_features_by_importance(X, y, feature_cols)
        selected_final, _ = remove_correlated_features(X, selected_by_importance)
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

    return {
        'threshold': threshold,
        'recall': recall,
        'precision': precision,
        'accuracy': accuracy,
        'specificity': specificity,
        'f1': f1,
        'f2': f2,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def find_optimal_thresholds(y_true, y_pred_proba):
    """Find optimal thresholds by various criteria."""
    thresholds = np.arange(0.10, 0.71, 0.01)
    results = [calculate_metrics_at_threshold(y_true, y_pred_proba, t) for t in thresholds]
    df_results = pd.DataFrame(results)

    # Baseline (0.50)
    baseline_mask = df_results['threshold'].round(2) == 0.50
    baseline = df_results[baseline_mask].iloc[0].to_dict() if baseline_mask.any() else None

    # Max F2
    idx_f2 = df_results['f2'].idxmax()
    best_f2 = df_results.loc[idx_f2].to_dict()

    # Recall >= 80% with max accuracy
    recall_80 = df_results[df_results['recall'] >= 0.80]
    best_r80 = recall_80.loc[recall_80['accuracy'].idxmax()].to_dict() if len(recall_80) > 0 else None

    # Recall >= 85% with max accuracy
    recall_85 = df_results[df_results['recall'] >= 0.85]
    best_r85 = recall_85.loc[recall_85['accuracy'].idxmax()].to_dict() if len(recall_85) > 0 else None

    return {
        'baseline_0.50': baseline,
        'max_f2': best_f2,
        'recall_80': best_r80,
        'recall_85': best_r85
    }


def run_experiment(cutoff, include_assessment, df_enroll):
    """Run a single experiment with threshold optimization."""
    label = f"week_{cutoff}_{'with' if include_assessment else 'without'}_assessment"

    df_features = load_features(cutoff, include_znorm=True)
    if df_features is None:
        return None

    X, y, feature_cols = prepare_data(df_features, df_enroll, include_assessment)

    if len(X) < 50:
        return None

    # Train with CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]

    roc_auc = roc_auc_score(y, y_pred_proba)

    # Find optimal thresholds
    optimal = find_optimal_thresholds(y, y_pred_proba)

    # Get feature importances
    model.fit(X, y)
    importances = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    return {
        'label': label,
        'cutoff': cutoff,
        'include_assessment': include_assessment,
        'n_samples': int(len(X)),
        'n_features': len(feature_cols),
        'failure_rate': float(y.mean()),
        'roc_auc': float(roc_auc),
        'thresholds': optimal,
        'top_features': top_features
    }


def main():
    print("=" * 70)
    print("THRESHOLD OPTIMIZATION - ALL TIME-LIMITED MODELS")
    print("=" * 70)
    print()

    df_enroll = load_enrollments()
    print(f"Loaded {len(df_enroll)} enrollments")
    print(f"Overall failure rate: {df_enroll['failed'].mean()*100:.1f}%")
    print()

    results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': []
    }

    # Run all experiments
    for cutoff in CUTOFFS:
        for include_assessment in [True, False]:
            label = f"week_{cutoff}_{'with' if include_assessment else 'without'}_assessment"
            print(f"Running: {label}...")

            exp = run_experiment(cutoff, include_assessment, df_enroll)
            if exp:
                results['experiments'].append(exp)

                # Show key metrics
                baseline = exp['thresholds']['baseline_0.50']
                best_f2 = exp['thresholds']['max_f2']

                if baseline and best_f2:
                    print(f"  ROC-AUC: {exp['roc_auc']:.3f}")
                    print(f"  Baseline (t=0.50): Acc={baseline['accuracy']*100:.1f}%, Recall={baseline['recall']*100:.1f}%")
                    print(f"  Optimized (t={best_f2['threshold']:.2f}): Acc={best_f2['accuracy']*100:.1f}%, Recall={best_f2['recall']*100:.1f}%")
                    print()

    # Print summary table
    print()
    print("=" * 100)
    print("SUMMARY: THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 100)
    print()
    print(f"{'Model':<45} {'ROC-AUC':>8} {'T=0.50':>12} {'T=Opt':>12} {'Opt':>6} {'Recall':>8} {'Acc':>8}")
    print(f"{'':45} {'':>8} {'Acc/Rec':>12} {'Acc/Rec':>12} {'Thr':>6} {'Gain':>8} {'Gain':>8}")
    print("-" * 100)

    for exp in results['experiments']:
        baseline = exp['thresholds']['baseline_0.50']
        best_f2 = exp['thresholds']['max_f2']

        if baseline and best_f2:
            recall_gain = (best_f2['recall'] - baseline['recall']) * 100
            acc_gain = (best_f2['accuracy'] - baseline['accuracy']) * 100

            print(f"{exp['label']:<45} {exp['roc_auc']:>8.3f} "
                  f"{baseline['accuracy']*100:>5.1f}/{baseline['recall']*100:<5.1f} "
                  f"{best_f2['accuracy']*100:>5.1f}/{best_f2['recall']*100:<5.1f} "
                  f"{best_f2['threshold']:>6.2f} "
                  f"{recall_gain:>+7.1f} "
                  f"{acc_gain:>+7.1f}")

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)

    print()
    print(f"Results saved to: {OUTPUT_FILE}")

    # Key recommendations
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best model for each cutoff
    for cutoff in CUTOFFS:
        exps = [e for e in results['experiments'] if e['cutoff'] == cutoff]
        if exps:
            best = max(exps, key=lambda x: x['roc_auc'])
            best_f2 = best['thresholds']['max_f2']
            print(f"\nWeek {cutoff}:")
            print(f"  Best config: {'WITH' if best['include_assessment'] else 'WITHOUT'} assessment")
            print(f"  ROC-AUC: {best['roc_auc']:.3f}")
            print(f"  Optimal threshold: {best_f2['threshold']:.2f}")
            print(f"  With t={best_f2['threshold']:.2f}: Accuracy={best_f2['accuracy']*100:.1f}%, Recall={best_f2['recall']*100:.1f}%")


if __name__ == "__main__":
    main()
