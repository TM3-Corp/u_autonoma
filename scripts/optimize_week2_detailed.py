#!/usr/bin/env python3
"""
Detailed Week 2 Analysis with 5% Percentile

Uses the full feature pipeline and finds optimal threshold
for various recall targets.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"

# XGBoost parameters
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


def load_week2_features():
    """Load all features for week 2 cutoff."""
    feature_dir = BASE_DIR / "data/enriched_features/cutoff_week_2"

    if not feature_dir.exists():
        print(f"ERROR: Directory not found: {feature_dir}")
        return None

    feature_files = list(feature_dir.glob("*_features.parquet"))
    print(f"Found {len(feature_files)} feature files:")
    for f in feature_files:
        print(f"  - {f.name}")

    dfs = []
    for fpath in feature_files:
        df = pd.read_parquet(fpath)
        print(f"  {fpath.name}: {len(df)} rows, {len(df.columns)} cols")
        dfs.append(df)

    if not dfs:
        return None

    # Merge all feature files
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
        df_merged = df_merged.drop(columns=dup_cols)

    # Add z-norm features
    df_merged = calculate_znorm_features(df_merged)

    return df_merged


def calculate_znorm_features(df):
    """Calculate z-score normalized features per course."""
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


def load_enrollments():
    """Load enrollment data with target variable."""
    df = pd.read_csv(ENROLLMENTS_FILE)
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


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
        'threshold': float(threshold),
        'recall': float(recall),
        'precision': float(precision),
        'accuracy': float(accuracy),
        'specificity': float(specificity),
        'f1': float(f1),
        'f2': float(f2),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }


def find_all_thresholds(y_true, y_pred_proba):
    """Calculate metrics for all thresholds."""
    thresholds = np.arange(0.10, 0.61, 0.01)
    results = [calculate_metrics_at_threshold(y_true, y_pred_proba, t) for t in thresholds]
    return pd.DataFrame(results)


def select_features_by_importance(X, y, feature_cols, threshold=0.005):
    """Select features with importance above threshold."""
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_))
    selected = [f for f, imp in importances.items() if imp >= threshold]
    return selected


def remove_correlated_features(X, feature_cols, threshold=0.85):
    """Remove highly correlated features."""
    if len(feature_cols) <= 1:
        return feature_cols
    corr_matrix = X[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [f for f in feature_cols if f not in to_drop]
    return selected


def main():
    print("=" * 70)
    print("WEEK 2 DETAILED ANALYSIS (5% Percentile)")
    print("=" * 70)
    print()

    # Load features
    df_features = load_week2_features()
    if df_features is None:
        print("Failed to load features")
        return

    df_enroll = load_enrollments()

    # Merge
    df = df_features.merge(
        df_enroll[['user_id', 'course_id', 'failed']],
        on=['user_id', 'course_id'],
        how='inner'
    )
    df = df.dropna(subset=['failed'])

    print(f"\nTotal samples: {len(df)}")
    print(f"Failure rate: {df['failed'].mean()*100:.1f}%")

    # Prepare features
    exclude_cols = ['user_id', 'course_id', 'failed', 'final_score', 'enrollment_state']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df['failed'].values

    X = X.fillna(0).replace([np.inf, -np.inf], 0)

    # Keep only numeric
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]

    print(f"Initial features: {len(numeric_cols)}")

    # Apply feature selection (same as optimize_all_thresholds.py)
    print("\nApplying feature selection...")
    selected_by_importance = select_features_by_importance(X, y, numeric_cols, threshold=0.005)
    print(f"  After importance filter (≥0.005): {len(selected_by_importance)}")

    selected_final = remove_correlated_features(X, selected_by_importance, threshold=0.85)
    print(f"  After correlation filter (≤0.85): {len(selected_final)}")

    X = X[selected_final]
    numeric_cols = selected_final

    print(f"\nFinal features: {len(numeric_cols)}")

    # Train with CV
    print("\nTraining XGBoost with 5-fold CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBClassifier(**XGBOOST_PARAMS)

    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    roc_auc = roc_auc_score(y, y_pred_proba)

    print(f"\nROC-AUC: {roc_auc:.3f}")

    # Calculate all thresholds
    df_thresholds = find_all_thresholds(y, y_pred_proba)

    # Print detailed table
    print("\n" + "=" * 90)
    print("THRESHOLD ANALYSIS - Week 2 (5% Percentile)")
    print("=" * 90)
    print(f"\n{'Threshold':>10} {'Accuracy':>10} {'Recall':>10} {'Precision':>10} {'F1':>10} {'F2':>10}")
    print("-" * 70)

    for _, row in df_thresholds.iterrows():
        print(f"{row['threshold']:>10.2f} {row['accuracy']*100:>9.1f}% {row['recall']*100:>9.1f}% "
              f"{row['precision']*100:>9.1f}% {row['f1']:>10.3f} {row['f2']:>10.3f}")

    # Find optimal thresholds for different targets
    print("\n" + "=" * 70)
    print("OPTIMAL THRESHOLDS BY TARGET")
    print("=" * 70)

    # Max Accuracy
    idx_acc = df_thresholds['accuracy'].idxmax()
    best_acc = df_thresholds.loc[idx_acc]
    print(f"\nMax Accuracy:")
    print(f"  Threshold: {best_acc['threshold']:.2f}")
    print(f"  Accuracy: {best_acc['accuracy']*100:.1f}%")
    print(f"  Recall: {best_acc['recall']*100:.1f}%")

    # Max F2
    idx_f2 = df_thresholds['f2'].idxmax()
    best_f2 = df_thresholds.loc[idx_f2]
    print(f"\nMax F2 (prioritizes recall):")
    print(f"  Threshold: {best_f2['threshold']:.2f}")
    print(f"  Accuracy: {best_f2['accuracy']*100:.1f}%")
    print(f"  Recall: {best_f2['recall']*100:.1f}%")

    # Recall >= 70% with max accuracy
    recall_70 = df_thresholds[df_thresholds['recall'] >= 0.70]
    if len(recall_70) > 0:
        best_r70 = recall_70.loc[recall_70['accuracy'].idxmax()]
        print(f"\nRecall ≥ 70% with best Accuracy:")
        print(f"  Threshold: {best_r70['threshold']:.2f}")
        print(f"  Accuracy: {best_r70['accuracy']*100:.1f}%")
        print(f"  Recall: {best_r70['recall']*100:.1f}%")

    # Recall >= 75% with max accuracy
    recall_75 = df_thresholds[df_thresholds['recall'] >= 0.75]
    if len(recall_75) > 0:
        best_r75 = recall_75.loc[recall_75['accuracy'].idxmax()]
        print(f"\nRecall ≥ 75% with best Accuracy:")
        print(f"  Threshold: {best_r75['threshold']:.2f}")
        print(f"  Accuracy: {best_r75['accuracy']*100:.1f}%")
        print(f"  Recall: {best_r75['recall']*100:.1f}%")

    # Recall >= 80% with max accuracy
    recall_80 = df_thresholds[df_thresholds['recall'] >= 0.80]
    if len(recall_80) > 0:
        best_r80 = recall_80.loc[recall_80['accuracy'].idxmax()]
        print(f"\nRecall ≥ 80% with best Accuracy:")
        print(f"  Threshold: {best_r80['threshold']:.2f}")
        print(f"  Accuracy: {best_r80['accuracy']*100:.1f}%")
        print(f"  Recall: {best_r80['recall']*100:.1f}%")

    # Get feature importances
    model.fit(X, y)
    importances = dict(zip(numeric_cols, [float(x) for x in model.feature_importances_]))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:10])

    print("\n" + "=" * 70)
    print("TOP 10 FEATURES")
    print("=" * 70)
    for i, (feat, imp) in enumerate(top_features.items(), 1):
        print(f"  {i:2d}. {feat}: {imp:.3f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY: WEEK 2 vs OTHER WEEKS (Best Config)")
    print("=" * 70)
    print(f"\n{'Week':<8} {'ROC-AUC':>10} {'Threshold':>10} {'Accuracy':>10} {'Recall':>10}")
    print("-" * 50)

    if len(recall_70) > 0:
        print(f"{'2':<8} {roc_auc:>10.3f} {best_r70['threshold']:>10.2f} {best_r70['accuracy']*100:>9.1f}% {best_r70['recall']*100:>9.1f}%")
    print(f"{'4':<8} {'0.758':>10} {'0.23':>10} {'71.4%':>10} {'76.8%':>10}")
    print(f"{'6':<8} {'0.823':>10} {'0.35':>10} {'76.2%':>10} {'70.3%':>10}")
    print(f"{'Full':<8} {'0.902':>10} {'0.33':>10} {'83.4%':>10} {'85.9%':>10}")


if __name__ == "__main__":
    main()
