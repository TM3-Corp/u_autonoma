#!/usr/bin/env python3
"""
Phase 4: Train multi-class XGBoost model with 4 grade classes.
Uses 5-fold stratified CV and LOCO validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, precision_recall_fscore_support, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import json
import pickle

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "puc"
FEATURES_FILE = DATA_DIR / "enriched_features" / "normalized_features_multiclass.parquet"
MODEL_DIR = DATA_DIR / "models" / "multiclass_baseline"
METRICS_FILE = DATA_DIR / "report" / "multiclass_model_metrics.json"
VIZ_DIR = DATA_DIR / "report" / "visualizations"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Grade class definitions
GRADE_CLASSES = {
    0: {'name': 'EXCELLENT', 'color': '#2ecc71'},
    1: {'name': 'GOOD', 'color': '#3498db'},
    2: {'name': 'MARGINAL', 'color': '#f39c12'},
    3: {'name': 'FAIL', 'color': '#e74c3c'}
}


def load_data():
    """Load features with multi-class labels."""
    print("Loading data...")
    df = pd.read_parquet(FEATURES_FILE)

    # Exclude metadata columns
    exclude_cols = {'student_id', 'course_id', 'grade', 'failed', 'grade_class', 'class_label'}
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    X = df[feature_cols].values
    y = df['grade_class'].values
    courses = df['course_id'].values

    print(f"Loaded {len(df)} students, {len(feature_cols)} features")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y, courses, feature_cols, df


def calculate_class_weights(y):
    """Calculate class weights inversely proportional to frequency."""
    class_counts = np.bincount(y)
    total = len(y)
    n_classes = len(class_counts)

    weights = {}
    for i in range(n_classes):
        weights[i] = total / (n_classes * class_counts[i])

    print("\nClass weights:")
    for i, w in weights.items():
        print(f"  Class {i} ({GRADE_CLASSES[i]['name']:9s}): {w:.3f} (n={class_counts[i]})")

    return weights


def train_model_cv(X, y, class_weights, feature_names):
    """Train model using 5-fold stratified cross-validation."""
    print("\n" + "="*60)
    print("5-Fold Stratified Cross-Validation")
    print("="*60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    cv_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}/5...")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Calculate sample weights
        sample_weights = np.array([class_weights[label] for label in y_train])

        # Train model
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')

        # Per-class metrics
        prec, rec, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None)

        print(f"  F1-Weighted: {f1_weighted:.3f}")
        print(f"  F1-Macro: {f1_macro:.3f}")
        print(f"  FAIL Recall: {rec[3]:.3f}")

        cv_metrics.append({
            'fold': fold,
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'fail_recall': float(rec[3]),
            'per_class_precision': prec.tolist(),
            'per_class_recall': rec.tolist(),
            'per_class_f1': f1.tolist()
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    # Compute overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # ROC-AUC (One-vs-Rest)
    y_bin = label_binarize(all_y_true, classes=[0, 1, 2, 3])
    roc_auc_ovr = roc_auc_score(y_bin, all_y_proba, average='macro', multi_class='ovr')

    print(f"\n{'='*60}")
    print("Overall CV Performance:")
    print(f"{'='*60}")
    print(f"  ROC-AUC (macro OvR): {roc_auc_ovr:.3f}")
    print(f"  F1-Weighted: {np.mean([m['f1_weighted'] for m in cv_metrics]):.3f} ± {np.std([m['f1_weighted'] for m in cv_metrics]):.3f}")
    print(f"  F1-Macro: {np.mean([m['f1_macro'] for m in cv_metrics]):.3f} ± {np.std([m['f1_macro'] for m in cv_metrics]):.3f}")
    print(f"  FAIL Recall: {np.mean([m['fail_recall'] for m in cv_metrics]):.3f} ± {np.std([m['fail_recall'] for m in cv_metrics]):.3f}")

    return {
        'cv_metrics': cv_metrics,
        'roc_auc_ovr': float(roc_auc_ovr),
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'all_y_proba': all_y_proba
    }


def train_model_loco(X, y, courses, class_weights, feature_names):
    """Train model using Leave-One-Course-Out validation."""
    print("\n" + "="*60)
    print("Leave-One-Course-Out Cross-Validation")
    print("="*60)

    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=courses)

    print(f"Training on {n_splits} course splits...")

    loco_metrics = []
    all_y_true = []
    all_y_pred = []
    all_y_proba = []

    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=courses), 1):
        test_course = courses[test_idx[0]]
        n_test = len(test_idx)

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip if test set is too small or has no variance
        if n_test < 5 or len(np.unique(y_test)) < 2:
            continue

        # Calculate sample weights
        sample_weights = np.array([class_weights[label] for label in y_train])

        # Train model
        model = XGBClassifier(
            objective='multi:softprob',
            num_class=4,
            learning_rate=0.1,
            max_depth=5,
            n_estimators=150,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            random_state=RANDOM_SEED,
            n_jobs=-1
        )

        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Metrics
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)

        # FAIL recall (if FAIL class present)
        fail_mask = y_test == 3
        fail_recall = (y_pred[fail_mask] == 3).mean() if fail_mask.any() else np.nan

        if fold % 5 == 0:
            print(f"  Fold {fold}/{n_splits}: Course {test_course}, n={n_test}, F1={f1_weighted:.3f}")

        loco_metrics.append({
            'fold': fold,
            'course_id': int(test_course),
            'n_test': int(n_test),
            'f1_weighted': float(f1_weighted),
            'f1_macro': float(f1_macro),
            'fail_recall': float(fail_recall) if not np.isnan(fail_recall) else None
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)

    # Compute overall metrics
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)

    # ROC-AUC (One-vs-Rest)
    y_bin = label_binarize(all_y_true, classes=[0, 1, 2, 3])
    roc_auc_ovr = roc_auc_score(y_bin, all_y_proba, average='macro', multi_class='ovr')

    print(f"\n{'='*60}")
    print("Overall LOCO Performance:")
    print(f"{'='*60}")
    print(f"  ROC-AUC (macro OvR): {roc_auc_ovr:.3f}")
    print(f"  F1-Weighted: {np.mean([m['f1_weighted'] for m in loco_metrics]):.3f} ± {np.std([m['f1_weighted'] for m in loco_metrics]):.3f}")

    # FAIL recall (excluding NaN)
    fail_recalls = [m['fail_recall'] for m in loco_metrics if m['fail_recall'] is not None]
    if fail_recalls:
        print(f"  FAIL Recall: {np.mean(fail_recalls):.3f} ± {np.std(fail_recalls):.3f}")

    return {
        'loco_metrics': loco_metrics,
        'roc_auc_ovr': float(roc_auc_ovr),
        'all_y_true': all_y_true,
        'all_y_pred': all_y_pred,
        'all_y_proba': all_y_proba
    }


def train_final_model(X, y, class_weights, feature_names):
    """Train final model on all data."""
    print("\nTraining final model on all data...")

    sample_weights = np.array([class_weights[label] for label in y])

    model = XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        learning_rate=0.1,
        max_depth=5,
        n_estimators=150,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='mlogloss',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    model.fit(X, y, sample_weight=sample_weights, verbose=False)

    # Feature importance
    importance = model.feature_importances_
    top_indices = np.argsort(importance)[::-1][:20]

    print("\nTop 20 Features:")
    for i, idx in enumerate(top_indices, 1):
        print(f"  {i:2d}. {feature_names[idx]:40s} {importance[idx]:.4f}")

    # Save model
    model_path = MODEL_DIR / "xgb_model_multiclass.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model: {model_path}")

    # Save feature names
    with open(MODEL_DIR / "feature_names.json", 'w') as f:
        json.dump(feature_names, f, indent=2)

    # Save class weights
    with open(MODEL_DIR / "class_weights.json", 'w') as f:
        json.dump(class_weights, f, indent=2)

    return model, importance


def plot_confusion_matrix(y_true, y_pred, title, output_name):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax, cbar=False)

    # Add annotations with counts and percentages
    for i in range(4):
        for j in range(4):
            text = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                   fontsize=11, fontweight='bold' if i == j else 'normal')

    class_labels = [GRADE_CLASSES[i]['name'] for i in range(4)]
    ax.set_xticklabels(class_labels, rotation=0)
    ax.set_yticklabels(class_labels, rotation=0)
    ax.set_xlabel('Predicted Class', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Class', fontweight='bold', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_name}")


def plot_roc_curves(y_true, y_proba, title, output_name):
    """Plot ROC curves for each class (One-vs-Rest)."""
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    fig, ax = plt.subplots(figsize=(10, 8))

    for i in range(4):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        class_name = GRADE_CLASSES[i]['name']
        color = GRADE_CLASSES[i]['color']

        ax.plot(fpr, tpr, color=color, lw=2, alpha=0.8,
               label=f'{class_name} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_name}")


def save_metrics(cv_results, loco_results):
    """Save all metrics to JSON."""
    metrics = {
        'cv': {
            'roc_auc_ovr': cv_results['roc_auc_ovr'],
            'fold_metrics': cv_results['cv_metrics'],
            'classification_report': classification_report(
                cv_results['all_y_true'],
                cv_results['all_y_pred'],
                target_names=[GRADE_CLASSES[i]['name'] for i in range(4)],
                output_dict=True
            )
        },
        'loco': {
            'roc_auc_ovr': loco_results['roc_auc_ovr'],
            'fold_metrics': loco_results['loco_metrics'],
            'classification_report': classification_report(
                loco_results['all_y_true'],
                loco_results['all_y_pred'],
                target_names=[GRADE_CLASSES[i]['name'] for i in range(4)],
                output_dict=True
            )
        }
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics: {METRICS_FILE}")


def main():
    print("="*60)
    print("Phase 4: Train Multi-Class XGBoost Model")
    print("="*60)

    # Load data
    X, y, courses, feature_names, df = load_data()

    # Calculate class weights
    class_weights = calculate_class_weights(y)

    # 5-Fold CV
    cv_results = train_model_cv(X, y, class_weights, feature_names)

    # LOCO CV
    loco_results = train_model_loco(X, y, courses, class_weights, feature_names)

    # Train final model
    final_model, feature_importance = train_final_model(X, y, class_weights, feature_names)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_confusion_matrix(
        cv_results['all_y_true'],
        cv_results['all_y_pred'],
        'Confusion Matrix - 5-Fold CV',
        'confusion_matrix_multiclass.png'
    )

    plot_roc_curves(
        cv_results['all_y_true'],
        cv_results['all_y_proba'],
        'ROC Curves (One-vs-Rest) - 5-Fold CV',
        'roc_curves_multiclass_ovr.png'
    )

    # Save metrics
    save_metrics(cv_results, loco_results)

    # Print classification report
    print("\n" + "="*60)
    print("Classification Report (5-Fold CV):")
    print("="*60)
    print(classification_report(
        cv_results['all_y_true'],
        cv_results['all_y_pred'],
        target_names=[GRADE_CLASSES[i]['name'] for i in range(4)]
    ))

    print("\n✓ Phase 4 complete!")


if __name__ == "__main__":
    main()
