#!/usr/bin/env python3
"""
Train early warning model for PUC data with LOCO validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, fbeta_score, confusion_matrix,
    roc_curve, classification_report
)
import xgboost as xgb

# Paths
DATA_FILE = Path('data/puc/enriched_features/normalized_features.parquet')
FEATURES_FILE = Path('data/puc/feature_selection/optimal_features.json')
MODEL_DIR = Path('data/puc/models/early_warning_baseline')
OUTPUT_FILE = Path('data/puc/report/early_warning_model_metrics.json')

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

print("="*70)
print("EARLY WARNING MODEL TRAINING - PUC DATA")
print("="*70)

# Load data
print("\nLoading data...")
df = pd.read_parquet(DATA_FILE)

with open(FEATURES_FILE) as f:
    optimal_features = json.load(f)['features']

print(f"Dataset: {len(df)} enrollments")
print(f"Features: {len(optimal_features)}")
print(f"Courses: {df['course_id'].nunique()}")

# Prepare data
X = df[optimal_features].fillna(0).values
y = df['failed'].astype(int).values
courses = df['course_id'].values

print(f"\nTarget distribution:")
print(f"  Pass: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
print(f"  Fail: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

# Model configuration
pos_weight = (y==0).sum() / (y==1).sum()
print(f"\nScale pos weight: {pos_weight:.1f}")

model = xgb.XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    subsample=0.8,
    min_child_weight=1,
    eval_metric='logloss',
    scale_pos_weight=pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

# ============================================================================
# 5-FOLD STRATIFIED CV
# ============================================================================
print("\n" + "="*70)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

scoring = {
    'roc_auc': 'roc_auc',
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

cv_results = cross_validate(
    model, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

print(f"\nCross-Validation Results:")
for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
    test_scores = cv_results[f'test_{metric}']
    print(f"  {metric:12s}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")

cv_metrics = {
    metric: {
        'mean': float(cv_results[f'test_{metric}'].mean()),
        'std': float(cv_results[f'test_{metric}'].std()),
        'scores': cv_results[f'test_{metric}'].tolist()
    }
    for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']
}

# ============================================================================
# LEAVE-ONE-COURSE-OUT (LOCO) VALIDATION
# ============================================================================
print("\n" + "="*70)
print("LEAVE-ONE-COURSE-OUT (LOCO) VALIDATION")
print("="*70)

unique_courses = np.unique(courses)
print(f"\nTraining on {len(unique_courses)} courses (LOCO)...")

loco_results = []

for test_course in unique_courses:
    # Split data
    train_mask = courses != test_course
    test_mask = courses == test_course

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Check if test set has both classes
    if len(np.unique(y_test)) < 2:
        print(f"  Course {test_course}: Skipped (insufficient class diversity)")
        continue

    # Train
    fold_model = xgb.XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=0.8,
        min_child_weight=1,
        eval_metric='logloss',
        scale_pos_weight=(y_train==0).sum() / ((y_train==1).sum() + 1),
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    fold_model.fit(X_train, y_train)

    # Predict
    y_pred_proba = fold_model.predict_proba(X_test)[:, 1]
    y_pred = fold_model.predict(X_test)

    # Metrics
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = np.nan

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    loco_results.append({
        'course_id': int(test_course),
        'n_test': len(y_test),
        'n_fail': int(y_test.sum()),
        'roc_auc': float(auc) if not np.isnan(auc) else None,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1)
    })

    print(f"  Course {test_course:5d}: n={len(y_test):3d}, fail={y_test.sum():2d}, AUC={auc:.3f}, Recall={rec:.3f}")

# Aggregate LOCO results
loco_df = pd.DataFrame(loco_results)
loco_metrics = {
    'roc_auc': {
        'mean': float(loco_df['roc_auc'].mean()),
        'std': float(loco_df['roc_auc'].std()),
        'min': float(loco_df['roc_auc'].min()),
        'max': float(loco_df['roc_auc'].max())
    },
    'accuracy': {
        'mean': float(loco_df['accuracy'].mean()),
        'std': float(loco_df['accuracy'].std())
    },
    'recall': {
        'mean': float(loco_df['recall'].mean()),
        'std': float(loco_df['recall'].std())
    },
    'f1': {
        'mean': float(loco_df['f1'].mean()),
        'std': float(loco_df['f1'].std())
    }
}

print(f"\n{'='*70}")
print("LOCO VALIDATION SUMMARY")
print(f"{'='*70}")
print(f"  ROC-AUC:  {loco_metrics['roc_auc']['mean']:.3f} ± {loco_metrics['roc_auc']['std']:.3f}")
print(f"  Recall:   {loco_metrics['recall']['mean']:.3f} ± {loco_metrics['recall']['std']:.3f}")
print(f"  F1:       {loco_metrics['f1']['mean']:.3f} ± {loco_metrics['f1']['std']:.3f}")

# ============================================================================
# TRAIN FINAL MODEL ON ALL DATA
# ============================================================================
print("\n" + "="*70)
print("TRAINING FINAL MODEL ON ALL DATA")
print("="*70)

model.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': optimal_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 10 Feature Importances:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# Save model
model_path = MODEL_DIR / 'xgb_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\n✓ Model saved to: {model_path}")

# Save feature list
features_path = MODEL_DIR / 'feature_names.json'
with open(features_path, 'w') as f:
    json.dump(optimal_features, f, indent=2)

# Save feature importance
importance_path = MODEL_DIR / 'feature_importance.csv'
feature_importance.to_csv(importance_path, index=False)

# ============================================================================
# SAVE METRICS
# ============================================================================
print("\nSaving metrics...")

metrics = {
    'dataset': {
        'n_enrollments': len(df),
        'n_courses': int(df['course_id'].nunique()),
        'n_features': len(optimal_features),
        'failure_rate': float(y.mean())
    },
    'model': {
        'type': 'XGBClassifier',
        'params': model.get_params()
    },
    'cv_5fold': cv_metrics,
    'loco_validation': {
        'summary': loco_metrics,
        'per_course': loco_results
    },
    'feature_importance': feature_importance.to_dict('records')
}

with open(OUTPUT_FILE, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"\n✓ Metrics saved to: {OUTPUT_FILE}")

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
print(f"5-Fold CV ROC-AUC:     {cv_metrics['roc_auc']['mean']:.3f} ± {cv_metrics['roc_auc']['std']:.3f}")
print(f"5-Fold CV Recall:      {cv_metrics['recall']['mean']:.3f} ± {cv_metrics['recall']['std']:.3f}")
print(f"LOCO ROC-AUC:          {loco_metrics['roc_auc']['mean']:.3f} ± {loco_metrics['roc_auc']['std']:.3f}")
print(f"LOCO Recall:           {loco_metrics['recall']['mean']:.3f} ± {loco_metrics['recall']['std']:.3f}")
print("="*70)
print("\n✓ Phase 4 complete: Model training successful")
