#!/usr/bin/env python3
"""
Quick SOTA model training to test impact of new features.

Trains XGBoost multi-class model with all SOTA features and compares to baseline.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
SOTA_FEATURES = Path('data/puc/enriched_features/all_features_sota.parquet')
BASELINE_METRICS = Path('data/puc/models/multiclass_baseline/metrics.json')
OUTPUT_DIR = Path('data/puc/models/multiclass_sota_quick')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("QUICK SOTA MODEL TRAINING - PUC Multi-Class")
print("="*70)

# Load data
print("\n1. Loading SOTA features...")
df = pd.read_parquet(SOTA_FEATURES)
print(f"   Total records: {len(df)}")
print(f"   Total columns: {len(df.columns)}")

# Prepare X and y
exclude_cols = ['student_id', 'course_id', 'grade_category', 'grade', 'failed', 'class_label',
                'grade_class']  # CRITICAL: Remove grade_class (data leakage!)
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].select_dtypes(include=['number']).fillna(0)
y = df['grade_category'].values

# Double-check for leakage
leakage_keywords = ['grade', 'score', 'final']
potential_leakage = [c for c in X.columns if any(kw in c.lower() for kw in leakage_keywords)]
if potential_leakage:
    print(f"\n⚠️  WARNING: Potential leakage features found:")
    for feat in potential_leakage[:10]:
        print(f"      - {feat}")
    X = X.drop(columns=[c for c in potential_leakage if c in X.columns])

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"   Features: {X.shape[1]}")
print(f"   Classes: {list(le.classes_)}")
print(f"   Distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")

# Count SOTA features
sota_feature_keywords = ['inactivity', 'decay', 'momentum', 'bigram', 'transition', 'early_deceleration',
                         'fade_score', 'days_until', 'stall_in_first']
sota_features = [c for c in X.columns if any(kw in c for kw in sota_feature_keywords)]
print(f"\n   SOTA features in model: {len(sota_features)} / {X.shape[1]}")

# Train XGBoost multi-class model
print("\n2. Training XGBoost multi-class model...")

# Calculate class weights (focus on FAIL recall)
class_counts = np.bincount(y_encoded)
weights = {i: len(y_encoded) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
# Boost FAIL class weight
fail_idx = list(le.classes_).index('FAIL')
weights[fail_idx] *= 3  # 3x weight for FAIL class

print(f"   Class weights: {dict(zip(le.classes_, [weights[i] for i in range(len(le.classes_))]))}")

model = XGBClassifier(
    objective='multi:softmax',
    num_class=len(le.classes_),
    n_estimators=200,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)

# 5-fold stratified cross-validation
print("\n3. Running 5-fold stratified cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Proper cross-validation: collect predictions
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(model, X, y_encoded, cv=cv, n_jobs=-1)
y_pred_proba = cross_val_predict(model, X, y_encoded, cv=cv, method='predict_proba', n_jobs=-1)

# Train on full data for feature importance only
model.fit(X, y_encoded)

# Calculate metrics
print("\n4. Calculating metrics (from cross-validation)...")

# Overall metrics
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_encoded, y_pred)
f1_macro = f1_score(y_encoded, y_pred, average='macro')
f1_weighted = f1_score(y_encoded, y_pred, average='weighted')

# ROC-AUC (one-vs-rest)
roc_auc_ovr = roc_auc_score(y_encoded, y_pred_proba, multi_class='ovr', average='macro')

# Per-class metrics
report = classification_report(y_encoded, y_pred, target_names=le.classes_, output_dict=True)

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred)

print("\n" + "="*70)
print("RESULTS - SOTA MODEL")
print("="*70)

print(f"\nOverall Metrics:")
print(f"  Accuracy: {accuracy:.3f}")
print(f"  F1-Macro: {f1_macro:.3f}")
print(f"  F1-Weighted: {f1_weighted:.3f}")
print(f"  ROC-AUC (OvR): {roc_auc_ovr:.3f}")

print(f"\nPer-Class Performance:")
for cls in le.classes_:
    metrics = report[cls]
    print(f"  {cls}:")
    print(f"    Precision: {metrics['precision']:.3f}")
    print(f"    Recall:    {metrics['recall']:.3f}")
    print(f"    F1-Score:  {metrics['f1-score']:.3f}")
    print(f"    Support:   {int(metrics['support'])}")

# FAIL class focus
fail_recall = report['FAIL']['recall']
print(f"\n🎯 FAIL Recall (KEY METRIC): {fail_recall:.1%}")

print(f"\nConfusion Matrix:")
print(f"  Rows=True, Cols=Predicted")
print(f"  Order: {', '.join(le.classes_)}")
for i, row in enumerate(cm):
    print(f"  {le.classes_[i]:10s}: {row}")

# Feature importance
print(f"\n5. Top 20 Features by Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for i, row in feature_importance.head(20).iterrows():
    is_sota = '✨' if any(kw in row['feature'] for kw in sota_feature_keywords) else '  '
    print(f"  {is_sota} {row['feature']:40s}: {row['importance']:.4f}")

# Count SOTA features in top 20
top20_sota = sum(1 for _, row in feature_importance.head(20).iterrows()
                 if any(kw in row['feature'] for kw in sota_feature_keywords))
print(f"\n  SOTA features in top 20: {top20_sota} / 20")

# Save results
print(f"\n6. Saving results...")

metrics = {
    'overall': {
        'accuracy': float(accuracy),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weighted),
        'roc_auc_ovr': float(roc_auc_ovr)
    },
    'per_class': {cls: {
        'precision': float(report[cls]['precision']),
        'recall': float(report[cls]['recall']),
        'f1_score': float(report[cls]['f1-score']),
        'support': int(report[cls]['support'])
    } for cls in le.classes_},
    'fail_recall': float(fail_recall),
    'confusion_matrix': cm.tolist(),
    'feature_count': X.shape[1],
    'sota_feature_count': len(sota_features),
    'top_20_features': feature_importance.head(20).to_dict('records')
}

with open(OUTPUT_DIR / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

print(f"   ✓ Metrics: {OUTPUT_DIR / 'metrics.json'}")
print(f"   ✓ Feature importance: {OUTPUT_DIR / 'feature_importance.csv'}")

# Compare to baseline
if BASELINE_METRICS.exists():
    print(f"\n7. Comparison to Baseline:")
    baseline = json.load(open(BASELINE_METRICS))

    print(f"\n  Metric              Baseline    SOTA        Change")
    print(f"  {'-'*55}")
    print(f"  ROC-AUC (OvR):      {baseline['overall']['roc_auc_ovr']:.3f}       {roc_auc_ovr:.3f}      {roc_auc_ovr - baseline['overall']['roc_auc_ovr']:+.3f}")
    print(f"  F1-Weighted:        {baseline['overall']['f1_weighted']:.3f}       {f1_weighted:.3f}      {f1_weighted - baseline['overall']['f1_weighted']:+.3f}")

    baseline_fail_recall = baseline['per_class']['FAIL']['recall']
    print(f"  FAIL Recall:        {baseline_fail_recall:.1%}     {fail_recall:.1%}    {fail_recall - baseline_fail_recall:+.1%}")

    if fail_recall >= 0.50:
        print(f"\n  🎉 SUCCESS! FAIL recall ≥50% target achieved!")
    elif fail_recall >= 0.40:
        print(f"\n  ✅ Good progress! FAIL recall ≥40% (Phase 4 may reach 50%)")
    else:
        print(f"\n  ⚠️  FAIL recall <40% - Phase 4 advanced techniques needed")

print("\n" + "="*70)
print("TRAINING COMPLETE")
print("="*70)
