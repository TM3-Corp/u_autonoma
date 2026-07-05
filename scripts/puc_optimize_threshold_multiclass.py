#!/usr/bin/env python3
"""
Phase 4.1: Threshold Optimization for PUC Multi-Class Early Warning Model

Instead of using argmax(probabilities), optimizes probability threshold for FAIL class
to maximize F2-score (weights recall 2x more than precision).

Expected gain: +10-15% FAIL recall
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, f1_score, fbeta_score,
    roc_auc_score, accuracy_score
)
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
SOTA_FEATURES = Path('data/puc/enriched_features/all_features_sota.parquet')
OUTPUT_DIR = Path('data/puc/models/multiclass_threshold_optimized')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 4.1: THRESHOLD OPTIMIZATION - PUC Multi-Class")
print("="*70)

# Load data
print("\n1. Loading SOTA features...")
df = pd.read_parquet(SOTA_FEATURES)

# Prepare X and y
exclude_cols = ['student_id', 'course_id', 'grade_category', 'grade', 'failed', 'class_label']
# Remove potential leakage features
leakage_keywords = ['grade', 'score', 'final']
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].select_dtypes(include=['number']).fillna(0)

# Remove leakage
leakage_features = [c for c in X.columns if any(kw in c.lower() for kw in leakage_keywords)]
X = X.drop(columns=leakage_features)

y = df['grade_category'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
fail_idx = list(le.classes_).index('FAIL')

print(f"   Features: {X.shape[1]}")
print(f"   Classes: {list(le.classes_)}")
print(f"   FAIL class index: {fail_idx}")
print(f"   Distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")

# Train model with 5-fold CV to get probabilities
print("\n2. Training XGBoost with 5-fold CV...")

# Calculate class weights
class_counts = np.bincount(y_encoded)
weights = {i: len(y_encoded) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
weights[fail_idx] *= 3  # Boost FAIL class

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

# Get cross-validated probabilities
from sklearn.model_selection import cross_val_predict
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba = cross_val_predict(model, X, y_encoded, cv=cv, method='predict_proba', n_jobs=-1)

print(f"   Probability shape: {y_proba.shape}")
print(f"   FAIL probabilities - min: {y_proba[:, fail_idx].min():.3f}, max: {y_proba[:, fail_idx].max():.3f}")

# Baseline: argmax predictions
y_pred_argmax = np.argmax(y_proba, axis=1)

print("\n3. Baseline Performance (argmax):")
report_argmax = classification_report(y_encoded, y_pred_argmax,
                                      target_names=le.classes_,
                                      output_dict=True, zero_division=0)
print(f"   FAIL Recall: {report_argmax['FAIL']['recall']:.1%}")
print(f"   FAIL Precision: {report_argmax['FAIL']['precision']:.1%}")
print(f"   FAIL F1-Score: {report_argmax['FAIL']['f1-score']:.3f}")

# Calculate F2-score for FAIL (weights recall 2x more than precision)
fail_f2_argmax = fbeta_score(
    (y_encoded == fail_idx).astype(int),
    (y_pred_argmax == fail_idx).astype(int),
    beta=2.0,
    zero_division=0
)
print(f"   FAIL F2-Score: {fail_f2_argmax:.3f}")

# Threshold optimization
print("\n4. Optimizing threshold for FAIL class...")
print("   Grid search over thresholds: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]")

thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
results = []

for threshold in thresholds:
    # Predict FAIL if P(FAIL) > threshold, else use argmax of others
    y_pred_threshold = np.argmax(y_proba, axis=1)

    # Override with FAIL if probability exceeds threshold
    fail_mask = y_proba[:, fail_idx] > threshold
    y_pred_threshold[fail_mask] = fail_idx

    # Calculate metrics
    report = classification_report(y_encoded, y_pred_threshold,
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)

    fail_recall = report['FAIL']['recall']
    fail_precision = report['FAIL']['precision']
    fail_f1 = report['FAIL']['f1-score']

    # F2-score (beta=2 weights recall 2x more than precision)
    fail_f2 = fbeta_score(
        (y_encoded == fail_idx).astype(int),
        (y_pred_threshold == fail_idx).astype(int),
        beta=2.0,
        zero_division=0
    )

    # Overall metrics
    accuracy = accuracy_score(y_encoded, y_pred_threshold)
    f1_weighted = f1_score(y_encoded, y_pred_threshold, average='weighted', zero_division=0)

    results.append({
        'threshold': threshold,
        'fail_recall': fail_recall,
        'fail_precision': fail_precision,
        'fail_f1': fail_f1,
        'fail_f2': fail_f2,
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'n_predicted_fail': (y_pred_threshold == fail_idx).sum()
    })

results_df = pd.DataFrame(results)

# Find optimal threshold (maximize F2-score)
optimal_idx = results_df['fail_f2'].idxmax()
optimal_threshold = results_df.loc[optimal_idx, 'threshold']
optimal_f2 = results_df.loc[optimal_idx, 'fail_f2']

print(f"\n   Optimal threshold: {optimal_threshold:.2f}")
print(f"   Optimal F2-score: {optimal_f2:.3f}")

# Display results table
print("\n5. Threshold Performance Comparison:")
print("\n   Threshold  FAIL Recall  FAIL Precision  F1-Score  F2-Score  Accuracy  Predicted FAIL")
print("   " + "-"*88)
for _, row in results_df.iterrows():
    marker = " ← OPTIMAL" if row['threshold'] == optimal_threshold else ""
    print(f"   {row['threshold']:5.2f}      {row['fail_recall']:7.1%}      {row['fail_precision']:9.1%}      "
          f"{row['fail_f1']:6.3f}    {row['fail_f2']:6.3f}    {row['accuracy']:6.1%}    "
          f"{int(row['n_predicted_fail']):4d}{marker}")

# Detailed results for optimal threshold
print(f"\n6. Detailed Performance at Optimal Threshold ({optimal_threshold:.2f}):")

# Get predictions with optimal threshold
y_pred_optimal = np.argmax(y_proba, axis=1)
fail_mask_optimal = y_proba[:, fail_idx] > optimal_threshold
y_pred_optimal[fail_mask_optimal] = fail_idx

# Full classification report
report_optimal = classification_report(y_encoded, y_pred_optimal,
                                       target_names=le.classes_,
                                       output_dict=True, zero_division=0)

print(f"\n   Per-Class Performance:")
for cls in le.classes_:
    metrics = report_optimal[cls]
    print(f"   {cls:10s}: Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
          f"Support={int(metrics['support'])}")

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred_optimal)
print(f"\n   Confusion Matrix:")
print(f"   Rows=True, Cols=Predicted")
print(f"   Order: {', '.join(le.classes_)}")
for i, row in enumerate(cm):
    print(f"   {le.classes_[i]:10s}: {row}")

# Calculate improvement
fail_recall_baseline = report_argmax['FAIL']['recall']
fail_recall_optimal = report_optimal['FAIL']['recall']
improvement = fail_recall_optimal - fail_recall_baseline

print(f"\n7. Improvement Summary:")
print(f"\n   Metric              Baseline (argmax)  Optimized (t={optimal_threshold:.2f})  Change")
print(f"   " + "-"*75)
print(f"   FAIL Recall:        {fail_recall_baseline:7.1%}            {fail_recall_optimal:7.1%}           {improvement:+6.1%}")
print(f"   FAIL Precision:     {report_argmax['FAIL']['precision']:7.1%}            "
      f"{report_optimal['FAIL']['precision']:7.1%}           "
      f"{report_optimal['FAIL']['precision'] - report_argmax['FAIL']['precision']:+6.1%}")
print(f"   FAIL F2-Score:      {fail_f2_argmax:7.3f}            {optimal_f2:7.3f}           {optimal_f2 - fail_f2_argmax:+6.3f}")

if improvement >= 0.10:
    print(f"\n   ✅ SUCCESS! Gained {improvement:.1%} FAIL recall (+10% or more)")
elif improvement >= 0.05:
    print(f"\n   ✅ Good progress! Gained {improvement:.1%} FAIL recall")
elif improvement > 0:
    print(f"\n   ↗️  Modest improvement: {improvement:.1%} FAIL recall")
else:
    print(f"\n   ⚠️  No improvement - threshold optimization may not help with current features")

# Check if we reached 50% target
if fail_recall_optimal >= 0.50:
    print(f"\n   🎉 TARGET ACHIEVED! FAIL recall ≥50%")
elif fail_recall_optimal >= 0.40:
    print(f"\n   📈 Strong progress! FAIL recall ≥40% (close to 50% target)")
else:
    print(f"\n   📊 Current FAIL recall: {fail_recall_optimal:.1%}")
    print(f"   🎯 Need {0.50 - fail_recall_optimal:.1%} more to reach 50% target")
    print(f"   💡 Next steps: Phase 4.2 (Proactivity features) + Phase 4.3 (Feature selection)")

# Plot threshold vs metrics
print("\n8. Generating threshold analysis plots...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Recall vs Threshold
axes[0, 0].plot(results_df['threshold'], results_df['fail_recall'], 'b-o', linewidth=2, markersize=8)
axes[0, 0].axvline(optimal_threshold, color='r', linestyle='--', label=f'Optimal ({optimal_threshold:.2f})')
axes[0, 0].axhline(0.50, color='g', linestyle=':', label='50% Target')
axes[0, 0].set_xlabel('Threshold', fontsize=12)
axes[0, 0].set_ylabel('FAIL Recall', fontsize=12)
axes[0, 0].set_title('FAIL Recall vs Threshold', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Plot 2: Precision vs Recall (trade-off)
axes[0, 1].plot(results_df['fail_recall'], results_df['fail_precision'], 'g-o', linewidth=2, markersize=8)
axes[0, 1].scatter(fail_recall_optimal, report_optimal['FAIL']['precision'],
                   color='r', s=200, marker='*', label='Optimal', zorder=5)
axes[0, 1].set_xlabel('FAIL Recall', fontsize=12)
axes[0, 1].set_ylabel('FAIL Precision', fontsize=12)
axes[0, 1].set_title('Precision-Recall Trade-off', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Plot 3: F1 and F2 scores vs Threshold
axes[1, 0].plot(results_df['threshold'], results_df['fail_f1'], 'b-o', label='F1-Score', linewidth=2, markersize=8)
axes[1, 0].plot(results_df['threshold'], results_df['fail_f2'], 'r-s', label='F2-Score', linewidth=2, markersize=8)
axes[1, 0].axvline(optimal_threshold, color='k', linestyle='--', alpha=0.5, label=f'Optimal ({optimal_threshold:.2f})')
axes[1, 0].set_xlabel('Threshold', fontsize=12)
axes[1, 0].set_ylabel('Score', fontsize=12)
axes[1, 0].set_title('F1 vs F2 Score (F2 weights recall 2x)', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Overall accuracy vs FAIL recall
axes[1, 1].scatter(results_df['fail_recall'], results_df['accuracy'], s=100, c=results_df['threshold'],
                   cmap='viridis', edgecolors='black', linewidth=1.5)
axes[1, 1].scatter(fail_recall_optimal, results_df.loc[optimal_idx, 'accuracy'],
                   color='r', s=300, marker='*', label='Optimal', zorder=5)
axes[1, 1].set_xlabel('FAIL Recall', fontsize=12)
axes[1, 1].set_ylabel('Overall Accuracy', fontsize=12)
axes[1, 1].set_title('Accuracy vs FAIL Recall Trade-off', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Threshold', fontsize=10)

plt.tight_layout()
plot_path = OUTPUT_DIR / 'threshold_optimization_analysis.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"   ✓ Saved: {plot_path}")

# Save results
print("\n9. Saving results...")

# Save metrics
metrics = {
    'optimal_threshold': float(optimal_threshold),
    'baseline_argmax': {
        'fail_recall': float(fail_recall_baseline),
        'fail_precision': float(report_argmax['FAIL']['precision']),
        'fail_f1': float(report_argmax['FAIL']['f1-score']),
        'fail_f2': float(fail_f2_argmax)
    },
    'optimized': {
        'fail_recall': float(fail_recall_optimal),
        'fail_precision': float(report_optimal['FAIL']['precision']),
        'fail_f1': float(report_optimal['FAIL']['f1-score']),
        'fail_f2': float(optimal_f2)
    },
    'improvement': {
        'fail_recall_gain': float(improvement),
        'fail_recall_gain_pct': float(improvement * 100)
    },
    'all_thresholds': results_df.to_dict('records'),
    'per_class_optimal': {cls: {
        'precision': float(report_optimal[cls]['precision']),
        'recall': float(report_optimal[cls]['recall']),
        'f1_score': float(report_optimal[cls]['f1-score']),
        'support': int(report_optimal[cls]['support'])
    } for cls in le.classes_},
    'confusion_matrix': cm.tolist()
}

with open(OUTPUT_DIR / 'threshold_optimization_results.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save threshold comparison table
results_df.to_csv(OUTPUT_DIR / 'threshold_comparison.csv', index=False)

print(f"   ✓ Metrics: {OUTPUT_DIR / 'threshold_optimization_results.json'}")
print(f"   ✓ Comparison table: {OUTPUT_DIR / 'threshold_comparison.csv'}")
print(f"   ✓ Plots: {OUTPUT_DIR / 'threshold_optimization_analysis.png'}")

print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION COMPLETE")
print("="*70)
print(f"\n✨ Key Takeaway:")
print(f"   Using threshold {optimal_threshold:.2f} instead of argmax improves")
print(f"   FAIL recall by {improvement:+.1%} (from {fail_recall_baseline:.1%} to {fail_recall_optimal:.1%})")

if fail_recall_optimal >= 0.50:
    print(f"\n🎉 TARGET ACHIEVED! Ready for production deployment.")
elif fail_recall_optimal >= 0.40:
    print(f"\n📈 Strong progress! Consider Phase 4.2-4.3 to reach 50% target.")
else:
    gap_to_50 = 0.50 - fail_recall_optimal
    print(f"\n📊 Current: {fail_recall_optimal:.1%} | Target: 50% | Gap: {gap_to_50:.1%}")
    print(f"   Recommended: Phase 4.2 (Proactivity) + Phase 4.3 (Selection) for remaining {gap_to_50:.1%}")

print("="*70)
