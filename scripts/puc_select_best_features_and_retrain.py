#!/usr/bin/env python3
"""
Phase 4.3: Feature Selection - Select Top 50 Features and Retrain

With 520 features, the model is likely overfitting. Select the top 50 features
by XGBoost importance and retrain for better generalization.

Expected: Better precision, maintain/improve recall (reduce overfitting)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
PROACTIVITY_MODEL_DIR = Path('data/puc/models/multiclass_with_proactivity')
FEATURE_IMPORTANCE_FILE = PROACTIVITY_MODEL_DIR / 'feature_importance.csv'
MERGED_FEATURES_FILE = Path('data/puc/enriched_features/all_features_with_pct.parquet')
OUTPUT_DIR = Path('data/puc/models/multiclass_final_optimized')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 4.3: FEATURE SELECTION + FINAL OPTIMIZATION")
print("="*70)

# Load feature importance from previous run
print("\n1. Loading feature importance rankings...")
feature_importance = pd.read_csv(FEATURE_IMPORTANCE_FILE)
print(f"   Total features ranked: {len(feature_importance)}")

# Select top N features
TOP_N = 50
top_features = feature_importance.head(TOP_N)['feature'].tolist()

print(f"\n2. Selected top {TOP_N} features:")
for i, feat in enumerate(top_features[:15], 1):
    imp = feature_importance[feature_importance['feature'] == feat]['importance'].values[0]
    print(f"   {i:2d}. {feat:45s} (importance: {imp:.4f})")
print(f"   ... and {TOP_N - 15} more")

# Count PCT features in top N
pct_keywords = ['_pct', '_hist_b', 'top25_rate', 'access_rate']
pct_in_topN = sum(1 for f in top_features if any(kw in f for kw in pct_keywords))
print(f"\n   PCT features in top {TOP_N}: {pct_in_topN} / {TOP_N} ({100*pct_in_topN/TOP_N:.0f}%)")

# Load full dataset and select top features
print(f"\n3. Loading dataset and selecting top {TOP_N} features...")

# Reconstruct the dataset (merge SOTA + PCT)
from pathlib import Path
df_sota = pd.read_parquet('data/puc/enriched_features/all_features_sota.parquet')
df_pct = pd.read_parquet('data/puc/enriched_features/proactivity_features.parquet')
df_merged = df_sota.merge(df_pct, on=['student_id', 'course_id'], how='left')

# Normalize PCT features
pct_feature_cols = [c for c in df_merged.columns
                    if any(kw in c for kw in pct_keywords)
                    and c not in ['student_id', 'course_id', 'grade_category']]

for col in pct_feature_cols:
    if col in df_merged.columns and df_merged[col].dtype in [np.float64, np.int64]:
        if not col.endswith('_znorm'):
            df_merged[f'{col}_znorm'] = df_merged.groupby('course_id')[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

# Prepare features
exclude_cols = ['student_id', 'course_id', 'grade_category', 'grade', 'failed', 'class_label']
leakage_keywords = ['grade', 'score', 'final']

feature_cols = [c for c in df_merged.columns if c not in exclude_cols]
X_full = df_merged[feature_cols].select_dtypes(include=['number']).fillna(0)
leakage_features = [c for c in X_full.columns if any(kw in c.lower() for kw in leakage_keywords)]
X_full = X_full.drop(columns=leakage_features)

# Select only top features
X = X_full[top_features].copy()
y = df_merged['grade_category'].values

print(f"   Selected features shape: {X.shape}")
print(f"   Reduced from {X_full.shape[1]} to {X.shape[1]} features ({100*(1-X.shape[1]/X_full.shape[1]):.0f}% reduction)")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
fail_idx = list(le.classes_).index('FAIL')

print(f"   Classes: {list(le.classes_)}")
print(f"   Distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")

# Train model with CV
print("\n4. Training XGBoost with top {TOP_N} features...")

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

# Cross-validated probabilities
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_proba = cross_val_predict(model, X, y_encoded, cv=cv, method='predict_proba', n_jobs=-1)

# Train on full data
model.fit(X, y_encoded)

# Calculate ROC-AUC
roc_auc = roc_auc_score(y_encoded, y_proba, multi_class='ovr', average='macro')
print(f"   ROC-AUC (OvR): {roc_auc:.3f}")

# Baseline (argmax)
y_pred_argmax = np.argmax(y_proba, axis=1)
report_argmax = classification_report(y_encoded, y_pred_argmax,
                                      target_names=le.classes_,
                                      output_dict=True, zero_division=0)
fail_recall_argmax = report_argmax['FAIL']['recall']

print(f"   Baseline (argmax) FAIL recall: {fail_recall_argmax:.1%}")

# Threshold optimization
print("\n5. Optimizing threshold...")
thresholds = [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25]
results = []

for threshold in thresholds:
    y_pred_threshold = np.argmax(y_proba, axis=1)
    fail_mask = y_proba[:, fail_idx] > threshold
    y_pred_threshold[fail_mask] = fail_idx

    report = classification_report(y_encoded, y_pred_threshold,
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)

    f2 = fbeta_score(
        (y_encoded == fail_idx).astype(int),
        (y_pred_threshold == fail_idx).astype(int),
        beta=2.0,
        zero_division=0
    )

    results.append({
        'threshold': threshold,
        'fail_recall': report['FAIL']['recall'],
        'fail_precision': report['FAIL']['precision'],
        'fail_f2': f2,
        'f1_weighted': float(pd.Series([report[cls]['f1-score'] * report[cls]['support']
                                        for cls in le.classes_]).sum() / len(y))
    })

results_df = pd.DataFrame(results)
optimal_idx = results_df['fail_f2'].idxmax()
best_threshold = results_df.loc[optimal_idx, 'threshold']
best_recall = results_df.loc[optimal_idx, 'fail_recall']
best_f2 = results_df.loc[optimal_idx, 'fail_f2']

print(f"\n   Threshold  FAIL Recall  FAIL Precision  F2-Score")
print(f"   {'-'*55}")
for _, row in results_df.iterrows():
    marker = " ← OPTIMAL" if row['threshold'] == best_threshold else ""
    print(f"   {row['threshold']:5.2f}      {row['fail_recall']:7.1%}      {row['fail_precision']:9.1%}      {row['fail_f2']:6.3f}{marker}")

# Final predictions
y_pred_optimal = np.argmax(y_proba, axis=1)
fail_mask_optimal = y_proba[:, fail_idx] > best_threshold
y_pred_optimal[fail_mask_optimal] = fail_idx

report_optimal = classification_report(y_encoded, y_pred_optimal,
                                       target_names=le.classes_,
                                       output_dict=True, zero_division=0)

print("\n" + "="*70)
print(f"FINAL RESULTS (TOP {TOP_N} FEATURES + OPTIMIZED THRESHOLD)")
print("="*70)

print(f"\nPer-Class Performance:")
for cls in le.classes_:
    metrics = report_optimal[cls]
    print(f"  {cls:10s}: Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}, "
          f"Support={int(metrics['support'])}")

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred_optimal)
print(f"\nConfusion Matrix (Rows=Actual, Cols=Predicted):")
for i, row in enumerate(cm):
    print(f"  {le.classes_[i]:10s}: {row}")

# Comparison
print(f"\n6. Comparison Across All Phases:")

phases = [
    ("Baseline (Multi-Class)", 0.692, 0.243),
    ("+ SOTA Features", 0.705, 0.220),
    ("+ Threshold Opt", 0.705, 0.622),
    ("+ Proactivity Features", 0.704, 0.622),
    (f"+ Feature Selection (top {TOP_N})", roc_auc, best_recall)
]

print(f"\n  Phase                              ROC-AUC    FAIL Recall    Change")
print(f"  {'-'*75}")
for i, (phase, auc, recall) in enumerate(phases):
    if i == 0:
        change_auc = "-"
        change_recall = "-"
    else:
        change_auc = f"{auc - phases[i-1][1]:+.3f}"
        change_recall = f"{recall - phases[i-1][2]:+.1%}"
    print(f"  {phase:35s}  {auc:.3f}      {recall:.1%}       {change_recall:>6s}")

# Final assessment
print(f"\n7. Final Assessment:")
improvement_from_baseline = best_recall - 0.243
print(f"   Total improvement from baseline: {improvement_from_baseline:+.1%}")
print(f"   FAIL students identified: {int(best_recall * 82)}/82")
print(f"   FAIL students missed: {82 - int(best_recall * 82)}/82")

if best_recall >= 0.70:
    print(f"\n   🎉 70% TARGET ACHIEVED!")
elif best_recall >= 0.65:
    print(f"\n   📈 Close to 70% target! ({best_recall:.1%})")
    print(f"   Gap to 70%: {0.70 - best_recall:.1%}")
else:
    print(f"\n   📊 Current: {best_recall:.1%}")
    print(f"   Gap to 70%: {0.70 - best_recall:.1%}")

print(f"\n   Model Status: {'✅ PRODUCTION-READY' if best_recall >= 0.60 else '⚠️  NEEDS MORE WORK'}")

# Save results
print(f"\n8. Saving final model and results...")

metrics = {
    'top_n_features': TOP_N,
    'selected_features': top_features,
    'pct_features_in_selection': pct_in_topN,
    'optimal_threshold': float(best_threshold),
    'performance': {
        'roc_auc_ovr': float(roc_auc),
        'fail_recall': float(best_recall),
        'fail_precision': float(report_optimal['FAIL']['precision']),
        'fail_f1': float(report_optimal['FAIL']['f1-score']),
        'fail_f2': float(best_f2)
    },
    'per_class': {cls: {
        'precision': float(report_optimal[cls]['precision']),
        'recall': float(report_optimal[cls]['recall']),
        'f1_score': float(report_optimal[cls]['f1-score']),
        'support': int(report_optimal[cls]['support'])
    } for cls in le.classes_},
    'confusion_matrix': cm.tolist(),
    'threshold_results': results_df.to_dict('records')
}

with open(OUTPUT_DIR / 'final_model_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save selected features list
with open(OUTPUT_DIR / 'selected_features.json', 'w') as f:
    json.dump({'features': top_features, 'count': len(top_features)}, f, indent=2)

print(f"   ✓ Final metrics: {OUTPUT_DIR / 'final_model_metrics.json'}")
print(f"   ✓ Selected features: {OUTPUT_DIR / 'selected_features.json'}")

print("\n" + "="*70)
print("PHASE 4.3 COMPLETE - FINAL MODEL READY")
print("="*70)
print(f"\n✨ Final Model Summary:")
print(f"   Features: {TOP_N} (reduced from 520)")
print(f"   ROC-AUC: {roc_auc:.3f}")
print(f"   FAIL Recall: {best_recall:.1%}")
print(f"   Optimal Threshold: {best_threshold:.2f}")
print(f"   Production Status: {'✅ READY' if best_recall >= 0.60 else '⚠️  REVIEW NEEDED'}")
print("="*70)
