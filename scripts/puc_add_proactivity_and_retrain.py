#!/usr/bin/env python3
"""
Phase 4.2: Add Proactivity Features and Retrain Model

Merges proactivity (PCT) features with existing SOTA features,
then retrains with threshold optimization.

Expected: +5-8% FAIL recall (from 62.2% to 67-70%)
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
SOTA_FEATURES = Path('data/puc/enriched_features/all_features_sota.parquet')
PROACTIVITY_FEATURES = Path('data/puc/enriched_features/proactivity_features.parquet')
OUTPUT_DIR = Path('data/puc/models/multiclass_with_proactivity')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 4.2: ADD PROACTIVITY FEATURES + RETRAIN")
print("="*70)

# Load existing SOTA features
print("\n1. Loading existing SOTA features...")
df_sota = pd.read_parquet(SOTA_FEATURES)
print(f"   Shape: {df_sota.shape}")

# Load proactivity features
print("\n2. Loading proactivity (PCT) features...")
df_pct = pd.read_parquet(PROACTIVITY_FEATURES)
print(f"   Shape: {df_pct.shape}")
print(f"   PCT feature columns: {len([c for c in df_pct.columns if c not in ['student_id', 'course_id']])}")

# Merge
print("\n3. Merging features...")
df_merged = df_sota.merge(df_pct, on=['student_id', 'course_id'], how='left')
print(f"   Merged shape: {df_merged.shape}")
print(f"   Total features (including IDs): {df_merged.shape[1]}")

# Apply course-relative normalization to PCT features
print("\n4. Applying course-relative normalization to PCT features...")

# Get PCT feature columns that are actually in merged dataframe
# (handle any column renaming that might have occurred during merge)
pct_keywords = ['_pct', '_hist_b', 'top25_rate', 'access_rate']
pct_feature_cols = [c for c in df_merged.columns
                    if any(kw in c for kw in pct_keywords)
                    and c not in ['student_id', 'course_id', 'grade_category']]

print(f"   Found {len(pct_feature_cols)} PCT-related columns to normalize")

for col in pct_feature_cols:
    if col in df_merged.columns and df_merged[col].dtype in [np.float64, np.int64]:
        # Only normalize if not already normalized
        if not col.endswith('_znorm'):
            df_merged[f'{col}_znorm'] = df_merged.groupby('course_id')[col].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
            )

print(f"   Added {len(pct_feature_cols)} normalized PCT features")
print(f"   Total columns now: {df_merged.shape[1]}")

# Prepare X and y
print("\n5. Preparing features and target...")
exclude_cols = ['student_id', 'course_id', 'grade_category', 'grade', 'failed', 'class_label']
leakage_keywords = ['grade', 'score', 'final']

feature_cols = [c for c in df_merged.columns if c not in exclude_cols]
X = df_merged[feature_cols].select_dtypes(include=['number']).fillna(0)

# Remove leakage
leakage_features = [c for c in X.columns if any(kw in c.lower() for kw in leakage_keywords)]
X = X.drop(columns=leakage_features)

y = df_merged['grade_category'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
fail_idx = list(le.classes_).index('FAIL')

print(f"   Features: {X.shape[1]} (was 293 before PCT)")
print(f"   PCT features added: {X.shape[1] - 293}")
print(f"   Classes: {list(le.classes_)}")
print(f"   Distribution: {dict(zip(le.classes_, np.bincount(y_encoded)))}")

# Check PCT features are included
pct_features_in_model = [c for c in X.columns if '_pct' in c or '_hist_b' in c]
print(f"   PCT-related features in model: {len(pct_features_in_model)}")

# Train model with CV
print("\n6. Training XGBoost with 5-fold CV...")

class_counts = np.bincount(y_encoded)
weights = {i: len(y_encoded) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
weights[fail_idx] *= 3

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

# Train on full data for feature importance
model.fit(X, y_encoded)

# Baseline (argmax)
y_pred_argmax = np.argmax(y_proba, axis=1)
report_argmax = classification_report(y_encoded, y_pred_argmax,
                                      target_names=le.classes_,
                                      output_dict=True, zero_division=0)
fail_recall_argmax = report_argmax['FAIL']['recall']

print(f"\n   Baseline (argmax) FAIL recall: {fail_recall_argmax:.1%}")

# Threshold optimization
print("\n7. Optimizing threshold...")
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
best_f2 = 0
best_threshold = 0.05
best_recall = 0

for threshold in thresholds:
    y_pred_threshold = np.argmax(y_proba, axis=1)
    fail_mask = y_proba[:, fail_idx] > threshold
    y_pred_threshold[fail_mask] = fail_idx

    f2 = fbeta_score(
        (y_encoded == fail_idx).astype(int),
        (y_pred_threshold == fail_idx).astype(int),
        beta=2.0,
        zero_division=0
    )

    report = classification_report(y_encoded, y_pred_threshold,
                                   target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    recall = report['FAIL']['recall']

    if f2 > best_f2:
        best_f2 = f2
        best_threshold = threshold
        best_recall = recall

print(f"   Optimal threshold: {best_threshold:.2f}")
print(f"   Optimal F2-score: {best_f2:.3f}")
print(f"   Optimal FAIL recall: {best_recall:.1%}")

# Generate final predictions with optimal threshold
y_pred_optimal = np.argmax(y_proba, axis=1)
fail_mask_optimal = y_proba[:, fail_idx] > best_threshold
y_pred_optimal[fail_mask_optimal] = fail_idx

report_optimal = classification_report(y_encoded, y_pred_optimal,
                                       target_names=le.classes_,
                                       output_dict=True, zero_division=0)

print("\n" + "="*70)
print("RESULTS WITH PROACTIVITY FEATURES")
print("="*70)

print(f"\nPer-Class Performance:")
for cls in le.classes_:
    metrics = report_optimal[cls]
    print(f"  {cls:10s}: Precision={metrics['precision']:.3f}, "
          f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")

# Confusion matrix
cm = confusion_matrix(y_encoded, y_pred_optimal)
print(f"\nConfusion Matrix:")
print(f"  Rows=True, Cols=Predicted")
for i, row in enumerate(cm):
    print(f"  {le.classes_[i]:10s}: {row}")

# Feature importance
print(f"\n8. Top 20 Features by Importance:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

pct_keywords = ['_pct', '_hist_b', 'top25', 'access_rate']
for i, row in feature_importance.head(20).iterrows():
    is_pct = '⭐' if any(kw in row['feature'] for kw in pct_keywords) else '  '
    print(f"  {is_pct} {row['feature']:45s}: {row['importance']:.4f}")

pct_in_top20 = sum(1 for _, row in feature_importance.head(20).iterrows()
                   if any(kw in row['feature'] for kw in pct_keywords))
print(f"\n  PCT features in top 20: {pct_in_top20} / 20")

# Comparison to previous
print(f"\n9. Comparison to Previous Model (without PCT):")

# Load previous results
prev_results_file = Path('data/puc/models/multiclass_threshold_optimized/threshold_optimization_results.json')
if prev_results_file.exists():
    prev_results = json.load(open(prev_results_file))
    prev_recall = prev_results['optimized']['fail_recall']

    print(f"\n  Metric                   Previous    With PCT    Change")
    print(f"  {'-'*65}")
    print(f"  FAIL Recall:             {prev_recall:.1%}       {best_recall:.1%}      {best_recall - prev_recall:+.1%}")
    print(f"  ROC-AUC (OvR):           0.705       {roc_auc_score(y_encoded, y_proba, multi_class='ovr', average='macro'):.3f}      "
          f"{roc_auc_score(y_encoded, y_proba, multi_class='ovr', average='macro') - 0.705:+.3f}")

    improvement = best_recall - prev_recall
    if improvement >= 0.05:
        print(f"\n  ✅ SUCCESS! Gained {improvement:.1%} FAIL recall with PCT features")
    elif improvement > 0:
        print(f"\n  ↗️  Modest improvement: {improvement:.1%} FAIL recall")
    else:
        print(f"\n  ⚠️  No significant improvement from PCT features")

    # Check if we hit 70% target
    if best_recall >= 0.70:
        print(f"\n  🎉 70% TARGET ACHIEVED!")
    elif best_recall >= 0.65:
        print(f"\n  📈 Close to 70% target! ({best_recall:.1%})")
    else:
        print(f"\n  📊 Current: {best_recall:.1%} | Target: 70% | Gap: {0.70 - best_recall:.1%}")

# Save results
print(f"\n10. Saving results...")

metrics = {
    'total_features': int(X.shape[1]),
    'pct_features': len(pct_features_in_model),
    'optimal_threshold': float(best_threshold),
    'baseline_argmax': {
        'fail_recall': float(fail_recall_argmax),
        'fail_precision': float(report_argmax['FAIL']['precision']),
        'fail_f1': float(report_argmax['FAIL']['f1-score'])
    },
    'optimized': {
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
    'top_20_features': feature_importance.head(20).to_dict('records'),
    'pct_features_in_top_20': int(pct_in_top20)
}

with open(OUTPUT_DIR / 'metrics_with_proactivity.json', 'w') as f:
    json.dump(metrics, f, indent=2)

feature_importance.to_csv(OUTPUT_DIR / 'feature_importance.csv', index=False)

print(f"   ✓ Metrics: {OUTPUT_DIR / 'metrics_with_proactivity.json'}")
print(f"   ✓ Feature importance: {OUTPUT_DIR / 'feature_importance.csv'}")

print("\n" + "="*70)
print("PHASE 4.2 COMPLETE")
print("="*70)
print(f"\nKey Results:")
print(f"  ✅ Added {len(pct_features_in_model)} proactivity features")
print(f"  ✅ FAIL Recall: {best_recall:.1%}")
print(f"  ✅ Optimal Threshold: {best_threshold:.2f}")
print(f"  ✅ PCT features in top 20: {pct_in_top20}/20")

print(f"\n💡 Next: Phase 4.3 - Feature Selection (reduce {X.shape[1]} → 50 features)")
print("="*70)
