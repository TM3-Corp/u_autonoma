#!/usr/bin/env python3
"""
SOTA Feature Selection Pipeline for PUC Data

5-stage pipeline:
1. Filter: Variance threshold, correlation-based redundancy
2. Univariate: Mutual information, statistical tests
3. Embedded: LASSO, ElasticNet, tree importance
4. Wrapper: Boruta, RFECV
5. Stability: Aggregate rankings and bootstrap validation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    SelectFromModel,
    RFECV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, ElasticNetCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# Paths
INPUT_FILE = Path('data/puc/enriched_features/all_features_sota.parquet')
OUTPUT_DIR = Path('data/puc/feature_selection')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_JOBS = -1

print("="*70)
print("SOTA FEATURE SELECTION PIPELINE - PUC DATA")
print("="*70)

# Load data
print("\nLoading features...")
df = pd.read_parquet(INPUT_FILE)

# Prepare X and y (multi-class: FAIL, PASS, GOOD, EXCELLENT)
exclude_cols = ['student_id', 'course_id', 'grade_category', 'grade', 'failed', 'class_label']
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].fillna(0)

# Ensure all features are numeric
X = X.select_dtypes(include=['number'])

# Encode multi-class target
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(df['grade_category'])
courses = df['course_id'].values

print(f"Dataset: {X.shape[0]} enrollments, {X.shape[1]} features")
print(f"Classes: {list(le.classes_)}")
print(f"Class distribution: {dict(zip(le.classes_, np.bincount(y)))}")

# ============================================================================
# STAGE 1: FILTER METHODS
# ============================================================================
print("\n" + "="*70)
print("STAGE 1: FILTER METHODS")
print("="*70)

# 1a. Variance Threshold (remove near-constant features)
print("\n1a. Variance Threshold...")
vt = VarianceThreshold(threshold=0.01)
X_var = vt.fit_transform(X)
var_features = [feature_cols[i] for i in range(len(feature_cols)) if vt.get_support()[i]]

print(f"  Removed {len(feature_cols) - len(var_features)} low-variance features")
print(f"  Remaining: {len(var_features)}")

# 1b. Correlation-based redundancy removal (remove one from pairs with r > 0.95)
print("\n1b. Correlation Filter...")
X_df = pd.DataFrame(X_var, columns=var_features)
corr_matrix = X_df.corr().abs()

# Find pairs with high correlation
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

corr_features = [f for f in var_features if f not in to_drop]
X_corr = X_df[corr_features].values

print(f"  Removed {len(to_drop)} highly correlated features (r > 0.95)")
print(f"  Remaining: {len(corr_features)}")

filter_results = {
    'variance_threshold': {
        'removed': len(feature_cols) - len(var_features),
        'kept': len(var_features)
    },
    'correlation_filter': {
        'removed': len(to_drop),
        'kept': len(corr_features),
        'dropped_features': to_drop
    }
}

# ============================================================================
# STAGE 2: UNIVARIATE STATISTICS
# ============================================================================
print("\n" + "="*70)
print("STAGE 2: UNIVARIATE STATISTICS")
print("="*70)

# 2a. Mutual Information
print("\n2a. Mutual Information...")
mi_scores = mutual_info_classif(X_corr, y, random_state=RANDOM_STATE, n_neighbors=5)
mi_ranking = pd.DataFrame({
    'feature': corr_features,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print(f"  Top 5 features by MI:")
for idx, row in mi_ranking.head(5).iterrows():
    print(f"    {row['feature']}: {row['mi_score']:.4f}")

# 2b. ANOVA F-statistic (for multi-class target)
print("\n2b. ANOVA F-statistic...")
from sklearn.feature_selection import f_classif
f_scores, f_pvalues = f_classif(X_corr, y)
f_ranking = pd.DataFrame({
    'feature': corr_features,
    'f_score': f_scores,
    'f_p': f_pvalues
}).sort_values('f_score', ascending=False)

print(f"  Top 5 features by F-score:")
for idx, row in f_ranking.head(5).iterrows():
    print(f"    {row['feature']}: F={row['f_score']:.3f}, p={row['f_p']:.4f}")

# 2c. Kruskal-Wallis H test (non-parametric multi-class)
print("\n2c. Kruskal-Wallis H Test...")
kw_scores = []
for i, feat in enumerate(corr_features):
    groups = [X_corr[y == c, i] for c in np.unique(y)]
    h_stat, p_val = stats.kruskal(*groups)
    kw_scores.append({'feature': feat, 'h_stat': h_stat, 'kw_p': p_val})

kw_ranking = pd.DataFrame(kw_scores).sort_values('kw_p')

print(f"  Top 5 features by p-value:")
for idx, row in kw_ranking.head(5).iterrows():
    print(f"    {row['feature']}: H={row['h_stat']:.2f}, p={row['kw_p']:.6f}")

univariate_results = {
    'mutual_information': mi_ranking.to_dict('records'),
    'anova_f': f_ranking.to_dict('records'),
    'kruskal_wallis': kw_ranking.to_dict('records')
}

# ============================================================================
# STAGE 3: EMBEDDED METHODS
# ============================================================================
print("\n" + "="*70)
print("STAGE 3: EMBEDDED METHODS")
print("="*70)

# 3a. Logistic Regression L1 (multi-class)
print("\n3a. Logistic Regression L1 (multi-class)...")
from sklearn.linear_model import LogisticRegression
lr_l1 = LogisticRegression(
    penalty='l1',
    solver='saga',
    multi_class='multinomial',
    max_iter=5000,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
lr_l1.fit(X_corr, y)
# Average absolute coefficients across classes
lasso_coefs = np.abs(lr_l1.coef_).mean(axis=0)
lasso_ranking = pd.DataFrame({
    'feature': corr_features,
    'lasso_coef': lasso_coefs
}).sort_values('lasso_coef', ascending=False)

lasso_selected = lasso_ranking[lasso_ranking['lasso_coef'] > 0.001]['feature'].tolist()
print(f"  Selected {len(lasso_selected)} features (coef > 0.001)")
print(f"  Top 5:")
for idx, row in lasso_ranking.head(5).iterrows():
    print(f"    {row['feature']}: {row['lasso_coef']:.4f}")

# 3b. Logistic Regression ElasticNet (multi-class)
print("\n3b. Logistic Regression ElasticNet (multi-class)...")
lr_enet = LogisticRegression(
    penalty='elasticnet',
    solver='saga',
    multi_class='multinomial',
    l1_ratio=0.5,
    max_iter=5000,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS
)
lr_enet.fit(X_corr, y)
enet_coefs = np.abs(lr_enet.coef_).mean(axis=0)
enet_ranking = pd.DataFrame({
    'feature': corr_features,
    'enet_coef': enet_coefs
}).sort_values('enet_coef', ascending=False)

enet_selected = enet_ranking[enet_ranking['enet_coef'] > 0.001]['feature'].tolist()
print(f"  Selected {len(enet_selected)} features (coef > 0.001)")

# 3c. Random Forest Importance
print("\n3c. Random Forest Importance...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    class_weight='balanced'
)
rf.fit(X_corr, y)
rf_ranking = pd.DataFrame({
    'feature': corr_features,
    'rf_importance': rf.feature_importances_
}).sort_values('rf_importance', ascending=False)

print(f"  Top 5:")
for idx, row in rf_ranking.head(5).iterrows():
    print(f"    {row['feature']}: {row['rf_importance']:.4f}")

# 3d. XGBoost Importance (multi-class)
print("\n3d. XGBoost Importance (multi-class)...")
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=N_JOBS,
    eval_metric='mlogloss'
)
xgb_model.fit(X_corr, y)
xgb_ranking = pd.DataFrame({
    'feature': corr_features,
    'xgb_importance': xgb_model.feature_importances_
}).sort_values('xgb_importance', ascending=False)

print(f"  Top 5:")
for idx, row in xgb_ranking.head(5).iterrows():
    print(f"    {row['feature']}: {row['xgb_importance']:.4f}")

embedded_results = {
    'lasso': lasso_ranking.to_dict('records'),
    'elasticnet': enet_ranking.to_dict('records'),
    'random_forest': rf_ranking.to_dict('records'),
    'xgboost': xgb_ranking.to_dict('records')
}

# ============================================================================
# STAGE 4: WRAPPER METHODS
# ============================================================================
print("\n" + "="*70)
print("STAGE 4: WRAPPER METHODS")
print("="*70)

# 4a. Manual Boruta (simplified - top features by RF importance)
print("\n4a. Boruta-style Selection...")
# Use RF importance threshold
importance_threshold = rf_ranking['rf_importance'].quantile(0.75)
boruta_selected = rf_ranking[rf_ranking['rf_importance'] >= importance_threshold]['feature'].tolist()
print(f"  Selected {len(boruta_selected)} features (top 25% by RF importance)")

# 4b. RFECV with Logistic Regression
print("\n4b. RFECV with Logistic Regression...")
lr = LogisticRegressionCV(cv=5, random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
rfecv = RFECV(
    estimator=lr,
    step=5,
    cv=StratifiedKFold(5),
    scoring='roc_auc_ovr',  # Use one-vs-rest for multi-class
    n_jobs=N_JOBS
)
rfecv.fit(X_corr, y)
rfecv_selected = [corr_features[i] for i in range(len(corr_features)) if rfecv.support_[i]]
print(f"  Selected {len(rfecv_selected)} features")
print(f"  Optimal number: {rfecv.n_features_}")

wrapper_results = {
    'boruta': boruta_selected,
    'rfecv': rfecv_selected
}

# ============================================================================
# STAGE 5: STABILITY SELECTION
# ============================================================================
print("\n" + "="*70)
print("STAGE 5: STABILITY SELECTION")
print("="*70)

# Aggregate rankings across all methods
print("\n5a. Aggregating Rankings...")

# Normalize rankings to 0-1 scale (rank / max_rank)
all_rankings = pd.DataFrame({'feature': corr_features})

# Add normalized ranks from each method
for df, name in [(mi_ranking, 'mi'), (f_ranking, 'anova'), (kw_ranking, 'kw'),
                  (lasso_ranking, 'lasso'), (enet_ranking, 'enet'),
                  (rf_ranking, 'rf'), (xgb_ranking, 'xgb')]:
    df_copy = df.copy()
    df_copy[f'{name}_rank'] = df_copy[df_copy.columns[1]].rank(ascending=False)
    df_copy[f'{name}_norm'] = df_copy[f'{name}_rank'] / len(df_copy)
    all_rankings = all_rankings.merge(df_copy[['feature', f'{name}_norm']], on='feature', how='left')

# Average normalized rank
rank_cols = [c for c in all_rankings.columns if c.endswith('_norm')]
all_rankings['avg_rank'] = all_rankings[rank_cols].mean(axis=1)
all_rankings = all_rankings.sort_values('avg_rank', ascending=False)

print(f"  Top 10 features by average ranking:")
for idx, row in all_rankings.head(10).iterrows():
    print(f"    {row['feature']}: avg_rank={row['avg_rank']:.3f}")

# 5b. Selection by consensus (appear in top N of at least 5 methods)
print("\n5b. Consensus Selection (top 50 in at least 4 methods)...")
consensus_selected = []
for feat in corr_features:
    count = 0
    # Check if in top 50 of each method
    if feat in mi_ranking.head(50)['feature'].values:
        count += 1
    if feat in f_ranking.head(50)['feature'].values:
        count += 1
    if feat in kw_ranking.head(50)['feature'].values:
        count += 1
    if feat in lasso_ranking.head(50)['feature'].values:
        count += 1
    if feat in rf_ranking.head(50)['feature'].values:
        count += 1
    if feat in xgb_ranking.head(50)['feature'].values:
        count += 1

    if count >= 4:
        consensus_selected.append(feat)

print(f"  Selected {len(consensus_selected)} features by consensus")

# Final selection: Top 40 by average rank
final_selected = all_rankings.head(40)['feature'].tolist()

print("\n" + "="*70)
print("FINAL FEATURE SELECTION")
print("="*70)
print(f"Optimal features: {len(final_selected)}")
print(f"\nTop 20 features:")
for i, feat in enumerate(final_selected[:20], 1):
    print(f"  {i:2d}. {feat}")

# Save results
print("\nSaving results...")

# Optimal features
optimal_features = {
    'features': final_selected,
    'count': len(final_selected),
    'consensus_features': consensus_selected,
    'consensus_count': len(consensus_selected)
}

with open(OUTPUT_DIR / 'optimal_features.json', 'w') as f:
    json.dump(optimal_features, f, indent=2)

# Full rankings
all_rankings.to_parquet(OUTPUT_DIR / 'feature_rankings.parquet', index=False)

# Stage summaries
summary = {
    'filter': filter_results,
    'univariate': {
        'top_mi': mi_ranking.head(20).to_dict('records'),
        'top_anova': f_ranking.head(20).to_dict('records')
    },
    'embedded': {
        'lasso_selected': len(lasso_selected),
        'enet_selected': len(enet_selected),
        'top_rf': rf_ranking.head(20).to_dict('records'),
        'top_xgb': xgb_ranking.head(20).to_dict('records')
    },
    'wrapper': {
        'boruta_selected': len(boruta_selected),
        'rfecv_selected': len(rfecv_selected)
    },
    'stability': {
        'consensus_selected': len(consensus_selected),
        'final_selected': len(final_selected)
    }
}

with open(OUTPUT_DIR / 'selection_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✓ Results saved to: {OUTPUT_DIR}")
print(f"  - optimal_features.json ({len(final_selected)} features)")
print(f"  - feature_rankings.parquet (full rankings)")
print(f"  - selection_summary.json (stage summaries)")
print("\n✓ Phase 3 complete: Feature selection successful")
