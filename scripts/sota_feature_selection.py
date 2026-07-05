#!/usr/bin/env python3
"""
SOTA Feature Selection Pipeline

Implements a multi-stage iterative feature selection process to address
the curse of dimensionality (280 features, 363 samples).

Stages:
1. Filter: Variance threshold, correlation-based redundancy removal
2. Univariate: Mutual information, statistical tests
3. Embedded: LASSO, ElasticNet, tree-based importance
4. Wrapper: Manual Boruta, RFECV
5. Stability: Aggregate rankings across methods and bootstrap samples

Key considerations:
- Leave-One-Course-Out (LOCO) validation (students within course not independent)
- Binary target: failed (grade < 57%)
- Focus on which new course-relative features survive selection
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    SelectFromModel,
    RFE,
    RFECV
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb
import shap

# Paths
DATA_DIR = Path('/home/paul/projects/uautonoma/data')
ENRICHED_DIR = DATA_DIR / 'enriched_features'
OUTPUT_DIR = DATA_DIR / 'feature_selection'
OUTPUT_DIR.mkdir(exist_ok=True)

# Constants
RANDOM_STATE = 42
N_JOBS = -1
FAILURE_THRESHOLD = 57.0  # Grade below this = failed

# New course-relative feature patterns (to track separately)
COURSE_RELATIVE_PATTERNS = [
    'first_access_pct', 'last_access_pct', 'activity_span_pct',
    'median_activity_pct', 'activity_std_pct',
    'early_10_', 'early_20_', 'early_33_',
    '_timing_hist_', '_mean_access_pct', '_median_access_pct', '_std_access_pct',
    '_early_access_rate', 'activity_bin_', 'curve_slope', 'curve_trend',
    'session_gap_cv', 'session_gap_median_hours', 'session_gap_max_days',
    'longest_inactive_period_pct'
]


def is_course_relative_feature(col):
    """Check if feature is one of our new course-relative features."""
    return any(pattern in col for pattern in COURSE_RELATIVE_PATTERNS)


def load_data():
    """Load features and create target variable."""
    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    # Load normalized features
    df = pd.read_parquet(ENRICHED_DIR / 'normalized_features.parquet')

    # Load grades from model courses enrollments
    enrollments_path = DATA_DIR / 'model_courses_enrollments.json'
    with open(enrollments_path) as f:
        enrollments_data = json.load(f)

    # Extract grades into a lookup
    grade_lookup = {}
    for enrollment in enrollments_data:
        if enrollment.get('type') != 'StudentEnrollment':
            continue
        user_id = enrollment.get('user_id')
        course_id = enrollment.get('course_id')
        grades = enrollment.get('grades', {})
        # Prefer final_score, fallback to current_score
        grade = grades.get('final_score') or grades.get('current_score')
        if grade is not None and user_id is not None and course_id is not None:
            grade_lookup[(user_id, course_id)] = grade

    # Add target variable
    df['grade'] = df.apply(
        lambda row: grade_lookup.get((row['user_id'], row['course_id']), np.nan),
        axis=1
    )
    df['failed'] = (df['grade'] < FAILURE_THRESHOLD).astype(int)

    # Remove rows without grades
    df_valid = df.dropna(subset=['grade'])

    print(f"Total samples: {len(df_valid)}")
    print(f"Failed: {df_valid['failed'].sum()} ({df_valid['failed'].mean()*100:.1f}%)")
    print(f"Passed: {(1-df_valid['failed']).sum()} ({(1-df_valid['failed']).mean()*100:.1f}%)")
    print(f"Courses: {df_valid['course_id'].nunique()}")

    # Separate features and target
    exclude_cols = ['user_id', 'course_id', 'grade', 'failed']
    feature_cols = [c for c in df_valid.columns if c not in exclude_cols]

    X = df_valid[feature_cols].copy()
    y = df_valid['failed'].values
    course_ids = df_valid['course_id'].values

    # Track which features are course-relative
    course_rel_features = [c for c in feature_cols if is_course_relative_feature(c)]
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Course-relative features: {len(course_rel_features)}")

    return X, y, course_ids, feature_cols, course_rel_features


def stage1_filter(X, feature_cols, variance_threshold=0.01, corr_threshold=0.95):
    """
    Stage 1: Filter Methods
    - Remove near-zero variance features
    - Remove highly correlated features (keep one from each cluster)
    """
    print("\n" + "=" * 70)
    print("STAGE 1: FILTER METHODS")
    print("=" * 70)

    results = {'removed': [], 'kept': []}

    # 1a. Variance threshold
    print(f"\n1a. Variance Threshold (>{variance_threshold})")

    # Handle NaN by filling with median
    X_filled = X.fillna(X.median())

    # Normalize for variance calculation
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_filled),
        columns=X.columns,
        index=X.index
    )

    variances = X_scaled.var()
    low_var_features = variances[variances < variance_threshold].index.tolist()
    print(f"   Low variance features: {len(low_var_features)}")
    for f in low_var_features[:5]:
        print(f"     - {f}: var={variances[f]:.6f}")
    if len(low_var_features) > 5:
        print(f"     ... and {len(low_var_features)-5} more")

    results['low_variance'] = low_var_features

    # Remove low variance features
    X_filtered = X_filled.drop(columns=low_var_features)

    # 1b. Correlation-based redundancy removal
    print(f"\n1b. Correlation-based Redundancy (threshold={corr_threshold})")

    corr_matrix = X_filtered.corr().abs()

    # Hierarchical clustering on correlation matrix
    # Convert correlation to distance
    dissimilarity = 1 - corr_matrix
    np.fill_diagonal(dissimilarity.values, 0)

    # Find highly correlated pairs
    high_corr_pairs = []
    upper_tri = np.triu_indices_from(corr_matrix, k=1)
    for i, j in zip(*upper_tri):
        if corr_matrix.iloc[i, j] > corr_threshold:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

    print(f"   Highly correlated pairs (>{corr_threshold}): {len(high_corr_pairs)}")

    # Remove one from each correlated pair (keep the one with higher variance)
    to_remove = set()
    for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: -x[2]):
        if f1 not in to_remove and f2 not in to_remove:
            # Keep the one with higher variance
            if variances.get(f1, 0) >= variances.get(f2, 0):
                to_remove.add(f2)
            else:
                to_remove.add(f1)

    print(f"   Removing {len(to_remove)} redundant features")
    for f in list(to_remove)[:5]:
        print(f"     - {f}")
    if len(to_remove) > 5:
        print(f"     ... and {len(to_remove)-5} more")

    results['high_correlation'] = list(to_remove)

    # Final filtered features
    X_final = X_filtered.drop(columns=list(to_remove))
    results['kept'] = X_final.columns.tolist()

    print(f"\n   Features after Stage 1: {len(results['kept'])} (removed {len(low_var_features) + len(to_remove)})")

    return X_final, results


def stage2_univariate(X, y, n_top=100):
    """
    Stage 2: Univariate Feature Importance
    - Mutual Information
    - Point-biserial correlation
    - Mann-Whitney U test
    """
    print("\n" + "=" * 70)
    print("STAGE 2: UNIVARIATE METHODS")
    print("=" * 70)

    X_filled = X.fillna(X.median())
    results = {}

    # 2a. Mutual Information
    print("\n2a. Mutual Information")
    mi_scores = mutual_info_classif(X_filled, y, random_state=RANDOM_STATE, n_neighbors=5)
    mi_df = pd.DataFrame({
        'feature': X.columns,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    results['mutual_info'] = mi_df.set_index('feature')['mi_score'].to_dict()

    print(f"   Top 10 by Mutual Information:")
    for _, row in mi_df.head(10).iterrows():
        marker = "★" if is_course_relative_feature(row['feature']) else " "
        print(f"   {marker} {row['feature']}: {row['mi_score']:.4f}")

    # 2b. Point-biserial correlation (for continuous features vs binary target)
    print("\n2b. Point-biserial Correlation")
    pb_scores = {}
    for col in X.columns:
        try:
            corr, pval = stats.pointbiserialr(y, X_filled[col])
            pb_scores[col] = {'corr': abs(corr), 'pval': pval}
        except:
            pb_scores[col] = {'corr': 0, 'pval': 1}

    pb_df = pd.DataFrame(pb_scores).T
    pb_df = pb_df.sort_values('corr', ascending=False)
    results['point_biserial'] = pb_df['corr'].to_dict()

    print(f"   Top 10 by |correlation|:")
    for feat, row in pb_df.head(10).iterrows():
        marker = "★" if is_course_relative_feature(feat) else " "
        sig = "*" if row['pval'] < 0.05 else ""
        print(f"   {marker} {feat}: r={row['corr']:.4f}{sig}")

    # 2c. Mann-Whitney U test
    print("\n2c. Mann-Whitney U Test")
    mw_scores = {}
    for col in X.columns:
        try:
            group0 = X_filled.loc[y == 0, col]
            group1 = X_filled.loc[y == 1, col]
            stat, pval = stats.mannwhitneyu(group0, group1, alternative='two-sided')
            # Effect size (rank-biserial correlation)
            n1, n2 = len(group0), len(group1)
            effect_size = 1 - (2 * stat) / (n1 * n2)
            mw_scores[col] = {'effect_size': abs(effect_size), 'pval': pval}
        except:
            mw_scores[col] = {'effect_size': 0, 'pval': 1}

    mw_df = pd.DataFrame(mw_scores).T
    mw_df = mw_df.sort_values('effect_size', ascending=False)
    results['mann_whitney'] = mw_df['effect_size'].to_dict()

    print(f"   Top 10 by effect size:")
    for feat, row in mw_df.head(10).iterrows():
        marker = "★" if is_course_relative_feature(feat) else " "
        sig = "*" if row['pval'] < 0.05 else ""
        print(f"   {marker} {feat}: effect={row['effect_size']:.4f}{sig}")

    # Aggregate univariate rankings
    print("\n2d. Aggregated Univariate Ranking")

    # Rank each method (lower = better)
    rank_mi = mi_df.reset_index(drop=True).reset_index().set_index('feature')['index']
    rank_pb = pb_df.reset_index().reset_index().set_index('index')['level_0']
    rank_pb.index = pb_df.index
    rank_mw = mw_df.reset_index().reset_index().set_index('index')['level_0']
    rank_mw.index = mw_df.index

    # Average rank
    avg_rank = pd.DataFrame({
        'mi_rank': rank_mi,
        'pb_rank': rank_pb,
        'mw_rank': rank_mw
    })
    avg_rank['mean_rank'] = avg_rank.mean(axis=1)
    avg_rank = avg_rank.sort_values('mean_rank')

    results['univariate_ranking'] = avg_rank['mean_rank'].to_dict()

    print(f"   Top 20 by average rank:")
    for feat in avg_rank.head(20).index:
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: rank={avg_rank.loc[feat, 'mean_rank']:.1f}")

    # Return top features
    top_features = avg_rank.head(n_top).index.tolist()
    results['top_univariate'] = top_features

    return results


def stage3_embedded(X, y, n_top=100):
    """
    Stage 3: Embedded Methods
    - LASSO (L1 regularization)
    - Elastic Net
    - Random Forest importance
    - XGBoost importance
    """
    print("\n" + "=" * 70)
    print("STAGE 3: EMBEDDED METHODS")
    print("=" * 70)

    X_filled = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

    results = {}

    # 3a. LASSO (Logistic Regression with L1)
    print("\n3a. LASSO (L1 Logistic Regression)")
    lasso = LogisticRegressionCV(
        penalty='l1',
        solver='saga',
        cv=5,
        random_state=RANDOM_STATE,
        max_iter=1000,
        n_jobs=N_JOBS
    )
    lasso.fit(X_scaled, y)

    lasso_coef = pd.Series(np.abs(lasso.coef_[0]), index=X.columns)
    lasso_coef = lasso_coef.sort_values(ascending=False)
    n_nonzero = (lasso_coef > 0).sum()

    results['lasso'] = lasso_coef.to_dict()
    print(f"   Non-zero coefficients: {n_nonzero}")
    print(f"   Top 10:")
    for feat, coef in lasso_coef.head(10).items():
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: {coef:.4f}")

    # 3b. Elastic Net
    print("\n3b. Elastic Net (L1+L2)")
    enet = LogisticRegressionCV(
        penalty='elasticnet',
        solver='saga',
        cv=5,
        l1_ratios=[0.5],
        random_state=RANDOM_STATE,
        max_iter=1000,
        n_jobs=N_JOBS
    )
    enet.fit(X_scaled, y)

    enet_coef = pd.Series(np.abs(enet.coef_[0]), index=X.columns)
    enet_coef = enet_coef.sort_values(ascending=False)

    results['elastic_net'] = enet_coef.to_dict()
    print(f"   Non-zero coefficients: {(enet_coef > 0).sum()}")
    print(f"   Top 10:")
    for feat, coef in enet_coef.head(10).items():
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: {coef:.4f}")

    # 3c. Random Forest
    print("\n3c. Random Forest Importance")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    rf.fit(X_filled, y)

    rf_importance = pd.Series(rf.feature_importances_, index=X.columns)
    rf_importance = rf_importance.sort_values(ascending=False)

    results['random_forest'] = rf_importance.to_dict()
    print(f"   Top 10:")
    for feat, imp in rf_importance.head(10).items():
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: {imp:.4f}")

    # 3d. XGBoost
    print("\n3d. XGBoost Importance")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb_model.fit(X_filled, y)

    xgb_importance = pd.Series(xgb_model.feature_importances_, index=X.columns)
    xgb_importance = xgb_importance.sort_values(ascending=False)

    results['xgboost'] = xgb_importance.to_dict()
    print(f"   Top 10:")
    for feat, imp in xgb_importance.head(10).items():
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: {imp:.4f}")

    # Aggregate embedded rankings
    print("\n3e. Aggregated Embedded Ranking")

    embedded_rank = pd.DataFrame({
        'lasso': lasso_coef.rank(ascending=False),
        'enet': enet_coef.rank(ascending=False),
        'rf': rf_importance.rank(ascending=False),
        'xgb': xgb_importance.rank(ascending=False)
    })
    embedded_rank['mean_rank'] = embedded_rank.mean(axis=1)
    embedded_rank = embedded_rank.sort_values('mean_rank')

    results['embedded_ranking'] = embedded_rank['mean_rank'].to_dict()

    print(f"   Top 20 by average rank:")
    for feat in embedded_rank.head(20).index:
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: rank={embedded_rank.loc[feat, 'mean_rank']:.1f}")

    # Return top features
    top_features = embedded_rank.head(n_top).index.tolist()
    results['top_embedded'] = top_features

    return results, rf, xgb_model


def stage4_wrapper(X, y, rf_model, n_features_to_select=50):
    """
    Stage 4: Wrapper Methods
    - Manual Boruta implementation
    - RFECV (Recursive Feature Elimination with CV)
    """
    print("\n" + "=" * 70)
    print("STAGE 4: WRAPPER METHODS")
    print("=" * 70)

    X_filled = X.fillna(X.median())
    results = {}

    # 4a. Manual Boruta Implementation
    print("\n4a. Boruta Algorithm (manual implementation)")
    print("   Creating shadow features and comparing importance...")

    n_iterations = 20
    confirmed = set()
    rejected = set()
    tentative = set(X.columns)

    for iteration in range(n_iterations):
        if not tentative:
            break

        # Create shadow features (shuffled copies)
        X_shadow = X_filled[list(tentative)].apply(
            lambda col: np.random.permutation(col), axis=0
        )
        X_shadow.columns = [f'shadow_{c}' for c in tentative]

        # Combine original and shadow
        X_combined = pd.concat([X_filled[list(tentative)], X_shadow], axis=1)

        # Fit Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            random_state=RANDOM_STATE + iteration,
            n_jobs=N_JOBS
        )
        rf.fit(X_combined, y)

        # Get importances
        importances = pd.Series(rf.feature_importances_, index=X_combined.columns)
        shadow_importances = importances[[c for c in importances.index if c.startswith('shadow_')]]
        max_shadow = shadow_importances.max()

        # Compare original features to max shadow
        for feat in list(tentative):
            if importances[feat] > max_shadow:
                confirmed.add(feat)
                tentative.discard(feat)
            elif importances[feat] < shadow_importances.min():
                rejected.add(feat)
                tentative.discard(feat)

    # Remaining tentative features after max iterations
    confirmed.update(tentative)  # Be conservative, keep tentative

    results['boruta_confirmed'] = list(confirmed)
    results['boruta_rejected'] = list(rejected)

    print(f"   Confirmed: {len(confirmed)}")
    print(f"   Rejected: {len(rejected)}")
    print(f"   Top confirmed features:")

    # Rank confirmed features by RF importance
    if confirmed:
        rf_final = RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=RANDOM_STATE, n_jobs=N_JOBS
        )
        rf_final.fit(X_filled[list(confirmed)], y)
        boruta_importance = pd.Series(rf_final.feature_importances_, index=list(confirmed))
        boruta_importance = boruta_importance.sort_values(ascending=False)

        for feat, imp in boruta_importance.head(15).items():
            marker = "★" if is_course_relative_feature(feat) else " "
            print(f"   {marker} {feat}: {imp:.4f}")

    # 4b. RFECV
    print("\n4b. Recursive Feature Elimination with CV (RFECV)")
    print("   Finding optimal number of features...")

    # Use RF as base estimator
    rf_rfe = RandomForestClassifier(
        n_estimators=100,
        max_depth=7,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )

    # RFECV with stratified 5-fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    rfecv = RFECV(
        estimator=rf_rfe,
        step=10,  # Remove 10 features at a time
        cv=cv,
        scoring='roc_auc',
        min_features_to_select=10,
        n_jobs=N_JOBS
    )
    rfecv.fit(X_filled, y)

    optimal_n = rfecv.n_features_
    selected_features = X.columns[rfecv.support_].tolist()

    results['rfecv_optimal_n'] = optimal_n
    results['rfecv_selected'] = selected_features
    results['rfecv_cv_scores'] = rfecv.cv_results_['mean_test_score'].tolist()

    print(f"   Optimal number of features: {optimal_n}")
    print(f"   Best CV AUC: {rfecv.cv_results_['mean_test_score'].max():.4f}")
    print(f"   Selected features:")
    for feat in selected_features[:15]:
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}")
    if len(selected_features) > 15:
        print(f"   ... and {len(selected_features)-15} more")

    return results


def stage5_stability(X, y, course_ids, n_bootstrap=30):
    """
    Stage 5: Stability Selection
    - Run feature selection multiple times with bootstrap samples
    - Track selection frequency
    - Use LOCO (Leave-One-Course-Out) as well
    """
    print("\n" + "=" * 70)
    print("STAGE 5: STABILITY SELECTION")
    print("=" * 70)

    X_filled = X.fillna(X.median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filled)

    selection_counts = pd.Series(0, index=X.columns)

    # 5a. Bootstrap stability
    print(f"\n5a. Bootstrap Stability ({n_bootstrap} iterations)")

    for i in range(n_bootstrap):
        # Bootstrap sample
        idx = np.random.choice(len(X), size=len(X), replace=True)
        X_boot = X_scaled[idx]
        y_boot = y[idx]

        # LASSO selection
        lasso = LogisticRegressionCV(
            penalty='l1', solver='saga', cv=3,
            random_state=RANDOM_STATE+i, max_iter=500, n_jobs=N_JOBS
        )
        try:
            lasso.fit(X_boot, y_boot)
            selected = np.abs(lasso.coef_[0]) > 0
            selection_counts[selected] += 1
        except:
            pass

        if (i+1) % 10 == 0:
            print(f"   Completed {i+1}/{n_bootstrap} iterations")

    # Calculate selection probability
    selection_prob = selection_counts / n_bootstrap
    selection_prob = selection_prob.sort_values(ascending=False)

    print(f"\n   Features selected in >50% of bootstraps:")
    stable_features = selection_prob[selection_prob > 0.5]
    for feat, prob in stable_features.head(20).items():
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: {prob*100:.1f}%")

    # 5b. LOCO stability
    print(f"\n5b. Leave-One-Course-Out Stability")

    loco_counts = pd.Series(0, index=X.columns)
    unique_courses = np.unique(course_ids)

    for course in unique_courses:
        # Leave out one course
        mask = course_ids != course
        X_train = X_scaled[mask]
        y_train = y[mask]

        # LASSO selection
        lasso = LogisticRegressionCV(
            penalty='l1', solver='saga', cv=3,
            random_state=RANDOM_STATE, max_iter=500, n_jobs=N_JOBS
        )
        try:
            lasso.fit(X_train, y_train)
            selected = np.abs(lasso.coef_[0]) > 0
            loco_counts[selected] += 1
        except:
            pass

    loco_prob = loco_counts / len(unique_courses)
    loco_prob = loco_prob.sort_values(ascending=False)

    print(f"\n   Features selected in >50% of LOCO folds:")
    loco_stable = loco_prob[loco_prob > 0.5]
    for feat, prob in loco_stable.head(20).items():
        marker = "★" if is_course_relative_feature(feat) else " "
        print(f"   {marker} {feat}: {prob*100:.1f}%")

    results = {
        'bootstrap_selection_prob': selection_prob.to_dict(),
        'loco_selection_prob': loco_prob.to_dict(),
        'stable_bootstrap': stable_features.index.tolist(),
        'stable_loco': loco_stable.index.tolist()
    }

    return results


def aggregate_and_select(stage2_results, stage3_results, stage4_results, stage5_results,
                         feature_cols, course_rel_features, max_features=50):
    """
    Aggregate all rankings and select final feature set.
    """
    print("\n" + "=" * 70)
    print("FINAL AGGREGATION AND SELECTION")
    print("=" * 70)

    # Create master ranking dataframe
    all_features = feature_cols
    master_df = pd.DataFrame(index=all_features)

    # Add univariate rank
    master_df['univariate_rank'] = pd.Series(stage2_results['univariate_ranking'])

    # Add embedded rank
    master_df['embedded_rank'] = pd.Series(stage3_results['embedded_ranking'])

    # Add Boruta (1 if confirmed, 0 otherwise)
    boruta_confirmed = set(stage4_results.get('boruta_confirmed', []))
    master_df['boruta_confirmed'] = master_df.index.isin(boruta_confirmed).astype(int)

    # Add RFECV (1 if selected, 0 otherwise)
    rfecv_selected = set(stage4_results.get('rfecv_selected', []))
    master_df['rfecv_selected'] = master_df.index.isin(rfecv_selected).astype(int)

    # Add stability scores
    master_df['bootstrap_stability'] = pd.Series(stage5_results['bootstrap_selection_prob'])
    master_df['loco_stability'] = pd.Series(stage5_results['loco_selection_prob'])

    # Fill NaN
    master_df = master_df.fillna(master_df.median())

    # Normalize ranks to 0-1 (invert so higher = better)
    max_rank = master_df[['univariate_rank', 'embedded_rank']].max().max()
    master_df['univariate_score'] = 1 - (master_df['univariate_rank'] / max_rank)
    master_df['embedded_score'] = 1 - (master_df['embedded_rank'] / max_rank)

    # Calculate composite score
    # Weight: univariate (0.15), embedded (0.25), boruta (0.15), rfecv (0.15), stability (0.30)
    master_df['composite_score'] = (
        0.15 * master_df['univariate_score'] +
        0.25 * master_df['embedded_score'] +
        0.15 * master_df['boruta_confirmed'] +
        0.15 * master_df['rfecv_selected'] +
        0.15 * master_df['bootstrap_stability'] +
        0.15 * master_df['loco_stability']
    )

    # Sort by composite score
    master_df = master_df.sort_values('composite_score', ascending=False)

    # Mark course-relative features
    master_df['is_course_relative'] = master_df.index.map(
        lambda x: x in course_rel_features
    )

    # Select top features
    selected_features = master_df.head(max_features).index.tolist()

    # Summary statistics
    n_course_rel_selected = sum(f in course_rel_features for f in selected_features)
    n_course_rel_total = len(course_rel_features)

    print(f"\nFinal Selection: {len(selected_features)} features")
    print(f"Course-relative features in selection: {n_course_rel_selected}/{n_course_rel_total}")
    print(f"\nTop 30 selected features (★ = course-relative):")

    for i, feat in enumerate(selected_features[:30], 1):
        marker = "★" if feat in course_rel_features else " "
        score = master_df.loc[feat, 'composite_score']
        print(f"  {i:2d}. {marker} {feat}: score={score:.4f}")

    # Show course-relative feature performance
    print(f"\n" + "-" * 50)
    print("COURSE-RELATIVE FEATURE ANALYSIS")
    print("-" * 50)

    cr_df = master_df[master_df['is_course_relative']].copy()
    cr_df = cr_df.sort_values('composite_score', ascending=False)

    print(f"\nTop 20 course-relative features:")
    for feat in cr_df.head(20).index:
        score = cr_df.loc[feat, 'composite_score']
        in_top50 = "✓" if feat in selected_features else " "
        print(f"  {in_top50} {feat}: {score:.4f}")

    # Which course-relative features made it?
    cr_selected = [f for f in selected_features if f in course_rel_features]
    print(f"\n{len(cr_selected)} course-relative features in final selection:")
    for f in cr_selected:
        print(f"  ★ {f}")

    return master_df, selected_features


def evaluate_selection(X, y, selected_features, course_ids):
    """
    Evaluate selected feature set with LOCO cross-validation.
    """
    print("\n" + "=" * 70)
    print("EVALUATION: LOCO Cross-Validation")
    print("=" * 70)

    X_selected = X[selected_features].fillna(X[selected_features].median())

    # XGBoost with selected features
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # LOCO CV
    unique_courses = np.unique(course_ids)
    auc_scores = []

    for course in unique_courses:
        train_mask = course_ids != course
        test_mask = course_ids == course

        X_train = X_selected.loc[train_mask]
        y_train = y[train_mask]
        X_test = X_selected.loc[test_mask]
        y_test = y[test_mask]

        if len(np.unique(y_test)) < 2:
            continue

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_test, y_pred_proba)
            auc_scores.append(auc)
            print(f"  Course {course}: AUC = {auc:.4f} (n={len(y_test)}, fail_rate={y_test.mean():.2f})")
        except:
            pass

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)

    print(f"\nLOCO CV Results:")
    print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"  Min AUC: {min(auc_scores):.4f}")
    print(f"  Max AUC: {max(auc_scores):.4f}")

    return {'mean_auc': mean_auc, 'std_auc': std_auc, 'course_aucs': auc_scores}


def main():
    print("=" * 70)
    print("SOTA FEATURE SELECTION PIPELINE")
    print("Addressing curse of dimensionality: 280 features, 363 samples")
    print("=" * 70)

    # Load data
    X, y, course_ids, feature_cols, course_rel_features = load_data()

    # Stage 1: Filter
    X_filtered, stage1_results = stage1_filter(X, feature_cols)

    # Stage 2: Univariate
    stage2_results = stage2_univariate(X_filtered, y)

    # Stage 3: Embedded
    stage3_results, rf_model, xgb_model = stage3_embedded(X_filtered, y)

    # Stage 4: Wrapper
    stage4_results = stage4_wrapper(X_filtered, y, rf_model)

    # Stage 5: Stability
    stage5_results = stage5_stability(X_filtered, y, course_ids)

    # Final aggregation
    master_df, selected_features = aggregate_and_select(
        stage2_results, stage3_results, stage4_results, stage5_results,
        X_filtered.columns.tolist(), course_rel_features,
        max_features=50
    )

    # Evaluate
    eval_results = evaluate_selection(X_filtered, y, selected_features, course_ids)

    # Save results
    output = {
        'n_original_features': int(len(feature_cols)),
        'n_after_filter': int(len(X_filtered.columns)),
        'n_selected': int(len(selected_features)),
        'selected_features': selected_features,
        'n_course_relative_in_selection': int(sum(f in course_rel_features for f in selected_features)),
        'course_relative_selected': [f for f in selected_features if f in course_rel_features],
        'stage1_removed': stage1_results['low_variance'] + stage1_results['high_correlation'],
        'rfecv_optimal_n': int(stage4_results.get('rfecv_optimal_n', 0)),
        'loco_mean_auc': float(eval_results['mean_auc']),
        'loco_std_auc': float(eval_results['std_auc'])
    }

    # Save selected features list
    with open(OUTPUT_DIR / 'selected_features.json', 'w') as f:
        json.dump(output, f, indent=2)

    # Save master ranking
    master_df.to_parquet(OUTPUT_DIR / 'feature_rankings.parquet')

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - selected_features.json")
    print(f"  - feature_rankings.parquet")

    return output


if __name__ == '__main__':
    output = main()
