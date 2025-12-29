#!/usr/bin/env python3
"""
Feature Agglomeration for Engagement Dynamics Features.

Reduces ~50 engagement features to 6-8 interpretable aggregate features
using Ward's hierarchical clustering (based on Oviedo et al. approach).

Also provides:
- PCA comparison
- Feature importance ranking
- Cluster interpretation/naming
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')


# Feature groups for interpretation
FEATURE_GROUPS = {
    'session_regularity': [
        'session_count', 'session_gap_min', 'session_gap_max', 'session_gap_mean',
        'session_gap_std', 'session_regularity', 'sessions_per_week'
    ],
    'time_blocks': [
        'weekday_morning_pct', 'weekday_afternoon_pct', 'weekday_evening_pct',
        'weekday_night_pct', 'weekend_morning_pct', 'weekend_afternoon_pct',
        'weekend_evening_pct', 'weekend_night_pct', 'weekday_morning_sd',
        'weekday_afternoon_sd', 'weekend_total_sd'
    ],
    'dct_coefficients': [
        'dct_coef_0', 'dct_coef_1', 'dct_coef_2', 'dct_coef_3', 'dct_coef_4',
        'dct_coef_5', 'dct_coef_6', 'dct_coef_7', 'dct_coef_8', 'dct_coef_9',
        'dct_coef_10', 'dct_coef_11'
    ],
    'engagement_trajectory': [
        'engagement_velocity', 'engagement_acceleration', 'weekly_cv',
        'trend_reversals', 'early_engagement_ratio', 'late_surge'
    ],
    'workload_dynamics': [
        'peak_count_type1', 'peak_count_type2', 'peak_count_type3', 'peak_ratio',
        'max_positive_slope', 'max_negative_slope', 'slope_std',
        'positive_slope_sum', 'negative_slope_sum', 'weekly_range'
    ],
    'time_to_access': [
        'first_access_day', 'first_module_day', 'first_assignment_day', 'access_time_pct'
    ],
    'raw_aggregates': [
        'total_page_views', 'total_participations', 'activity_span_days', 'unique_active_hours'
    ]
}


def load_student_features(filepath: str = 'data/engagement_dynamics/student_features.csv') -> pd.DataFrame:
    """Load student features from CSV."""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} students with {len(df.columns)} columns")
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding IDs, targets, normalized versions)."""
    exclude = ['course_id', 'user_id', 'user_role', 'final_score', 'failed']
    feature_cols = [c for c in df.columns
                   if c not in exclude
                   and not c.endswith('_norm')]
    return feature_cols


def prepare_feature_matrix(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
    """Prepare feature matrix, handling missing values."""
    X_df = df[feature_cols].copy()

    # Fill missing values with column median
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(X_df[col].median())

    # Replace infinities
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    for col in X_df.columns:
        X_df[col] = X_df[col].fillna(0)

    return X_df.values, X_df


def run_feature_agglomeration(
    X: np.ndarray,
    feature_names: List[str],
    n_clusters: int = 8
) -> Tuple[np.ndarray, Dict[int, List[str]]]:
    """Run Feature Agglomeration with Ward linkage."""
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Feature Agglomeration
    fa = FeatureAgglomeration(n_clusters=n_clusters, linkage='ward')
    X_reduced = fa.fit_transform(X_scaled)

    # Get feature-to-cluster mapping
    cluster_mapping = {}
    for i, label in enumerate(fa.labels_):
        if label not in cluster_mapping:
            cluster_mapping[label] = []
        cluster_mapping[label].append(feature_names[i])

    return X_reduced, cluster_mapping


def interpret_clusters(cluster_mapping: Dict[int, List[str]]) -> Dict[int, str]:
    """Assign interpretable names to clusters based on feature composition."""
    cluster_names = {}

    for cluster_id, features in cluster_mapping.items():
        # Count features from each group
        group_counts = {}
        for group_name, group_features in FEATURE_GROUPS.items():
            count = len([f for f in features if f in group_features])
            if count > 0:
                group_counts[group_name] = count

        if group_counts:
            # Name by dominant group
            dominant_group = max(group_counts.items(), key=lambda x: x[1])[0]
            cluster_names[cluster_id] = dominant_group.upper()
        else:
            cluster_names[cluster_id] = f'CLUSTER_{cluster_id}'

    return cluster_names


def compare_with_pca(X: np.ndarray, n_components: int = 8) -> Tuple[np.ndarray, float]:
    """Compare with PCA for same number of components."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    variance_explained = sum(pca.explained_variance_ratio_)
    return X_pca, variance_explained


def calculate_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str]
) -> pd.DataFrame:
    """Calculate feature importance using Random Forest."""
    # Remove NaN targets
    valid_mask = ~np.isnan(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    if len(y_valid) < 10:
        return pd.DataFrame()

    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_valid, y_valid)

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    return importance_df


def evaluate_prediction_performance(
    X_original: np.ndarray,
    X_reduced: np.ndarray,
    y: np.ndarray
) -> Dict[str, float]:
    """Compare prediction performance with original vs reduced features."""
    valid_mask = ~np.isnan(y)
    y_valid = y[valid_mask]
    X_orig_valid = X_original[valid_mask]
    X_red_valid = X_reduced[valid_mask]

    if len(y_valid) < 20:
        return {'original_r2': np.nan, 'reduced_r2': np.nan}

    rf_orig = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_red = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    try:
        scores_orig = cross_val_score(rf_orig, X_orig_valid, y_valid, cv=5, scoring='r2')
        scores_red = cross_val_score(rf_red, X_red_valid, y_valid, cv=5, scoring='r2')
    except Exception as e:
        print(f"Warning: Cross-validation failed: {e}")
        return {'original_r2': np.nan, 'reduced_r2': np.nan}

    return {
        'original_r2': np.mean(scores_orig),
        'reduced_r2': np.mean(scores_red),
        'original_r2_std': np.std(scores_orig),
        'reduced_r2_std': np.std(scores_red)
    }


def main():
    """Main feature agglomeration pipeline."""
    print("\n" + "=" * 70)
    print("FEATURE AGGLOMERATION ANALYSIS")
    print("=" * 70)

    # Load data
    df = load_student_features()
    feature_cols = get_feature_columns(df)
    print(f"Feature columns: {len(feature_cols)}")

    # Prepare feature matrix
    X, X_df = prepare_feature_matrix(df, feature_cols)
    y = df['final_score'].values

    # Run Feature Agglomeration with different cluster counts
    print("\n" + "-" * 50)
    print("Testing different cluster counts...")
    print("-" * 50)

    for n_clusters in [4, 6, 8, 10]:
        X_reduced, cluster_mapping = run_feature_agglomeration(X, feature_cols, n_clusters)
        X_pca, pca_variance = compare_with_pca(X, n_clusters)

        performance = evaluate_prediction_performance(X, X_reduced, y)

        print(f"\nn_clusters = {n_clusters}:")
        print(f"  FA Prediction R²: {performance['reduced_r2']:.3f} (±{performance.get('reduced_r2_std', 0):.3f})")
        print(f"  Original R²:      {performance['original_r2']:.3f} (±{performance.get('original_r2_std', 0):.3f})")
        print(f"  PCA variance explained: {pca_variance:.1%}")

    # Use n_clusters=8 as default (per plan)
    n_clusters = 8
    print(f"\n" + "=" * 70)
    print(f"DETAILED ANALYSIS WITH {n_clusters} CLUSTERS")
    print("=" * 70)

    X_reduced, cluster_mapping = run_feature_agglomeration(X, feature_cols, n_clusters)
    cluster_names = interpret_clusters(cluster_mapping)

    print("\nCluster Composition:")
    print("-" * 50)
    for cluster_id in sorted(cluster_mapping.keys()):
        features = cluster_mapping[cluster_id]
        name = cluster_names[cluster_id]
        print(f"\nCluster {cluster_id} ({name}): {len(features)} features")
        for feat in features[:5]:  # Show first 5
            print(f"  - {feat}")
        if len(features) > 5:
            print(f"  ... and {len(features) - 5} more")

    # Feature importance
    print("\n" + "-" * 50)
    print("Top 15 Most Important Features (Random Forest)")
    print("-" * 50)

    importance_df = calculate_feature_importance(X, y, feature_cols)
    if not importance_df.empty:
        print(f"\n{'Feature':<35} {'Importance':>12}")
        print("-" * 47)
        for _, row in importance_df.head(15).iterrows():
            print(f"{row['feature']:<35} {row['importance']:>12.4f}")

    # Create aggregated feature matrix
    print("\n" + "-" * 50)
    print("Creating Aggregated Feature DataFrame")
    print("-" * 50)

    # Add reduced features to dataframe
    for i in range(n_clusters):
        col_name = f"agg_{cluster_names[i].lower()}"
        df[col_name] = X_reduced[:, i]

    agg_cols = [c for c in df.columns if c.startswith('agg_')]
    print(f"Added {len(agg_cols)} aggregated features: {agg_cols}")

    # Save results
    output_dir = 'data/engagement_dynamics'

    # Save aggregated features
    output_cols = ['course_id', 'user_id', 'final_score', 'failed'] + agg_cols
    df[output_cols].to_csv(f'{output_dir}/aggregated_features.csv', index=False)
    print(f"\nSaved aggregated features to {output_dir}/aggregated_features.csv")

    # Save cluster mapping
    cluster_info = {
        'n_clusters': n_clusters,
        'cluster_mapping': {str(k): v for k, v in cluster_mapping.items()},
        'cluster_names': {str(k): v for k, v in cluster_names.items()}
    }
    with open(f'{output_dir}/cluster_mapping.json', 'w') as f:
        json.dump(cluster_info, f, indent=2)
    print(f"Saved cluster mapping to {output_dir}/cluster_mapping.json")

    # Save feature importance
    if not importance_df.empty:
        importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False)
        print(f"Saved feature importance to {output_dir}/feature_importance.csv")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Original features: {len(feature_cols)}")
    print(f"Aggregated features: {n_clusters}")
    print(f"Dimensionality reduction: {len(feature_cols)} -> {n_clusters} ({100 * n_clusters / len(feature_cols):.1f}%)")

    performance = evaluate_prediction_performance(X, X_reduced, y)
    r2_retention = performance['reduced_r2'] / performance['original_r2'] if performance['original_r2'] > 0 else 0
    print(f"Prediction performance retention: {r2_retention:.1%}")

    return df, X_reduced, cluster_mapping


if __name__ == '__main__':
    main()
