#!/usr/bin/env python3
"""
Train early warning models with time-limited data.

This script trains models using features calculated from different
time cutoffs (2, 4, 6, 8 weeks) and compares performance with/without
assessment features.

Output:
    data/analysis/time_cutoff_results.json
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/analysis/time_cutoff_results.json"

# Cutoffs to evaluate
CUTOFFS = [2, 4, 6, 8, 'full']

# Assessment-related feature patterns (to exclude when testing without)
EXCLUDE_PATTERNS = [
    'quiz', 'quizzes',
    'assi', 'assignment',
    'grade', 'grad',
    'score',
    'submission',
]

# XGBoost hyperparameters (same as optimized model)
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

# Feature selection parameters (same as optimized model)
FEATURE_SELECTION_ENABLED = True
FEATURE_IMPORTANCE_THRESHOLD = 0.005  # Remove features with importance < 0.5%
CORRELATION_THRESHOLD = 0.85  # Remove highly correlated features


def calculate_znorm_features(df):
    """Calculate z-score normalized features per course (on-the-fly)."""
    # Get numeric columns (excluding identifiers)
    exclude_cols = ['user_id', 'course_id', 'enrollment_state', 'failed', 'final_score']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.endswith('_znorm')]

    # Calculate z-scores per course
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


def load_features(cutoff, include_znorm=True):
    """Load all features for a given cutoff.

    Args:
        cutoff: Time cutoff (2, 4, 6, 8, or 'full')
        include_znorm: Whether to include z-score normalized features
    """
    if cutoff == 'full':
        feature_dir = BASE_DIR / "data/enriched_features"
        feature_files = [
            'session_features.parquet',
            'category_features.parquet',
            'proactivity_features.parquet',
            'pca_features.parquet',
            'weekly_features.parquet',
            'ngram_features.parquet',
            'graph_features.parquet',
            'time_features.parquet',
        ]
        # Add normalized features for full data (to match baseline 0.860)
        if include_znorm:
            feature_files.append('normalized_features.parquet')
    else:
        feature_dir = BASE_DIR / f"data/enriched_features/cutoff_week_{cutoff}"
        feature_files = list(feature_dir.glob("*_features.parquet"))
        feature_files = [f.name for f in feature_files]

    # Load and merge
    dfs = []
    for fname in feature_files:
        fpath = feature_dir / fname
        if fpath.exists():
            df = pd.read_parquet(fpath)
            # For normalized_features, only keep the _znorm columns
            if fname == 'normalized_features.parquet':
                znorm_cols = [c for c in df.columns if c.endswith('_znorm')]
                df = df[['user_id', 'course_id'] + znorm_cols]
                print(f"    Loaded {len(znorm_cols)} z-normalized features")
            dfs.append(df)

    if not dfs:
        return None

    # Merge on user_id, course_id
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_dup'))
        # Remove duplicate columns
        dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
        df_merged = df_merged.drop(columns=dup_cols)

    # For temporal cutoffs, calculate z-norm on the fly if requested
    if cutoff != 'full' and include_znorm:
        print(f"    Calculating z-norm features on-the-fly...")
        df_merged = calculate_znorm_features(df_merged)
        znorm_count = len([c for c in df_merged.columns if c.endswith('_znorm')])
        print(f"    Added {znorm_count} calculated z-normalized features")

    return df_merged


def load_enrollments():
    """Load enrollment data with target variable."""
    df = pd.read_csv(ENROLLMENTS_FILE)
    # Use same threshold as optimized model: 57% (Chilean 4.0/7.0 scale)
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


def filter_assessment_features(columns):
    """Filter out assessment-related features."""
    filtered = []
    for col in columns:
        col_lower = col.lower()
        is_excluded = any(pattern in col_lower for pattern in EXCLUDE_PATTERNS)
        if not is_excluded:
            filtered.append(col)
    return filtered


def select_features_by_importance(X, y, feature_cols, threshold=0.005):
    """Select features with importance above threshold."""
    model = XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X, y)

    importances = dict(zip(feature_cols, model.feature_importances_))
    selected = [f for f, imp in importances.items() if imp >= threshold]

    return selected, importances


def remove_correlated_features(X, feature_cols, threshold=0.85):
    """Remove highly correlated features, keeping the first in each pair."""
    if len(feature_cols) <= 1:
        return feature_cols, []

    corr_matrix = X[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    selected = [f for f in feature_cols if f not in to_drop]

    return selected, to_drop


def prepare_data(df_features, df_enroll, include_assessment=True, apply_feature_selection=True):
    """Prepare features and target for training.

    Args:
        df_features: DataFrame with features
        df_enroll: DataFrame with enrollment data
        include_assessment: Whether to include assessment-related features
        apply_feature_selection: Whether to apply feature selection (importance + correlation)
    """
    # Merge with enrollments
    df = df_features.merge(df_enroll[['user_id', 'course_id', 'failed', 'final_score']],
                           on=['user_id', 'course_id'], how='inner')

    # Drop rows with missing target
    df = df.dropna(subset=['failed'])

    # Get feature columns
    exclude_cols = ['user_id', 'course_id', 'failed', 'final_score', 'enrollment_state']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Filter assessment features if needed
    if not include_assessment:
        feature_cols = filter_assessment_features(feature_cols)

    # Prepare X, y
    X = df[feature_cols].copy()
    y = df['failed'].values

    # Handle missing values
    X = X.fillna(0)

    # Handle infinite values
    X = X.replace([np.inf, -np.inf], 0)

    # Drop object/string columns (not supported by XGBoost)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    dropped_cols = [c for c in feature_cols if c not in numeric_cols]
    if dropped_cols:
        print(f"  Dropped non-numeric columns: {dropped_cols}")
    X = X[numeric_cols]
    feature_cols = numeric_cols

    # Apply feature selection (same as optimized model)
    if apply_feature_selection and FEATURE_SELECTION_ENABLED and len(feature_cols) > 10:
        initial_count = len(feature_cols)

        # Step 1: Remove low-importance features
        selected_by_importance, _ = select_features_by_importance(
            X, y, feature_cols, threshold=FEATURE_IMPORTANCE_THRESHOLD
        )

        # Step 2: Remove highly correlated features
        selected_final, _ = remove_correlated_features(
            X, selected_by_importance, threshold=CORRELATION_THRESHOLD
        )

        print(f"  Feature selection: {initial_count} -> {len(selected_final)} features")
        feature_cols = selected_final
        X = X[feature_cols]

    return X, y, feature_cols


def train_and_evaluate(X, y, feature_cols):
    """Train XGBoost and evaluate with cross-validation."""
    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model = XGBClassifier(**XGBOOST_PARAMS)

    # Get predictions via cross-validation
    y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # ROC curve points
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    metrics['fpr'] = fpr.tolist()
    metrics['tpr'] = tpr.tolist()

    # Train final model for feature importance
    model.fit(X, y)
    importances = dict(zip(feature_cols, model.feature_importances_))
    top_features = dict(sorted(importances.items(), key=lambda x: -x[1])[:20])

    return metrics, top_features


def run_experiment(cutoff, include_assessment, df_enroll, include_znorm=True):
    """Run a single experiment.

    Args:
        cutoff: Time cutoff (2, 4, 6, 8, or 'full')
        include_assessment: Whether to include assessment-related features
        df_enroll: Enrollment dataframe
        include_znorm: Whether to include z-score normalized features
    """
    label = f"cutoff_{cutoff}_{'with' if include_assessment else 'without'}_assessment"
    print(f"\n--- Experiment: {label} ---")

    # Load features (with z-norm for fair comparison)
    df_features = load_features(cutoff, include_znorm=include_znorm)
    if df_features is None:
        print(f"  No features found for cutoff {cutoff}")
        return None

    # Prepare data
    X, y, feature_cols = prepare_data(df_features, df_enroll, include_assessment)

    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Failure rate: {y.mean()*100:.1f}%")

    # Train and evaluate
    metrics, top_features = train_and_evaluate(X, y, feature_cols)

    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")

    # Count z-norm features
    znorm_count = len([c for c in feature_cols if c.endswith('_znorm')])

    return {
        'cutoff': cutoff,
        'include_assessment': include_assessment,
        'include_znorm': include_znorm,
        'n_samples': len(X),
        'n_features': len(feature_cols),
        'n_znorm_features': znorm_count,
        'failure_rate': float(y.mean()),
        'metrics': metrics,
        'top_features': top_features,
    }


def main():
    parser = argparse.ArgumentParser(description='Train time-limited models')
    parser.add_argument('--cutoff', type=str, help='Specific cutoff (2,4,6,8,full)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    args = parser.parse_args()

    print("="*60)
    print("Time-Limited Early Warning Model Training")
    print("="*60)

    # Load enrollments
    df_enroll = load_enrollments()
    print(f"\nLoaded {len(df_enroll)} enrollments")
    print(f"Overall failure rate: {df_enroll['failed'].mean()*100:.1f}%")

    # Determine cutoffs to run
    if args.all:
        cutoffs = CUTOFFS
    elif args.cutoff:
        cutoffs = [int(args.cutoff) if args.cutoff != 'full' else 'full']
    else:
        cutoffs = CUTOFFS  # Default to all

    # Run experiments
    results = {
        'timestamp': datetime.now().isoformat(),
        'experiments': [],
        'summary': {}
    }

    for cutoff in cutoffs:
        # With assessment features
        exp_with = run_experiment(cutoff, True, df_enroll)
        if exp_with:
            results['experiments'].append(exp_with)

        # Without assessment features
        exp_without = run_experiment(cutoff, False, df_enroll)
        if exp_without:
            results['experiments'].append(exp_without)

    # Create summary table
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"\n{'Cutoff':<10} {'With Assess':<15} {'Without Assess':<15} {'Delta':<10}")
    print("-"*50)

    for cutoff in cutoffs:
        with_exp = next((e for e in results['experiments']
                        if e['cutoff'] == cutoff and e['include_assessment']), None)
        without_exp = next((e for e in results['experiments']
                           if e['cutoff'] == cutoff and not e['include_assessment']), None)

        if with_exp and without_exp:
            auc_with = with_exp['metrics']['roc_auc']
            auc_without = without_exp['metrics']['roc_auc']
            delta = auc_with - auc_without
            print(f"{cutoff:<10} {auc_with:.3f}          {auc_without:.3f}          {delta:+.3f}")

            results['summary'][str(cutoff)] = {
                'with_assessment': auc_with,
                'without_assessment': auc_without,
                'delta': delta
            }

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {OUTPUT_FILE}")

    # Key insights
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)

    # Find minimum cutoff for ROC-AUC >= 0.80
    for cutoff in cutoffs:
        exp = next((e for e in results['experiments']
                   if e['cutoff'] == cutoff and e['include_assessment']), None)
        if exp and exp['metrics']['roc_auc'] >= 0.80:
            print(f"\n* First cutoff with ROC-AUC >= 0.80 (with assessment): {cutoff} weeks")
            print(f"  ROC-AUC: {exp['metrics']['roc_auc']:.3f}")
            break

    for cutoff in cutoffs:
        exp = next((e for e in results['experiments']
                   if e['cutoff'] == cutoff and not e['include_assessment']), None)
        if exp and exp['metrics']['roc_auc'] >= 0.80:
            print(f"\n* First cutoff with ROC-AUC >= 0.80 (without assessment): {cutoff} weeks")
            print(f"  ROC-AUC: {exp['metrics']['roc_auc']:.3f}")
            break


if __name__ == "__main__":
    main()
