#!/usr/bin/env python3
"""
Evaluate the predictive power of new session features.

Compares model performance WITH and WITHOUT the new features:
- session_density (clicks per minute)
- session_spread_days (unique active days)

Also provides feature importance ranking for all session features.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import json

warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"


def load_data():
    """Load session features and grades."""
    # Load session features (with new features)
    df_session = pd.read_parquet(DATA_DIR / "enriched_features" / "session_features.parquet")

    # Load grades
    df_enroll = pd.read_csv(DATA_DIR / "page_views" / "student_enrollments.csv")
    df_enroll = df_enroll[df_enroll['final_score'].notna()]

    # Merge
    df = df_session.merge(
        df_enroll[['user_id', 'course_id', 'final_score']],
        on=['user_id', 'course_id'],
        how='inner'
    )

    # Create target
    df['failed'] = (df['final_score'] < 57).astype(int)

    return df


def evaluate_features(df, feature_sets, model_name='XGBoost'):
    """Evaluate different feature sets."""
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for set_name, features in feature_sets.items():
        available_features = [f for f in features if f in df.columns]
        if not available_features:
            continue

        X = df[available_features].fillna(0).replace([np.inf, -np.inf], 0)
        y = df['failed'].values

        # Normalize within course
        X_norm = X.copy()
        for course_id in df['course_id'].unique():
            mask = df['course_id'] == course_id
            if mask.sum() > 1:
                scaler = StandardScaler()
                X_norm.loc[mask, :] = scaler.fit_transform(X_norm.loc[mask, :])

        X_norm = X_norm.fillna(0).values

        # Train model
        if HAS_XGBOOST and model_name == 'XGBoost':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                scale_pos_weight=sum(y == 0) / max(sum(y == 1), 1),
                random_state=42, n_jobs=-1, use_label_encoder=False,
                eval_metric='logloss', verbosity=0
            )
        else:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_leaf=5,
                random_state=42, class_weight='balanced', n_jobs=-1
            )

        # Cross-validation
        scores = cross_val_score(model, X_norm, y, cv=cv, scoring='roc_auc')

        # Fit on full data for feature importance
        model.fit(X_norm, y)

        results[set_name] = {
            'roc_auc_mean': scores.mean(),
            'roc_auc_std': scores.std(),
            'n_features': len(available_features),
            'features': available_features,
            'importance': dict(zip(available_features, model.feature_importances_))
        }

    return results


def main():
    print("="*70)
    print("EVALUATING NEW SESSION FEATURES")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"  Loaded {len(df)} student-course pairs")
    print(f"  Passed: {(df['failed'] == 0).sum()} | Failed: {(df['failed'] == 1).sum()}")

    # Define feature sets
    BASE_FEATURES = [
        'session_count', 'session_duration_mean', 'session_duration_std',
        'session_duration_median', 'sessions_per_week', 'views_per_session',
        'short_sessions_pct', 'long_sessions_pct', 'session_regularity',
        'total_views', 'total_time_min'
    ]

    NEW_FEATURES = ['session_density', 'session_spread_days']

    feature_sets = {
        'Base (sin nuevas)': BASE_FEATURES,
        'Con session_density': BASE_FEATURES + ['session_density'],
        'Con session_spread_days': BASE_FEATURES + ['session_spread_days'],
        'Con AMBAS nuevas': BASE_FEATURES + NEW_FEATURES,
    }

    # Evaluate
    print("\nEvaluating feature sets with XGBoost...")
    print("-"*70)

    results = evaluate_features(df, feature_sets)

    # Print results
    print(f"\n{'Feature Set':<30} {'ROC-AUC':>15} {'# Features':>12}")
    print("-"*60)

    base_auc = results['Base (sin nuevas)']['roc_auc_mean']
    for name, res in results.items():
        auc = res['roc_auc_mean']
        std = res['roc_auc_std']
        n = res['n_features']
        diff = auc - base_auc
        diff_str = f"({diff:+.3f})" if name != 'Base (sin nuevas)' else ""
        print(f"{name:<30} {auc:.3f} +/- {std:.3f} {diff_str:>8} {n:>6}")

    # Feature importance for full model
    print("\n" + "="*70)
    print("FEATURE IMPORTANCE (Full Model with All Features)")
    print("="*70)

    full_model_imp = results['Con AMBAS nuevas']['importance']
    sorted_imp = sorted(full_model_imp.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<30} {'Importance':>12} {'Rank':>6}")
    print("-"*50)
    for i, (feat, imp) in enumerate(sorted_imp, 1):
        marker = " ***" if feat in NEW_FEATURES else ""
        print(f"{feat:<30} {imp:>12.4f} {i:>6}{marker}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    base = results['Base (sin nuevas)']['roc_auc_mean']
    full = results['Con AMBAS nuevas']['roc_auc_mean']
    improvement = (full - base) / base * 100

    print(f"\nBase model ROC-AUC:       {base:.4f}")
    print(f"Full model ROC-AUC:       {full:.4f}")
    print(f"Improvement:              {improvement:+.2f}%")

    # Check where new features rank
    density_rank = next((i+1 for i, (f, _) in enumerate(sorted_imp) if f == 'session_density'), None)
    spread_rank = next((i+1 for i, (f, _) in enumerate(sorted_imp) if f == 'session_spread_days'), None)

    print(f"\nsession_density rank:     #{density_rank} of {len(sorted_imp)}")
    print(f"session_spread_days rank: #{spread_rank} of {len(sorted_imp)}")

    # Save results
    output_path = DATA_DIR / "report" / "analysis" / "new_features_evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON
    json_results = {
        k: {
            'roc_auc_mean': float(v['roc_auc_mean']),
            'roc_auc_std': float(v['roc_auc_std']),
            'n_features': v['n_features'],
            'features': v['features'],
            'importance': {kk: float(vv) for kk, vv in v['importance'].items()}
        }
        for k, v in results.items()
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
