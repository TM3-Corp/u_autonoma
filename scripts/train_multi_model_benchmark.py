#!/usr/bin/env python3
"""
Multi-Model Benchmark Training for Early Warning System

Trains 15 model architectures across 10 configurations:
- 5 time cutoffs: Week 2, 4, 6, 8, Full
- 2 feature modes: WITH/WITHOUT assessment features

Outputs:
- data/analysis/multi_model_benchmark_results.json
- data/analysis/BENCHMARK_RESULTS.md

Based on U Autónoma methodology for benchmarking early warning models.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# Optional imports
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available")


# Configuration
BASE_DIR = Path(__file__).parent.parent
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_JSON = BASE_DIR / "data/analysis/multi_model_benchmark_results.json"
OUTPUT_MD = BASE_DIR / "data/analysis/BENCHMARK_RESULTS.md"

# Time cutoffs to evaluate
CUTOFFS = [2, 4, 6, 8, 'full']

# Assessment-related feature patterns (to exclude in WITHOUT mode)
ASSESSMENT_PATTERNS = [
    'quiz', 'quizzes',
    'assi', 'assignment',
    'grade', 'grad',
    'score',
    'submission',
]

# Threshold optimization strategies
STRATEGIES = ['max_g_mean', 'acc_at_recall_75', 'acc_at_recall_80', 'max_f1']


def get_models() -> dict:
    """Get all model architectures to benchmark."""
    models = {}

    # Tree-based models
    if HAS_XGBOOST:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, eval_metric='logloss', verbosity=0, random_state=42
        )
        models['XGBoost_balanced'] = XGBClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, scale_pos_weight=1.5,
            eval_metric='logloss', verbosity=0, random_state=42
        )

    if HAS_LIGHTGBM:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, verbosity=-1, random_state=42
        )
        models['LightGBM_balanced'] = LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, class_weight='balanced', verbosity=-1, random_state=42
        )

    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=42
    )
    models['RandomForest_balanced'] = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight='balanced', random_state=42
    )

    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )

    # Linear models
    models['LogisticRegression'] = LogisticRegression(
        C=1.0, max_iter=1000, random_state=42
    )
    models['LogisticRegression_balanced'] = LogisticRegression(
        C=1.0, class_weight='balanced', max_iter=1000, random_state=42
    )

    # SVM models
    models['SVM_RBF'] = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        cv=3
    )
    models['SVM_balanced'] = CalibratedClassifierCV(
        SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', random_state=42),
        cv=3
    )

    # Neural network models
    models['MLP'] = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation='relu',
        learning_rate_init=0.001, max_iter=500, random_state=42
    )
    models['MLP_deep'] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation='relu',
        learning_rate_init=0.001, max_iter=500, random_state=42
    )

    return models


def create_ensembles() -> dict:
    """Create ensemble models (requires XGBoost)."""
    if not HAS_XGBOOST:
        return {}

    ensembles = {}

    base_estimators = [
        ('xgb', XGBClassifier(n_estimators=50, max_depth=4, verbosity=0, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=42)),
    ]

    ensembles['VotingEnsemble'] = VotingClassifier(
        estimators=base_estimators, voting='soft'
    )

    ensembles['StackingEnsemble'] = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=3
    )

    return ensembles


def load_enrollments() -> pd.DataFrame:
    """Load enrollment data with target variable."""
    df = pd.read_csv(ENROLLMENTS_FILE)
    df['failed'] = (df['final_score'] < 57).astype(int)
    return df


def calculate_znorm_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate z-score normalized features per course."""
    exclude_cols = ['user_id', 'course_id', 'enrollment_state', 'failed', 'final_score']
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols and not c.endswith('_znorm')]

    result = []
    for course_id in df['course_id'].unique():
        course_df = df[df['course_id'] == course_id].copy()
        for col in feature_cols:
            mean_val = course_df[col].mean()
            std_val = course_df[col].std()
            if std_val > 0:
                course_df[f'{col}_znorm'] = (course_df[col] - mean_val) / std_val
            else:
                course_df[f'{col}_znorm'] = 0
        result.append(course_df)

    return pd.concat(result, ignore_index=True)


def load_features(cutoff) -> pd.DataFrame:
    """Load all features for a given cutoff.

    Args:
        cutoff: Time cutoff (2, 4, 6, 8, or 'full')
    """
    if cutoff == 'full':
        feature_dir = BASE_DIR / "data/enriched_features"
        # Include normalized_features for full data
        feature_files = [
            'session_features.parquet',
            'category_features.parquet',
            'proactivity_features.parquet',
            'pca_features.parquet',
            'weekly_features.parquet',
            'ngram_features.parquet',
            'graph_features.parquet',
            'time_features.parquet',
            'normalized_features.parquet',
        ]
    else:
        feature_dir = BASE_DIR / f"data/enriched_features/cutoff_week_{cutoff}"
        feature_files = [f.name for f in feature_dir.glob("*_features.parquet")]

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
            dfs.append(df)

    if not dfs:
        return None

    # Merge on user_id, course_id
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_dup'))
        dup_cols = [c for c in df_merged.columns if c.endswith('_dup')]
        df_merged = df_merged.drop(columns=dup_cols)

    # For temporal cutoffs, calculate z-norm on the fly
    if cutoff != 'full':
        df_merged = calculate_znorm_features(df_merged)

    return df_merged


def filter_assessment_features(columns: list, with_assessment: bool) -> list:
    """Filter out assessment-related features if with_assessment is False."""
    if with_assessment:
        return columns

    filtered = []
    for col in columns:
        col_lower = col.lower()
        is_excluded = any(pattern in col_lower for pattern in ASSESSMENT_PATTERNS)
        if not is_excluded:
            filtered.append(col)
    return filtered


def calculate_metrics(y_true, y_proba, thresholds) -> pd.DataFrame:
    """Calculate metrics at multiple thresholds."""
    results = []

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        if len(np.unique(y_pred)) < 2:
            continue

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape != (2, 2):
            continue

        tn, fp, fn, tp = cm.ravel()

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        g_mean = np.sqrt(recall * specificity)
        balanced_acc = (recall + specificity) / 2
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0

        results.append({
            'threshold': t,
            'accuracy': accuracy,
            'recall': recall,
            'specificity': specificity,
            'precision': precision,
            'g_mean': g_mean,
            'balanced_acc': balanced_acc,
            'f1': f1,
            'f2': f2,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        })

    return pd.DataFrame(results)


def find_optimal_thresholds(metrics_df: pd.DataFrame) -> dict:
    """Find optimal thresholds for each strategy."""
    if len(metrics_df) == 0:
        return {}

    optimal = {}

    # Max G-Mean
    idx = metrics_df['g_mean'].idxmax()
    optimal['max_g_mean'] = metrics_df.loc[idx].to_dict()

    # Max Accuracy with Recall >= 75%
    subset = metrics_df[metrics_df['recall'] >= 0.75]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['acc_at_recall_75'] = subset.loc[idx].to_dict()

    # Max Accuracy with Recall >= 80%
    subset = metrics_df[metrics_df['recall'] >= 0.80]
    if len(subset) > 0:
        idx = subset['accuracy'].idxmax()
        optimal['acc_at_recall_80'] = subset.loc[idx].to_dict()

    # Max F1
    idx = metrics_df['f1'].idxmax()
    optimal['max_f1'] = metrics_df.loc[idx].to_dict()

    return optimal


def train_and_evaluate(model, X, y, model_name: str) -> dict:
    """Train model and return evaluation metrics."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    try:
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    except Exception as e:
        print(f"      Error with {model_name}: {e}")
        return None

    roc_auc = roc_auc_score(y, y_pred_proba)

    # Calculate metrics at various thresholds
    thresholds = np.arange(0.10, 0.70, 0.01)
    metrics_df = calculate_metrics(y, y_pred_proba, thresholds)

    # Find optimal thresholds
    optimal = find_optimal_thresholds(metrics_df)

    return {
        'roc_auc': float(roc_auc),
        'optimal': optimal,
    }


def run_configuration(cutoff, with_assessment: bool, df_enroll: pd.DataFrame) -> dict:
    """Run all models for a specific configuration."""
    config_name = f"week_{cutoff}_{'with' if with_assessment else 'without'}_assessment"
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")

    # Load features
    df_features = load_features(cutoff)
    if df_features is None:
        print(f"  No features found for cutoff {cutoff}")
        return None

    # Merge with enrollments
    df = df_features.merge(
        df_enroll[['user_id', 'course_id', 'failed', 'final_score']],
        on=['user_id', 'course_id'], how='inner'
    )
    df = df.dropna(subset=['failed'])

    if len(df) < 50:
        print(f"  Insufficient samples: {len(df)}")
        return None

    # Get feature columns
    exclude_cols = ['user_id', 'course_id', 'failed', 'final_score', 'enrollment_state']
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    # Filter assessment features if needed
    feature_cols = filter_assessment_features(feature_cols, with_assessment)

    # Prepare X, y
    X = df[feature_cols].copy()
    y = df['failed'].values

    # Handle missing/infinite values
    X = X.fillna(0)
    X = X.replace([np.inf, -np.inf], 0)

    # Keep only numeric columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    feature_cols = numeric_cols

    print(f"  Samples: {len(y)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Failure rate: {y.mean()*100:.1f}%")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Get models
    models = get_models()
    models.update(create_ensembles())

    results = {
        'n_samples': len(y),
        'failure_rate': float(y.mean()),
        'n_features': len(feature_cols),
        'with_assessment': with_assessment,
        'cutoff': cutoff,
        'models': {}
    }

    # Train each model
    print(f"\n  Training {len(models)} models...")
    for model_name, model in models.items():
        print(f"    {model_name}...", end=' ')
        model_result = train_and_evaluate(model, X_scaled, y, model_name)
        if model_result:
            results['models'][model_name] = model_result
            print(f"AUC={model_result['roc_auc']:.3f}")
        else:
            print("FAILED")

    return results


def generate_markdown_report(all_results: dict) -> str:
    """Generate markdown summary report."""
    lines = [
        "# Multi-Model Benchmark Results",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        "| Config | Samples | Features | Failure Rate | Best AUC | Best Model |",
        "|--------|---------|----------|--------------|----------|------------|",
    ]

    for config_name, result in sorted(all_results.items()):
        best_model = max(result['models'].items(), key=lambda x: x[1]['roc_auc'])
        lines.append(
            f"| {config_name} | {result['n_samples']} | {result['n_features']} | "
            f"{result['failure_rate']*100:.1f}% | {best_model[1]['roc_auc']:.3f} | {best_model[0]} |"
        )

    # Best model per cutoff (WITH vs WITHOUT)
    lines.extend([
        "",
        "## WITH vs WITHOUT Assessment Comparison",
        "",
        "| Cutoff | WITH AUC | WITHOUT AUC | Delta | Best Model (WITH) |",
        "|--------|----------|-------------|-------|-------------------|",
    ])

    for cutoff in CUTOFFS:
        with_key = f"week_{cutoff}_with_assessment"
        without_key = f"week_{cutoff}_without_assessment"

        with_result = all_results.get(with_key)
        without_result = all_results.get(without_key)

        if with_result and without_result:
            with_best = max(with_result['models'].items(), key=lambda x: x[1]['roc_auc'])
            without_best = max(without_result['models'].items(), key=lambda x: x[1]['roc_auc'])
            delta = with_best[1]['roc_auc'] - without_best[1]['roc_auc']
            lines.append(
                f"| Week {cutoff} | {with_best[1]['roc_auc']:.3f} | {without_best[1]['roc_auc']:.3f} | "
                f"{delta:+.3f} | {with_best[0]} |"
            )

    # Best accuracy at 80% recall
    lines.extend([
        "",
        "## Best Accuracy at 80% Recall",
        "",
        "| Config | Model | Threshold | Accuracy | Recall | Specificity |",
        "|--------|-------|-----------|----------|--------|-------------|",
    ])

    for config_name, result in sorted(all_results.items()):
        best_acc = None
        best_model_name = None
        for model_name, model_result in result['models'].items():
            opt = model_result['optimal'].get('acc_at_recall_80')
            if opt and (best_acc is None or opt['accuracy'] > best_acc['accuracy']):
                best_acc = opt
                best_model_name = model_name

        if best_acc:
            lines.append(
                f"| {config_name} | {best_model_name} | {best_acc['threshold']:.2f} | "
                f"{best_acc['accuracy']*100:.1f}% | {best_acc['recall']*100:.1f}% | "
                f"{best_acc['specificity']*100:.1f}% |"
            )

    # Model rankings by G-Mean
    lines.extend([
        "",
        "## Model Rankings (by G-Mean at Full Data WITH Assessment)",
        "",
    ])

    full_with = all_results.get('week_full_with_assessment')
    if full_with:
        model_ranks = []
        for model_name, model_result in full_with['models'].items():
            gm_opt = model_result['optimal'].get('max_g_mean', {})
            model_ranks.append({
                'name': model_name,
                'roc_auc': model_result['roc_auc'],
                'g_mean': gm_opt.get('g_mean', 0),
                'recall': gm_opt.get('recall', 0),
                'accuracy': gm_opt.get('accuracy', 0),
            })

        model_ranks.sort(key=lambda x: -x['g_mean'])

        lines.append("| Rank | Model | ROC-AUC | G-Mean | Recall | Accuracy |")
        lines.append("|------|-------|---------|--------|--------|----------|")
        for i, m in enumerate(model_ranks[:10], 1):
            lines.append(
                f"| {i} | {m['name']} | {m['roc_auc']:.3f} | {m['g_mean']:.3f} | "
                f"{m['recall']*100:.1f}% | {m['accuracy']*100:.1f}% |"
            )

    return '\n'.join(lines)


def main():
    print("=" * 70)
    print("MULTI-MODEL BENCHMARK TRAINING")
    print("U Autónoma Early Warning System")
    print("=" * 70)
    print()

    # Load enrollments
    df_enroll = load_enrollments()
    print(f"Loaded {len(df_enroll)} enrollments")
    print(f"Overall failure rate: {df_enroll['failed'].mean()*100:.1f}%")
    print()

    # Run all configurations
    all_results = {}

    for cutoff in CUTOFFS:
        # WITH assessment
        result = run_configuration(cutoff, with_assessment=True, df_enroll=df_enroll)
        if result:
            config_name = f"week_{cutoff}_with_assessment"
            all_results[config_name] = result

        # WITHOUT assessment
        result = run_configuration(cutoff, with_assessment=False, df_enroll=df_enroll)
        if result:
            config_name = f"week_{cutoff}_without_assessment"
            all_results[config_name] = result

    # Save JSON results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nJSON results saved to: {OUTPUT_JSON}")

    # Generate and save markdown report
    md_report = generate_markdown_report(all_results)
    with open(OUTPUT_MD, 'w') as f:
        f.write(md_report)
    print(f"Markdown report saved to: {OUTPUT_MD}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<40} {'Best AUC':>10} {'Best Model':<25}")
    print("-" * 75)

    for config_name, result in sorted(all_results.items()):
        best_model = max(result['models'].items(), key=lambda x: x[1]['roc_auc'])
        print(f"{config_name:<40} {best_model[1]['roc_auc']:>10.3f} {best_model[0]:<25}")

    # WITH vs WITHOUT comparison
    print("\n" + "=" * 70)
    print("WITH vs WITHOUT ASSESSMENT COMPARISON")
    print("=" * 70)
    print(f"\n{'Cutoff':<15} {'WITH AUC':>12} {'WITHOUT AUC':>15} {'Delta':>10}")
    print("-" * 55)

    for cutoff in CUTOFFS:
        with_key = f"week_{cutoff}_with_assessment"
        without_key = f"week_{cutoff}_without_assessment"

        with_result = all_results.get(with_key)
        without_result = all_results.get(without_key)

        if with_result and without_result:
            with_best = max(with_result['models'].values(), key=lambda x: x['roc_auc'])['roc_auc']
            without_best = max(without_result['models'].values(), key=lambda x: x['roc_auc'])['roc_auc']
            delta = with_best - without_best
            print(f"Week {cutoff:<10} {with_best:>12.3f} {without_best:>15.3f} {delta:>+10.3f}")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
