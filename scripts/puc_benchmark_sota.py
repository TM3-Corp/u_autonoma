#!/usr/bin/env python3
"""
PUC SOTA Benchmark
==================
Unified benchmark for PUC early warning models. Computes features on-the-fly
at each temporal cutoff, runs multi-model benchmark with threshold optimization.

Phase 1 (fast, ~1h): No hyperparameter tuning, quick feature selection.
Phase 2 (thorough, ~3-5h): Optuna tuning on top-3 models, SMOTE experiments.

Usage:
    python scripts/puc_benchmark_sota.py              # Phase 1 (default)
    python scripts/puc_benchmark_sota.py --phase 2    # Phase 2
    python scripts/puc_benchmark_sota.py --quick       # Quick smoke test (1 config)

Input:
    data/puc/puc_fixed_data.parquet   (from puc_fix_data.py)
    data/puc/puc_grades_clean.parquet

Output:
    data/puc/sota_results/benchmark_results.json
    data/puc/sota_results/BENCHMARK_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.stats import entropy

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import lightgbm as lgb

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from imblearn.over_sampling import SMOTE, BorderlineSMOTE
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False

from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ── Configuration ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "puc"
RESULTS_DIR = DATA_DIR / "sota_results"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

SESSION_GAP_MINUTES = 30
PERCENTILES = [0.05, 0.10, 0.15, 0.20]
CUTOFF_WEEKS = [2, 4, 6, 8, "full"]

CATEGORIES = [
    "files", "discussions", "quizzes", "assignments",
    "pages", "modules", "grades", "announcements",
    "navigation", "external_tools",
]

# Assessment-related feature name patterns
ASSESSMENT_PATTERNS = [
    "quiz", "quizzes", "assign", "grade", "submission",
]


# ── Classification Schemes ─────────────────────────────────────────────────
def create_labels(
    grades: pd.DataFrame, scheme: str,
) -> pd.Series | pd.DataFrame:
    """Create target labels from grades according to classification scheme.

    Returns Series for binary schemes, DataFrame with multiple columns for Oviedo.
    """
    g = grades["grade"]
    if scheme == "binary_4.0":
        return (g < 4.0).astype(int)
    elif scheme == "binary_4.5":
        return (g < 4.5).astype(int)
    elif scheme == "binary_5.0":
        return (g < 5.0).astype(int)
    elif scheme == "3class":
        return pd.cut(
            g,
            bins=[0, 3.99, 5.99, 7.01],
            labels=[0, 1, 2],  # FAIL, PASS, GOOD
            include_lowest=True,
        ).astype(int)
    elif scheme == "4class":
        return pd.cut(
            g,
            bins=[0, 3.99, 4.99, 5.99, 7.01],
            labels=[0, 1, 2, 3],  # FAIL, MARGINAL, GOOD, EXCELLENT
            include_lowest=True,
        ).astype(int)
    elif scheme == "3class_marginal":
        return pd.cut(
            g,
            bins=[0, 3.99, 4.49, 7.01],
            labels=[0, 1, 2],  # FAIL (<4), MARGINAL (4-4.5), OK (>4.5)
            include_lowest=True,
        ).astype(int)
    elif scheme == "oviedo":
        # 3 independent binary classifiers
        return pd.DataFrame({
            "at_risk": (g <= 2.5).astype(int),
            "pass_fail": (g < 5.0).astype(int),
            "excellent": (g >= 6.0).astype(int),
        })
    else:
        raise ValueError(f"Unknown scheme: {scheme}")


CLASSIFICATION_SCHEMES = ["binary_4.0", "binary_4.5", "binary_5.0", "3class", "4class", "3class_marginal", "oviedo"]

# ── SOTA risk-score config (2026-06 review) ─────────────────────────────────
# Probabilities feed operating-point selection and the deployed risk score, so:
#   - CALIBRATE_PROBABILITIES: wrap binary tree models in Platt/sigmoid so a
#     "70% risk" means 70% (Brier/ECE improve ~3x, ROC-AUC unchanged).
#   - USE_SMOTE=False: retire resampling. The 2022-2026 risk-scoring consensus
#     (van den Goorbergh JAMIA'22, Carriero Stat.Med.'25) and our own ablation
#     show SMOTE does NOT improve AUC and damages calibration. Production stack =
#     class-weights (scale_pos_weight) + calibration + threshold-moving.
# Evidence: data/puc/sota_results/few_feature_sweep/CONSOLIDATED_FINDINGS.md
CALIBRATE_PROBABILITIES = True
USE_SMOTE = False


# ── Model Definitions ──────────────────────────────────────────────────────
def get_models() -> dict:
    """Return dict of model_name -> model instance."""
    models = {}

    # Tree-based
    models["XGBoost"] = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE,
    )
    models["XGBoost_balanced"] = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, scale_pos_weight=3.0,
        eval_metric="logloss", verbosity=0, random_state=RANDOM_STATE,
    )
    if HAS_LGBM:
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, verbosity=-1, random_state=RANDOM_STATE,
        )
        models["LightGBM_balanced"] = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.1,
            subsample=0.8, class_weight="balanced", verbosity=-1, random_state=RANDOM_STATE,
        )
    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100, max_depth=8, random_state=RANDOM_STATE,
    )
    models["RandomForest_balanced"] = RandomForestClassifier(
        n_estimators=100, max_depth=8, class_weight="balanced", random_state=RANDOM_STATE,
    )
    models["GradientBoosting"] = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE,
    )

    # Linear
    models["LogisticRegression"] = LogisticRegression(
        C=1.0, max_iter=1000, random_state=RANDOM_STATE,
    )
    models["LogisticRegression_balanced"] = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE,
    )

    # SVM (wrapped for predict_proba)
    models["SVM_RBF"] = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE), cv=3,
    )
    models["SVM_balanced"] = CalibratedClassifierCV(
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", random_state=RANDOM_STATE), cv=3,
    )

    # Neural nets
    models["MLP"] = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu",
        learning_rate_init=0.001, max_iter=500, random_state=RANDOM_STATE,
    )
    models["MLP_deep"] = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu",
        learning_rate_init=0.001, max_iter=500, random_state=RANDOM_STATE,
    )

    return models


def get_ensembles() -> dict:
    """Return ensemble models (slower, so separate)."""
    base_estimators = [
        ("xgb", XGBClassifier(n_estimators=50, max_depth=4, verbosity=0, random_state=RANDOM_STATE)),
        ("rf", RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_STATE)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(32,), max_iter=300, random_state=RANDOM_STATE)),
    ]
    return {
        "VotingEnsemble": VotingClassifier(estimators=base_estimators, voting="soft"),
        "StackingEnsemble": StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000), cv=3,
        ),
    }


# ── Feature Computation ───────────────────────────────────────────────────

def calculate_session_features(df_user: pd.DataFrame) -> dict:
    """Session-based features from a single user's page views."""
    if len(df_user) < 2:
        return {}

    ts = df_user["created_at"].sort_values()
    gaps_min = ts.diff().dt.total_seconds() / 60
    session_starts = (gaps_min >= SESSION_GAP_MINUTES) | gaps_min.isna()
    session_ids = session_starts.cumsum()

    # Per-session stats
    durations = []
    views_per = []
    for _, grp in df_user.groupby(session_ids):
        t = grp["created_at"]
        dur = (t.max() - t.min()).total_seconds() / 60
        durations.append(dur)
        views_per.append(len(grp))

    durations = np.array(durations)
    views_per = np.array(views_per)
    n_sessions = len(durations)

    total_span_weeks = max(
        (ts.max() - ts.min()).total_seconds() / (7 * 86400), 1 / 7,
    )

    feats = {
        "n_sessions": n_sessions,
        "sessions_per_week": n_sessions / total_span_weeks,
        "session_duration_mean": durations.mean(),
        "session_duration_std": durations.std() if n_sessions > 1 else 0,
        "session_duration_max": durations.max(),
        "total_time_min": durations.sum(),
        "views_per_session": views_per.mean(),
        "short_sessions_pct": (durations < 5).sum() / n_sessions,
        "long_sessions_pct": (durations > 30).sum() / n_sessions,
    }

    # Session regularity (coefficient of variation of inter-session gaps)
    session_start_times = []
    for _, grp in df_user.groupby(session_ids):
        session_start_times.append(grp["created_at"].min())
    if len(session_start_times) > 1:
        session_start_times = pd.Series(session_start_times).sort_values()
        inter_gaps = session_start_times.diff().dropna().dt.total_seconds() / 3600
        if len(inter_gaps) > 1 and inter_gaps.mean() > 0:
            feats["session_regularity"] = max(0, 1 - (inter_gaps.std() / inter_gaps.mean()))
        else:
            feats["session_regularity"] = 0
    else:
        feats["session_regularity"] = 0

    return feats


def calculate_category_features(df_user: pd.DataFrame) -> dict:
    """Category-based features: views, unique resources, percentages, time per category."""
    total = len(df_user)
    if total == 0:
        return {}

    feats = {"total_views": total}
    has_interaction_seconds = "interaction_seconds" in df_user.columns

    for cat in CATEGORIES:
        cat_data = df_user[df_user["category"] == cat]
        feats[f"{cat}_views"] = len(cat_data)
        feats[f"{cat}_unique"] = cat_data["resource_id"].dropna().nunique()
        feats[f"{cat}_pct"] = len(cat_data) / total

        # Time spent per category (minutes)
        if has_interaction_seconds:
            feats[f"{cat}_time_min"] = cat_data["interaction_seconds"].fillna(0).sum() / 60
        else:
            feats[f"{cat}_time_min"] = 0.0

    # Content vs assessment ratio
    content_views = feats.get("files_views", 0) + feats.get("pages_views", 0) + feats.get("modules_views", 0)
    assess_views = feats.get("assignments_views", 0) + feats.get("quizzes_views", 0) + feats.get("grades_views", 0)
    feats["content_vs_assessment_ratio"] = content_views / (assess_views + 1)

    return feats


def calculate_time_features(df_user: pd.DataFrame) -> dict:
    """Time-of-day and day-of-week engagement patterns."""
    if len(df_user) < 5:
        return {}

    hours = df_user["hour"]
    days = df_user["day_of_week"]
    total = len(df_user)

    feats = {}
    # Time-of-day bins
    feats["morning_pct"] = ((hours >= 6) & (hours < 12)).sum() / total
    feats["afternoon_pct"] = ((hours >= 12) & (hours < 18)).sum() / total
    feats["evening_pct"] = ((hours >= 18) & (hours < 24)).sum() / total
    feats["night_pct"] = ((hours >= 0) & (hours < 6)).sum() / total
    feats["weekend_pct"] = (days >= 5).sum() / total

    # Entropy-based diversity (FIX: use nunique of actual hours, not value_counts nunique)
    hour_dist = hours.value_counts(normalize=True)
    feats["hour_entropy"] = entropy(hour_dist.values, base=2) / np.log2(24)  # normalized [0,1]

    day_dist = days.value_counts(normalize=True)
    feats["day_entropy"] = entropy(day_dist.values, base=2) / np.log2(7)  # normalized [0,1]

    # FIX: unique_hours / 24 and unique_days / 7 (correct diversity measures)
    feats["unique_hours"] = hours.nunique() / 24
    feats["unique_days"] = days.nunique() / 7

    return feats


def calculate_weekly_features(
    df_user: pd.DataFrame, total_weeks: int = 16,
) -> dict:
    """Weekly temporal patterns including DCT coefficients."""
    if len(df_user) < 3:
        return {}

    weeks = df_user["week_number"]
    views_per_week = weeks.value_counts().sort_index()

    # Pad to total_weeks
    weekly_array = np.zeros(total_weeks)
    for w, count in views_per_week.items():
        if 1 <= w <= total_weeks:
            weekly_array[w - 1] = count

    active_weeks = (weekly_array > 0).sum()
    feats = {
        "active_weeks": int(active_weeks),
        "weekly_mean": weekly_array[weekly_array > 0].mean() if active_weeks > 0 else 0,
        "weekly_std": weekly_array[weekly_array > 0].std() if active_weeks > 1 else 0,
        "first_active_week": int(np.argmax(weekly_array > 0) + 1) if active_weeks > 0 else 0,
        "last_active_week": int(total_weeks - np.argmax(weekly_array[::-1] > 0)) if active_weeks > 0 else 0,
    }

    # Early vs late ratio
    mid = total_weeks // 2
    early_sum = weekly_array[:mid].sum()
    late_sum = weekly_array[mid:].sum()
    feats["early_late_ratio"] = early_sum / (late_sum + 1)

    # Weekly trend (slope of linear fit on active weeks)
    if active_weeks >= 3:
        active_idx = np.where(weekly_array > 0)[0]
        active_vals = weekly_array[active_idx]
        if len(active_idx) > 1:
            slope = np.polyfit(active_idx, active_vals, 1)[0]
            feats["weekly_trend"] = slope
        else:
            feats["weekly_trend"] = 0
    else:
        feats["weekly_trend"] = 0

    # FIX: weeks_since_last uses total_weeks (course end proxy), not student max week
    feats["weeks_since_last"] = total_weeks - feats["last_active_week"]

    # DCT coefficients (from normalized weekly activity)
    total_activity = weekly_array.sum()
    if total_activity > 0:
        normalized = weekly_array / total_activity
        dct_coeffs = dct(normalized[:8], norm="ortho")
        for i in range(min(4, len(dct_coeffs))):
            feats[f"dct_{i}"] = float(dct_coeffs[i])
    else:
        for i in range(4):
            feats[f"dct_{i}"] = 0.0

    return feats


def calculate_gap_features(df_user: pd.DataFrame) -> dict:
    """Inter-event gap features (replaces broken inactivity_episodes)."""
    if len(df_user) < 3:
        return {}

    ts = df_user["created_at"].sort_values()
    gaps_hours = ts.diff().dropna().dt.total_seconds() / 3600

    if len(gaps_hours) == 0:
        return {}

    return {
        "mean_gap_hours": gaps_hours.mean(),
        "max_gap_hours": gaps_hours.max(),
        "gap_std_hours": gaps_hours.std(),
    }


def calculate_transition_features(df_user: pd.DataFrame) -> dict:
    """Sequential navigation bigrams from sessions."""
    if len(df_user) < 3:
        return {}

    df_sorted = df_user.sort_values("created_at")

    # Session detection
    gaps_min = df_sorted["created_at"].diff().dt.total_seconds() / 60
    session_starts = (gaps_min >= SESSION_GAP_MINUTES) | gaps_min.isna()
    session_ids = session_starts.cumsum()

    # Extract bigrams (FIX: use category column directly)
    all_bigrams: list[str] = []
    for _, session in df_sorted.groupby(session_ids):
        if len(session) < 2:
            continue
        cats = session["category"].tolist()
        for i in range(len(cats) - 1):
            all_bigrams.append(f"{cats[i]}->{cats[i+1]}")

    if not all_bigrams:
        return {}

    bigram_counts = Counter(all_bigrams)
    total = sum(bigram_counts.values())

    feats = {
        "total_transitions": total,
        "unique_transitions": len(bigram_counts),
        "transition_diversity": len(bigram_counts) / (len(CATEGORIES) ** 2),
    }

    # Transition entropy (normalized)
    probs = np.array(list(bigram_counts.values())) / total
    max_ent = np.log2(len(CATEGORIES) ** 2)
    feats["transition_entropy"] = entropy(probs, base=2) / max_ent if max_ent > 0 else 0

    # Self-loop ratio
    self_loops = sum(c for b, c in bigram_counts.items() if b.split("->")[0] == b.split("->")[1])
    feats["self_loop_ratio"] = self_loops / total

    # Specific pedagogically meaningful bigrams
    key_bigrams = [
        "assignments->grades", "quizzes->grades", "modules->assignments",
        "modules->quizzes", "modules->files", "discussions->discussions",
        "navigation->modules", "files->assignments",
    ]
    for bg in key_bigrams:
        feats[f"bigram_{bg.replace('->', '_to_')}"] = bigram_counts.get(bg, 0)

    return feats


def calculate_proactivity_features(
    df_user: pd.DataFrame, course_start: pd.Timestamp,
) -> dict:
    """Proactivity and engagement density features."""
    if len(df_user) < 2:
        return {}

    ts = df_user["created_at"]
    first_activity = ts.min()
    days_to_first = (first_activity - course_start).total_seconds() / 86400

    total_days = max((ts.max() - ts.min()).total_seconds() / 86400, 1)
    unique_active_days = ts.dt.date.nunique()

    feats = {
        "days_to_first": max(0, days_to_first),
        "activity_density": len(df_user) / total_days,
        "daily_consistency": unique_active_days / max(total_days, 1),
    }

    return feats


def calculate_rich_proactivity_features(
    df_user: pd.DataFrame, cutoff_days: int,
) -> dict:
    """Rich per-category proactivity features (Ignacio-style).

    For each category, measures how early/late the student accessed resources
    relative to the course period. Lower mean_pct = accessed later; higher = earlier.
    """
    if len(df_user) < 2 or cutoff_days <= 0:
        return {}

    course_start = df_user["created_at"].min()
    feats = {}
    cat_means = []
    proact_cats = ["files", "assignments", "quizzes", "discussions", "pages", "modules"]

    for cat in proact_cats:
        cat_data = df_user[df_user["category"] == cat]
        prefix = f"{cat}_proact"

        if len(cat_data) == 0:
            feats[f"{prefix}_mean_pct"] = 1.0  # never accessed = maximally late
            feats[f"{prefix}_median_pct"] = 1.0
            feats[f"{prefix}_std_pct"] = 0.0
            feats[f"{prefix}_top50_rate"] = 0.0
            cat_means.append(1.0)
            continue

        days_since = (cat_data["created_at"] - course_start).dt.total_seconds() / 86400
        pct_vals = (days_since / cutoff_days).clip(0, 1)

        mean_pct = float(pct_vals.mean())
        feats[f"{prefix}_mean_pct"] = mean_pct
        feats[f"{prefix}_median_pct"] = float(pct_vals.median())
        feats[f"{prefix}_std_pct"] = float(pct_vals.std()) if len(pct_vals) > 1 else 0.0
        feats[f"{prefix}_top50_rate"] = float((pct_vals <= 0.5).mean())
        cat_means.append(mean_pct)

    # Global proactivity: 1 - mean(all category means), so higher = more proactive
    feats["overall_proactivity"] = 1.0 - np.mean(cat_means)

    return feats


def calculate_first_access_week_features(df_user: pd.DataFrame) -> dict:
    """Per-category first access week (Ignacio-style).

    999 = never accessed that category.
    """
    if len(df_user) == 0:
        return {}

    feats = {}
    for cat in ["assignments", "quizzes", "discussions", "grades", "files"]:
        cat_data = df_user[df_user["category"] == cat]
        if len(cat_data) > 0:
            feats[f"{cat}_first_access_week"] = int(cat_data["week_number"].min())
        else:
            feats[f"{cat}_first_access_week"] = 999

    return feats


def calculate_coverage_features(df_user: pd.DataFrame, df_course: pd.DataFrame) -> dict:
    """Resource coverage and category diversity (Oviedo-style).

    FIX: Uses real resource_ids instead of synthetic hashes.
    """
    if len(df_user) < 2:
        return {}

    # Total unique resources in the course
    course_resources = df_course["resource_id"].dropna().nunique()
    user_resources = df_user["resource_id"].dropna().nunique()

    feats = {
        "resource_coverage_rate": user_resources / course_resources if course_resources > 0 else 0,
    }

    # Category diversity: how many categories did the student use?
    user_cats = df_user["category"].nunique()
    feats["category_diversity"] = user_cats / len(CATEGORIES)

    # Per-category coverage
    for cat in ["files", "discussions", "quizzes", "assignments", "pages", "modules"]:
        course_cat_res = df_course.loc[df_course["category"] == cat, "resource_id"].dropna().nunique()
        user_cat_res = df_user.loc[df_user["category"] == cat, "resource_id"].dropna().nunique()
        feats[f"{cat}_coverage"] = user_cat_res / course_cat_res if course_cat_res > 0 else 0

    return feats


def calculate_all_features(
    df_pv: pd.DataFrame,
    course_starts: dict,
    compute_pct: bool = True,
    total_weeks: int = 16,
    cutoff_weeks: int | str = "full",
) -> pd.DataFrame:
    """Compute all feature families for each (student, course) pair.

    Args:
        df_pv: Page views DataFrame with category, resource_id, hour, day_of_week, etc.
        course_starts: {course_id: pd.Timestamp}
        compute_pct: Whether to compute PCT ranking features (slow).
        total_weeks: Number of weeks for weekly features (matches cutoff).
        cutoff_weeks: Temporal cutoff in weeks (int) or "full".
    """
    cutoff_days = cutoff_weeks * 7 if isinstance(cutoff_weeks, int) else total_weeks * 7

    results = []
    grouped_course = df_pv.groupby("course_id")

    for course_id, df_course in grouped_course:
        course_start = course_starts.get(course_id)
        if course_start is None:
            continue

        enrolled_students = set(df_course["student_id"].unique())

        # Pre-compute PCT rankings per course (avoids redundant work per student)
        pct_cache = {}
        if compute_pct:
            pct_cache = _precompute_pct_rankings(df_course, enrolled_students)

        grouped_user = df_course.groupby("student_id")

        for student_id, df_user in grouped_user:
            if len(df_user) < 2:
                continue

            row = {"student_id": student_id, "course_id": course_id}

            # All feature families
            row.update(calculate_session_features(df_user))
            row.update(calculate_category_features(df_user))
            row.update(calculate_time_features(df_user))
            row.update(calculate_weekly_features(df_user, total_weeks=total_weeks))
            row.update(calculate_gap_features(df_user))
            row.update(calculate_transition_features(df_user))
            row.update(calculate_proactivity_features(df_user, course_start))
            row.update(calculate_coverage_features(df_user, df_course))

            # New feature families
            row.update(calculate_rich_proactivity_features(df_user, cutoff_days))
            row.update(calculate_first_access_week_features(df_user))

            # PCT features from pre-computed cache
            if compute_pct and pct_cache:
                row.update(_lookup_pct_features(pct_cache, student_id))

            results.append(row)

    return pd.DataFrame(results)


def _precompute_pct_rankings(
    df_course: pd.DataFrame, enrolled_students: set,
) -> dict:
    """Pre-compute PCT rankings for all resources in a course.

    Returns {resource_type: {resource_id: {student_id: pct_value}}}
    """
    resource_types = ["files", "discussions", "quizzes", "assignments", "pages", "modules"]
    rankings = {}

    for rtype in resource_types:
        type_data = df_course[df_course["category"] == rtype]
        resources = type_data["resource_id"].dropna().unique()
        rtype_rankings = {}

        for rid in resources:
            rid_data = type_data[type_data["resource_id"] == rid]
            first_access = rid_data.groupby("student_id")["created_at"].min().sort_values()
            n_accessors = len(first_access)

            rid_pct = {}
            for rank, (sid, _) in enumerate(first_access.items(), 1):
                rid_pct[sid] = (n_accessors - rank + 1) / n_accessors
            rtype_rankings[rid] = rid_pct

        rankings[rtype] = rtype_rankings

    return rankings


def _lookup_pct_features(pct_cache: dict, student_id: int) -> dict:
    """Look up pre-computed PCT features for a student.

    Uses _rank_ prefix to avoid collision with proactivity features.
    """
    feats = {}
    for rtype, rtype_rankings in pct_cache.items():
        if not rtype_rankings:
            prefix = rtype[:4]
            feats[f"{prefix}_rank_pct_mean"] = 0
            feats[f"{prefix}_rank_access_rate"] = 0
            continue

        pct_values = []
        for rid, rid_pct in rtype_rankings.items():
            pct_values.append(rid_pct.get(student_id, 0.0))

        pct_arr = np.array(pct_values)
        prefix = rtype[:4]
        feats[f"{prefix}_rank_pct_mean"] = float(pct_arr.mean())
        feats[f"{prefix}_rank_access_rate"] = float((pct_arr > 0).mean())

    return feats


# ── Z-normalization ────────────────────────────────────────────────────────
def calculate_znorm(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Per-course z-normalization of features."""
    znorm_parts = []
    for course_id, course_df in df.groupby("course_id"):
        course_df = course_df.copy()
        znorm_cols = {}
        for col in feature_cols:
            if col not in course_df.columns:
                continue
            vals = course_df[col]
            mean_val = vals.mean()
            std_val = vals.std()
            if std_val > 0:
                znorm_cols[f"{col}_znorm"] = (vals - mean_val) / std_val
            else:
                znorm_cols[f"{col}_znorm"] = 0
        if znorm_cols:
            znorm_df = pd.DataFrame(znorm_cols, index=course_df.index)
            course_df = pd.concat([course_df, znorm_df], axis=1)
        znorm_parts.append(course_df)
    return pd.concat(znorm_parts, ignore_index=True)


# ── Feature Selection ──────────────────────────────────────────────────────
def sota_feature_selection(
    X: pd.DataFrame, y: pd.Series,
    correlation_threshold: float = 0.85,
    variance_threshold: float = 0.01,
    min_features: int = 10,
    max_features: int = 80,
    return_ranked: bool = False,
) -> list[str]:
    """
    SOTA 4-stage feature selection (per-fold safe).

    If return_ranked=True, returns the FULL composite-ranked feature list
    (best first), after the variance+correlation pre-filter, without applying
    the n_select cap — callers can take head(N) for an N-feature sweep.

    Stage 1: Variance filter + correlation-based redundancy removal
    Stage 2: Statistical — Mutual Information + ANOVA F-test
    Stage 3: Embedded — RF importance + L1 LogReg (LASSO) + XGBoost importance
    Stage 4: Simplified Boruta — single-pass shadow feature test

    Combines via weighted composite percentile rank across all methods.
    """
    if X.shape[1] <= min_features:
        return list(X.columns)

    # ── Stage 1: Filter ──────────────────────────────────────────────────
    # 1a. Variance threshold — remove near-constant features
    variances = X.var()
    keep_var = variances[variances >= variance_threshold].index.tolist()
    if len(keep_var) < min_features:
        keep_var = variances.nlargest(min_features).index.tolist()
    X_filt = X[keep_var]

    # 1b. Correlation removal — drop redundant features (keep first encountered)
    features = list(X_filt.columns)
    if len(features) > min_features:
        corr_matrix = X_filt.corr().abs()
        to_drop = set()
        for i in range(len(corr_matrix)):
            if corr_matrix.columns[i] in to_drop:
                continue
            for j in range(i + 1, len(corr_matrix)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    to_drop.add(corr_matrix.columns[j])
        features = [f for f in features if f not in to_drop]

    if len(features) <= min_features:
        return features

    X_filt = X_filt[features]
    n_feat = len(features)

    # ── Stage 2: Statistical ─────────────────────────────────────────────
    # Mutual Information (non-parametric, captures non-linear dependencies)
    try:
        mi_scores = mutual_info_classif(
            X_filt, y, random_state=RANDOM_STATE, n_neighbors=3,
        )
    except Exception:
        mi_scores = np.zeros(n_feat)
    mi_pctrank = pd.Series(mi_scores, index=features).rank(
        pct=True, ascending=False
    )

    # ANOVA F-test (linear discriminative power)
    try:
        f_scores, _ = f_classif(X_filt, y)
        f_scores = np.nan_to_num(f_scores, nan=0.0)
    except Exception:
        f_scores = np.zeros(n_feat)
    f_pctrank = pd.Series(f_scores, index=features).rank(
        pct=True, ascending=False
    )

    # ── Stage 3: Embedded ────────────────────────────────────────────────
    # Random Forest importance
    try:
        rf = RandomForestClassifier(
            n_estimators=30, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        rf.fit(X_filt, y)
        rf_imp = pd.Series(rf.feature_importances_, index=features)
    except Exception:
        rf_imp = pd.Series(1.0 / n_feat, index=features)
    rf_pctrank = rf_imp.rank(pct=True, ascending=False)

    # L1 Logistic Regression (LASSO) — identifies sparse predictive set
    try:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_filt)
        lasso = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=300,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        lasso.fit(X_sc, y)
        coefs = np.abs(lasso.coef_)
        if coefs.ndim > 1:
            coefs = coefs.mean(axis=0)
        else:
            coefs = coefs.ravel()
        lasso_imp = pd.Series(coefs, index=features)
    except Exception:
        lasso_imp = pd.Series(1.0 / n_feat, index=features)
    lasso_pctrank = lasso_imp.rank(pct=True, ascending=False)

    # LightGBM importance (much faster than XGBoost for ranking purposes)
    try:
        lgb_sel = lgb.LGBMClassifier(
            n_estimators=30, max_depth=4,
            class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1,
        )
        lgb_sel.fit(X_filt, y)
        lgb_imp = pd.Series(lgb_sel.feature_importances_, index=features)
    except Exception:
        lgb_imp = pd.Series(1.0 / n_feat, index=features)
    lgb_pctrank = lgb_imp.rank(pct=True, ascending=False)

    # ── Stage 4: Simplified Boruta (single-pass) ─────────────────────────
    # Fit RF on real + shuffled shadow features; features beating max shadow pass
    try:
        rng = np.random.RandomState(RANDOM_STATE)
        X_shadow = X_filt.values.copy()
        for col_idx in range(X_shadow.shape[1]):
            rng.shuffle(X_shadow[:, col_idx])
        X_combined = np.hstack([X_filt.values, X_shadow])

        rf_boruta = RandomForestClassifier(
            n_estimators=30, max_depth=6,
            class_weight="balanced", random_state=RANDOM_STATE,
        )
        rf_boruta.fit(X_combined, y)
        real_imp = rf_boruta.feature_importances_[:n_feat]
        shadow_max = rf_boruta.feature_importances_[n_feat:].max()
        # Continuous score: how much the feature exceeds the shadow threshold
        boruta_score = real_imp / (shadow_max + 1e-10)
    except Exception:
        boruta_score = np.ones(n_feat)
    boruta_pctrank = pd.Series(boruta_score, index=features).rank(
        pct=True, ascending=False
    )

    # ── Composite Score ──────────────────────────────────────────────────
    # Weighted average of percentile ranks (lower = better feature)
    # Weights: tree ensembles 0.45 (RF+LGB), statistical 0.25 (MI+ANOVA),
    #          sparse 0.15 (LASSO), wrapper 0.15 (Boruta)
    composite = (
        0.15 * mi_pctrank
        + 0.10 * f_pctrank
        + 0.20 * rf_pctrank
        + 0.15 * lasso_pctrank
        + 0.25 * lgb_pctrank
        + 0.15 * boruta_pctrank
    )

    composite_sorted = composite.sort_values()
    if return_ranked:
        return composite_sorted.index.tolist()
    n_select = min(max_features, max(min_features, n_feat // 2))
    selected = composite_sorted.head(n_select).index.tolist()

    return selected if len(selected) >= 3 else list(X.columns[:min_features])


# ── Threshold Optimization ─────────────────────────────────────────────────
def calculate_metrics_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float,
) -> dict:
    """Calculate all metrics at a given threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        return {}

    total = tn + fp + fn + tp
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / total if total > 0 else 0

    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    f2 = 5 * precision * recall / (4 * precision + recall) if (4 * precision + recall) > 0 else 0
    f3 = 10 * precision * recall / (9 * precision + recall) if (9 * precision + recall) > 0 else 0

    youden = recall + specificity - 1
    g_mean = np.sqrt(recall * specificity)

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0

    return {
        "threshold": threshold,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "recall": recall, "precision": precision, "specificity": specificity,
        "accuracy": accuracy, "f1": f1, "f2": f2, "f3": f3,
        "youden_j": youden, "g_mean": g_mean, "mcc": mcc,
        "cost_3x": fn * 3 + fp,
        "cost_5x": fn * 5 + fp,
    }


def find_optimal_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray,
) -> dict[str, dict]:
    """Find optimal threshold for each of 12 criteria."""
    thresholds = np.arange(0.05, 0.96, 0.01)
    all_metrics = [calculate_metrics_at_threshold(y_true, y_proba, t) for t in thresholds]
    all_metrics = [m for m in all_metrics if m]

    if not all_metrics:
        return {}

    df = pd.DataFrame(all_metrics)

    criteria = {}

    # Direct maximization criteria
    for name, col in [
        ("max_f1", "f1"), ("max_f2", "f2"), ("max_f3", "f3"),
        ("youden_j", "youden_j"), ("mcc", "mcc"), ("g_mean", "g_mean"),
        ("max_accuracy", "accuracy"),
    ]:
        idx = df[col].idxmax()
        criteria[name] = df.iloc[idx].to_dict()

    # Cost minimization
    for name, col in [("cost_3x", "cost_3x"), ("cost_5x", "cost_5x")]:
        idx = df[col].idxmin()
        criteria[name] = df.iloc[idx].to_dict()

    # Recall-constrained criteria
    for target_recall, name in [(0.80, "recall_80"), (0.85, "recall_85"), (0.90, "recall_90")]:
        subset = df[df["recall"] >= target_recall]
        if len(subset) > 0:
            idx = subset["accuracy"].idxmax()
            criteria[name] = df.iloc[idx].to_dict()
        else:
            # Fall back to highest recall available
            idx = df["recall"].idxmax()
            criteria[name] = df.iloc[idx].to_dict()

    return criteria


# ── Evaluation ─────────────────────────────────────────────────────────────
def precompute_fold_selections(
    X: pd.DataFrame, y: pd.Series,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
) -> list[list[str]]:
    """Pre-compute SOTA feature selection for each CV fold.

    Call once per (percentile, cutoff, scheme, assessment_mode) combination,
    then pass result to all evaluate_model calls sharing the same (X, y).
    This avoids redundant computation across the 15 model evaluations.
    """
    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_iter = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_iter = list(cv.split(X, y))

    fold_selections = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
        y_train = y.iloc[train_idx]
        if y_train.sum() == 0:
            fold_selections.append(list(X.columns))
            continue
        X_train_raw = X.iloc[train_idx]
        if X_train_raw.shape[1] > 5:
            selected = sota_feature_selection(X_train_raw, y_train)
        else:
            selected = list(X_train_raw.columns)
        fold_selections.append(selected)
    return fold_selections


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins). Noisy at small N."""
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        e += abs(y_prob[m].mean() - y_true[m].mean()) * m.sum() / len(y_true)
    return float(e)


def evaluate_model(
    model, model_name: str, X: pd.DataFrame, y: pd.Series,
    is_binary: bool = True,
    groups: np.ndarray | None = None,
    do_feature_selection: bool = True,
    resampling: str = "none",
    fold_selections: list[list[str]] | None = None,
    calibrate: bool = CALIBRATE_PROBABILITIES,
) -> dict | None:
    """Evaluate a single model using grouped stratified CV.

    Uses StratifiedGroupKFold with groups=course_id so no course leaks
    across folds. Per-fold: feature selection, scaling, optional SMOTE.

    Args:
        model: sklearn-compatible classifier.
        model_name: Display name.
        X: Feature DataFrame (unscaled, pre-filtered but NOT feature-selected).
        y: Target Series.
        is_binary: Binary vs multiclass.
        groups: Group labels (course_id) for StratifiedGroupKFold.
        do_feature_selection: Run per-fold feature selection.
        resampling: "none", "smote", or "borderline_smote".
        fold_selections: Pre-computed feature lists per fold (from
            precompute_fold_selections). If provided, skips per-fold selection.

    Returns metrics dict or None on failure.
    """
    n_splits = 5

    if groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_iter = list(cv.split(X, y, groups))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        cv_iter = list(cv.split(X, y))

    try:
        oof_proba = np.full(len(y), np.nan) if is_binary else np.full((len(y), y.nunique()), np.nan)
        per_course_results = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_proba": []})
        fold_feature_sets = []
        valid_folds = 0

        for fold_idx, (train_idx, test_idx) in enumerate(cv_iter):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Skip folds with no positive samples
            if y_train.sum() == 0 or y_test.sum() == 0:
                continue

            valid_folds += 1
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]

            # Per-fold feature selection (use pre-computed if available)
            if fold_selections is not None:
                selected = fold_selections[fold_idx]
            elif do_feature_selection and X_train_raw.shape[1] > 5:
                selected = sota_feature_selection(X_train_raw, y_train)
            else:
                selected = list(X_train_raw.columns)
            fold_feature_sets.append(selected)

            X_train_sel = X_train_raw[selected]
            X_test_sel = X_test_raw[selected]

            # Per-fold scaling
            scaler = StandardScaler()
            X_train_sc = pd.DataFrame(
                scaler.fit_transform(X_train_sel), columns=selected, index=X_train_sel.index,
            )
            X_test_sc = pd.DataFrame(
                scaler.transform(X_test_sel), columns=selected, index=X_test_sel.index,
            )

            # Optional resampling (binary and multi-class)
            y_train_fit = y_train
            X_train_fit = X_train_sc
            if resampling != "none":
                try:
                    n_minority = int(y_train.value_counts().min())
                    k = min(5, n_minority - 1)
                    if k >= 1:
                        if resampling == "smote":
                            from imblearn.over_sampling import SMOTE
                            sampler = SMOTE(k_neighbors=k, random_state=RANDOM_STATE)
                            X_train_fit, y_train_fit = sampler.fit_resample(X_train_sc, y_train)
                        elif resampling == "borderline_smote":
                            from imblearn.over_sampling import BorderlineSMOTE
                            sampler = BorderlineSMOTE(k_neighbors=k, random_state=RANDOM_STATE)
                            X_train_fit, y_train_fit = sampler.fit_resample(X_train_sc, y_train)
                except Exception:
                    pass  # Fall back to no resampling

            # Clone and fit (Platt-calibrate binary models on natural prevalence)
            model_fold = clone(model)
            if calibrate and is_binary and resampling == "none":
                try:
                    model_fold = CalibratedClassifierCV(model_fold, method="sigmoid", cv=3)
                except Exception:
                    model_fold = clone(model)
            model_fold.fit(X_train_fit, y_train_fit)

            # Predict
            if is_binary:
                proba = model_fold.predict_proba(X_test_sc)[:, 1]
                oof_proba[test_idx] = proba

                # Per-course tracking
                if groups is not None:
                    test_groups = groups[test_idx]
                    preds_05 = (proba >= 0.5).astype(int)
                    for cid in np.unique(test_groups):
                        mask = test_groups == cid
                        per_course_results[cid]["y_true"].extend(y_test.values[mask].tolist())
                        per_course_results[cid]["y_pred"].extend(preds_05[mask].tolist())
                        per_course_results[cid]["y_proba"].extend(proba[mask].tolist())
            else:
                proba = model_fold.predict_proba(X_test_sc)
                oof_proba[test_idx] = proba

                # Per-course tracking for fail class (class 0) — OVR approach
                if groups is not None:
                    test_groups = groups[test_idx]
                    p_fail = proba[:, 0]
                    y_fail_test = (y_test.values == 0).astype(int)
                    preds_05 = (p_fail >= 0.5).astype(int)
                    for cid in np.unique(test_groups):
                        mask = test_groups == cid
                        per_course_results[cid]["y_true"].extend(y_fail_test[mask].tolist())
                        per_course_results[cid]["y_pred"].extend(preds_05[mask].tolist())
                        per_course_results[cid]["y_proba"].extend(p_fail[mask].tolist())

        if valid_folds == 0:
            return None

        # Aggregate out-of-fold predictions
        valid_mask = ~np.isnan(oof_proba) if is_binary else ~np.isnan(oof_proba[:, 0])
        if valid_mask.sum() < 10:
            return None

        y_valid = y.values[valid_mask]

        if is_binary:
            y_proba_valid = oof_proba[valid_mask]
            y_pred = (y_proba_valid >= 0.5).astype(int)

            result = {
                "model": model_name,
                "accuracy": accuracy_score(y_valid, y_pred),
                "recall": recall_score(y_valid, y_pred, zero_division=0),
                "precision": precision_score(y_valid, y_pred, zero_division=0),
                "f1": f1_score(y_valid, y_pred, zero_division=0),
                "f2": fbeta_score(y_valid, y_pred, beta=2, zero_division=0),
                "mcc": matthews_corrcoef(y_valid, y_pred),
                "resampling": resampling,
            }

            try:
                result["roc_auc"] = roc_auc_score(y_valid, y_proba_valid)
            except ValueError:
                result["roc_auc"] = 0.5

            # Calibration quality (risk-score reliability)
            try:
                result["brier"] = float(brier_score_loss(y_valid, y_proba_valid))
                result["ece"] = _ece(y_valid, y_proba_valid)
            except Exception:
                pass
            result["calibrated"] = bool(calibrate and resampling == "none")

            # Threshold optimization
            result["thresholds"] = find_optimal_thresholds(y_valid, y_proba_valid)

            # Per-course metrics
            if per_course_results:
                course_metrics = {}
                for cid, data in per_course_results.items():
                    yt = np.array(data["y_true"])
                    yp = np.array(data["y_pred"])
                    n_fail = int(yt.sum())
                    course_metrics[int(cid)] = {
                        "n_students": len(yt),
                        "n_fail": n_fail,
                        "recall": float(recall_score(yt, yp, zero_division=0)) if n_fail > 0 else None,
                        "precision": float(precision_score(yt, yp, zero_division=0)),
                    }
                result["per_course_metrics"] = course_metrics

        else:
            y_proba_valid = oof_proba[valid_mask]
            y_pred = y_proba_valid.argmax(axis=1)

            result = {
                "model": model_name,
                "accuracy": accuracy_score(y_valid, y_pred),
                "f1_macro": f1_score(y_valid, y_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_valid, y_pred, average="weighted", zero_division=0),
                "recall_macro": recall_score(y_valid, y_pred, average="macro", zero_division=0),
                "resampling": resampling,
            }

            try:
                result["roc_auc_ovr"] = roc_auc_score(
                    y_valid, y_proba_valid, multi_class="ovr", average="weighted",
                )
            except ValueError:
                result["roc_auc_ovr"] = 0.5

            # ── Fail-class (class 0) OVR metrics ──
            # Treat as binary: fail (class 0) vs rest
            y_fail = (y_valid == 0).astype(int)
            p_fail = y_proba_valid[:, 0]
            y_fail_pred = (y_pred == 0).astype(int)

            result["fail_recall"] = float(recall_score(y_fail, y_fail_pred, zero_division=0))
            result["fail_precision"] = float(precision_score(y_fail, y_fail_pred, zero_division=0))
            result["fail_f1"] = float(f1_score(y_fail, y_fail_pred, zero_division=0))
            result["fail_f2"] = float(fbeta_score(y_fail, y_fail_pred, beta=2, zero_division=0))

            try:
                result["fail_roc_auc"] = float(roc_auc_score(y_fail, p_fail))
            except ValueError:
                result["fail_roc_auc"] = 0.5

            # Threshold optimization for fail class
            result["thresholds"] = find_optimal_thresholds(y_fail, p_fail)

            # Per-course metrics for fail class
            if per_course_results:
                course_metrics = {}
                for cid, data in per_course_results.items():
                    yt = np.array(data["y_true"])
                    yp = np.array(data["y_pred"])
                    n_fail = int(yt.sum())
                    course_metrics[int(cid)] = {
                        "n_students": len(yt),
                        "n_fail": n_fail,
                        "recall": float(recall_score(yt, yp, zero_division=0)) if n_fail > 0 else None,
                        "precision": float(precision_score(yt, yp, zero_division=0)),
                    }
                result["per_course_metrics"] = course_metrics

        # Feature importance (fit on full data with last fold's features)
        try:
            if fold_feature_sets:
                all_selected = fold_feature_sets[-1]
                scaler_full = StandardScaler()
                X_full = pd.DataFrame(
                    scaler_full.fit_transform(X[all_selected]),
                    columns=all_selected, index=X.index,
                )
                model_full = clone(model)
                model_full.fit(X_full, y)
                if hasattr(model_full, "feature_importances_"):
                    imp = pd.Series(model_full.feature_importances_, index=all_selected)
                    result["top_features"] = imp.nlargest(10).to_dict()
        except Exception:
            pass

        return result

    except Exception as e:
        print(f"    ERROR {model_name}: {e}")
        return None


# ── Optuna Hyperparameter Tuning (Phase 2) ─────────────────────────────────
def get_optuna_search_space(trial, model_name: str) -> dict:
    """Return hyperparameter search space for a model family."""
    if "XGBoost" in model_name:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 20.0),
            "eval_metric": "logloss",
            "verbosity": 0,
            "random_state": RANDOM_STATE,
            "device": "cpu",
            "tree_method": "hist",
            "n_jobs": 1,
        }
    elif "LightGBM" in model_name:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "verbosity": -1,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
    elif "RandomForest" in model_name:
        cw_choice = trial.suggest_categorical("class_weight_type", ["none", "balanced", "custom"])
        if cw_choice == "balanced":
            cw = "balanced"
        elif cw_choice == "custom":
            w = trial.suggest_float("pos_weight", 5.0, 20.0)
            cw = {0: 1, 1: w}
        else:
            cw = None
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": cw,
            "random_state": RANDOM_STATE,
            "n_jobs": 1,
        }
    elif "LogisticRegression" in model_name:
        cw_choice = trial.suggest_categorical("class_weight_type", ["none", "balanced", "custom"])
        if cw_choice == "balanced":
            cw = "balanced"
        elif cw_choice == "custom":
            w = trial.suggest_categorical("pos_weight", [5, 10, 15, 20])
            cw = {0: 1, 1: w}
        else:
            cw = None
        return {
            "C": trial.suggest_float("C", 0.01, 10.0, log=True),
            "penalty": trial.suggest_categorical("penalty", ["l1", "l2"]),
            "solver": "saga",
            "class_weight": cw,
            "max_iter": 1000,
            "random_state": RANDOM_STATE,
        }
    elif "GradientBoosting" in model_name:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": RANDOM_STATE,
        }
    else:
        return {}


def _get_model_class(model_name: str):
    """Return the model class for a given model name."""
    if "XGBoost" in model_name:
        return XGBClassifier
    elif "LightGBM" in model_name and HAS_LGBM:
        return lgb.LGBMClassifier
    elif "RandomForest" in model_name:
        return RandomForestClassifier
    elif "LogisticRegression" in model_name:
        return LogisticRegression
    elif "GradientBoosting" in model_name:
        return GradientBoostingClassifier
    return None


def optuna_tune_model(
    model_class, model_name: str,
    X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
    n_trials: int = 50,
    selected_features: list[str] | None = None,
    is_binary: bool = True,
) -> dict:
    """Optuna HPO with inner 3-fold grouped CV. Objective: F-beta(2)."""
    X_sel = X[selected_features] if selected_features else X

    def objective(trial):
        params = get_optuna_search_space(trial, model_name)
        if not params:
            return 0.0

        model = model_class(**params)
        cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        scores = []

        for train_idx, test_idx in cv.split(X_sel, y, groups):
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            if y_train.sum() == 0 or y_test.sum() == 0:
                continue

            X_train_raw, X_test_raw = X_sel.iloc[train_idx], X_sel.iloc[test_idx]

            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train_raw)
            X_te_sc = scaler.transform(X_test_raw)

            try:
                m = clone(model)
                m.fit(X_tr_sc, y_train)
                y_pred = m.predict(X_te_sc)
                if is_binary:
                    scores.append(fbeta_score(y_test, y_pred, beta=2, zero_division=0))
                else:
                    y_fail = (y_test == 0).astype(int)
                    y_fail_pred = (y_pred == 0).astype(int)
                    scores.append(fbeta_score(y_fail, y_fail_pred, beta=2, zero_division=0))
            except Exception:
                scores.append(0.0)

        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    return {
        "best_params": study.best_params,
        "best_f2": study.best_value,
        "n_trials": len(study.trials),
    }


# ── Filter assessment features ─────────────────────────────────────────────
def filter_assessment_features(X: pd.DataFrame, include: bool) -> pd.DataFrame:
    """Filter out or keep assessment-related features."""
    if include:
        return X
    # Remove columns matching assessment patterns
    cols_to_keep = []
    for col in X.columns:
        col_lower = col.lower()
        if not any(pat in col_lower for pat in ASSESSMENT_PATTERNS):
            cols_to_keep.append(col)
    return X[cols_to_keep] if cols_to_keep else X


# ── Course start computation ───────────────────────────────────────────────
def get_course_starts(df_pv: pd.DataFrame, percentile: float) -> dict:
    """Compute course start dates using given percentile."""
    return df_pv.groupby("course_id")["created_at"].quantile(percentile).to_dict()


def filter_by_cutoff(
    df_pv: pd.DataFrame, course_starts: dict, cutoff_weeks: int | str,
) -> pd.DataFrame:
    """Filter page views to temporal cutoff."""
    if cutoff_weeks == "full":
        return df_pv.copy()

    parts = []
    for course_id, start in course_starts.items():
        cutoff_date = start + timedelta(weeks=cutoff_weeks)
        df_course = df_pv[
            (df_pv["course_id"] == course_id) & (df_pv["created_at"] <= cutoff_date)
        ]
        if len(df_course) > 0:
            parts.append(df_course)

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ── Main Experiment Loop ───────────────────────────────────────────────────
def run_benchmark(phase: int = 1, quick: bool = False, course_ids: list[int] | None = None, scheme_filter: list[str] | None = None, resume: bool = False) -> None:
    """Run the full SOTA benchmark."""
    print("=" * 70)
    label = "PUC SOTA Benchmark"
    if course_ids:
        label += f" — {len(course_ids)} courses"
    print(f"{label} — Phase {phase}")
    print("=" * 70)
    start_time = time.time()

    # ── Load data ──────────────────────────────────────────────────────
    print("\n[1] Loading fixed data...")
    df_pv = pd.read_parquet(DATA_DIR / "puc_fixed_data.parquet")
    df_grades = pd.read_parquet(DATA_DIR / "puc_grades_clean.parquet")
    print(f"  Page views: {len(df_pv):,}, Grades: {len(df_grades):,}")

    # ── Course filter ──────────────────────────────────────────────────
    if course_ids:
        df_pv = df_pv[df_pv["course_id"].isin(course_ids)]
        df_grades = df_grades[df_grades["course_id"].isin(course_ids)]
        n_students = df_grades["student_id"].nunique()
        n_fail = (df_grades["grade"] < 4.0).sum()
        print(f"  Filtered to {len(course_ids)} courses: {course_ids}")
        print(f"  Page views: {len(df_pv):,}, Grades: {len(df_grades):,}")
        print(f"  Students: {n_students}, Failures: {n_fail} ({100*n_fail/len(df_grades):.1f}%)")

    # Ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df_pv["created_at"]):
        df_pv["created_at"] = pd.to_datetime(df_pv["created_at"], utc=True)

    # Grade lookup: one row per (student_id, course_id)
    grade_lookup = df_grades.set_index(["student_id", "course_id"])["grade"]

    # ── Experiment configuration ───────────────────────────────────────
    percentiles = [0.05] if quick else PERCENTILES
    cutoffs = [4] if quick else CUTOFF_WEEKS
    if scheme_filter:
        schemes = scheme_filter
    elif quick:
        schemes = ["binary_4.0", "binary_5.0"]
    else:
        schemes = CLASSIFICATION_SCHEMES
    assessment_modes = [True, False]

    models_dict = get_models()
    if not quick:
        models_dict.update(get_ensembles())
    n_models = len(models_dict)

    n_experiments = (
        len(percentiles) * len(cutoffs) * len(schemes) * len(assessment_modes) * n_models
    )
    print(f"\n[2] Experiment grid: {len(percentiles)} percentiles x {len(cutoffs)} cutoffs "
          f"x {len(schemes)} schemes x {len(assessment_modes)} assessment modes "
          f"x {n_models} models = {n_experiments} experiments")

    all_results = []
    feature_cache: dict[tuple, pd.DataFrame] = {}
    exp_count = 0

    # ── Checkpoint helper (incremental save) ──────────────────────────
    if course_ids and scheme_filter:
        out_dir = RESULTS_DIR / "7courses_multiclass"
    elif course_ids:
        out_dir = RESULTS_DIR / "7courses"
    else:
        out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: load existing results and identify completed Phase 2 configs
    completed_p2_configs: set[tuple] = set()
    if resume:
        results_file = out_dir / "benchmark_results.json"
        if results_file.exists():
            with open(results_file) as f:
                prev_data = json.load(f)
            prev_results = prev_data["results"]
            # Load all previous results
            all_results = prev_results
            exp_count = len(prev_results)
            # Identify completed Phase 2 config groups
            for r in prev_results:
                if r.get("phase") == 2:
                    completed_p2_configs.add(
                        (r["scheme"], r["percentile"], str(r["cutoff_week"]), r["include_assessment"])
                    )
            print(f"  [resume] Loaded {len(prev_results)} existing results")
            print(f"  [resume] {len(completed_p2_configs)} Phase 2 config groups already done")

    def make_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return obj

    def _clean_result(r: dict) -> dict:
        rc = {}
        for k, v in r.items():
            if k == "thresholds":
                rc[k] = {
                    tk: {mk: make_serializable(mv) for mk, mv in tv.items()}
                    for tk, tv in v.items()
                }
            elif k in ("top_features", "per_course_metrics", "optuna_best_params"):
                if isinstance(v, dict):
                    rc[k] = {
                        str(fk): make_serializable(fv) if not isinstance(fv, dict)
                        else {mk: make_serializable(mv) for mk, mv in fv.items()}
                        for fk, fv in v.items()
                    }
                else:
                    rc[k] = make_serializable(v)
            elif k == "selected_features":
                rc[k] = v
            else:
                rc[k] = make_serializable(v)
        return rc

    def save_checkpoint(label: str = "") -> None:
        results_clean = [_clean_result(r) for r in all_results]
        results_file = out_dir / "benchmark_results.json"
        elapsed = (time.time() - start_time) / 60
        with open(results_file, "w") as f:
            json.dump({
                "metadata": {
                    "phase": phase,
                    "total_experiments": exp_count,
                    "successful": len(all_results),
                    "duration_minutes": elapsed,
                    "timestamp": pd.Timestamp.now().isoformat(),
                    "course_ids": course_ids,
                    "checkpoint": label,
                },
                "results": results_clean,
            }, f, indent=2, default=make_serializable)
        print(f"  [checkpoint] {label}: {len(all_results)} results saved ({elapsed:.1f} min)", flush=True)

    # ── Outer loop: percentile + cutoff (features cached) ──────────────
    skip_phase1 = resume and any(r.get("phase") == 1 for r in all_results)
    if skip_phase1:
        print(f"\n[resume] Skipping Phase 1 — {sum(1 for r in all_results if r.get('phase')==1)} results already loaded")
    for pct_idx, percentile in enumerate(percentiles):
        course_starts = get_course_starts(df_pv, percentile)
        print(f"\n{'='*60}")
        print(f"Percentile {percentile} ({pct_idx+1}/{len(percentiles)})")

        for cut_idx, cutoff in enumerate(cutoffs):
            cache_key = (percentile, cutoff)

            # Filter and compute features (cached)
            if cache_key not in feature_cache:
                print(f"\n  Cutoff week={cutoff}: computing features...", flush=True)
                t0 = time.time()

                df_filtered = filter_by_cutoff(df_pv, course_starts, cutoff)
                if len(df_filtered) == 0:
                    print(f"    No data at cutoff={cutoff}, skipping", flush=True)
                    continue

                n_pvs = len(df_filtered)
                n_courses = df_filtered["course_id"].nunique()
                print(f"    {n_pvs:,} page views, {n_courses} courses", flush=True)

                # Determine total weeks for weekly features
                total_weeks = cutoff if isinstance(cutoff, int) else 16

                # PCT features are expensive — enable for early weeks (small data)
                # and skip for 'full' cutoff (2.3M rows x 4K resources is too slow)
                compute_pct = isinstance(cutoff, int) and cutoff <= 8
                df_features = calculate_all_features(
                    df_filtered, course_starts,
                    compute_pct=compute_pct,
                    total_weeks=total_weeks,
                    cutoff_weeks=cutoff,
                )

                if len(df_features) == 0:
                    print(f"    No features computed, skipping", flush=True)
                    continue

                # Merge grades
                df_features = df_features.merge(
                    df_grades[["student_id", "course_id", "grade", "failed"]],
                    on=["student_id", "course_id"],
                    how="inner",
                )

                # Z-normalization
                exclude = ["student_id", "course_id", "grade", "failed"]
                feat_cols = [c for c in df_features.columns if c not in exclude and df_features[c].dtype in ["float64", "int64", "float32", "int32"]]
                df_features = calculate_znorm(df_features, feat_cols)

                dt = time.time() - t0
                print(f"    Features: {df_features.shape[1]} cols, {len(df_features)} rows ({dt:.1f}s)")
                feature_cache[cache_key] = df_features

            df_features = feature_cache[cache_key]

            # ── Inner loop: scheme + assessment + model ────────────────
            if skip_phase1:
                continue  # features are cached, skip to next (pct, cutoff) for caching
            for scheme in schemes:
                # Create labels
                try:
                    y_raw = create_labels(df_features, scheme)
                except Exception as e:
                    print(f"    Scheme {scheme} error: {e}")
                    continue

                # Handle Oviedo (3 independent binary problems)
                if scheme == "oviedo":
                    sub_problems = list(y_raw.columns) if isinstance(y_raw, pd.DataFrame) else [scheme]
                else:
                    sub_problems = [scheme]

                for sub_problem in sub_problems:
                    if scheme == "oviedo":
                        y = y_raw[sub_problem]
                        scheme_label = f"oviedo_{sub_problem}"
                    else:
                        y = y_raw
                        scheme_label = scheme

                    # Skip if < 5 samples in minority class
                    if isinstance(y, pd.Series):
                        min_class = y.value_counts().min()
                        if min_class < 5:
                            print(f"    Skipping {scheme_label}: minority class has {min_class} samples")
                            continue

                    is_binary = y.nunique() == 2

                    # Groups for StratifiedGroupKFold
                    groups = df_features["course_id"].values

                    for include_assessment in assessment_modes:
                        # Prepare feature matrix (no standalone feature selection — done per-fold)
                        exclude = ["student_id", "course_id", "grade", "failed"]
                        feature_cols_all = [
                            c for c in df_features.columns
                            if c not in exclude and df_features[c].dtype in ["float64", "int64", "float32", "int32"]
                        ]
                        X = df_features[feature_cols_all].copy()
                        X = filter_assessment_features(X, include_assessment)
                        X = X.fillna(0).replace([np.inf, -np.inf], 0)

                        if X.shape[1] < 3:
                            continue

                        assess_label = "with_assess" if include_assessment else "no_assess"

                        # Pre-compute feature selection once for this (X, y) combo
                        # All 15 models share the same fold selections
                        cached_selections = precompute_fold_selections(X, y, groups)

                        for model_name, model in models_dict.items():
                            exp_count += 1
                            if exp_count % 50 == 0 or exp_count <= 5:
                                print(f"    [{exp_count}/{n_experiments}] "
                                      f"pct={percentile} wk={cutoff} "
                                      f"{scheme_label} {assess_label} {model_name}",
                                      flush=True)

                            result = evaluate_model(
                                model, model_name, X, y,
                                is_binary=is_binary,
                                groups=groups,
                                fold_selections=cached_selections,
                            )

                            if result is not None:
                                result.update({
                                    "percentile": percentile,
                                    "cutoff_week": cutoff,
                                    "scheme": scheme_label,
                                    "include_assessment": include_assessment,
                                    "n_features": X.shape[1],
                                    "n_samples": len(y),
                                    "fail_rate": float(y.mean()) if is_binary else float((y == 0).mean()),
                                    "phase": 1,
                                })
                                all_results.append(result)

            # Checkpoint after each (percentile, cutoff) block
            if all_results and exp_count % 90 == 0:
                save_checkpoint(f"phase1_pct{percentile}_wk{cutoff}")

    phase1_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Phase 1: {exp_count} experiments in {phase1_time/60:.1f} minutes")
    print(f"Successful: {len(all_results)}")
    save_checkpoint("phase1_complete")

    # ── Phase 2: Optuna Tuning + SMOTE ──────────────────────────────────
    if phase >= 2:
        if not HAS_OPTUNA:
            print("\n[!] Optuna not installed — skipping Phase 2 HPO. pip install optuna")
        else:
            print(f"\n{'='*70}")
            print("Phase 2: Optuna Hyperparameter Tuning + SMOTE")
            print("=" * 70)

            # Identify top-3 Phase 1 models for fail-targeting schemes
            tune_schemes = ["binary_4.0"]
            # Add multi-class schemes if they were run
            for mc_scheme in ["4class", "3class_marginal"]:
                if any(r.get("scheme") == mc_scheme for r in all_results):
                    tune_schemes.append(mc_scheme)

            phase1_binary40 = [
                r for r in all_results
                if r.get("scheme") in tune_schemes and r.get("phase") == 1
            ]

            if not phase1_binary40:
                print("  No fail-targeting Phase 1 results to tune")
            else:
                # Group by config key (including scheme)
                config_results = defaultdict(list)
                for r in phase1_binary40:
                    key = (r["scheme"], r["percentile"], r["cutoff_week"], r["include_assessment"])
                    config_results[key].append(r)

                for config_key, results_list in config_results.items():
                    scheme_p2, pct, cutoff, incl_assess = config_key

                    # Skip if already completed in a previous run (resume mode)
                    resume_key = (scheme_p2, pct, str(cutoff), incl_assess)
                    if resume_key in completed_p2_configs:
                        assess_label = "with_assess" if incl_assess else "no_assess"
                        print(f"\n  [skip] pct={pct} wk={cutoff} {scheme_p2} {assess_label} — already done")
                        continue

                    # Top-3 by ROC-AUC (or fail_roc_auc for multi-class)
                    sorted_results = sorted(
                        results_list,
                        key=lambda r: r.get("roc_auc", r.get("fail_roc_auc", 0)),
                        reverse=True,
                    )
                    top3 = sorted_results[:3]

                    cache_key = (pct, cutoff)
                    if cache_key not in feature_cache:
                        continue
                    df_feat = feature_cache[cache_key]

                    # Recreate X and y for this scheme
                    y_p2 = create_labels(df_feat, scheme_p2)
                    is_binary_p2 = y_p2.nunique() == 2
                    if y_p2.value_counts().min() < 5:
                        continue

                    exclude_cols = ["student_id", "course_id", "grade", "failed"]
                    feat_cols = [
                        c for c in df_feat.columns
                        if c not in exclude_cols and df_feat[c].dtype in ["float64", "int64", "float32", "int32"]
                    ]
                    X_p2 = df_feat[feat_cols].copy()
                    X_p2 = filter_assessment_features(X_p2, incl_assess)
                    X_p2 = X_p2.fillna(0).replace([np.inf, -np.inf], 0)
                    groups_p2 = df_feat["course_id"].values
                    assess_label = "with_assess" if incl_assess else "no_assess"

                    # Pre-compute feature selection ONCE for this config
                    # (avoids running it 150+ times per model inside Optuna)
                    selected_features_p2 = sota_feature_selection(X_p2, y_p2)
                    cached_selections_p2 = precompute_fold_selections(X_p2, y_p2, groups_p2)

                    for r in top3:
                        mname = r["model"]
                        model_class = _get_model_class(mname)
                        if model_class is None:
                            continue

                        roc_val = r.get("roc_auc", r.get("fail_roc_auc", 0))
                        print(f"\n  Optuna: pct={pct} wk={cutoff} {scheme_p2} {assess_label} {mname} "
                              f"(Phase 1 ROC-AUC={roc_val:.3f})")

                        n_trials = 20 if quick else 50
                        tune_result = optuna_tune_model(
                            model_class, mname, X_p2, y_p2, groups_p2,
                            n_trials=n_trials,
                            selected_features=selected_features_p2,
                            is_binary=is_binary_p2,
                        )
                        print(f"    Best F2={tune_result['best_f2']:.3f} "
                              f"({tune_result['n_trials']} trials)")

                        # Re-evaluate tuned model with outer grouped CV
                        # Remove non-model params from best_params
                        best_params = {k: v for k, v in tune_result["best_params"].items()
                                       if k not in ("class_weight_type", "pos_weight")}
                        # Reconstruct class_weight if needed
                        bp = tune_result["best_params"]
                        if "class_weight_type" in bp:
                            if bp["class_weight_type"] == "balanced":
                                best_params["class_weight"] = "balanced"
                            elif bp["class_weight_type"] == "custom" and "pos_weight" in bp:
                                best_params["class_weight"] = {0: 1, 1: bp["pos_weight"]}
                        # Add fixed params
                        if "XGBoost" in mname:
                            best_params.update({"eval_metric": "logloss", "verbosity": 0, "random_state": RANDOM_STATE,
                                                "device": "cpu", "tree_method": "hist", "n_jobs": 1})
                        elif "LightGBM" in mname:
                            best_params.update({"verbosity": -1, "random_state": RANDOM_STATE, "n_jobs": 1})
                        elif "LogisticRegression" in mname:
                            best_params.update({"solver": "saga", "max_iter": 1000, "random_state": RANDOM_STATE})
                        else:
                            best_params["random_state"] = RANDOM_STATE

                        tuned_model = model_class(**best_params)
                        tuned_name = f"{mname}_tuned"

                        result_tuned = evaluate_model(
                            tuned_model, tuned_name, X_p2, y_p2,
                            is_binary=is_binary_p2, groups=groups_p2,
                            fold_selections=cached_selections_p2,
                        )

                        if result_tuned is not None:
                            fail_rate = float(y_p2.mean()) if is_binary_p2 else float((y_p2 == 0).mean())
                            result_tuned.update({
                                "percentile": pct,
                                "cutoff_week": cutoff,
                                "scheme": scheme_p2,
                                "include_assessment": incl_assess,
                                "n_features": X_p2.shape[1],
                                "n_samples": len(y_p2),
                                "fail_rate": fail_rate,
                                "phase": 2,
                                "optuna_best_params": tune_result["best_params"],
                                "optuna_best_f2": tune_result["best_f2"],
                            })
                            all_results.append(result_tuned)
                            roc_key = "roc_auc" if is_binary_p2 else "fail_roc_auc"
                            f2_key = "f2" if is_binary_p2 else "fail_f2"
                            print(f"    Tuned ROC-AUC={result_tuned.get(roc_key, 0):.3f} "
                                  f"F2={result_tuned.get(f2_key, 0):.3f}")

                        # SMOTE experiments on tuned model (retired by default — see USE_SMOTE)
                        if USE_SMOTE and HAS_IMBLEARN:
                            for resamp in ["smote", "borderline_smote"]:
                                result_smote = evaluate_model(
                                    tuned_model, f"{tuned_name}_{resamp}", X_p2, y_p2,
                                    is_binary=is_binary_p2, groups=groups_p2,
                                    resampling=resamp,
                                    fold_selections=cached_selections_p2,
                                )
                                if result_smote is not None:
                                    result_smote.update({
                                        "percentile": pct,
                                        "cutoff_week": cutoff,
                                        "scheme": scheme_p2,
                                        "include_assessment": incl_assess,
                                        "n_features": X_p2.shape[1],
                                        "n_samples": len(y_p2),
                                        "fail_rate": fail_rate,
                                        "phase": 2,
                                        "optuna_best_params": tune_result["best_params"],
                                    })
                                    all_results.append(result_smote)
                                    print(f"    {resamp}: ROC-AUC={result_smote.get(roc_key, 0):.3f} "
                                          f"F2={result_smote.get(f2_key, 0):.3f}")

                    # Checkpoint after each config group
                    save_checkpoint(f"phase2_pct{pct}_wk{cutoff}_{scheme_p2}_{assess_label}")

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Completed total in {total_time/60:.1f} minutes")
    print(f"Successful: {len(all_results)}")

    # ── Final save + report ─────────────────────────────────────────────
    save_checkpoint("final")
    print(f"\nSaved results: {out_dir / 'benchmark_results.json'}")
    generate_report(all_results, out_dir)


def generate_report(results: list[dict], output_dir: Path) -> None:
    """Generate markdown summary report with enhanced metrics."""
    if not results:
        print("No results to report!")
        return

    exclude_keys = {"thresholds", "top_features", "selected_features", "per_course_metrics", "optuna_best_params"}
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in exclude_keys}
        for r in results
    ])

    report_lines = [
        "# PUC SOTA Benchmark Report",
        f"\nGenerated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        f"\nTotal experiments: {len(results)}",
    ]

    # Phase breakdown
    if "phase" in df.columns:
        for p in sorted(df["phase"].dropna().unique()):
            n = len(df[df["phase"] == p])
            report_lines.append(f"- Phase {int(p)}: {n} experiments")
    report_lines.append("")

    # ── Best models per scheme ─────────────────────────────────────────
    report_lines.append("## Best Models by Classification Scheme\n")

    for scheme in df["scheme"].unique():
        subset = df[df["scheme"] == scheme]
        report_lines.append(f"### {scheme}\n")

        # Determine primary metric and whether this scheme has fail-class OVR metrics
        has_fail_roc = "fail_roc_auc" in subset.columns and subset["fail_roc_auc"].notna().any()
        if "roc_auc" in subset.columns and subset["roc_auc"].notna().any():
            primary_metric = "roc_auc"
        elif has_fail_roc:
            primary_metric = "fail_roc_auc"
        elif "roc_auc_ovr" in subset.columns:
            primary_metric = "roc_auc_ovr"
        else:
            primary_metric = "accuracy"

        best = subset.sort_values(primary_metric, ascending=False).head(5)

        has_f2 = "f2" in best.columns and best["f2"].notna().any()
        has_fail_f2 = "fail_f2" in best.columns and best["fail_f2"].notna().any()

        if has_fail_roc and not has_f2:
            # Multi-class with fail-class metrics
            header = ("| Rank | Model | Pct | Week | Assess | Fail_ROC_AUC | ROC_AUC_OVR | "
                      "Fail_Recall | Fail_F2 | Accuracy |")
            sep = "|------|-------|-----|------|--------|-------------|-------------|-------------|---------|----------|"
            report_lines.append(header)
            report_lines.append(sep)
            for i, (_, row) in enumerate(best.iterrows(), 1):
                assess = "Yes" if row["include_assessment"] else "No"
                report_lines.append(
                    f"| {i} | {row['model']} | {row['percentile']} | {row['cutoff_week']} | "
                    f"{assess} | {row.get('fail_roc_auc', 0):.3f} | {row.get('roc_auc_ovr', 0):.3f} | "
                    f"{row.get('fail_recall', 0):.3f} | {row.get('fail_f2', 0):.3f} | "
                    f"{row['accuracy']:.3f} |",
                )
        else:
            # Binary or legacy multi-class
            header = (f"| Rank | Model | Pct | Week | Assess | {primary_metric.upper()} | "
                      f"{'Recall' if 'recall' in best.columns else 'F1_macro'} | "
                      f"{'F2 | ' if has_f2 else ''}Accuracy |")
            sep = ("|------|-------|-----|------|--------|---------|--------|"
                   f"{'------|' if has_f2 else ''}----------|")
            report_lines.append(header)
            report_lines.append(sep)
            for i, (_, row) in enumerate(best.iterrows(), 1):
                recall_col = "recall" if "recall" in row.index else "recall_macro"
                recall_val = row.get(recall_col, "N/A")
                assess = "Yes" if row["include_assessment"] else "No"
                recall_str = f"{recall_val:.3f}" if isinstance(recall_val, (int, float)) else str(recall_val)
                f2_str = f"{row['f2']:.3f} | " if has_f2 and pd.notna(row.get("f2")) else ("N/A | " if has_f2 else "")
                report_lines.append(
                    f"| {i} | {row['model']} | {row['percentile']} | {row['cutoff_week']} | "
                    f"{assess} | {row[primary_metric]:.3f} | "
                    f"{recall_str} | "
                    f"{f2_str}"
                    f"{row['accuracy']:.3f} |",
                )
        report_lines.append("")

    # ── Best thresholds for models with threshold optimization ─────────
    threshold_results = [r for r in results if "thresholds" in r and r["thresholds"]]
    if threshold_results:
        report_lines.append("## Threshold Optimization (Top Models)\n")

        # Sort by roc_auc for binary, fail_roc_auc for multi-class
        binary_results_sorted = sorted(
            threshold_results,
            key=lambda r: r.get("roc_auc", r.get("fail_roc_auc", 0)),
            reverse=True,
        )[:5]

        for r in binary_results_sorted:
            phase_str = f", phase={r.get('phase', 1)}" if r.get("phase", 1) != 1 else ""
            roc_val = r.get('roc_auc', r.get('fail_roc_auc', 0))
            roc_label = "ROC-AUC" if 'roc_auc' in r else "Fail-ROC-AUC"
            report_lines.append(
                f"### {r['model']} (pct={r['percentile']}, wk={r['cutoff_week']}, "
                f"{r['scheme']}, {roc_label}={roc_val:.3f}{phase_str})\n",
            )
            report_lines.append("| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |")
            report_lines.append("|-----------|-----------|--------|-----------|-----|-----|----------|-----|")

            for criterion, metrics in r["thresholds"].items():
                report_lines.append(
                    f"| {criterion} | {metrics['threshold']:.2f} | "
                    f"{metrics['recall']:.3f} | {metrics['precision']:.3f} | "
                    f"{metrics['f1']:.3f} | {metrics.get('f2', 0):.3f} | "
                    f"{metrics['accuracy']:.3f} | "
                    f"{metrics['mcc']:.3f} |",
                )
            report_lines.append("")

    # ── Per-course recall table ────────────────────────────────────────
    results_with_course = [r for r in results if "per_course_metrics" in r and r["per_course_metrics"]]
    if results_with_course:
        report_lines.append("## Per-Course Recall (Fail Class)\n")

        # Show best model per scheme that has per-course metrics
        fail_schemes = ["binary_4.0", "4class", "3class_marginal"]
        for scheme_name in fail_schemes:
            scheme_results = sorted(
                [r for r in results_with_course if r.get("scheme") == scheme_name],
                key=lambda r: r.get("roc_auc", r.get("fail_roc_auc", 0)),
                reverse=True,
            )
            if not scheme_results:
                continue
            r = scheme_results[0]
            roc_val = r.get("roc_auc", r.get("fail_roc_auc", 0))
            report_lines.append(
                f"### {scheme_name}\n"
                f"Model: **{r['model']}** (pct={r['percentile']}, wk={r['cutoff_week']}, "
                f"ROC-AUC={roc_val:.3f})\n",
            )
            report_lines.append("| Course ID | Students | Failures | Recall | Precision |")
            report_lines.append("|-----------|----------|----------|--------|-----------|")

            for cid, cm in sorted(r["per_course_metrics"].items(), key=lambda x: x[0]):
                recall_str = f"{cm['recall']:.3f}" if cm["recall"] is not None else "N/A (0 fail)"
                report_lines.append(
                    f"| {cid} | {cm['n_students']} | {cm['n_fail']} | "
                    f"{recall_str} | {cm['precision']:.3f} |",
                )
            report_lines.append("")
        report_lines.append("")

    # ── Feature importance consensus ──────────────────────────────────
    results_with_feats = [r for r in results if "top_features" in r and r["top_features"]]
    if results_with_feats:
        report_lines.append("## Feature Importance Consensus\n")
        report_lines.append("Top features across best 5 models (by frequency in top-10):\n")

        # Take top-5 models by ROC-AUC (or fail_roc_auc) that have top_features
        top5 = sorted(
            results_with_feats,
            key=lambda r: r.get("roc_auc", r.get("fail_roc_auc", 0)),
            reverse=True,
        )[:5]

        feat_counter = Counter()
        for r in top5:
            for feat_name in r["top_features"]:
                feat_counter[feat_name] += 1

        report_lines.append("| Feature | Appearances (out of 5) |")
        report_lines.append("|---------|----------------------|")
        for feat, count in feat_counter.most_common(15):
            report_lines.append(f"| {feat} | {count} |")
        report_lines.append("")

    # ── Phase 1 vs Phase 2 comparison ─────────────────────────────────
    if "phase" in df.columns and df["phase"].nunique() > 1:
        report_lines.append("## Phase 1 vs Phase 2 Comparison (binary_4.0)\n")

        df_40 = df[df["scheme"] == "binary_4.0"]
        for p in [1, 2]:
            subset = df_40[df_40["phase"] == p]
            if len(subset) == 0:
                continue
            best_row = subset.sort_values("roc_auc", ascending=False).iloc[0]
            f2_str = f", F2={best_row['f2']:.3f}" if "f2" in best_row and pd.notna(best_row.get("f2")) else ""
            resamp = f", resamp={best_row['resampling']}" if best_row.get("resampling", "none") != "none" else ""
            report_lines.append(
                f"**Phase {p}**: {best_row['model']} — "
                f"ROC-AUC={best_row['roc_auc']:.3f}, "
                f"Recall={best_row['recall']:.3f}, "
                f"Precision={best_row['precision']:.3f}"
                f"{f2_str}{resamp}",
            )
        report_lines.append("")

    # ── Cutoff week comparison ─────────────────────────────────────────
    report_lines.append("## Performance by Cutoff Week\n")
    df_phase1 = df[df["phase"] == 1] if "phase" in df.columns else df
    if "roc_auc" in df_phase1.columns and df_phase1["roc_auc"].notna().any():
        agg_cols = {"roc_auc": ["mean", "max"], "recall": ["mean", "max"], "accuracy": ["mean", "max"]}
        if "f2" in df_phase1.columns:
            agg_cols["f2"] = ["mean", "max"]
        weekly = df_phase1.groupby("cutoff_week").agg(agg_cols).round(3)
        report_lines.append("| Week | Mean ROC-AUC | Max ROC-AUC | Mean Recall | Max Recall |")
        report_lines.append("|------|-------------|-------------|-------------|------------|")
        for week in weekly.index:
            row = weekly.loc[week]
            report_lines.append(
                f"| {week} | {row[('roc_auc', 'mean')]:.3f} | {row[('roc_auc', 'max')]:.3f} | "
                f"{row[('recall', 'mean')]:.3f} | {row[('recall', 'max')]:.3f} |",
            )
        report_lines.append("")

    # ── Classification scheme comparison ───────────────────────────────
    report_lines.append("## Classification Scheme Comparison\n")
    report_lines.append("| Scheme | N_experiments | Best Metric | Best Model |")
    report_lines.append("|--------|--------------|-------------|------------|")
    for scheme in df["scheme"].unique():
        subset = df[df["scheme"] == scheme]
        if "roc_auc" in subset.columns and subset["roc_auc"].notna().any():
            best_metric = f"ROC-AUC={subset['roc_auc'].max():.3f}"
            best_model = subset.loc[subset["roc_auc"].idxmax(), "model"]
        elif "roc_auc_ovr" in subset.columns and subset["roc_auc_ovr"].notna().any():
            best_metric = f"ROC-AUC-OVR={subset['roc_auc_ovr'].max():.3f}"
            best_model = subset.loc[subset["roc_auc_ovr"].idxmax(), "model"]
        else:
            best_metric = f"Acc={subset['accuracy'].max():.3f}"
            best_model = subset.loc[subset["accuracy"].idxmax(), "model"]
        report_lines.append(f"| {scheme} | {len(subset)} | {best_metric} | {best_model} |")
    report_lines.append("")

    # ── Deployment recommendations ─────────────────────────────────────
    report_lines.append("## Deployment Recommendations\n")

    if threshold_results:
        best_threshold_results = []
        for r in threshold_results:
            for criterion, metrics in r.get("thresholds", {}).items():
                best_threshold_results.append({
                    "model": r["model"],
                    "scheme": r["scheme"],
                    "cutoff_week": r["cutoff_week"],
                    "percentile": r["percentile"],
                    "criterion": criterion,
                    **metrics,
                })

        if best_threshold_results:
            df_thresh = pd.DataFrame(best_threshold_results)

            # Aggressive: maximize recall
            aggressive = df_thresh.sort_values("recall", ascending=False).iloc[0]
            report_lines.append(f"**Aggressive** (maximize recall): {aggressive['model']} "
                              f"at t={aggressive['threshold']:.2f} — "
                              f"Recall={aggressive['recall']:.1%}, "
                              f"Precision={aggressive['precision']:.1%}")

            # Balanced: best F2 (recall-weighted)
            f2_df = df_thresh[df_thresh["criterion"] == "max_f2"]
            if len(f2_df) > 0:
                best_f2 = f2_df.sort_values("f2", ascending=False).iloc[0]
                report_lines.append(f"\n**Recall-focused** (max F2): {best_f2['model']} "
                                  f"at t={best_f2['threshold']:.2f} — "
                                  f"Recall={best_f2['recall']:.1%}, "
                                  f"Precision={best_f2['precision']:.1%}, "
                                  f"F2={best_f2['f2']:.3f}")

            # Balanced: best Youden's J
            balanced_df = df_thresh[df_thresh["criterion"] == "youden_j"]
            if len(balanced_df) > 0:
                balanced = balanced_df.sort_values("youden_j", ascending=False).iloc[0]
                report_lines.append(f"\n**Balanced** (Youden's J): {balanced['model']} "
                                  f"at t={balanced['threshold']:.2f} — "
                                  f"Recall={balanced['recall']:.1%}, "
                                  f"Precision={balanced['precision']:.1%}")

            # Conservative: best MCC
            conservative_df = df_thresh[df_thresh["criterion"] == "mcc"]
            if len(conservative_df) > 0:
                conservative = conservative_df.sort_values("mcc", ascending=False).iloc[0]
                report_lines.append(f"\n**Conservative** (max MCC): {conservative['model']} "
                                  f"at t={conservative['threshold']:.2f} — "
                                  f"Recall={conservative['recall']:.1%}, "
                                  f"Precision={conservative['precision']:.1%}")

    report_lines.append("")

    # Write report
    report_file = output_dir / "BENCHMARK_REPORT.md"
    with open(report_file, "w") as f:
        f.write("\n".join(report_lines))
    print(f"Saved report: {report_file}")


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="PUC SOTA Benchmark")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                       help="Phase 1 (fast) or Phase 2 (thorough)")
    parser.add_argument("--quick", action="store_true",
                       help="Quick smoke test (1 config)")
    parser.add_argument("--courses", type=str, default=None,
                       help="Comma-separated course IDs to filter (e.g. 54503,54529,55010)")
    parser.add_argument("--schemes", type=str, default=None,
                       help="Comma-separated schemes to run (e.g. binary_4.0,4class,3class_marginal)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint: skip Phase 1 and completed Phase 2 configs")
    args = parser.parse_args()

    course_ids = None
    if args.courses:
        course_ids = [int(c.strip()) for c in args.courses.split(",")]

    scheme_filter = None
    if args.schemes:
        scheme_filter = [s.strip() for s in args.schemes.split(",")]

    run_benchmark(phase=args.phase, quick=args.quick, course_ids=course_ids, scheme_filter=scheme_filter, resume=args.resume)


if __name__ == "__main__":
    main()
