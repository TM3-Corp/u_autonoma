#!/usr/bin/env python3
"""
PUC Few-Feature Sweep — re-derive per-week PUC models under a FEW-FEATURE
methodology, for both grade thresholds (binary_4.0 and binary_4.5).

Reuses puc_benchmark_sota.py wholesale (same feature engineering, same SOTA
composite feature ranking, same grouped-by-course CV, same evaluation) and
only varies how many top-ranked features enter the model:

  - Approach 1 (fix N a priori, vary N): N in {2,3,5,8,13,21,34} + full baseline.
  - Approach 2 (optimal N): pick the smallest N where the AUC-vs-N curve plateaus
    (done in analysis, downstream of this sweep).

Per-fold safety: the composite ranking is computed on TRAIN data of each fold
only (no leakage). Because the ranking order is identical across N for a given
(week, threshold, assessment), it is computed ONCE per such block and sliced
head(N) for each N — so the expensive part runs 5x2x2 = 20 times, not per-N.

Grouped CV uses StratifiedGroupKFold(groups=course_id): a course's students are
entirely in train OR test per fold, so the out-of-fold ROC-AUC is already a
cross-course (LOCO-style) generalization estimate.

Output: data/puc/sota_results/few_feature_sweep/sweep_results.json
        (NEW dir — does NOT touch the authoritative benchmark_results.json)
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import puc_benchmark_sota as B

# ── Config ──────────────────────────────────────────────────────────────────
COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]  # the 7 benchmark courses
PERCENTILE = 0.05                 # course-start windowing (fixed: sweep is about N, not windowing)
CUTOFF_WEEKS = [2, 4, 6, 8, "full"]
THRESHOLDS = ["binary_4.0", "binary_4.5"]
N_SWEEP = [2, 3, 5, 8, 13, 21, 34]   # + a full-feature baseline (N=None)
ASSESS_MODES = [True, False]
MODELS = ["XGBoost", "XGBoost_balanced", "LightGBM", "RandomForest", "RandomForest_balanced"]
N_SPLITS = 5

OUT_DIR = B.RESULTS_DIR / "few_feature_sweep"
OUT_FILE = OUT_DIR / "sweep_results.json"


def compute_fold_rankings(X: pd.DataFrame, y: pd.Series, groups: np.ndarray) -> list[list[str]]:
    """Composite SOTA ranking per fold, computed on TRAIN only (leak-free).

    Folds match evaluate_model's: same StratifiedGroupKFold(n_splits, shuffle,
    random_state) over the same (X, y, groups) is deterministic, so index k here
    aligns with index k inside evaluate_model.
    """
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=B.RANDOM_STATE)
    rankings: list[list[str]] = []
    for train_idx, _ in cv.split(X, y, groups):
        y_tr = y.iloc[train_idx]
        X_tr = X.iloc[train_idx]
        if y_tr.sum() == 0 or X_tr.shape[1] <= 5:
            rankings.append(list(X.columns))
            continue
        rankings.append(B.sota_feature_selection(X_tr, y_tr, return_ranked=True))
    return rankings


def slim(result: dict) -> dict:
    """Keep the fields we need; drop bulky per-fold internals."""
    keep = ("model", "accuracy", "recall", "precision", "f1", "f2", "mcc",
            "roc_auc", "thresholds", "top_features", "per_course_metrics")
    return {k: result[k] for k in keep if k in result}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("PUC Few-Feature Sweep — binary_4.0 vs binary_4.5")
    print("=" * 70)

    # ── Load + filter to the 7 benchmark courses ───────────────────────────
    df_pv = pd.read_parquet(B.DATA_DIR / "puc_fixed_data.parquet")
    df_grades = pd.read_parquet(B.DATA_DIR / "puc_grades_clean.parquet")
    df_pv = df_pv[df_pv["course_id"].isin(COURSE_IDS)]
    df_grades = df_grades[df_grades["course_id"].isin(COURSE_IDS)]
    if not pd.api.types.is_datetime64_any_dtype(df_pv["created_at"]):
        df_pv["created_at"] = pd.to_datetime(df_pv["created_at"], utc=True)
    print(f"  page views: {len(df_pv):,} | grades: {len(df_grades):,} | "
          f"students: {df_grades['student_id'].nunique()} | courses: {df_grades['course_id'].nunique()}")

    course_starts = B.get_course_starts(df_pv, PERCENTILE)
    models_dict = B.get_models()
    models = {k: models_dict[k] for k in MODELS if k in models_dict}
    print(f"  models: {list(models)}")
    print(f"  N sweep: {N_SWEEP} + full baseline | thresholds: {THRESHOLDS}")

    all_rows: list[dict] = []
    n_done = 0

    for cutoff in CUTOFF_WEEKS:
        print(f"\n{'='*60}\nCutoff week = {cutoff}: computing features...", flush=True)
        t0 = time.time()
        df_filtered = B.filter_by_cutoff(df_pv, course_starts, cutoff)
        if len(df_filtered) == 0:
            print(f"  no data at cutoff {cutoff}, skipping")
            continue
        total_weeks = cutoff if isinstance(cutoff, int) else 16
        compute_pct = isinstance(cutoff, int) and cutoff <= 8
        df_feat = B.calculate_all_features(
            df_filtered, course_starts, compute_pct=compute_pct,
            total_weeks=total_weeks, cutoff_weeks=cutoff,
        )
        df_feat = df_feat.merge(
            df_grades[["student_id", "course_id", "grade", "failed"]],
            on=["student_id", "course_id"], how="inner",
        )
        exclude = ["student_id", "course_id", "grade", "failed"]
        feat_cols = [c for c in df_feat.columns
                     if c not in exclude and df_feat[c].dtype in ("float64", "int64", "float32", "int32")]
        df_feat = B.calculate_znorm(df_feat, feat_cols)
        print(f"  features: {df_feat.shape[1]} cols, {len(df_feat)} rows ({time.time()-t0:.1f}s)", flush=True)

        groups = df_feat["course_id"].values

        for scheme in THRESHOLDS:
            y = B.create_labels(df_feat, scheme)
            if not isinstance(y, pd.Series):
                continue
            min_class = int(y.value_counts().min())
            fail_rate = float(y.mean())
            if min_class < 5:
                print(f"  {scheme}: minority class {min_class} < 5, skipping")
                continue
            print(f"  {scheme}: fail_rate={fail_rate:.4f} (minority n={min_class})", flush=True)

            for include_assessment in ASSESS_MODES:
                exclude = ["student_id", "course_id", "grade", "failed"]
                fcols = [c for c in df_feat.columns
                         if c not in exclude and df_feat[c].dtype in ("float64", "int64", "float32", "int32")]
                X = df_feat[fcols].copy()
                X = B.filter_assessment_features(X, include_assessment)
                X = X.fillna(0).replace([np.inf, -np.inf], 0)
                if X.shape[1] < 3:
                    continue
                assess_label = "with_assess" if include_assessment else "no_assess"

                # Expensive ranking — once per (week, threshold, assess)
                tr0 = time.time()
                fold_rankings = compute_fold_rankings(X, y, groups)
                max_avail = int(np.median([len(r) for r in fold_rankings]))
                print(f"    [{assess_label}] ranked features per fold "
                      f"(median {max_avail}) in {time.time()-tr0:.1f}s", flush=True)

                sweep = [(n, [r[:n] for r in fold_rankings]) for n in N_SWEEP if n <= max_avail]
                sweep.append(("full", fold_rankings))  # full-feature baseline at same conditions

                for n_tag, fold_selections in sweep:
                    n_eff = X.shape[1] if n_tag == "full" else n_tag
                    for model_name, model in models.items():
                        res = B.evaluate_model(
                            model, model_name, X, y,
                            is_binary=True, groups=groups,
                            fold_selections=fold_selections,
                        )
                        n_done += 1
                        if res is None:
                            continue
                        row = slim(res)
                        row.update({
                            "cutoff_week": cutoff,
                            "scheme": scheme,
                            "threshold": 4.0 if scheme == "binary_4.0" else 4.5,
                            "include_assessment": include_assessment,
                            "n_features_req": n_tag,
                            "n_features_eff": n_eff,
                            "n_samples": int(len(y)),
                            "fail_rate": fail_rate,
                            "percentile": PERCENTILE,
                        })
                        all_rows.append(row)

                # checkpoint after each (week, threshold) finishes a block
                _save(all_rows, t_start, n_done)

    _save(all_rows, t_start, n_done, final=True)
    print(f"\nDone. {len(all_rows)} rows | {n_done} evals | {(time.time()-t_start)/60:.1f} min")
    print(f"Saved: {OUT_FILE}")


def _save(rows: list[dict], t_start: float, n_done: int, final: bool = False) -> None:
    payload = {
        "metadata": {
            "courses": COURSE_IDS,
            "percentile": PERCENTILE,
            "n_sweep": N_SWEEP,
            "thresholds": THRESHOLDS,
            "models": MODELS,
            "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS}, groups=course_id)",
            "note": "OOF ROC-AUC under grouped CV = cross-course (LOCO-style) generalization.",
            "n_rows": len(rows),
            "n_evals": n_done,
            "duration_minutes": (time.time() - t_start) / 60,
            "final": final,
        },
        "results": [_serializable(r) for r in rows],
    }
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f)


def _serializable(obj):
    if isinstance(obj, dict):
        return {k: _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


if __name__ == "__main__":
    main()
