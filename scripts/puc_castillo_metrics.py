#!/usr/bin/env python3
"""
PUC — imbalance-honest metrics pack for a technical retention director.

Per cutoff week, under the PRODUCTION config (Platt-calibrated XGBoost +
scale_pos_weight, no SMOTE, per-fold leak-free top-40 SOTA features,
StratifiedGroupKFold by course = LOCO / cross-course generalization):

  Ranking (threshold-free):  ROC-AUC [95% CI], PR-AUC / Average Precision
                             [95% CI] vs its chance baseline (= prevalence).
  Operating points:          max-F1 and recall>=0.80 — precision, recall, F1,
                             F2, MCC, balanced accuracy, lift (prec/prevalence),
                             flag rate (% of cohort marked).
  Naive baselines:           flag-everyone and majority-class rows, to show the
                             metrics that expose them (precision, MCC, bal-acc).
  Calibration:               Brier, ECE (probabilities are Platt-calibrated).

Output: data/puc/sota_results/few_feature_sweep/castillo_metrics.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, f1_score, fbeta_score,
                             matthews_corrcoef, precision_score, recall_score,
                             balanced_accuracy_score, accuracy_score)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import puc_benchmark_sota as B

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
PERCENTILE = 0.05
CUTOFF_WEEKS = [2, 4, 6, 8, "full"]
N_SPLITS = 5
TOPK = 40
N_BOOT = 2000
RNG = np.random.RandomState(B.RANDOM_STATE)
OUT_FILE = B.RESULTS_DIR / "few_feature_sweep" / "castillo_metrics.json"


def boot_ci(y, p, fn, n=N_BOOT):
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    v = np.array(vals)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def op_metrics(y, p, thr):
    yp = (p >= thr).astype(int)
    prev = float(y.mean())
    prec = precision_score(y, yp, zero_division=0)
    return {
        "threshold": float(thr),
        "flag_rate": float(yp.mean()),
        "recall": float(recall_score(y, yp, zero_division=0)),
        "precision": float(prec),
        "f1": float(f1_score(y, yp, zero_division=0)),
        "f2": float(fbeta_score(y, yp, beta=2, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, yp)),
        "balanced_accuracy": float(balanced_accuracy_score(y, yp)),
        "accuracy": float(accuracy_score(y, yp)),
        "lift": float(prec / prev) if prev > 0 else None,
    }


def ece(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum():
            e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def main():
    t0 = time.time()
    print("PUC Castillo metrics — production config, LOCO CV")
    df_pv = pd.read_parquet(B.DATA_DIR / "puc_fixed_data.parquet")
    df_grades = pd.read_parquet(B.DATA_DIR / "puc_grades_clean.parquet")
    df_pv = df_pv[df_pv["course_id"].isin(COURSE_IDS)].copy()
    df_grades = df_grades[df_grades["course_id"].isin(COURSE_IDS)]
    if not pd.api.types.is_datetime64_any_dtype(df_pv["created_at"]):
        df_pv["created_at"] = pd.to_datetime(df_pv["created_at"], utc=True)
    course_starts = B.get_course_starts(df_pv, PERCENTILE)

    weeks_out = {}
    for cutoff in CUTOFF_WEEKS:
        print(f"\nWeek {cutoff}: features...", flush=True)
        dff = B.filter_by_cutoff(df_pv, course_starts, cutoff)
        if len(dff) == 0:
            continue
        total_weeks = cutoff if isinstance(cutoff, int) else 16
        compute_pct = isinstance(cutoff, int) and cutoff <= 8
        dfe = B.calculate_all_features(dff, course_starts, compute_pct=compute_pct,
                                       total_weeks=total_weeks, cutoff_weeks=cutoff)
        dfe = dfe.merge(df_grades[["student_id", "course_id", "grade", "failed"]],
                        on=["student_id", "course_id"], how="inner")
        excl = ["student_id", "course_id", "grade", "failed"]
        fcols = [c for c in dfe.columns if c not in excl and dfe[c].dtype in ("float64", "int64", "float32", "int32")]
        dfe = B.calculate_znorm(dfe, fcols)
        groups = dfe["course_id"].values
        y = (dfe["grade"].values < 4.0).astype(int)
        fcols = [c for c in dfe.columns if c not in excl and dfe[c].dtype in ("float64", "int64", "float32", "int32")]
        X = B.filter_assessment_features(dfe[fcols].copy(), True).fillna(0).replace([np.inf, -np.inf], 0)

        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=B.RANDOM_STATE)
        oof = np.full(len(y), np.nan)
        for tr, te in cv.split(X, y, groups):
            ytr = y[tr]
            if ytr.sum() < 2:
                continue
            ranked = B.sota_feature_selection(X.iloc[tr], pd.Series(ytr), return_ranked=True)
            feats = ranked[:TOPK] if len(ranked) >= TOPK else ranked
            spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
            clf = CalibratedClassifierCV(
                XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                              scale_pos_weight=spw, eval_metric="logloss",
                              random_state=B.RANDOM_STATE, verbosity=0),
                method="sigmoid", cv=3)
            clf.fit(X.iloc[tr][feats].values, ytr)
            oof[te] = clf.predict_proba(X.iloc[te][feats].values)[:, 1]

        m = ~np.isnan(oof)
        yv, pv = y[m], oof[m]
        prev = float(yv.mean())

        # op points: max-F1 and smallest threshold reaching recall>=0.80
        ths = np.unique(np.round(pv, 3))
        f1s = [(f1_score(yv, (pv >= t).astype(int), zero_division=0), t) for t in ths]
        t_f1 = max(f1s)[1]
        rec_ths = [t for t in sorted(ths, reverse=True)
                   if recall_score(yv, (pv >= t).astype(int), zero_division=0) >= 0.80]
        t_r80 = rec_ths[0] if rec_ths else float(ths.min())

        weeks_out[str(cutoff)] = {
            "n": int(m.sum()), "positives": int(yv.sum()), "prevalence": prev,
            "roc_auc": float(roc_auc_score(yv, pv)),
            "roc_auc_ci95": boot_ci(yv, pv, roc_auc_score),
            "pr_auc": float(average_precision_score(yv, pv)),
            "pr_auc_ci95": boot_ci(yv, pv, average_precision_score),
            "pr_auc_chance_baseline": prev,
            "brier": float(brier_score_loss(yv, pv)),
            "ece": ece(yv, pv),
            "op_max_f1": op_metrics(yv, pv, t_f1),
            "op_recall80": op_metrics(yv, pv, t_r80),
            "baseline_flag_everyone": op_metrics(yv, pv, -1.0),
            "baseline_majority_class": {
                "accuracy": float(1 - prev), "recall": 0.0, "precision": 0.0,
                "f1": 0.0, "mcc": 0.0, "balanced_accuracy": 0.5,
            },
        }
        w = weeks_out[str(cutoff)]
        print(f"  AUC={w['roc_auc']:.3f} {w['roc_auc_ci95']} | PR-AUC={w['pr_auc']:.3f} "
              f"(chance {prev:.3f}, x{w['pr_auc']/prev:.1f}) | maxF1={w['op_max_f1']['f1']:.3f} "
              f"MCC={w['op_max_f1']['mcc']:.3f}", flush=True)

    payload = {"metadata": {"config": "Platt-calibrated XGBoost + scale_pos_weight, no SMOTE, "
                                      "top-40 per-fold SOTA features, LOCO (group-by-course) CV",
                            "courses": COURSE_IDS, "n_boot": N_BOOT, "target": "grade < 4.0",
                            "duration_minutes": (time.time() - t0) / 60},
               "weeks": weeks_out}
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nDone {(time.time()-t0)/60:.1f} min -> {OUT_FILE}")


if __name__ == "__main__":
    main()
