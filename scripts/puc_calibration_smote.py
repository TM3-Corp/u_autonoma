#!/usr/bin/env python3
"""
PUC SOTA risk-score plumbing — close the two real gaps the SOTA review flagged:
  (1) NO probability calibration for tree models -> add Platt/sigmoid + report Brier/ECE.
  (2) SMOTE reliance -> ablate; the 2022-2026 risk-scoring consensus says resampling
      does not improve AUC and harms calibration.

On the production target (binary <4.0), 7 courses / 560 students, LOCO group-by-course
CV, per-fold SOTA top-40 feature basis (leak-free). Three configs per week:
  A. raw_weighted  : XGBoost with scale_pos_weight = neg/pos (no calibration)
  B. cal_weighted  : same, wrapped in CalibratedClassifierCV(method='sigmoid', cv=3)
                     -- Platt, not isotonic (isotonic overfits at this minority count)
  C. smote         : XGBoost (no class weight) on SMOTE-resampled train folds

Metrics (OOF): ROC-AUC (discrimination), Brier + ECE (calibration). Calibration is
monotonic so AUC is ~unchanged by B; the point is Brier/ECE. Reliability bins for wk8.

Output: data/puc/sota_results/few_feature_sweep/calibration_smote.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

import puc_benchmark_sota as B

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
PERCENTILE = 0.05
CUTOFF_WEEKS = [2, 4, 6, 8, "full"]
N_SPLITS = 5
TOPK = 40
OUT_DIR = B.RESULTS_DIR / "few_feature_sweep"
OUT_FILE = OUT_DIR / "calibration_smote.json"


def ece(y, p, n_bins=10):
    """Expected Calibration Error (equal-width bins). Noisy at small N -> few bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        e += abs(p[m].mean() - y[m].mean()) * m.sum() / len(y)
    return float(e)


def reliability(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out.append({"bin": [round(lo, 2), round(hi, 2)], "n": int(m.sum()),
                    "conf": float(p[m].mean()), "obs": float(y[m].mean())})
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 70)
    print("PUC calibration + SMOTE ablation (target <4.0, LOCO CV)")
    print("=" * 70)

    df_pv = pd.read_parquet(B.DATA_DIR / "puc_fixed_data.parquet")
    df_grades = pd.read_parquet(B.DATA_DIR / "puc_grades_clean.parquet")
    df_pv = df_pv[df_pv["course_id"].isin(COURSE_IDS)].copy()
    df_grades = df_grades[df_grades["course_id"].isin(COURSE_IDS)]
    if not pd.api.types.is_datetime64_any_dtype(df_pv["created_at"]):
        df_pv["created_at"] = pd.to_datetime(df_pv["created_at"], utc=True)
    course_starts = B.get_course_starts(df_pv, PERCENTILE)

    weeks_out = {}
    for cutoff in CUTOFF_WEEKS:
        print(f"\n{'='*60}\nWeek {cutoff}: features...", flush=True)
        tf = time.time()
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
        print(f"  X {X.shape} | pos={int(y.sum())} ({y.mean():.3f}) | {time.time()-tf:.1f}s", flush=True)

        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=B.RANDOM_STATE)
        n = len(y)
        oof = {k: np.full(n, np.nan) for k in ("raw", "cal", "smote")}

        for tr, te in cv.split(X, y, groups):
            ytr = y[tr]
            if ytr.sum() < 2:
                continue
            ranked = B.sota_feature_selection(X.iloc[tr], pd.Series(ytr), return_ranked=True)
            feats = ranked[:TOPK] if len(ranked) >= TOPK else ranked
            Xtr, Xte = X.iloc[tr][feats].values, X.iloc[te][feats].values
            spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))

            base = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                                 scale_pos_weight=spw, eval_metric="logloss",
                                 random_state=B.RANDOM_STATE, verbosity=0)
            base.fit(Xtr, ytr)
            oof["raw"][te] = base.predict_proba(Xte)[:, 1]

            try:
                cal = CalibratedClassifierCV(
                    XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                                  scale_pos_weight=spw, eval_metric="logloss",
                                  random_state=B.RANDOM_STATE, verbosity=0),
                    method="sigmoid", cv=3)
                cal.fit(Xtr, ytr)
                oof["cal"][te] = cal.predict_proba(Xte)[:, 1]
            except Exception as e:
                print(f"    cal failed: {e}")

            try:
                from imblearn.over_sampling import SMOTE
                k = min(5, int(ytr.sum()) - 1)
                if k >= 1:
                    Xs, ys = SMOTE(k_neighbors=k, random_state=B.RANDOM_STATE).fit_resample(Xtr, ytr)
                    sm = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                                       eval_metric="logloss", random_state=B.RANDOM_STATE, verbosity=0)
                    sm.fit(Xs, ys)
                    oof["smote"][te] = sm.predict_proba(Xte)[:, 1]
            except Exception as e:
                print(f"    smote failed: {e}")

        res = {"n": int(n), "pos": int(y.sum()), "prev": float(y.mean()), "configs": {}}
        for k, p in oof.items():
            mm = ~np.isnan(p)
            if mm.sum() < 10:
                continue
            res["configs"][k] = {
                "auc": float(roc_auc_score(y[mm], p[mm])),
                "brier": float(brier_score_loss(y[mm], p[mm])),
                "ece": ece(y[mm], p[mm]),
            }
        if str(cutoff) == "8":
            mm = ~np.isnan(oof["raw"])
            res["reliability_wk8_raw"] = reliability(y[mm], oof["raw"][mm])
            mm = ~np.isnan(oof["cal"])
            res["reliability_wk8_cal"] = reliability(y[mm], oof["cal"][mm])
        weeks_out[str(cutoff)] = res
        c = res["configs"]
        def fmt(k): return (f"AUC={c[k]['auc']:.3f} Brier={c[k]['brier']:.3f} ECE={c[k]['ece']:.3f}"
                            if k in c else "n/a")
        print(f"  raw:   {fmt('raw')}")
        print(f"  cal:   {fmt('cal')}")
        print(f"  smote: {fmt('smote')}")

    payload = {
        "metadata": {"courses": COURSE_IDS, "percentile": PERCENTILE, "topk": TOPK, "target": "<4.0",
                     "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS}, groups=course)",
                     "calibration": "Platt/sigmoid via CalibratedClassifierCV(cv=3)",
                     "duration_minutes": (time.time() - t0) / 60},
        "weeks": weeks_out,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nDone {(time.time()-t0)/60:.1f} min. Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
