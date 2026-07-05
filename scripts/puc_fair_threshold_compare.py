#!/usr/bin/env python3
"""
PUC Fair Threshold Comparison — 4.0 vs 4.5, done honestly.

Addresses the SOTA-review findings: the earlier 4.0-vs-4.5 AUC drop was a
TUNED-vs-UNTUNED comparison and had NO confidence intervals (7-fold LOCO with
~5-25 positives/course => sampling SD ~0.05-0.10). This script gives 4.5 a fair,
construct-valid shot and quantifies whether any drop is real.

Three fair comparisons, all under identical StratifiedGroupKFold(groups=course)
LOCO CV, identical per-fold SOTA feature basis (composite top-40 ranked on TRAIN
only, leak-free), with-assessment features:

  A. REGRESSION-THEN-THRESHOLD (the construct-valid arbiter): one XGBoost grade
     regressor, OOF predicted grade ranked ONCE, scored against BOTH labels
     (<4.0 and <4.5). Any AUC difference reflects intrinsic label separability,
     not pipeline asymmetry — there is no per-threshold tuning to be unfair about.

  B. TUNED-EQUIVALENT BINARY: XGBoost classifier per threshold with the CORRECT
     scale_pos_weight = neg/pos for THAT threshold (the fairness fix that was
     missing). Compared 4.0 vs 4.5 head to head.

  C. BOOTSTRAP CIs: student-level resampling (B=2000), paired for the 4.0-vs-4.5
     difference, to test whether any gap is distinguishable from zero. Per-week
     positive counts reported.

Plus an OVERLAP DIAGNOSTIC: OOF predicted-grade distribution of the 3 bands
(clear-fail <4.0, marginal [4.0,4.5), clear-pass >=4.5) — assumption-light test
of whether the marginal cohort is intrinsically hard (Bayes-error signature).

Output: data/puc/sota_results/few_feature_sweep/fair_compare.json  (no authoritative files touched)
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from xgboost import XGBClassifier, XGBRegressor

import puc_benchmark_sota as B

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
PERCENTILE = 0.05
CUTOFF_WEEKS = [2, 4, 6, 8, "full"]
N_SPLITS = 5
TOPK = 40              # SOTA-consensus feature count (matches dedicated FS ~33-40)
N_BOOT = 2000
RNG = np.random.RandomState(B.RANDOM_STATE)

OUT_DIR = B.RESULTS_DIR / "few_feature_sweep"
OUT_FILE = OUT_DIR / "fair_compare.json"


def boot_auc_ci(y, score, n=N_BOOT):
    """Percentile bootstrap CI for a single AUC (student-level resample)."""
    idx = np.arange(len(y))
    aucs = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        yb = y[b]
        if yb.min() == yb.max():
            continue
        aucs.append(roc_auc_score(yb, score[b]))
    a = np.array(aucs)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def boot_diff_ci(y_a, score_a, y_b, score_b, n=N_BOOT):
    """Paired bootstrap CI for AUC(a) - AUC(b) on the SAME resampled students.

    y_a/y_b are the two labelings of the SAME students; score_a/score_b the
    corresponding risk scores (for the regressor arbiter, score_a == score_b).
    """
    idx = np.arange(len(y_a))
    diffs = []
    for _ in range(n):
        bb = RNG.choice(idx, size=len(idx), replace=True)
        ya, yb = y_a[bb], y_b[bb]
        if ya.min() == ya.max() or yb.min() == yb.max():
            continue
        diffs.append(roc_auc_score(ya, score_a[bb]) - roc_auc_score(yb, score_b[bb]))
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 70)
    print("PUC Fair Threshold Comparison — 4.0 vs 4.5 (regression arbiter + tuned binary + bootstrap CIs)")
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
        grade = dfe["grade"].values.astype(float)
        y40 = (grade < 4.0).astype(int)
        y45 = (grade < 4.5).astype(int)
        fcols = [c for c in dfe.columns if c not in excl and dfe[c].dtype in ("float64", "int64", "float32", "int32")]
        X = dfe[fcols].copy()
        X = B.filter_assessment_features(X, True).fillna(0).replace([np.inf, -np.inf], 0)
        print(f"  X {X.shape} | <4.0 pos={int(y40.sum())} ({y40.mean():.3f}) | "
              f"<4.5 pos={int(y45.sum())} ({y45.mean():.3f}) | {time.time()-tf:.1f}s", flush=True)

        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=B.RANDOM_STATE)
        n = len(y40)
        oof_reg = np.full(n, np.nan)     # predicted grade
        oof_b40 = np.full(n, np.nan)     # P(fail) binary 4.0
        oof_b45 = np.full(n, np.nan)     # P(fail) binary 4.5
        per_fold_pos = []

        for fold, (tr, te) in enumerate(cv.split(X, y40, groups)):
            # per-fold SOTA feature basis (ranked on TRAIN 4.0 target, leak-free)
            ranked = B.sota_feature_selection(X.iloc[tr], pd.Series(y40[tr]), return_ranked=True)
            feats = ranked[:TOPK] if len(ranked) >= TOPK else ranked
            Xtr, Xte = X.iloc[tr][feats].values, X.iloc[te][feats].values

            # A. grade regressor (one model, threshold-agnostic ranking)
            reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=B.RANDOM_STATE, verbosity=0)
            reg.fit(Xtr, grade[tr])
            oof_reg[te] = reg.predict(Xte)

            # B. tuned-equivalent binary classifiers — correct scale_pos_weight per threshold
            for ytr_full, oof in ((y40, oof_b40), (y45, oof_b45)):
                ytr = ytr_full[tr]
                if ytr.sum() == 0 or ytr.sum() == len(ytr):
                    continue
                spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))  # neg/pos for THIS threshold
                clf = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                                    subsample=0.8, scale_pos_weight=spw, eval_metric="logloss",
                                    random_state=B.RANDOM_STATE, verbosity=0)
                clf.fit(Xtr, ytr)
                oof[te] = clf.predict_proba(Xte)[:, 1]
            per_fold_pos.append((int(y40[te].sum()), int(y45[te].sum()), int(len(te))))

        # risk score from regressor = lower predicted grade => higher risk
        risk_reg = -oof_reg
        m = ~np.isnan(oof_reg)

        def auc(yv, sv, mask):
            mm = mask & ~np.isnan(sv)
            return roc_auc_score(yv[mm], sv[mm])

        res = {
            "n_samples": int(n),
            "pos_40": int(y40.sum()), "prev_40": float(y40.mean()),
            "pos_45": int(y45.sum()), "prev_45": float(y45.mean()),
            "per_fold_pos_40_45_n": per_fold_pos,
            # A. regression-then-threshold arbiter (SAME ranking vs both labels)
            "reg_auc_at_40": auc(y40, risk_reg, m),
            "reg_auc_at_45": auc(y45, risk_reg, m),
            # B. tuned binary head-to-head
            "bin_auc_40": auc(y40, oof_b40, ~np.isnan(oof_b40)),
            "bin_auc_45": auc(y45, oof_b45, ~np.isnan(oof_b45)),
            # raw calibration (Brier) for the binary scores
            "brier_40": float(brier_score_loss(y40[~np.isnan(oof_b40)], oof_b40[~np.isnan(oof_b40)])),
            "brier_45": float(brier_score_loss(y45[~np.isnan(oof_b45)], oof_b45[~np.isnan(oof_b45)])),
        }
        # CIs
        res["reg_ci_40"] = boot_auc_ci(y40[m], risk_reg[m])
        res["reg_ci_45"] = boot_auc_ci(y45[m], risk_reg[m])
        dmean, dlo, dhi = boot_diff_ci(y40[m], risk_reg[m], y45[m], risk_reg[m])
        res["reg_diff_40_minus_45"] = {"mean": dmean, "ci95": [dlo, dhi], "significant": (dlo > 0 or dhi < 0)}
        mb = ~np.isnan(oof_b40) & ~np.isnan(oof_b45)
        dmean2, dlo2, dhi2 = boot_diff_ci(y40[mb], oof_b40[mb], y45[mb], oof_b45[mb])
        res["bin_diff_40_minus_45"] = {"mean": dmean2, "ci95": [dlo2, dhi2], "significant": (dlo2 > 0 or dhi2 < 0)}
        # overlap diagnostic: predicted grade by true band
        band = np.where(grade < 4.0, "fail", np.where(grade < 4.5, "marginal", "pass"))
        res["pred_grade_by_band"] = {
            b: {"n": int((band[m] == b).sum()),
                "pred_mean": float(np.mean(oof_reg[m][band[m] == b])),
                "pred_std": float(np.std(oof_reg[m][band[m] == b]))}
            for b in ("fail", "marginal", "pass")
        }
        weeks_out[str(cutoff)] = res
        print(f"  reg-arbiter AUC @4.0={res['reg_auc_at_40']:.3f} @4.5={res['reg_auc_at_45']:.3f} "
              f"| diff CI95 {res['reg_diff_40_minus_45']['ci95']} sig={res['reg_diff_40_minus_45']['significant']}")
        print(f"  tuned-binary AUC 4.0={res['bin_auc_40']:.3f} 4.5={res['bin_auc_45']:.3f} "
              f"| diff CI95 {res['bin_diff_40_minus_45']['ci95']} sig={res['bin_diff_40_minus_45']['significant']}")

    payload = {
        "metadata": {
            "courses": COURSE_IDS, "percentile": PERCENTILE, "topk_features": TOPK,
            "cv": f"StratifiedGroupKFold(n_splits={N_SPLITS}, groups=course_id) = LOCO-style",
            "n_boot": N_BOOT,
            "notes": "reg-arbiter = one XGB grade regressor ranked once vs both label cuts (no per-threshold tuning). "
                     "tuned-binary = XGB with correct scale_pos_weight per threshold. CIs = student-level bootstrap.",
            "duration_minutes": (time.time() - t0) / 60,
        },
        "weeks": weeks_out,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min. Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
