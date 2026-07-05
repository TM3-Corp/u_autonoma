#!/usr/bin/env python3
"""
PUC Phase 2 — the construct-valid way to surface marginal students WITHOUT a
second hard binary at 4.5. Two complementary framings, both under LOCO CV,
top-40 SOTA features, with-assessment, calibrated (Platt) probabilities:

  1. REGRESSION-ON-GRADE: predict expected grade. Threshold-agnostic — the
     institution flags any rate it can staff. We report recall of true <4.0 and
     <4.5 students at flag-rates 10/15/20%, plus predicted-vs-observed grade
     calibration by decile.

  2. ORDINAL 3-TIER (Frank-Hall): the two ordinal boundaries ARE our thresholds.
     From calibrated binaries: P(>=4.0) and P(>=4.5) (monotone-clipped), derive
        P(fail)=1-P(>=4.0)   red    -> intervene
        P(marginal)=P(>=4.0)-P(>=4.5)  amber -> human review
        P(pass)=P(>=4.5)     green  -> monitor
     Assign tier by argmax; report tier x true-band crosstab and per-tier
     composition. The amber tier routes the irreducible-overlap cohort to review
     instead of forcing a machine call where Bayes error is high.

Output: data/puc/sota_results/few_feature_sweep/ordinal_regression.json
"""
from __future__ import annotations
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier, XGBRegressor

import puc_benchmark_sota as B

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
PERCENTILE = 0.05
CUTOFF_WEEKS = [2, 4, 6, 8, "full"]
N_SPLITS = 5
TOPK = 40
FLAG_RATES = [0.10, 0.15, 0.20]
OUT_DIR = B.RESULTS_DIR / "few_feature_sweep"
OUT_FILE = OUT_DIR / "ordinal_regression.json"


def cal_binary(Xtr, ytr, Xte, spw):
    clf = CalibratedClassifierCV(
        XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                      scale_pos_weight=spw, eval_metric="logloss",
                      random_state=B.RANDOM_STATE, verbosity=0),
        method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 70)
    print("PUC Phase 2 — regression-on-grade + ordinal 3-tier (fail/marginal/pass)")
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
        X = B.filter_assessment_features(dfe[fcols].copy(), True).fillna(0).replace([np.inf, -np.inf], 0)
        print(f"  X {X.shape} | <4.0={int(y40.sum())} <4.5={int(y45.sum())} | {time.time()-tf:.1f}s", flush=True)

        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=B.RANDOM_STATE)
        n = len(grade)
        oof_grade = np.full(n, np.nan)
        oof_p40 = np.full(n, np.nan)   # P(<4.0)
        oof_p45 = np.full(n, np.nan)   # P(<4.5)

        for tr, te in cv.split(X, y40, groups):
            ranked = B.sota_feature_selection(X.iloc[tr], pd.Series(y40[tr]), return_ranked=True)
            feats = ranked[:TOPK] if len(ranked) >= TOPK else ranked
            Xtr, Xte = X.iloc[tr][feats].values, X.iloc[te][feats].values

            reg = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                               colsample_bytree=0.8, reg_lambda=1.0,
                               random_state=B.RANDOM_STATE, verbosity=0)
            reg.fit(Xtr, grade[tr])
            oof_grade[te] = reg.predict(Xte)

            for yv, oof in ((y40, oof_p40), (y45, oof_p45)):
                ytr = yv[tr]
                if ytr.sum() < 2 or ytr.sum() == len(ytr):
                    continue
                spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
                oof[te] = cal_binary(Xtr, ytr, Xte, spw)

        m = ~np.isnan(oof_grade) & ~np.isnan(oof_p40) & ~np.isnan(oof_p45)
        g, pg = grade[m], oof_grade[m]
        p40, p45 = oof_p40[m], oof_p45[m]
        true_band = np.where(g < 4.0, "fail", np.where(g < 4.5, "marginal", "pass"))

        # --- 1. regression-on-grade: recall at flag rates (flag lowest predicted grades) ---
        order = np.argsort(pg)  # lowest predicted grade first
        reg_flags = {}
        for r in FLAG_RATES:
            k = int(round(r * len(pg)))
            flagged = np.zeros(len(pg), bool); flagged[order[:k]] = True
            rec40 = float((flagged & (g < 4.0)).sum() / max((g < 4.0).sum(), 1))
            rec45 = float((flagged & (g < 4.5)).sum() / max((g < 4.5).sum(), 1))
            prec_fail = float((flagged & (g < 4.0)).sum() / max(flagged.sum(), 1))
            reg_flags[f"{int(r*100)}pct"] = {"recall_lt40": rec40, "recall_lt45": rec45, "precision_fail": prec_fail}
        reg_auc40 = float(roc_auc_score((g < 4.0).astype(int), -pg))
        reg_auc45 = float(roc_auc_score((g < 4.5).astype(int), -pg))
        # predicted-grade calibration by quartile of predicted grade
        q = np.quantile(pg, [0, .25, .5, .75, 1])
        gcal = []
        for i in range(4):
            mm = (pg >= q[i]) & (pg <= q[i+1]) if i == 3 else (pg >= q[i]) & (pg < q[i+1])
            if mm.sum():
                gcal.append({"pred_mean": float(pg[mm].mean()), "obs_mean": float(g[mm].mean()), "n": int(mm.sum())})

        # --- 2. ordinal 3-tier (Frank-Hall) ---
        p_ge40 = 1 - p40
        p_ge45 = np.minimum(1 - p45, p_ge40)        # monotone clip
        P = np.vstack([1 - p_ge40, p_ge40 - p_ge45, p_ge45]).T   # [fail, marginal, pass]
        tier = np.array(["fail", "marginal", "pass"])[P.argmax(1)]
        # crosstab tier x true band
        bands = ["fail", "marginal", "pass"]
        crosstab = {t: {b: int(((tier == t) & (true_band == b)).sum()) for b in bands} for t in bands}
        tier_comp = {}
        for t in bands:
            sel = tier == t
            tier_comp[t] = {"n": int(sel.sum()),
                            "pct_truly_fail": float(((sel) & (true_band == "fail")).sum() / max(sel.sum(), 1)),
                            "pct_truly_marginal": float(((sel) & (true_band == "marginal")).sum() / max(sel.sum(), 1)),
                            "pct_truly_pass": float(((sel) & (true_band == "pass")).sum() / max(sel.sum(), 1))}
        # operational read: red OR amber = "needs attention"; how much of fail+marginal does it catch?
        needs = (tier == "fail") | (tier == "marginal")
        catch_fail = float((needs & (true_band == "fail")).sum() / max((true_band == "fail").sum(), 1))
        catch_marg = float((needs & (true_band == "marginal")).sum() / max((true_band == "marginal").sum(), 1))
        attention_rate = float(needs.mean())

        weeks_out[str(cutoff)] = {
            "n": int(m.sum()),
            "regression": {"auc_at_40": reg_auc40, "auc_at_45": reg_auc45,
                           "flag_rates": reg_flags, "grade_calibration_quartiles": gcal},
            "ordinal_3tier": {"crosstab_tier_x_band": crosstab, "tier_composition": tier_comp,
                              "red_amber_catch_fail": catch_fail, "red_amber_catch_marginal": catch_marg,
                              "attention_rate": attention_rate},
        }
        print(f"  reg AUC @4.0={reg_auc40:.3f} @4.5={reg_auc45:.3f} | "
              f"flag@15%: recall<4.0={reg_flags['15pct']['recall_lt40']:.2f} <4.5={reg_flags['15pct']['recall_lt45']:.2f}")
        print(f"  3-tier red+amber catches {catch_fail:.0%} of fails, {catch_marg:.0%} of marginals "
              f"at {attention_rate:.0%} attention rate")

    payload = {"metadata": {"courses": COURSE_IDS, "topk": TOPK, "cv": "LOCO group-by-course",
                            "calibration": "Platt/sigmoid", "flag_rates": FLAG_RATES,
                            "duration_minutes": (time.time() - t0) / 60},
               "weeks": weeks_out}
    with open(OUT_FILE, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"\nDone {(time.time()-t0)/60:.1f} min. Saved {OUT_FILE}")


if __name__ == "__main__":
    main()
