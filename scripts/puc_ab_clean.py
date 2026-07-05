#!/usr/bin/env python3
"""T3 — PUC A/B: old (UTC, undeduped) vs clean (local, deduped). THE key measurement.

Production config (calibrated XGB + spw=neg/pos, top-40 per-fold return_ranked
selection, LOCO StratifiedGroupKFold(5), seed 42) run TWICE per week on IDENTICAL
folds — once on old features, once on clean features (both aligned to the same
560-pair universe, same row order => same fold + bootstrap indices). Reports per
week ROC-AUC / PR-AUC / Brier / ECE for both arms and a paired bootstrap CI
(B=2000, shared resample indices) on ΔAUC and ΔPR-AUC (clean − old).

Decision rule: ADOPT clean as canonical unless some week shows a *significant*
degradation worse than −0.03 AUC (ΔAUC CI upper bound < −0.03) => BLOCKED-FOR-REVIEW.

Reads cached matrices from tier1_clean/features/week_{w}_{old,clean}.parquet.
Output: tier1_clean/ab_results.json
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "data/puc/sota_results/tier1_clean/features"
OUT = REPO / "data/puc/sota_results/tier1_clean/ab_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
TOPK, N_SPLITS, N_BOOT = 40, 5, 2000
DEGRADE_LIMIT = -0.03
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += (m.mean()) * abs(y[m].mean() - p[m].mean())
    return float(e)


def load_arm(wk, arm):
    df = pd.read_parquet(FEAT_DIR / f"week_{wk}_{arm}.parquet")
    ids = df[["student_id", "course_id"]].reset_index(drop=True)
    y = df["_y"].to_numpy().astype(int)
    groups = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, groups, ids


def oof_for_arm(X, y, groups, folds):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        ranked = B.sota_feature_selection(X.iloc[tr], pd.Series(y[tr]), return_ranked=True)
        feats = ranked[:TOPK] if len(ranked) >= TOPK else ranked
        spw = float((len(y[tr]) - y[tr].sum()) / max(y[tr].sum(), 1))
        clf = CalibratedClassifierCV(
            XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                          scale_pos_weight=spw, eval_metric="logloss", verbosity=0,
                          random_state=RS),
            method="sigmoid", cv=3)
        clf.fit(X.iloc[tr][feats].values, y[tr])
        oof[te] = clf.predict_proba(X.iloc[te][feats].values)[:, 1]
    return oof


def metrics(y, p):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(ece(y, p), 4),
    }


def paired_delta_ci(y, p_clean, p_old):
    """Paired bootstrap: shared indices; delta = clean - old for AUC and PR-AUC."""
    idx = np.arange(len(y))
    d_auc, d_pr = [], []
    for _ in range(N_BOOT):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        d_auc.append(roc_auc_score(y[b], p_clean[b]) - roc_auc_score(y[b], p_old[b]))
        d_pr.append(average_precision_score(y[b], p_clean[b]) - average_precision_score(y[b], p_old[b]))
    da, dp = np.array(d_auc), np.array(d_pr)
    return {
        "delta_auc_mean": round(float(da.mean()), 4),
        "delta_auc_ci95": [round(float(np.percentile(da, 2.5)), 4), round(float(np.percentile(da, 97.5)), 4)],
        "delta_pr_mean": round(float(dp.mean()), 4),
        "delta_pr_ci95": [round(float(np.percentile(dp, 2.5)), 4), round(float(np.percentile(dp, 97.5)), 4)],
    }


def run_week(wk):
    Xo, yo, go, ido = load_arm(wk, "old")
    Xc, yc, gc, idc = load_arm(wk, "clean")
    assert ido.equals(idc), f"week {wk}: old/clean id order mismatch"
    assert np.array_equal(yo, yc) and np.array_equal(go, gc), f"week {wk}: y/groups mismatch"

    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    folds = list(cv.split(Xo, yo, go))  # identical for both arms (same y, groups, order)

    oof_old = oof_for_arm(Xo, yo, go, folds)
    oof_clean = oof_for_arm(Xc, yc, gc, folds)
    m = ~np.isnan(oof_old) & ~np.isnan(oof_clean)
    y = yo[m]
    delta = paired_delta_ci(y, oof_clean[m], oof_old[m])
    res = {
        "n": int(m.sum()), "prevalence": round(float(y.mean()), 4),
        "old": metrics(y, oof_old[m]), "clean": metrics(y, oof_clean[m]),
        "paired": delta,
        "significant_degradation": bool(delta["delta_auc_ci95"][1] < DEGRADE_LIMIT),
    }
    print(f"[T3] wk{wk}: old AUC={res['old']['roc_auc']} PR={res['old']['pr_auc']} | "
          f"clean AUC={res['clean']['roc_auc']} PR={res['clean']['pr_auc']} | "
          f"ΔAUC={delta['delta_auc_mean']} CI{delta['delta_auc_ci95']} | "
          f"sig_degr={res['significant_degradation']}", flush=True)
    return res


def main():
    weeks = {}
    decision = "ADOPT clean as canonical"
    for wk in WEEKS:
        weeks[wk] = run_week(wk)
        # incremental
        blocked = any(w["significant_degradation"] for w in weeks.values())
        decision = "BLOCKED-FOR-REVIEW" if blocked else "ADOPT clean as canonical"
        OUT.write_text(json.dumps({
            "decision_rule": f"ADOPT unless any week ΔAUC CI upper < {DEGRADE_LIMIT}",
            "decision": decision,
            "blocked_weeks": [w for w, r in weeks.items() if r["significant_degradation"]],
            "weeks": weeks,
        }, indent=2))
    print(f"[T3] DECISION: {decision}", flush=True)
    print(f"[T3] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
