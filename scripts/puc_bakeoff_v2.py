#!/usr/bin/env python3
"""P2 — Pre-registered bake-off (FROZEN 10-config candidate list).

Uncalibrated (AUC and recall@capacity are rank metrics; calibration is applied
only to the P3 winner). weeks {2,4,6,8,full} x CV seeds {42..46}
x StratifiedGroupKFold(5, shuffle, seed), groups=course. Per (week,seed,fold):
compute the composite feature ranking ONCE per feature-set (clean, v2), slice
top-30/40, reuse across all models. rank-avg = mean of per-fold percentile-ranks
(scipy.stats.rankdata / n) across members.

Selection: winner = highest mean paired ΔAUC vs C1 averaged over seeds AND weeks
{2,4,8}. Tie-break (<0.003): higher mean recall@20% over same cells; then fewer
features; then simpler (single over ensemble).

Run with the catboost venv: .venv-tier1/bin/python scripts/puc_bakeoff_v2.py
Output: tier2_push/bakeoff_results.json
"""
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
CLEAN_FEAT = REPO / "data/puc/sota_results/tier1_clean/features"
V2_FEAT = REPO / "data/puc/sota_results/tier2_push/features"
OUT = REPO / "data/puc/sota_results/tier2_push/bakeoff_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
SEEDS = [42, 43, 44, 45, 46]
SEL_WEEKS = ["2", "4", "8"]
FLAG_RATES = [0.10, 0.15, 0.20]
RS = B.RANDOM_STATE  # 42, model base seed (fixed; only CV fold seed varies)

# base fits needed: (featureset, N, model)
BASE_FITS = [
    ("clean", 40, "xgb"), ("clean", 40, "cat"), ("clean", 40, "hist"),
    ("clean", 30, "xgb"), ("clean", 30, "cat"), ("clean", 30, "hist"),
    ("v2", 40, "xgb"), ("v2", 40, "cat"), ("v2", 40, "hist"),
    ("v2", 30, "cat"),
]
# config -> ('single', key) | ('ens', [keys])
CONFIGS = {
    "C1": ("single", ("clean", 40, "xgb")),
    "C2": ("single", ("clean", 40, "cat")),
    "C3": ("single", ("clean", 30, "cat")),
    "C4": ("single", ("clean", 40, "hist")),
    "C5": ("ens", [("clean", 40, "xgb"), ("clean", 40, "cat"), ("clean", 40, "hist")]),
    "C6": ("ens", [("clean", 30, "xgb"), ("clean", 30, "cat"), ("clean", 30, "hist")]),
    "C7": ("single", ("v2", 40, "xgb")),
    "C8": ("single", ("v2", 40, "cat")),
    "C9": ("single", ("v2", 30, "cat")),
    "C10": ("ens", [("v2", 40, "xgb"), ("v2", 40, "cat"), ("v2", 40, "hist")]),
}
CONFIG_DESC = {
    "C1": "XGB prod (spw) 40 clean [baseline]", "C2": "CatBoost Balanced 40 clean",
    "C3": "CatBoost Balanced 30 clean", "C4": "HistGB balanced 40 clean",
    "C5": "rank-avg(XGB,CB,HGB) 40 clean", "C6": "rank-avg(XGB,CB,HGB) 30 clean",
    "C7": "XGB prod 40 v2", "C8": "CatBoost Balanced 40 v2",
    "C9": "CatBoost Balanced 30 v2", "C10": "rank-avg(XGB,CB,HGB) 40 v2",
}
CONFIG_NFEAT = {"C1": 40, "C2": 40, "C3": 30, "C4": 40, "C5": 40, "C6": 30,
                "C7": 40, "C8": 40, "C9": 30, "C10": 40}
CONFIG_ENS = {"C1": 0, "C2": 0, "C3": 0, "C4": 0, "C5": 1, "C6": 1,
              "C7": 0, "C8": 0, "C9": 0, "C10": 1}


def load_matrix(feat_dir, wk, suffix):
    df = pd.read_parquet(feat_dir / f"week_{wk}_{suffix}.parquet")
    y = df["_y"].to_numpy().astype(int)
    g = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, g


def make_model(kind, ytr):
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    if kind == "xgb":
        return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             subsample=0.8, scale_pos_weight=spw,
                             eval_metric="logloss", verbosity=0, random_state=RS)
    if kind == "hist":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=RS)
    if kind == "cat":
        return CatBoostClassifier(auto_class_weights="Balanced", random_seed=RS,
                                  verbose=False, allow_writing_files=False)
    raise ValueError(kind)


def recall_at_flag(y, p, rate):
    k = max(1, int(np.ceil(rate * len(y))))
    flagged = np.argsort(p)[::-1][:k]
    return float(y[flagged].sum() / max(y.sum(), 1))


def max_f2(y, p):
    best = 0.0
    for t in np.unique(np.round(p, 3)):
        f = fbeta_score(y, (p >= t).astype(int), beta=2, zero_division=0)
        best = max(best, f)
    return float(best)


def metrics(y, p):
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "f2_max": round(max_f2(y, p), 4),
        "recall_at_flag": {str(r): round(recall_at_flag(y, p, r), 4) for r in FLAG_RATES},
    }


def run_week_seed(wk, seed, mats):
    """Return {config: metrics} for one (week, seed) on shared folds."""
    (Xc, yc, gc), (Xv, yv, gv) = mats["clean"], mats["v2"]
    assert np.array_equal(yc, yv) and np.array_equal(gc, gv)
    y, g = yc, gc
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = list(cv.split(Xc, y, g))

    Xset = {"clean": Xc, "v2": Xv}
    base_oof = {k: np.full(len(y), np.nan) for k in BASE_FITS}

    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        # rank ONCE per feature-set per fold
        ranked = {}
        for fs in ("clean", "v2"):
            ranked[fs] = B.sota_feature_selection(
                Xset[fs].iloc[tr], pd.Series(y[tr]), return_ranked=True)
        for (fs, N, kind) in BASE_FITS:
            r = ranked[fs]
            feats = r[:N] if len(r) >= N else r
            m = make_model(kind, y[tr])
            m.fit(Xset[fs].iloc[tr][feats].values, y[tr])
            base_oof[(fs, N, kind)][te] = m.predict_proba(Xset[fs].iloc[te][feats].values)[:, 1]

    # ensembles: per-fold percentile-rank average across members
    ens_oof = {}
    for cid, spec in CONFIGS.items():
        if spec[0] != "ens":
            continue
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            if y[tr].sum() < 2:
                continue
            ranks = []
            for key in spec[1]:
                p = base_oof[key][te]
                ranks.append(rankdata(p) / len(p))
            oof[te] = np.mean(ranks, axis=0)
        ens_oof[cid] = oof

    results = {}
    for cid, spec in CONFIGS.items():
        oof = base_oof[spec[1]] if spec[0] == "single" else ens_oof[cid]
        mask = ~np.isnan(oof)
        results[cid] = metrics(y[mask], oof[mask])
    return results


def main():
    t0 = time.time()
    out = {"config_desc": CONFIG_DESC, "seeds": SEEDS, "weeks": WEEKS,
           "selection_weeks": SEL_WEEKS, "note": "uncalibrated; rank metrics",
           "cells": {}}
    # cells[week][seed][config] = metrics
    for wk in WEEKS:
        mats = {"clean": load_matrix(CLEAN_FEAT, wk, "clean"),
                "v2": load_matrix(V2_FEAT, wk, "v2")}
        out["cells"][wk] = {}
        for seed in SEEDS:
            res = run_week_seed(wk, seed, mats)
            out["cells"][wk][str(seed)] = res
            c1 = res["C1"]["roc_auc"]
            best = max(res, key=lambda c: res[c]["roc_auc"])
            print(f"[P2] wk{wk} seed{seed}: C1={c1} best={best}({res[best]['roc_auc']}) "
                  f"| C8={res['C8']['roc_auc']} C10={res['C10']['roc_auc']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            OUT.write_text(json.dumps(out, indent=2))  # incremental

    # ---- selection: mean paired ΔAUC vs C1 over seeds x weeks{2,4,8}
    summary = {}
    for cid in CONFIGS:
        # per-week mean/sd ΔAUC over seeds (all weeks, for the table)
        per_week = {}
        for wk in WEEKS:
            deltas = [out["cells"][wk][str(s)][cid]["roc_auc"]
                      - out["cells"][wk][str(s)]["C1"]["roc_auc"] for s in SEEDS]
            per_week[wk] = {"mean_dAUC": round(float(np.mean(deltas)), 4),
                            "sd_dAUC": round(float(np.std(deltas)), 4)}
        # selection metric: mean over seeds x SEL_WEEKS
        sel_deltas, sel_rec20 = [], []
        for wk in SEL_WEEKS:
            for s in SEEDS:
                cell = out["cells"][wk][str(s)]
                sel_deltas.append(cell[cid]["roc_auc"] - cell["C1"]["roc_auc"])
                sel_rec20.append(cell[cid]["recall_at_flag"]["0.2"])
        summary[cid] = {
            "desc": CONFIG_DESC[cid], "n_feat": CONFIG_NFEAT[cid],
            "is_ensemble": bool(CONFIG_ENS[cid]),
            "sel_mean_dAUC": round(float(np.mean(sel_deltas)), 4),
            "sel_mean_recall20": round(float(np.mean(sel_rec20)), 4),
            "per_week_dAUC": per_week,
        }
    out["summary"] = summary

    # apply pre-registered selection rule
    ranked_cids = sorted(CONFIGS, key=lambda c: summary[c]["sel_mean_dAUC"], reverse=True)
    top = ranked_cids[0]
    tied = [c for c in ranked_cids
            if summary[top]["sel_mean_dAUC"] - summary[c]["sel_mean_dAUC"] < 0.003]
    if len(tied) > 1:
        tied.sort(key=lambda c: (-summary[c]["sel_mean_recall20"],
                                 summary[c]["n_feat"], summary[c]["is_ensemble"]))
        winner = tied[0]
    else:
        winner = top
    out["winner"] = winner
    out["winner_desc"] = CONFIG_DESC[winner]
    out["selection_rule"] = ("highest mean paired ΔAUC vs C1 over seeds x weeks{2,4,8}; "
                             "tie<0.003 -> higher recall@20%, fewer feats, single>ens")
    out["selection_ranking"] = [
        {"config": c, "sel_mean_dAUC": summary[c]["sel_mean_dAUC"],
         "sel_mean_recall20": summary[c]["sel_mean_recall20"]} for c in ranked_cids]
    out["tied_within_0.003"] = tied
    OUT.write_text(json.dumps(out, indent=2))

    print(f"\n[P2] WINNER = {winner} ({CONFIG_DESC[winner]})")
    print(f"[P2] sel_mean_dAUC ranking:")
    for c in ranked_cids:
        print(f"   {c:>4} dAUC={summary[c]['sel_mean_dAUC']:+.4f} "
              f"rec20={summary[c]['sel_mean_recall20']:.4f}  {CONFIG_DESC[c]}")
    print(f"[P2] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
