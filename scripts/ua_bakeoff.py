#!/usr/bin/env python3
"""UA-2 — UA mini bake-off (FROZEN 5-config candidate list).

Configs (uncalibrated):
  U1 XGB old-params, HISTORICAL global FS (importance>=0.005 + corr<=0.85) —
     the pipeline UA actually had; anchors to the T4 numbers. Baseline.
  U2 XGB + PUC-style per-fold composite top-40 (sota_feature_selection).
  U3 CatBoost default (Balanced) + per-fold sota top-40.
  U4 rank-avg(XGB, CatBoost, HistGB) + per-fold sota top-40.
  U5 CatBoost top-40 with pre_assessment INCLUDED vs EXCLUDED (isolates their
     value; FULL cutoff only — pre_assessment is full-horizon, see UA-1 leak note).

Protocol: 2 arms (KEEP 373 / DROP-A 322) x weeks {2,4,8,full} x seeds {42..46}
x {StratifiedKFold(5), StratifiedGroupKFold(5, groups=course)}. Metrics: ROC-AUC,
PR-AUC, recall@{10,20}%, F2max; paired ΔAUC vs U1 WITHIN arm. NEVER compare across
arms. Model seed fixed 42; only the CV fold seed varies.

Selection (per arm independently): highest mean paired ΔAUC vs U1 over seeds x
weeks{2,4,8}, STRATIFIED CV primary (LOCO reported alongside).

Run: .venv-tier1/bin/python scripts/ua_bakeoff.py
Output: tier2_push/ua_bakeoff_results.json
"""
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_time_limited_model as TT
import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
UA_FEAT = REPO / "data/puc/sota_results/tier2_push/ua_features"
PRE_ASSESS = REPO / "data/enriched_features/pre_assessment_features.parquet"
OUT = REPO / "data/puc/sota_results/tier2_push/ua_bakeoff_results.json"
WEEKS = ["2", "4", "8", "full"]
SEL_WEEKS = ["2", "4", "8"]
SEEDS = [42, 43, 44, 45, 46]
ARMS = {"KEEP": "arm_keep", "DROP_A": "arm_dropA"}
TOPK = 40
RS = B.RANDOM_STATE
ID = ["user_id", "course_id"]
PA_COLS = [c for c in pd.read_parquet(PRE_ASSESS).columns if c not in ID]


def load_arm(wk, arm_col):
    df = pd.read_parquet(UA_FEAT / f"week_{wk}.parquet")
    arms = pd.read_parquet(UA_FEAT / "arms.parquet")
    sub = arms[arms[arm_col]][ID + ["failed"]]
    m = df.merge(sub, on=ID, how="inner").sort_values(ID).reset_index(drop=True)
    y = m["failed"].to_numpy().astype(int)
    g = m["course_id"].to_numpy()
    X = m.drop(columns=ID + ["failed"]).reset_index(drop=True)
    return X, y, g


def make_model(kind, ytr):
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    if kind == "xgb":
        return XGBClassifier(learning_rate=0.1, max_depth=5, min_child_weight=1,
                             n_estimators=100, subsample=0.8, scale_pos_weight=spw,
                             eval_metric="logloss", verbosity=0, random_state=RS)
    if kind == "hist":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=RS)
    if kind == "cat":
        return CatBoostClassifier(auto_class_weights="Balanced", random_seed=RS,
                                  verbose=False, allow_writing_files=False)
    raise ValueError(kind)


def sota_topk(Xtr, ytr):
    r = B.sota_feature_selection(Xtr, pd.Series(ytr), return_ranked=True)
    return r[:TOPK] if len(r) >= TOPK else r


def global_fs(X, y):
    """Historical pipeline FS: importance>=0.005 then corr<=0.85 (global)."""
    cols = list(X.columns)
    sel, _ = TT.select_features_by_importance(X, y, cols, threshold=0.005)
    sel, _ = TT.remove_correlated_features(X, sel, threshold=0.85)
    return sel if len(sel) >= 3 else cols[:10]


def recall_at_flag(y, p, rate):
    k = max(1, int(np.ceil(rate * len(y))))
    return float(y[np.argsort(p)[::-1][:k]].sum() / max(y.sum(), 1))


def max_f2(y, p):
    best = 0.0
    for t in np.unique(np.round(p, 3)):
        best = max(best, fbeta_score(y, (p >= t).astype(int), beta=2, zero_division=0))
    return float(best)


def metrics(y, p):
    return {"roc_auc": round(float(roc_auc_score(y, p)), 4),
            "pr_auc": round(float(average_precision_score(y, p)), 4),
            "recall_at_10": round(recall_at_flag(y, p, 0.10), 4),
            "recall_at_20": round(recall_at_flag(y, p, 0.20), 4),
            "f2_max": round(max_f2(y, p), 4)}


def run_cell(X, y, g, feats_global, cv_kind, seed, do_u5):
    if cv_kind == "strat":
        splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y))
    else:
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = list(splitter.split(X, y, g))

    oof = {k: np.full(len(y), np.nan) for k in ["U1", "U2", "U3", "U4"]}
    memb = {k: np.full(len(y), np.nan) for k in ["xgb_s", "cat_s", "hist_s"]}
    u5 = {"incl": np.full(len(y), np.nan), "excl": np.full(len(y), np.nan)} if do_u5 else None

    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats_sota = sota_topk(X.iloc[tr], y[tr])
        # U1: global FS + XGB
        mg = make_model("xgb", y[tr]); mg.fit(X.iloc[tr][feats_global].values, y[tr])
        oof["U1"][te] = mg.predict_proba(X.iloc[te][feats_global].values)[:, 1]
        # sota-based members
        for kind, key in [("xgb", "xgb_s"), ("cat", "cat_s"), ("hist", "hist_s")]:
            m = make_model(kind, y[tr]); m.fit(X.iloc[tr][feats_sota].values, y[tr])
            memb[key][te] = m.predict_proba(X.iloc[te][feats_sota].values)[:, 1]
        oof["U2"][te] = memb["xgb_s"][te]
        oof["U3"][te] = memb["cat_s"][te]
        oof["U4"][te] = np.mean([rankdata(memb[k][te]) / len(te)
                                 for k in ("xgb_s", "cat_s", "hist_s")], axis=0)
        if do_u5:
            excl_feats = [c for c in X.columns if c not in PA_COLS]
            Xi_tr = X.iloc[tr]
            fi = sota_topk(Xi_tr, y[tr])                      # incl pre_assessment
            fe = sota_topk(Xi_tr[excl_feats], y[tr])          # excl pre_assessment
            mi = make_model("cat", y[tr]); mi.fit(X.iloc[tr][fi].values, y[tr])
            me = make_model("cat", y[tr]); me.fit(X.iloc[tr][fe].values, y[tr])
            u5["incl"][te] = mi.predict_proba(X.iloc[te][fi].values)[:, 1]
            u5["excl"][te] = me.predict_proba(X.iloc[te][fe].values)[:, 1]

    res = {}
    for k in ["U1", "U2", "U3", "U4"]:
        mask = ~np.isnan(oof[k])
        res[k] = metrics(y[mask], oof[k][mask])
        res[k]["delta_auc_vs_u1"] = round(res[k]["roc_auc"] - metrics(y[mask], oof["U1"][mask])["roc_auc"], 4)
    if do_u5:
        for k in ["incl", "excl"]:
            mask = ~np.isnan(u5[k])
            res[f"U5_{k}"] = metrics(y[mask], u5[k][mask])
        res["U5_delta_incl_minus_excl"] = round(res["U5_incl"]["roc_auc"] - res["U5_excl"]["roc_auc"], 4)
    return res


def main():
    t0 = time.time()
    out = {"arms": list(ARMS), "weeks": WEEKS, "seeds": SEEDS,
           "cv_schemes": ["strat", "loco"], "sel_weeks": SEL_WEEKS,
           "u1_note": "U1 = historical pipeline (global importance+corr FS); "
                      "U2-U5 use leak-free per-fold sota FS, so ΔAUC vs U1 is a "
                      "conservative estimate of the improvement.",
           "label_caveat_KEEP": ("target = recorded Canvas outcome; includes 51 "
                                 "active-zero enrollments whose true grades are external"),
           "cells": {}}
    for arm, col in ARMS.items():
        out["cells"][arm] = {}
        for wk in WEEKS:
            X, y, g = load_arm(wk, col)
            feats_global = global_fs(X, y)
            out["cells"][arm][wk] = {}
            do_u5 = (wk == "full")
            for cv in ["strat", "loco"]:
                out["cells"][arm][wk][cv] = {}
                for seed in SEEDS:
                    res = run_cell(X, y, g, feats_global, cv, seed, do_u5 and cv in ("strat", "loco"))
                    out["cells"][arm][wk][cv][str(seed)] = res
                    OUT.write_text(json.dumps(out, indent=2))
                u1a = out["cells"][arm][wk][cv]["42"]["U1"]["roc_auc"]
                print(f"[UA-2] {arm} wk{wk} {cv}: U1(s42)={u1a} "
                      f"U3(s42)={out['cells'][arm][wk][cv]['42']['U3']['roc_auc']} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- per-arm selection: mean ΔAUC vs U1 over seeds x SEL_WEEKS, STRAT primary
    out["selection"] = {}
    for arm in ARMS:
        summ = {}
        for cid in ["U2", "U3", "U4"]:
            d_strat, d_loco, rec20, auc_strat = [], [], [], []
            for wk in SEL_WEEKS:
                for s in SEEDS:
                    d_strat.append(out["cells"][arm][wk]["strat"][str(s)][cid]["delta_auc_vs_u1"])
                    d_loco.append(out["cells"][arm][wk]["loco"][str(s)][cid]["delta_auc_vs_u1"])
                    rec20.append(out["cells"][arm][wk]["strat"][str(s)][cid]["recall_at_20"])
                    auc_strat.append(out["cells"][arm][wk]["strat"][str(s)][cid]["roc_auc"])
            summ[cid] = {"sel_mean_dAUC_strat": round(float(np.mean(d_strat)), 4),
                         "sel_mean_dAUC_loco": round(float(np.mean(d_loco)), 4),
                         "sel_mean_recall20": round(float(np.mean(rec20)), 4),
                         "sel_mean_auc_strat": round(float(np.mean(auc_strat)), 4)}
        # interpretive: clean leak-free CatBoost(U3) vs XGB(U2), both sota per-fold
        u3u2 = [out["cells"][arm][wk]["strat"][str(s)]["U3"]["roc_auc"]
                - out["cells"][arm][wk]["strat"][str(s)]["U2"]["roc_auc"]
                for wk in SEL_WEEKS for s in SEEDS]
        summ["_interpretive_U3_vs_U2_strat"] = round(float(np.mean(u3u2)), 4)
        winner = max(["U2", "U3", "U4"], key=lambda c: summ[c]["sel_mean_dAUC_strat"])
        out["selection"][arm] = {"summary": summ, "winner": winner,
                                 "winner_dAUC_strat": summ[winner]["sel_mean_dAUC_strat"]}
        print(f"[UA-2] {arm} WINNER={winner} "
              f"(ΔAUC_strat={summ[winner]['sel_mean_dAUC_strat']:+.4f})", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[UA-2] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
