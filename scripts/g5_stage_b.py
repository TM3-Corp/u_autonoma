#!/usr/bin/env python3
"""G5 — Stage B: model × features on R2-pooled (FROZEN).

R2-pooled ONLY (pre-committed primary). {CatBoost, XGB} × N ∈ {20, 30, 40,
full-after-corr-prefilter} × seeds {42..46}, weeks {4,8}. Shared per-(seed,fold)
ExtraTrees rankings sliced per N (identical folds+rankings across models/N within
a (week,seed)). Uncalibrated (rank metric).

Selection: highest mean AUC over seeds × weeks; tie (<0.003) → higher recall@20%
→ fewer features. Output: tier3_pooled/stageB_results.json (+ selected config).
"""
import json, time
from pathlib import Path
import numpy as np
import tier3_common as T

POOL = Path(T.POOL)
OUT = POOL / "stageB_results.json"
WEEKS = ["4", "8"]
SEEDS = [42, 43, 44, 45, 46]
MODELS = ["cat", "xgb"]
NSPECS = [("N20", 20, False), ("N30", 30, False), ("N40", 40, False), ("full", None, True)]


def run_cell(d, week, seed):
    """Return {config_id: {pooled_auc, recall20}} for all model×N configs on one (week,seed).
    Shared folds + per-fold rankings computed once, sliced per N."""
    X = d[T.MODEL_FEATURES].reset_index(drop=True)
    y = d["y"].to_numpy().astype(int)
    g = d["course_id"].to_numpy()
    folds, nsp = T.loco_splits(X, y, g, seed)

    # precompute per-fold ranking + corr-prefilter set (shared across models/N)
    fold_feats = []
    for tr, te in folds:
        if y[tr].sum() < 2:
            fold_feats.append(None)
            continue
        ranked = T.rank_features(X.iloc[tr], y[tr], seed)
        corr_set = T.corr_prefilter(X.iloc[tr])
        fold_feats.append({"ranked": ranked, "corr": corr_set})

    out = {}
    for kind in MODELS:
        for nid, N, is_full in NSPECS:
            cfg = f"{kind}_{nid}"
            oof = np.full(len(y), np.nan)
            for (tr, te), ff in zip(folds, fold_feats):
                if ff is None:
                    continue
                sel = ff["corr"] if is_full else (ff["ranked"][:N] if len(ff["ranked"]) >= N else ff["ranked"])
                m = T.make_model(kind, y[tr], seed)
                m.fit(X.iloc[tr][sel].values, y[tr])
                oof[te] = m.predict_proba(X.iloc[te][sel].values)[:, 1]
            out[cfg] = {"pooled_auc": T.pooled_auc(y, oof), "recall20": T.recall_at(y, oof, 0.20),
                        "n_feat": (len(fold_feats[0]["corr"]) if is_full and fold_feats[0] else N)}
    return out, nsp


def main():
    t0 = time.time()
    dfw = {w: T.load_week(w) for w in WEEKS}
    T.assert_rules(dfw["8"])
    r2 = {w: T.subset(dfw[w], "R2", "pooled") for w in WEEKS}
    print(f"[G5] R2-pooled: wk4 n={len(r2['4'])} f={int(r2['4'].y.sum())}, "
          f"wk8 n={len(r2['8'])} f={int(r2['8'].y.sum())}, "
          f"courses={r2['8'].course_id.nunique()}", flush=True)

    cells = {}  # cells[week][seed] = {cfg: {...}}
    for w in WEEKS:
        cells[w] = {}
        for s in SEEDS:
            res, nsp = run_cell(r2[w], w, s)
            cells[w][str(s)] = res
            print(f"[G5] wk{w} seed{s} (nsplits={nsp}): " +
                  " ".join(f"{c}={res[c]['pooled_auc']}" for c in sorted(res)) +
                  f" [{time.time()-t0:.0f}s]", flush=True)

    # aggregate per config over weeks × seeds
    configs = [f"{k}_{nid}" for k in MODELS for nid, _, _ in NSPECS]
    agg = {}
    for cfg in configs:
        aucs, recs, nf = [], [], []
        for w in WEEKS:
            for s in SEEDS:
                c = cells[w][str(s)][cfg]
                if c["pooled_auc"] is not None:
                    aucs.append(c["pooled_auc"])
                if c["recall20"] is not None:
                    recs.append(c["recall20"])
                nf.append(c["n_feat"])
        agg[cfg] = {"mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
                    "mean_recall20": round(float(np.mean(recs)), 4) if recs else None,
                    "n_feat": int(np.median(nf))}

    # selection: highest mean AUC; tie<0.003 → higher recall@20 → fewer feats
    ranked = sorted(agg.items(), key=lambda kv: (-(kv[1]["mean_auc"] or 0)))
    best_auc = ranked[0][1]["mean_auc"]
    tied = [(c, v) for c, v in agg.items() if v["mean_auc"] is not None
            and best_auc - v["mean_auc"] < 0.003]
    tied.sort(key=lambda kv: (-(kv[1]["mean_recall20"] or 0), kv[1]["n_feat"]))
    winner = tied[0][0]

    out = {"scope": "R2-pooled only (pre-committed primary)",
           "weeks": WEEKS, "seeds": SEEDS, "models": MODELS,
           "n_specs": [n[0] for n in NSPECS],
           "aggregate": agg,
           "ranking_by_mean_auc": [c for c, _ in ranked],
           "tie_threshold": 0.003,
           "tied_within_threshold": [c for c, _ in tied],
           "winner": winner,
           "winner_desc": f"{winner} ({agg[winner]})",
           "selection_rule": "highest mean AUC over weeks×seeds; tie<0.003 → higher recall@20% → fewer feats",
           "cells": cells,
           "note_n_collapse": "Model uses 23 institution-invariant features (guardrail 2); "
                              "N∈{30,40,full} therefore ≈ all features — the N grid partially "
                              "collapses (only N20 truly subsets). Logged honestly.",
           }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\n[G5] aggregate (mean AUC / rec20 / nfeat):")
    for c, _ in ranked:
        print(f"   {c}: AUC={agg[c]['mean_auc']} rec20={agg[c]['mean_recall20']} nfeat={agg[c]['n_feat']}")
    print(f"[G5] WINNER = {winner} (tie set {[c for c,_ in tied]}). wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
