#!/usr/bin/env python3
"""Explore the promising combination: CatBoost/XGBoost x feature-count x recall.

Synthesis of two findings:
  - CatBoost > XGBoost (this session's T6)
  - fewer top features (~15-20) may match 40 and help operational recall (Ignacio)

Clean PUC data, LOCO 5-fold, per-fold leak-free ranking (computed ONCE per fold,
then sliced to N in {15,20,30,40}). Metrics: ROC-AUC, PR-AUC (ranking) AND the
operational metric that matters for early warning: recall at a fixed review
capacity (flag the top 10/15/20% by risk -> what fraction of true fails caught),
which is threshold-free and calibration-invariant. Weeks 4, 8, full.

Run with the catboost venv:  .venv-tier1/bin/python scripts/puc_promising_explore.py
Output: tier1_clean/promising_explore.json
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "data/puc/sota_results/tier1_clean/features"
OUT = REPO / "data/puc/sota_results/tier1_clean/promising_explore.json"
WEEKS = ["4", "8", "full"]
NS = [15, 20, 30, 40]
FLAG_RATES = [0.10, 0.15, 0.20]
RS = B.RANDOM_STATE


def load_week(wk):
    df = pd.read_parquet(FEAT_DIR / f"week_{wk}_clean.parquet")
    y = df["_y"].to_numpy().astype(int)
    g = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, g


def make(kind, ytr):
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    if kind == "xgb":
        return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                             scale_pos_weight=spw, eval_metric="logloss", verbosity=0, random_state=RS)
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=RS, verbose=False,
                              allow_writing_files=False)


def recall_at_flag(y, p, rate):
    k = max(1, int(np.ceil(rate * len(y))))
    flagged = np.argsort(p)[::-1][:k]
    return float(y[flagged].sum() / max(y.sum(), 1))


def max_f2(y, p):
    best = 0.0
    for t in np.unique(np.round(p, 3)):
        f = fbeta_score(y, (p >= t).astype(int), beta=2, zero_division=0)
        if f > best:
            best = f
    return float(best)


def run_week(wk):
    X, y, g = load_week(wk)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    folds = list(cv.split(X, y, g))

    # rank features ONCE per fold (the expensive step), reuse for all N
    fold_rank = []
    for tr, te in folds:
        ranked = B.sota_feature_selection(X.iloc[tr], pd.Series(y[tr]), return_ranked=True)
        fold_rank.append(ranked)

    out = {}
    for kind in ("xgb", "catboost"):
        for N in NS:
            oof = np.full(len(y), np.nan)
            for (tr, te), ranked in zip(folds, fold_rank):
                if y[tr].sum() < 2:
                    continue
                feats = ranked[:N] if len(ranked) >= N else ranked
                m = make(kind, y[tr])
                m.fit(X.iloc[tr][feats].values, y[tr])
                oof[te] = m.predict_proba(X.iloc[te][feats].values)[:, 1]
            mask = ~np.isnan(oof)
            yv, pv = y[mask], oof[mask]
            out[f"{kind}_N{N}"] = {
                "roc_auc": round(float(roc_auc_score(yv, pv)), 4),
                "pr_auc": round(float(average_precision_score(yv, pv)), 4),
                "f2_max": round(max_f2(yv, pv), 4),
                "recall_at_flag": {str(r): round(recall_at_flag(yv, pv, r), 4) for r in FLAG_RATES},
            }
    return out


def main():
    res = {"config": {"data": "clean PUC", "cv": "LOCO5", "Ns": NS,
                      "flag_rates": FLAG_RATES, "note": "uncalibrated (ranking comparison)"},
           "weeks": {}}
    for wk in WEEKS:
        res["weeks"][wk] = run_week(wk)
        OUT.write_text(json.dumps(res, indent=2))
        # console summary
        w = res["weeks"][wk]
        print(f"\n=== week {wk} ===")
        print(f"{'config':<14}{'AUC':>7}{'PR-AUC':>8}{'F2max':>7}{'rec@10%':>9}{'rec@20%':>9}")
        for k, v in w.items():
            print(f"{k:<14}{v['roc_auc']:>7}{v['pr_auc']:>8}{v['f2_max']:>7}"
                  f"{v['recall_at_flag']['0.1']:>9}{v['recall_at_flag']['0.2']:>9}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
