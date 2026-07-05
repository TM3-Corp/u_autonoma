#!/usr/bin/env python3
"""P3 — Confirmatory run of the P2 winner (the ONLY quotable PUC numbers).

Reads the winner from tier2_push/bakeoff_results.json. Per week {2,4,6,8,full}:
 1. Nested CV: outer LOCO StratifiedGroupKFold(5, seed 42). Per outer-train:
    top-N leak-free ranking + (if winner involves CatBoost) inner 3-fold Optuna
    30 trials on the CatBoost member (depth 4-8, lr log .01-.3, l2 1-10,
    iters 100-500; F2) -> fit on outer-train -> predict outer-test.
 2. 5-seed bagging: fit each member with model seeds {42..46}, average probs.
 3. Platt calibration (sigmoid, cv=3) around the seed-42 per-fold model for
    probability-quality metrics; AUC reported raw-bagged AND calibrated.
 4. ROC-AUC + PR-AUC bootstrap CI (B=2000, seed 42), Brier, ECE, capacity curve
    recall@{5,10,15,20,25}%.
Leak guard: if nested AUC exceeds the bake-off seed-mean by >0.02 -> flag + STOP.

Run: .venv-tier1/bin/python scripts/puc_confirmatory_v2.py
Output: tier2_push/confirmatory_results.json
"""
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, fbeta_score)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
REPO = Path(__file__).resolve().parents[1]
CLEAN_FEAT = REPO / "data/puc/sota_results/tier1_clean/features"
V2_FEAT = REPO / "data/puc/sota_results/tier2_push/features"
BAKEOFF = REPO / "data/puc/sota_results/tier2_push/bakeoff_results.json"
OUT = REPO / "data/puc/sota_results/tier2_push/confirmatory_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
SEEDS = [42, 43, 44, 45, 46]
FLAG_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
N_TRIALS, N_BOOT = 30, 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)

# same frozen config table as P2
CONFIGS = {
    "C1": ("single", ("clean", 40, "xgb")), "C2": ("single", ("clean", 40, "cat")),
    "C3": ("single", ("clean", 30, "cat")), "C4": ("single", ("clean", 40, "hist")),
    "C5": ("ens", [("clean", 40, "xgb"), ("clean", 40, "cat"), ("clean", 40, "hist")]),
    "C6": ("ens", [("clean", 30, "xgb"), ("clean", 30, "cat"), ("clean", 30, "hist")]),
    "C7": ("single", ("v2", 40, "xgb")), "C8": ("single", ("v2", 40, "cat")),
    "C9": ("single", ("v2", 30, "cat")),
    "C10": ("ens", [("v2", 40, "xgb"), ("v2", 40, "cat"), ("v2", 40, "hist")]),
}


def load_matrix(featureset, wk):
    fdir = V2_FEAT if featureset == "v2" else CLEAN_FEAT
    suf = "v2" if featureset == "v2" else "clean"
    df = pd.read_parquet(fdir / f"week_{wk}_{suf}.parquet")
    y = df["_y"].to_numpy().astype(int)
    g = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, g


def topn(Xtr, ytr, N):
    r = B.sota_feature_selection(Xtr, pd.Series(ytr), return_ranked=True)
    return r[:N] if len(r) >= N else r


def make_model(kind, ytr, seed, cat_params=None):
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    if kind == "xgb":
        return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                             subsample=0.8, scale_pos_weight=spw,
                             eval_metric="logloss", verbosity=0, random_state=seed)
    if kind == "hist":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
    if kind == "cat":
        p = dict(cat_params or {})
        p.update(auto_class_weights="Balanced", random_seed=seed,
                 verbose=False, allow_writing_files=False)
        return CatBoostClassifier(**p)
    raise ValueError(kind)


def tune_catboost(Xf, y, g):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)
    splits = list(inner.split(Xf, y, g))

    def objective(trial):
        params = {"depth": trial.suggest_int("depth", 4, 8),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                  "iterations": trial.suggest_int("iterations", 100, 500)}
        scores = []
        for tr, va in splits:
            if y[tr].sum() < 2:
                continue
            m = make_model("cat", y[tr], RS, params)
            m.fit(Xf[tr], y[tr])
            p = m.predict_proba(Xf[va])[:, 1]
            scores.append(fbeta_score(y[va], (p >= 0.5).astype(int), beta=2, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def recall_at_flag(y, p, rate):
    k = max(1, int(np.ceil(rate * len(y))))
    flagged = np.argsort(p)[::-1][:k]
    return float(y[flagged].sum() / max(y.sum(), 1))


def boot_ci(y, p, fn):
    idx = np.arange(len(y)); vals = []
    for _ in range(N_BOOT):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    v = np.array(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def run_week(wk, featureset, N, members, has_cat):
    X, y, g = load_matrix(featureset, wk)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    folds = list(outer.split(X, y, g))

    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    cat_params_log = []

    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = topn(X.iloc[tr], y[tr], N)
        Xtr_f = X.iloc[tr][feats].values
        Xte_f = X.iloc[te][feats].values
        cat_params = tune_catboost(Xtr_f, y[tr], g[tr]) if has_cat else None
        if has_cat:
            cat_params_log.append(cat_params)

        raw_members, cal_members = [], []
        for kind in members:
            cp = cat_params if kind == "cat" else None
            # raw-bagged over model seeds
            preds = []
            for s in SEEDS:
                m = make_model(kind, y[tr], s, cp)
                m.fit(Xtr_f, y[tr])
                preds.append(m.predict_proba(Xte_f)[:, 1])
            raw_members.append(np.mean(preds, axis=0))
            # calibrated (seed 42) for probability quality
            base = make_model(kind, y[tr], RS, cp)
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(Xtr_f, y[tr])
            cal_members.append(cal.predict_proba(Xte_f)[:, 1])

        if len(members) == 1:
            oof_raw[te] = raw_members[0]
            oof_cal[te] = cal_members[0]
        else:
            oof_raw[te] = np.mean([rankdata(r) / len(r) for r in raw_members], axis=0)
            oof_cal[te] = np.mean(cal_members, axis=0)

    m = ~np.isnan(oof_raw)
    yv, praw, pcal = y[m], oof_raw[m], oof_cal[m]
    res = {
        "n_eval": int(m.sum()), "prevalence": round(float(yv.mean()), 4),
        "roc_auc_raw_bagged": round(float(roc_auc_score(yv, praw)), 4),
        "roc_auc_raw_ci95": boot_ci(yv, praw, roc_auc_score),
        "pr_auc_raw_bagged": round(float(average_precision_score(yv, praw)), 4),
        "pr_auc_raw_ci95": boot_ci(yv, praw, average_precision_score),
        "roc_auc_calibrated": round(float(roc_auc_score(yv, pcal)), 4),
        "pr_auc_calibrated": round(float(average_precision_score(yv, pcal)), 4),
        "brier_calibrated": round(float(brier_score_loss(yv, pcal)), 4),
        "ece_calibrated": round(ece(yv, pcal), 4),
        "capacity_curve": {str(r): round(recall_at_flag(yv, praw, r), 4) for r in FLAG_RATES},
        "catboost_params_per_fold": cat_params_log if has_cat else None,
    }
    return res


def main():
    t0 = time.time()
    bo = json.loads(BAKEOFF.read_text())
    winner = bo["winner"]
    spec = CONFIGS[winner]
    if spec[0] == "single":
        featureset, N, kind = spec[1]
        members = [kind]
    else:
        featureset, N = spec[1][0][0], spec[1][0][1]
        members = [k[2] for k in spec[1]]
    has_cat = "cat" in members
    print(f"[P3] winner={winner} ({bo['winner_desc']}) featureset={featureset} "
          f"N={N} members={members} has_cat={has_cat}", flush=True)

    out = {"winner": winner, "winner_desc": bo["winner_desc"],
           "featureset": featureset, "n_feat": N, "members": members,
           "protocol": "nested LOCO5 outer, inner 3-fold Optuna30 on CatBoost, "
                       "5-seed bagging, Platt sigmoid; seed 42",
           "weeks": {}, "leak_flags": []}
    for wk in WEEKS:
        res = run_week(wk, featureset, N, members, has_cat)
        # leak guard vs bake-off seed-mean AUC for the winner at this week
        bo_seed_aucs = [bo["cells"][wk][str(s)][winner]["roc_auc"] for s in SEEDS]
        bo_mean = float(np.mean(bo_seed_aucs))
        exceed = res["roc_auc_raw_bagged"] - bo_mean
        res["bakeoff_seed_mean_auc"] = round(bo_mean, 4)
        res["nested_minus_bakeoff"] = round(exceed, 4)
        res["leak_flag"] = bool(exceed > 0.02)
        if res["leak_flag"]:
            out["leak_flags"].append(wk)
        out["weeks"][wk] = res
        OUT.write_text(json.dumps(out, indent=2))
        print(f"[P3] wk{wk}: AUC_raw={res['roc_auc_raw_bagged']} CI{res['roc_auc_raw_ci95']} "
              f"AUC_cal={res['roc_auc_calibrated']} PR={res['pr_auc_raw_bagged']} "
              f"Brier={res['brier_calibrated']} ECE={res['ece_calibrated']} "
              f"rec20={res['capacity_curve']['0.2']} | bo_mean={bo_mean:.4f} "
              f"leak={res['leak_flag']} [{time.time()-t0:.0f}s]", flush=True)
        if res["leak_flag"]:
            print(f"[P3] LEAK FLAG at wk{wk} (nested exceeds bake-off mean by "
                  f"{exceed:+.4f}) — STOP recorded.", flush=True)
            break
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[P3] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
