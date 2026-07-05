#!/usr/bin/env python3
"""T5 — Nested CV on clean PUC (honest headline numbers).

Outer = LOCO StratifiedGroupKFold(5). Per outer-train: top-40 leak-free feature
selection, inner 3-fold StratifiedGroupKFold Optuna (30 trials, F2 objective,
XGBoost space) -> fit tuned + Platt-calibrated on outer-train -> predict
outer-test. Nested OOF ROC-AUC/PR-AUC (+ bootstrap CI) vs non-nested tuned
(params chosen once on all data => mildly optimistic) on the SAME clean folds,
plus the register's reference non-nested numbers.

Reads cached clean matrices from tier1_clean/features/week_{w}_clean.parquet.
Output: tier1_clean/nested_cv_results.json
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from xgboost import XGBClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "data/puc/sota_results/tier1_clean/features"
OUT = REPO / "data/puc/sota_results/tier1_clean/nested_cv_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
REFERENCE_NONNESTED = {"2": 0.831, "4": 0.872, "6": 0.863, "8": 0.863, "full": 0.854}
TOPK, N_SPLITS, N_TRIALS, N_BOOT = 40, 5, 30, 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)


def load_week(wk):
    df = pd.read_parquet(FEAT_DIR / f"week_{wk}_clean.parquet")
    y = df["_y"].to_numpy().astype(int)
    groups = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"])
    return X.reset_index(drop=True), y, groups


def topk_feats(Xtr, ytr):
    ranked = B.sota_feature_selection(Xtr, pd.Series(ytr), return_ranked=True)
    return ranked[:TOPK] if len(ranked) >= TOPK else ranked


def tune(Xtr, ytr, gtr, feats):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)
    Xf = Xtr[feats].values
    splits = list(inner.split(Xf, ytr, gtr))

    def objective(trial):
        params = B.get_optuna_search_space(trial, "XGBoost")
        scores = []
        for tr, va in splits:
            if ytr[tr].sum() < 2:
                continue
            m = XGBClassifier(**params)
            m.fit(Xf[tr], ytr[tr])
            p = m.predict_proba(Xf[va])[:, 1]
            scores.append(fbeta_score(ytr[va], (p >= 0.5).astype(int), beta=2, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def fit_predict(Xtr, ytr, Xte, feats, params):
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    p = dict(params); p.setdefault("scale_pos_weight", spw)
    p.update(eval_metric="logloss", verbosity=0, random_state=RS)
    clf = CalibratedClassifierCV(XGBClassifier(**p), method="sigmoid", cv=3)
    clf.fit(Xtr[feats].values, ytr)
    return clf.predict_proba(Xte[feats].values)[:, 1]


def boot_ci(y, p, fn, n=N_BOOT):
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    v = np.array(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def run_week(wk):
    X, y, groups = load_week(wk)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    folds = list(cv.split(X, y, groups))

    # ---- non-nested: tune ONCE on all data (optimistic), fixed params per fold
    feats_global = topk_feats(X, y)
    params_global = tune(X, y, groups, feats_global)
    oof_nn = np.full(len(y), np.nan)
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = topk_feats(X.iloc[tr], y[tr])
        oof_nn[te] = fit_predict(X.iloc[tr], y[tr], X.iloc[te], feats, params_global)

    # ---- nested: re-tune within each outer-train
    oof_nest = np.full(len(y), np.nan)
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = topk_feats(X.iloc[tr], y[tr])
        params = tune(X.iloc[tr], y[tr], groups[tr], feats)
        oof_nest[te] = fit_predict(X.iloc[tr], y[tr], X.iloc[te], feats, params)

    def metrics(oof):
        m = ~np.isnan(oof)
        yv, pv = y[m], oof[m]
        return {
            "roc_auc": round(float(roc_auc_score(yv, pv)), 4),
            "roc_auc_ci95": boot_ci(yv, pv, roc_auc_score),
            "pr_auc": round(float(average_precision_score(yv, pv)), 4),
            "pr_auc_ci95": boot_ci(yv, pv, average_precision_score),
            "n_eval": int(m.sum()),
        }

    nn, nest = metrics(oof_nn), metrics(oof_nest)
    gap = round(nn["roc_auc"] - nest["roc_auc"], 4)
    res = {
        "n": int(len(y)), "prevalence": round(float(y.mean()), 4),
        "nested": nest, "non_nested_clean": nn,
        "reference_nonnested_auc": REFERENCE_NONNESTED[wk],
        "auc_gap_nonnested_minus_nested": gap,
        "leakage_flag": bool(nest["roc_auc"] - nn["roc_auc"] > 0.02),
    }
    print(f"[T5] wk{wk}: nested AUC={nest['roc_auc']} PR={nest['pr_auc']} | "
          f"non-nested(clean)={nn['roc_auc']} | ref={REFERENCE_NONNESTED[wk]} | gap={gap} | "
          f"leak_flag={res['leakage_flag']}", flush=True)
    return res


def main():
    out = {"config": {"topk": TOPK, "outer_splits": N_SPLITS, "inner_splits": 3,
                      "n_trials": N_TRIALS, "objective": "F2", "seed": RS,
                      "data": "clean (T1/T2)"},
           "weeks": {}}
    for wk in WEEKS:
        out["weeks"][wk] = run_week(wk)
        OUT.write_text(json.dumps(out, indent=2))  # incremental save
    print(f"[T5] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
