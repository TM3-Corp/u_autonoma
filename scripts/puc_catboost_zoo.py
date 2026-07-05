#!/usr/bin/env python3
"""T6 — CatBoost + HistGradientBoosting into the zoo (clean data).

Extends a LOCAL model set (no edit to puc_benchmark_sota.py) with CatBoost
(auto_class_weights='Balanced') and HistGradientBoostingClassifier
(class_weight='balanced'), evaluated under the production protocol on clean
data: LOCO StratifiedGroupKFold(5), top-40 leak-free per-fold selection, Platt
calibration, seed 42. XGBoost (same config) is the same-fold baseline. CatBoost
gets a 30-trial Optuna pass at weeks 4 and 8; other weeks use sensible defaults.
Paired bootstrap CIs on ΔAUC vs XGBoost (shared resample indices).

Run with the catboost venv:  .venv-tier1/bin/python scripts/puc_catboost_zoo.py
Reads cached clean matrices from tier1_clean/features/week_{w}_clean.parquet.
Output: tier1_clean/catboost_results.json
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, fbeta_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "data/puc/sota_results/tier1_clean/features"
OUT = REPO / "data/puc/sota_results/tier1_clean/catboost_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
OPTUNA_WEEKS = {"4", "8"}
TOPK, N_SPLITS, N_TRIALS, N_BOOT = 40, 5, 30, 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)


def load_week(wk):
    df = pd.read_parquet(FEAT_DIR / f"week_{wk}_clean.parquet")
    y = df["_y"].to_numpy().astype(int)
    groups = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"])
    return X.reset_index(drop=True), y, groups


def topk(Xtr, ytr):
    ranked = B.sota_feature_selection(Xtr, pd.Series(ytr), return_ranked=True)
    return ranked[:TOPK] if len(ranked) >= TOPK else ranked


def make_model(kind, ytr, params=None):
    spw = float((len(ytr) - ytr.sum()) / max(ytr.sum(), 1))
    if kind == "xgb":
        return XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                             scale_pos_weight=spw, eval_metric="logloss", verbosity=0,
                             random_state=RS)
    if kind == "hist":
        return HistGradientBoostingClassifier(class_weight="balanced", random_state=RS)
    if kind == "catboost":
        p = dict(params or {})
        p.update(auto_class_weights="Balanced", random_seed=RS, verbose=False,
                 allow_writing_files=False)
        return CatBoostClassifier(**p)
    raise ValueError(kind)


def oof_predict(X, y, groups, folds, kind, params=None, calibrate=True):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = topk(X.iloc[tr], y[tr])
        base = make_model(kind, y[tr], params)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3) if calibrate else base
        clf.fit(X.iloc[tr][feats].values, y[tr])
        oof[te] = clf.predict_proba(X.iloc[te][feats].values)[:, 1]
    return oof


def tune_catboost(X, y, groups):
    inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)
    feats = topk(X, y)
    Xf = X[feats].values
    splits = list(inner.split(Xf, y, groups))

    def objective(trial):
        params = {
            "depth": trial.suggest_int("depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "iterations": trial.suggest_int("iterations", 100, 500),
        }
        scores = []
        for tr, va in splits:
            if y[tr].sum() < 2:
                continue
            m = make_model("catboost", y[tr], params)
            m.fit(Xf[tr], y[tr])
            p = m.predict_proba(Xf[va])[:, 1]
            scores.append(fbeta_score(y[va], (p >= 0.5).astype(int), beta=2, zero_division=0))
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RS),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=10))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def paired_delta_ci(y, p_a, p_b, fn, n=N_BOOT):
    """CI on fn(a)-fn(b) with shared resample indices."""
    idx = np.arange(len(y)); vals = []
    for _ in range(n):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p_a[b]) - fn(y[b], p_b[b]))
    v = np.array(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def run_week(wk):
    X, y, groups = load_week(wk)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RS)
    folds = list(cv.split(X, y, groups))

    cb_params = tune_catboost(X, y, groups) if wk in OPTUNA_WEEKS else None
    oof = {
        "xgb": oof_predict(X, y, groups, folds, "xgb"),
        "hist": oof_predict(X, y, groups, folds, "hist"),
        "catboost": oof_predict(X, y, groups, folds, "catboost", cb_params),
    }
    m = ~np.isnan(oof["xgb"])
    yv = y[m]

    def met(p):
        return {"roc_auc": round(float(roc_auc_score(yv, p[m])), 4),
                "pr_auc": round(float(average_precision_score(yv, p[m])), 4)}

    res = {"n": int(len(y)), "prevalence": round(float(y.mean()), 4),
           "catboost_tuned": wk in OPTUNA_WEEKS, "catboost_params": cb_params,
           "xgb": met(oof["xgb"]), "hist": met(oof["hist"]), "catboost": met(oof["catboost"])}
    for kind in ("catboost", "hist"):
        d = round(res[kind]["roc_auc"] - res["xgb"]["roc_auc"], 4)
        res[f"delta_auc_{kind}_vs_xgb"] = d
        res[f"delta_auc_{kind}_vs_xgb_ci95"] = paired_delta_ci(yv, oof[kind][m], oof["xgb"][m], roc_auc_score)
    print(f"[T6] wk{wk}: xgb={res['xgb']['roc_auc']} cat={res['catboost']['roc_auc']} "
          f"hist={res['hist']['roc_auc']} dCat={res['delta_auc_catboost_vs_xgb']} "
          f"dHist={res['delta_auc_hist_vs_xgb']}", flush=True)
    return res


def verdict(weeks):
    dcat = [w["delta_auc_catboost_vs_xgb"] for w in weeks.values()]
    mean_d = float(np.mean(dcat))
    sig = any((lo > 0) for lo in [w["delta_auc_catboost_vs_xgb_ci95"][0] for w in weeks.values()])
    if mean_d > 0.005 and sig:
        return f"CatBoost BEATS XGBoost (mean ΔAUC={mean_d:+.4f}, ≥1 week CI>0)"
    if mean_d < -0.005:
        return f"CatBoost LOSES to XGBoost (mean ΔAUC={mean_d:+.4f})"
    return f"CatBoost MATCHES XGBoost (mean ΔAUC={mean_d:+.4f}, no significant separation)"


def main():
    out = {"config": {"protocol": "LOCO5 + top40/fold + Platt + seed42, clean data",
                      "n_trials": N_TRIALS, "optuna_weeks": sorted(OPTUNA_WEEKS)}, "weeks": {}}
    for wk in WEEKS:
        out["weeks"][wk] = run_week(wk)
        OUT.write_text(json.dumps(out, indent=2))
    out["conclusion"] = verdict(out["weeks"])
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[T6] {out['conclusion']}")
    print(f"[T6] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
