#!/usr/bin/env python3
"""H2 (Tier-2B) — PUC stratified nested run ("alumnos nuevos en cursos conocidos").

Same winner config as P3 (CatBoost Balanced, top-40 per-fold leak-free FS, 5-seed
bagging, Platt sigmoid calibration) but under **StratifiedKFold(5, shuffle, seed 42)**
OUTER folds (NOT grouped by course). This measures generalization to new students
within already-seen courses — the same (easier, honest) question the retired UA 0.90
answered, now measured under a clean nested protocol.

Because this is a NEW protocol run (not the P3 LOCO folds), inner tuning is re-run:
inner StratifiedKFold(3, shuffle, seed 42) Optuna 30-trial F2 per outer-train fold.

Persists, exactly like H1:
  - OOF vectors -> tier2_push/oof_stratified_week_{w}.parquet (student_id,course_id,y,p,p_raw)
  - roc_auc_calibrated / raw_bagged (+ CI95, B=2000)
  - pr_auc_calibrated (+ CI95), Brier, ECE
  - calibrated capacity curve recall@{5,10,15,20,25}%
  - threshold sweep 0.05..0.95 (recall/precision/FPR on calibrated p)
  - fpr/tpr ROC arrays from calibrated OOF

Run: .venv-tier1/bin/python scripts/puc_stratified_nested.py
Output: tier2_push/stratified_nested_results.json + oof_stratified_week_*.parquet
"""
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, fbeta_score, roc_curve)
from catboost import CatBoostClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
REPO = Path(__file__).resolve().parents[1]
CLEAN_FEAT = REPO / "data/puc/sota_results/tier1_clean/features"
LOCO_CI = REPO / "data/puc/sota_results/tier2_push/confirmatory_calibrated_ci.json"
OUTDIR = REPO / "data/puc/sota_results/tier2_push"
OUT = OUTDIR / "stratified_nested_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
SEEDS = [42, 43, 44, 45, 46]
FLAG_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90
THRESHOLDS.append(0.95)
N_TRIALS, N_BOOT = 30, 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)


def load_matrix(wk):
    df = pd.read_parquet(CLEAN_FEAT / f"week_{wk}_clean.parquet")
    y = df["_y"].to_numpy().astype(int)
    sid = df["student_id"].to_numpy()
    cid = df["course_id"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, sid, cid


def topn(Xtr, ytr, N):
    r = B.sota_feature_selection(Xtr, pd.Series(ytr), return_ranked=True)
    return r[:N] if len(r) >= N else r


def make_cat(seed, cat_params):
    p = dict(cat_params or {})
    p.update(auto_class_weights="Balanced", random_seed=seed,
             verbose=False, allow_writing_files=False)
    return CatBoostClassifier(**p)


def tune_catboost(Xf, y):
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
    splits = list(inner.split(Xf, y))

    def objective(trial):
        params = {"depth": trial.suggest_int("depth", 4, 8),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                  "iterations": trial.suggest_int("iterations", 100, 500)}
        scores = []
        for tr, va in splits:
            if y[tr].sum() < 2:
                continue
            m = make_cat(RS, params)
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


def threshold_sweep(y, p):
    P = float(y.sum()); Nn = float((1 - y).sum())
    rows = []
    for t in THRESHOLDS:
        pred = (p >= t).astype(int)
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
        recall = tp / P if P else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / Nn if Nn else 0.0
        rows.append({"threshold": t, "recall": round(recall, 4),
                     "precision": round(precision, 4), "fpr": round(fpr, 4),
                     "flagged": int(pred.sum())})
    return rows


def run_week(wk):
    X, y, sid, cid = load_matrix(wk)
    outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    folds = list(outer.split(X, y))

    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    params_log = []
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = topn(X.iloc[tr], y[tr], 40)
        Xtr_f = X.iloc[tr][feats].values
        Xte_f = X.iloc[te][feats].values
        cp = tune_catboost(Xtr_f, y[tr])
        params_log.append(cp)

        preds = []
        for s in SEEDS:
            m = make_cat(s, cp)
            m.fit(Xtr_f, y[tr])
            preds.append(m.predict_proba(Xte_f)[:, 1])
        oof_raw[te] = np.mean(preds, axis=0)

        base = make_cat(RS, cp)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal.fit(Xtr_f, y[tr])
        oof_cal[te] = cal.predict_proba(Xte_f)[:, 1]

    m = ~np.isnan(oof_cal)
    yv, praw, pcal = y[m], oof_raw[m], oof_cal[m]
    sidv, cidv = sid[m], cid[m]

    oof_df = pd.DataFrame({"student_id": sidv, "course_id": cidv,
                           "y": yv.astype(int), "p": pcal, "p_raw": praw})
    oof_path = OUTDIR / f"oof_stratified_week_{wk}.parquet"
    oof_df.to_parquet(oof_path, index=False)

    fpr, tpr, _ = roc_curve(yv, pcal)
    res = {
        "n_eval": int(m.sum()), "prevalence": round(float(yv.mean()), 4),
        "roc_auc_calibrated": round(float(roc_auc_score(yv, pcal)), 4),
        "roc_auc_calibrated_ci95": boot_ci(yv, pcal, roc_auc_score),
        "pr_auc_calibrated": round(float(average_precision_score(yv, pcal)), 4),
        "pr_auc_calibrated_ci95": boot_ci(yv, pcal, average_precision_score),
        "roc_auc_raw_bagged": round(float(roc_auc_score(yv, praw)), 4),
        "roc_auc_raw_ci95": boot_ci(yv, praw, roc_auc_score),
        "brier_calibrated": round(float(brier_score_loss(yv, pcal)), 4),
        "ece_calibrated": round(ece(yv, pcal), 4),
        "capacity_curve_calibrated": {str(r): round(recall_at_flag(yv, pcal, r), 4)
                                      for r in FLAG_RATES},
        "threshold_sweep_calibrated": threshold_sweep(yv, pcal),
        "roc_fpr": [round(float(x), 5) for x in fpr],
        "roc_tpr": [round(float(x), 5) for x in tpr],
        "catboost_params_per_fold": params_log,
        "oof_parquet": str(oof_path.relative_to(REPO)),
        "oof_rows": int(len(oof_df)),
    }
    return res


def main():
    t0 = time.time()
    loco = json.loads(LOCO_CI.read_text()) if LOCO_CI.exists() else None
    out = {
        "source": "H2 Tier-2B — stratified nested confirmatory with persisted OOF",
        "winner": "C2", "winner_desc": "CatBoost Balanced 40 clean",
        "featureset": "clean", "n_feat": 40,
        "cv_scheme": "stratified (StratifiedKFold, not grouped)",
        "cv_label": "cursos conocidos, alumnos nuevos",
        "protocol": "nested StratifiedKFold(5, shuffle, seed 42) outer, per-fold top-40 "
                    "leak-free FS, inner StratifiedKFold(3) Optuna30 F2, 5-seed bagging, "
                    "Platt sigmoid(cv=3) calibrated production probs; bootstrap CI B=2000",
        "seeds": SEEDS, "n_boot": N_BOOT, "random_state": RS,
        "weeks": {},
    }
    for wk in WEEKS:
        res = run_week(wk)
        if loco is not None:
            loco_cal = loco["weeks"][wk]["roc_auc_calibrated"]
            res["loco_roc_auc_calibrated"] = loco_cal
            res["strat_minus_loco"] = round(res["roc_auc_calibrated"] - loco_cal, 4)
            res["implausible_flag"] = bool(res["strat_minus_loco"] > 0.10)
        out["weeks"][wk] = res
        OUT.write_text(json.dumps(out, indent=2))
        flag = ""
        if loco is not None and res.get("implausible_flag"):
            flag = "  <<< IMPLAUSIBLE strat-loco>0.10 — investigate"
        print(f"[H2] wk{wk}: AUC_cal={res['roc_auc_calibrated']} "
              f"CI{res['roc_auc_calibrated_ci95']} AUC_raw={res['roc_auc_raw_bagged']} "
              f"PR_cal={res['pr_auc_calibrated']} rec20={res['capacity_curve_calibrated']['0.2']} "
              f"rows={res['oof_rows']}"
              + (f" | LOCO_cal={res.get('loco_roc_auc_calibrated')} "
                 f"Δ={res.get('strat_minus_loco'):+.4f}" if loco is not None else "")
              + f" [{time.time()-t0:.0f}s]{flag}", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[H2] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
