#!/usr/bin/env python3
"""H1 (Tier-2B) — Calibrated CIs + persisted OOF for the PUC production artifact.

Re-runs the P3 confirmatory CALIBRATED arm per week {2,4,6,8,full} under the
IDENTICAL protocol, but:
  - REUSES the stored per-fold tuned CatBoost params from confirmatory_results.json
    (`catboost_params_per_fold`) instead of re-running Optuna (no new tuning), and
  - PERSISTS the OOF probability vectors to
    tier2_push/oof_calibrated_week_{w}.parquet (student_id, course_id, y, p, p_raw).

Same LOCO folds (StratifiedGroupKFold(5, shuffle, seed 42)), same per-outer-train
top-40 leak-free selection, 5-seed raw bagging {42..46}, Platt sigmoid(cv=3) around
the seed-42 per-fold model for the calibrated (production) probabilities.

Computes for each week:
  - roc_auc_calibrated (+ CI95, bootstrap B=2000)
  - pr_auc_calibrated  (+ CI95)
  - roc_auc_raw_bagged (+ CI95)  [carried for cross-check vs P3]
  - calibrated capacity curve recall@{5,10,15,20,25}%
  - threshold sweep 0.05..0.95 step 0.05: recall, precision, FPR (on calibrated p)
  - Brier + ECE (calibrated)
  - fpr/tpr ROC arrays (from calibrated OOF) for the page's operating curves

Run: .venv-tier1/bin/python scripts/puc_confirmatory_calibrated_ci.py
Output: tier2_push/confirmatory_calibrated_ci.json
        tier2_push/oof_calibrated_week_{2,4,6,8,full}.parquet
"""
import json, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, roc_curve)
from catboost import CatBoostClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
CLEAN_FEAT = REPO / "data/puc/sota_results/tier1_clean/features"
CONF = REPO / "data/puc/sota_results/tier2_push/confirmatory_results.json"
OUTDIR = REPO / "data/puc/sota_results/tier2_push"
OUT = OUTDIR / "confirmatory_calibrated_ci.json"
WEEKS = ["2", "4", "6", "8", "full"]
SEEDS = [42, 43, 44, 45, 46]
FLAG_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 19)]  # 0.05 .. 0.90
THRESHOLDS.append(0.95)
N_BOOT = 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)


def load_matrix(wk):
    df = pd.read_parquet(CLEAN_FEAT / f"week_{wk}_clean.parquet")
    y = df["_y"].to_numpy().astype(int)
    g = df["_group"].to_numpy()
    sid = df["student_id"].to_numpy()
    cid = df["course_id"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, g, sid, cid


def topn(Xtr, ytr, N):
    r = B.sota_feature_selection(Xtr, pd.Series(ytr), return_ranked=True)
    return r[:N] if len(r) >= N else r


def make_cat(seed, cat_params):
    p = dict(cat_params or {})
    p.update(auto_class_weights="Balanced", random_seed=seed,
             verbose=False, allow_writing_files=False)
    return CatBoostClassifier(**p)


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


def run_week(wk, params_per_fold):
    X, y, g, sid, cid = load_matrix(wk)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    folds = list(outer.split(X, y, g))

    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    fi = 0
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        cp = params_per_fold[fi]
        fi += 1
        feats = topn(X.iloc[tr], y[tr], 40)
        Xtr_f = X.iloc[tr][feats].values
        Xte_f = X.iloc[te][feats].values

        # raw-bagged over model seeds {42..46}
        preds = []
        for s in SEEDS:
            m = make_cat(s, cp)
            m.fit(Xtr_f, y[tr])
            preds.append(m.predict_proba(Xte_f)[:, 1])
        oof_raw[te] = np.mean(preds, axis=0)

        # calibrated (seed 42) — the production probabilities
        base = make_cat(RS, cp)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal.fit(Xtr_f, y[tr])
        oof_cal[te] = cal.predict_proba(Xte_f)[:, 1]

    m = ~np.isnan(oof_cal)
    yv, praw, pcal = y[m], oof_raw[m], oof_cal[m]
    sidv, cidv = sid[m], cid[m]

    # persist OOF parquet
    oof_df = pd.DataFrame({"student_id": sidv, "course_id": cidv,
                           "y": yv.astype(int), "p": pcal, "p_raw": praw})
    oof_path = OUTDIR / f"oof_calibrated_week_{wk}.parquet"
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
        "oof_parquet": str(oof_path.relative_to(REPO)),
        "oof_rows": int(len(oof_df)),
    }
    return res


def main():
    t0 = time.time()
    conf = json.loads(CONF.read_text())
    out = {
        "source": "H1 Tier-2B — calibrated confirmatory with persisted OOF",
        "winner": conf["winner"], "winner_desc": conf["winner_desc"],
        "featureset": conf["featureset"], "n_feat": conf["n_feat"],
        "cv_scheme": "LOCO (StratifiedGroupKFold by course)",
        "cv_label": "cursos nunca vistos",
        "protocol": "nested LOCO5 outer (seed 42), per-fold top-40 leak-free FS, "
                    "REUSED stored per-fold CatBoost params (no Optuna), 5-seed bagging, "
                    "Platt sigmoid(cv=3) for calibrated production probs; bootstrap CI B=2000",
        "seeds": SEEDS, "n_boot": N_BOOT, "random_state": RS,
        "weeks": {},
    }
    for wk in WEEKS:
        ppf = conf["weeks"][wk]["catboost_params_per_fold"]
        res = run_week(wk, ppf)
        stored_cal = conf["weeks"][wk]["roc_auc_calibrated"]
        drift = round(res["roc_auc_calibrated"] - stored_cal, 4)
        res["stored_roc_auc_calibrated"] = stored_cal
        res["cal_drift_vs_stored"] = drift
        out["weeks"][wk] = res
        OUT.write_text(json.dumps(out, indent=2))
        flag = "OK" if abs(drift) <= 0.005 else "DRIFT>0.005"
        print(f"[H1] wk{wk}: AUC_cal={res['roc_auc_calibrated']} "
              f"CI{res['roc_auc_calibrated_ci95']} (stored {stored_cal}, drift {drift:+.4f} {flag}) "
              f"AUC_raw={res['roc_auc_raw_bagged']} PR_cal={res['pr_auc_calibrated']} "
              f"rec20={res['capacity_curve_calibrated']['0.2']} rows={res['oof_rows']} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[H1] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
