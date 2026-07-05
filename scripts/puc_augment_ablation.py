#!/usr/bin/env python3
"""P4 — Train-only augmentation ablation (winner config, weeks {2,4,8}).

Rebuilds TRAIN folds with the 3 zero-fail courses' students appended as extra
negatives (features from the same clean pipeline, aligned to the 560-matrix
schema). TEST = the identical 560-pair LOCO folds as P3 (seed 42) — augmentation
students NEVER enter test. Paired comparison (augmented − non-augmented) on the
same test folds.

To isolate the effect of the added negatives (not confound with re-tuning/FS):
per outer fold the top-40 feature ranking and the F2-Optuna CatBoost params are
computed ONCE on the NON-augmented outer-train, then reused for BOTH arms; only
the training rows differ. 5-seed bagging both arms.

Run: .venv-tier1/bin/python scripts/puc_augment_ablation.py
Output: tier2_push/augment_ablation.json
"""
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import puc_features_clean as FC
import puc_confirmatory_v2 as P3  # make_model, tune_catboost, topn, recall_at_flag
import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
CLEAN = REPO / "data/puc/puc_clean_data.parquet"
GRADES = REPO / "data/puc/puc_grades_clean.parquet"
CLEAN_FEAT = REPO / "data/puc/sota_results/tier1_clean/features"
BAKEOFF = REPO / "data/puc/sota_results/tier2_push/bakeoff_results.json"
OUT = REPO / "data/puc/sota_results/tier2_push/augment_ablation.json"
AUG_COURSES = [53493, 54947, 56867]
WEEKS = ["2", "4", "8"]
SEEDS = [42, 43, 44, 45, 46]
FLAG_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
N, N_BOOT = 40, 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)


def load_base(wk):
    df = pd.read_parquet(CLEAN_FEAT / f"week_{wk}_clean.parquet")
    y = df["_y"].to_numpy().astype(int)
    g = df["_group"].to_numpy()
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, g


def build_aug(wk, base_cols):
    clean = pd.read_parquet(CLEAN)
    aug = clean[clean["course_id"].isin(AUG_COURSES)].copy()
    aug["hour"] = aug["hour_local"].astype("int32")
    aug["day_of_week"] = aug["dow_local"].astype("int32")
    grades = pd.read_parquet(GRADES)
    grades_aug = grades[grades["course_id"].isin(AUG_COURSES)]
    cutoff = "full" if wk == "full" else int(wk)
    Xa, ya, ga, ida, meta = FC.build_week_matrix(aug, grades_aug, cutoff)
    assert int(ya.sum()) == 0, f"aug arm has {int(ya.sum())} positives (expected 0)"
    Xa = Xa.reindex(columns=base_cols, fill_value=0.0)
    return Xa.reset_index(drop=True), ya


def bagged_predict(Xtr, ytr, Xte, params):
    preds = []
    for s in SEEDS:
        m = P3.make_model("cat", ytr, s, params)
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xte)[:, 1])
    return np.mean(preds, axis=0)


def paired_ci(y, p_a, p_b, fn):
    idx = np.arange(len(y)); vals = []
    for _ in range(N_BOOT):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p_a[b]) - fn(y[b], p_b[b]))
    v = np.array(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def run_week(wk):
    X, y, g = load_base(wk)
    Xa, ya = build_aug(wk, list(X.columns))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    folds = list(outer.split(X, y, g))
    # assert test folds cover exactly the 560 base rows, disjoint, in [0,560)
    all_test = np.concatenate([te for _, te in folds])
    assert len(all_test) == len(y) == 560 and set(all_test) == set(range(560)), "test folds != P3 560"

    oof_base = np.full(len(y), np.nan)
    oof_aug = np.full(len(y), np.nan)
    aug_train_sizes = []
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = P3.topn(X.iloc[tr], y[tr], N)
        params = P3.tune_catboost(X.iloc[tr][feats].values, y[tr], g[tr])
        Xtr_b = X.iloc[tr][feats].values
        Xte = X.iloc[te][feats].values
        # non-augmented arm
        oof_base[te] = bagged_predict(Xtr_b, y[tr], Xte, params)
        # augmented arm: append the 167 aug negatives to train
        Xtr_a = np.vstack([Xtr_b, Xa[feats].values])
        ytr_a = np.concatenate([y[tr], ya])
        aug_train_sizes.append(int(len(ytr_a)))
        oof_aug[te] = bagged_predict(Xtr_a, ytr_a, Xte, params)

    m = ~np.isnan(oof_base)
    yv, pb, pa = y[m], oof_base[m], oof_aug[m]

    def met(p):
        return {"roc_auc": round(float(roc_auc_score(yv, p)), 4),
                "pr_auc": round(float(average_precision_score(yv, p)), 4),
                "capacity_curve": {str(r): round(P3.recall_at_flag(yv, p, r), 4) for r in FLAG_RATES}}

    res = {
        "n_test": int(m.sum()), "aug_train_sizes": aug_train_sizes,
        "n_aug_negatives": int(len(ya)),
        "non_augmented": met(pb), "augmented": met(pa),
        "delta_auc_aug_minus_base": round(float(roc_auc_score(yv, pa) - roc_auc_score(yv, pb)), 4),
        "delta_auc_ci95": paired_ci(yv, pa, pb, roc_auc_score),
        "delta_pr_aug_minus_base": round(float(average_precision_score(yv, pa) - average_precision_score(yv, pb)), 4),
        "delta_pr_ci95": paired_ci(yv, pa, pb, average_precision_score),
    }
    return res


def main():
    t0 = time.time()
    out = {"winner": "C2 CatBoost Balanced 40 clean", "aug_courses": AUG_COURSES,
           "note": ("per-fold FS + F2-Optuna params computed on NON-aug train, reused "
                    "for both arms; only training rows differ; test=identical P3 560 folds; "
                    "5-seed bag both arms"),
           "weeks": {}}
    for wk in WEEKS:
        res = run_week(wk)
        out["weeks"][wk] = res
        OUT.write_text(json.dumps(out, indent=2))
        print(f"[P4] wk{wk}: base AUC={res['non_augmented']['roc_auc']} "
              f"aug AUC={res['augmented']['roc_auc']} "
              f"ΔAUC={res['delta_auc_aug_minus_base']:+.4f} CI{res['delta_auc_ci95']} "
              f"(aug_train~{res['aug_train_sizes'][0]}) [{time.time()-t0:.0f}s]", flush=True)
    # verdict
    deltas = [out["weeks"][wk]["delta_auc_aug_minus_base"] for wk in WEEKS]
    md = float(np.mean(deltas))
    sig = any(out["weeks"][wk]["delta_auc_ci95"][0] > 0 for wk in WEEKS)
    out["verdict"] = (f"augmentation {'HELPS' if md>0.003 else ('HURTS' if md<-0.003 else 'NEUTRAL')} "
                      f"(mean ΔAUC={md:+.4f}, any-week CI>0: {sig})")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[P4] {out['verdict']}")
    print(f"[P4] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
