#!/usr/bin/env python3
"""G6 — Confirmatory (the ONLY quotable Tier-3 numbers).

Winner config (from stageB_results.json) on R2-pooled, all 5 weeks:
 1. Nested LOCO: outer grouped 5-fold (seed 42); per outer-train top-N leak-free
    ranking + inner 3-fold Optuna 150-trial F2 tuning of the winning family →
    5-seed bagging → Platt sigmoid. Bootstrap CI (B=2000), capacity curve
    {5..25}%, per-course AUC, persisted OOF parquets.
 2. Leave-institution-out: train UA(R2)→test PUC(R2) and train PUC(R2)→test UA(R2).
    Tuned on the train side; AUC+CI each direction (expect asymmetry — UA test
    labels noisier). Pristine-label test = train-UA→test-PUC.
 3. R3 max-map (INTERNAL, quotable:false): greedy forward course-subset by AUC
    seeded from G3's per-course profiles; reference config, single seed 42.

Leak guard: nested ≤ Stage-B mean +0.02. Output: confirmatory_results.json,
oof_pooled_week_{w}.parquet.
"""
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, fbeta_score)
import tier3_common as T

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
POOL = Path(T.POOL)
STAGEB = POOL / "stageB_results.json"
PROFILES = POOL / "course_profiles.json"
OUT = POOL / "confirmatory_results.json"
WEEKS = ["2", "4", "6", "8", "full"]
SEEDS = [42, 43, 44, 45, 46]
FLAG_RATES = [0.05, 0.10, 0.15, 0.20, 0.25]
N_TRIALS, N_BOOT = 150, 2000
RS = T.RANDOM_STATE
RNG = np.random.RandomState(RS)


def parse_winner(w):
    kind = "cat" if w.startswith("cat") else "xgb"
    spec = w.split("_", 1)[1]
    if spec == "full":
        return kind, None, True
    return kind, int(spec.replace("N", "")), False


def select_feats(X, ytr, tr, N, is_full):
    if is_full:
        return T.corr_prefilter(X.iloc[tr])
    ranked = T.rank_features(X.iloc[tr], ytr, RS)
    return ranked[:N] if len(ranked) >= N else ranked


def tune(kind, Xf, y, g, use_group=True):
    if use_group and len(np.unique(g)) >= 3:
        inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RS)
        splits = list(inner.split(Xf, y, g))
    else:
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RS)
        splits = list(inner.split(Xf, y))

    def objective(trial):
        if kind == "cat":
            params = {"depth": trial.suggest_int("depth", 4, 8),
                      "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                      "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                      "iterations": trial.suggest_int("iterations", 100, 500)}
        else:
            params = {"max_depth": trial.suggest_int("max_depth", 3, 8),
                      "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                      "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                      "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                      "min_child_weight": trial.suggest_int("min_child_weight", 1, 8)}
        scores = []
        for tr, va in splits:
            if y[tr].sum() < 2:
                continue
            m = T.make_model(kind, y[tr], RS, params)
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


def recall_at(y, p, rate):
    k = max(1, int(np.ceil(rate * len(y))))
    idx = np.argsort(p)[::-1][:k]
    return round(float(y[idx].sum() / max(y.sum(), 1)), 4)


def boot_ci(y, p, fn):
    idx = np.arange(len(y)); vals = []
    for _ in range(N_BOOT):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    v = np.array(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def nested_week(d, kind, N, is_full):
    X = d[T.MODEL_FEATURES].reset_index(drop=True)
    y = d["y"].to_numpy().astype(int)
    g = d["course_id"].to_numpy()
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RS)
    folds = list(outer.split(X, y, g))
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    params_log = []
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        sel = select_feats(X, y[tr], tr, N, is_full)
        Xtr, Xte = X.iloc[tr][sel].values, X.iloc[te][sel].values
        params = tune(kind, Xtr, y[tr], g[tr], use_group=True)
        params_log.append(params)
        preds = []
        for s in SEEDS:
            m = T.make_model(kind, y[tr], s, params)
            m.fit(Xtr, y[tr])
            preds.append(m.predict_proba(Xte)[:, 1])
        oof_raw[te] = np.mean(preds, axis=0)
        base = T.make_model(kind, y[tr], RS, params)
        cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        cal.fit(Xtr, y[tr])
        oof_cal[te] = cal.predict_proba(Xte)[:, 1]
    m = ~np.isnan(oof_raw)
    yv, praw, pcal = y[m], oof_raw[m], oof_cal[m]
    res = {
        "n_eval": int(m.sum()), "prevalence": round(float(yv.mean()), 4),
        "roc_auc_raw_bagged": round(float(roc_auc_score(yv, praw)), 4),
        "roc_auc_raw_ci95": boot_ci(yv, praw, roc_auc_score),
        "pr_auc_raw": round(float(average_precision_score(yv, praw)), 4),
        "roc_auc_calibrated": round(float(roc_auc_score(yv, pcal)), 4),
        "brier_calibrated": round(float(brier_score_loss(yv, pcal)), 4),
        "ece_calibrated": round(ece(yv, pcal), 4),
        "capacity_curve": {str(r): recall_at(yv, praw, r) for r in FLAG_RATES},
        "per_course_auc": T.per_course_auc(d.iloc[m.nonzero()[0]], yv, praw),
        "params_per_fold": params_log,
    }
    oof_df = d[["inst", "sid", "course_id"]].copy()
    oof_df["y"] = y
    oof_df["p_raw"] = oof_raw
    oof_df["p_cal"] = oof_cal
    return res, oof_df


def transfer(train_d, test_d, kind, N, is_full, label):
    Xtr_all = train_d[T.MODEL_FEATURES].reset_index(drop=True)
    ytr = train_d["y"].to_numpy().astype(int)
    gtr = train_d["course_id"].to_numpy()
    Xte_all = test_d[T.MODEL_FEATURES].reset_index(drop=True)
    yte = test_d["y"].to_numpy().astype(int)
    sel = select_feats(Xtr_all, ytr, np.arange(len(ytr)), N, is_full)
    Xtr = Xtr_all[sel].values
    Xte = Xte_all[sel].values
    params = tune(kind, Xtr, ytr, gtr, use_group=(train_d.course_id.nunique() >= 3))
    preds = []
    for s in SEEDS:
        m = T.make_model(kind, ytr, s, params)
        m.fit(Xtr, ytr)
        preds.append(m.predict_proba(Xte)[:, 1])
    p = np.mean(preds, axis=0)
    return {"direction": label, "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "test_prevalence": round(float(yte.mean()), 4),
            "roc_auc": round(float(roc_auc_score(yte, p)), 4),
            "roc_auc_ci95": boot_ci(yte, p, roc_auc_score),
            "pr_auc": round(float(average_precision_score(yte, p)), 4),
            "recall20": recall_at(yte, p, 0.20)}


def r3_max_map(df8):
    """INTERNAL, quotable:false. Greedy forward course-subset by pooled LOCO AUC,
    seeded by the top-2 courses by G3 per-course AUC, reference config seed 42, wk8."""
    prof = json.loads(PROFILES.read_text())["profiles"]
    order = [p["course_id"] for p in sorted(prof, key=lambda x: -(x["loco_auc_wk8"] or 0))]

    def pooled_of(courses):
        d = df8[df8.course_id.isin(courses)]
        if d.course_id.nunique() < 2 or d.y.sum() < 2:
            return None
        oof, y, g, nsp = T.oof_predict(d, kind="cat", N=40, seed=RS)
        return T.pooled_auc(y, oof)

    selected = order[:2]
    best = pooled_of(selected)
    trace = [{"added": selected[:], "auc": best}]
    remaining = [c for c in order if c not in selected]
    no_improve = 0
    while remaining and no_improve < 3:
        scored = []
        for c in remaining:
            a = pooled_of(selected + [c])
            if a is not None:
                scored.append((c, a))
        if not scored:
            break
        scored.sort(key=lambda t: -t[1])
        c, a = scored[0]
        selected.append(c)
        remaining.remove(c)
        if best is None or a > best:
            best = a
            no_improve = 0
        else:
            no_improve += 1
        trace.append({"added": c, "auc": a, "subset_size": len(selected)})
    peak_idx = int(np.argmax([t.get("auc") or 0 for t in trace]))
    peak_courses = order[:2] + [t["added"] for t in trace[1:peak_idx + 1]]
    return {"quotable": False,
            "procedure": "greedy forward course-subset by pooled LOCO AUC, seeded by top-2 "
                         "G3 per-course AUC, CatBoost Balanced top-40, seed 42, week 8; "
                         "stop after 3 non-improving steps",
            "max_auc": round(float(best), 4) if best else None,
            "peak_subset": [int(c) for c in peak_courses],
            "peak_subset_size": len(peak_courses),
            "trace": trace}


def main():
    t0 = time.time()
    sb = json.loads(STAGEB.read_text())
    winner = sb["winner"]
    kind, N, is_full = parse_winner(winner)
    print(f"[G6] winner={winner} → kind={kind} N={N} full={is_full}", flush=True)

    dfw = {w: T.load_week(w) for w in WEEKS}
    T.assert_rules(dfw["8"])
    r2 = {w: T.subset(dfw[w], "R2", "pooled") for w in WEEKS}

    out = {"winner": winner, "kind": kind, "N": N, "is_full": is_full,
           "scope": "R2-pooled (10 courses, ~400 pairs, ~91 fails, ~23% prevalence)",
           "protocol": "nested LOCO5 outer seed42, inner 3-fold Optuna150 F2, 5-seed bag, Platt sigmoid",
           "weeks": {}, "leak_flags": [], "transfer": {}, "r3_max_map": None}

    # 1. nested LOCO all weeks
    for w in WEEKS:
        res, oof_df = nested_week(r2[w], kind, N, is_full)
        oof_df.to_parquet(POOL / f"oof_pooled_week_{w}.parquet", index=False)
        # leak guard vs Stage-B mean AUC for this config
        sb_mean = sb["aggregate"].get(winner, {}).get("mean_auc")
        res["stageB_mean_auc"] = sb_mean
        res["nested_minus_stageB"] = round(res["roc_auc_raw_bagged"] - sb_mean, 4) if sb_mean else None
        res["leak_flag"] = bool(sb_mean and (res["roc_auc_raw_bagged"] - sb_mean) > 0.02)
        if res["leak_flag"]:
            out["leak_flags"].append(w)
        out["weeks"][w] = res
        OUT.write_text(json.dumps(out, indent=2))
        print(f"[G6] nested wk{w}: AUC_raw={res['roc_auc_raw_bagged']} CI{res['roc_auc_raw_ci95']} "
              f"AUC_cal={res['roc_auc_calibrated']} PR={res['pr_auc_raw']} "
              f"rec20={res['capacity_curve']['0.2']} leak={res['leak_flag']} [{time.time()-t0:.0f}s]", flush=True)

    # 2. leave-institution-out
    puc_r2 = {w: T.subset(dfw[w], "R2", "PUC") for w in WEEKS}
    ua_r2 = {w: T.subset(dfw[w], "R2", "UA") for w in WEEKS}
    for w in WEEKS:
        out["transfer"][w] = {
            "train_UA_test_PUC": transfer(ua_r2[w], puc_r2[w], kind, N, is_full, "train_UA_test_PUC"),
            "train_PUC_test_UA": transfer(puc_r2[w], ua_r2[w], kind, N, is_full, "train_PUC_test_UA"),
        }
        tp = out["transfer"][w]
        print(f"[G6] transfer wk{w}: UA→PUC AUC={tp['train_UA_test_PUC']['roc_auc']} "
              f"{tp['train_UA_test_PUC']['roc_auc_ci95']} | "
              f"PUC→UA AUC={tp['train_PUC_test_UA']['roc_auc']} "
              f"{tp['train_PUC_test_UA']['roc_auc_ci95']} [{time.time()-t0:.0f}s]", flush=True)
        out["transfer_note"] = ("Asymmetry expected: PUC test labels are pristine official actas; "
                                "UA test labels are Canvas-recorded (DROP-A). train-UA→test-PUC is the "
                                "pristine-label transfer test.")
        OUT.write_text(json.dumps(out, indent=2))

    # 3. R3 max-map (internal)
    print("[G6] R3 max-map (greedy forward, internal/non-quotable)...", flush=True)
    out["r3_max_map"] = r3_max_map(dfw["8"])
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[G6] R3 max AUC={out['r3_max_map']['max_auc']} on "
          f"{out['r3_max_map']['peak_subset_size']} courses (quotable=False) [{time.time()-t0:.0f}s]", flush=True)

    print(f"[G6] DONE. leak_flags={out['leak_flags']}. wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
