#!/usr/bin/env python3
"""UA-3 — UA confirmatory + honest range.

Per arm (KEEP 373 / DROP-A 322) the UA-2 winner -> nested CV (inner 3-fold Optuna
30 trials if the winner has a CatBoost member), 5-seed bagging, Platt calibration,
bootstrap CIs, capacity curve {10,15,20,25}%. Reports BOTH StratifiedKFold (primary,
matches historical UA reporting) and StratifiedGroupKFold(LOCO). A+ (286) sensitivity
row: winner config, single seed-42, one line per week.

Honest range per week/cv: "AUC = [DROP-A] - [KEEP]" with the KEEP label caveat.
HARD guardrail: the KEEP number NEVER appears without its caveat sentence.
Leak guard: nested AUC > UA-2 bake-off strat seed-mean +0.02 -> flag.

Run: .venv-tier1/bin/python scripts/ua_confirmatory.py
Output: tier2_push/ua_confirmatory.json
"""
import json, sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ua_bakeoff as UB          # load_arm, make_model, sota_topk, PA_COLS, metrics helpers
import puc_confirmatory_v2 as P3  # tune_catboost, make_model(seed), ece, recall_at_flag
import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
UA_BAKEOFF = REPO / "data/puc/sota_results/tier2_push/ua_bakeoff_results.json"
OUT = REPO / "data/puc/sota_results/tier2_push/ua_confirmatory.json"
WEEKS = ["2", "4", "8", "full"]
SEEDS = [42, 43, 44, 45, 46]
FLAG_RATES = [0.10, 0.15, 0.20, 0.25]
N_BOOT = 2000
RS = B.RANDOM_STATE
RNG = np.random.RandomState(RS)
KEEP_CAVEAT = ("target = recorded Canvas outcome; includes 51 active-zero "
               "enrollments whose true grades are external")
# winner config -> member model kinds (UA-2 uses these ids)
WINNER_MEMBERS = {"U2": ["xgb"], "U3": ["cat"], "U4": ["xgb", "cat", "hist"]}


def boot_ci(y, p, fn):
    idx = np.arange(len(y)); vals = []
    for _ in range(N_BOOT):
        b = RNG.choice(idx, size=len(idx), replace=True)
        if y[b].min() == y[b].max():
            continue
        vals.append(fn(y[b], p[b]))
    v = np.array(vals)
    return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]


def nested_oof(X, y, g, members, cv_kind, tune):
    has_cat = "cat" in members
    if cv_kind == "strat":
        folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X, y))
    else:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=RS).split(X, y, g))
    oof_raw = np.full(len(y), np.nan)
    oof_cal = np.full(len(y), np.nan)
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        feats = UB.sota_topk(X.iloc[tr], y[tr])
        Xtr = X.iloc[tr][feats].values
        Xte = X.iloc[te][feats].values
        cat_params = P3.tune_catboost(Xtr, y[tr], g[tr]) if (has_cat and tune) else None
        from scipy.stats import rankdata
        raw_m, cal_m = [], []
        for kind in members:
            cp = cat_params if kind == "cat" else None
            preds = []
            for s in SEEDS:
                m = P3.make_model(kind, y[tr], s, cp)
                m.fit(Xtr, y[tr])
                preds.append(m.predict_proba(Xte)[:, 1])
            raw_m.append(np.mean(preds, axis=0))
            base = P3.make_model(kind, y[tr], RS, cp)
            cal = CalibratedClassifierCV(base, method="sigmoid", cv=3)
            cal.fit(Xtr, y[tr])
            cal_m.append(cal.predict_proba(Xte)[:, 1])
        if len(members) == 1:
            oof_raw[te], oof_cal[te] = raw_m[0], cal_m[0]
        else:
            oof_raw[te] = np.mean([rankdata(r) / len(r) for r in raw_m], axis=0)
            oof_cal[te] = np.mean(cal_m, axis=0)
    return oof_raw, oof_cal


def summarize(y, praw, pcal):
    m = ~np.isnan(praw)
    yv, pr, pc = y[m], praw[m], pcal[m]
    return {
        "n_eval": int(m.sum()), "prevalence": round(float(yv.mean()), 4),
        "roc_auc_raw_bagged": round(float(roc_auc_score(yv, pr)), 4),
        "roc_auc_raw_ci95": boot_ci(yv, pr, roc_auc_score),
        "pr_auc_raw": round(float(average_precision_score(yv, pr)), 4),
        "roc_auc_calibrated": round(float(roc_auc_score(yv, pc)), 4),
        "brier_calibrated": round(float(brier_score_loss(yv, pc)), 4),
        "ece_calibrated": round(P3.ece(yv, pc), 4),
        "capacity_curve": {str(r): round(P3.recall_at_flag(yv, pr, r), 4) for r in FLAG_RATES},
    }


def main():
    t0 = time.time()
    bo = json.loads(UA_BAKEOFF.read_text())
    winners = {arm: bo["selection"][arm]["winner"] for arm in ["KEEP", "DROP_A"]}
    print(f"[UA-3] winners: {winners}", flush=True)

    out = {"winners_per_arm": winners, "label_caveat_KEEP": KEEP_CAVEAT,
           "protocol": "nested (strat + loco) outer, inner Optuna30 on CatBoost, "
                       "5-seed bag, Platt sigmoid; seed 42",
           "arms": {}, "honest_range": {}, "leak_flags": [], "sensitivity_Aplus": {}}
    arm_cols = {"KEEP": "arm_keep", "DROP_A": "arm_dropA"}

    for arm in ["KEEP", "DROP_A"]:
        members = WINNER_MEMBERS[winners[arm]]
        out["arms"][arm] = {"winner": winners[arm], "members": members, "weeks": {}}
        for wk in WEEKS:
            X, y, g = UB.load_arm(wk, arm_cols[arm])
            entry = {}
            for cv in ["strat", "loco"]:
                praw, pcal = nested_oof(X, y, g, members, cv, tune=True)
                s = summarize(y, praw, pcal)
                # leak guard vs UA-2 strat seed-mean for this winner
                bo_aucs = [bo["cells"][arm][wk]["strat" if cv == "strat" else "loco"][str(sd)][winners[arm]]["roc_auc"]
                           for sd in SEEDS]
                bo_mean = float(np.mean(bo_aucs))
                s["bakeoff_seed_mean_auc"] = round(bo_mean, 4)
                s["leak_flag"] = bool(s["roc_auc_raw_bagged"] - bo_mean > 0.02)
                if s["leak_flag"]:
                    out["leak_flags"].append(f"{arm}/{wk}/{cv}")
                entry[cv] = s
            out["arms"][arm]["weeks"][wk] = entry
            OUT.write_text(json.dumps(out, indent=2))
            print(f"[UA-3] {arm} wk{wk}: strat AUC={entry['strat']['roc_auc_raw_bagged']} "
                  f"loco AUC={entry['loco']['roc_auc_raw_bagged']} "
                  f"[{time.time()-t0:.0f}s]", flush=True)

    # honest range per week/cv: DROP_A (quotable alone) -> KEEP (with caveat)
    for wk in WEEKS:
        out["honest_range"][wk] = {}
        for cv in ["strat", "loco"]:
            dropa = out["arms"]["DROP_A"]["weeks"][wk][cv]["roc_auc_raw_bagged"]
            keep = out["arms"]["KEEP"]["weeks"][wk][cv]["roc_auc_raw_bagged"]
            lo, hi = sorted([dropa, keep])
            out["honest_range"][wk][cv] = {
                "range_auc": [lo, hi],
                "DROP_A_quotable": dropa,
                "KEEP_with_caveat": {"auc": keep, "caveat": KEEP_CAVEAT},
                "statement": f"wk{wk} {cv} AUC = {dropa} (DROP-A, quotable) – {keep} "
                             f"(KEEP; {KEEP_CAVEAT})",
            }

    # A+ sensitivity row (286): winner config (DROP_A's winner), single seed-42, per week
    win_ap = winners["DROP_A"]
    members_ap = WINNER_MEMBERS[win_ap]
    for wk in WEEKS:
        X, y, g = UB.load_arm(wk, "arm_aplus")
        # single seed-42, per-fold sota + cat default (no bag), strat CV, quick
        folds = list(StratifiedKFold(5, shuffle=True, random_state=RS).split(X, y))
        oof = np.full(len(y), np.nan)
        for tr, te in folds:
            if y[tr].sum() < 2:
                continue
            feats = UB.sota_topk(X.iloc[tr], y[tr])
            probs = []
            for kind in members_ap:
                m = P3.make_model(kind, y[tr], RS, None)
                m.fit(X.iloc[tr][feats].values, y[tr])
                probs.append(m.predict_proba(X.iloc[te][feats].values)[:, 1])
            from scipy.stats import rankdata
            oof[te] = probs[0] if len(probs) == 1 else np.mean([rankdata(p)/len(te) for p in probs], axis=0)
        mask = ~np.isnan(oof)
        out["sensitivity_Aplus"][wk] = {
            "n": 286, "winner": win_ap,
            "roc_auc_strat_seed42": round(float(roc_auc_score(y[mask], oof[mask])), 4),
            "prevalence": round(float(y.mean()), 4)}
        print(f"[UA-3] A+ wk{wk}: AUC(strat,s42)={out['sensitivity_Aplus'][wk]['roc_auc_strat_seed42']}", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"[UA-3] leak_flags: {out['leak_flags']}")
    print(f"[UA-3] wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
