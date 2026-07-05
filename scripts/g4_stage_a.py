#!/usr/bin/env python3
"""G4 — Stage A: course-set × mix (FROZEN grid).

Reference config (CatBoost Balanced, top-40/fold, uncalibrated) on weeks {4,8}:
rules {R0,R1,R2} × mixes {pooled, PUC, UA} × seeds {42..46}, LOCO grouped by course.
Skip cells with <4 courses for LOCO (justified). Report pooled-OOF AUC + mean
per-course AUC + recall@20% (per seed, then mean over seeds).

Output: tier3_pooled/stageA_results.json
"""
import json, time
from pathlib import Path
import numpy as np
import tier3_common as T

POOL = Path(T.POOL)
OUT = POOL / "stageA_results.json"
WEEKS = ["4", "8"]
RULES = ["R0", "R1", "R2"]
MIXES = ["pooled", "PUC", "UA"]
SEEDS = [42, 43, 44, 45, 46]
N = 40
MIN_COURSES_LOCO = 4


def main():
    t0 = time.time()
    results = {"config": "CatBoost Balanced, top-40/fold, uncalibrated, LOCO grouped by course",
               "weeks": WEEKS, "rules": RULES, "mixes": MIXES, "seeds": SEEDS,
               "cells": {}}
    dfw = {w: T.load_week(w) for w in WEEKS}
    T.assert_rules(dfw["8"])

    for w in WEEKS:
        results["cells"][w] = {}
        for rule in RULES:
            results["cells"][w][rule] = {}
            for mix in MIXES:
                d = T.subset(dfw[w], rule, mix)
                ncourses = d.course_id.nunique()
                key = f"{rule}/{mix}"
                if ncourses < MIN_COURSES_LOCO:
                    results["cells"][w][rule][mix] = {
                        "skipped": True, "reason": f"{ncourses} courses < {MIN_COURSES_LOCO} for LOCO",
                        "n_courses": int(ncourses), "n": int(len(d)), "fails": int(d.y.sum())}
                    print(f"[G4] wk{w} {key}: SKIP ({ncourses} courses)", flush=True)
                    continue
                per_seed = []
                for s in SEEDS:
                    oof, y, g, nsp = T.oof_predict(d, kind="cat", N=N, seed=s)
                    pc = T.per_course_auc(d, y, oof)
                    per_seed.append({
                        "seed": s,
                        "pooled_auc": T.pooled_auc(y, oof),
                        "mean_percourse_auc": T.mean_per_course_auc(pc),
                        "recall20": T.recall_at(y, oof, 0.20),
                        "n_splits": nsp,
                        "per_course_auc": pc,
                    })
                pooled_aucs = [c["pooled_auc"] for c in per_seed if c["pooled_auc"] is not None]
                pc_aucs = [c["mean_percourse_auc"] for c in per_seed if c["mean_percourse_auc"] is not None]
                rec = [c["recall20"] for c in per_seed if c["recall20"] is not None]
                results["cells"][w][rule][mix] = {
                    "skipped": False, "n_courses": int(ncourses),
                    "n": int(len(d)), "fails": int(d.y.sum()),
                    "prevalence": round(float(d.y.mean()), 4),
                    "mean_pooled_auc": round(float(np.mean(pooled_aucs)), 4) if pooled_aucs else None,
                    "std_pooled_auc": round(float(np.std(pooled_aucs)), 4) if pooled_aucs else None,
                    "mean_percourse_auc": round(float(np.mean(pc_aucs)), 4) if pc_aucs else None,
                    "mean_recall20": round(float(np.mean(rec)), 4) if rec else None,
                    "per_seed": per_seed,
                }
                r = results["cells"][w][rule][mix]
                print(f"[G4] wk{w} {key}: n={r['n']} f={r['fails']} courses={ncourses} "
                      f"pooledAUC={r['mean_pooled_auc']} pcAUC={r['mean_percourse_auc']} "
                      f"rec20={r['mean_recall20']} [{time.time()-t0:.0f}s]", flush=True)
                OUT.write_text(json.dumps(results, indent=2))

    OUT.write_text(json.dumps(results, indent=2))
    # verifier summary
    r2p = results["cells"]["8"]["R2"]["pooled"]
    print(f"\n[G4] R2-pooled wk8: pooledAUC={r2p['mean_pooled_auc']} "
          f"pcAUC={r2p['mean_percourse_auc']} rec20={r2p['mean_recall20']} (n={r2p['n']}, {r2p['fails']} fails)")
    n_cells = sum(1 for w in WEEKS for rule in RULES for mix in MIXES)
    n_skip = sum(1 for w in WEEKS for rule in RULES for mix in MIXES
                 if results["cells"][w][rule][mix].get("skipped"))
    print(f"[G4] {n_cells} cells, {n_skip} skipped (R2/PUC each week), each ran × {len(SEEDS)} seeds. "
          f"wrote {OUT} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
