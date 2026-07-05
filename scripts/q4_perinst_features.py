#!/usr/bin/env python3
"""Q4 — per-institution feature importance + top-N learning curves (Paul's request).

For PUC and UA SEPARATELY (no cross-institution invariance constraint — each
institution may use its own full feature set):
 1. Feature-importance ranking (mean ExtraTrees importance across LOCO train folds).
 2. Top-N performance: N ∈ {2,5,10,20,30,40,60}, per-fold leak-free top-N selection,
    CatBoost Balanced LOCO, seeds {42,43,44}. → optimal N per institution.
 3. Overlap of the two rankings (Jaccard at top-5/10/20) → are the same signals
    predictive at both institutions? (This is the mechanism behind the pooling null.)

Uses the 62 per-course z-norm features (course-relative; within-institution).
Weeks: 8 (reference) and full. Output: tier3_pooled/q4_perinst_features.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
import tier3_common as T
import warnings; warnings.filterwarnings("ignore")

POOL = Path(T.POOL)
SCHEMA = json.loads((POOL / "feature_schema.json").read_text())
ZFEATS = SCHEMA["znorm_features"]  # all 62 (no invariance drop, within-institution)
NS = [2, 5, 10, 20, 30, 40, 60]
SEEDS = [42, 43, 44]
WEEKS = ["8", "full"]


def inst_rank(d):
    """Mean ExtraTrees importance across LOCO train folds → ranked feature list."""
    X = d[ZFEATS].reset_index(drop=True)
    y = d["y"].to_numpy().astype(int)
    g = d["course_id"].to_numpy()
    folds, _ = T.loco_splits(X, y, g, 42)
    imp = np.zeros(len(ZFEATS))
    nf = 0
    for tr, _ in folds:
        if y[tr].sum() < 2:
            continue
        m = ExtraTreesClassifier(n_estimators=400, random_state=42, n_jobs=-1)
        m.fit(X.iloc[tr].values, y[tr])
        imp += m.feature_importances_
        nf += 1
    imp /= max(nf, 1)
    order = np.argsort(imp)[::-1]
    return [ZFEATS[i] for i in order], {ZFEATS[i]: round(float(imp[i]), 5) for i in order}


def topn_curve(d):
    """For each N, LOCO CatBoost with per-fold top-N ExtraTrees selection, seeds avg."""
    out = {}
    for N in NS:
        aucs = []
        for s in SEEDS:
            oof, y, g, _ = T.oof_predict(d, kind="cat", N=N, seed=s, features=ZFEATS)
            a = T.pooled_auc(y, oof)
            if a is not None:
                aucs.append(a)
        out[str(N)] = round(float(np.mean(aucs)), 4) if aucs else None
    return out


def jaccard(a, b, k):
    sa, sb = set(a[:k]), set(b[:k])
    return round(len(sa & sb) / len(sa | sb), 3)


def main():
    out = {"note": "Per-institution feature analysis on the 62 per-course z-norm features. "
                   "PUC (7 courses) and UA (10 courses) analyzed separately; no invariance drop.",
           "features_used": len(ZFEATS), "n_specs": NS, "seeds": SEEDS, "weeks": {}}
    for w in WEEKS:
        df = T.load_week(w)
        puc = df[df.inst == "PUC"].reset_index(drop=True)
        ua = df[df.inst == "UA"].reset_index(drop=True)
        rank_p, imp_p = inst_rank(puc)
        rank_u, imp_u = inst_rank(ua)
        curve_p = topn_curve(puc)
        curve_u = topn_curve(ua)
        best_p = max(curve_p, key=lambda k: curve_p[k] if curve_p[k] else -1)
        best_u = max(curve_u, key=lambda k: curve_u[k] if curve_u[k] else -1)
        out["weeks"][w] = {
            "PUC": {"top15": rank_p[:15], "topn_auc": curve_p, "optimal_N": int(best_p),
                    "optimal_auc": curve_p[best_p]},
            "UA": {"top15": rank_u[:15], "topn_auc": curve_u, "optimal_N": int(best_u),
                   "optimal_auc": curve_u[best_u]},
            "overlap_jaccard": {"top5": jaccard(rank_p, rank_u, 5),
                                "top10": jaccard(rank_p, rank_u, 10),
                                "top20": jaccard(rank_p, rank_u, 20)},
            "shared_top10": sorted(set(rank_p[:10]) & set(rank_u[:10])),
            "puc_only_top10": [f for f in rank_p[:10] if f not in rank_u[:10]],
            "ua_only_top10": [f for f in rank_u[:10] if f not in rank_p[:10]],
        }
        r = out["weeks"][w]
        print(f"\n[Q4] wk{w}:")
        print(f"  PUC top-N AUC: {curve_p}  → optimal N={best_p} ({curve_p[best_p]})")
        print(f"  UA  top-N AUC: {curve_u}  → optimal N={best_u} ({curve_u[best_u]})")
        print(f"  Jaccard top5/10/20: {r['overlap_jaccard']}")
        print(f"  shared top10: {r['shared_top10']}")
        (POOL / "q4_perinst_features.json").write_text(json.dumps(out, indent=2))
    (POOL / "q4_perinst_features.json").write_text(json.dumps(out, indent=2))
    print("\n[Q4] wrote q4_perinst_features.json", flush=True)


if __name__ == "__main__":
    main()
