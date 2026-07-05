#!/usr/bin/env python3
"""Q7 — top-N sweep {2,5,10,20,40,60,80,100} on BASELINE (62) vs AUGMENTED (62+28),
per set. Fair test of whether the added families help when N is not capped at 40.
Reuses Q6's feature computation. Output: tier3_pooled/q7_topn_augmented.json
"""
import json, time
from pathlib import Path
import numpy as np
import tier3_common as T
import common_features as CF
import q6_feature_ablation as Q6
import warnings; warnings.filterwarnings("ignore")

POOL = Path(T.POOL)
NS = [2, 5, 10, 20, 40, 60, 80, 100]
SEEDS = [42, 43, 44]


def curve(mz, feats, courses, mix=None):
    d = mz[mz.course_id.isin(courses)]
    if mix:
        d = d[d.inst == mix]
    d = d.reset_index(drop=True)
    out = {}
    for N in NS:
        if N > len(feats) and NS.index(N) > 0 and NS[NS.index(N)-1] >= len(feats):
            out[str(N)] = out[str(NS[NS.index(N)-1])]  # capped: same as max
            continue
        aucs = []
        for s in SEEDS:
            oof, y, g, _ = T.oof_predict(d, kind="cat", N=min(N, len(feats)), seed=s, features=feats)
            a = T.pooled_auc(y, oof)
            if a is not None:
                aucs.append(a)
        out[str(N)] = round(float(np.mean(aucs)), 4) if aucs else None
    return out


def best(c):
    v = {k: x for k, x in c.items() if x is not None}
    bk = max(v, key=v.get)
    return int(bk), v[bk]


def main():
    t0 = time.time()
    ev, _ = Q6.load_events_ext(Q6.CUTOFF)
    newf, newcols = Q6.new_features(ev)
    base8 = T.load_week("8")
    base_cols = json.loads((POOL / "feature_schema.json").read_text())["base_features"]
    uni = base8[["inst", "sid", "course_id", "y"] + base_cols].copy()
    m = uni.merge(newf, on=["inst", "sid", "course_id"], how="left")
    m[newcols] = m[newcols].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    mz = Q6.znorm(m, base_cols + newcols)
    base_z = [f"{c}_z" for c in base_cols]           # 62
    aug_z = base_z + [f"{c}_z" for c in newcols]      # 90
    print(f"[Q7] baseline {len(base_z)} feats, augmented {len(aug_z)} feats", flush=True)

    sets = [("PUC_only", CF.PUC_COURSES, "PUC"),
            ("R2_UA_only", [c for c in T.R2_EXPECTED if c in CF.UA_COURSES], "UA"),
            ("R2_pooled", T.R2_EXPECTED, None)]
    out = {"n_specs": NS, "seeds": SEEDS, "n_base": len(base_z), "n_aug": len(aug_z), "sets": {}}
    for name, courses, mix in sets:
        cb = curve(mz, base_z, courses, mix)
        ca = curve(mz, aug_z, courses, mix)
        bnb, bvb = best(cb); bna, bva = best(ca)
        out["sets"][name] = {"baseline_curve": cb, "augmented_curve": ca,
                             "baseline_bestN": bnb, "baseline_best": bvb,
                             "augmented_bestN": bna, "augmented_best": bva,
                             "delta_best": round(bva - bvb, 4)}
        print(f"[Q7] {name}: baseline best {bvb}@N{bnb} | augmented best {bva}@N{bna} | "
              f"Δbest={bva-bvb:+.4f} [{time.time()-t0:.0f}s]", flush=True)
        print(f"       base curve: {cb}")
        print(f"       aug  curve: {ca}", flush=True)
        (POOL / "q7_topn_augmented.json").write_text(json.dumps(out, indent=2))
    print(f"[Q7] DONE [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
