#!/usr/bin/env python3
"""Q3 sensitivity — percentile-rank vs z-norm course-relative features.

Paul's point: PUC and UA raw activity differ ~9x; without course-relative context
the pooled model is off. G2 used per-course z-norm (course-relative). This tests
whether per-course PERCENTILE-RANK (distribution-free; maps every course's marginal
to uniform [0,1], scale- AND shape-invariant) is better for the wild activity gap:
 (a) does the institution probe pass with MORE features retained? and
 (b) does pooled R2 AUC improve vs z-norm?

Reuses the BASE features already in the pooled parquets (no re-featurization).
Output: tier3_pooled/q3_pctrank_results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import tier3_common as T
import warnings; warnings.filterwarnings("ignore")

POOL = Path(T.POOL)
SCHEMA = json.loads((POOL / "feature_schema.json").read_text())
BASE = SCHEMA["base_features"]
RS = 42


def pctrank_per_course(df):
    """Per-course percentile rank in [0,1] for every base feature."""
    parts = []
    for cid, g in df.groupby("course_id"):
        g = g.copy()
        cols = {}
        for c in BASE:
            v = g[c].to_numpy()
            cols[f"{c}_pct"] = (rankdata(v, method="average") - 0.5) / len(v) if len(v) else 0.0
        parts.append(pd.concat([g, pd.DataFrame(cols, index=g.index)], axis=1))
    return pd.concat(parts).sort_index()


def probe(df, cols):
    X = df[cols].values
    y = (df["inst"].values == "UA").astype(int)
    g = df["course_id"].values
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=RS)
    m = HistGradientBoostingClassifier(random_state=RS)
    p = cross_val_predict(m, X, y, cv=list(cv.split(X, y, g)), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def select_invariant(dfs, cols):
    pooled = pd.concat(list(dfs.values()), ignore_index=True)
    cur = list(cols)
    dropped = []
    while True:
        worst = max(probe(dfs[w], cur) for w in dfs)
        if worst <= 0.75 or len(cur) <= 15:
            break
        rf = RandomForestClassifier(n_estimators=300, random_state=RS, n_jobs=-1)
        rf.fit(pooled[cur].values, (pooled["inst"].values == "UA").astype(int))
        imp = sorted(zip(cur, rf.feature_importances_), key=lambda t: -t[1])
        k = 8 if worst > 0.90 else 4 if worst > 0.82 else 2 if worst > 0.77 else 1
        drop = [c for c, _ in imp[:k]]
        dropped += drop
        cur = [c for c in cur if c not in drop]
    after = {w: round(probe(dfs[w], cur), 4) for w in dfs}
    return cur, dropped, after


def pooled_r2_auc(df, cols, week_label):
    """CatBoost Balanced top-40 LOCO on R2-pooled using `cols`. Report per-seed mean."""
    d = df[df.course_id.isin(T.R2_EXPECTED)].reset_index(drop=True)
    aucs, pcs = [], []
    for s in [42, 43, 44, 45, 46]:
        oof, y, g, nsp = T.oof_predict(d, kind="cat", N=40, seed=s, features=cols)
        aucs.append(T.pooled_auc(y, oof))
        pcs.append(T.mean_per_course_auc(T.per_course_auc(d, y, oof)))
    return round(float(np.mean(aucs)), 4), round(float(np.mean(pcs)), 4)


def main():
    weeks = T.WEEKS
    dfs_z, dfs_p = {}, {}
    for w in weeks:
        df = T.load_week(w)
        dfs_z[w] = df
        dfs_p[w] = pctrank_per_course(df)
    pct_cols = [f"{c}_pct" for c in BASE]

    out = {"note": "Per-course percentile-rank (distribution-free) vs the shipped per-course "
                   "z-norm, on the same base features. Tests Paul's Q3: does rank-context beat "
                   "z-norm context given the ~9x PUC/UA activity gap.",
           "activity_gap": {"PUC_events_per_student_median": 2275, "UA_events_per_student_median": 246}}

    # 1. institution probe on ALL pct-rank features, per week (vs z-norm's 0.98/0.997 before drop)
    probe_all = {w: round(probe(dfs_p[w], pct_cols), 4) for w in weeks}
    out["probe_pctrank_all_features"] = probe_all
    out["probe_znorm_all_features_before_drop"] = {"2": 0.9815, "4": 0.9969, "6": 0.8631, "8": 0.8237, "full": 0.8313}
    print("[Q3] institution probe on ALL pct-rank feats (per week):", probe_all, flush=True)

    # 2. invariant selection on pct-rank
    kept, dropped, after = select_invariant(dfs_p, pct_cols)
    out["pctrank_invariant"] = {"n_kept": len(kept), "n_dropped": len(dropped),
                                "probe_after": after, "kept": kept, "dropped": dropped}
    print(f"[Q3] pct-rank invariance: kept {len(kept)}/{len(pct_cols)} (dropped {len(dropped)}); "
          f"probe after {after}", flush=True)

    # 3. pooled R2 AUC: z-norm (shipped 23) vs pct-rank invariant, weeks 2/4/8/full
    out["pooled_R2"] = {}
    for w in ["2", "4", "8", "full"]:
        z_auc, z_pc = pooled_r2_auc(dfs_z[w], T.MODEL_FEATURES, w)
        p_auc, p_pc = pooled_r2_auc(dfs_p[w], kept, w)
        out["pooled_R2"][w] = {"znorm_23feat": {"pooled_auc": z_auc, "mean_pc_auc": z_pc},
                               "pctrank_invariant": {"n_feat": len(kept), "pooled_auc": p_auc, "mean_pc_auc": p_pc}}
        print(f"[Q3] R2-pooled wk{w}: z-norm(23)={z_auc}/{z_pc}  vs  pct-rank({len(kept)})={p_auc}/{p_pc}", flush=True)
        (POOL / "q3_pctrank_results.json").write_text(json.dumps(out, indent=2))

    (POOL / "q3_pctrank_results.json").write_text(json.dumps(out, indent=2))
    print("[Q3] wrote q3_pctrank_results.json", flush=True)


if __name__ == "__main__":
    main()
