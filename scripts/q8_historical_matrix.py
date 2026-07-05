#!/usr/bin/env python3
"""Q8 — DEFINITIVE test: the ACTUAL historical enriched-feature matrices (real
implementations, ~230-240 feats/institution, 116 shared) vs the Tier-3 basics (62).
Full-horizon (both masters are full-horizon; PUC has no cutoff snapshots).
Top-N sweep {2,5,10,20,40,60,80,100}. Leak-free: grade-value & jaccard_to_passing
columns dropped. Output: tier3_pooled/q8_historical.json
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tier3_common as T
import common_features as CF
import warnings; warnings.filterwarnings("ignore")

POOL = Path(T.POOL)
UA_MASTER = Path("data/enriched_features/normalized_features.parquet")
PUC_MASTER = Path("data/puc/enriched_features/all_features_sota.parquet")
HCOLS = json.load(open("/home/paul/.claude/jobs/3c9df0b8/tmp/histcols.json"))
NS = [2, 5, 10, 20, 40, 60, 80, 100]
SEEDS = [42, 43, 44]


def znorm(df, cols):
    parts = []
    for cid, g in df.groupby("course_id"):
        g = g.copy(); z = {}
        for c in cols:
            v = g[c]; sd = v.std()
            z[f"{c}_z"] = (v - v.mean()) / sd if sd and sd > 0 else 0.0
        parts.append(pd.concat([g, pd.DataFrame(z, index=g.index)], axis=1))
    return pd.concat(parts).sort_index()


def align(master_path, idcol, feat_cols, uni_inst):
    m = pd.read_parquet(master_path)
    m = m.rename(columns={idcol: "sid"})
    if idcol == "user_id":
        m["sid"] = m["sid"].map(CF.normalize_user_id)
    m["sid"] = m["sid"].astype("int64"); m["course_id"] = m["course_id"].astype("int64")
    keep = [c for c in feat_cols if c in m.columns]
    m = m[["sid", "course_id"] + keep].drop_duplicates(["sid", "course_id"])
    merged = uni_inst.merge(m, on=["sid", "course_id"], how="left")
    match = merged[keep].notna().any(axis=1).mean()
    merged[keep] = merged[keep].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return merged, keep, float(match)


def curve(df, feats, courses, mix=None):
    d = df[df.course_id.isin(courses)]
    if mix: d = d[d.inst == mix]
    d = d.reset_index(drop=True)
    out = {}
    for N in NS:
        aucs = []
        for s in SEEDS:
            oof, y, g, _ = T.oof_predict(d, kind="cat", N=min(N, len(feats)), seed=s, features=feats)
            a = T.pooled_auc(y, oof)
            if a is not None: aucs.append(a)
        out[str(N)] = round(float(np.mean(aucs)), 4) if aucs else None
    return out


def best(c):
    v = {k: x for k, x in c.items() if x is not None}
    bk = max(v, key=v.get); return int(bk), v[bk]


def probe_drop(df, cols):
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    def probe(cs):
        X=df[cs].values; y=(df.inst.values=="UA").astype(int); g=df.course_id.values
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=42)
        p=cross_val_predict(HistGradientBoostingClassifier(random_state=42),X,y,cv=list(cv.split(X,y,g)),method="predict_proba")[:,1]
        return float(roc_auc_score(y,p))
    cur=list(cols); dropped=[]
    while probe(cur)>0.75 and len(cur)>15:
        rf=RandomForestClassifier(n_estimators=300,random_state=42,n_jobs=-1)
        rf.fit(df[cur].values,(df.inst.values=="UA").astype(int))
        imp=sorted(zip(cur,rf.feature_importances_),key=lambda t:-t[1])
        k=8 if probe(cur)>0.85 else 3
        for c,_ in imp[:k]: cur.remove(c); dropped.append(c)
    return cur, dropped, round(probe(cur),4)


def main():
    t0=time.time()
    unifull = T.load_week("full")
    uni = unifull[["inst","sid","course_id","y"]].copy()
    base_cols = json.loads((POOL/"feature_schema.json").read_text())["base_features"]
    base_z = [f"{c}_znorm" for c in base_cols]
    # baseline frames already have _znorm in pooled_week_full
    baseP = unifull[unifull.inst=="PUC"].reset_index(drop=True)
    baseUA = unifull[unifull.inst=="UA"].reset_index(drop=True)

    out={"scope":"full-horizon; historical enriched matrices vs Tier-3 basics","n_specs":NS,"seeds":SEEDS,"sets":{}}

    # PUC-only
    pucU = uni[uni.inst=="PUC"].reset_index(drop=True)
    pf, pkeep, pmatch = align(PUC_MASTER,"student_id",HCOLS["puc"],pucU)
    pfz = znorm(pf, pkeep); pfz_cols=[f"{c}_z" for c in pkeep]
    print(f"[Q8] PUC hist: {len(pkeep)} feats, match={pmatch:.2f}", flush=True)
    hb=curve(pfz,pfz_cols,CF.PUC_COURSES); bb=curve(baseP,base_z,CF.PUC_COURSES)
    hbn,hbv=best(hb); bbn,bbv=best(bb)
    out["sets"]["PUC_only"]={"hist_curve":hb,"base_curve":bb,"hist_best":hbv,"hist_bestN":hbn,
        "base_best":bbv,"base_bestN":bbn,"delta_best":round(hbv-bbv,4),"n_hist":len(pkeep),"match":round(pmatch,3)}
    print(f"[Q8] PUC-only: hist {hbv}@N{hbn} vs base {bbv}@N{bbn} Δ={hbv-bbv:+.4f} [{time.time()-t0:.0f}s]",flush=True)
    (POOL/"q8_historical.json").write_text(json.dumps(out,indent=2))

    # UA-R2
    uaU = uni[uni.inst=="UA"].reset_index(drop=True)
    uf, ukeep, umatch = align(UA_MASTER,"user_id",HCOLS["ua"],uaU)
    ufz = znorm(uf, ukeep); ufz_cols=[f"{c}_z" for c in ukeep]
    r2ua=[c for c in T.R2_EXPECTED if c in CF.UA_COURSES]
    print(f"[Q8] UA hist: {len(ukeep)} feats, match={umatch:.2f}", flush=True)
    hb=curve(ufz,ufz_cols,r2ua); bb=curve(baseUA,base_z,r2ua)
    hbn,hbv=best(hb); bbn,bbv=best(bb)
    out["sets"]["R2_UA_only"]={"hist_curve":hb,"base_curve":bb,"hist_best":hbv,"hist_bestN":hbn,
        "base_best":bbv,"base_bestN":bbn,"delta_best":round(hbv-bbv,4),"n_hist":len(ukeep),"match":round(umatch,3)}
    print(f"[Q8] UA-R2: hist {hbv}@N{hbn} vs base {bbv}@N{bbn} Δ={hbv-bbv:+.4f} [{time.time()-t0:.0f}s]",flush=True)
    (POOL/"q8_historical.json").write_text(json.dumps(out,indent=2))

    # R2-pooled on SHARED cols
    shared=HCOLS["shared"]
    pf2,_,_=align(PUC_MASTER,"student_id",shared,pucU); pf2["inst"]="PUC"
    uf2,_,_=align(UA_MASTER,"user_id",shared,uaU); uf2["inst"]="UA"
    skeep=[c for c in shared if c in pf2.columns and c in uf2.columns]
    pooled=pd.concat([pf2[["inst","sid","course_id","y"]+skeep],uf2[["inst","sid","course_id","y"]+skeep]],ignore_index=True)
    poolz=znorm(pooled,skeep); poolz_cols=[f"{c}_z" for c in skeep]
    r2p=poolz[poolz.course_id.isin(T.R2_EXPECTED)].reset_index(drop=True)
    kept,dropped,probe=probe_drop(r2p,poolz_cols)
    print(f"[Q8] pooled shared: {len(skeep)} feats → invariant {len(kept)} (probe={probe})", flush=True)
    hb=curve(poolz,kept,T.R2_EXPECTED); bb=curve(unifull,T.MODEL_FEATURES,T.R2_EXPECTED)
    hbn,hbv=best(hb); bbn,bbv=best(bb)
    out["sets"]["R2_pooled"]={"hist_curve":hb,"base_curve":bb,"hist_best":hbv,"hist_bestN":hbn,
        "base_best":bbv,"base_bestN":bbn,"delta_best":round(hbv-bbv,4),"n_shared":len(skeep),
        "n_invariant":len(kept),"probe_auc":probe}
    print(f"[Q8] R2-pooled: hist {hbv}@N{hbn} vs base {bbv}@N{bbn} Δ={hbv-bbv:+.4f} [{time.time()-t0:.0f}s]",flush=True)
    out["verdict"]=("ROI CONFIRMED — full rebuild justified" if any(out["sets"][s]["delta_best"]>0.015 for s in out["sets"])
                    else "modest/mixed — feature ceiling near basics")
    (POOL/"q8_historical.json").write_text(json.dumps(out,indent=2))
    print(f"[Q8] VERDICT: {out['verdict']} [{time.time()-t0:.0f}s]",flush=True)


if __name__=="__main__":
    main()
