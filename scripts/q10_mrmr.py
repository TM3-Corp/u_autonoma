#!/usr/bin/env python3
"""Q10 — mRMR (min-Redundancy-Max-Relevance) feature selection vs ExtraTrees top-N.
Directly targets the redundancy we found. Combined matrix (basics+historical),
per-fold leak-free mRMR ranking, LOCO CatBoost, sweep N. Output: q10_mrmr.json"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedGroupKFold
import tier3_common as T, q8_historical_matrix as Q8
import warnings; warnings.filterwarnings("ignore")
POOL=Path(T.POOL); HCOLS=Q8.HCOLS; NS=[10,20,30,40,60]; SEEDS=[42,43,44]

def mrmr_rank(X, y, k):
    rel=pd.Series(mutual_info_classif(X.values,y,random_state=42),index=X.columns)
    corr=X.corr().abs().fillna(0.0)
    sel=[rel.idxmax()]; cand=[c for c in X.columns if c!=sel[0]]
    while len(sel)<min(k,X.shape[1]) and cand:
        red=corr.loc[cand,sel].mean(axis=1)
        nxt=(rel[cand]-red).idxmax(); sel.append(nxt); cand.remove(nxt)
    return sel

def oof_mrmr(df, feats, N, seed):
    X=df[feats].reset_index(drop=True); y=df["y"].to_numpy().astype(int); g=df["course_id"].to_numpy()
    cv=StratifiedGroupKFold(min(5,len(np.unique(g))),shuffle=True,random_state=seed)
    oof=np.full(len(y),np.nan)
    for tr,te in cv.split(X,y,g):
        if y[tr].sum()<2: continue
        rk=mrmr_rank(X.iloc[tr],y[tr],N)[:N]
        m=T.make_model("cat",y[tr],seed); m.fit(X.iloc[tr][rk].values,y[tr])
        oof[te]=m.predict_proba(X.iloc[te][rk].values)[:,1]
    return T.pooled_auc(y,oof)

def curve(df,feats,courses,mix=None):
    d=df[df.course_id.isin(courses)]
    if mix: d=d[d.inst==mix]
    d=d.reset_index(drop=True); out={}
    for N in NS:
        a=[oof_mrmr(d,feats,N,s) for s in SEEDS]; a=[x for x in a if x is not None]
        out[str(N)]=round(float(np.mean(a)),4) if a else None
    return out

def build_combined():
    uf=T.load_week("full"); uni=uf[["inst","sid","course_id","y"]].copy()
    base_cols=json.loads((POOL/"feature_schema.json").read_text())["base_features"]
    basics=uf[["inst","sid","course_id"]+base_cols].copy()
    pucU=uni[uni.inst=="PUC"].reset_index(drop=True); uaU=uni[uni.inst=="UA"].reset_index(drop=True)
    pf,_,_=Q8.align(Q8.PUC_MASTER,"student_id",HCOLS["shared"],pucU); pf["inst"]="PUC"
    uff,_,_=Q8.align(Q8.UA_MASTER,"user_id",HCOLS["shared"],uaU); uff["inst"]="UA"
    sk=[c for c in HCOLS["shared"] if c in pf.columns and c in uff.columns]
    hist=pd.concat([pf[["inst","sid","course_id","y"]+sk],uff[["inst","sid","course_id","y"]+sk]],ignore_index=True)
    hist=hist.rename(columns={c:f"H_{c}" for c in sk}); hk=[f"H_{c}" for c in sk]
    comb=hist.merge(basics.drop(columns=["inst"]),on=["sid","course_id"],how="left")
    allc=hk+base_cols; comb[allc]=comb[allc].fillna(0.0)
    return Q8.znorm(comb,allc),[f"{c}_z" for c in allc]

def main():
    t0=time.time()
    cz,czc=build_combined()
    print(f"[Q10] combined matrix {len(czc)} feats",flush=True)
    out={"method":"mRMR (MI relevance − mean|corr| redundancy), per-fold leak-free","NS":NS,"sets":{},
         "baselines_extratrees":{"PUC_only":0.8056,"R2_UA_only":0.7588,"R2_pooled":0.7034}}
    for name,courses,mix in [("PUC_only",T.PUC_COURSES,"PUC"),
                              ("R2_UA_only",[c for c in T.R2_EXPECTED if c in Q8.CF.UA_COURSES],"UA"),
                              ("R2_pooled",T.R2_EXPECTED,None)]:
        c=curve(cz,czc,courses,mix)
        v={k:x for k,x in c.items() if x}; bn=max(v,key=v.get)
        base=out["baselines_extratrees"][name]
        out["sets"][name]={"mrmr_curve":c,"mrmr_best":v[bn],"mrmr_bestN":int(bn),
                           "extratrees_best":base,"delta_vs_extratrees":round(v[bn]-base,4)}
        print(f"[Q10] {name}: mRMR best {v[bn]}@N{bn} vs ExtraTrees {base} Δ={v[bn]-base:+.4f} [{time.time()-t0:.0f}s]",flush=True)
        (POOL/"q10_mrmr.json").write_text(json.dumps(out,indent=2))
    print(f"[Q10] DONE [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__": main()
