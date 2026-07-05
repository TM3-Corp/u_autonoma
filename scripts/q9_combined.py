#!/usr/bin/env python3
"""Q9 — final: basics(62) + real historical COMBINED vs basics alone (full horizon).
Directly tests whether adding the validated historical features to the basics lifts AUC."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import tier3_common as T, common_features as CF, q8_historical_matrix as Q8
import warnings; warnings.filterwarnings("ignore")
POOL=Path(T.POOL); NS=[40,60,80,100,120]; SEEDS=[42,43,44]
HCOLS=Q8.HCOLS

def main():
    t0=time.time()
    uf=T.load_week("full")
    uni=uf[["inst","sid","course_id","y"]].copy()
    base_cols=json.loads((POOL/"feature_schema.json").read_text())["base_features"]
    # raw basics from pooled_week_full
    basics=uf[["inst","sid","course_id"]+base_cols].copy()
    out={"scope":"basics + real historical, combined, full horizon","NS":NS,"sets":{}}

    # PUC-only combined
    pucU=uni[uni.inst=="PUC"].reset_index(drop=True)
    pf,pk,_=Q8.align(Q8.PUC_MASTER,"student_id",HCOLS["puc"],pucU)
    pf=pf.rename(columns={c:f"H_{c}" for c in pk}); hk=[f"H_{c}" for c in pk]
    comb=pf.merge(basics[basics.inst=="PUC"].drop(columns=["inst"]),on=["sid","course_id"],how="left")
    allc=hk+base_cols
    comb[allc]=comb[allc].fillna(0.0)
    cz=Q8.znorm(comb,allc); czc=[f"{c}_z" for c in allc]
    hb=Q8.curve(cz,czc,CF.PUC_COURSES)
    bn,bv=Q8.best(hb)
    out["sets"]["PUC_only"]={"combined_curve":hb,"combined_best":bv,"combined_bestN":bn,
        "basics_alone_best":0.8056,"n_feat":len(allc),"delta_vs_basics":round(bv-0.8056,4)}
    print(f"[Q9] PUC combined({len(allc)}): best {bv}@N{bn} vs basics 0.8056 Δ={bv-0.8056:+.4f} [{time.time()-t0:.0f}s]",flush=True)
    (POOL/"q9_combined.json").write_text(json.dumps(out,indent=2))

    # R2-pooled combined (basics + shared historical)
    pf2,_,_=Q8.align(Q8.PUC_MASTER,"student_id",HCOLS["shared"],pucU); pf2["inst"]="PUC"
    uaU=uni[uni.inst=="UA"].reset_index(drop=True)
    uf2,_,_=Q8.align(Q8.UA_MASTER,"user_id",HCOLS["shared"],uaU); uf2["inst"]="UA"
    sk=[c for c in HCOLS["shared"] if c in pf2.columns and c in uf2.columns]
    hist=pd.concat([pf2[["inst","sid","course_id","y"]+sk],uf2[["inst","sid","course_id","y"]+sk]],ignore_index=True)
    hist=hist.rename(columns={c:f"H_{c}" for c in sk}); hk=[f"H_{c}" for c in sk]
    comb=hist.merge(basics.drop(columns=["inst"]),on=["sid","course_id"],how="left")
    allc=hk+base_cols; comb[allc]=comb[allc].fillna(0.0)
    cz=Q8.znorm(comb,allc); czc=[f"{c}_z" for c in allc]
    r2=cz[cz.course_id.isin(T.R2_EXPECTED)].reset_index(drop=True)
    kept,dropped,probe=Q8.probe_drop(r2,czc)
    hb=Q8.curve(cz,kept,T.R2_EXPECTED); bn,bv=Q8.best(hb)
    out["sets"]["R2_pooled"]={"combined_curve":hb,"combined_best":bv,"combined_bestN":bn,
        "basics_alone_best":0.7034,"n_feat":len(allc),"n_invariant":len(kept),"probe":probe,
        "delta_vs_basics":round(bv-0.7034,4)}
    print(f"[Q9] R2-pooled combined({len(allc)}→{len(kept)}inv,probe={probe}): best {bv}@N{bn} vs basics 0.7034 Δ={bv-0.7034:+.4f} [{time.time()-t0:.0f}s]",flush=True)
    (POOL/"q9_combined.json").write_text(json.dumps(out,indent=2))
    print(f"[Q9] DONE [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__": main()
