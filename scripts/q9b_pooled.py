#!/usr/bin/env python3
"""Q9b — finish the interrupted R2-pooled combined (basics + real shared historical)."""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
import tier3_common as T, q8_historical_matrix as Q8
import warnings; warnings.filterwarnings("ignore")
POOL=Path(T.POOL); HCOLS=Q8.HCOLS
def main():
    t0=time.time()
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
    cz=Q8.znorm(comb,allc); czc=[f"{c}_z" for c in allc]
    r2=cz[cz.course_id.isin(T.R2_EXPECTED)].reset_index(drop=True)
    kept,dropped,probe=Q8.probe_drop(r2,czc)
    hb=Q8.curve(cz,kept,T.R2_EXPECTED); bn,bv=Q8.best(hb)
    r=json.load(open(POOL/"q9_combined.json"))
    r["sets"]["R2_pooled"]={"combined_curve":hb,"combined_best":bv,"combined_bestN":bn,
        "basics_alone_best":0.7034,"n_feat":len(allc),"n_invariant":len(kept),"probe":probe,
        "delta_vs_basics":round(bv-0.7034,4)}
    (POOL/"q9_combined.json").write_text(json.dumps(r,indent=2))
    print(f"[Q9b] R2-pooled combined({len(allc)}→{len(kept)}inv,probe={probe}): best {bv}@N{bn} vs basics 0.7034 Δ={bv-0.7034:+.4f} [{time.time()-t0:.0f}s]",flush=True)
if __name__=="__main__": main()
