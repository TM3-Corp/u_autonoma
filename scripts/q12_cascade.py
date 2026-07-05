#!/usr/bin/env python3
"""Q12 — temporal cascade: feed earlier-cutoff OOF risk into the later model.
w4 gets risk_w2; w6 gets risk_w2+risk_w4; w8 gets risk_w2+risk_w4+risk_w6.
Leak-free (OOF calibrated risks are course-out by construction). Same per-fold
top-40 ExtraTrees selection — we also track whether the risk feature is SELECTED
and its importance rank (explainability / redundancy). Output: q12_cascade.json"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import warnings; warnings.filterwarnings("ignore")

FEAT=Path("data/puc/sota_results/tier1_clean/features")
OOF=Path("data/puc/sota_results/tier2_push")
OUT=Path("data/puc/sota_results/tier3_pooled/q12_cascade.json")
N=40; SEEDS=[42,43,44]; CASCADE={"4":[2],"6":[2,4],"8":[2,4,6]}

def risk(w):
    return pd.read_parquet(OOF/f"oof_calibrated_week_{w}.parquet")[["student_id","course_id","p"]].rename(columns={"p":f"risk_w{w}"})

def run(df, feats, track=None):
    X=df[feats].reset_index(drop=True); y=df["_y"].to_numpy().astype(int); g=df["course_id"].to_numpy()
    aucs=[]; sel={r:0 for r in (track or [])}; rank={r:[] for r in (track or [])}; nfold=0
    for seed in SEEDS:
        cv=StratifiedGroupKFold(5,shuffle=True,random_state=seed)
        oof=np.full(len(y),np.nan)
        for tr,te in cv.split(X,y,g):
            if y[tr].sum()<2: continue
            et=ExtraTreesClassifier(n_estimators=300,random_state=seed,n_jobs=-1); et.fit(X.iloc[tr].values,y[tr])
            ranked=[feats[i] for i in np.argsort(et.feature_importances_)[::-1]]
            s=ranked[:N]
            for r in (track or []):
                if r in s: sel[r]+=1
                rank[r].append(ranked.index(r)+1)
            nfold+=1
            m=CatBoostClassifier(auto_class_weights="Balanced",random_seed=seed,verbose=False,allow_writing_files=False)
            m.fit(X.iloc[tr][s].values,y[tr]); oof[te]=m.predict_proba(X.iloc[te][s].values)[:,1]
        mm=~np.isnan(oof); aucs.append(roc_auc_score(y[mm],oof[mm]))
    out={"auc":round(float(np.mean(aucs)),4)}
    if track:
        out["risk_selected_pct"]={r:round(sel[r]/nfold,2) for r in track}
        out["risk_mean_importance_rank"]={r:round(float(np.mean(rank[r])),1) for r in track}
        out["n_features_pool"]=len(feats)
    return out

def main():
    t0=time.time(); res={"config":"PUC clean, LOCO CatBoost Bal top-40, seeds{42,43,44}; risk=OOF calibrated (course-out)","weeks":{}}
    for w,priors in CASCADE.items():
        df=pd.read_parquet(FEAT/f"week_{w}_clean.parquet")
        rcols=[]
        for rw in priors:
            df=df.merge(risk(rw),on=["student_id","course_id"],how="left"); rcols.append(f"risk_w{rw}")
        df[rcols]=df[rcols].fillna(0.0)
        base_feats=[c for c in df.columns if c not in ["student_id","course_id","_y","_group"] and not c.startswith("risk_w")]
        base=run(df,base_feats)
        casc=run(df,base_feats+rcols,track=rcols)
        res["weeks"][w]={"priors_fed":priors,"baseline_auc":base["auc"],"cascade_auc":casc["auc"],
                         "delta":round(casc["auc"]-base["auc"],4),
                         "risk_selected_pct":casc["risk_selected_pct"],
                         "risk_mean_importance_rank":casc["risk_mean_importance_rank"],
                         "n_feature_pool":casc["n_features_pool"]}
        r=res["weeks"][w]
        print(f"[Q12] wk{w} (+{['risk_w'+str(p) for p in priors]}): baseline={base['auc']} cascade={casc['auc']} "
              f"Δ={r['delta']:+.4f} | risk selected%={r['risk_selected_pct']} mean_rank={r['risk_mean_importance_rank']} of {r['n_feature_pool']} [{time.time()-t0:.0f}s]",flush=True)
        OUT.write_text(json.dumps(res,indent=2))
    print(f"[Q12] DONE [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__": main()
