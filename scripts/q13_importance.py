#!/usr/bin/env python3
"""Q13 — feature importance across methods (gain / ExtraTrees / permutation / SHAP)
for PUC-only, UA-only, R2-pooled. Also SHAP direction (does high value raise/lower
risk) and cross-method agreement. Output: q13_importance.json"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr
import tier3_common as T
import warnings; warnings.filterwarnings("ignore")

POOL=Path(T.POOL); OUT=POOL/"q13_importance.json"
SCHEMA=json.loads((POOL/"feature_schema.json").read_text())
ZNORM=SCHEMA["znorm_features"]           # 62 (single-institution)
INVAR=SCHEMA["model_feature_cols"]       # 23 (pooled invariant)
WEEK="full"

def clean_name(f): return f.replace("_znorm","").replace("_z","")

def analyze(df, feats, label):
    X=df[feats].reset_index(drop=True); y=df["y"].to_numpy().astype(int)
    m=CatBoostClassifier(auto_class_weights="Balanced",random_seed=42,verbose=False,allow_writing_files=False)
    m.fit(X.values,y)
    pool=Pool(X.values,y)
    gain=pd.Series(m.get_feature_importance(pool,type="PredictionValuesChange"),index=feats)
    shap=m.get_feature_importance(pool,type="ShapValues")           # (n, k+1)
    shv=shap[:,:-1]
    shap_imp=pd.Series(np.abs(shv).mean(0),index=feats)
    # SHAP direction: corr(feature value, shap value) sign
    direction={}
    for i,f in enumerate(feats):
        c=np.corrcoef(X[f].values, shv[:,i])[0,1] if X[f].std()>0 else 0.0
        direction[f]="higher→more risk" if c>0.05 else ("higher→less risk" if c<-0.05 else "mixed")
    et=ExtraTreesClassifier(n_estimators=400,random_state=42,n_jobs=-1); et.fit(X.values,y)
    etimp=pd.Series(et.feature_importances_,index=feats)
    perm=permutation_importance(m,X.values,y,n_repeats=20,random_state=42,scoring="roc_auc",n_jobs=-1)
    permimp=pd.Series(perm.importances_mean,index=feats)
    # rankings + agreement
    def top(s,k=12): return [{"feature":clean_name(f),"value":round(float(s[f]),4),"dir":direction[f]} for f in s.sort_values(ascending=False).index[:k]]
    def jac(a,b,k=10): 
        A=set(a.sort_values(ascending=False).index[:k]); B=set(b.sort_values(ascending=False).index[:k]); return round(len(A&B)/len(A|B),2)
    out={"cohort":label,"n":int(len(df)),"fails":int(y.sum()),"n_features":len(feats),
         "top_by_shap":top(shap_imp),"top_by_gain":top(gain),"top_by_permutation":top(permimp),"top_by_extratrees":top(etimp),
         "agreement":{"shap_vs_gain_spearman":round(float(spearmanr(shap_imp,gain).correlation),3),
                      "shap_vs_permutation_spearman":round(float(spearmanr(shap_imp,permimp).correlation),3),
                      "gain_vs_extratrees_spearman":round(float(spearmanr(gain,etimp).correlation),3),
                      "shap_vs_gain_top10_jaccard":jac(shap_imp,gain),
                      "shap_vs_permutation_top10_jaccard":jac(shap_imp,permimp)}}
    return out

def main():
    t0=time.time()
    df=T.load_week(WEEK)
    res={"week":WEEK,"note":"importance descriptive (fit on full cohort); PUC/UA on 62 znorm, pooled on 23 invariant",
         "cohorts":{}}
    for label,sub,feats in [
        ("PUC_only", df[df.inst=="PUC"], ZNORM),
        ("UA_only", df[df.inst=="UA"], ZNORM),
        ("R2_pooled", df[df.course_id.isin(T.R2_EXPECTED)], INVAR)]:
        res["cohorts"][label]=analyze(sub.reset_index(drop=True),feats,label)
        a=res["cohorts"][label]
        print(f"[Q13] {label}: SHAP-vs-gain spearman={a['agreement']['shap_vs_gain_spearman']} "
              f"top10-jac={a['agreement']['shap_vs_gain_top10_jaccard']} | top SHAP: "
              f"{[t['feature'] for t in a['top_by_shap'][:5]]} [{time.time()-t0:.0f}s]",flush=True)
        OUT.write_text(json.dumps(res,indent=2))
    print(f"[Q13] DONE [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__": main()
