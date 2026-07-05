#!/usr/bin/env python3
"""Q11 — time-to-disengagement survival framing. Event = student goes silent
before course end (censored if active to the end). Cox PH → disengagement risk;
does that risk predict FAILURE, and does it add to the classifier?
Output: q11_survival.json"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import roc_auc_score
import tier3_common as T, common_features as CF
import warnings; warnings.filterwarnings("ignore")
POOL=Path(T.POOL)
COVARS=["total_events","active_days","n_active_weeks","mean_weekly_views","trend_slope",
        "session_regularity","weekend_share","share_evening","first_event_day","max_inactivity_gap_days"]

def build_surv():
    """Per (inst,sid,course): surv_time (last active week+1), event (disengaged before end)."""
    frames=[]
    for path,idc,inst,cids in [(CF.PUC_CLEAN,"student_id","PUC",CF.PUC_COURSES),
                               (CF.UA_CLEAN,"user_id","UA",CF.UA_COURSES)]:
        df=pd.read_parquet(path,columns=[idc,"course_id","created_at"])
        df=df[df.course_id.isin(cids)].copy(); df["course_id"]=df.course_id.astype("int64")
        df=df.rename(columns={idc:"sid","created_at":"ts"})
        if not pd.api.types.is_datetime64_any_dtype(df["ts"]): df["ts"]=pd.to_datetime(df["ts"],utc=True)
        starts=df.groupby("course_id")["ts"].quantile(0.05).to_dict()
        df["wk"]=((df["ts"]-df["course_id"].map(starts)).dt.total_seconds()/86400/7).clip(lower=0).astype(int)
        last=df.groupby(["sid","course_id"])["wk"].max().rename("last_wk").reset_index()
        cend=df.groupby("course_id")["wk"].quantile(0.90).rename("course_end").reset_index()
        last=last.merge(cend,on="course_id"); last["inst"]=inst
        frames.append(last)
    s=pd.concat(frames,ignore_index=True)
    s["surv_time"]=s["last_wk"]+1
    s["event"]=(s["last_wk"]<s["course_end"]-2).astype(int)  # disengaged before end
    return s[["inst","sid","course_id","surv_time","event"]]

def znorm(df,cols):
    parts=[]
    for cid,g in df.groupby("course_id"):
        g=g.copy(); z={}
        for c in cols:
            v=g[c]; sd=v.std(); z[c]=(v-v.mean())/sd if sd and sd>0 else 0.0
        for c in cols: g[c]=z[c]
        parts.append(g)
    return pd.concat(parts).sort_index()

def main():
    t0=time.time()
    uf=T.load_week("full")
    surv=build_surv()
    d=uf[["inst","sid","course_id","y"]+COVARS].merge(surv,on=["inst","sid","course_id"],how="inner")
    out={"framing":"time-to-disengagement (event=silent >2wk before course end); Cox PH per institution",
         "sets":{}}
    for inst in ["PUC","UA"]:
        di=d[d.inst==inst].copy()
        # disengagement rate + KM separation fail vs pass
        km_fail=di[di.y==1]; km_pass=di[di.y==0]
        lr=logrank_test(km_fail["surv_time"],km_pass["surv_time"],km_fail["event"],km_pass["event"])
        # Cox (penalized) on znormed covars
        dz=znorm(di.copy(),COVARS)
        cph=CoxPHFitter(penalizer=0.1)
        cph.fit(dz[["surv_time","event"]+COVARS],duration_col="surv_time",event_col="event")
        risk=cph.predict_partial_hazard(dz[COVARS]).values
        auc_risk=float(roc_auc_score(di["y"].values, risk))
        # does survival risk ADD to the classifier? CatBoost covars vs covars+risk, LOCO
        base=dz[["inst","sid","course_id","y"]+COVARS].copy()
        aug=base.copy(); aug["surv_risk"]=(risk-risk.mean())/ (risk.std() or 1)
        courses=CF.PUC_COURSES if inst=="PUC" else CF.UA_COURSES
        def ev(df,feats):
            aucs=[]
            for s in [42,43,44]:
                oof,y,g,_=T.oof_predict(df,kind="cat",N=len(feats),seed=s,features=feats)
                a=T.pooled_auc(y,oof)
                if a: aucs.append(a)
            return round(float(np.mean(aucs)),4)
        auc_base=ev(base,COVARS); auc_aug=ev(aug,COVARS+["surv_risk"])
        out["sets"][inst]={
            "n":int(len(di)),"disengagement_rate":round(float(di["event"].mean()),3),
            "cox_concordance":round(float(cph.concordance_index_),4),
            "logrank_p_fail_vs_pass":round(float(lr.p_value),5),
            "auc_survrisk_predicts_fail":round(auc_risk,4),
            "clf_covars_only":auc_base,"clf_covars_plus_survrisk":auc_aug,
            "delta_adding_survrisk":round(auc_aug-auc_base,4),
            "top_cox_coefs":cph.params_.reindex(cph.params_.abs().sort_values(ascending=False).index).head(5).round(3).to_dict()}
        print(f"[Q11] {inst}: diseng_rate={out['sets'][inst]['disengagement_rate']} "
              f"cox_C={out['sets'][inst]['cox_concordance']} logrank_p={out['sets'][inst]['logrank_p_fail_vs_pass']} "
              f"AUC(risk→fail)={auc_risk:.3f} | clf {auc_base}→{auc_aug} Δ={auc_aug-auc_base:+.4f} [{time.time()-t0:.0f}s]",flush=True)
        (POOL/"q11_survival.json").write_text(json.dumps(out,indent=2))
    print(f"[Q11] DONE [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__": main()
