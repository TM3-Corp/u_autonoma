#!/usr/bin/env python3
"""Q6 — quick ablation: do the historically-top MISSING feature families lift AUC?

Adds shared-computable families the Tier-3 pipeline skipped, at week 8:
 - proactivity access-percentile per content category (peer first-access rank) + access_rate
 - course-relative timing histogram (activity_bin_1..5, first/last/median access_pct)
 - resource-diversity / graph core (unique_resources, coverage, diversity, category entropy, repetition)
 - device (mobile_pct), participation_rate, grades_check_per_week
Compares baseline (current 62 features) vs augmented, on PUC-only and R2-pooled,
CatBoost Balanced top-40 LOCO, seeds {42,43,44}. Output: tier3_pooled/q6_ablation.json
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata
import tier3_common as T
import common_features as CF
import warnings; warnings.filterwarnings("ignore")

POOL = Path(T.POOL)
CUTOFF = 8
CONTENT_BINS = ["files", "assignments", "quizzes", "discussions", "pages", "modules"]
SEEDS = [42, 43, 44]
MOBILE = r'(?i)Mobile|Android|iPhone|iPad|Windows Phone'


def load_events_ext(cutoff):
    """Both institutions, cutoff-filtered, with resource_id/user_agent/participated/http_method."""
    frames = []
    for path, idcol, catcol, cmap, inst, cids in [
        (CF.PUC_CLEAN, "student_id", "category", CF.PUC_CAT_MAP, "PUC", CF.PUC_COURSES),
        (CF.UA_CLEAN, "user_id", "resource_type", CF.UA_RT_MAP, "UA", CF.UA_COURSES)]:
        cols = [idcol, "course_id", "created_at", catcol, "resource_id", "user_agent",
                "participated", "http_method"]
        df = pd.read_parquet(path, columns=cols)
        df = df[df["course_id"].isin(cids)].copy()
        df["course_id"] = df["course_id"].astype("int64")
        df = df.rename(columns={idcol: "sid", "created_at": "ts", catcol: "raw"})
        if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        df["bin"] = df["raw"].map(cmap).fillna("other")
        df["inst"] = inst
        frames.append(df)
    ev = pd.concat(frames, ignore_index=True)
    starts = ev.groupby("course_id")["ts"].quantile(CF.PERCENTILE).to_dict()
    ev["start"] = ev["course_id"].map(starts)
    bound = ev["start"] + pd.Timedelta(weeks=cutoff)
    ev = ev[ev["ts"] <= bound].copy()
    ev["dss"] = (ev["ts"] - ev["start"]).dt.total_seconds() / 86400.0
    ev["is_mobile"] = ev["user_agent"].fillna("").str.contains(MOBILE, regex=True).astype(int)
    ev["is_post"] = (ev["http_method"].fillna("") == "POST").astype(int)
    ev["part"] = ev["participated"].fillna(False).astype(int)
    return ev, starts


def new_features(ev):
    g = ev.groupby(["inst", "sid", "course_id"])
    base = g.agg(n=("ts", "size"), mobile_pct=("is_mobile", "mean"),
                 part_rate=("part", "mean"), post_rate=("is_post", "mean"),
                 uniq=("resource_id", "nunique")).reset_index()
    base["resource_diversity"] = base["uniq"] / base["n"]
    base["repetition"] = base["n"] / base["uniq"].clip(lower=1)

    # course-level unique resources for coverage
    cu = ev.groupby("course_id")["resource_id"].nunique().rename("course_uniq").reset_index()
    base = base.merge(cu, on="course_id", how="left")
    base["resource_coverage"] = base["uniq"] / base["course_uniq"].clip(lower=1)

    # category entropy (diversity over bins) + grades_check_per_week
    def cat_ent(s):
        c = s.value_counts().values.astype(float)
        p = c / c.sum()
        return float(-(p * np.log(p)).sum())
    ent = g["bin"].apply(cat_ent).rename("category_diversity").reset_index()
    base = base.merge(ent, on=["inst", "sid", "course_id"], how="left")
    grades = ev[ev.bin == "grades"].groupby(["inst", "sid", "course_id"]).size().rename("grades_n").reset_index()
    span_wk = g["dss"].apply(lambda d: max((d.max() - d.min()) / 7.0, 1.0)).rename("span_wk").reset_index()
    base = base.merge(grades, on=["inst", "sid", "course_id"], how="left").merge(span_wk, on=["inst", "sid", "course_id"], how="left")
    base["grades_check_per_week"] = base["grades_n"].fillna(0) / base["span_wk"]

    # course-relative timing: activity_bin_1..5 (share of events in each 5th of [start,cutoff]) + access_pct
    # normalize dss by each course's max dss at cutoff
    cmax = ev.groupby("course_id")["dss"].max().rename("cmax").reset_index()
    ev2 = ev.merge(cmax, on="course_id", how="left")
    ev2["pos"] = (ev2["dss"] / ev2["cmax"].clip(lower=1e-9)).clip(0, 1)
    ev2["binid"] = np.minimum((ev2["pos"] * 5).astype(int), 4)
    hist = ev2.groupby(["inst", "sid", "course_id", "binid"]).size().unstack(fill_value=0)
    hist = hist.div(hist.sum(axis=1).clip(lower=1), axis=0)
    hist.columns = [f"activity_bin_{i+1}" for i in hist.columns]
    hist = hist.reset_index()
    apct = ev2.groupby(["inst", "sid", "course_id"])["pos"].agg(
        first_access_pct="min", last_access_pct="max", median_access_pct="median").reset_index()
    base = base.merge(hist, on=["inst", "sid", "course_id"], how="left").merge(apct, on=["inst", "sid", "course_id"], how="left")

    # proactivity: per (course,resource) rank students by first-access (earlier=1); per (sid,cat) mean pct + access_rate
    fa = ev.groupby(["course_id", "resource_id", "inst", "sid", "bin"])["dss"].min().reset_index()
    fa["acc_pct"] = 1 - (fa.groupby(["course_id", "resource_id"])["dss"].rank(method="average") - 1) / \
        (fa.groupby(["course_id", "resource_id"])["dss"].transform("count") - 1).clip(lower=1)
    # course resources per bin (for access_rate)
    cres = ev.groupby(["course_id", "bin"])["resource_id"].nunique().rename("cat_nres").reset_index()
    for b in CONTENT_BINS:
        sub = fa[fa.bin == b]
        agg = sub.groupby(["inst", "sid", "course_id"]).agg(
            mp=("acc_pct", "mean"), nres=("resource_id", "nunique")).reset_index()
        agg = agg.merge(cres[cres.bin == b][["course_id", "cat_nres"]], on="course_id", how="left")
        agg[f"{b}_mean_pct"] = agg["mp"]
        agg[f"{b}_access_rate"] = agg["nres"] / agg["cat_nres"].clip(lower=1)
        base = base.merge(agg[["inst", "sid", "course_id", f"{b}_mean_pct", f"{b}_access_rate"]],
                          on=["inst", "sid", "course_id"], how="left")

    drop = ["n", "uniq", "course_uniq", "grades_n", "span_wk"]
    base = base.drop(columns=[c for c in drop if c in base.columns])
    newcols = [c for c in base.columns if c not in ["inst", "sid", "course_id"]]
    return base, newcols


def znorm(df, cols):
    parts = []
    for cid, gg in df.groupby("course_id"):
        gg = gg.copy()
        z = {}
        for c in cols:
            v = gg[c]; sd = v.std()
            z[f"{c}_z"] = (v - v.mean()) / sd if sd and sd > 0 else 0.0
        parts.append(pd.concat([gg, pd.DataFrame(z, index=gg.index)], axis=1))
    return pd.concat(parts).sort_index()


def eval_set(df, feats, courses, mix=None):
    d = df[df.course_id.isin(courses)]
    if mix:
        d = d[d.inst == mix]
    d = d.reset_index(drop=True)
    aucs = []
    for s in SEEDS:
        oof, y, g, _ = T.oof_predict(d, kind="cat", N=40, seed=s, features=feats)
        a = T.pooled_auc(y, oof)
        if a is not None:
            aucs.append(a)
    return round(float(np.mean(aucs)), 4)


def probe_drop(df, cols):
    """wk8-only invariance drop for the augmented set (probe <=0.75)."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    def probe(cs):
        X = df[cs].values; y = (df.inst.values == "UA").astype(int); g = df.course_id.values
        cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
        p = cross_val_predict(HistGradientBoostingClassifier(random_state=42), X, y,
                              cv=list(cv.split(X, y, g)), method="predict_proba")[:, 1]
        return float(roc_auc_score(y, p))
    cur = list(cols); dropped = []
    while probe(cur) > 0.75 and len(cur) > 15:
        rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(df[cur].values, (df.inst.values == "UA").astype(int))
        imp = sorted(zip(cur, rf.feature_importances_), key=lambda t: -t[1])
        k = 6 if probe(cur) > 0.85 else 2
        for c, _ in imp[:k]:
            cur.remove(c); dropped.append(c)
    return cur, dropped, round(probe(cur), 4)


def main():
    t0 = time.time()
    ev, starts = load_events_ext(CUTOFF)
    print(f"[Q6] events wk8: {len(ev)} rows", flush=True)
    newf, newcols = new_features(ev)
    print(f"[Q6] computed {len(newcols)} new features", flush=True)

    base8 = T.load_week("8")
    base_cols = json.loads((POOL / "feature_schema.json").read_text())["base_features"]
    # merge new onto universe, fillna 0
    uni = base8[["inst", "sid", "course_id", "y"] + base_cols].copy()
    m = uni.merge(newf, on=["inst", "sid", "course_id"], how="left")
    m[newcols] = m[newcols].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # znorm base+new together
    allcols = base_cols + newcols
    mz = znorm(m, allcols)
    base_z = [f"{c}_z" for c in base_cols]
    new_z = [f"{c}_z" for c in newcols]
    aug_z = base_z + new_z

    out = {"cutoff": 8, "seeds": SEEDS, "n_new_features": len(newcols), "new_features": newcols,
           "results": {}}

    # PUC-only: baseline (62) vs augmented (62+new)
    puc_base = eval_set(mz, base_z, CF.PUC_COURSES, "PUC")
    puc_aug = eval_set(mz, aug_z, CF.PUC_COURSES, "PUC")
    out["results"]["PUC_only"] = {"baseline_62": puc_base, "augmented": puc_aug,
                                  "delta": round(puc_aug - puc_base, 4), "n_feat_aug": len(aug_z)}
    print(f"[Q6] PUC-only: baseline={puc_base} augmented={puc_aug} Δ={puc_aug-puc_base:+.4f} [{time.time()-t0:.0f}s]", flush=True)
    (POOL / "q6_ablation.json").write_text(json.dumps(out, indent=2))

    # UA-only (R2 courses): baseline vs augmented
    r2_ua = [c for c in T.R2_EXPECTED if c in CF.UA_COURSES]
    ua_base = eval_set(mz, base_z, r2_ua, "UA")
    ua_aug = eval_set(mz, aug_z, r2_ua, "UA")
    out["results"]["R2_UA_only"] = {"baseline_62": ua_base, "augmented": ua_aug, "delta": round(ua_aug - ua_base, 4)}
    print(f"[Q6] R2-UA-only: baseline={ua_base} augmented={ua_aug} Δ={ua_aug-ua_base:+.4f} [{time.time()-t0:.0f}s]", flush=True)
    (POOL / "q6_ablation.json").write_text(json.dumps(out, indent=2))

    # R2-pooled: baseline (23 invariant) vs augmented (probe-dropped 62+new)
    r2 = mz[mz.course_id.isin(T.R2_EXPECTED)].reset_index(drop=True)
    kept, dropped, probe_auc = probe_drop(r2, aug_z)
    r2_base_feats = [c.replace("_znorm", "_z") for c in T.MODEL_FEATURES]  # match this script's _z suffix
    r2_base = eval_set(mz, r2_base_feats, T.R2_EXPECTED)
    r2_aug = eval_set(mz, kept, T.R2_EXPECTED)
    out["results"]["R2_pooled"] = {"baseline_23invariant": r2_base, "augmented": r2_aug,
                                   "delta": round(r2_aug - r2_base, 4),
                                   "n_feat_aug_after_probe": len(kept), "probe_auc_after": probe_auc,
                                   "n_dropped_leakers": len(dropped)}
    print(f"[Q6] R2-pooled: baseline(23)={r2_base} augmented({len(kept)}feat,probe={probe_auc})={r2_aug} "
          f"Δ={r2_aug-r2_base:+.4f} [{time.time()-t0:.0f}s]", flush=True)
    out["verdict"] = ("ROI positive → full rebuild warranted" if (puc_aug - puc_base > 0.01 or r2_aug - r2_base > 0.01)
                      else "marginal — reassess")
    (POOL / "q6_ablation.json").write_text(json.dumps(out, indent=2))
    print(f"[Q6] VERDICT: {out['verdict']}. wrote q6_ablation.json [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
