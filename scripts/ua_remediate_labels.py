#!/usr/bin/env python3
"""T4 — UA label remediation A+ (independent).

Remediation A+ = drop the 51 LMS-active zero-score enrollments (>=20 views,
final_score==0, external-LTI-gradebook artifact) AND all of course 86676
(partial gradebook). Expected clean set: n=286, fails(<57)=73, prevalence 25.5%.

Remediation is a *relabeling*: page-view features are label-independent, so we
reuse the existing enriched features (data/enriched_features/*) unchanged and
re-run the weekly models under identical eval code for OLD (373) vs NEW (286)
arms, isolating the remediation effect. Metrics reported at the max-F1 operating
point, under both LOCO (StratifiedGroupKFold by course) and Stratified 5-fold.

Outputs:
  data/ua_remediated/student_enrollments_clean.csv
  data/ua_remediated/ua_clean_results.json
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             precision_recall_curve)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

import train_time_limited_model as T  # reuse load_features / feature machinery

REPO = Path(__file__).resolve().parents[1]
PV = REPO / "data/page_views/categorized_page_views.parquet"
ENROLL = REPO / "data/page_views/student_enrollments.csv"
OUT_DIR = REPO / "data/ua_remediated"
CLEAN_CSV = OUT_DIR / "student_enrollments_clean.csv"
OUT_JSON = OUT_DIR / "ua_clean_results.json"

MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
DROP_COURSE = 86676
ACTIVE_VIEW_MIN = 20
FAIL_THRESHOLD = 57      # unify label to <57 everywhere (was inconsistently >=60 in graph feature)
CUTOFFS = [2, 4, 6, 8, "full"]
RANDOM_STATE = 42


def normalize_user_id(uid):
    return uid % 10_000_000_000 if uid > 10_000_000_000 else uid


def compute_active_zero_set():
    """(user_id, course_id) pairs: final_score==0 AND >=20 page views."""
    pv = pd.read_parquet(PV, columns=["user_id", "course_id"])
    pv = pv[pv["course_id"].isin(MODEL_COURSES)].copy()
    pv["user_id"] = pv["user_id"].map(normalize_user_id)
    pv["course_id"] = pv["course_id"].astype(int)
    views = pv.groupby(["user_id", "course_id"]).size().rename("n_views").reset_index()

    enr = pd.read_csv(ENROLL)
    enr = enr.merge(views, on=["user_id", "course_id"], how="left")
    enr["n_views"] = enr["n_views"].fillna(0).astype(int)
    active_zero = enr[(enr["final_score"] == 0.0) & (enr["n_views"] >= ACTIVE_VIEW_MIN)]
    pairs = set(map(tuple, active_zero[["user_id", "course_id"]].values.tolist()))
    return enr, pairs


def build_clean_enrollments():
    enr, active_zero = compute_active_zero_set()
    base = pd.read_csv(ENROLL)
    keep = base[
        (base["course_id"] != DROP_COURSE)
        & (~base.apply(lambda r: (r["user_id"], r["course_id"]) in active_zero, axis=1))
    ].copy()
    keep["failed"] = (keep["final_score"] < FAIL_THRESHOLD).astype(int)
    meta = {
        "n_active_zero": len(active_zero),
        "n_input": len(base),
        "n_dropped_course_86676": int((base["course_id"] == DROP_COURSE).sum()),
        "n_clean": len(keep),
        "fails_clean": int(keep["failed"].sum()),
        "prevalence_clean": round(float(keep["failed"].mean()), 4),
        "courses_remaining": sorted(keep["course_id"].unique().tolist()),
    }
    return keep, meta


def max_f1_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    # precision_recall_curve returns thr of len n-1; align to f1[:-1]
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best]) if len(thr) else 0.5


def evaluate(X, y, groups, cv_kind):
    if cv_kind == "loco":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splitter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        splitter = cv.split(X, y)
    model = XGBClassifier(**T.XGBOOST_PARAMS)
    p = cross_val_predict(model, X, y, cv=list(splitter), method="predict_proba")[:, 1]
    thr = max_f1_threshold(y, p)
    yhat = (p >= thr).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "threshold_maxf1": round(thr, 4),
        "n": int(len(y)),
        "prevalence": round(float(y.mean()), 4),
    }


def run_arm(df_enroll, cutoff, include_assessment):
    df_features = T.load_features(cutoff, include_znorm=True)
    if df_features is None:
        return None
    X, y, feats = T.prepare_data(df_features, df_enroll, include_assessment=include_assessment)
    # groups aligned to X rows: re-derive via the same inner-merge order
    merged = df_features.merge(
        df_enroll[["user_id", "course_id", "failed", "final_score"]],
        on=["user_id", "course_id"], how="inner").dropna(subset=["failed"])
    groups = merged["course_id"].values
    out = {"n_samples": int(len(X)), "n_features": int(len(feats)),
           "loco": evaluate(X, y, groups, "loco"),
           "stratified": evaluate(X, y, groups, "stratified")}
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    clean, meta = build_clean_enrollments()
    clean.to_csv(CLEAN_CSV, index=False)
    print("[T4] clean enrollment meta:", json.dumps(meta), flush=True)

    # verifier assertions
    assert meta["n_clean"] == 286, f"expected n=286, got {meta['n_clean']}"
    assert meta["fails_clean"] == 73, f"expected fails=73, got {meta['fails_clean']}"
    assert abs(meta["prevalence_clean"] - 0.2552) <= 0.001, f"prevalence {meta['prevalence_clean']}"
    assert meta["n_active_zero"] == 51, f"expected 51 active-zero, got {meta['n_active_zero']}"
    print("[T4] VERIFIER PASS: n=286, fails=73, prevalence=0.2552, active_zero=51", flush=True)

    old_enroll = pd.read_csv(ENROLL)
    old_enroll["failed"] = (old_enroll["final_score"] < FAIL_THRESHOLD).astype(int)

    results = {"remediation_meta": meta,
               "threshold_note": "Unified fail label to final_score<57 in both arms "
                                  "(the graph-feature passing set used >=60 in the old code; "
                                  "inconsistency flagged and resolved to 57).",
               "weeks": {}}
    for cutoff in CUTOFFS:
        wk = str(cutoff)
        results["weeks"][wk] = {}
        for inc in (True, False):
            key = "with_assessment" if inc else "without_assessment"
            old = run_arm(old_enroll, cutoff, inc)
            new = run_arm(clean, cutoff, inc)
            results["weeks"][wk][key] = {"old": old, "new_Aplus": new}
            if old and new:
                print(f"[T4] wk{wk} {key}: OLD loco AUC={old['loco']['roc_auc']} "
                      f"F1={old['loco']['f1']} (n={old['n_samples']}) | "
                      f"NEW loco AUC={new['loco']['roc_auc']} F1={new['loco']['f1']} "
                      f"(n={new['n_samples']})", flush=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"[T4] wrote {CLEAN_CSV} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
