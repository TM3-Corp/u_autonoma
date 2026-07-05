#!/usr/bin/env python3
"""Verification: isolate what drives the UA old->A+ drop.

Label sets (same features/CV each): OLD(373) / drop active-zeros only(322) /
drop course 86676 only(333) / A+ both(286). ROC-AUC (threshold-independent),
weeks 8 & full, with AND without assessment features, LOCO + stratified.
"""
import numpy as np, pandas as pd, warnings
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import train_time_limited_model as T
warnings.filterwarnings("ignore")

MC = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
ENROLL = T.ENROLLMENTS_FILE
PV = "../data/page_views/categorized_page_views.parquet"


def active_zero_pairs():
    pv = pd.read_parquet(PV, columns=["user_id", "course_id"])
    pv = pv[pv.course_id.isin(MC)].copy()
    pv["user_id"] = pv["user_id"].map(lambda u: u % 10_000_000_000 if u > 10_000_000_000 else u)
    pv["course_id"] = pv["course_id"].astype(int)
    v = pv.groupby(["user_id", "course_id"]).size().rename("n").reset_index()
    e = pd.read_csv(ENROLL).merge(v, on=["user_id", "course_id"], how="left")
    e["n"] = e["n"].fillna(0)
    az = e[(e.final_score == 0) & (e.n >= 20)]
    return set(map(tuple, az[["user_id", "course_id"]].values.tolist()))


def label_sets():
    base = pd.read_csv(ENROLL)
    base["failed"] = (base.final_score < 57).astype(int)
    AZ = active_zero_pairs()
    is_az = base.apply(lambda r: (r.user_id, r.course_id) in AZ, axis=1)
    return {
        "OLD(373)": base,
        "drop_activezeros(322)": base[~is_az],
        "drop_86676(333)": base[base.course_id != 86676],
        "A+(286)": base[(base.course_id != 86676) & (~is_az)],
    }


def auc(dfe, cutoff, inc, cv):
    feats = T.load_features(cutoff, include_znorm=True)
    X, y, fc = T.prepare_data(feats, dfe, include_assessment=inc)
    merged = feats.merge(dfe[["user_id", "course_id", "failed", "final_score"]],
                         on=["user_id", "course_id"], how="inner").dropna(subset=["failed"])
    g = merged["course_id"].values
    if cv == "loco":
        sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(X, y, g))
    else:
        sp = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    p = cross_val_predict(XGBClassifier(**T.XGBOOST_PARAMS), X, y, cv=sp, method="predict_proba")[:, 1]
    return round(roc_auc_score(y, p), 3), len(y), int(y.sum())


def main():
    LS = label_sets()
    for cutoff in [8, "full"]:
        for inc in [True, False]:
            tag = "WITH" if inc else "WITHOUT"
            print(f"\n=== week {cutoff} | {tag} assessment features | ROC-AUC ===")
            print(f"{'labelset':<24}{'LOCO (n,fails)':>22}{'stratified':>12}")
            for name, dfe in LS.items():
                a_l, n, f = auc(dfe, cutoff, inc, "loco")
                a_s, _, _ = auc(dfe, cutoff, inc, "strat")
                print(f"{name:<24}{f'{a_l} (n={n},f={f})':>22}{a_s:>12}")


if __name__ == "__main__":
    main()
