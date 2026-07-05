#!/usr/bin/env python3
"""Follow-up (internal, quotable:false) — full metric suite for the top course
sets + per-course characteristics, for Paul's exploration HTML.

For a panel of course sets (incl. the R3 max-map subsets, which are selected BY
AUC and therefore internal/non-quotable), run LOCO OOF (reference config:
CatBoost Balanced, top-40/fold, seed 42, week 8) and report ROC-AUC + metrics at
the max-F1 operating point (F1, precision, recall, accuracy) + recall@20%.

Per course: institution, total LMS activity (Σ events, full horizon), total
students (n), failure rate (prevalence), per-course LOCO AUC.

Output: tier3_pooled/topsets_metrics.json  ("quotable": false)
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, precision_recall_curve)
import tier3_common as T

POOL = Path(T.POOL)
OUT = POOL / "topsets_metrics.json"
REF_WEEK = "8"


def metrics_at_maxf1(y, p):
    m = ~np.isnan(p)
    y, p = y[m], p[m]
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    best = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    t = float(thr[best]) if len(thr) else 0.5
    yhat = (p >= t).astype(int)
    k = max(1, int(np.ceil(0.20 * len(y))))
    top = np.argsort(p)[::-1][:k]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "f1": round(float(f1_score(y, yhat, zero_division=0)), 4),
        "precision": round(float(precision_score(y, yhat, zero_division=0)), 4),
        "recall": round(float(recall_score(y, yhat, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, yhat)), 4),
        "recall_at_20pct": round(float(y[top].sum() / max(y.sum(), 1)), 4),
        "threshold_maxf1": round(t, 4),
    }


def main():
    df8 = T.load_week(REF_WEEK)
    dffull = T.load_week("full")
    T.assert_rules(df8)

    R2 = T.R2_EXPECTED
    R2_UA = [c for c in R2 if c in T.UA_COURSES]
    R1 = T.R1_EXPECTED
    ALL = sorted(df8.course_id.unique().tolist())

    sets = [
        ("R3 max-map peak — 3 courses", [84936, 54503, 55010], "R3"),
        ("R3 seed — 2 courses", [84936, 54503], "R3"),
        ("R3 +88381 — 4 courses", [84936, 54503, 55010, 88381], "R3"),
        ("R3 +55183 — 5 courses", [84936, 54503, 55010, 88381, 55183], "R3"),
        ("R2-pooled (balanced · HEADLINE)", R2, "R2"),
        ("R2 UA-only (8)", R2_UA, "R2"),
        ("R0 PUC-only (7)", T.PUC_COURSES, "R0"),
        ("R0 UA-only (10)", T.UA_COURSES, "R0"),
        ("R1 pooled (13)", R1, "R1"),
        ("R0 all courses (17)", ALL, "R0"),
    ]

    # per-course characteristics
    total_activity = dffull.groupby("course_id")["total_events"].sum()
    stats = T.course_stats(df8).set_index("course_id")
    # per-course AUC from a single R0 LOCO run (reference config)
    r0 = df8[df8.course_id.isin(ALL)].reset_index(drop=True)
    oof_r0, y_r0, g_r0, _ = T.oof_predict(r0, kind="cat", N=40, seed=T.RANDOM_STATE)
    pc_auc = T.per_course_auc(r0, y_r0, oof_r0)

    courses = []
    for cid in ALL:
        courses.append({
            "course_id": int(cid),
            "institution": stats.loc[cid, "inst"],
            "total_lms_activity": int(total_activity.loc[cid]),
            "total_students": int(stats.loc[cid, "n"]),
            "fails": int(stats.loc[cid, "fails"]),
            "failure_rate": round(float(stats.loc[cid, "prev"]), 4),
            "per_course_auc_wk8": pc_auc.get(int(cid)),
        })

    set_results = []
    for name, cids, rule in sets:
        d = df8[df8.course_id.isin(cids)].reset_index(drop=True)
        oof, y, g, nsp = T.oof_predict(d, kind="cat", N=40, seed=T.RANDOM_STATE)
        met = metrics_at_maxf1(y, oof)
        n_puc = sum(1 for c in cids if c in T.PUC_COURSES)
        n_ua = sum(1 for c in cids if c in T.UA_COURSES)
        set_results.append({
            "name": name, "rule": rule, "courses": [int(c) for c in cids],
            "n_courses": len(cids), "n_puc": n_puc, "n_ua": n_ua,
            "mixed_institution": bool(n_puc > 0 and n_ua > 0),
            "n_students": int(len(d)), "n_fails": int(d.y.sum()),
            "failure_rate": round(float(d.y.mean()), 4),
            "total_lms_activity": int(dffull[dffull.course_id.isin(cids)]["total_events"].sum()),
            **met,
        })
        print(f"[TS] {name}: ROC-AUC={met['roc_auc']} F1={met['f1']} "
              f"P={met['precision']} R={met['recall']} Acc={met['accuracy']} "
              f"(PUC {n_puc}+UA {n_ua}, {int(d.y.sum())} fails)", flush=True)

    set_results.sort(key=lambda s: -s["roc_auc"])
    out = {
        "quotable": False,
        "internal_note": "Sets tagged R3 are selected BY measured AUC (max-map) — INTERNAL, "
                         "never for client/sales material. R2-pooled is the pre-registered honest "
                         "headline. High-AUC sets are dominated by low-failure-rate courses (few, "
                         "clearly-disengaged failers) — metrics are optimistic/noisy on small fail "
                         "counts. Accuracy is inflated under class imbalance.",
        "reference_config": "CatBoost Balanced, top-40/fold, LOCO, seed 42, week 8; "
                            "F1/precision/recall/accuracy at the max-F1 operating point.",
        "sets": set_results,
        "courses": courses,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[TS] wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
