#!/usr/bin/env python3
"""Q1 — full metric suite for the top sets + per-course AUC at EVERY week
{2,4,6,8,full} (the dashboard was week-8 only). Reference config CatBoost Balanced
top-40 LOCO seed 42; F1/precision/recall/accuracy at the max-F1 point.

Output: tier3_pooled/topsets_by_week.json
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                             accuracy_score, precision_recall_curve)
import tier3_common as T

POOL = Path(T.POOL)
WEEKS = ["2", "4", "6", "8", "full"]

SETS = [
    ("R3 max-map peak — 3", [84936, 54503, 55010]),
    ("R3 seed — 2", [84936, 54503]),
    ("R3 +88381 — 4", [84936, 54503, 55010, 88381]),
    ("R3 +55183 — 5", [84936, 54503, 55010, 88381, 55183]),
    ("PUC only — 7", T.PUC_COURSES),
    ("R2 UA-only — 8", [c for c in T.R2_EXPECTED if c in T.UA_COURSES]),
    ("R1 pooled — 13", T.R1_EXPECTED),
    ("All 17 (R0)", None),
    ("R2-pooled — headline", T.R2_EXPECTED),
    ("UA only — 10", T.UA_COURSES),
]


def met(y, p):
    m = ~np.isnan(p); y, p = y[m], p[m]
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    b = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    t = float(thr[b]) if len(thr) else 0.5
    yh = (p >= t).astype(int)
    k = max(1, int(np.ceil(0.20 * len(y))))
    top = np.argsort(p)[::-1][:k]
    return {"roc": round(float(roc_auc_score(y, p)), 4), "f1": round(float(f1_score(y, yh, zero_division=0)), 4),
            "prec": round(float(precision_score(y, yh, zero_division=0)), 4),
            "rec": round(float(recall_score(y, yh, zero_division=0)), 4),
            "acc": round(float(accuracy_score(y, yh)), 4),
            "r20": round(float(y[top].sum() / max(y.sum(), 1)), 4)}


def main():
    out = {"weeks": WEEKS, "sets": {}, "per_course_auc": {}}
    dfw = {w: T.load_week(w) for w in WEEKS}
    T.assert_rules(dfw["8"])

    # per-course AUC per week (single R0 LOCO per week)
    for w in WEEKS:
        oof, y, g, _ = T.oof_predict(dfw[w].reset_index(drop=True), kind="cat", N=40, seed=42)
        out["per_course_auc"][w] = T.per_course_auc(dfw[w].reset_index(drop=True), y, oof)

    for name, cids in SETS:
        out["sets"][name] = {}
        for w in WEEKS:
            d = dfw[w] if cids is None else dfw[w][dfw[w].course_id.isin(cids)]
            d = d.reset_index(drop=True)
            oof, y, g, _ = T.oof_predict(d, kind="cat", N=40, seed=42)
            out["sets"][name][w] = met(y, oof)
        r = out["sets"][name]
        print(f"[Q1] {name}: " + " ".join(f"wk{w}={r[w]['roc']}" for w in WEEKS), flush=True)
        (POOL / "topsets_by_week.json").write_text(json.dumps(out, indent=2))

    (POOL / "topsets_by_week.json").write_text(json.dumps(out, indent=2))
    print("[Q1] wrote topsets_by_week.json", flush=True)


if __name__ == "__main__":
    main()
