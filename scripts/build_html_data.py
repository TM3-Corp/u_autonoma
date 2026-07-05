#!/usr/bin/env python3
"""H3 (Tier-2B) — build the regenerated window.DATA payload + number map for
tm3-diagnostico.html, entirely from persisted OOF vectors and confirmatory JSONs.

Design (recorded in html_number_map.json):
  - The page's institution toggle is repurposed into a CV-SCHEME toggle:
      LOCO ("cursos nunca vistos")  -> oof_calibrated_week_*.parquet  (H1)
      Estratificada ("cursos conocidos, alumnos nuevos") -> oof_stratified_week_*.parquet (H2)
    Both are PUC, calibrated CatBoost (production artifact), real OOF.
  - UA appears only as DROP-A summary (segunda institucion), from ua_confirmatory.json.

Emits:
  tier2_push/html_window_data.json  — the new window.DATA object (drop-in)
  tier2_push/html_number_map.json   — inventory + best-defensible table + provenance

Every metric traces to a source path. Run AFTER H1 + H2.
Run: .venv-tier1/bin/python scripts/build_html_data.py
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

REPO = Path(__file__).resolve().parents[1]
T2 = REPO / "data/puc/sota_results/tier2_push"
T1 = REPO / "data/puc/sota_results/tier1_clean"
LOCO_CI = T2 / "confirmatory_calibrated_ci.json"
STRAT = T2 / "stratified_nested_results.json"
UA = T2 / "ua_confirmatory.json"
XGB = T1 / "nested_cv_results.json"
OUT_DATA = T2 / "html_window_data.json"
OUT_MAP = T2 / "html_number_map.json"
WEEKS = ["2", "4", "6", "8", "full"]
WLABEL = {"2": "Sem 2", "4": "Sem 4", "6": "Sem 6", "8": "Sem 8", "full": "Semestre"}
WNUM = {"2": 2, "4": 4, "6": 6, "8": 8, "full": 14}


def oof(scheme, wk):
    fn = f"oof_{'calibrated' if scheme=='loco' else 'stratified'}_week_{wk}.parquet"
    d = pd.read_parquet(T2 / fn)
    return d


def real_ops(y, p):
    """Real operating points (tp/fp/tn/fn) at a spread of thresholds, deduped by fpr."""
    P = int(y.sum()); N = int((1 - y).sum())
    # thresholds spanning the useful operating range
    cand = sorted(set([round(float(np.quantile(p, q)), 3)
                       for q in [0.55, 0.65, 0.75, 0.82, 0.88, 0.92, 0.95, 0.97]]))
    ops, seen_fpr = [], []
    for t in cand:
        pred = p >= t
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        tn = N - fp; fn = P - tp
        if tp == 0:
            continue
        fpr = fp / N if N else 0
        if any(abs(fpr - f) < 0.02 for f in seen_fpr):
            continue
        seen_fpr.append(fpr)
        ops.append({"t": round(float(t), 2), "tp": tp, "fp": fp, "tn": tn, "fn": fn})
    return ops


def roc_array(y, p, maxpts=60):
    fpr, tpr, thr = roc_curve(y, p)
    # subsample to <= maxpts, always keep endpoints
    if len(fpr) > maxpts:
        idx = np.unique(np.linspace(0, len(fpr) - 1, maxpts).astype(int))
        fpr, tpr, thr = fpr[idx], tpr[idx], thr[idx]
    out = []
    for f, t, th in zip(fpr, tpr, thr):
        th = float(th) if np.isfinite(th) else 1.0
        out.append([round(float(f), 4), round(float(t), 4), round(min(th, 1.0), 4)])
    return out


def per_course_loco():
    """Per-course held-out performance from LOCO wk8 OOF (grouped CV => held out)."""
    d = oof("loco", "8")
    rows = []
    for cid, g in d.groupby("course_id"):
        yv = g.y.values
        nfail = int(yv.sum())
        auc = float(roc_auc_score(yv, g.p.values)) if (yv.sum() > 0 and yv.sum() < len(yv)) else None
        rows.append({"cid": str(cid), "n": int(len(g)), "nfail": nfail, "auc": auc})
    # keep courses with >=1 fail and computable AUC, sort by AUC
    rows = [r for r in rows if r["auc"] is not None]
    rows.sort(key=lambda r: r["auc"])
    return rows


def scheme_weeks(scheme, src):
    weeks = []
    for wk in WEEKS:
        d = oof(scheme, wk)
        y = d.y.values.astype(int); p = d.p.values
        w = src["weeks"][wk]
        weeks.append({
            "label": WLABEL[wk], "week": WNUM[wk], "key": wk,
            "model": "CatBoost calibrado",
            "auc": round(float(w["roc_auc_calibrated"]), 4),
            "auc_ci": [round(x, 3) for x in w["roc_auc_calibrated_ci95"]],
            "pr_auc": round(float(w["pr_auc_calibrated"]), 4),
            "brier": round(float(w["brier_calibrated"]), 4),
            "ece": round(float(w["ece_calibrated"]), 4),
            "prevalence": round(float(y.mean()), 4),
            "n_samples": int(len(y)),
            "P": int(y.sum()), "N": int((1 - y).sum()),
            "capacity": {k: round(float(v), 4) for k, v in w["capacity_curve_calibrated"].items()},
            "real_ops": real_ops(y, p),
            "roc": roc_array(y, p),
        })
    return weeks


def main():
    loco = json.loads(LOCO_CI.read_text())
    strat = json.loads(STRAT.read_text())
    ua = json.loads(UA.read_text())
    xgb = json.loads(XGB.read_text())

    puc_loco = scheme_weeks("loco", loco)
    puc_strat = scheme_weeks("strat", strat)
    pc = per_course_loco()

    # UA DROP-A summary (segunda institucion) — calibrated, both schemes, DROP-A only
    ua_drop = ua["arms"]["DROP_A"]["weeks"]
    ua_summary = []
    for wk in ["2", "4", "8", "full"]:
        s = ua_drop[wk]["strat"]; l = ua_drop[wk]["loco"]
        ua_summary.append({
            "label": WLABEL[wk], "key": wk,
            "strat_auc": round(float(s["roc_auc_calibrated"]), 4),
            "strat_auc_raw": round(float(s["roc_auc_raw_bagged"]), 4),
            "strat_ci": [round(x, 3) for x in s["roc_auc_raw_ci95"]],
            "loco_auc": round(float(l["roc_auc_calibrated"]), 4),
            "loco_auc_raw": round(float(l["roc_auc_raw_bagged"]), 4),
            "prevalence": round(float(s["prevalence"]), 4),
            "n": int(s["n_eval"]),
        })

    # best-defensible cross-model LOCO row (per-week max of calibrated CatBoost vs tuned XGB nested)
    best_loco = []
    for wk in WEEKS:
        cat = float(loco["weeks"][wk]["roc_auc_calibrated"])
        xg = float(xgb["weeks"][wk]["nested"]["roc_auc"])
        if xg > cat:
            best_loco.append({"week": wk, "auc": round(xg, 4), "model": "XGBoost tuned (nested)",
                              "source": "tier1_clean/nested_cv_results.json"})
        else:
            best_loco.append({"week": wk, "auc": round(cat, 4), "model": "CatBoost calibrado",
                              "source": "tier2_push/confirmatory_calibrated_ci.json"})

    DATA = {
        "_provenance": "Regenerado desde OOF persistido (H1/H2) + ua_confirmatory (H3). "
                       "PUC: CatBoost calibrado (Platt), CV anidada, top-40 por fold, IC bootstrap 2000. "
                       "LOCO=cursos nunca vistos; estratificada=cursos conocidos, alumnos nuevos.",
        "loco": {"label": "Cursos nunca vistos", "sublabel": "validación leave-course-out (LOCO)",
                 "weeks": puc_loco},
        "strat": {"label": "Cursos conocidos", "sublabel": "alumnos nuevos (estratificada)",
                  "weeks": puc_strat},
        "puc_meta": {"students": 560, "courses": 7, "pairs": 560, "prevalence": 0.0732,
                     "events": "1.77M", "features": "~280"},
        "per_course_loco": {"model": "CatBoost calibrado (Sem 8, LOCO)", "courses": pc,
                            "loco_pooled_auc": round(float(loco["weeks"]["8"]["roc_auc_calibrated"]), 4)},
        "ua_dropA": {"label": "U. Autónoma · DROP-A (segunda institución)", "n": 322,
                     "prevalence": 0.3043,
                     "provenance": "arm DROP-A (n=322): descarta 51 active-zeros LTI; CatBoost calibrado, "
                                   "nested strat+LOCO, seed 42.",
                     "weeks": ua_summary},
        "best_defensible_loco": best_loco,
    }
    OUT_DATA.write_text(json.dumps(DATA, indent=2, ensure_ascii=False))

    # ---- number map / inventory ----
    fmt = lambda v: f"{v:.2f}"
    best_table = {}
    for wk in WEEKS:
        best_table[wk] = {
            "loco_cursos_nunca_vistos": {
                "auc": fmt(loco["weeks"][wk]["roc_auc_calibrated"]),
                "ci95": [round(x, 2) for x in loco["weeks"][wk]["roc_auc_calibrated_ci95"]],
                "source": "tier2_push/confirmatory_calibrated_ci.json",
            },
            "estratificada_cursos_conocidos": {
                "auc": fmt(strat["weeks"][wk]["roc_auc_calibrated"]),
                "ci95": [round(x, 2) for x in strat["weeks"][wk]["roc_auc_calibrated_ci95"]],
                "source": "tier2_push/stratified_nested_results.json",
            },
            "best_per_week_loco_cross_model": best_loco[WEEKS.index(wk)],
        }

    number_map = {
        "design_decision": "Institution toggle repurposed as CV-scheme toggle (LOCO vs estratificada), "
                           "both PUC real OOF. UA DROP-A moved to technical annex as segunda institución. "
                           "Every interactive metric traces to a persisted OOF vector or confirmatory JSON.",
        "forbidden_removed": [
            "0.903 / 0.9033 / 0.90 / 0.89 (UA KEEP strat header family)",
            "ua_best 0.8605 / 0.86 hold-out (UA KEEP)",
            "UA without_assessment full 0.8485 & all ua_weeks KEEP AUCs (0.7428..0.9033)",
            "UA LOCO 0.745/0.7454 (KEEP)",
            "UA per_course_auc KEEP (0.692..0.944)",
            "PUC old non-nested benchmark AUCs (0.8308/0.8716/0.8632/0.8632/0.8537) as headline",
            "PUC best_models 0.872/0.863/0.88 (old non-nested)",
        ],
        "best_defensible_table": best_table,
        "ua_dropA_summary": ua_summary,
        "per_course_loco": pc,
        "sources_exist": {
            "confirmatory_calibrated_ci.json": LOCO_CI.exists(),
            "stratified_nested_results.json": STRAT.exists(),
            "ua_confirmatory.json": UA.exists(),
            "nested_cv_results.json": XGB.exists(),
        },
    }
    OUT_MAP.write_text(json.dumps(number_map, indent=2, ensure_ascii=False))

    print("[H3] wrote", OUT_DATA.name, "and", OUT_MAP.name)
    print("[H3] LOCO cal AUC:", {w: puc_loco[i]["auc"] for i, w in enumerate(WEEKS)})
    print("[H3] STRAT cal AUC:", {w: puc_strat[i]["auc"] for i, w in enumerate(WEEKS)})
    print("[H3] best-per-week LOCO:", [(b["week"], b["auc"], b["model"].split()[0]) for b in best_loco])
    print("[H3] UA DROP-A strat:", {u["key"]: u["strat_auc"] for u in ua_summary})
    print("[H3] per-course LOCO n=", len(pc), "AUC range",
          round(min(r["auc"] for r in pc), 3), "-", round(max(r["auc"] for r in pc), 3))


if __name__ == "__main__":
    main()
