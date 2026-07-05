#!/usr/bin/env python3
"""T7 — SHAP activation on the winning clean-data config (weeks 4 & 8).

Best config from T3/T5/T6 = XGBoost production (calibrated XGB + spw + top-40).
For TreeSHAP we fit the UNCALIBRATED booster on full clean data with the same
top-40 leak-free feature set and spw=neg/pos (calibration wraps the booster and
is not TreeSHAP-friendly; SHAP explains the underlying risk model).

Exports per week w in {4,8}:
  tier1_clean/shap_week{w}_summary.png
  tier1_clean/shap_week{w}_global_importance.json   (top-20 mean|SHAP|)
  tier1_clean/shap_week{w}_per_student.csv          (student_id, course_id,
      risk_score, top-3 signed factors w/ plain-language names)   [560 rows]
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from xgboost import XGBClassifier

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
FEAT_DIR = REPO / "data/puc/sota_results/tier1_clean/features"
OUT_DIR = REPO / "data/puc/sota_results/tier1_clean"
WEEKS = [4, 8]
TOPK = 40
RS = B.RANDOM_STATE


def humanize(feat):
    z = feat.endswith("_znorm")
    base = feat[:-6] if z else feat
    words = {
        "views": "vistas", "total": "total", "session": "sesión", "sessions": "sesiones",
        "regularity": "regularidad", "hour": "hora", "entropy": "entropía",
        "weekend": "fin de semana", "pct": "%", "day": "día", "unique": "únicos/as",
        "morning": "mañana", "afternoon": "tarde", "evening": "noche", "night": "madrugada",
        "gap": "brecha", "proactivity": "proactividad", "decay": "decaimiento",
        "momentum": "momentum", "inactivity": "inactividad", "weekly": "semanal",
        "trend": "tendencia", "quiz": "quiz", "assignment": "tarea", "grade": "nota",
        "files": "archivos", "discussions": "foros", "modules": "módulos",
        "pages": "páginas", "announcements": "anuncios", "first": "primer",
        "access": "acceso", "week": "semana", "count": "conteo", "ratio": "ratio",
        "time": "tiempo", "active": "activo", "density": "densidad", "coverage": "cobertura",
        "std": "desv", "mean": "media", "max": "máx", "min": "mín", "duration": "duración",
    }
    pretty = " ".join(words.get(w, w) for w in base.split("_"))
    if z:
        pretty += " (relativo al curso)"
    return pretty.strip().capitalize()


def load_week(w):
    df = pd.read_parquet(FEAT_DIR / f"week_{w}_clean.parquet")
    ids = df[["student_id", "course_id"]].reset_index(drop=True)
    y = df["_y"].to_numpy().astype(int)
    X = df.drop(columns=["student_id", "course_id", "_group", "_y"]).reset_index(drop=True)
    return X, y, ids


def run_week(w):
    X, y, ids = load_week(w)
    ranked = B.sota_feature_selection(X, pd.Series(y), return_ranked=True)
    feats = ranked[:TOPK] if len(ranked) >= TOPK else ranked
    Xs = X[feats]
    spw = float((len(y) - y.sum()) / max(y.sum(), 1))
    model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                          scale_pos_weight=spw, eval_metric="logloss", verbosity=0,
                          random_state=RS)
    model.fit(Xs.values, y)
    risk = model.predict_proba(Xs.values)[:, 1]

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs.values)
    if isinstance(sv, list):  # older API safety
        sv = sv[1]

    # summary plot
    plt.figure()
    shap.summary_plot(sv, Xs, max_display=20, show=False,
                      feature_names=[humanize(f) for f in feats])
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_week{w}_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # global importance top-20 by mean|SHAP|
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:20]
    global_imp = [{"feature": feats[i], "feature_es": humanize(feats[i]),
                   "mean_abs_shap": round(float(mean_abs[i]), 5)} for i in order]
    (OUT_DIR / f"shap_week{w}_global_importance.json").write_text(json.dumps(
        {"week": w, "n_features_used": len(feats), "top20_mean_abs_shap": global_imp}, indent=2))

    # per-student top-3 signed factors
    rows = []
    for r in range(len(Xs)):
        contrib = sv[r]
        top3 = np.argsort(np.abs(contrib))[::-1][:3]
        rec = {"student_id": int(ids.iloc[r]["student_id"]),
               "course_id": int(ids.iloc[r]["course_id"]),
               "risk_score": round(float(risk[r]), 4)}
        for k, j in enumerate(top3, 1):
            direction = "aumenta" if contrib[j] > 0 else "reduce"
            rec[f"factor{k}"] = humanize(feats[j])
            rec[f"factor{k}_effect"] = direction  # aumenta/reduce el riesgo
            rec[f"factor{k}_shap"] = round(float(contrib[j]), 4)
        rows.append(rec)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_DIR / f"shap_week{w}_per_student.csv", index=False)
    print(f"[T7] week{w}: {len(feats)} feats, per-student rows={len(df_out)}; "
          f"top global: {global_imp[0]['feature_es']} ({global_imp[0]['mean_abs_shap']})", flush=True)
    return len(df_out)


def main():
    for w in WEEKS:
        run_week(w)
    print("[T7] done", flush=True)


if __name__ == "__main__":
    main()
