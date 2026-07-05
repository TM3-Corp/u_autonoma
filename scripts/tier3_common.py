#!/usr/bin/env python3
"""Tier-3 shared harness: rule membership, per-fold leak-free FS, model factory,
grouped LOCO CV, metrics. Imported by G3/G4/G5/G6.

Feature set = the institution-invariant `model_feature_cols` from G2's schema
(23 znorm features surviving the guardrail-2 probe). Grouping = course_id (globally
unique across institutions). Labels y already unified (PUC grade<4.0 ≡ UA <57).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "data/puc/sota_results/tier3_pooled"
FEAT_DIR = POOL / "features"
SCHEMA = json.loads((POOL / "feature_schema.json").read_text())
MODEL_FEATURES = SCHEMA["model_feature_cols"]
RANDOM_STATE = 42
WEEKS = ["2", "4", "6", "8", "full"]

# Frozen course inventory (characteristics only; matches TIER3_EXECUTION.md)
PUC_COURSES = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
UA_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Pre-registered rule membership (derived from characteristics; asserted below).
R1_EXPECTED = sorted([54529, 54570, 55010, 55410,
                      79875, 79913, 84941, 84944, 86020, 86676, 88381, 89099, 89390])
R2_EXPECTED = sorted([54570, 55410,
                      79875, 79913, 84941, 84944, 86020, 88381, 89099, 89390])


def load_week(wk):
    df = pd.read_parquet(FEAT_DIR / f"pooled_week_{wk}.parquet")
    return df


def course_stats(df):
    s = df.groupby(["inst", "course_id"]).agg(n=("y", "size"), fails=("y", "sum")).reset_index()
    s["prev"] = s["fails"] / s["n"]
    return s


def rule_courses(df, rule):
    """Return the set of course_ids selected by a pre-registered rule (characteristics only)."""
    s = course_stats(df)
    if rule == "R0":
        sel = s
    elif rule == "R1":
        sel = s[(s.fails >= 4) & (s.n >= 15)]
    elif rule == "R2":
        sel = s[(s.fails >= 4) & (s.n >= 15) & (s.prev >= 0.08) & (s.prev <= 0.50)]
    else:
        raise ValueError(rule)
    return sorted(sel.course_id.tolist())


def assert_rules(df):
    assert rule_courses(df, "R1") == R1_EXPECTED, "R1 membership drift"
    assert rule_courses(df, "R2") == R2_EXPECTED, "R2 membership drift"


def subset(df, rule="R0", mix="pooled"):
    """Filter df to (rule × mix). mix ∈ {pooled, PUC, UA}."""
    courses = set(rule_courses(df, rule))
    d = df[df.course_id.isin(courses)].copy()
    if mix == "PUC":
        d = d[d.inst == "PUC"]
    elif mix == "UA":
        d = d[d.inst == "UA"]
    return d.reset_index(drop=True)


# ── models ──
def make_catboost(y, seed, params=None):
    from catboost import CatBoostClassifier
    p = dict(params or {})
    p.update(auto_class_weights="Balanced", random_seed=seed,
             verbose=False, allow_writing_files=False)
    return CatBoostClassifier(**p)


def make_xgb(y, seed, params=None):
    from xgboost import XGBClassifier
    spw = float((len(y) - y.sum()) / max(y.sum(), 1))
    base = dict(n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
                scale_pos_weight=spw, eval_metric="logloss", verbosity=0, random_state=seed)
    base.update(params or {})
    return XGBClassifier(**base)


def make_model(kind, y, seed, params=None):
    return make_catboost(y, seed, params) if kind in ("cat", "catboost") else make_xgb(y, seed, params)


# ── per-fold leak-free feature ranking (model-agnostic, deterministic) ──
def rank_features(Xtr, ytr, seed=RANDOM_STATE):
    """ExtraTrees importance ranking on the training fold only (leak-free)."""
    from sklearn.ensemble import ExtraTreesClassifier
    m = ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    m.fit(Xtr, ytr)
    order = np.argsort(m.feature_importances_)[::-1]
    return [Xtr.columns[i] for i in order]


def corr_prefilter(X, thresh=0.95):
    """Drop features with |corr|>thresh (keep the earlier column). For the 'full' N config."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop = [c for c in upper.columns if any(upper[c] > thresh)]
    return [c for c in X.columns if c not in drop]


# ── grouped LOCO CV ──
def loco_splits(X, y, g, seed=RANDOM_STATE):
    from sklearn.model_selection import StratifiedGroupKFold
    ng = len(np.unique(g))
    n_splits = min(5, ng)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y, g)), n_splits


def oof_predict(df, kind="cat", N=40, seed=RANDOM_STATE, features=None, n_seeds_bag=1,
                corr_full=False, cat_params_fixed=None):
    """Return OOF probabilities (grouped LOCO), per-fold selected feature counts.
    Feature selection per fold: rank on train, slice top-N (or corr-prefilter for full).
    n_seeds_bag>1 → average probs over model seeds {seed..seed+n_seeds_bag-1}."""
    feats_all = features or MODEL_FEATURES
    X = df[feats_all].reset_index(drop=True)
    y = df["y"].to_numpy().astype(int)
    g = df["course_id"].to_numpy()
    folds, n_splits = loco_splits(X, y, g, seed)
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if y[tr].sum() < 2:
            continue
        if corr_full:
            sel = corr_prefilter(X.iloc[tr])
        else:
            ranked = rank_features(X.iloc[tr], y[tr], seed)
            sel = ranked[:N] if len(ranked) >= N else ranked
        Xtr, Xte = X.iloc[tr][sel].values, X.iloc[te][sel].values
        seeds = range(seed, seed + n_seeds_bag)
        preds = []
        for s in seeds:
            m = make_model(kind, y[tr], s, cat_params_fixed if kind in ("cat", "catboost") else None)
            m.fit(Xtr, y[tr])
            preds.append(m.predict_proba(Xte)[:, 1])
        oof[te] = np.mean(preds, axis=0)
    return oof, y, g, n_splits


# ── metrics ──
def pooled_auc(y, p):
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(p)
    if m.sum() == 0 or y[m].min() == y[m].max():
        return None
    return float(roc_auc_score(y[m], p[m]))


def per_course_auc(df, y, p):
    from sklearn.metrics import roc_auc_score
    out = {}
    cid = df["course_id"].to_numpy()
    for c in np.unique(cid):
        m = (cid == c) & ~np.isnan(p)
        yy = y[m]
        if yy.sum() < 2 or (len(yy) - yy.sum()) < 1 or yy.min() == yy.max():
            out[int(c)] = None
        else:
            out[int(c)] = round(float(roc_auc_score(yy, p[m])), 4)
    return out


def recall_at(y, p, rate):
    m = ~np.isnan(p)
    yy, pp = y[m], p[m]
    if yy.sum() == 0:
        return None
    k = max(1, int(np.ceil(rate * len(yy))))
    idx = np.argsort(pp)[::-1][:k]
    return round(float(yy[idx].sum() / yy.sum()), 4)


def mean_per_course_auc(pc):
    vals = [v for v in pc.values() if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None
