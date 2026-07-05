#!/usr/bin/env python3
"""G2 (Tier-3) — Shared cross-institution feature pipeline.

ONE pipeline over both clean clickstreams producing an IDENTICAL schema per
(student, course, cutoff ∈ {2,4,6,8,full}):
  - PUC: data/puc/puc_clean_data.parquet  (7 courses, `category`)
  - UA : data/ua_clean/ua_clean_data.parquet (10 DROP-A courses, `resource_type`)

Uses ONLY signals computable identically at both institutions. Category taxonomy
mapped to 10 shared bins. Families: session (30-min gap), category counts+shares,
temporal (hour/dow local), weekly (views/sessions/trend/momentum/inactivity),
first-access timing. Base features per-course z-normed (znorm alongside raw).

Labels: PUC grade<4.0 · UA DROP-A final_score<57 (drop 51 active-zeros, keep 86676).
Institution column kept for grouping/audit, NEVER used as a feature.

Cutoff = target course start (0.05 quantile of that course's clean events) + w weeks;
full = all events. Leak-free by construction (per-course start; no future events).

Outputs:
  tier3_pooled/features/pooled_week_{w}.parquet  (one row per pair, both insts)
  tier3_pooled/feature_schema.json
  tier3_pooled/category_mapping.json
  tier3_pooled/g2_build_report.json  (row counts, unmapped share, probe, leak spot-check)

Run: .venv-tier1/bin/python scripts/common_features.py [--week W]
"""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
PUC_CLEAN = REPO / "data/puc/puc_clean_data.parquet"
UA_CLEAN = REPO / "data/ua_clean/ua_clean_data.parquet"
PUC_GRADES = REPO / "data/puc/puc_grades_clean.parquet"
UA_ENROLL = REPO / "data/page_views/student_enrollments.csv"
OUT_DIR = REPO / "data/puc/sota_results/tier3_pooled"
FEAT_DIR = OUT_DIR / "features"

PUC_COURSES = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
UA_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
UA_DROP_COURSE_NONE = None  # DROP-A keeps 86676
UA_ACTIVE_VIEW_MIN = 20
UA_FAIL_THRESHOLD = 57
PERCENTILE = 0.05
CUTOFFS = [2, 4, 6, 8, "full"]
SESSION_GAP_MIN = 30.0
RANDOM_STATE = 42

SHARED_BINS = ["files", "assignments", "quizzes", "discussions", "pages",
               "modules", "grades", "announcements", "navigation", "other"]

# Explicit, documented mapping tables (raw taxonomy → shared bin)
PUC_CAT_MAP = {
    "files": "files", "assignments": "assignments", "quizzes": "quizzes",
    "discussions": "discussions", "pages": "pages", "modules": "modules",
    "grades": "grades", "announcements": "announcements", "navigation": "navigation",
    "other": "other", "external_tools": "other",
}
UA_RT_MAP = {
    "files": "files", "assignments": "assignments", "quizzes": "quizzes",
    "discussions": "discussions", "pages": "pages", "modules": "modules",
    "grades": "grades", "announcements": "announcements", "home": "navigation",
    "other": "other", "navigation": "navigation",
}


def normalize_user_id(uid):
    return uid % 10_000_000_000 if uid > 10_000_000_000 else uid


# ── event loading (standardized frame: inst, sid, course_id, ts, hour_local, dow_local, bin) ──
def _standardize(df, inst, raw_bin_col, bin_map):
    unmapped = float((~df[raw_bin_col].isin(bin_map)).mean())
    out = pd.DataFrame({
        "inst": inst,
        "sid": df["sid"].astype("int64").to_numpy(),
        "course_id": df["course_id"].astype("int64").to_numpy(),
        "hour_local": df["hour_local"].astype("int32").to_numpy(),
        "dow_local": df["dow_local"].astype("int32").to_numpy(),
        "bin": df[raw_bin_col].map(bin_map).fillna("other").to_numpy(),
    }, index=df.index)
    out["ts"] = df["ts"]  # preserve tz-aware dtype (align on index)
    return out.reset_index(drop=True), unmapped


def load_puc_events():
    df = pd.read_parquet(PUC_CLEAN, columns=["student_id", "course_id", "created_at",
                                             "category", "hour_local", "dow_local"])
    df = df[df["course_id"].isin(PUC_COURSES)].copy()
    df = df.rename(columns={"student_id": "sid", "created_at": "ts"})
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return _standardize(df, "PUC", "category", PUC_CAT_MAP)


def load_ua_events():
    df = pd.read_parquet(UA_CLEAN, columns=["user_id", "course_id", "created_at",
                                            "resource_type", "hour_local", "dow_local"])
    df = df[df["course_id"].isin(UA_COURSES)].copy()
    df = df.rename(columns={"user_id": "sid", "created_at": "ts"})
    if not pd.api.types.is_datetime64_any_dtype(df["ts"]):
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return _standardize(df, "UA", "resource_type", UA_RT_MAP)


# ── labels ──
def puc_labels():
    g = pd.read_parquet(PUC_GRADES)
    g = g[g["course_id"].isin(PUC_COURSES)].copy()
    g = g[["student_id", "course_id", "grade"]].drop_duplicates(["student_id", "course_id"])
    g = g.rename(columns={"student_id": "sid"})
    g["y"] = (g["grade"] < 4.0).astype(int)
    g["inst"] = "PUC"
    return g[["inst", "sid", "course_id", "y"]].reset_index(drop=True)


def ua_drop_a_labels():
    """DROP-A: drop 51 active-zeros (>=20 views & final_score==0), keep 86676, <57."""
    pv = pd.read_parquet(UA_CLEAN, columns=["user_id", "course_id"])
    pv["user_id"] = pv["user_id"].astype("int64")  # already normalized in G1
    pv["course_id"] = pv["course_id"].astype("int64")
    views = pv.groupby(["user_id", "course_id"]).size().rename("n_views").reset_index()

    enr = pd.read_csv(UA_ENROLL)
    enr["user_id"] = enr["user_id"].map(normalize_user_id).astype("int64")
    enr["course_id"] = enr["course_id"].astype("int64")
    enr = enr[enr["course_id"].isin(UA_COURSES)].copy()
    enr = enr.merge(views, on=["user_id", "course_id"], how="left")
    enr["n_views"] = enr["n_views"].fillna(0).astype(int)
    active_zero = (enr["final_score"] == 0.0) & (enr["n_views"] >= UA_ACTIVE_VIEW_MIN)
    keep = enr[~active_zero].copy()
    keep["y"] = (keep["final_score"] < UA_FAIL_THRESHOLD).astype(int)
    keep = keep.rename(columns={"user_id": "sid"})
    keep["inst"] = "UA"
    meta = {"n_active_zero": int(active_zero.sum()), "n_clean": int(len(keep)),
            "fails": int(keep["y"].sum()), "prevalence": round(float(keep["y"].mean()), 4)}
    return keep[["inst", "sid", "course_id", "y"]].reset_index(drop=True), meta


# ── cutoff ──
def course_starts(events):
    return events.groupby("course_id")["ts"].quantile(PERCENTILE).to_dict()


def filter_cutoff(events, starts, cutoff):
    if cutoff == "full":
        return events
    parts = []
    for c, start in starts.items():
        bound = start + pd.Timedelta(weeks=cutoff)
        parts.append(events[(events["course_id"] == c) & (events["ts"] <= bound)])
    return pd.concat(parts, ignore_index=True) if parts else events.iloc[:0]


# ── feature families ──
def _entropy(counts):
    c = np.asarray(counts, dtype=float)
    s = c.sum()
    if s <= 0:
        return 0.0
    p = c[c > 0] / s
    return float(-(p * np.log(p)).sum())


def featurize_pair(sub, start):
    """sub: events for one (inst,sid,course), cutoff-filtered, with ts/hour/dow/bin.
    start: that course's start timestamp. Returns dict of base features."""
    sub = sub.sort_values("ts")
    ts = sub["ts"].values
    n = len(sub)
    f = {}
    f["total_events"] = float(n)

    # days since start
    dss = (sub["ts"] - start).dt.total_seconds().values / 86400.0
    dates = sub["ts"].dt.tz_convert("America/Santiago").dt.date
    day_ords = np.sort(np.unique([d.toordinal() for d in dates]))
    active_days = len(day_ords)
    f["active_days"] = float(active_days)
    f["events_per_active_day"] = float(n / active_days) if active_days else 0.0
    f["first_event_day"] = float(dss.min())
    f["last_event_day"] = float(dss.max())
    f["span_days"] = float(dss.max() - dss.min())
    f["first_event_hour_local"] = float(sub["hour_local"].iloc[0])

    # inter-event gaps (hours)
    if n >= 2:
        gaps_h = np.diff(ts).astype("timedelta64[s]").astype(float) / 3600.0
        f["mean_intergap_hours"] = float(np.mean(gaps_h))
        f["median_intergap_hours"] = float(np.median(gaps_h))
    else:
        f["mean_intergap_hours"] = 0.0
        f["median_intergap_hours"] = 0.0

    # max inactivity gap in days (between distinct active local days)
    if active_days >= 2:
        f["max_inactivity_gap_days"] = float(np.max(np.diff(day_ords)))
    else:
        f["max_inactivity_gap_days"] = 0.0

    # sessions (30-min gap)
    if n >= 1:
        gap_min = np.concatenate([[np.inf], np.diff(ts).astype("timedelta64[s]").astype(float) / 60.0])
        new_sess = gap_min > SESSION_GAP_MIN
        sess_id = np.cumsum(new_sess) - 1
        n_sess = int(sess_id.max() + 1)
        sess_minutes, sess_counts = [], []
        starts_list = []
        for s in range(n_sess):
            m = sess_id == s
            tss = ts[m]
            dur = (tss.max() - tss.min()).astype("timedelta64[s]").astype(float) / 60.0
            sess_minutes.append(dur)
            sess_counts.append(int(m.sum()))
            starts_list.append(tss.min())
        sess_minutes = np.array(sess_minutes)
        sess_counts = np.array(sess_counts)
        f["n_sessions"] = float(n_sess)
        f["total_session_minutes"] = float(sess_minutes.sum())
        f["mean_session_minutes"] = float(sess_minutes.mean())
        f["median_session_minutes"] = float(np.median(sess_minutes))
        f["max_session_minutes"] = float(sess_minutes.max())
        f["mean_events_per_session"] = float(sess_counts.mean())
        f["short_session_share"] = float((sess_counts == 1).mean())
        f["dwell_per_event"] = float(sess_minutes.sum() / n)
        if n_sess >= 2:
            inter = np.diff(np.array(starts_list)).astype("timedelta64[s]").astype(float) / 3600.0
            f["session_regularity"] = float(1.0 / (1.0 + np.std(inter)))
        else:
            f["session_regularity"] = 0.0
    else:
        for k in ["n_sessions", "total_session_minutes", "mean_session_minutes",
                  "median_session_minutes", "max_session_minutes", "mean_events_per_session",
                  "short_session_share", "dwell_per_event", "session_regularity"]:
            f[k] = 0.0

    # category counts + shares
    vc = sub["bin"].value_counts()
    for b in SHARED_BINS:
        c = float(vc.get(b, 0))
        f[f"cnt_{b}"] = c
        f[f"share_{b}"] = float(c / n) if n else 0.0

    # temporal
    h = sub["hour_local"].values
    dow = sub["dow_local"].values
    f["share_night"] = float(((h >= 0) & (h < 6)).mean())
    f["share_morning"] = float(((h >= 6) & (h < 12)).mean())
    f["share_afternoon"] = float(((h >= 12) & (h < 18)).mean())
    f["share_evening"] = float(((h >= 18) & (h < 24)).mean())
    f["weekend_share"] = float((dow >= 5).mean())
    f["prep_evening_night_share"] = f["share_evening"] + f["share_night"]
    f["distinct_hours_local"] = float(len(np.unique(h)))
    f["distinct_dows"] = float(len(np.unique(dow)))
    f["hour_entropy"] = _entropy(np.bincount(h, minlength=24))
    f["dow_entropy"] = _entropy(np.bincount(dow, minlength=7))

    # weekly
    wk = np.clip(np.floor(dss / 7.0).astype(int), 0, None)
    W = int(wk.max() + 1)
    weekly_views = np.bincount(wk, minlength=W).astype(float)
    # weekly sessions: count sessions per week by session-start week
    if n >= 1:
        sess_week = np.clip(np.floor(
            ((np.array(starts_list) - start.to_datetime64()).astype("timedelta64[s]").astype(float) / 86400.0) / 7.0
        ).astype(int), 0, None)
        weekly_sessions = np.bincount(sess_week, minlength=W).astype(float)
    else:
        weekly_sessions = np.zeros(W)
    f["mean_weekly_views"] = float(weekly_views.mean())
    f["std_weekly_views"] = float(weekly_views.std())
    f["mean_weekly_sessions"] = float(weekly_sessions.mean())
    f["n_active_weeks"] = float((weekly_views > 0).sum())
    f["inactive_weeks"] = float((weekly_views == 0).sum())
    f["active_week_ratio"] = float((weekly_views > 0).mean())
    if W >= 2:
        x = np.arange(W)
        slope = float(np.polyfit(x, weekly_views, 1)[0])
        f["trend_slope"] = slope
        f["momentum"] = float(weekly_views[-1] - weekly_views[-2])
        mu, sd = weekly_views.mean(), weekly_views.std()
        f["last_week_deviation"] = float((weekly_views[-1] - mu) / sd) if sd > 0 else 0.0
    else:
        f["trend_slope"] = 0.0
        f["momentum"] = 0.0
        f["last_week_deviation"] = 0.0
    f["first_week_events"] = float(weekly_views[0]) if W >= 1 else 0.0
    front = dss <= (f["first_event_day"] + f["last_event_day"]) / 2.0
    f["front_load_share"] = float(front.mean()) if n else 0.0

    # first-access timing to assessment bins
    assign_days = dss[sub["bin"].values == "assignments"]
    quiz_days = dss[sub["bin"].values == "quizzes"]
    f["days_to_first_assignment"] = float(assign_days.min()) if len(assign_days) else float(f["last_event_day"] + 7)
    f["days_to_first_quiz"] = float(quiz_days.min()) if len(quiz_days) else float(f["last_event_day"] + 7)
    return f


# canonical feature order (from a synthetic non-empty featurize)
BASE_FEATURES = None


def get_base_feature_order():
    global BASE_FEATURES
    if BASE_FEATURES is not None:
        return BASE_FEATURES
    start = pd.Timestamp("2025-01-01", tz="UTC")
    ts = pd.to_datetime(["2025-01-02 10:00", "2025-01-02 10:20", "2025-01-10 15:00",
                         "2025-01-20 22:00"], utc=True)
    sub = pd.DataFrame({"ts": ts, "hour_local": [7, 7, 12, 19], "dow_local": [3, 3, 4, 0],
                        "bin": ["quizzes", "assignments", "files", "modules"]})
    BASE_FEATURES = list(featurize_pair(sub, start).keys())
    return BASE_FEATURES


# ── per-course z-norm ──
def znorm_per_course(df, feature_cols):
    parts = []
    for cid, g in df.groupby("course_id"):
        g = g.copy()
        zcols = {}
        for col in feature_cols:
            v = g[col]
            sd = v.std()
            zcols[f"{col}_znorm"] = (v - v.mean()) / sd if sd and sd > 0 else 0.0
        parts.append(pd.concat([g, pd.DataFrame(zcols, index=g.index)], axis=1))
    return pd.concat(parts).sort_index()


# ── build one week ──
def build_week(cutoff, puc_ev, ua_ev, puc_starts, ua_starts, universe):
    base_order = get_base_feature_order()
    rows = []
    for ev, starts, inst in [(puc_ev, puc_starts, "PUC"), (ua_ev, ua_starts, "UA")]:
        evc = filter_cutoff(ev, starts, cutoff)
        for (cid, sid), sub in evc.groupby(["course_id", "sid"]):
            f = featurize_pair(sub, pd.Timestamp(starts[cid]))
            f["inst"] = inst
            f["sid"] = int(sid)
            f["course_id"] = int(cid)
            rows.append(f)
    feats = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["inst", "sid", "course_id"] + base_order)

    merged = universe.merge(feats, on=["inst", "sid", "course_id"], how="left")
    for col in base_order:
        if col not in merged.columns:
            merged[col] = 0.0
    merged[base_order] = merged[base_order].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    zdf = znorm_per_course(merged, base_order)
    id_cols = ["inst", "sid", "course_id", "y"]
    znorm_cols = [f"{c}_znorm" for c in base_order]
    ordered = id_cols + base_order + znorm_cols
    return zdf[ordered].reset_index(drop=True)


def institution_probe(df, cols):
    """Train inst-classifier (HGB) on `cols`; grouped-CV AUC must be ≤0.75."""
    from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = df[cols].values
    y = (df["inst"].values == "UA").astype(int)
    g = df["course_id"].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    m = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    p = cross_val_predict(m, X, y, cv=list(cv.split(X, y, g)), method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def select_invariant_features(dfs, znorm_cols):
    """Greedy backward elimination (guardrail 2): drop institution-leaking znorm
    features until EVERY week's HGB grouped-CV probe ≤0.75. Ranks leakers by
    RandomForest institution-importance on the pooled-across-weeks matrix.
    Returns (kept, dropped, before_per_week, after_per_week)."""
    from sklearn.ensemble import RandomForestClassifier
    pooled = pd.concat(list(dfs.values()), ignore_index=True)
    before = {w: round(institution_probe(dfs[w], znorm_cols), 4) for w in dfs}
    cur = list(znorm_cols)
    dropped = []
    while True:
        worst = max(institution_probe(dfs[w], cur) for w in dfs)
        if worst <= 0.75 or len(cur) <= 15:
            break
        X = pooled[cur].values
        y = (pooled["inst"].values == "UA").astype(int)
        rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        imp = sorted(zip(cur, rf.feature_importances_), key=lambda t: -t[1])
        # finer steps near the 0.75 boundary to avoid over-dropping
        k = 8 if worst > 0.90 else 4 if worst > 0.82 else 2 if worst > 0.77 else 1
        todrop = [c for c, _ in imp[:k]]
        dropped += todrop
        cur = [c for c in cur if c not in todrop]
    after = {w: round(institution_probe(dfs[w], cur), 4) for w in dfs}
    return cur, dropped, before, after


def leak_spotcheck(cutoff, puc_ev, ua_ev, puc_starts, ua_starts, feat_df):
    """Recount raw events ≤ cutoff for 3 cells per inst; match feature total_events."""
    checks = []
    for ev, starts, inst in [(puc_ev, puc_starts, "PUC"), (ua_ev, ua_starts, "UA")]:
        sub_feat = feat_df[(feat_df["inst"] == inst) & (feat_df["total_events"] > 0)]
        sample = sub_feat.sort_values("total_events", ascending=False).head(3)
        for _, r in sample.iterrows():
            cid, sid = int(r["course_id"]), int(r["sid"])
            if cutoff == "full":
                raw = ev[(ev["course_id"] == cid) & (ev["sid"] == sid)]
            else:
                bound = pd.Timestamp(starts[cid]) + pd.Timedelta(weeks=cutoff)
                raw = ev[(ev["course_id"] == cid) & (ev["sid"] == sid) & (ev["ts"] <= bound)]
            checks.append({"inst": inst, "course_id": cid, "sid": sid,
                           "feat_total_events": int(r["total_events"]),
                           "raw_recount": int(len(raw)),
                           "match": bool(int(r["total_events"]) == int(len(raw)))})
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default=None, help="only build this week (2/4/6/8/full)")
    args = ap.parse_args()
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    puc_ev, puc_unmapped = load_puc_events()
    ua_ev, ua_unmapped = load_ua_events()
    puc_starts = course_starts(puc_ev)
    ua_starts = course_starts(ua_ev)

    plab = puc_labels()
    ulab, umeta = ua_drop_a_labels()
    universe = pd.concat([plab, ulab], ignore_index=True)
    print(f"[G2] universe: PUC {len(plab)} ({plab.y.sum()} fails) + UA {len(ulab)} "
          f"({ulab.y.sum()} fails) = {len(universe)} pairs, {universe.y.sum()} fails", flush=True)
    print(f"[G2] unmapped-event share: PUC {puc_unmapped:.4f} UA {ua_unmapped:.4f}", flush=True)

    base_order = get_base_feature_order()
    znorm_cols = [f"{c}_znorm" for c in base_order]

    # mapping table
    (OUT_DIR / "category_mapping.json").write_text(json.dumps({
        "shared_bins": SHARED_BINS, "puc_category_to_bin": PUC_CAT_MAP,
        "ua_resource_type_to_bin": UA_RT_MAP,
        "unmapped_share": {"PUC": round(puc_unmapped, 4), "UA": round(ua_unmapped, 4)},
    }, indent=2))

    weeks = [args.week] if args.week else [str(w) for w in CUTOFFS]
    report = {"universe": {"puc_pairs": int(len(plab)), "puc_fails": int(plab.y.sum()),
                           "ua_pairs": int(len(ulab)), "ua_fails": int(ulab.y.sum()),
                           "ua_drop_a_meta": umeta, "total_pairs": int(len(universe))},
              "unmapped_share": {"PUC": round(puc_unmapped, 4), "UA": round(ua_unmapped, 4)},
              "n_base_features": len(base_order), "n_znorm_features": len(znorm_cols),
              "weeks": {}}

    # Build every week, persist full-feature parquets, keep in memory for the probe.
    dfs = {}
    for wk in weeks:
        cutoff = int(wk) if wk != "full" else "full"
        df = build_week(cutoff, puc_ev, ua_ev, puc_starts, ua_starts, universe)
        spot = leak_spotcheck(cutoff, puc_ev, ua_ev, puc_starts, ua_starts, df)
        df.to_parquet(FEAT_DIR / f"pooled_week_{wk}.parquet", index=False)
        dfs[wk] = df
        report["weeks"][wk] = {
            "n_rows": int(len(df)),
            "n_rows_puc": int((df["inst"] == "PUC").sum()),
            "n_rows_ua": int((df["inst"] == "UA").sum()),
            "n_fails": int(df["y"].sum()),
            "leak_spotcheck": spot,
            "leak_spotcheck_all_match": bool(all(c["match"] for c in spot)),
            "n_cols": int(df.shape[1]),
        }
        print(f"[G2] built wk{wk}: rows={len(df)} fails={int(df['y'].sum())} "
              f"leak_match={report['weeks'][wk]['leak_spotcheck_all_match']}", flush=True)

    # Guardrail 2: select an institution-invariant model feature set (drop leakers).
    if len(dfs) >= 2:
        model_cols, dropped, probe_before, probe_after = select_invariant_features(dfs, znorm_cols)
    else:  # single-week smoke build: cannot design the full drop-set; report probe only
        wk0 = next(iter(dfs))
        probe_before = {wk0: round(institution_probe(dfs[wk0], znorm_cols), 4)}
        model_cols, dropped, probe_after = znorm_cols, [], probe_before
    for wk in dfs:
        report["weeks"][wk]["institution_probe_auc_allznorm"] = probe_before.get(wk)
        report["weeks"][wk]["institution_probe_auc_model"] = probe_after.get(wk)
        report["weeks"][wk]["probe_pass"] = bool(probe_after.get(wk, 1.0) <= 0.75)
    report["institution_invariance"] = {
        "dropped_leaking_features": dropped,
        "n_dropped": len(dropped),
        "n_model_features": len(model_cols),
        "probe_auc_before_max": round(max(probe_before.values()), 4),
        "probe_auc_after_max": round(max(probe_after.values()), 4),
        "all_weeks_probe_pass": bool(all(v <= 0.75 for v in probe_after.values())),
        "note": "Guardrail 2: per-course z-norm leaves residual institution signal "
                "(distribution-shape). Greedy backward elimination by RandomForest "
                "institution-importance drops leakers until every week's HGB grouped "
                "probe ≤0.75. Dropped features stay in the parquet for audit; model "
                "uses model_feature_cols only.",
    }
    (OUT_DIR / "g2_build_report.json").write_text(json.dumps(report, indent=2))
    print(f"[G2] invariance: dropped {len(dropped)} leakers → probe max "
          f"{report['institution_invariance']['probe_auc_before_max']} → "
          f"{report['institution_invariance']['probe_auc_after_max']} "
          f"(pass={report['institution_invariance']['all_weeks_probe_pass']}); "
          f"model features = {len(model_cols)}", flush=True)

    # feature schema
    (OUT_DIR / "feature_schema.json").write_text(json.dumps({
        "id_cols": ["inst", "sid", "course_id", "y"],
        "base_features": base_order,
        "znorm_features": znorm_cols,
        "model_feature_cols": model_cols,
        "dropped_institution_leakers": dropped,
        "families": {
            "session": ["n_sessions", "total_session_minutes", "mean_session_minutes",
                        "median_session_minutes", "max_session_minutes", "mean_events_per_session",
                        "short_session_share", "dwell_per_event", "session_regularity"],
            "category": [f"cnt_{b}" for b in SHARED_BINS] + [f"share_{b}" for b in SHARED_BINS],
            "temporal": ["share_night", "share_morning", "share_afternoon", "share_evening",
                         "weekend_share", "prep_evening_night_share", "distinct_hours_local",
                         "distinct_dows", "hour_entropy", "dow_entropy"],
            "weekly": ["mean_weekly_views", "std_weekly_views", "mean_weekly_sessions",
                       "n_active_weeks", "inactive_weeks", "active_week_ratio", "trend_slope",
                       "momentum", "last_week_deviation", "first_week_events", "front_load_share"],
            "totals": ["total_events", "active_days", "events_per_active_day", "span_days",
                       "mean_intergap_hours", "median_intergap_hours", "max_inactivity_gap_days"],
            "first_access": ["first_event_day", "last_event_day", "first_event_hour_local",
                             "days_to_first_assignment", "days_to_first_quiz"],
        },
    }, indent=2))
    print(f"[G2] wrote features + schema + mapping to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
