#!/usr/bin/env python3
"""P1 — Features v2: thesis families on clean data.

Computes NEW feature families per (student, TARGET-course) pair at cutoffs
{2,4,6,8,full} and appends them (raw + per-course z-norm) to the T2 clean
matrices, aligned to the SAME 560-pair universe and row order. The T2 base is
NOT recomputed — it is loaded from tier1_clean/features/week_{w}_clean.parquet
and joined.

Leak-freedom (hard rule): the cutoff clock is the TARGET course's course_start
(0.05 quantile, identical to T2 because get_course_starts is per-course). EVERY
feature — including cross-course ones — uses only events with
created_at <= target_start + cutoff_weeks (for "full": <= the target course's
last event). Session gap = 30 min (T8). Degenerate/empty cases return 0 (not NaN)
so pre-fill NaN stays ~0.

Families:
  A Cross-course context (all 20 courses of the clean parquet)
  B Intensity (weekly views vs own mean, within target course)
  C Workload slope (signed weekly-view deltas)
  D Peaks (local maxima / weeks above own-mean thresholds)
  E Composites (proactivity x regularity; pooled hour/weekday entropy)

Output: tier2_push/features/week_{w}_v2.parquet, tier2_push/features_v2_report.json
"""
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd

import puc_benchmark_sota as B

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
CLEAN = REPO / "data/puc/puc_clean_data.parquet"
T2_FEAT = REPO / "data/puc/sota_results/tier1_clean/features"
OUT_FEAT = REPO / "data/puc/sota_results/tier2_push/features"
OUT_REPORT = REPO / "data/puc/sota_results/tier2_push/features_v2_report.json"

TARGET_COURSES = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
PERCENTILE = 0.05
CUTOFFS = [2, 4, 6, 8, "full"]
SESSION_GAP_MIN = 30.0
DAY_NS = 86_400 * 1_000_000_000
WEEK_NS = 7 * DAY_NS
EPS = 1e-9

# new RAW base columns produced by this script (order fixed for the report)
XC_COLS = ["xc_total_views", "xc_total_sessions", "xc_course_share_views",
           "xc_course_share_sessions", "xc_n_active_other_courses",
           "xc_sessions_between_mean", "xc_sessions_between_max",
           "xc_sessions_between_total", "xc_relative_neglect",
           "xc_max_other_course_share"]
B_COLS = ["intensity_max_dev", "intensity_min_dev", "intensity_std_dev",
          "intensity_last_week_dev"]
C_COLS = ["slope_pos_sum", "slope_neg_sum", "slope_pos_count",
          "slope_neg_count", "slope_ratio"]
D_COLS = ["n_local_peaks", "weeks_above_25pct", "weeks_above_50pct",
          "weeks_above_100pct", "weeks_above_150pct"]
E_COLS = ["procrastination_x_regularity", "pdh_entropy", "pwd_entropy"]
NEW_RAW_COLS = XC_COLS + B_COLS + C_COLS + D_COLS + E_COLS  # 27 new base features


# ---------- session helpers (operate on int64-ns sorted arrays) ----------
def _session_count(ts_ns):
    if ts_ns.size == 0:
        return 0
    if ts_ns.size == 1:
        return 1
    gaps_min = np.diff(ts_ns) / (60 * 1e9)
    return int(1 + (gaps_min > SESSION_GAP_MIN).sum())


def _session_starts(ts_ns):
    """Return int64-ns start timestamps of each 30-min-gap session (ts sorted)."""
    if ts_ns.size == 0:
        return np.empty(0, dtype=np.int64)
    if ts_ns.size == 1:
        return ts_ns.copy()
    gaps_min = np.diff(ts_ns) / (60 * 1e9)
    boundary = np.concatenate([[True], gaps_min > SESSION_GAP_MIN])
    return ts_ns[boundary]


def _entropy(counts):
    counts = np.asarray(counts, dtype=float)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts[counts > 0] / tot
    return float(-(p * np.log2(p)).sum())


def _weekly_views(ts_ns, start_ns, upper_ns):
    """Views per week-index (0-based, relative to target start) within window."""
    m = ts_ns <= upper_ns
    t = ts_ns[m]
    if t.size == 0:
        return np.zeros(0, dtype=float)
    widx = np.floor((t - start_ns) / WEEK_NS).astype(int)
    widx = np.clip(widx, 0, None)
    n_weeks = int(widx.max()) + 1
    v = np.zeros(n_weeks, dtype=float)
    np.add.at(v, widx, 1.0)
    return v


# ---------- family computations for ONE (student, target) pair ----------
def _cross_course(stu_ns, stu_course, upper_ns, target):
    """Family A. stu_ns/stu_course = student's ALL-course events (sorted by ns)."""
    m = stu_ns <= upper_ns
    ns_w, cid_w = stu_ns[m], stu_course[m]
    total_views = int(ns_w.size)
    if total_views == 0:
        return dict.fromkeys(XC_COLS, 0.0)

    is_tgt = cid_w == target
    tgt_views = int(is_tgt.sum())
    total_sessions = _session_count(ns_w)  # events already sorted
    tgt_sessions = _session_count(ns_w[is_tgt])

    share_views = tgt_views / max(total_views, 1)
    share_sessions = tgt_sessions / max(total_sessions, 1)

    # per-other-course view shares (active = >=5 events)
    other_courses = np.unique(cid_w[~is_tgt])
    shares, n_active_other, max_other_share = [], 0, 0.0
    for c in other_courses:
        cv = int((cid_w == c).sum())
        if cv >= 5:
            n_active_other += 1
        s = cv / max(total_views, 1)
        shares.append(s)
        max_other_share = max(max_other_share, s)
    # relative neglect: target share minus mean share over all the student's
    # active courses (target + others with >=5 events)
    active_shares = [share_views] if tgt_views >= 5 else []
    active_shares += [s for c, s in zip(other_courses, shares)
                      if int((cid_w == c).sum()) >= 5]
    mean_active_share = float(np.mean(active_shares)) if active_shares else share_views
    rel_neglect = share_views - mean_active_share

    # sessions of OTHER courses falling between consecutive target sessions
    tgt_starts = _session_starts(ns_w[is_tgt])
    oth_starts = _session_starts(ns_w[~is_tgt])
    if tgt_starts.size >= 2 and oth_starts.size > 0:
        counts = []
        for a, b in zip(tgt_starts[:-1], tgt_starts[1:]):
            counts.append(int(((oth_starts > a) & (oth_starts < b)).sum()))
        counts = np.asarray(counts, dtype=float)
        sb_mean, sb_max, sb_total = float(counts.mean()), float(counts.max()), float(counts.sum())
    else:
        sb_mean = sb_max = sb_total = 0.0

    return {
        "xc_total_views": float(total_views),
        "xc_total_sessions": float(total_sessions),
        "xc_course_share_views": float(share_views),
        "xc_course_share_sessions": float(share_sessions),
        "xc_n_active_other_courses": float(n_active_other),
        "xc_sessions_between_mean": sb_mean,
        "xc_sessions_between_max": sb_max,
        "xc_sessions_between_total": sb_total,
        "xc_relative_neglect": float(rel_neglect),
        "xc_max_other_course_share": float(max_other_share),
    }


def _intensity_slope_peaks(v):
    """Families B, C, D from the weekly-views series v (0 for empty/short)."""
    out = dict.fromkeys(B_COLS + C_COLS + D_COLS, 0.0)
    if v.size == 0:
        return out
    mean_v = float(v.mean())
    denom = mean_v + EPS
    dev = (v - mean_v) / denom
    out["intensity_max_dev"] = float(dev.max())
    out["intensity_min_dev"] = float(dev.min())
    out["intensity_std_dev"] = float(v.std() / denom)
    out["intensity_last_week_dev"] = float(dev[-1])

    if v.size >= 2:
        d = np.diff(v)
        pos, neg = d[d > 0], d[d < 0]
        out["slope_pos_sum"] = float(pos.sum())
        out["slope_neg_sum"] = float(neg.sum())
        out["slope_pos_count"] = float(pos.size)
        out["slope_neg_count"] = float(neg.size)
        out["slope_ratio"] = float(pos.sum() / (abs(neg.sum()) + EPS))

    if v.size >= 3:
        peaks = ((v[1:-1] > v[:-2]) & (v[1:-1] > v[2:])).sum()
        out["n_local_peaks"] = float(peaks)
    out["weeks_above_25pct"] = float((v > 0.25 * mean_v).sum())
    out["weeks_above_50pct"] = float((v > 0.50 * mean_v).sum())
    out["weeks_above_100pct"] = float((v > 1.00 * mean_v).sum())
    out["weeks_above_150pct"] = float((v > 1.50 * mean_v).sum())
    return out


def _pooled_entropy(hours, dows):
    """Family E entropies from pooled target-course hour/weekday histograms."""
    hh = np.bincount(hours, minlength=24) if hours.size else np.zeros(24)
    ww = np.bincount(dows, minlength=7) if dows.size else np.zeros(7)
    return _entropy(hh), _entropy(ww)


def compute_week(cutoff, base_df, students, course_end_ns, target_starts_ns):
    """Return a DataFrame of NEW raw columns aligned to base_df row order."""
    rows = []
    for sid, cid in zip(base_df["student_id"].to_numpy(),
                        base_df["course_id"].to_numpy()):
        stu = students.get(sid)
        start_ns = target_starts_ns[cid]
        if cutoff == "full":
            upper_ns = course_end_ns[cid]
        else:
            upper_ns = start_ns + cutoff * WEEK_NS

        if stu is None:
            rec = dict.fromkeys(NEW_RAW_COLS, 0.0)
            rows.append(rec)
            continue

        stu_ns, stu_course = stu["ns"], stu["course"]
        rec = _cross_course(stu_ns, stu_course, upper_ns, cid)

        # target-course events within window (for B/C/D/E)
        tm = (stu_course == cid) & (stu_ns <= upper_ns)
        t_ns = stu_ns[tm]
        v = _weekly_views(t_ns, start_ns, upper_ns)
        rec.update(_intensity_slope_peaks(v))
        pdh, pwd = _pooled_entropy(stu["hour"][tm], stu["dow"][tm])
        rec["pdh_entropy"] = pdh
        rec["pwd_entropy"] = pwd
        rec["procrastination_x_regularity"] = np.nan  # filled from base below
        rows.append(rec)

    new = pd.DataFrame(rows, columns=NEW_RAW_COLS)
    # Family E composite from the base matrix (raw, non-znorm) columns.
    # "proactivity index" -> quizzes_proact_mean_pct (top SHAP driver); documented.
    prox = base_df["quizzes_proact_mean_pct"].to_numpy()
    reg = base_df["session_regularity"].to_numpy()
    new["procrastination_x_regularity"] = prox * reg
    return new


def znorm_per_course(new_raw, course_ids):
    """Per-course z-norm (T2 policy) preserving row order; std==0 -> 0."""
    df = new_raw.copy()
    df["_c"] = course_ids
    z = df.groupby("_c")[NEW_RAW_COLS].transform(
        lambda x: (x - x.mean()) / x.std() if x.std(ddof=1) > 0 else x * 0.0)
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    z.columns = [f"{c}_znorm" for c in NEW_RAW_COLS]
    return z


def main():
    OUT_FEAT.mkdir(parents=True, exist_ok=True)
    clean = pd.read_parquet(CLEAN, columns=["student_id", "course_id",
                                            "created_at", "hour_local", "dow_local"])
    clean["ns"] = clean["created_at"].astype("int64")

    # target course starts (0.05 quantile) — per-course, identical to T2
    target_starts_ns = {c: int(v.value) for c, v in
                        B.get_course_starts(clean, PERCENTILE).items()
                        if c in TARGET_COURSES}
    course_end_ns = {c: int(clean.loc[clean["course_id"] == c, "ns"].max())
                     for c in TARGET_COURSES}

    # per-student event arrays (ALL 20 courses), sorted by created_at
    clean = clean.sort_values("ns")
    students = {}
    for sid, g in clean.groupby("student_id"):
        students[sid] = {
            "ns": g["ns"].to_numpy(),
            "course": g["course_id"].to_numpy(),
            "hour": g["hour_local"].to_numpy().astype(int),
            "dow": g["dow_local"].to_numpy().astype(int),
        }
    print(f"[P1] {len(students)} students indexed; targets {sorted(target_starts_ns)}", flush=True)

    report = {"cutoffs": {}, "new_raw_columns": NEW_RAW_COLS,
              "n_new_base_features": len(NEW_RAW_COLS),
              "families": {"A_cross_course": XC_COLS, "B_intensity": B_COLS,
                           "C_slope": C_COLS, "D_peaks": D_COLS, "E_composite": E_COLS},
              "nan_policy": "fill 0 after znorm; composite=quizzes_proact_mean_pct*session_regularity",
              "leak_note": "cutoff clock = target course start(0.05q); created_at<=start+cutoff for all families; full=target course last event"}

    for cutoff in CUTOFFS:
        wk = str(cutoff)
        base = pd.read_parquet(T2_FEAT / f"week_{wk}_clean.parquet")
        new_raw = compute_week(cutoff, base, students, course_end_ns, target_starts_ns)

        pre_nan = float(new_raw.isna().to_numpy().mean())
        new_z = znorm_per_course(new_raw, base["course_id"].to_numpy())
        new_raw_filled = new_raw.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        out = pd.concat([base.reset_index(drop=True),
                         new_raw_filled.reset_index(drop=True),
                         new_z.reset_index(drop=True)], axis=1)
        assert len(out) == 560, f"week {wk}: {len(out)} rows != 560"
        out.to_parquet(OUT_FEAT / f"week_{wk}_v2.parquet", index=False)

        stats = {c: {"mean": round(float(new_raw_filled[c].mean()), 4),
                     "std": round(float(new_raw_filled[c].std()), 4),
                     "nan_pre": round(float(new_raw[c].isna().mean()), 4)}
                 for c in NEW_RAW_COLS}
        report["cutoffs"][wk] = {
            "n_rows": int(len(out)), "n_base_cols": int(base.shape[1] - 4),
            "n_new_base": len(NEW_RAW_COLS), "n_new_znorm": len(NEW_RAW_COLS),
            "n_total_cols": int(out.shape[1]),
            "nan_rate_new_pre_fill": round(pre_nan, 5),
            "nan_ok_le_5pct": bool(pre_nan <= 0.05),
            "stats": stats,
        }
        print(f"[P1] week {wk}: rows={len(out)} new_base={len(NEW_RAW_COLS)} "
              f"total_cols={out.shape[1]} nan_pre={pre_nan:.5f}", flush=True)

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[P1] wrote {OUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
