#!/usr/bin/env python3
"""T2 — Clean feature rebuild.

Reuses puc_benchmark_sota feature machinery but reads the cleaned parquet and
uses hour_local/dow_local for all time-of-day / weekday features (by aliasing
the `hour`/`day_of_week` columns the feature functions read — no edit to
puc_benchmark_sota.py). Builds per-week feature matrices aligned to the full
560-pair grade universe for BOTH arms (old=UTC, clean=local+deduped) so T3 can
do a strictly paired A/B on identical folds.

CLI computes all weeks for both arms, logs shapes + NaN rates, and writes
tier1_clean/feature_build_report.json.
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

import puc_benchmark_sota as B

REPO = Path(__file__).resolve().parents[1]
OLD_PARQUET = REPO / "data/puc/puc_fixed_data.parquet"
CLEAN_PARQUET = REPO / "data/puc/puc_clean_data.parquet"
GRADES = REPO / "data/puc/puc_grades_clean.parquet"
OUT_DIR = REPO / "data/puc/sota_results/tier1_clean"
FEAT_DIR = OUT_DIR / "features"

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
PERCENTILE = 0.05
CUTOFFS = [2, 4, 6, 8, "full"]


def load_pv(path, use_local):
    df = pd.read_parquet(path)
    df = df[df["course_id"].isin(COURSE_IDS)].copy()
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    if use_local:
        # T2 requirement: local time drives ALL time-of-day / weekday features.
        # feature funcs read literal columns "hour"/"day_of_week"; alias them.
        assert "hour_local" in df.columns and "dow_local" in df.columns, "clean parquet missing local cols"
        df["hour"] = df["hour_local"].astype("int32")
        df["day_of_week"] = df["dow_local"].astype("int32")
    return df


def build_week_matrix(df_pv, df_grades, cutoff):
    """Return (X, y, groups, meta) aligned to the full grade universe.

    X: features per (student_id, course_id), reindexed to every graded pair,
       missing (no activity before cutoff) -> 0. with-assessment (production).
    """
    course_starts = B.get_course_starts(df_pv, PERCENTILE)
    dfw = B.filter_by_cutoff(df_pv, course_starts, cutoff)
    feats = B.calculate_all_features(dfw, course_starts, cutoff_weeks=cutoff)

    # canonical universe = every graded (student, course) pair, sorted
    grades = df_grades[["student_id", "course_id", "grade"]].drop_duplicates(
        ["student_id", "course_id"]).sort_values(["course_id", "student_id"]).reset_index(drop=True)

    merged = grades.merge(feats, on=["student_id", "course_id"], how="left")

    id_cols = {"student_id", "course_id", "grade", "failed"}
    fcols = [c for c in merged.columns
             if c not in id_cols and pd.api.types.is_numeric_dtype(merged[c])]

    nan_rate = float(merged[fcols].isna().to_numpy().mean()) if fcols else 0.0
    n_missing_pairs = int(merged[fcols].isna().all(axis=1).sum()) if fcols else 0

    znormed = B.calculate_znorm(merged.copy(), fcols)
    all_feat_cols = [c for c in znormed.columns
                     if c not in id_cols and pd.api.types.is_numeric_dtype(znormed[c])]

    X = B.filter_assessment_features(znormed[all_feat_cols], True)
    X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = (znormed["grade"].to_numpy() < 4.0).astype(int)
    groups = znormed["course_id"].to_numpy()

    meta = {
        "cutoff": str(cutoff),
        "n_rows": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_positives": int(y.sum()),
        "prevalence": round(float(y.mean()), 4),
        "raw_nan_rate": round(nan_rate, 4),
        "n_pairs_no_activity": n_missing_pairs,
    }
    ids = znormed[["student_id", "course_id"]].reset_index(drop=True)
    return X.reset_index(drop=True), y, groups, ids, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true", help="write per-week matrices to parquet")
    args = ap.parse_args()

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    df_grades = pd.read_parquet(GRADES)
    df_grades = df_grades[df_grades["course_id"].isin(COURSE_IDS)].copy()
    print(f"[T2] grade universe: {df_grades[['student_id','course_id']].drop_duplicates().shape[0]} pairs, "
          f"{int((df_grades['grade']<4.0).sum())} fails", flush=True)

    pv_old = load_pv(OLD_PARQUET, use_local=False)
    pv_clean = load_pv(CLEAN_PARQUET, use_local=True)

    report = {"grade_pairs": int(df_grades[["student_id", "course_id"]].drop_duplicates().shape[0]),
              "weeks": {}}
    for cutoff in CUTOFFS:
        wk = str(cutoff)
        Xo, yo, go, ido, mo = build_week_matrix(pv_old, df_grades, cutoff)
        Xc, yc, gc, idc, mc = build_week_matrix(pv_clean, df_grades, cutoff)
        # NaN-rate comparison. Relative "within 20%" is meaningless when the
        # baseline NaN rate is ~0 (e.g. week 2: old 0.0018 vs clean 0.0 -> the
        # clean pipeline REDUCED NaN). Pass if within 20% relative OR within an
        # absolute 2pp (both effectively zero => no NaN inflation).
        old_nan = max(mo["raw_nan_rate"], 1e-9)
        rel = abs(mc["raw_nan_rate"] - mo["raw_nan_rate"]) / old_nan
        abs_diff = abs(mc["raw_nan_rate"] - mo["raw_nan_rate"])
        within20 = (rel <= 0.20) or (abs_diff <= 0.02)
        report["weeks"][wk] = {
            "old": mo, "clean": mc,
            "nan_rate_rel_diff": round(rel, 4),
            "nan_within_20pct": bool(within20),
            "n_match_560": bool(mo["n_rows"] == mc["n_rows"] == report["grade_pairs"]),
        }
        print(f"[T2] week {wk}: old n={mo['n_rows']} f={mo['n_features']} nan={mo['raw_nan_rate']} | "
              f"clean n={mc['n_rows']} f={mc['n_features']} nan={mc['raw_nan_rate']} | "
              f"rel_nan_diff={rel:.3f} within20={within20}", flush=True)
        if args.cache:
            for arm, X, y, g, ids in [("old", Xo, yo, go, ido), ("clean", Xc, yc, gc, idc)]:
                out = X.copy()
                out.insert(0, "_y", y)
                out.insert(0, "_group", g)
                out.insert(0, "course_id", ids["course_id"].values)
                out.insert(0, "student_id", ids["student_id"].values)
                out.to_parquet(FEAT_DIR / f"week_{wk}_{arm}.parquet", index=False)

    (OUT_DIR / "feature_build_report.json").write_text(json.dumps(report, indent=2))
    print(f"[T2] wrote {OUT_DIR/'feature_build_report.json'}", flush=True)
    return report


if __name__ == "__main__":
    main()
