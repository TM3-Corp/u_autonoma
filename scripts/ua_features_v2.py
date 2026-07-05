#!/usr/bin/env python3
"""UA-1 — UA feature completion.

Assembles the full UA feature matrix per cutoff {2,4,6,8,full}:
  - existing enriched features via train_time_limited_model.load_features(
    include_znorm=True) — the same loader the historical UA pipeline used;
  - PLUS data/enriched_features/pre_assessment_features.parquet (34 features),
    joined on user_id/course_id.

Leak handling (documented): pre_assessment features are FULL-HORIZON (computed
over all data, incl. behavior around deadlines). Joining them at a temporal
cutoff would leak post-cutoff information. Therefore pre_assessment is included
ONLY in the `full` cutoff; temporal cutoffs {2,4,6,8} exclude them. U5 (isolating
pre_assessment value) is a full-cutoff-only comparison in UA-2/UA-3.

Also freezes the 3 label arms (identical membership reused downstream):
  KEEP  = all 373 enrollments, failed=final_score<57 (recorded Canvas outcome;
          includes 51 active-zero LTI artifacts — carries the label caveat).
  DROP-A= 322 (drop the 51 active-zeros: final_score==0 & >=20 views; KEEP 86676).
  A+    = 286 (drop the 51 active-zeros AND course 86676) — sensitivity only.

Output: tier2_push/ua_features/week_{w}.parquet (feature matrix, user_id/course_id
keyed), tier2_push/ua_features/arms.parquet (arm membership + labels),
tier2_push/ua_features_report.json.
"""
import json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_time_limited_model as TT
import ua_remediate_labels as UR  # reuse the EXACT T4 active-zero logic

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[1]
PV = REPO / "data/page_views/categorized_page_views.parquet"
ENROLL = REPO / "data/page_views/student_enrollments.csv"
PRE_ASSESS = REPO / "data/enriched_features/pre_assessment_features.parquet"
OUT_DIR = REPO / "data/puc/sota_results/tier2_push/ua_features"
OUT_REPORT = REPO / "data/puc/sota_results/tier2_push/ua_features_report.json"
CUTOFFS = [2, 4, 6, 8, "full"]
DROP_COURSE = 86676
ACTIVE_VIEW_MIN = 20
FAIL_THRESHOLD = 57
ID = ["user_id", "course_id"]


def build_arms():
    # reuse the EXACT T4 active-zero set (handles user_id normalization +
    # MODEL_COURSES filter) so DROP-A/A+ membership is identical to Tier-1.
    enr_ur, az = UR.compute_active_zero_set()
    enr = pd.read_csv(ENROLL)
    enr["failed"] = (enr["final_score"] < FAIL_THRESHOLD).astype(int)
    assert len(az) == 51, f"expected 51 active-zero, got {len(az)}"

    def in_az(r):
        return (r["user_id"], r["course_id"]) in az

    enr["is_active_zero"] = enr.apply(in_az, axis=1)
    enr["arm_keep"] = True
    enr["arm_dropA"] = ~enr["is_active_zero"]
    enr["arm_aplus"] = (~enr["is_active_zero"]) & (enr["course_id"] != DROP_COURSE)
    counts = {"KEEP": int(enr["arm_keep"].sum()),
              "DROP_A": int(enr["arm_dropA"].sum()),
              "A_plus": int(enr["arm_aplus"].sum())}
    fails = {"KEEP": int(enr.loc[enr["arm_keep"], "failed"].sum()),
             "DROP_A": int(enr.loc[enr["arm_dropA"], "failed"].sum()),
             "A_plus": int(enr.loc[enr["arm_aplus"], "failed"].sum())}
    assert counts["KEEP"] == 373 and counts["DROP_A"] == 322 and counts["A_plus"] == 286, counts
    return enr, counts, fails, len(az)


def load_feature_matrix(cutoff, universe):
    """Feature matrix aligned to the full KEEP universe (373 pairs); no-activity
    students zero-filled (informative for early warning; PUC-consistent)."""
    df = TT.load_features(cutoff, include_znorm=True)
    # keep ids + numeric features; drop dup/object columns
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = ID + [c for c in num if c not in ID]
    df = df[keep].copy()
    df = df.replace([np.inf, -np.inf], 0.0)
    df = df.drop_duplicates(subset=ID)  # outer merges can repeat pairs
    n_base = df.shape[1] - 2
    added_pa = False
    if cutoff == "full":
        pa = pd.read_parquet(PRE_ASSESS)
        pa_cols = [c for c in pa.columns if c not in ID]
        pa = pa[ID + pa_cols].drop_duplicates(subset=ID)
        df = df.merge(pa, on=ID, how="left", suffixes=("", "_pa"))
        df[pa_cols] = df[pa_cols].fillna(0.0)
        added_pa = True
    n_active = int(df.merge(universe, on=ID, how="inner").shape[0])
    # align to the full arm universe; no-activity pairs -> 0
    df = universe.merge(df, on=ID, how="left").fillna(0.0)
    return df, n_base, added_pa, n_active


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    enr, counts, fails, n_az = build_arms()
    enr[ID + ["failed", "final_score", "is_active_zero",
              "arm_keep", "arm_dropA", "arm_aplus"]].to_parquet(
        OUT_DIR / "arms.parquet", index=False)
    print(f"[UA-1] arms: {counts} fails={fails} active_zero={n_az}", flush=True)

    report = {"arms": counts, "arm_fails": fails, "n_active_zero": n_az,
              "fail_threshold": FAIL_THRESHOLD,
              "leak_note": ("pre_assessment_features are FULL-HORIZON; included ONLY "
                            "in the `full` cutoff to avoid leaking post-cutoff info. "
                            "Temporal cutoffs {2,4,6,8} exclude them. U5 (pre_assessment "
                            "value) is a full-cutoff-only comparison."),
              "residual_note": ("jaccard_to_passing graph feature reused as-is (built with "
                                ">=60 passing set in legacy code); not recomputed here "
                                "(would require regenerating graph features). Optional "
                                "cross-course xc feature not added (time budget)."),
              "cutoffs": {}}

    PA_COLS = [c for c in pd.read_parquet(PRE_ASSESS).columns if c not in ID]
    universe = enr.loc[enr["arm_keep"], ID].drop_duplicates().sort_values(ID).reset_index(drop=True)
    for cutoff in CUTOFFS:
        wk = str(cutoff)
        df, n_base, added_pa, n_active = load_feature_matrix(cutoff, universe)
        df.to_parquet(OUT_DIR / f"week_{wk}.parquet", index=False)
        # arm sizes (feature matrix aligned to full universe -> exact arm sizes)
        joins = {}
        for arm, col in [("KEEP", "arm_keep"), ("DROP_A", "arm_dropA"), ("A_plus", "arm_aplus")]:
            sub = enr[enr[col]][ID + ["failed"]]
            merged = df.merge(sub, on=ID, how="inner")
            joins[arm] = {"n_joined": int(len(merged)),
                          "n_fail": int(merged["failed"].sum()),
                          "prevalence": round(float(merged["failed"].mean()), 4)}
        pa_present = added_pa and all(c in df.columns for c in PA_COLS)
        report["cutoffs"][wk] = {
            "n_feature_rows": int(len(df)), "n_active_students": n_active,
            "n_zero_filled": int(len(df) - n_active),
            "n_features": int(df.shape[1] - 2), "n_base_features": int(n_base),
            "pre_assessment_included": bool(added_pa),
            "pre_assessment_cols_present": bool(pa_present),
            "n_pre_assessment": len(PA_COLS) if added_pa else 0,
            "arm_sizes": joins,
        }
        print(f"[UA-1] wk{wk}: feat_rows={len(df)} active={n_active} zero_filled={len(df)-n_active} "
              f"n_feat={df.shape[1]-2} pre_assess={added_pa} | KEEP={joins['KEEP']['n_joined']} "
              f"DROP_A={joins['DROP_A']['n_joined']} A+={joins['A_plus']['n_joined']}",
              flush=True)

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"[UA-1] wrote {OUT_REPORT}", flush=True)


if __name__ == "__main__":
    main()
