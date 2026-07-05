#!/usr/bin/env python3
"""G3 — Course profile table (predictability map's raw material).

Per course (17): institution, n, fails, prevalence, events/student (median),
sessions/student (median), active-weeks coverage, grade-distribution stats
(std, ceiling share, zeros share on a [0,1]-normalized score), and per-course
LOCO AUC under the reference config (CatBoost Balanced, top-40/fold, pooled R0
training, seed 42) at the reference week (8).

Output: tier3_pooled/course_profiles.json + course_profiles.md
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import tier3_common as T

REPO = Path(__file__).resolve().parents[1]
POOL = REPO / "data/puc/sota_results/tier3_pooled"
PUC_GRADES = REPO / "data/puc/puc_grades_clean.parquet"
UA_ENROLL = REPO / "data/page_views/student_enrollments.csv"
REF_WEEK = "8"


def load_scores():
    """Normalized [0,1] score per (inst, course_id, sid): PUC grade/7, UA final_score/100 (DROP-A)."""
    g = pd.read_parquet(PUC_GRADES)
    g = g[g.course_id.isin(T.PUC_COURSES)].drop_duplicates(["student_id", "course_id"])
    puc = pd.DataFrame({"inst": "PUC", "course_id": g.course_id.values,
                        "score01": (g.grade / 7.0).clip(0, 1).values})
    # UA DROP-A scores: reuse the same active-zero drop as G2
    import common_features as CF
    ulab, _ = CF.ua_drop_a_labels()
    enr = pd.read_csv(UA_ENROLL)
    enr["user_id"] = enr["user_id"].map(CF.normalize_user_id).astype("int64")
    enr = enr.merge(ulab.rename(columns={"sid": "user_id"})[["user_id", "course_id"]],
                    on=["user_id", "course_id"], how="inner")
    ua = pd.DataFrame({"inst": "UA", "course_id": enr.course_id.values,
                       "score01": (enr.final_score / 100.0).clip(0, 1).values})
    return pd.concat([puc, ua], ignore_index=True)


def grade_dist(scores, cid):
    s = scores[scores.course_id == cid]["score01"].values
    return {"score_std": round(float(np.std(s)), 4),
            "ceiling_share": round(float((s >= 0.95).mean()), 4),
            "zeros_share": round(float((s <= 0.01).mean()), 4)}


def main():
    t0 = time.time()
    df8 = T.load_week(REF_WEEK)
    dffull = T.load_week("full")
    T.assert_rules(df8)
    scores = load_scores()

    # per-course LOCO AUC: single pooled-R0 LOCO run, reference config, seed 42
    print("[G3] running reference LOCO (CatBoost Balanced top-40, pooled R0, seed 42, wk8)...", flush=True)
    r0 = T.subset(df8, "R0", "pooled")
    oof, y, g, nsp = T.oof_predict(r0, kind="cat", N=40, seed=T.RANDOM_STATE)
    pc_auc = T.per_course_auc(r0, y, oof)
    pooled = T.pooled_auc(y, oof)
    print(f"[G3] pooled R0 wk8 LOCO AUC={pooled} (nsplits={nsp}) [{time.time()-t0:.0f}s]", flush=True)

    stats8 = T.course_stats(df8).set_index("course_id")
    profiles = []
    for cid in sorted(df8.course_id.unique()):
        row8 = stats8.loc[cid]
        fpair = dffull[dffull.course_id == cid]
        prof = {
            "course_id": int(cid),
            "institution": row8["inst"],
            "n": int(row8["n"]),
            "fails": int(row8["fails"]),
            "prevalence": round(float(row8["prev"]), 4),
            "events_per_student_median": round(float(fpair["total_events"].median()), 1),
            "sessions_per_student_median": round(float(fpair["n_sessions"].median()), 1),
            "active_weeks_mean": round(float(fpair["n_active_weeks"].mean()), 2),
            "active_week_ratio_mean": round(float(fpair["active_week_ratio"].mean()), 3),
            **grade_dist(scores, cid),
            "loco_auc_wk8": pc_auc.get(int(cid)),
            "in_R1": int(cid) in T.R1_EXPECTED,
            "in_R2": int(cid) in T.R2_EXPECTED,
        }
        profiles.append(prof)

    out = {"reference_config": "CatBoost Balanced, top-40/fold, pooled R0 LOCO, seed 42, week 8",
           "pooled_R0_wk8_loco_auc": pooled,
           "n_courses": len(profiles),
           "profiles": profiles}
    (POOL / "course_profiles.json").write_text(json.dumps(out, indent=2))

    # markdown table
    hdr = ("| course | inst | n | fails | prev | ev/stu | sess/stu | act_wk | score_std | "
           "ceil | zeros | LOCO AUC wk8 | R1 | R2 |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for p in sorted(profiles, key=lambda x: (x["institution"], -x["prevalence"])):
        lines.append(f"| {p['course_id']} | {p['institution']} | {p['n']} | {p['fails']} | "
                     f"{p['prevalence']:.1%} | {p['events_per_student_median']:.0f} | "
                     f"{p['sessions_per_student_median']:.0f} | {p['active_weeks_mean']:.1f} | "
                     f"{p['score_std']:.3f} | {p['ceiling_share']:.0%} | {p['zeros_share']:.0%} | "
                     f"{p['loco_auc_wk8'] if p['loco_auc_wk8'] is not None else 'null'} | "
                     f"{'✓' if p['in_R1'] else ''} | {'✓' if p['in_R2'] else ''} |")
    (POOL / "course_profiles.md").write_text(
        f"# G3 — Course profiles (17 courses)\nReference LOCO AUC: CatBoost Balanced top-40, "
        f"pooled R0, seed 42, week 8. Pooled R0 wk8 LOCO AUC = {pooled}.\n\n" + hdr + "\n".join(lines) + "\n")

    n_null = sum(1 for p in profiles if p["loco_auc_wk8"] is None)
    print(f"[G3] wrote {len(profiles)} profiles; {n_null} null AUC (courses <2 fails). "
          f"[{time.time()-t0:.0f}s]", flush=True)
    # verifier
    assert len(profiles) == 17, f"expected 17 courses, got {len(profiles)}"
    for p in profiles:
        if p["fails"] < 2:
            assert p["loco_auc_wk8"] is None
    print("[G3] VERIFIER PASS: 17 rows, all fields present, null-AUC rule honored.", flush=True)


if __name__ == "__main__":
    main()
