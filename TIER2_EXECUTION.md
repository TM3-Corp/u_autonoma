# TIER-2 EXECUTION PACK — PUC metrics push (features v2 + model bake-off + confirmatory)
**Designed by Fable 5 · 2026-07-03 · to be executed by Opus 4.8 in a fresh session**
Prerequisites (all DONE, do not redo): Tier-1 pack (`TIER1_EXECUTION.md`), results (`TIER1_RESULTS.md`), verifier log (`TIER1_PROGRESS.md`). Clean data is canonical: `data/puc/puc_clean_data.parquet` (T3 decision: ADOPT). Cached per-week matrices: `data/puc/sota_results/tier1_clean/features/week_{2,4,6,8,full}_{old,clean}.parquet`.

## Mission

Two tracks, PUC first (priority for the Enrique document), UA second.
**PUC**: close the gap between the honest nested numbers (0.77/0.81/0.79/0.85/0.77) and the old optimistic table (0.83/0.87/0.86/0.86/0.85) — honestly. Levers, in ROI order: CatBoost adoption, rank-average ensembling, cross-course context features (thesis family, never ported), seed bagging, train-only augmentation. Target: honest nested wk4 ≥ 0.83–0.85, wk8 ≥ 0.86.
**UA**: give UA the same optimized pipeline it never received (old pipeline = untuned XGB, crude FS, no calibration, no CatBoost, and the 34 `pre_assessment` features never loaded), under a **two-arm label design**: Arm KEEP = all 373 (target = *recorded* Canvas outcome `final_score<57`, includes the 51 active-zero LTI artifacts — real, well-defined, but not identical to true reprobación); Arm DROP = remediation A, n=322 (drop the 51 active-zeros, KEEP course 86676 — Paul's standing position; A+ n=286 reported only as a sensitivity row). The pair brackets true performance until official actas arrive.

## Non-negotiable guardrails (inherit all Tier-1 guardrails, plus)

1. Never modify: `benchmark_results.json` + backups, `few_feature_sweep/`, existing parquets, anything under `tier1_clean/` except NEW files. All Tier-2 outputs go to `data/puc/sota_results/tier2_push/` (create it).
2. Do not touch `~/projects/tm3-roi-diagnostico/` files except as specified in P6 (and P6 is GATED on Paul's approval — prepare numbers, do NOT edit HTML).
3. `RANDOM_STATE=42` base; CV repeat seeds exactly {42,43,44,45,46}; identical folds within each paired comparison.
4. No SMOTE, no deep/sequential models, no threshold-cut relitigating (4.0 stays). All retired with evidence.
5. **Winner's-curse discipline (the core rule of this pack):**
   - The candidate list in P2 is FROZEN. Do not add configs mid-run, do not report any single-seed max.
   - Selection = highest mean paired ΔAUC vs baseline C1, averaged over CV seeds and over weeks {2,4,8} (early-warning weighting; pre-registered here).
   - Only the P3 confirmatory nested number is quotable. The P2 sweep is internal.
   - Log every config evaluated in the results JSON — no silent drops.
6. Feature computation must be leak-free: only events with `created_at <= course_start(target course) + cutoff_weeks` enter ANY feature, including cross-course features (the cutoff clock is the TARGET course's). Per-fold feature selection stays per-fold.
7. After 3 failed attempts on a task: `BLOCKED: <reason>` in TIER2_PROGRESS.md, move on. Long runs in background bash with logs. Serialize heavy jobs (16 cores; the Tier-1 session measured thrash when two Boruta-heavy jobs run concurrently — run ONE at a time).
8. Work on branch `sota-tier2` (from `sota-tier1`). Keep `TIER2_PROGRESS.md` with one verifier-stamped entry per task.

## Key facts (trust; do not rediscover)

- Universe: 7 courses `[54503, 54529, 55010, 55183, 55410, 54570, 54581]`, 560 pairs, 41 fails, target `grade < 4.0` from `puc_grades_clean.parquet`.
- Clean parquet: 1,767,329 rows, **20 courses / 714 students** — the extra 13 courses are the raw material for cross-course features. Columns include `created_at` (UTC), `created_at_local`, `hour_local`, `dow_local`, `normalized_url`.
- Zero-fail augmentation courses: `[53493, 54947, 56867]` (+167 pairs, 0 fails, coverage-100%) — grades for them are in `puc_grades_clean.parquet`.
- Machinery: `puc_benchmark_sota.py` (`sota_feature_selection(return_ranked=True)`, `get_course_starts`, `filter_by_cutoff`, `calculate_all_features`, `calculate_znorm`, `get_optuna_search_space`), `puc_features_clean.py` (T2 wrapper: local-time aliasing + 560-row alignment — REUSE `build_week_matrix`), `puc_ab_clean.py` (`oof_for_arm`, `metrics`, paired bootstrap), `puc_nested_cv.py` (nested protocol), `puc_catboost_zoo.py` (CatBoost/HistGB configs + venv), `puc_promising_explore.py` (recall@capacity + shared-ranking pattern).
- CatBoost lives in `.venv-tier1` (`.venv-tier1/bin/python`); system python is externally-managed.
- Baselines to anchor against (clean data, LOCO, seed 42): calibrated prod XGB = 0.756/0.794/0.815/0.841/0.776 (T3); nested tuned XGB = 0.772/0.812/0.785/0.848/0.767 (T5); untuned CatBoost N40 = 0.829/0.854/0.829 at wk4/8/full (promising_explore, uncalibrated single-seed — treat as noisy).
- Session gap stays 30 min (T8: empirically justified).

---

## TASKS

### P0 — Setup
Branch `sota-tier2`; create `data/puc/sota_results/tier2_push/`; init `TIER2_PROGRESS.md`.
**Verifier**: branch + dir exist; PROGRESS initialized.

### P1 — Features v2: thesis families on clean data (`scripts/puc_features_v2.py`)
Compute NEW feature families per (student, course) pair, per cutoff week ∈ {2,4,6,8,full}, and append them to the T2 clean matrices (do not recompute the T2 base — load `week_{w}_clean.parquet` and join). All time filtering uses the TARGET course's `course_start` (percentile 0.05, same as T2) and `created_at <= start + cutoff`.

**Family A — Cross-course context** (uses ALL 20 courses of `puc_clean_data.parquet`; this is the novel signal):
1. `xc_total_views`, `xc_total_sessions` — student's whole-LMS activity within the target course's cutoff window (sessions = 30-min gap over the student's ALL-course event stream).
2. `xc_course_share_views` = target-course views / whole-LMS views (thesis `Course_to_Weekly` concentration analog); same for sessions.
3. `xc_n_active_other_courses` — other courses with ≥5 events in window.
4. `xc_sessions_between_course` — mean/max count of other-course sessions occurring between consecutive target-course sessions (thesis loyalty/inactivity family).
5. `xc_relative_neglect` — target-course share minus the student's mean share across their active courses.

**Family B — Intensity** (within target course): weekly views/sessions vs the student's own running mean: `intensity_max_dev`, `intensity_std_dev`, `intensity_last_week_dev`.

**Family C — Workload slope**: weekly-views deltas split by sign: `slope_pos_sum`, `slope_neg_sum`, `slope_pos_count`, `slope_neg_count`, `slope_ratio`.

**Family D — Peaks**: `n_local_peaks` (local maxima in weekly series), `weeks_above_25/50/100pct` of own mean.

**Family E — Composites**: `procrastination_x_regularity` (existing proactivity index × session_regularity, computed from the base matrix columns; if exact names differ, pick the closest existing pair and document), `pdh_entropy`/`pwd_entropy` (entropy of pooled hour/weekday histograms — near-dupes of existing entropies are fine; correlation filter in selection will handle them).

Output: `tier2_push/features/week_{w}_v2.parquet` = T2 clean matrix + new columns (z-norm the new columns per course, same as T2 policy: znorm ADDED alongside raw). Also `tier2_push/features_v2_report.json` (per week: n new features, NaN policy = fill 0 after znorm, summary stats).
**Verifier (all must hold)**: each week matrix has exactly 560 rows, same `student_id/course_id/_y/_group` as T2 clean (assert equality); ≥25 new base features; leak spot-check — pick 3 (student, course, week-2) cells and recompute `xc_total_views` directly from the raw parquet with an independent snippet: values must match exactly; NaN rate of new columns ≤ 5% pre-fill.

### P2 — Pre-registered bake-off (`scripts/puc_bakeoff_v2.py`)
**FROZEN candidate list** (uncalibrated — AUC and recall@capacity are rank metrics; calibration is applied only to the P3 winner):

| ID | Model | N feats | Features |
|----|-------|---------|----------|
| C1 | XGB prod (spw) | 40 | T2 clean (baseline) |
| C2 | CatBoost default (Balanced) | 40 | T2 clean |
| C3 | CatBoost default | 30 | T2 clean |
| C4 | HistGB (balanced) | 40 | T2 clean |
| C5 | rank-avg(C1,C2,C4) | 40 | T2 clean |
| C6 | rank-avg(XGB,CB,HGB) | 30 | T2 clean |
| C7 | XGB prod | 40 | **v2** |
| C8 | CatBoost default | 40 | **v2** |
| C9 | CatBoost default | 30 | **v2** |
| C10 | rank-avg(XGB,CB,HGB) | 40 | **v2** |

rank-avg = average of the per-model OOF probability *ranks* (scipy.stats.rankdata per fold's test block, then mean across models).
Protocol: weeks {2,4,6,8,full} × CV seeds {42..46} (StratifiedGroupKFold(5, shuffle, seed), groups=course). Per (week, seed, fold): compute the composite feature ranking ONCE per feature-set (T2-clean and v2 separately), slice top-30/top-40 — reuse across all models (the `puc_promising_explore.py` pattern). Metrics per config: ROC-AUC, PR-AUC, recall@{10,15,20}% flag rate, F2max; plus per-seed paired ΔAUC vs C1.
Output: `tier2_push/bakeoff_results.json` (every cell) + printed summary table of mean±sd ΔAUC.
**Selection rule (pre-registered)**: winner = highest mean ΔAUC vs C1 averaged over seeds and weeks {2,4,8}. Tie-break (<0.003 apart): higher mean recall@20% over the same cells; then fewer features; then simpler (single model over ensemble).
**Verifier**: JSON contains 10 configs × 5 weeks × 5 seeds; C1 seed-42 AUCs within ±0.02 of the uncalibrated analog of T3 (sanity anchor — use `puc_promising_explore.py` xgb_N40 values: 0.814/0.834/0.784 at wk4/8/full); winner + rule application recorded in the JSON.

### P3 — Confirmatory run of the winner (`scripts/puc_confirmatory_v2.py`)
The ONLY quotable numbers. Winner config → per week {2,4,6,8,full}:
1. **Nested CV**: outer LOCO 5-fold (seed 42); per outer-train: feature ranking + (if winner is CatBoost or ensemble-with-CatBoost) inner 3-fold Optuna 30 trials on the CatBoost member (space: depth 4–8, lr log 0.01–0.3, l2_leaf_reg 1–10, iterations 100–500; F2 objective) → fit on outer-train → predict outer-test.
2. **5-seed bagging**: within each outer fold, fit the (tuned) model with model seeds {42..46} and average probabilities (fold structure unchanged).
3. **Platt calibration** (`CalibratedClassifierCV(sigmoid, cv=3)`) wrapped around the final per-fold model for the probability-quality metrics; report AUC both raw-bagged and calibrated (should be ≈equal).
4. Report: ROC-AUC + PR-AUC with bootstrap CI (B=2000, seed-42 RNG, same style as T5), Brier, ECE, and the **capacity curve**: recall at flag rates {5,10,15,20,25}% .
Output: `tier2_push/confirmatory_results.json`.
**Verifier**: JSON complete for 5 weeks; nested ≤ bake-off values expected (flag if nested exceeds the bake-off mean by >0.02 → possible leak, STOP and record); CI arrays present; capacity curve monotone non-decreasing in flag rate.

### P4 — Train-only augmentation ablation (`scripts/puc_augment_ablation.py`)
Winner config, weeks {2,4,8}: rebuild TRAIN folds with the 3 zero-fail courses' students appended as negatives (their features computed by the same v2/T2 pipeline; they never appear in test folds — test = the same 560-pair LOCO folds as P3, assert identical test indices). Paired comparison vs P3-without-augmentation on the same folds.
**Verifier**: test-fold indices identical to P3 (assert); paired ΔAUC + CI reported per week; augmented train sizes logged (+~167 per full-data fold, less at early cutoffs).

### P5 — `TIER2_RESULTS.md` (repo root)
Executive table: per-week honest confirmatory AUC (+CI) vs Tier-1 nested XGB vs the old optimistic reference; capacity-curve table; bake-off summary (all 10 configs, mean±sd); augmentation verdict; **UA section**: two-arm honest range per week + per-arm winners + pre_assessment feature verdict (with the KEEP caveat everywhere); explicit section **"Números para el documento de Enrique"** with the exact per-week PUC values and the recall@20% line, PLUS a one-paragraph honest-provenance note (clean data, LOCO, nested, calibrated). End with open items.
**Verifier**: file exists with the tables, the UA range section, and the Enrique section.

### UA-1 — UA feature completion (`scripts/ua_features_v2.py`)
Assemble the full UA feature matrix per cutoff {2,4,6,8,full}: the existing enriched features (as loaded by `train_time_limited_model.load_features`, include_znorm=True) **plus** `data/enriched_features/pre_assessment_features.parquet` (34 features, join on user_id/course_id; for temporal cutoffs join the same full-data file and DOCUMENT that these features are full-horizon — if that leaks future info relative to the cutoff, compute a cutoff-limited variant only if cheap, else include them ONLY in the `full` cutoff and note it). Optional (if <30 min): `xc_course_share_views` within the 10-course parquet. Fix in-pipeline: rebuild `jaccard_to_passing`'s passing set with `<57` (the 57-vs-60 inconsistency) if graph features are recomputed; otherwise reuse and note.
**Verifier**: per-cutoff matrix row counts match the enrollment arms after inner-join; pre_assessment columns present; a leak-handling note exists in the report JSON.

### UA-2 — UA mini bake-off (`scripts/ua_bakeoff.py`)
**FROZEN candidates** (uncalibrated): U1 XGB old-params (baseline, current pipeline FS), U2 XGB + composite top-40 per-fold FS (PUC-style `sota_feature_selection`), U3 CatBoost default (Balanced) top-40, U4 rank-avg(U2,U3,HistGB) top-40, U5 = U3 with pre_assessment features included vs excluded (isolates their value; full cutoff only if leak-limited per UA-1).
Protocol: **2 label arms (KEEP 373 / DROP-A 322)** × weeks {2,4,8,full} × CV seeds {42..46}; report BOTH StratifiedKFold(5) and StratifiedGroupKFold(5, groups=course). Metrics: ROC-AUC, PR-AUC, recall@{10,20}%, F2max; paired ΔAUC vs U1 within arm.
**Selection rule (pre-registered)**: per arm independently, highest mean paired ΔAUC vs U1 over seeds × weeks {2,4,8}, stratified CV primary (matches historical UA reporting; LOCO reported alongside). **Never compare configs across arms** — a model better at true prediction can score worse against the KEEP arm's contaminated labels.
**Verifier**: JSON has 5 configs × 2 arms × 4 weeks × 5 seeds × 2 CV schemes; U1 seed-42 within ±0.02 of the T4 re-run anchors (KEEP-full stratified ≈0.892; DROP-A-full ≈0.850); per-arm winners recorded.

### UA-3 — UA confirmatory + honest range (`scripts/ua_confirmatory.py`)
Per arm: winner → nested CV (inner Optuna 30 trials if CatBoost member), 5-seed bagging, Platt calibration, bootstrap CIs, capacity curve {10,15,20,25}%. A+ (286) sensitivity row: winner config, single seed-42 run, one line.
Output: `tier2_push/ua_confirmatory.json` + a results section reporting the **UA honest range**: "wkX AUC = [DROP-A value] – [KEEP value]" with the label caveat attached to the KEEP end.
**Hard guardrail**: the KEEP-arm number must NEVER appear in any output without its caveat sentence ("target = recorded Canvas outcome; includes 51 active-zero enrollments whose true grades are external"). The DROP-A number is the quotable-alone one.
**Verifier**: JSON complete for both arms + sensitivity row; every KEEP mention in TIER2_RESULTS.md carries the caveat; nested ≤ bake-off mean +0.02 (leak flag otherwise).

### P6 — HTML numbers prep (GATED — do not edit the HTML)
Produce `tier2_push/html_update_proposal.md`: the drop-in per-week numbers for `~/projects/tm3-roi-diagnostico/metricas-tecnicas-udla.html` (PUC-only, neutral descriptive tone per Paul's standing direction; remove the "Cada afirmación" section concept), showing OLD table vs NEW proposed table side by side. **STOP there — Paul decides which numbers ship.**
**Verifier**: proposal file exists; HTML untouched (git status clean on that repo).

## Success / abort criteria (pre-registered)

- **Success**: confirmatory nested wk4 ≥ 0.83 AND wk8 ≥ 0.86 (stretch 0.88), with recall@20% ≥ 0.65 at wk8. If met, the Enrique curve is fully honest at old-optimistic levels.
- **Partial**: winner beats C1 (mean paired ΔAUC > +0.01) but misses targets → still adopt; report actuals; the HTML decision goes to Paul with real numbers.
- **Null result**: no config beats C1 by > +0.005 mean → report honestly, keep Tier-1 numbers, do NOT torture the data further. A null here is informative, not a failure.
- **UA success**: per-arm winner beats U1 by mean paired ΔAUC ≥ +0.01; report the honest range (DROP-A is the quotable-alone number). UA never headlines over PUC unless DROP-A confirmatory exceeds the PUC confirmatory at the same week — if that happens, flag it prominently rather than deciding placement.
- Compute budget ≈ 5–8 h total (P2 rankings dominate; UA track ≈ 1–1.5 h — run AFTER the PUC track completes; serialize everything).

## Task order
P0 → P1 → P2 → P3 → P4 → UA-1 → UA-2 → UA-3 → P5 → P6. PUC first: it feeds the Enrique document. UA runs while P5 drafting can begin if convenient, but never concurrently with another Boruta-heavy job.

## When done
All tasks DONE/BLOCKED → stop. Adoption into the register, HTML edits, and anything sales-facing is Paul's + a Fable session's call.
