# TIER-1 SOTA EXECUTION PACK — PUC clean rebuild + UA label remediation
**Designed by Fable 5 · 2026-07-03 · to be executed by Opus 4.8 in a fresh session**
Ground truth for all context: `EXPERIMENT_REGISTER.md` (this repo) — read it FIRST, especially the 2026-07-03 addenda. Memory summary lives in the project memory (v21–v25).

## Mission

Execute the Tier-1 upgrades that turn our pipeline into the defensible "best possible models from this data": clean the PUC clickstream (dedup + timezone + duration model), remediate UA labels (A+), re-measure everything with honest A/B comparisons, add nested CV + CatBoost, and activate SHAP. All compute is local CPU. Every task has a binary verifier — a task is DONE only when its verifier passes and its outputs are written.

## Non-negotiable guardrails

1. **Never modify or delete**: `data/puc/sota_results/7courses_multiclass/benchmark_results.json` (authoritative, 2,640 runs), its `.binary40_backup.json`, anything under `few_feature_sweep/`, or `data/puc/puc_fixed_data.parquet` / `puc_grades_clean.parquet`. All new outputs go to **new files/dirs**.
2. **Do not touch** `~/projects/tm3-roi-diagnostico/` (sales materials) — a later Fable session re-derives those from your results.
3. No SMOTE (retired with evidence). Production config = XGBoost + `scale_pos_weight` + Platt calibration (`CalibratedClassifierCV(method='sigmoid', cv=3)`).
4. `RANDOM_STATE = 42` everywhere; identical folds for any A/B (StratifiedGroupKFold(5, shuffle=True, random_state=42), groups=course_id).
5. Do NOT trust `interaction_seconds` as a signal (user-verified unreliable). It appears only in T1's *quantification of its own unreliability*.
6. Every dropped row must be counted and reported. No silent drops.
7. After 3 failed attempts on a task: write `BLOCKED: <reason>` to PROGRESS and move to the next independent task. Do not loop.
8. If the repo is a git repo, work on branch `sota-tier1`. Long runs via background bash with logs under the session scratchpad.
9. Use `/exec-lite` to scaffold this batch if available (hash-chained PROGRESS.jsonl + verifier capture); otherwise keep a plain `TIER1_PROGRESS.md` appending one entry per task: timestamp, what ran, verifier output, PASS/FAIL.

## Key facts you must not rediscover (from the register — trust these)

- PUC universe: 7 courses `[54503, 54529, 55010, 55183, 55410, 54570, 54581]`, 560 student-course pairs, 41 fails, target = official Nota Final < 4.0 from `puc_grades_clean.parquet`. Pairing verified solid.
- PUC clickstream: `data/puc/puc_fixed_data.parquet` (2.3M rows, 714 students, 20 courses, UTC timestamps, **zero dedup applied**). Thesis-proven recipes below.
- UA: enrollments in `data/page_views/student_enrollments.csv` (373 rows, 10 courses); page views in `data/page_views/categorized_page_views.parquet` with **Canvas GLOBAL user ids** — always normalize `user_id % 10**13` before joining (a `normalize_user_id` exists in `scripts/calculate_features_with_cutoff.py:51`).
- UA labels are contaminated: 51 enrollments are LMS-ACTIVE (≥20 views) with `final_score == 0.0` (external LTI gradebook), and course **86676's entire gradebook is partial** (4 tight score bands, ceiling 82.2). Remediation A+ = drop those 51 AND all of 86676 → expected **n=286, fails=73, prevalence 25.5%** (these exact numbers are your verifier).
- Session standard: 30-min inactivity gap (`SESSION_GAP_MINUTES=30` in `scripts/puc_benchmark_sota.py:99`).
- Reusable machinery in `scripts/puc_benchmark_sota.py`: `calculate_all_features`, `filter_by_cutoff`, `get_course_starts`, `calculate_znorm`, `filter_assessment_features`, `sota_feature_selection(..., return_ranked=True)`, `evaluate_model` (already calibrates: `CALIBRATE_PROBABILITIES=True`), `get_optuna_search_space`, `optuna_tune_model`. Follow the pattern of `scripts/puc_castillo_metrics.py` for LOCO + bootstrap-CI evaluation.

---

## TASKS (in order; T4 is independent and may run in parallel with T1-T3)

### T0 — Setup
Create `data/puc/sota_results/tier1_clean/` and `data/ua_remediated/`. Git branch `sota-tier1` if repo. Record baseline references (paths of old results; do not copy large files).
**Verifier**: dirs exist; PROGRESS initialized.

### T1 — PUC clean rebuild (`scripts/puc_clean_rebuild.py`)
Input `data/puc/puc_fixed_data.parquet` → output `data/puc/puc_clean_data.parquet` + `tier1_clean/cleaning_report.json`.
Apply IN THIS ORDER, counting rows removed at each level:
1. **Dedup L1 — exact duplicate rows**: `drop_duplicates()` over all columns (thesis removed ~18% at raw level; expect less here since this parquet is already course-filtered — any % is fine, just count it).
2. **Dedup L2 — HTML+API twin**: normalize URL by stripping leading `/api/v1` from the path; drop duplicates on `(student_id, course_id, normalized_url, created_at rounded to 1s)` keeping first.
3. **Dedup L3 — rapid same-URL repeats**: within `(student_id, course_id)` sorted by time, drop rows where normalized_url equals the previous row's AND time-delta < 10s (debounce; keep the FIRST of each run — note the thesis comment/code mismatch, we standardize on keep-first).
4. **Timezone**: add `created_at_local = created_at.dt.tz_convert('America/Santiago')`; derive `hour_local`, `dow_local`. Keep UTC column.
5. **interaction_seconds unreliability quantification** (report only): % zeros, % > 1800s, top repeated values, distribution by controller → into the cleaning report as `interaction_seconds_audit`.
**Verifier (all must hold)**: row counts strictly monotone non-increasing across L1→L3 with L1 > 0 removed; running the script twice on its own output removes 0 additional rows (idempotent); the modal `hour_local` differs from modal UTC hour by 3 or 4; report JSON has all fields.

### T2 — Clean feature rebuild
Write `scripts/puc_features_clean.py`: reuse `puc_benchmark_sota` feature machinery but reading `puc_clean_data.parquet` and using `hour_local`/`dow_local` for ALL time-of-day/weekday features (monkeypatch or parameterize — do NOT edit `puc_benchmark_sota.py` defaults; a thin wrapper module that patches the loaded dataframe columns is acceptable). Cutoffs 2/4/6/8/full, percentile 0.05, with-assessment.
**Verifier**: per-week feature matrices computed with n=560 rows each (grades unchanged); shapes and NaN-rates logged and within 20% of old pipeline's (log both).

### T3 — PUC A/B: old vs clean (THE key measurement)
Write `scripts/puc_ab_clean.py` following `puc_castillo_metrics.py`: production config (calibrated XGB + spw=neg/pos, top-40 per-fold `return_ranked` selection, LOCO folds, seed 42) run TWICE per week — once on old features (from `puc_fixed_data.parquet`) and once on clean (T2) — same fold indices. Report per week: ROC-AUC, PR-AUC, Brier, ECE for old & clean + **paired bootstrap CI (B=2000, student-level, same resample indices for both arms) on ΔAUC and ΔPR-AUC**.
Output `tier1_clean/ab_results.json`.
**Decision rule (bake in, then apply)**: ADOPT clean data as canonical if no week shows a *significant degradation* worse than −0.03 AUC (CI upper bound < −0.03). Cleaner data is more correct by construction; small insignificant drops do not block adoption. If a significant large drop appears → mark BLOCKED-FOR-REVIEW (likely a rebuild bug, not a data truth) and stop T5/T6 until reviewed.
**Verifier**: JSON has 5 weeks × 2 arms × 4 metrics + deltas with CIs; decision field populated.

### T4 — UA remediation A+ (independent; parallelizable)
Write `scripts/ua_remediate_labels.py`:
1. Recompute the active-zero set: enrollments × page views (normalize global ids), `final_score==0` AND ≥20 views → expect **51**.
2. Build `data/ua_remediated/student_enrollments_clean.csv` = drop those 51 AND all rows of course 86676.
3. Rebuild weekly features/labels with the existing UA scripts (`calculate_features_with_cutoff.py` pattern) against the clean enrollments; re-run the weekly models (`train_time_limited_model.py` logic; weeks 2/4/6/8/full, with/without assessment) with LOCO-grouped CV if course counts allow (9 courses), else the original 5-fold stratified + report both.
4. Output `data/ua_remediated/ua_clean_results.json`: per-week AUC/F1/precision/recall @máx-F1 + prevalence, old-vs-new side by side.
**Verifier**: clean CSV has exactly n=286, fails(<57)=73, prevalence 0.2552±0.001; results JSON complete. Flag in the report: the 57-vs-60 threshold inconsistency (label `<57` vs "passing" `>=60` in two feature scripts) — unify to 57 in anything you rebuild.

### T5 — Nested CV on clean PUC (honest headline numbers)
Write `scripts/puc_nested_cv.py`: outer = the same LOCO 5-fold; per outer-train: inner 3-fold StratifiedGroupKFold Optuna (30 trials, F2 objective, XGBoost space from `get_optuna_search_space`) → fit tuned+calibrated on outer-train → predict outer-test. Report per week: nested OOF ROC-AUC/PR-AUC (+ bootstrap CI) vs the non-nested tuned numbers (benchmark: 0.831/0.872/0.863/0.863/0.854).
Output `tier1_clean/nested_cv_results.json`.
**Verifier**: JSON complete; expect nested ≤ non-nested (small gap normal); if nested > non-nested by >0.02 anywhere, re-check fold leakage before accepting.

### T6 — CatBoost + HistGradientBoosting into the zoo
`pip install catboost` (user-level OK). Extend a LOCAL copy of the model dict (do not edit `puc_benchmark_sota.py`): CatBoost (`auto_class_weights='Balanced'`, sensible defaults + an Optuna space: depth 4-8, lr, l2_leaf_reg, iterations 100-500) and `HistGradientBoostingClassifier` (+class_weight). Evaluate on clean data, weeks 2/4/6/8/full, production protocol + a 30-trial Optuna pass for CatBoost at weeks 4 and 8. Compare vs XGBoost same-fold with paired CIs.
Output `tier1_clean/catboost_results.json`.
**Verifier**: JSON with per-week results for both models + paired ΔAUC vs XGB; a one-line conclusion field ("beats/matches/loses").

### T7 — SHAP activation on the winning model
Using the best clean-data config (from T3/T5/T6 winner) at weeks 4 and 8: fit on full data (per production config, uncalibrated booster for TreeSHAP), compute TreeSHAP; export `tier1_clean/shap_week{4,8}_summary.png`, `shap_week{4,8}_global_importance.json` (top-20 mean|SHAP|), and `shap_week{4,8}_per_student.csv` (student_id, risk_score, top-3 signed factors with plain-language feature names). Reuse patterns from `scripts/generate_shap_explanations.py`.
**Verifier**: all 6 files exist; per-student CSV has 560 rows; spot-check 3 students for sane factor names.

### T8 — Session-timeout sensitivity (cheap defensibility)
On clean data: inter-click gap histogram (log-scale summary into JSON) + re-run the week-4 production config with session gap ∈ {15, 30, 60} min (only session-derived features change — recompute those). Report AUC per timeout.
Output `tier1_clean/timeout_sensitivity.json`.
**Verifier**: 3 AUCs present; conclusion field states whether 30-min is within noise of alternatives.

### T9 — Consolidated results report
Write `TIER1_RESULTS.md` (repo root): executive table OLD vs CLEAN per week (PUC: AUC/PR-AUC/Brier/ECE; UA: AUC/F1/prevalence old-vs-A+), nested-CV honest numbers, CatBoost verdict, SHAP artifacts list, timeout sensitivity, interaction_seconds audit summary, every row-drop accounting, and an explicit section **"Numbers that must change in sales materials"** (do not change them yourself). End with open items.
**Verifier**: file exists with old-vs-new tables and the sales-impact section.

## Execution notes
- Heavy steps (T2 feature computation ~30-60s/week; T5 nested Optuna is the longest, possibly 1-3h) → background bash + check logs; don't block on them, run T4/T6-prep meanwhile.
- Keep context lean: never print full dataframes; log aggregates. Read `puc_benchmark_sota.py` sections on demand, not whole.
- Subagents: optional; this is mostly sequential CPU. If used, keep them Opus.
- When ALL tasks are DONE/BLOCKED: stop. The review/judgment pass (adopting canonical numbers, updating EXPERIMENT_REGISTER.md verdicts, re-deriving sales materials) belongs to a Fable session — do not attempt it.
