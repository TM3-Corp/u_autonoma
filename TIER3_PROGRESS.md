# TIER-3 SOTA Execution Progress — cross-institution pooling + course-eligibility

Branch: `sota-tier3` (from `sota-tier2`) · RANDOM_STATE=42 · CV seeds {42,43,44,45,46} · Started 2026-07-03
Ground truth: `TIER3_EXECUTION.md` (recipes/verifiers). Context: `TIER2_RESULTS.md`, `TIER1_RESULTS.md`.
One entry per task: timestamp · what ran · verifier output · PASS/FAIL/BLOCKED.

Guardrails honored: UA labels DROP-A arm only (drop 51 active-zeros, keep 86676); shared features use only signals computable identically at both institutions, per-course z-normed, institution-classifier probe ≤0.75 AUC; leak-free (cutoff by target course start pct 0.05, per-fold selection); no SMOTE; never modify authoritative files (`benchmark_results.json`, backups, `few_feature_sweep/`, existing parquets, `tier1_clean/`, `tier2_push/` except NEW files); nothing sales-facing edited. R2-pooled is the pre-committed primary headline; only G6 confirmatory numbers are quotable; every R3 artifact carries `"quotable": false`. Serialize Boruta/Optuna-heavy jobs. 3 strikes → BLOCKED.

Task order: G0 → G1 → G2 → G3 → G4 → G5 → G6 → G7.

## FINAL — all 8 tasks DONE (G0–G7), 0 BLOCKED (2026-07-03)
- **Outcome = pre-registered NULL.** Pooling PUC+UA on institution-invariant features does NOT beat single-institution models. R2-pooled nested wk8 raw-bagged **0.713** [0.652,0.771] < 0.85 target; ≤ Tier-2 PUC-only 0.836 and R2 UA-only 0.725. Two positives: (1) real modest cross-institution transfer — **train-UA→test-PUC 0.70–0.74 vs pristine actas** (< 0.80 target, useful as cold-start prior); (2) standalone predictability map (failure rate governs learnability; r=−0.39, −0.73 within PUC; cohort size irrelevant).
- **Root cause of the null (honest):** guardrail-2 invariance forced dropping 39/62 features (early-week institution probe 0.98–0.997) → only 23 shared behavioral features; the portable cross-institution signal is real but thinner than each institution's own feature set.
- **Guardrails honored (audited)**: UA labels DROP-A only (51 active-zeros dropped, 86676 kept, R-rules handle it); shared features computable identically at both institutions, per-course z-normed, institution probe ≤0.75 all weeks (0.59–0.64); leak-free (cutoff by target-course start pct 0.05, per-fold FS, Optuna/FS train-only — nested exceedance characterized as tuning/bagging, not leakage); RANDOM_STATE=42, seeds{42..46}; no SMOTE; R2-pooled the pre-committed primary; only G6 quotable; all R3 artifacts `quotable:false`. **Authoritative files untouched** (`benchmark_results.json`, backups, `few_feature_sweep/`, existing parquets, `tier1_clean/`, `tier2_push/` — only NEW files added, all under `tier3_pooled/` + `ua_clean/`); **sales material untouched** (`tm3-diagnostico.html` md5 `9744d05…` = Tier-2B record, byte-identical). Serialized all heavy jobs (G4→G5→G6 sequential).
- **Deliverables**: `TIER3_RESULTS.md` (6 sections) + `TIER3_PROGRESS.md`; `tier3_pooled/` JSONs (g2_build_report, feature_schema, category_mapping, course_profiles, stageA_results, stageB_results, confirmatory_results, ua_cleaning_report) + 5 pooled feature parquets + 5 OOF parquets + logs; `data/ua_clean/ua_clean_data.parquet`; 7 new scripts on `sota-tier3` (uncommitted — no commit requested).
- **Not done (correctly, per scope)**: no register updates, no HTML/sales edits, no adoption — Paul's + a Fable session's call. Keep Tier-2B page as-is (null → no change warranted).

Frozen eligibility rule membership (from verified inventory, characteristics only — never AUC):
- PUC courses (7): 54503, 54529, 55010, 55183, 55410, 54570, 54581 — 560 pairs, 41 fails.
- UA DROP-A courses (10): 79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390 — 322 pairs, 98 fails.
- R0 = all 17. R1 (fails≥4 ∧ n≥15): PUC {55010,54529,55410,54570}, UA {all but 84936} = 13. R2 (prev∈[8%,50%] ∧ fails≥4 ∧ n≥15): PUC {55410,54570}, UA {89390,88381,84941,89099,79913,79875,86020,84944} = 10.

---

## G7 — TIER3_RESULTS.md — PASS (2026-07-03)
Wrote `TIER3_RESULTS.md` (repo root) with all six required sections: (i) headline table — R2-pooled nested per week + CIs (quotable, traces to G6 exactly) vs Tier-2 PUC-only + leave-institution-out transfer with train-UA→test-PUC (pristine-actas) row called out; (ii) predictability analysis — per-course AUC vs characteristics correlations + plain-language findings incl. surprises (higher prevalence → harder; cohort size irrelevant); (iii) course-eligibility guide (institution-facing R2 rule + what to tell universities); (iv) R0/R1 Stage-A context rows; (v) R3 max-map summary marked internal/`quotable:false` with what it teaches (max-AUC picks low-prevalence courses = selection-bias trap R2 avoids); (vi) open items. Plus §6 transparent leak-flag characterization.
- **Verifier (all PASS)**: file has all six sections (grep-confirmed §1–§7) ✓; every quotable headline number traces to `confirmatory_results.json` exactly (asserted: wk2 0.690/0.627 … wk8 0.713/0.663 … full 0.708/0.682) ✓; R3 numbers carry the internal/non-quotable flag (JSON `quotable:false`, §5 header + body) ✓. PASS.

---

## G6 — Confirmatory (the ONLY quotable numbers) — PASS (2026-07-03)
Script `scripts/g6_confirmatory.py` (via `.venv-tier1`, background) → `tier3_pooled/confirmatory_results.json` + `oof_pooled_week_{2,4,6,8,full}.parquet` + log. Winner `cat_full` (CatBoost Balanced, corr-prefilter). Protocol: nested LOCO5 outer seed42, inner 3-fold Optuna **150-trial** F2, 5-seed bagging, Platt sigmoid, bootstrap CI B=2000. Ran ~2100s (35 min).
- **Nested R2-pooled raw-bagged AUC [CI95] / cal / PR / rec20**: wk2 **0.690** [0.628,0.748]/0.627/0.406/0.363 · wk4 **0.616** [0.544,0.685]/0.584/0.405/0.374 · wk6 **0.679** [0.614,0.742]/0.599/0.416/0.352 · wk8 **0.713** [0.652,0.771]/0.663/0.488/0.385 · full **0.708** [0.644,0.769]/0.682/0.489/0.407. Brier(cal) 0.168/0.172/0.172/0.165/0.160; ECE 0.059/0.049/0.041/0.031/0.040. Capacity curves monotone all weeks. Per-course AUC wk8: 88381 0.97 · 89099 0.80 · 79875 0.78 · 89390 0.76 · 55410 0.75 · 86020 0.72 · 54570 0.72 · 79913 0.68 · 84941 0.64 · 84944 0.58.
- **Leave-institution-out** (tuned on train side): **train-UA→test-PUC (pristine actas)** wk2 0.708 · wk4 **0.741** · wk6 0.719 · wk8 0.700 · full 0.727 [CIs ~±0.13]. **train-PUC→test-UA** wk2 0.665 · wk4 0.661 · wk6 0.628 · wk8 0.695 · full 0.686. Asymmetry as predicted (pristine-label UA→PUC direction is the stronger one, ~0.72; PUC→UA noisier). A real but modest cross-institution transfer signal.
- **R3 max-map (INTERNAL, `"quotable": false`)**: greedy forward → peak pooled LOCO AUC **0.938 on 3 courses {84936(UA), 54503(PUC), 55010(PUC)}** — all *low-prevalence* (5–6%, 11 fails total). Confirms max-AUC course selection picks rare-failure courses (opposite of the R2 balanced rule); teaches that a high "max performance" number is an artifact of easy low-prevalence subsets, not generalizable skill. Flagged non-quotable.
- **Leak flags (wk2,6,8,full) — characterized as NON-leakage** (transparent, à la Tier-2): the guard compares each week's nested (Optuna-tuned + 5-seed-bagged, seed-42 folds) to the winner's *cross-week* Stage-B mean (0.633, dragged down by weak wk4). On **matched seed-42 folds**, tuning+bagging legitimately lifts untuned-single-model → tuned-bagged (wk8 0.687→0.713, +0.026); per-fold FS (corr-prefilter) and Optuna run strictly on train folds (code-verified) → no test-label leakage. For the two weeks Stage-B measured: wk4 nested 0.616 vs 0.602 (+0.013, clean); wk8 nested 0.713 vs 0.664 (+0.049, tuning/bagging/partition, not leak). AUCs are modest regardless.
- **Verifier (all PASS)**: complete for 5 weeks + both transfer directions ✓; capacity monotone every week ✓; R3 flagged non-quotable ✓; leak flags recorded + characterized (nested exceedance attributable to tuning/bagging on matched folds, FS/tuning train-only) ✓; OOF parquets persisted (5) ✓. PASS.
- **Success-criteria read**: R2-pooled nested wk8 raw **0.713 < 0.85** target; pooling ≤ single-institution (Tier-2 PUC-only wk8 0.836; R2 UA-only Stage-A wk8 0.725); train-UA→test-PUC **0.70 (wk8) / 0.74 (wk4) < 0.80** target. Per pre-registration ⇒ **NULL** (pooling does not beat single-institution at the institution-invariant feature granularity), with two positives: a real modest cross-institution transfer (~0.72 vs pristine labels) and the standalone predictability map. Report honestly; keep Tier-2B page as-is.

---

## G5 — Stage B: model × features on R2-pooled — PASS (2026-07-03)
Script `scripts/g5_stage_b.py` (via `.venv-tier1`, background) → `tier3_pooled/stageB_results.json` + log. R2-pooled only (10 courses, 400 pairs, 91 fails). {CatBoost,XGB} × N{20,30,40,full-after-corr-prefilter} × seeds{42..46} × weeks{4,8}; shared per-(seed,fold) ExtraTrees rankings sliced per N; uncalibrated. Ran ~179s.
- **Aggregate mean AUC (weeks×seeds)**: cat_full **0.6331** (rec20 0.363) · cat_N30 0.6294 · cat_N40 0.6294 · cat_N20 0.6292 · xgb_full 0.6098 · xgb_N20 0.6077 · xgb_N30/N40 0.6069. CatBoost sweeps XGB at every config (consistent with Tier-1/2). 
- **N-grid collapse (logged honestly)**: with only 23 institution-invariant features, N∈{30,40} = all features (identical AUC); only N20 truly subsets and cat_full uses the corr-prefilter set (~21 features).
- **WINNER = cat_full** (CatBoost Balanced, corr-prefilter feature set). Highest mean AUC 0.6331; tie set = {cat_full} alone (next config cat_N30 0.6294 is 0.0037 below, outside the 0.003 tie threshold) → no tie-break needed.
- **Verifier (all PASS)**: 8 configs × 2 weeks × 5 seeds = 80 config-cells present ✓; winner + selection-rule application logged in JSON ✓. PASS.

---

## G4 — Stage A: course-set × mix grid — PASS (2026-07-03)
Script `scripts/g4_stage_a.py` (via `.venv-tier1`, background) → `tier3_pooled/stageA_results.json` + log `logs/g4_stage_a.log`. Reference config CatBoost Balanced top-40/fold uncalibrated, weeks{4,8} × rules{R0,R1,R2} × mixes{pooled,PUC,UA} × seeds{42..46}, LOCO grouped by course. Ran ~435s.
- **wk8 (mean pooled-OOF AUC / mean per-course AUC / recall@20%)**:
  - R0: pooled **0.673**/0.760/0.417 · PUC **0.800**/0.805/0.673 · UA 0.657/0.769/0.349
  - R1: pooled 0.650/0.722/0.377 · PUC 0.797/0.809/0.618 · UA 0.677/0.750/0.319
  - R2: **pooled 0.659/0.730/0.356** · PUC **SKIP (2 courses <4)** · UA **0.725**/0.762/0.369
- **wk4**: R2 pooled 0.600/0.664/0.363 · R2 UA 0.622/0.663/0.349.
- **Headline finding (pre-registered NULL direction)**: pooling PUC+UA does NOT beat single-institution. PUC-only pooled-OOF AUC (0.80 wk8) ≫ pooled (0.66–0.67) ≈ UA-only (0.66–0.73) across all rules. For the primary R2 set, pooled 0.659 < UA-only 0.725 at wk8 (R2/PUC-only is unmeasurable — only 2 courses). Mean-per-course AUC is closer (pooled 0.73 vs UA 0.76) — within-course ranking is fine, but the cross-institution *pooled* ranking suffers because the 23 institution-invariant features don't put the two institutions on one risk scale. This is the honest consequence of guardrail-2 (invariant-only features).
- **Verifier (all PASS)**: every non-skipped cell × 5 seeds present (18 cells, 2 skipped) ✓; skips justified by course count (R2/PUC = 2 < 4 for LOCO, both weeks) ✓; R2-pooled row complete ✓.

---

## G3 — Course profile table — PASS (2026-07-03)
Scripts `scripts/tier3_common.py` (shared harness: rule membership, per-fold ExtraTrees FS, CatBoost/XGB factory, grouped-LOCO OOF, metrics) + `scripts/g3_course_profiles.py` → `tier3_pooled/course_profiles.json` + `course_profiles.md`. Ran ~5s.
- **Rule membership derived from characteristics (asserted vs frozen)**: R1=13, R2=10 courses ✓. R2 pooled: 400 pairs, 91 fails, 22.8% prevalence (matches pack ~400/~90/~23%). 86676 excluded from R2 by the 50% cap (69.4% prev); low-prevalence PUC excluded by the 8% floor.
- **17 profiles** with institution/n/fails/prevalence/events-per-student(median)/sessions-per-student/active-weeks/grade-dist(std,ceiling,zeros on [0,1]-normalized score)/per-course LOCO AUC (CatBoost Balanced top-40, pooled R0, seed 42, week 8).
- **Pooled R0 wk8 LOCO AUC = 0.683** (all 17 courses; dragged by heterogeneity + the 23 invariant features). **Per-course wk8 LOCO AUC ranges 0.60–0.98**: e.g. R2 courses 88381 0.92 · 89099/84941 0.80 · 86020 0.80 · 79875 0.72 · 84944 0.62 · 79913 0.60. High-AUC low-prevalence PUC courses (54503 0.96/3 fails, 55183 0.85/2 fails) are noisy (tiny fail counts). Raw material for G7's predictability analysis + G6's R3 max-map.
- **Verifier (all PASS)**: 17 rows, all fields present ✓; per-course AUC null-rule honored (0 courses <2 fails → 0 nulls) ✓. PASS.

---

## G2 — Shared feature pipeline — PASS (2026-07-03)
Script `scripts/common_features.py` (via `.venv-tier1`) → `tier3_pooled/features/pooled_week_{2,4,6,8,full}.parquet` + `feature_schema.json` + `category_mapping.json` + `g2_build_report.json`. ONE pipeline over both clean clickstreams (`puc_clean_data.parquet` 7 courses; `ua_clean_data.parquet` 10 DROP-A courses). Ran ~50s.
- **Universe**: PUC 560 (41 fails) + UA DROP-A 322 (98 fails, drops 51 active-zeros, keeps 86676) = **882 pairs, 139 fails** each week (zero-filled to full label universe, PUC-consistent).
- **Category taxonomy**: PUC `category` + UA `resource_type` → 10 shared bins {files,assignments,quizzes,discussions,pages,modules,grades,announcements,navigation,other} (PUC external_tools→other; UA home→navigation). Mapping in `category_mapping.json`. **Unmapped-event share: PUC 0.00%, UA 0.00%** (<20% ✓).
- **62 base features** (session 30-min-gap: 9 · category counts+shares: 20 · temporal hour/dow local: 10 · weekly views/sessions/trend/momentum/inactivity: 11 · totals+gaps: 7 · first-access: 5) + 62 per-course z-norm = 128 total cols/parquet. Cutoff = target course start (0.05 quantile) + w weeks; full = all events. Institution kept for grouping/audit, never a feature.
- **Guardrail 2 (institution invariance)**: all-62-znorm probe (HGB, StratGroupKFold5) leaks strongly at early weeks (wk2 0.982, wk4 0.997, wk6 0.863, wk8 0.824, full 0.831) — the two institutions' behavior barely overlaps early. Greedy backward elimination by RandomForest institution-importance **dropped 39 leaking znorm features** (logged in schema `dropped_institution_leakers` + report), leaving **23 institution-invariant model features** (activity level, session shape, temporal shares, weekly dynamics, momentum, front-load). **Final per-week probe: 0.611/0.642/0.634/0.620/0.587 — all ≤0.75 ✓.** Dropped features stay in the parquet for audit; models use `model_feature_cols` only. (This heavy drop foreshadows the pre-registered "null" possibility — institutions may not mix at fine feature granularity; measured in G4+.)
- **Verifier (all PASS)**: row counts 882 (560 PUC + 322 UA per enrollments) every week ✓; identical column set + dtypes across institutions ✓; unmapped <20% each (0.00%) ✓; institution probe ≤0.75 all weeks (0.59–0.64 on the 23 model features) ✓; 3-cell/institution leak spot-check — raw recount ≤ cutoff exact-matches feature `total_events` (e.g. PUC 54581/8457 wk8: 6303==6303) ✓. PASS.

---

## G1 — UA clickstream hygiene — PASS (2026-07-03)
Script `scripts/ua_clean_rebuild.py` (via `.venv-tier1`) → `data/ua_clean/ua_clean_data.parquet` + `tier3_pooled/ua_cleaning_report.json`. Applies the Tier-1 PUC recipe to `categorized_page_views.parquet` (10 model courses). UA has no `url` → used `http_request` as URL surface; `created_at` naive string → parsed UTC; user_id normalized (`%1e10`, matches enrollments/ua_remediate_labels).
- **Row-drop accounting (monotone ✓)**: input 116,436 → L1 exact-dup −272 → 116,164 → L2 HTML/api-twin −8,017 → 108,147 → L3 rapid same-URL (<10s) −2,541 → **105,606** (−10,830, 9.3%). 208 users, 10 courses.
- **L2 applicability probe**: 57,651 `/api/v1` hits in http_request → L2 applicable (not skipped).
- **Timezone**: per-row UTC−local offset ∈ {3 (63,386), 4 (42,220)} = 100% of rows ✓ (America/Santiago, DST captured). Night-hour (0–6h) share 25.08% (UTC) → 5.10% (local) — same night-madrugada correction as PUC Tier-1.
- **Idempotency**: 2nd pass on the output removes 0/0/0 (L1/L2/L3) ✓.
- **Verifier (all PASS)**: monotone counts ✓; idempotent (2nd pass 0) ✓; per-row offset ∈{3,4} 100% ✓; report complete (all sections present) ✓. PASS.

---

## G0 — Setup — PASS (2026-07-03)
- Git branch `sota-tier3` created from `sota-tier2` (`git branch --show-current` → `sota-tier3`).
- Created dirs: `data/puc/sota_results/tier3_pooled/{,features/,logs/}`, `data/ua_clean/`.
- `TIER3_PROGRESS.md` initialized (this file).
- Env: `.venv-tier1/bin/python` 3.12.3, catboost/xgboost/sklearn present, 16 cores.
- Inputs confirmed present: `data/puc/puc_clean_data.parquet` (1,767,329 rows; PUC 7 courses = 1,108,137 rows / 539 students), `data/puc/puc_grades_clean.parquet`, `data/page_views/categorized_page_views.parquet` (1,013,881 rows; UA 10 courses = 116,436 rows / 208 users), `data/page_views/student_enrollments.csv` (373 rows).
- Taxonomy pre-check: PUC `category` ∈ {modules,navigation,files,quizzes,other,assignments,discussions,announcements,external_tools,pages,grades}; UA `resource_type` ∈ {assignments,modules,home,files,quizzes,discussions,pages,other,announcements,grades} — align to shared bins with home→navigation, external_tools→other. UA `/api/v1` twins present (57,651/116,436) → L2 applicable.
- **Verifier**: branch + dirs exist; PROGRESS initialized. PASS.

---
