# TIER-3 EXECUTION PACK — cross-institution pooling + course-eligibility analytics
**Designed by Fable 5 · 2026-07-03 · to be executed by Opus 4.8 in a fresh session**
Prerequisites (DONE): Tier-1/2/2B (`TIER1_RESULTS.md`, `TIER2_RESULTS.md`, `TIER2_PROGRESS.md`, `tier2_push/`). Read those first. Branch `sota-tier3` (from `sota-tier2`). Outputs → `data/puc/sota_results/tier3_pooled/` + `data/ua_clean/` (new dirs).

## Mission

Pool PUC + UA into one cross-institution training corpus (per-course z-normalized shared features; binary target already unified: PUC `grade<4.0` ≡ UA `final_score<57`), select courses by **pre-registered characteristics rules** (never by measured AUC), and measure: (a) the honest headline on the eligible pooled set, (b) **leave-institution-out transfer** (the "generalizes across institutions" claim; train-UA→test-PUC tests against pristine acta labels), (c) the **course-predictability map** — which course characteristics make models work (Paul's future eligibility guide for new universities).

## Verified course inventory (2026-07-03; do not re-derive)

PUC (labels: official actas): 55183 2/99 (2.0%) · 55010 6/117 (5.1%) · 54503 3/51 (5.9%) · 54529 8/131 (6.1%) · 55410 15/124 (12.1%) · 54581 2/16 (12.5%) · 54570 5/22 (22.7%). Total 560/41.
UA DROP-A (labels: Canvas recorded, 51 active-zeros removed; A+ counts shown, 86676 adds ~36/~25 in DROP-A): 84936 2/32 (6.2%) · 89390 4/30 (13.3%) · 88381 4/19 (21.1%) · 84941 4/18 (22.2%) · 89099 8/33 (24.2%) · 79913 11/41 (26.8%) · 79875 7/26 (26.9%) · 86020 17/49 (34.7%) · 84944 16/38 (42.1%) · 86676 ~25/~36 (~72%). Pooled fails ≈ 139.

## Pre-registered eligibility rules (FROZEN — characteristics only, never AUC)

- **R0** — all courses (7 PUC + 10 UA DROP-A incl. 86676). Context row.
- **R1 "evaluable"** — fails ≥ 4 AND n ≥ 15.
- **R2 "balanced" (PRIMARY — the pre-registered headline)** — prevalence ∈ [8%, 50%] AND fails ≥ 4 AND n ≥ 15. Expected members: PUC {55410, 54570} + UA {89390, 88381, 84941, 89099, 79913, 79875, 86020, 84944} ≈ 10 courses, ~400 pairs, ~90 fails, ~23% prevalence. (86676 exits via the 50% cap; the low-prevalence PUC courses exit via the 8% floor. No per-course judgment calls.)
- **R3 "max-map" (INTERNAL, never quotable)** — greedy forward course-subset selection by AUC. Exists to answer "what is the max performance and on which courses" and to feed G7's characteristics analysis. Every artifact it touches carries `"quotable": false`.

Training mixes: pooled (primary) · PUC-only · UA-only (reference rows; skip LOCO where a mix has <4 courses — note it, don't force it).

## Guardrails (inherit Tier-1/2; plus)

1. UA labels: DROP-A arm ONLY (drop the 51 active-zeros; keep 86676 and let R-rules handle it). Never the KEEP arm.
2. The shared feature pipeline may use ONLY signals computable identically at both institutions; z-norm per course everything that carries level; no feature may encode institution identity (verify: an "institution classifier" trained on the final feature matrix must not exceed 0.75 AUC on znormed features — if it does, find and drop the leaking features and log them).
3. Leak rules as always: cutoff by target course's start (percentile 0.05, both institutions); per-fold selection; RANDOM_STATE=42; seeds {42..46} for repeats; identical folds within comparisons.
4. Selection discipline: stages are sequential and FROZEN (below); R2-pooled is the pre-committed primary headline; only G6 confirmatory numbers are quotable. Everything evaluated gets logged.
5. No SMOTE; no new UA data requests; nothing sales-facing edited. TIER3_PROGRESS.md verifier-stamped per task. 3 strikes → BLOCKED. Serialize Boruta-heavy jobs.

---

## TASKS

### G0 — Setup
Branch `sota-tier3`; dirs `data/puc/sota_results/tier3_pooled/`, `data/ua_clean/`; PROGRESS init.
**Verifier**: exist; init entry written.

### G1 — UA clickstream hygiene (`scripts/ua_clean_rebuild.py`)
Apply the Tier-1 recipe to `data/page_views/categorized_page_views.parquet` (10 model courses; normalize user ids): L1 exact dups; L2 HTML/api-twin ONLY IF the twin pattern exists (probe `http_request` for `/api/v1` twins first — if absent, log "L2 not applicable" and skip); L3 same-URL <10s debounce; `created_at` → tz-aware → `hour_local`/`dow_local` (America/Santiago). Count every drop.
Output: `data/ua_clean/ua_clean_data.parquet` + `tier3_pooled/ua_cleaning_report.json`.
**Verifier**: monotone counts; idempotent (2nd pass removes 0); per-row UTC−local offset ∈ {3,4} for 100% of rows; report complete.

### G2 — Shared feature pipeline (`scripts/common_features.py`) — the long pole
ONE pipeline over both clean clickstreams (`puc_clean_data.parquet` 7 courses; `ua_clean_data.parquet` 10 courses) producing an IDENTICAL schema per (student, course, cutoff∈{2,4,6,8,full}):
- Category taxonomy: build + document an explicit mapping table PUC `category`/`controller` ↔ UA `resource_type`/`controller` → shared bins {files, assignments, quizzes, discussions, pages, modules, grades, announcements, navigation, other}. Log unmapped share per institution (must be <20% of events).
- Families: session (30-min gap: count, duration-by-dwell, regularity, short-session share) · category counts + shares · temporal (hour/dow local: morning/afternoon/evening/night shares, weekend share, hour/day entropies) · weekly (per-week views/sessions, trend slope, momentum, last-week deviation, inactivity gaps) · first-access timing. Target 60–120 base features; then per-course z-norm (znorm added alongside raw).
- Labels: PUC `grade<4.0`; UA DROP-A `final_score<57`. Institution column kept for grouping/audit, NEVER as a feature.
Output: `tier3_pooled/features/pooled_week_{w}.parquet` (one row per pair, both institutions) + `feature_schema.json` + mapping table.
**Verifier**: row counts = 560 + n(DROP-A per its enrollments) each week; identical column set/dtypes across institutions; unmapped-events <20% each; institution-classifier probe on znormed features ≤0.75 AUC (guardrail 2); 3-cell leak spot-check per institution (recount raw ≤ cutoff, exact match).

### G3 — Course profile table (`tier3_pooled/course_profiles.json` + md table)
Per course (17): institution, n, fails, prevalence, events/student (median), sessions/student, active-weeks coverage, grade-distribution stats (std, ceiling share, zeros share), and per-course LOCO AUC under the reference config (CatBoost Balanced, top-40/fold, pooled R0 training, seed 42) — the predictability map's raw material.
**Verifier**: 17 rows, all fields; per-course AUC null where a course has <2 fails (log, don't fake).

### G4 — Stage A: course-set × mix (FROZEN grid)
Reference config (CatBoost Balanced, top-40/fold, uncalibrated) on weeks {4,8}: rules {R0,R1,R2} × mixes {pooled, PUC-only, UA-only} (skip cells with <4 courses for LOCO) × seeds {42..46}, LOCO grouped by course. Report pooled-OOF AUC + mean per-course AUC + recall@20%.
Output: `tier3_pooled/stageA_results.json`.
**Verifier**: every non-skipped cell × 5 seeds present; skips justified by course count; R2-pooled row complete.

### G5 — Stage B: model × features on R2-pooled (FROZEN)
On R2-pooled only (regardless of Stage A ordering — R2-pooled is pre-committed): {CatBoost, XGB} × N ∈ {20, 30, 40, full-after-corr-prefilter} × seeds {42..46}, weeks {4,8}, shared per-(seed,fold) rankings sliced per N.
Selection: highest mean AUC over seeds×weeks; tie (<0.003) → higher recall@20% → fewer features.
Output: `tier3_pooled/stageB_results.json` (+ selected config recorded).
**Verifier**: 8 configs × 2 weeks × 5 seeds; winner + rule application logged.

### G6 — Confirmatory (the ONLY quotable numbers)
Winner config on R2-pooled, all 5 weeks:
1. **Nested LOCO** (outer grouped 5-fold seed 42; inner 3-fold Optuna **150 trials** F2 for the winning model family) + Platt + 5-seed bagging + bootstrap CIs (B=2000) + capacity curve {5..25}% + per-course AUC breakdown + persisted OOF parquets.
2. **Leave-institution-out** (same winner, tuned on train side): train UA(R2)→test PUC(R2) and train PUC(R2)→test UA(R2). Report AUC+CI each direction; expect asymmetry (UA test labels noisier) and say so in the JSON notes.
3. R3 max-map (internal): greedy forward subset by AUC from G3's per-course profiles, reference config, single seed — output marked `"quotable": false`.
Output: `tier3_pooled/confirmatory_results.json`, `oof_pooled_week_{w}.parquet`.
**Verifier**: complete for 5 weeks + both transfer directions; leak rule (nested ≤ Stage-B mean +0.02); capacity monotone; R3 flagged non-quotable.

### G7 — `TIER3_RESULTS.md` (repo root)
(i) Headline table: R2-pooled nested per week + CIs vs Tier-2 PUC-only, + transfer results with the train-UA→test-PUC row called out (pristine-label test). (ii) **Predictability analysis**: per-course AUC vs characteristics (prevalence, activity, n) — correlations + a plain-language findings list, including surprises. (iii) **Course-eligibility guide** (Paul's sales artifact): the R2 rule in institution-facing language + what to tell universities about course requirements. (iv) R0/R1 context rows. (v) R3 max-map summary, marked internal/non-quotable, with what it teaches. (vi) Open items.
**Verifier**: file has all six sections; every quotable number traces to G6; R3 numbers carry the internal flag.

## Success / abort criteria (pre-registered)
- **Success**: R2-pooled nested wk8 ≥ 0.85 with ~90 fails backing it (CI markedly tighter than PUC-only ±0.07), AND train-UA→test-PUC ≥ 0.80 (the transfer claim stands).
- **Partial**: pooling beats the corresponding single-institution rows but misses targets → adopt as the new working corpus, report actuals.
- **Null**: pooling ≤ PUC-only on R2 → institutions don't mix at this feature granularity; report honestly, keep Tier-2B page as-is, the predictability map is still delivered (it has standalone value).
- Compute ≈ 6–9 h (G2 dominates; G6's 150-trial Optuna next). Serialize.

## When done
Stop. Register updates, HTML/sales changes, adoption = Paul + Fable session.
