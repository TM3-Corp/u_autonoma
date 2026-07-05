# TIER-1 SOTA Execution Progress

Branch: `sota-tier1` · RANDOM_STATE=42 · Started 2026-07-03
One entry per task: timestamp · what ran · verifier output · PASS/FAIL.

Guardrails honored: never modify `benchmark_results.json`/backups/`few_feature_sweep/`/existing parquets; new files/dirs only; do not touch `~/projects/tm3-roi-diagnostico/`.

---

## T0 — Setup — PASS (2026-07-03)
- Git branch `sota-tier1` created (from `main`).
- Created dirs: `data/puc/sota_results/tier1_clean/`, `data/ua_remediated/`.
- Baseline references (not copied):
  - Authoritative PUC benchmark: `data/puc/sota_results/7courses_multiclass/benchmark_results.json`
  - Old PUC clickstream: `data/puc/puc_fixed_data.parquet` (2,315,314 rows, UTC ts)
  - PUC grades: `data/puc/puc_grades_clean.parquet` (1,607 rows; modeling universe 7 courses/560 pairs/41 fails)
  - UA enrollments: `data/page_views/student_enrollments.csv`; UA page views: `data/page_views/categorized_page_views.parquet`
- Env: python 3.12, pandas 2.3.1, xgboost 3.0.4, sklearn 1.7.1, optuna 4.5.0, shap 0.48.0. catboost installing (background) for T6.
- **Verifier**: dirs exist ✓; PROGRESS initialized ✓. PASS.

---

## T1 — PUC clean rebuild — PASS (2026-07-03)
Script `scripts/puc_clean_rebuild.py` → `data/puc/puc_clean_data.parquet` (1,767,329 rows) + `tier1_clean/cleaning_report.json`.
Row-drop accounting (monotone ✓): input 2,315,314 → **L1 exact dups −5,721** → 2,309,593 → **L2 HTML/api-twin −492,494** → 1,817,099 → **L3 rapid same-URL <10s −49,770** → 1,767,329 (total −547,985, 23.7%).
- **Verifier 1** (monotone non-increasing, L1>0): PASS (L1=5,721>0).
- **Verifier 2** (idempotent — 2nd pass on own output): PASS (L1=L2=L3=0 removed).
- **Verifier 3** (timezone): the literal modal-hour proxy gave diff=1 (modal UTC 12 vs local 11) — **NOT because the conversion is wrong**, but because the UTC midday distribution is near-flat (hours 12/13/14/15 within ~7%), making the mode unstable. Definitive check substituted: **per-row UTC−local offset is exactly 3h (444,144 rows) or 4h (1,323,185 rows) for 100% of rows** (Chile UTC-4 winter / UTC-3 DST, DST boundary captured). Corroborating: night-hour (0-6h) share drops 17.55%→4.37% — exactly the "23:00 local miscounted as madrugada" artifact the register flagged. Timezone step substantively CORRECT; deviation from the proxy documented.
- **Verifier 4** (report has all fields): PASS.
- `interaction_seconds_audit`: pct_zero=0.0%, pct_gt_1800=3.32%, median=3.41s, top repeated values 11.0s (5,182×), 12.0s, 0.012s → into report (documents the unreliability; not used as a signal).

---

## T4 — UA remediation A+ — PASS (2026-07-03)
Script `scripts/ua_remediate_labels.py` → `data/ua_remediated/student_enrollments_clean.csv` + `ua_clean_results.json`.
Remediation = relabeling (page-view features are label-independent), so existing `data/enriched_features/*` reused unchanged; old(373) vs new-A+(286) run under identical eval code (max-F1 op point; LOCO + stratified CV).
- **Verifier**: clean CSV n=**286**, fails(<57)=**73**, prevalence=**0.2552**, active-zero set=**51** — all exact. PASS. (86676 dropped 40 rows; 51 active-zeros with 4 overlapping 86676 → 373−87=286.)
- 57-vs-60 inconsistency: **unified to <57** in both arms (documented in JSON `threshold_note`); the `>=60` only fed the `jaccard_to_passing` graph feature (1 of ~100), reused as-is — residual noted.
- **KEY FINDING — remediation LOWERS the numbers (old numbers were inflated):**
  - Stratified AUC (with-assessment), old→new-A+: wk2 0.743→0.740, wk4 0.742→0.727, wk6 0.746→0.742, wk8 **0.828→0.760**, full **0.892→0.788**.
  - LOCO AUC (harder, unseen-course) full: 0.737→0.723; wk8 0.736→0.628.
  - Prevalence 40%→**25.5%**. The old full-week 0.89 was inflated by the 51 active-zeros (behaviorally-active students mislabeled as fails = easy cases) + course 86676's partial gradebook.
- This contradicts the register's optimistic "upside asimétrico" hypothesis and confirms its contrarian caveat ("parte del recall venía de abandonos triviales"). **Sales materials must drop UA prevalence 39-40%→25.5% and full-week AUC ~0.89→~0.79 (stratified) / ~0.72 (LOCO).**

---

## T2 — Clean feature rebuild — PASS (2026-07-03)
Script `scripts/puc_features_clean.py` (thin wrapper; reuses `puc_benchmark_sota` machinery, aliases `hour`←`hour_local`/`day_of_week`←`dow_local` for the clean arm — no edit to `puc_benchmark_sota.py`). Builds per-week matrices for BOTH arms aligned to the full 560-pair grade universe (missing early-activity pairs zero-filled), cached to `tier1_clean/features/week_{w}_{old,clean}.parquet`. Report `tier1_clean/feature_build_report.json`.
- **Verifier**: every week n=**560** rows both arms (grades unchanged: 560 pairs / 41 fails) ✓; 284 features each ✓; NaN-rates logged for both arms and within tolerance ✓.
- NaN-check note: the literal "within 20%" is a relative test that misfires when the baseline is ~0 (week 2: old 0.18% vs clean 0.00% — the clean pipeline *reduced* NaN). Criterion made robust: pass if within 20% relative OR ≤2pp absolute. All weeks pass.

---

## T3 — PUC A/B old vs clean (KEY measurement) — PASS · DECISION: ADOPT (2026-07-03)
Script `scripts/puc_ab_clean.py`. Production config (calibrated XGB + spw=neg/pos, top-40 per-fold `return_ranked`, LOCO StratifiedGroupKFold(5), seed 42) run twice/week on IDENTICAL 560-row folds (both arms aligned → same fold + paired-bootstrap indices). Output `tier1_clean/ab_results.json`.
- **Verifier**: 5 weeks × 2 arms × 4 metrics (ROC-AUC/PR-AUC/Brier/ECE) + paired ΔAUC/ΔPR CIs (B=2000, shared indices) + decision field. PASS.
- **Results (old→clean ROC-AUC, ΔAUC [CI95]):** wk2 0.743→0.756 (Δ+0.013 [−0.043,0.064]) · wk4 0.795→0.794 (Δ−0.002 [−0.043,0.040]) · wk6 0.787→0.815 (Δ+0.029 [−0.039,0.100]) · wk8 0.815→0.841 (Δ+0.026 [−0.012,0.063]) · full 0.780→0.776 (Δ−0.003 [−0.058,0.051]).
- **DECISION = ADOPT clean as canonical**: no week's ΔAUC CI upper bound < −0.03 (no significant degradation). Clean data trends *better* at wk6/wk8 (revealing signal, as hypothesized). Not flagged BLOCKED-FOR-REVIEW → T5/T6 cleared to proceed. All Δ CIs straddle 0 (differences insignificant at n=560/41 fails), but the direction + correctness-by-construction justify adoption per the pre-baked rule.

---

## T5 — Nested CV on clean PUC (honest headline) — PASS (2026-07-03)
Script `scripts/puc_nested_cv.py`. Outer LOCO 5-fold; per outer-train: top-40 leak-free selection → inner 3-fold Optuna (30 trials, F2, XGBoost space) → tuned+Platt-calibrated fit → predict outer-test. Non-nested arm = tune once globally (optimistic). Output `tier1_clean/nested_cv_results.json`.
- **Verifier**: JSON complete; nested vs non-nested gap small everywhere; **no leakage flag** (nested never exceeds non-nested by >0.02). PASS.
- **Honest nested ROC-AUC (clean) [+ non-nested / register-reference]:** wk2 **0.772** [0.757 / 0.831] · wk4 **0.812** [0.808 / 0.872] · wk6 **0.785** [0.783 / 0.863] · wk8 **0.848** [0.852 / 0.863] · full **0.767** [0.790 / 0.854].
- Takeaway: the honest nested numbers run **~0.05–0.09 below the previously-cited reference** at early weeks (the reference was non-nested and optimistic). wk8 is the strongest honest week (0.848). These are the numbers to publish as "under nested CV".

---

## T6 — CatBoost + HistGradientBoosting into the zoo — PASS (2026-07-03)
Script `scripts/puc_catboost_zoo.py` (run via `.venv-tier1` — catboost installed there; system python is externally-managed). Clean data, production protocol (LOCO5 + top-40/fold + Platt + seed42); CatBoost 30-trial Optuna at weeks 4 & 8; paired ΔAUC vs XGBoost (shared bootstrap indices). Output `tier1_clean/catboost_results.json`.
- **Verifier**: per-week results for CatBoost + HistGB + paired ΔAUC-vs-XGB CIs + one-line conclusion. PASS.
- **CatBoost vs XGBoost ΔAUC [CI95]:** wk2 +0.024 [−0.022,0.075] · wk4 +0.027 [−0.012,0.072] · wk6 +0.012 [−0.022,0.046] · wk8 +0.001 [−0.037,0.039] · **full +0.051 [0.010,0.100]** ← only significant week.
- **Conclusion: CatBoost BEATS XGBoost** (mean ΔAUC=+0.023), but **significant only at full week**; a tie at wk8 (0.842 vs 0.841, where HistGB actually leads at 0.854). HistGB vs XGB mixed (−0.02 to +0.025).
- Guardrail #3 pins production to XGBoost+Platt and result adoption is deferred to the review session → **CatBoost logged as a strong adoption candidate**, not switched in. T7 SHAP therefore runs on XGBoost (deployable today).

---

## T7 — SHAP activation on winning (production XGBoost) config — PASS (2026-07-03)
Script `scripts/puc_shap_tier1.py`. Weeks 4 & 8, clean data, production XGBoost (spw, top-40 leak-free selection on full data, **uncalibrated booster** for TreeSHAP). Guardrail #3 keeps production = XGBoost (CatBoost adoption deferred to review).
- **Verifier**: all 6 files exist (`shap_week{4,8}_summary.png`, `shap_week{4,8}_global_importance.json` top-20 mean|SHAP|, `shap_week{4,8}_per_student.csv`) ✓; per-student CSV = **560 rows** each ✓; 3-student spot-check shows sane plain-language factors (e.g. "Quizzes vistas", "N sesiones (relativo al curso)", "External tools % (relativo al curso)") with signed effect (aumenta/reduce el riesgo) + risk_score ✓. PASS.
- Top global signal both weeks: **quiz views** (mean|SHAP| 0.735 wk4, 0.847 wk8).

## T8 — Session-timeout sensitivity — PASS (2026-07-03)
Script `scripts/puc_timeout_sensitivity.py`. Inter-click gap histogram (log-scale) on clean data + week-4 production config re-run at gap ∈ {15,30,60} min (monkeypatch `SESSION_GAP_MINUTES`, recompute session features). Output `tier1_clean/timeout_sensitivity.json`.
- **Verifier**: 3 AUCs present (15=0.790, 30=0.794, 60=0.768) ✓; conclusion field ✓. PASS.
- Gap histogram: median 0.067 min (~4 s inter-click), 12.35% of gaps ≥30 min (session boundaries).
- **Conclusion**: 30-min is JUSTIFIED — it's the top performer, tied with 15-min (Δ=0.003), only 60-min degrades (−0.026). Sanity: gap=30 AUC (0.7935) exactly matches T3 week-4 clean AUC → pipeline consistent.

---

## T9 — Consolidated results report — PASS (2026-07-03)
Wrote `TIER1_RESULTS.md` (repo root): PUC old-vs-clean table (AUC/PR-AUC/Brier/ECE per week), nested-CV honest headline, CatBoost/HistGB verdict, SHAP artifact list + top drivers, timeout sensitivity, `interaction_seconds` audit, full row-drop accounting, UA old-vs-A+ table (LOCO+stratified), and the explicit **"Numbers that must change in sales materials"** section + open items.
- **Verifier**: file exists with old-vs-new tables (PUC §1, UA §6) and the sales-impact section (§7). PASS.

---

## FINAL — all 10 tasks DONE, 0 BLOCKED (2026-07-03)
- **Guardrails honored**: `benchmark_results.json` + backups + `few_feature_sweep/` + existing parquets **unmodified** (git-verified clean); all outputs new files under `tier1_clean/` and `ua_remediated/`; `tm3-roi-diagnostico` never touched; RANDOM_STATE=42; identical folds per A/B; no SMOTE; `interaction_seconds` excluded.
- **Headline**: PUC → ADOPT clean data (no degradation, better at wk6/8); honest nested AUCs published; CatBoost a strong (deferred) adoption candidate. UA → honest numbers are LOWER (prevalence 40%→25.5%, full AUC 0.89→~0.79) — sales materials must change.
- **Not done (correctly, per scope)**: result adoption, register verdict updates, sales-material edits — all deferred to the review/Fable session.
- 8 new scripts on branch `sota-tier1` (uncommitted — no commit requested).

---

## POST-HOC VERIFICATION — UA remediation challenged by Paul (2026-07-03)
Paul questioned (a) accuracy of the 86676/label analysis and (b) whether the late-week drop is a feature-filtering artifact. Verified with `scripts/ua_verify_decomposition.py` + direct data inspection:
- **86676 gradebook**: register's claim EXACT — 40 scores in 4 tight bands (14.7–20 ×13, 34.5–43.5 ×11, 52.7–64.2 ×3, 76.2–82.2 ×9), ceiling 82.19. Bands ≈ multiples of 20 → consistent with partial/completion-based gradebook. Paul's boxplot matches this data.
- **Active-zeros**: 51 confirmed (final_score=0, ≥20 views, median 84–131). Concentrated in 84941 (20), 84936 (10), 79875 (6). Course 84941's boxplot median=0 is explained by 20/38 being active-zeros → corroborates contamination.
- **Feature audit**: with-assessment arm includes **all 57 evaluation-activity features** (quizzes/assignments/grades views, `grades_check_per_week`, etc.); NOT filtered. Drop persists with AND without assessment → not a feature bug. Separately: `pre_assessment_features.parquet` (34 feats) is never loaded by the pipeline (pre-existing gap, affects both arms equally).
- **Decomposition (full-week stratified ROC-AUC)**: OLD 0.892 → drop-active-zeros 0.850 → drop-86676 **0.889 (neutral)** → A+ 0.788. Under LOCO, dropping 86676 *helps* (0.737→0.769). **CORRECTION**: the drop is driven by the active-zeros (verified mislabels = spurious credit removed), NOT by 86676 as the earlier writeup implied. `TIER1_RESULTS.md` §6/§7/§8 updated accordingly.
