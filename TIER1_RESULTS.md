# TIER-1 SOTA RESULTS — PUC clean rebuild + UA label remediation
**Executed by Opus 4.8 · 2026-07-03 · branch `sota-tier1`**
Ground truth: `TIER1_EXECUTION.md` (recipes/verifiers) + `EXPERIMENT_REGISTER.md` (evidence). Per-task verifier log: `TIER1_PROGRESS.md`.
All compute local CPU, `RANDOM_STATE=42`, identical folds per A/B (StratifiedGroupKFold(5, shuffle, seed 42), groups=course_id). Authoritative artifacts (`benchmark_results.json`, backups, existing parquets) untouched; all outputs are new files under `data/puc/sota_results/tier1_clean/` and `data/ua_remediated/`.

## Status: 10/10 tasks DONE (T0–T9), 0 BLOCKED
No task hit BLOCKED-FOR-REVIEW (T3) or a fold-leakage flag (T5). **PUC decision: ADOPT clean data as canonical.**

---

## 1. PUC — old vs clean data (T3, THE key measurement)

Production config (calibrated XGBoost + `scale_pos_weight=neg/pos`, top-40 leak-free per-fold selection, LOCO 5-fold, Platt sigmoid) run twice per week on **identical 560-row folds** (n=560 pairs, 41 fails). ΔAUC = clean − old, paired bootstrap B=2000 (shared indices).

| Week | AUC old→clean | PR-AUC old→clean | Brier old→clean | ECE old→clean | ΔAUC [CI95] |
|------|---------------|------------------|-----------------|---------------|-------------|
| 2    | 0.743 → **0.756** | 0.317 → 0.270 | 0.063 → 0.061 | 0.029 → 0.017 | +0.013 [−0.043, 0.064] |
| 4    | 0.795 → 0.794 | 0.280 → 0.261 | 0.061 → 0.062 | 0.025 → **0.009** | −0.002 [−0.043, 0.040] |
| 6    | 0.787 → **0.815** | 0.260 → 0.277 | 0.063 → 0.061 | 0.016 → 0.016 | +0.029 [−0.039, 0.100] |
| 8    | 0.815 → **0.841** | 0.334 → **0.396** | 0.058 → 0.057 | 0.015 → 0.015 | +0.026 [−0.012, 0.063] |
| full | 0.780 → 0.776 | 0.386 → 0.382 | 0.057 → 0.057 | 0.015 → 0.018 | −0.003 [−0.058, 0.051] |

**Decision = ADOPT clean as canonical.** No week shows significant degradation (no ΔAUC CI upper bound < −0.03). All ΔAUC CIs straddle 0 (differences insignificant at n=560/41 fails), but clean data trends **better** at weeks 6 & 8 and improves calibration (ECE) — consistent with "cleaning reveals signal". Adoption is justified by correctness-by-construction (dedup + local timezone), not by a significant point gain.

### Row-drop accounting (T1, no silent drops)
Input **2,315,314** → L1 exact-dup −5,721 → 2,309,593 → L2 HTML/api-twin −492,494 → 1,817,099 → L3 rapid same-URL (<10s) −49,770 → **1,767,329** (−547,985, 23.7%). Monotone ✓, idempotent ✓ (2nd pass removes 0). Output: `data/puc/puc_clean_data.parquet`.

### Timezone correction (T1)
Per-row UTC→America/Santiago offset is exactly **3h (444,144 rows) or 4h (1,323,185 rows)** — 100% of rows, DST boundary captured. Night-hour (0–6h) share drops **17.55% (UTC) → 4.37% (local)**, correcting the "23:00 local miscounted as madrugada" artifact that biased the whole time-of-day feature family. (The literal modal-hour verifier proxy misfired — flat midday UTC distribution — so the per-row offset is the definitive evidence; see `TIER1_PROGRESS.md` T1.)

### `interaction_seconds` unreliability audit (T1) — confirms the exclusion
0% null, **0% exact-zero**, **3.32% > 1800s**, median 3.41s, with suspicious repeated spikes (11.0s ×5,182; 12.0s ×2,819). Consistent with Canvas's imprecise heartbeat → remains **excluded as a time-active signal** (used only to quantify its own unreliability). Full audit in `tier1_clean/cleaning_report.json`.

---

## 2. PUC — honest headline numbers under nested CV (T5)

Outer LOCO 5-fold; per outer-train: top-40 selection → inner 3-fold Optuna (30 trials, F2) → tuned+calibrated fit → predict outer-test. "Non-nested (clean)" tunes once globally (optimistic). "Reference" = previously-cited non-nested benchmark.

| Week | **Nested AUC (honest)** | Nested CI95 | Nested PR-AUC | Non-nested (clean) | Reference (old, optimistic) |
|------|------------------------|-------------|---------------|--------------------|-----------------------------|
| 2    | **0.772** | [0.686, 0.846] | 0.277 | 0.757 | 0.831 |
| 4    | **0.812** | [0.741, 0.871] | 0.280 | 0.808 | 0.872 |
| 6    | **0.785** | [0.699, 0.869] | 0.276 | 0.783 | 0.863 |
| 8    | **0.848** | [0.777, 0.908] | 0.384 | 0.852 | 0.863 |
| full | **0.767** | [0.669, 0.846] | 0.382 | 0.790 | 0.854 |

No fold-leakage flag (nested never exceeds non-nested by >0.02). **The honest nested AUCs sit ~0.05–0.09 below the previously-cited reference at early weeks** — the reference was non-nested and optimistic. Week 8 is the strongest honest cut (0.848). These nested numbers are what should be published as "reported under nested cross-validation".

---

## 3. PUC — CatBoost + HistGB added to the zoo (T6)

Clean data, production protocol, same folds as XGBoost; CatBoost 30-trial Optuna at weeks 4 & 8. Paired ΔAUC vs XGBoost (shared bootstrap indices).

| Week | XGBoost | **CatBoost** | HistGB | ΔAUC CatBoost−XGB [CI95] |
|------|---------|--------------|--------|--------------------------|
| 2    | 0.756 | 0.779 | 0.757 | +0.024 [−0.022, 0.075] |
| 4    | 0.794 | 0.821 | 0.773 | +0.027 [−0.012, 0.072] |
| 6    | 0.815 | 0.827 | 0.793 | +0.012 [−0.022, 0.046] |
| 8    | 0.841 | 0.842 | **0.854** | +0.001 [−0.037, 0.039] |
| full | 0.776 | **0.826** | 0.801 | **+0.051 [0.010, 0.100]** |

**Verdict: CatBoost BEATS XGBoost** (mean ΔAUC +0.023) but **significant only at full week**; a tie at week 8 (where HistGB actually leads). **CatBoost is a strong adoption candidate — NOT switched in here** (guardrail #3 pins production to XGBoost+Platt; model adoption is deferred to the review session).

---

## 4. Explainability — SHAP activated (T7)

Production XGBoost (uncalibrated booster for TreeSHAP), clean data, weeks 4 & 8. Artifacts in `tier1_clean/`:
- `shap_week{4,8}_summary.png`, `shap_week{4,8}_global_importance.json` (top-20 mean|SHAP|), `shap_week{4,8}_per_student.csv` (560 rows: `student_id, course_id, risk_score, factor1-3 + effect + shap`, plain-language ES names).

Top global drivers — **quiz views dominate** (mean|SHAP| 0.735 wk4 / 0.847 wk8), then weekly trend, DCT spectral (relative-to-course), quiz proactivity, session count (relative-to-course). Per-student profiles are deployment-ready (Shapley-value explanations per student).

---

## 5. Session-timeout sensitivity (T8)

Inter-click gap median **0.067 min (~4 s)**; 12.35% of gaps ≥30 min (session boundaries). Week-4 production AUC by timeout: **15 min = 0.790, 30 min = 0.794, 60 min = 0.768**. **30-min is empirically justified** — it is the top performer, tied with 15-min (Δ=0.003); only 60-min degrades. (Sanity: gap=30 recompute reproduces T3's week-4 clean AUC 0.7935 exactly.) `tier1_clean/timeout_sensitivity.json`.

---

## 6. UA — label remediation A+ (T4)

Remediation A+ = drop the **51** LMS-active zero-score enrollments (≥20 views, `final_score==0`, external-LTI-gradebook artifact) **AND** all of course **86676** (partial gradebook). Clean set **n=286, fails(<57)=73, prevalence 25.5%** (verifier-exact). Features are label-independent, so existing enriched features were reused unchanged; old(373) vs new-A+(286) run under identical eval code (max-F1 operating point). 57-vs-60 threshold inconsistency **unified to <57**.

| Week | LOCO AUC old→new-A+ | Stratified AUC old→new-A+ | F1 (LOCO) old→new |
|------|---------------------|---------------------------|-------------------|
| 2    | 0.689 → 0.586 | 0.743 → 0.740 | 0.618 → 0.441 |
| 4    | 0.664 → 0.629 | 0.742 → 0.727 | 0.614 → 0.418 |
| 6    | 0.684 → 0.540 | 0.746 → 0.742 | 0.634 → 0.385 |
| 8    | 0.736 → 0.628 | 0.828 → 0.760 | 0.661 → 0.457 |
| full | 0.737 → 0.723 | **0.892 → 0.788** | 0.641 → 0.520 |

**The honest UA numbers are LOWER than the contaminated ones**, but a decomposition (`scripts/ua_verify_decomposition.py`) shows **the drop is driven by removing the 51 active-zeros, NOT by dropping course 86676** (correcting an earlier over-attribution to 86676). Full-week stratified ROC-AUC by label set: OLD 0.892 → drop-active-zeros-only **0.850** → drop-86676-only **0.889** (≈neutral) → A+ (both) 0.788. Under LOCO, dropping 86676 actually *helps* (0.737→0.769; its banded grades don't generalize across courses). Interpretation: the old 0.89 earned **spurious credit** for correctly ranking mislabeled active students as "fails" (verified: active students, median 84–131 views, `final_score=0` — an external-LTI artifact, corroborated by course 84941's boxplot median sitting at 0 because 20/38 of its students are active-zeros). Removing that spurious credit is correct even though the number falls. The drop persists **with and without** assessment features (full: −0.10 with, −0.13 without), so it is not a feature-filtering artifact — the with-assessment arm includes all 57 evaluation-activity features. Honest full-week AUC ≈ **0.79 (stratified) / 0.72 (LOCO)**. Whether to also drop 86676 is a genuine judgment call (barely affects AUC; mainly changes prevalence/n) — deferred to you. Outputs: `data/ua_remediated/student_enrollments_clean.csv`, `ua_clean_results.json`, `scripts/ua_verify_decomposition.py`. Definitive fix remains option C (official UA acta grades).

---

## 7. NUMBERS THAT MUST CHANGE IN SALES MATERIALS
*(Do NOT edit sales materials here — this section flags what a later Fable session must re-derive. Sources: this repo's `tier1_clean/` and `ua_remediated/` JSONs.)*

**UA (highest urgency — current materials are materially wrong):**
1. **Prevalence 39–40% → 25.5–30.4%** (n 373→286 under A+, or 322 if only active-zeros are dropped). The old prevalence counted 51 mislabeled active-zeros as fails.
2. **UA AUC must drop**: full-week ~0.89 → **~0.79 (stratified) / ~0.72 (LOCO)**; week-8 ~0.83 → ~0.76. The drop is driven by the active-zeros (verified mislabels), not by 86676. Any UA "0.74–0.90" headline is inflated by contaminated labels.
3. Course **86676**: its gradebook is banded/ceiling-capped at 82.2 (4 clusters near multiples of 20 — consistent with a partial/completion-based gradebook), but dropping it barely changes AUC. Decision to keep or drop it is open (see §6); at minimum stop citing it as a *clean* "good-variance" case.

**PUC (adopt honest numbers; direction is favorable):**
4. Publish **nested-CV** AUCs as the honest headline: wk2 0.77 · wk4 0.81 · wk6 0.79 · **wk8 0.85** · full 0.77 — replacing the optimistic non-nested 0.83/0.87/0.86/0.86/0.85.
5. PUC pipeline is now **clean-from-raw-click** (3-level dedup + America/Santiago timezone) — the defensible claim "aligned with C&E / JLA from click to prediction, reported under nested CV, with Shapley per-student explanations" is now earned.
6. Optional upgrade pending review: **CatBoost** (mean +0.023 AUC, +0.051 significant at full week) as the production model.

---

## 8. Open items (deferred to the review/judgment session)
- **Adoption decisions (explicitly out of scope here):** promote clean PUC data to canonical in `EXPERIMENT_REGISTER.md`; decide XGBoost→CatBoost switch (re-run SHAP on CatBoost if adopted); re-derive `tm3-roi-diagnostico` sales materials from these JSONs.
- **UA option C:** request official UA acta grades (only Canvas keys existed; no external grades) — the permanent fix; A+ is the best honest dataset available today.
- **Residual in T4:** the `jaccard_to_passing` graph feature was computed with the old `>=60` passing set (1 of ~100 features, reused as-is); rebuild it under `<57` if UA features are ever recomputed.
- **UA features left on the table:** `data/enriched_features/pre_assessment_features.parquet` (34 pre-deadline preparation-behavior features — activity 24/48/72h before deadlines, quiz/assignment access timing, `preparation_intensity`, `late_surge_ratio`) exists but is **never loaded** by `train_time_limited_model.load_features()`. Absent from both old and new UA arms (so not the cause of the drop), but a real feature-coverage gap worth wiring in and re-benchmarking.
- **UA 86676 decision:** verified banded/ceiling-82.2 gradebook; AUC-neutral to drop. Keep-vs-drop is a labeling-trust call for the UA team, not a modeling one.
- **Not re-run (Tier-2):** auth-session validation of the 30-min timeout, URL→resource match-rate, course-start unification, thesis inactivity/intensity/slope features. All documented in the register.
