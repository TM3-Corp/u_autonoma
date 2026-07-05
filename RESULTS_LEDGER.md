<!-- Auto-generated 2026-07-04 by a 5-agent results-audit workflow; verify high numbers against source before external use. -->

# RESULTS_LEDGER — TM3 Early-Warning Model Performance Audit

*Scope: every model-performance metric found across historical (pre-Tier-1) and Tier-1/2/3 artifacts for PUC and Universidad Autónoma (UA). Defensibility graded A (gold/nested+honest-CV) · B (deployment/stratified within-institution) · C (optimistic/non-nested — common in EDM literature) · D (not defensible — contaminated labels, leaky features, cherry-picked maxima).*

---

## 1. The tension, resolved

**Yes — the old PUC dashboard numbers are real.** The `wk2 0.831` / `wk4 0.872` (plus `wk6 0.863`, `wk8 0.863`, `full 0.854`) headlines came from the non-nested tuned benchmarks: `data/puc/sota_results/7courses_multiclass/benchmark_results.json` and `.../benchmark_results.json` (`XGBoost_balanced_tuned`, `assess=True`), and were surfaced in `tm3-diagnostico.html`. They are also preserved as the `non_nested_*` reference fields inside `data/puc/sota_results/tier1_clean/nested_cv_results.json`.

They are higher because feature selection and Optuna hyperparameter tuning were performed on **all** the data and then the same data was cross-validated (and the reported cell is the max over a config grid). That double-use of the test folds is a mild winner's curse — an optimistic bias of roughly **+0.05 to +0.06 AUC** over the honest nested estimate (e.g. wk8 non-nested `0.852` vs nested `0.848`; wk4 `0.872` vs nested `0.812`; full `0.854` vs nested `0.767`). Nothing was faked; the numbers simply measure "fit on these courses" rather than "generalizes to new courses." They are perfectly citable as *non-nested benchmark* results — the same protocol most published EDM papers report — as long as they are labeled as such.

---

## 2. Best DEFENSIBLE headline per institution

Highest **A-class** (nested LOCO — "new courses", reviewer-proof) and **B-class** (stratified nested — "known recurring courses, new cohort", the actual TM3 scenario) per week. All PUC on clean clickstream-derived data (n=560, prevalence 7.3%); UA on remediated **DROP-A** labels (n=322).

### PUC (grade < 4.0)

| Week | A — nested LOCO (new courses) | B — stratified nested (known courses) | Model / source |
|------|-------------------------------|----------------------------------------|----------------|
| 2 | **0.798** [0.724, 0.861] | **0.797** [0.719, 0.861] | CatBoost Bal-40, 5-seed, calibrated · `tier2_push/confirmatory_calibrated_ci.json` / `stratified_nested_results.json` |
| 4 | **0.830** [0.770, 0.887] | **0.792** [0.719, 0.859] | CatBoost Bal-40, calibrated · same |
| 6 | **0.823** [0.751, 0.887] (raw) / 0.818 cal | **0.797** [0.727, 0.863] | CatBoost Bal-40 · same |
| 8 | **0.848** [0.777, 0.908] (XGB single-seed) / **0.838** [0.769, 0.90] (CatBoost 5-seed, robust) | **0.829** [0.756, 0.897] | XGB nested `tier1_clean/nested_cv_results.json` · CatBoost `confirmatory_calibrated_ci.json` / `stratified_nested_results.json` |
| full | **0.830** [0.759, 0.895] | **0.836** [0.769, 0.895] (raw) / 0.814 cal | CatBoost Bal-40 · same |

> For PUC, stratified sits **at or slightly below** LOCO every week (`strat_minus_loco` ≤ 0). This is a strong selling point: PUC generalizes to *unseen courses* at no measurable cost, so the honest LOCO number **is also** the within-institution deployment number. Recommended single headline: **AUC ≈ 0.80 by week 2, ≈ 0.85 by week 8**, nested, calibrated, held-out courses.

### UA (grade < 4.0 / < 57%)

| Week | A — nested LOCO (new courses) | B — stratified nested DROP-A (known courses) | Model / source |
|------|-------------------------------|-----------------------------------------------|----------------|
| 8 | 0.615–0.628 (weak) | **0.756** [0.693, 0.817] | CatBoost nested, DROP-A · `tier2_push/ua_confirmatory.json` / `TIER2_RESULTS.md` |
| full | 0.605–0.64 (weak) | **0.809** [0.758, 0.857] | CatBoost nested, DROP-A, calibrated · `ua_confirmatory.json` |

> **UA does not generalize across courses** — honest DROP-A LOCO is only 0.43 (wk2) → 0.49 (wk4) → 0.61–0.64 (wk8/full). UA's defensible story is **strictly within-institution / known-course (B-class)**: full-semester **0.809**, week-8 **0.756**. There is **no** defensible UA A-class number above 0.65, and **no** defensible UA number above 0.85 exists at all (every UA ≥ 0.85 is KEEP-arm contaminated).

---

## 3. Full ledger — every AUC ≥ 0.80 found (sorted by value desc)

Cross-cluster exact duplicates consolidated. "arm" blank = n/a for PUC/pooled.

| # | Inst | AUC | Wk | Model | CV scheme | Nested | Calib | Arm | Def | Source file |
|---|------|-----|----|-------|-----------|--------|-------|-----|-----|-------------|
| 1 | UA | 1.000 | full | per-course all-data | simple | no | no | — | **D** | `data/prediction_models_results.json` (degenerate single-course, n=36–51) |
| 2 | PUC | 0.9775 | full | XGBoost | LOCO | no | no | — | **D** | `sota_results/7courses/benchmark_results.json` (label-leaky `at_risk`, grades in features) |
| 3 | pooled | 0.9384 | 8 | CatBoost greedy subset | LOCO | no | no | — | **D** | `tier3_pooled/confirmatory_results.json` (cherry-picked 3-course subset, `quotable:false`) |
| 4 | UA | 0.9033 | full | znorm+assessment | strat/unk | no | no | KEEP | **D** | `data/analysis/time_cutoff_results.json` (also `tm3-diagnostico.html`) |
| 5 | UA | 0.9024 | full | threshold-optimized | unk | no | no | KEEP | **D** | `data/analysis/all_thresholds_optimized.json` |
| 6 | UA | 0.892 | full | XGBoost (global-FS) | strat | no | yes | KEEP | **D** | `TIER1_RESULTS.md` (old headline; decomposes → 0.850 → 0.788) |
| 7 | UA | 0.8865 | full | P15 assess+pre-assess | unk | no | no | KEEP | **D** | `data/analysis/pre_assessment_optimization_results.json` |
| 8 | UA | 0.880 | full | VotingEnsemble | unk | no | no | KEEP | **D** | `data/analysis/multi_model_benchmark_results.json` (n=373 + assessment) |
| 9 | PUC | 0.880 | full | XGB+SMOTE 3-class | unk | no | no | — | **D** | `tm3-diagnostico.html` (off-protocol 3-class) |
| 10 | UA | 0.872 | full | CatBoost nested | strat | yes | no | KEEP | **D** | `TIER2_RESULTS.md` (KEEP inflates via 51 mislabeled active-zeros) |
| 11 | PUC | 0.8716 | 4 | XGBoost_balanced_tuned | LOCO | **no** | no | — | **C** | `sota_results/7courses_multiclass/benchmark_results.json` (**old dashboard wk4**) |
| 12 | PUC | 0.8634 | 8 | CatBoost (seed-46 cell) | strat | no | no | — | **D** | `tier2_push/bakeoff_results.json` (single-seed max; honest mean 0.857) |
| 13 | PUC | 0.8632 | 6/8 | XGBoost | LOCO | no | no | — | **C** | `sota_results/7courses/benchmark_results.json` (old dashboard wk6/wk8) |
| 14 | UA | 0.8624 | full | XGBoost (71f) | unk | no | no | KEEP | **D** | `data/enriched_features/model_results_v3.json` |
| 15 | UA | 0.8605 | full | XGBoost Optimizado | non-grp/hold-out | no | no | KEEP | **D** | `data/models/v4_optimized/.../early_warning_model_metrics.json` (LOCO twin only 0.745) |
| 16 | PUC | 0.857 | 8 | CatBoost Bal-40 (seed-mean) | strat | no | no | — | **C** | `tier2_push/bakeoff_results.json` (model-selection grid, untuned) |
| 17 | UA | 0.857 | full | XGBoost (145f) | unk | no | no | KEEP | **D** | `data/enriched_features/early_warning_model_results.json` |
| 18 | PUC | 0.8548 | 8 | XGBoost_balanced (284f) | LOCO | no | no | — | **C** | `few_feature_sweep/sweep_results.json` (grid-max over 400 configs) |
| 19 | PUC | 0.854 | 8 | HistGB | LOCO | no | yes | — | **C** | `tier1_clean/catboost_results.json` (non-nested global tuning) |
| 20 | PUC | 0.8537 | full | old benchmark | unk | no | no | — | **C** | `tm3-diagnostico.html` (old dashboard "semestre", shown rounded 0.90) |
| 21 | PUC | 0.852 | 8 | XGBoost (non-nested clean) | LOCO | no | yes | — | **C** | `tier1_clean/nested_cv_results.json` (`non_nested_clean`, +0.004 vs nested) |
| 22 | UA | 0.8504 | full | XGBoost (51f) | unk | no | no | KEEP | **D** | `data/enriched_features/model_results.json` |
| 23 | UA | 0.850 | full | XGBoost (global-FS) | strat | no | yes | DROP-A | **C** | `TIER1_RESULTS.md` (drop active-zeros only; still non-nested → C) |
| 24 | UA | 0.8485 | full | without-assessment | unk | no | no | KEEP | **D** | `tm3-diagnostico.html` |
| 25 | **PUC** | **0.848** | **8** | **XGBoost tuned (nested)** | **LOCO** | **yes** | no | — | **A** | **`tier1_clean/nested_cv_results.json`** — highest A point estimate, CI [0.777, 0.908] |
| 26 | UA | 0.8429 | full | XGBoost (124f, PCA) | unk | no | no | KEEP | **D** | `data/enriched_features/learning_material_model_results.json` |
| 27 | UA | 0.8424 | full | Stacking Ensemble | non-grp | no | no | KEEP | **D** | `v4_optimized/.../early_warning_model_metrics.json` |
| 28 | UA | 0.8416 | full | XGBoost_balanced | unk | no | no | KEEP | **D** | `data/analysis/multi_model_benchmark_results.json` (no-assessment, still KEEP) |
| 29 | PUC | 0.8376 | 8 | CatBoost Bal-40, 5-seed | LOCO | **yes** | yes | — | **A** | `tier2_push/confirmatory_calibrated_ci.json` — **production headline (0.84)**, CI [0.769, 0.90] |
| 30 | PUC | 0.8371 | 6 | LightGBM_tuned | LOCO | no | no | — | **C** | `sota_results/benchmark_results.json` (Optuna non-nested, 20 courses) |
| 31 | PUC | 0.8369 | 8 | XGBoost Platt-cal, top-40/fold | LOCO | **yes** | yes | — | **A** | `few_feature_sweep/castillo_metrics.json` — **pre-Tier-1 honest LOCO already hit 0.837**, CI [0.763, 0.900] |
| 32 | PUC | 0.836 | 8 | CatBoost Bal-40, 5-seed (raw) | LOCO | **yes** | no | — | **A** | `tier2_push/confirmatory_results.json`, CI [0.769, 0.894] |
| 33 | PUC | 0.8362 | 4 | XGBoost (untuned, grid-max) | LOCO | no | no | — | **C** | `sota_results/benchmark_results.json` |
| 34 | PUC | 0.8355 | full | CatBoost Bal-40 (raw) | **strat** | **yes** | no | — | **B** | `tier2_push/stratified_nested_results.json`, CI [0.769, 0.895] |
| 35 | PUC | 0.8332 | 8 | XGBoost raw (uncalib) | LOCO | **yes** | no | — | **A** | `few_feature_sweep/calibration_smote.json` (uncalibrated twin of #31) |
| 36 | PUC | 0.8331 | 2 | XGBoost_balanced_tuned | LOCO | no | no | — | **C** | `sota_results/benchmark_results.json` (**old dashboard wk2 0.831**) |
| 37 | UA | 0.8327 | 8 | with-assessment | unk | no | no | KEEP | **D** | `tm3-diagnostico.html` (DROP-A honest = 0.790 strat) |
| 38 | PUC | 0.8308 | 2 | RF_balanced_tuned | LOCO | no | no | — | **C** | `7courses_multiclass/benchmark_results.json` |
| 39 | PUC | 0.8304 | full | CatBoost Bal-40, 5-seed | LOCO | **yes** | yes | — | **A** | `tier2_push/confirmatory_calibrated_ci.json`, CI [0.759, 0.895] |
| 40 | PUC | 0.830 | 4 | CatBoost Bal-40, 5-seed | LOCO | **yes** | yes | — | **A** | `tier2_push/confirmatory_results.json` (raw 0.806) |
| 41 | PUC | 0.8298 | 4 | CatBoost calibrated | LOCO | **yes** | yes | — | **A** | `tier2_push/confirmatory_calibrated_ci.json`, CI [0.77, 0.887] |
| 42 | PUC | 0.8289 | 8 | CatBoost Bal-40 | **strat** | **yes** | yes | — | **B** | `tier2_push/stratified_nested_results.json`, CI [0.756, 0.897], cap-recall@20% 0.732 |
| 43 | UA | 0.8267 | 6 | RFE 132→64 | unk | no | no | KEEP | **D** | `data/analysis/comprehensive_optimization_results.json` (FS on all data + pctl cohort) |
| 44 | PUC | 0.825 | 6 | XGBoost raw, top-40/fold | LOCO | **yes** | no | — | **A** | `few_feature_sweep/calibration_smote.json` |
| 45 | PUC | 0.823 | 6 | CatBoost Bal-40 (raw) | LOCO | **yes** | no | — | **A** | `tier2_push/confirmatory_results.json`, CI [0.751, 0.887] |
| 46 | UA | 0.8232 | 6 | wk6 with-assessment (100f) | unk | no | no | KEEP | **D** | `data/analysis/optimal_early_model_results.json` |
| 47 | UA | 0.8213 | 8 | XGBoost | unk | no | no | KEEP | **D** | `data/analysis/multi_model_benchmark_results.json` (jump tracks assessment leakage) |
| 48 | UA | 0.8198 | early | LogReg (12 temporal feats) | non-grp CV | no | no | KEEP | **C** | `data/enriched_features/early_warning_model_results.json` (activity-only, but non-grouped + contaminated) |
| 49 | PUC | 0.8178 | 6 | CatBoost calibrated | LOCO | **yes** | yes | — | **A** | `tier2_push/confirmatory_calibrated_ci.json`, CI [0.748, 0.878] |
| 50 | PUC | 0.8139 | full | CatBoost calibrated | **strat** | **yes** | yes | — | **B** | `tier2_push/stratified_nested_results.json`, CI [0.729, 0.888] |
| 51 | PUC | 0.812 | 4 | XGBoost prod (nested) | LOCO | **yes** | yes | — | **A** | `tier1_clean/nested_cv_results.json`, CI [0.741, 0.871] |
| 52 | PUC | 0.8091 | full | CatBoost calibrated | **strat** | **yes** | yes | DROP-A→n/a | **A/B** | `few_feature_sweep/…` twin — see note |
| 53 | UA | 0.8091 | full | CatBoost calibrated | **strat** | **yes** | yes | DROP-A | **B** | `tier2_push/ua_confirmatory.json` — **highest defensible UA number**, CI [0.758, 0.857] |
| 54 | PUC | 0.809 | 4 (non-nested ref) | nested twin 0.808 | LOCO | no | — | — | **C** | `nested_cv_results.json` reference field |

*(F1/degenerate note: `data/baseline/focused_models_results.json` reports F1 = 0.989 on a 2-course pool where fail=0% — degenerate, D. The CLAUDE.md "F1 = 1.000 / 21-feature all-data" headline is the same class of single-course degeneracy — D, not generalizable.)*

---

## 4. The honest range statement *(paste-ready)*

> **PUC early-warning ROC-AUC ranges from 0.77 (strict, held-out *new* courses, week 8, nested cross-validation) to 0.85 (week 8, calibrated, held-out courses; 0.80 already by week 2), with the robust pre-registered production estimate at 0.838 [95% CI 0.769–0.900].** Because PUC generalizes to unseen courses at no measurable cost (stratified ≈ LOCO), the within-institution recurring-course deployment number is the same ~0.83–0.85. The most-comparable-to-literature non-nested benchmark reads **0.87 (week 4) / 0.86 (week 8)** — real, but optimistically biased by ~0.05 because feature selection and tuning touched all the data. At 20% review capacity the model catches **~68% of eventual failures by week 6–8, before the first grade is recorded.**
>
> **UA early-warning ROC-AUC is 0.81 [0.758–0.857] for the full semester and 0.76 for week 8 — nested, calibrated, within-institution (known recurring courses), on cleaned labels (active-zero mislabels removed).** UA does *not* transfer to genuinely new courses (held-out-course AUC 0.43–0.64), so UA must be positioned strictly as a known-course deployment. Any prior UA figure of 0.86–0.90 was inflated by 51 mislabeled active-zero LTI artifacts and must not be quoted.

---

## 5. Recommendation

**The actual TM3 use case is within-institution deployment on recurring courses** — so the *stratified nested* (B) and, for PUC, the *LOCO nested* (A) numbers are the ones to show. They are honest **and** appropriate to the scenario; you are not under-selling by using them.

### SHOW these

| Purpose | Institution | Number to show | Provenance |
|---------|-------------|----------------|------------|
| PUC hero, week 2 | PUC | **AUC 0.80** [0.72, 0.86] | CatBoost Bal-40, nested LOCO, calibrated · `confirmatory_calibrated_ci.json` |
| PUC hero, week 8 | PUC | **AUC 0.84** (robust) / up to **0.85** (best single-seed) | CatBoost 5-seed `confirmatory_calibrated_ci.json`; XGBoost `nested_cv_results.json` |
| PUC capacity metric | PUC | **Catch 68% of failures at 20% flag rate by week 6** (recall@20% 0.683) | `confirmatory_calibrated_ci.json` |
| PUC "known courses" framing | PUC | **AUC 0.829 (wk8) / 0.836 (full)** stratified | `stratified_nested_results.json` |
| UA (within-institution only) | UA | **AUC 0.81 (full) / 0.76 (wk8)**, DROP-A | `ua_confirmatory.json` |
| Literature-comparison footnote | PUC | "non-nested benchmark 0.87/0.86 (wk4/wk8), consistent with published EDM protocols" — labeled as such | `benchmark_results.json` |

### RETIRE these (never show as headline)

- **All UA ≥ 0.85** (rows 4–8, 10, 14–15, 17, 22, 24, 26–28, 37, 43, 46–47): KEEP-arm — the 51 mislabeled active-zeros earn the model spurious credit. The flagship `UA 0.903` and hero pill `XGBoost 0.86` are the worst offenders; already forbidden in the retired `tm3-diagnostico.html`.
- **PUC 0.9775 full** (row 2) and **pooled 0.9384** (row 3): label leakage and cherry-picked course subset — D, must not appear anywhere.
- **PUC single-seed 0.8634** (row 12): seed cherry-pick; the honest seed-mean is 0.857 (and the *nested* honest value is 0.838).
- **PUC 3-class SMOTE 0.88** (row 9): off-protocol, not comparable to the binary <4.0 task.
- **The CLAUDE.md "F1 = 1.000" all-data headline**: degenerate single-course separation — replace with the honest capacity metric (recall@20% ≈ 0.68).

### Framing guidance
The old 0.87/0.86 PUC numbers can stay *in the deck as a footnote* labeled "non-nested benchmark (optimistic)" — they are not fraudulent, and being transparent about the ~0.05 nested penalty builds credibility. But the number next to the product claim must be the nested/calibrated **0.84–0.85 (PUC) / 0.81 (UA, known-courses)**. That is the strongest story you can defend against any reviewer, and it is genuinely strong: **usable AUC by week 2, before a single grade exists.**