# TIER-2B — `tm3-diagnostico.html` update report (OLD → NEW, with sources)
**2026-07-03 · for Paul's review.** Every number in the updated page traces to a persisted prediction vector (OOF parquet) or a confirmatory JSON. This is the side-by-side to check the metrics.

## Files
| Item | Path | md5 |
|------|------|-----|
| **Updated page** | `~/projects/tm3-roi-diagnostico/tm3-diagnostico.html` | `9744d0529abae55b8747f742477d6c48` |
| **Byte-identical backup (old version)** | `~/projects/tm3-roi-diagnostico/tm3-diagnostico_v1_2026-07-03.bak.html` | `d16680dd14518b2915e0818115e25127` |
| Untouched (verified) | `index.html` | `b357b910c7af77e5e9734d22f6e09cbe` |
| Untouched (verified) | `metricas-tecnicas-udla.html` | `6d671a0e3fccf82ae323cc41088ee5a8` |

Compute sources: `data/puc/sota_results/tier2_push/{confirmatory_calibrated_ci.json, stratified_nested_results.json, ua_confirmatory.json, oof_*_week_*.parquet}` + `tier1_clean/nested_cv_results.json`. Full inventory: `tier2_push/html_number_map.json`. Regenerated interactive payload: `tier2_push/html_window_data.json`.

---

## Core design change
The page's **institution toggle (U. Autónoma / PUC)** became a **CV-scheme toggle**:
- **Cursos nunca vistos (LOCO)** — leave-course-out; the model never saw the course. Source: `confirmatory_calibrated_ci.json` + `oof_calibrated_week_*.parquet`.
- **Cursos conocidos (estratificada)** — new students inside already-seen courses. Source: `stratified_nested_results.json` + `oof_stratified_week_*.parquet`.

Both are **PUC, calibrated CatBoost** (the production artifact), so every interactive number now corresponds to a real out-of-fold prediction vector. **U. Autónoma** moved to the technical annex as **segunda institución, DROP-A arm only (n=322)**.

---

## Header / hero
| Element | OLD | NEW | Source |
|---|---|---|---|
| Lead AUC | "En la U. Autónoma … AUC de **0.74**" (UA KEEP, sem 2 sin notas) | "En PUC … cursos nunca vistos … AUC **0.80** en la semana 2 … a **0.84** en la semana 8" | `confirmatory_calibrated_ci.json` wk2/wk8 |
| Lead capacity | — | "marcando al 20% … se detecta cerca del **68%** hacia la semana 6" | `confirmatory_calibrated_ci.json` capacity wk6/wk8 @0.20 = 0.683 |
| Pill 1 | "U. Autónoma · 280 features · **XGBoost AUC 0.86**" | "PUC · 560 estudiantes-curso · 7 cursos" | — |
| Pill 2 | "PUC · 560 estudiantes · 7 cursos" | "Cursos nunca vistos (LOCO) · AUC **0.80→0.84**" | LOCO wk2/wk8 |
| Pill 3 | "Validación semanal + leave-course-out" | "CatBoost calibrado · CV anidada · IC 95%" | — |

---

## Section 01 — technical annex tables

### PUC main table  (was "XGBoost conservador, cross-course"; now **LOCO calibrated CatBoost**)
| Semana | OLD ROC-AUC [IC] | NEW ROC-AUC [IC] | Source |
|---|---|---|---|
| Sem 2 | 0.762 [0.674–0.840] | **0.798 [0.724–0.861]** | `confirmatory_calibrated_ci.json` |
| Sem 4 | 0.789 [0.702–0.862] | **0.830 [0.770–0.887]** | " |
| Sem 6 | 0.791 [0.707–0.868] | **0.818 [0.748–0.878]** | " |
| Sem 8 | 0.837 [0.763–0.900] | **0.838 [0.769–0.900]** | " |
| Semestre | 0.763 [0.674–0.841] | **0.830 [0.759–0.895]** | " |

PR-AUC, Brier, ECE, F1/MCC/Bal.Acc (max-F1) in that table all regenerated from the same OOF vectors.

### NEW second PUC table — **estratificada** (didn't exist before)
| Semana | ROC-AUC [IC] | Source |
|---|---|---|
| Sem 2 | 0.797 [0.719–0.861] | `stratified_nested_results.json` |
| Sem 4 | 0.792 [0.719–0.859] | " |
| Sem 6 | 0.797 [0.727–0.863] | " |
| Sem 8 | 0.829 [0.756–0.897] | " |
| Semestre | 0.814 [0.729–0.888] | " |

Honest finding surfaced: estratificada sits **at or below** LOCO — PUC generalizes to unseen courses without cost.

### Naive-classifier comparison — "Modelo TM³ (máx-F1)" row (PUC Sem 8)
| Metric | OLD | NEW (wk8 LOCO max-F1) |
|---|---|---|
| Recall | 0.561 | **0.683** |
| Precisión | 0.359 | **0.322** |
| F1 | 0.438 | 0.438 |
| MCC | 0.395 | **0.409** |
| Bal.Acc | 0.741 | **0.785** |
| Accuracy | 0.895 | **0.871** |

Source: `oof_calibrated_week_8.parquet`. The two naive rows («Marcar a todos» / «Nadie reprueba») unchanged (prevalence 7.3%).

### U. Autónoma table  (was **KEEP-arm strat, forbidden**; now **DROP-A segunda institución**)
| Semana | OLD ROC-AUC (KEEP) | NEW estratificada (DROP-A) | NEW LOCO (DROP-A) |
|---|---|---|---|
| Sem 2 | 0.740 | **0.665** | 0.430 |
| Sem 4 | 0.742 | **0.693** | 0.485 |
| Sem 8 | 0.828 | **0.790** | 0.615 |
| Semestre | **0.903** | **0.805** | 0.635 |

Source: `ua_confirmatory.json` (arm `DROP_A`, n=322, calibrated). OLD Sem 2/6 "con evaluaciones" rows and the KEEP 0.903 removed.

---

## Section 02–03 — interactive `window.DATA` (drives the charts)
| Field | OLD (forbidden / stale) | NEW | Source |
|---|---|---|---|
| `ua_weeks[].auc` | 0.7428 / 0.7423 / 0.7455 / 0.8278 / **0.9033** (UA KEEP) | LOCO 0.798/0.830/0.818/0.838/0.830 · STRAT 0.797/0.792/0.797/0.829/0.814 | `confirmatory_calibrated_ci.json`, `stratified_nested_results.json` |
| `ua_best.holdout_auc` | **0.8605** (UA KEEP hold-out) | *removed* | — |
| `ua_best.loco_auc` | **0.7454** (UA KEEP) | *removed* | — |
| `ua_best.per_course_auc` | 0.692…0.944 (UA KEEP, 10 cursos) | per-course **LOCO PUC** AUC 0.54…0.95 (7 cursos), pooled 0.838 | `oof_calibrated_week_8.parquet` by `course_id` |
| `puc.weeks[].auc` | 0.8308 / 0.8716 / 0.8632 / 0.8632 / **0.8537** (old non-nested benchmark) | LOCO calibrated 0.798/0.830/0.818/0.838/0.830 | `confirmatory_calibrated_ci.json` |
| `puc.best_models` | 0.872 / 0.863 / 0.88 (old non-nested) | *removed* | — |
| `real_ops` (ROC operating points) | old benchmark confusion counts | **recomputed TP/FP/TN/FN** from OOF at real thresholds, per scheme/week | `oof_{calibrated,stratified}_week_*.parquet` |
| ROC `fpr/tpr` arrays | binormal-only | real empirical ROC (≤60 pts) from OOF | same |
| Generalization cards | "hold-out 0.86 / LOCO 0.75 / 10 cursos" (UA) | "AUC pooled LOCO **0.84** / sem 2 **0.80** / 7 cursos / 7.3%" | `confirmatory_calibrated_ci.json` |
| `roiAuc` (ROI calc) | 0.828 (UA) / 0.863 (PUC) | 0.79 (UA DROP-A) / 0.838 (PUC); base rate 39%→30% | ROI is illustrative |

---

## Deleted claims (with reasons)
1. **"AUC 0.86 / 0.861 hold-out" (UA)** — UA KEEP-arm, contaminated labels (51 active-zero LTI artifacts). Forbidden.
2. **"AUC 0.903 / 0.90 / 0.89 semestre" (UA)** — UA KEEP-arm stratified header family. Forbidden.
3. **"LOCO 0.745 / 0.75, 10 cursos" (UA)** — UA KEEP-arm generalization. Forbidden; UA DROP-A LOCO is actually 0.43–0.64 (weak) and now shown honestly in the annex.
4. **"sin ninguna nota el modelo pierde solo 0.003 de AUC"** — derived from UA KEEP with/without-assessment pair. Removed (UA KEEP).
5. **PUC "mejor modelo por semana AUC 0.83–0.87" (non-nested benchmark)** — superseded by nested; only permissible in an annex as prior non-nested validation, so removed from the headline/charts.
6. **PUC `best_models` 0.872 / 0.863 / 0.88 (incl. 3-class SMOTE)** — old non-nested, off-protocol. Removed.
7. **"primeras notas → / ← sin notas aún" milestone** — UA-specific narrative (grades appear ~sem 8); not meaningful for PUC schemes. Disabled.

---

## Final best-defensible per-week table (what the page now leads with)

**PUC — calibrated CatBoost, nested CV, top-40 per fold, IC bootstrap 95%:**

| Semana | Cursos nunca vistos (LOCO) | Cursos conocidos (estratificada) |
|---|---|---|
| Sem 2 | **0.80** [0.72–0.86] | 0.80 [0.72–0.86] |
| Sem 4 | **0.83** [0.77–0.89] | 0.79 [0.72–0.86] |
| Sem 6 | **0.82** [0.75–0.88] | 0.80 [0.73–0.86] |
| Sem 8 | **0.84** [0.77–0.90] | 0.83 [0.76–0.90] |
| Semestre | **0.83** [0.76–0.90] | 0.81 [0.73–0.89] |

- **Recall a capacidad 20%** (LOCO): ~61% (sem 2) → ~68% (sem 6–8).
- **Best-per-week cross-model LOCO** (documented in the map, not needed for the header): wk8 = **0.848** (tuned XGB nested, `nested_cv_results.json`) slightly exceeds CatBoost 0.838; all other weeks CatBoost calibrated is the max.
- **U. Autónoma (segunda institución, DROP-A n=322, estratificada):** 0.67 / 0.69 / 0.79 / 0.81; LOCO weak (0.43–0.64). Quotable-alone value is the estratificada.

**Provenance sentence (one line, as it appears on the page):** *Cifras bajo validación cruzada anidada, LOCO ("cursos nunca vistos") o estratificada ("cursos conocidos, alumnos nuevos") según la fila, sobre datos limpios desde el clickstream, con CatBoost calibrado (Platt) e IC bootstrap 95%.*

---

## Verification performed
- Chrome (149) render of the served page: **0 console errors / 0 warnings**; charts build (wkChart/rocChart/locoChart populated); scheme, ROC-week and ROI toggles functional; **no forbidden strings** in the rendered body.
- `node --check` passes on both `<script>` blocks; `window.DATA` JSON parses.
- `index.html` and `metricas-tecnicas-udla.html` md5-unchanged; backup byte-identical to the original.

Adoption / publication is Paul's call.
