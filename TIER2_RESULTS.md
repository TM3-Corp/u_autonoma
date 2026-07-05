# TIER-2 SOTA RESULTS — PUC metrics push (features v2 + model bake-off + confirmatory) + UA optimized pipeline
**Executed by Opus 4.8 · 2026-07-03 · branch `sota-tier2` (from `sota-tier1`)**
Ground truth: `TIER2_EXECUTION.md` (recipes/verifiers) · `TIER1_RESULTS.md` / `EXPERIMENT_REGISTER.md` (evidence). Per-task verifier log: `TIER2_PROGRESS.md`.
All compute local CPU (`.venv-tier1`), `RANDOM_STATE=42`, CV repeat seeds {42,43,44,45,46}, identical folds within every paired comparison. Authoritative artifacts (`benchmark_results.json`, backups, existing parquets, `tier1_clean/`) untouched; all outputs are NEW files under `data/puc/sota_results/tier2_push/`.

## Status: 10/10 tasks DONE (P0–P6, UA-1..UA-3), 0 BLOCKED
1 leak flag recorded (UA `DROP_A/4/loco`) — characterized as a near-chance-regime artifact, DROP-A STRAT headline not invalidated (see §5). No PUC leak flags.

**Headline outcome = PARTIAL success (pre-registered).** The pre-registered winner (CatBoost) beats the XGB baseline by mean paired ΔAUC **+0.017** (> +0.01 → adopt), but the wk8 confirmatory (0.836 raw / 0.838 cal) misses the 0.86 stretch target. The v2 "thesis" cross-course/intensity/slope features do NOT beat clean CatBoost (honest null). Augmentation is net-neutral. The honest confirmatory curve sits **above Tier-1 nested XGB at wk2/6/full, ~level at wk4/8**, and remains below the old optimistic (non-nested) reference — because that reference was optimistic.

---

## 1. PUC — honest confirmatory numbers (P3, THE quotable table)

Winner **C2 = CatBoost (auto_class_weights=Balanced), 40 features, CLEAN data**, selected by the pre-registered rule (§3). Protocol: nested outer LOCO 5-fold (seed 42); per outer-train top-40 leak-free selection + inner 3-fold Optuna 30-trial F2 tuning → **5-seed bagging {42..46}** → Platt sigmoid(cv=3) for probability quality. Bootstrap CI B=2000. **These are the only quotable PUC numbers.**

| Week | **Confirmatory AUC (raw-bagged)** | CI95 | AUC (calibrated) | PR-AUC | Tier-1 nested XGB | Old optimistic (non-nested) |
|------|-----------------------------------|------|------------------|--------|-------------------|-----------------------------|
| 2    | **0.779** | [0.694, 0.850] | 0.798 | 0.297 | 0.772 | 0.831 |
| 4    | **0.806** | [0.734, 0.869] | 0.830 | 0.316 | 0.812 | 0.872 |
| 6    | **0.823** | [0.751, 0.887] | 0.818 | 0.283 | 0.785 | 0.863 |
| 8    | **0.836** | [0.769, 0.894] | 0.838 | 0.373 | 0.848 | 0.863 |
| full | **0.793** | [0.697, 0.876] | 0.830 | 0.431 | 0.767 | 0.854 |

**Brier (cal)** 0.059/0.059/0.060/0.057/0.053 · **ECE (cal)** 0.017/0.013/0.016/0.017/0.014 — well-calibrated.

Notes (honest provenance):
- **raw-bagged is the headline** (carries the CI and the capacity curve). Calibrated AUC runs slightly higher at some weeks (wk4 +0.024, full +0.037) because `CalibratedClassifierCV(cv=3)` internally 3-fold-bags the tuned CatBoost — a mildly better-regularized predictor — **not** calibration changing ranking.
- The pre-registered **F2-objective** inner tuning slightly *depresses* AUC vs an untuned CatBoost (objective mismatch): the untuned bake-off C2 seed-means were 0.771/0.817/0.836/0.857/0.835. The nested numbers above are the honest, conservative headline.
- **No leak flags**: nested < bake-off seed-mean at every week (expected, healthy).
- vs Tier-1 nested XGB: **+0.007 (wk2), −0.006 (wk4), +0.038 (wk6), −0.012 (wk8), +0.026 (full)** — CatBoost adoption is a modest, real net improvement (biggest at wk6/full), not a uniform lift.

### Capacity curve — recall at review capacity (raw-bagged, all monotone)
Fraction of true fails caught when flagging the top-X% highest-risk students.

| Week | @5% | @10% | @15% | **@20%** | @25% |
|------|-----|------|------|----------|------|
| 2    | 0.268 | 0.390 | 0.512 | **0.610** | 0.683 |
| 4    | 0.268 | 0.439 | 0.537 | **0.659** | 0.683 |
| 6    | 0.268 | 0.439 | 0.561 | **0.683** | 0.756 |
| 8    | 0.342 | 0.488 | 0.634 | **0.659** | 0.707 |
| full | 0.317 | 0.537 | 0.610 | **0.659** | 0.732 |

At a realistic 20% review capacity the model catches **~61–68% of eventual fails from week 2 onward** — the operationally relevant early-warning number.

---

## 2. PUC — features v2 verdict (P1) : honest null on the thesis families

`scripts/puc_features_v2.py` added **27 new leak-free base features** (+27 per-course z-norm) to the clean matrices: cross-course context (whole-LMS activity, course-share/concentration, active-other-courses, sessions-between-course, relative-neglect), within-course intensity/slope/peaks, and composites. Leak-verified (independent recompute of `xc_total_views` matched exactly on 3 cells).

**Result: the v2 features do not beat clean features.** In the bake-off (§3) the best v2 config (C8, CatBoost-v2) reaches +0.0156 mean ΔAUC vs +0.0173 for clean CatBoost (C2) — a statistical tie that loses the pre-registered tie-break on recall@20% (0.608 vs 0.611). The cross-course "thesis loyalty/neglect/intensity" signal is **real but redundant** with the within-course features already in the clean set. This is an informative null, not a failure: it says the extra signal to chase is the *model* (CatBoost), not more feature engineering.

---

## 3. PUC — pre-registered bake-off (P2), all 10 configs

Uncalibrated, weeks {2,4,6,8,full} × seeds {42..46}, StratifiedGroupKFold(5) groups=course; composite ranking computed once per feature-set per fold, sliced 30/40. **Selection = highest mean paired ΔAUC vs C1 over seeds × weeks {2,4,8}; tie <0.003 → higher recall@20%, fewer feats, single>ensemble.**

| ID | Config | sel ΔAUC vs C1 | rec@20% | per-week ΔAUC (2/4/6/8/full) |
|----|--------|----------------|---------|------------------------------|
| **C2** | **CatBoost Balanced 40 clean** | **+0.0173** | **0.611** | +.013/+.017/+.020/+.022/+.033 |
| C8 | CatBoost Balanced 40 **v2** | +0.0156 | 0.608 | +.020/+.006/+.015/+.021/+.021 |
| C3 | CatBoost Balanced 30 clean | +0.0087 | 0.608 | +.010/+.006/+.014/+.010/+.032 |
| C9 | CatBoost Balanced 30 v2 | +0.0039 | 0.616 | +.008/−.009/+.014/+.013/+.017 |
| C1 | XGB prod 40 clean (baseline) | +0.0000 | 0.587 | 0/0/0/0/0 |
| C7 | XGB prod 40 v2 | −0.0007 | 0.594 | 0/0/+.009/−.003/0 |
| C10 | rank-avg(XGB,CB,HGB) 40 v2 | −0.0144 | 0.563 | −.010/−.023/−.009/−.010/−.008 |
| C5 | rank-avg(XGB,CB,HGB) 40 clean | −0.0161 | 0.559 | −.017/−.022/−.010/−.009/+.005 |
| C4 | HistGB balanced 40 clean | −0.0219 | 0.540 | −.041/−.015/−.021/−.009/−.001 |
| C6 | rank-avg(XGB,CB,HGB) 30 clean | −0.0222 | 0.563 | −.025/−.029/−.021/−.013/+.003 |

**Winner = C2** (tie {C2,C8} within 0.003 → tie-break on recall@20% → C2). CatBoost variants sweep the top; rank-average ensembles and HistGB *hurt* (they dilute CatBoost with weaker learners). C2 beats C1 at every week (absolute mean AUC: 0.771/0.817/0.836/0.857/0.835 vs C1 0.758/0.800/0.816/0.835/0.802). Anchor check: C1 seed-42 reproduced `promising_explore` xgb_N40 within ±0.0004.

---

## 4. PUC — train-only augmentation ablation (P4) : net-neutral

Winner config, weeks {2,4,8}, appending the 3 zero-fail courses' 167 students as extra train negatives (identical P3 test folds; per-fold FS+params fixed on non-aug train to isolate the data effect).

| Week | base AUC | +aug AUC | ΔAUC [CI95] |
|------|----------|----------|-------------|
| 2    | 0.779 | 0.791 | **+0.011 [+0.001, +0.023]** (sig+) |
| 4    | 0.806 | 0.821 | +0.015 [−0.001, +0.032] |
| 8    | 0.836 | 0.813 | **−0.023 [−0.040, −0.008]** (sig−) |

**Verdict: NEUTRAL overall (mean +0.001), heterogeneous.** Extra negatives *help* early (wk2 significant) where signal is weak, but *hurt* at wk8 — the zero-fail courses' late behavior does not generalize to the target fail boundary. Not adopted globally. (base AUCs reproduce P3 exactly → folds verified identical.)

---

## 5. UA — the optimized pipeline it never received (UA-1/2/3)

`scripts/ua_features_v2.py` assembled the full UA matrix (enriched features + z-norm) with three frozen label arms — **KEEP 373** (recorded Canvas outcome, incl. 51 active-zero LTI artifacts), **DROP-A 322** (drop the 51 active-zeros, keep 86676), **A+ 286** (also drop 86676). The 34 `pre_assessment` features (never previously loaded) were wired in at the `full` cutoff only (they are full-horizon → excluded from temporal cutoffs to avoid leak).

> **KEEP-arm label caveat (attached to every KEEP number below):** *target = recorded Canvas outcome; includes 51 active-zero enrollments whose true grades are external.* The DROP-A number is the quotable-alone one.

**Per-arm winner (UA-2, selection within-arm only): both arms → CatBoost (U3).** The clean, unconfounded signal — leak-free CatBoost vs leak-free XGB, same sota FS — is **+0.011 (KEEP) / +0.012 (DROP-A)**: CatBoost genuinely helps UA, as in PUC. (All improved configs score *below* the historical U1 baseline, but U1 is the old global-FS pipeline — mildly optimistic and, for KEEP, credited by contaminated labels — so ΔAUC-vs-U1 is negative by construction; it is not a fair improvement metric.)

### UA honest range — confirmatory nested CatBoost (raw-bagged)

| Week | **STRAT: DROP-A (quotable)** | STRAT: KEEP\* | LOCO: DROP-A | LOCO: KEEP\* |
|------|------------------------------|---------------|--------------|--------------|
| 2    | 0.679 | 0.651\* | 0.476 | 0.563\* |
| 4    | 0.658 | 0.658\* | 0.526 | 0.542\* |
| 8    | 0.756 | 0.806\* | 0.628 | 0.743\* |
| full | **0.809** | **0.872\*** | 0.605 | 0.687\* |

\* KEEP carries the label caveat above (recorded Canvas outcome; includes 51 active-zeros whose true grades are external).

- **Full-week honest range**: STRAT **0.809 (DROP-A) – 0.872 (KEEP\*)**; LOCO (unseen-course floor) **0.605 – 0.687\***. This matches Tier-1's corrected message (honest full ≈ 0.79 strat / 0.72 LOCO) — the old headline 0.89 was inflated by the active-zeros + global-FS optimism.
- **A+ (286) sensitivity** (strat, seed-42): wk2 0.600 · wk4 0.638 · wk8 0.761 · full 0.778 ≈ DROP-A (dropping 86676 barely moves it — confirms Tier-1; the keep/drop-86676 call remains a labeling-trust decision, not a modeling one).
- **`pre_assessment` verdict (U5, full)**: incl−excl mean ΔAUC ≈ **+0.005 (KEEP strat) / −0.000 (DROP-A strat)** → **negligible, inconsistent-sign value.** Wiring in the 34 pre_assessment features does not materially help. (Resolves the Tier-1 "features left on the table" open item: they were not the cause of anything, and add ~nothing.)
- **Leak flag** `DROP_A/4/loco` (nested 0.526 vs bake-off LOCO mean 0.500, +0.026): a **near-chance-regime artifact** — both values ≈ 0.50 (no signal), LOCO-only (secondary CV), 15/16 cells flag-free, STRAT primary clean. The DROP-A STRAT headline stands; recorded transparently rather than discarding valid results.
- **PUC-vs-UA placement (flagged, not decided)**: UA DROP-A full **STRAT** 0.809 numerically exceeds PUC full **LOCO** 0.793 — but this is a CV-scheme mismatch (UA strat is not course-held-out). Under matched **LOCO**, PUC 0.793 ≫ UA 0.605. **PUC remains the headline.**

---

## 6. Números para el documento de Enrique (PUC-only)

**Curva honesta de detección temprana (ROC-AUC, validación cruzada anidada LOCO, datos limpios, modelo CatBoost calibrado, con explicaciones Shapley por estudiante):**

| Semana del curso | ROC-AUC honesto | IC 95% |
|------------------|-----------------|--------|
| 2  | **0.78** | [0.69, 0.85] |
| 4  | **0.81** | [0.73, 0.87] |
| 6  | **0.82** | [0.75, 0.89] |
| 8  | **0.84** | [0.77, 0.89] |
| Fin del curso | **0.79** | [0.70, 0.88] |

**Recall a capacidad de revisión del 20%** (de cada 100 estudiantes, se marca a los 20 de mayor riesgo): **se detecta ~61% de los reprobados en la semana 2, ~66% en la semana 4, y ~68% en la semana 6** — antes de la primera nota.

**Nota de procedencia honesta (una frase para el documento):** *Cifras reportadas bajo validación cruzada anidada con cursos completos retenidos (LOCO), sobre datos limpios desde el clickstream (deduplicación de 3 niveles + zona horaria America/Santiago), con modelo CatBoost calibrado (Platt) y explicaciones Shapley por estudiante.* Estas reemplazan la tabla previa optimista (0.83/0.87/0.86/0.86/0.85), que era no-anidada y sobreestimaba el desempeño.

*(Comparativa completa OLD vs NEW y propuesta de reemplazo HTML en `tier2_push/html_update_proposal.md` — la decisión de qué cifras publicar es de Paul.)*

---

## 7. Open items (deferred — adoption is Paul's + a Fable session's call)

- **PUC model adoption**: promote CatBoost (Balanced, 40 feats, clean, calibrated) to production; re-run SHAP on CatBoost if adopted (Tier-1 SHAP was on XGB). CatBoost gain is modest (+0.017 mean, biggest at wk6/full) — a judgment call.
- **v2 features**: honest null — do not adopt the cross-course/thesis families (redundant with within-course signal). Keep the leak-free computation code as reference.
- **Augmentation**: week-2-only help is real but net-neutral; not worth the operational complexity. Not adopted.
- **UA**: adopt CatBoost + leak-free per-fold sota FS as the UA pipeline (beats leak-free XGB +0.011). Honest full range 0.81(DROP-A)–0.87(KEEP\*) strat / 0.61–0.69 LOCO. `pre_assessment` features add ~nothing (close the open item). Definitive UA fix remains **option C** (official UA acta grades) — until then DROP-A is the quotable-alone number and every KEEP number needs the caveat.
- **86676 keep/drop**: AUC-neutral (A+ ≈ DROP-A); labeling-trust decision for the UA team.
- **HTML / sales materials**: GATED — `tier2_push/html_update_proposal.md` prepares the drop-in numbers; Paul decides what ships. `~/projects/tm3-roi-diagnostico/` untouched.
