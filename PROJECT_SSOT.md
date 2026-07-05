# PROJECT SSOT — Canvas-LMS Student-Failure Prediction (PUC + UA / TM3)
**THE single source of truth. Start here. Last updated 2026-07-04.**

This repo predicts which students will fail a course (PUC grade < 4.0 · UA final_score < 57) from Canvas LMS clickstream, for the TM3 early-warning product. The repo accumulated ~98 markdown docs across many sessions — most are historical or superseded. **This file is the index and the arbiter: nothing else holds headline authority.** If a number here disagrees with another doc, this file (and the canonical docs it names) wins.

---

## 1. Headline findings (the current truth)

**Best DEFENSIBLE metrics** (nested LOCO cross-validation, per-fold selection, calibrated, held-out courses — reviewer-proof):

| | PUC (grade<4.0, n=560, 7.3% fail) | UA (final<57, DROP-A n=322) |
|---|---|---|
| Week 2 | ROC-AUC **0.80** [0.72, 0.86] | (weak early) |
| Week 8 | **0.84–0.85** [0.77, 0.90] | **0.76** (within-institution) |
| Full semester | **0.83** | **0.81** [0.76, 0.86] (within-institution) |
| Operational | catches **~68% of failures at 20% review capacity**, before first grade | — |
| Winning model | CatBoost (auto_class_weights=Balanced), calibrated | CatBoost, DROP-A labels |

Full provenance + every number ever computed → **`RESULTS_LEDGER.md`**.

**Four established conclusions** (do not re-litigate; each is evidenced):
1. **The old 0.87 (wk4) / 0.90 (UA) numbers were real but not defensible** — PUC 0.86–0.87 = non-nested optimism (~+0.05 bias); UA ≥0.85 = contaminated by 51 mislabeled active-zeros; PUC 0.97+ = grades leaked into features. The honest ceiling is PUC ~0.85 / UA ~0.81.
2. **Feature ceiling is at the 62 "basics."** The full ~325-feature historical corpus, mRMR selection, survival framing, and temporal cascade all move AUC ±0.005 (null). Signal saturates fast → `PIPELINE_REVIEW.md`.
3. **Cross-institution pooling is a NULL.** PUC and UA predict failure from largely non-overlapping features (top-10 Jaccard 0.18); pooling ≤ single-institution → `TIER3_RESULTS.md`.
4. **The binding constraint is data volume + label quality, not modeling.** Highest-ROI next lever = official UA acta grades, not any model/feature.

---

## 2. Documentation map — the ONLY authoritative docs

| Topic | Canonical doc | Everything else on this topic is archive |
|---|---|---|
| **Front door / index** | `PROJECT_SSOT.md` (this file) | — |
| **All metrics + provenance** | `RESULTS_LEDGER.md` | + `TIER{1,2,3}_RESULTS.md` for per-tier detail |
| **Features (the ~325-feature corpus)** | `FEATURE_CATALOG.md` (root) | retire `docs/03_feature_engineering/FEATURE_CATALOG.md` |
| **Methodology + roadmap** | `PIPELINE_REVIEW.md` | end-to-end audit, 7 stages, next-steps |
| **Per-tier detail** | `TIER1_RESULTS.md` · `TIER2_RESULTS.md` · `TIER3_RESULTS.md` | `TIER*_PROGRESS.md` = verifier traces (archive) |
| **Experiment log** | this file §4 + `EXPERIMENT_REGISTER.md` | register = pre-execution plan; §4 below = what actually ran |
| **Data / cleaning** | code: `scripts/{puc_clean_rebuild,ua_clean_rebuild,ua_remediate_labels}.py`; provenance `data/report/DATASET_COMPARISON.md` | access docs `docs/data_access_discovery.md` |
| **Canvas API reference** | `docs/canvas-api-reference.md` (lowercase) | retire uppercase `docs/CANVAS_API_REFERENCE.md` |
| **Client-facing (safe to hand out)** | `RESULTADOS_PREDICCION_PUC.md` (PUC, honest numbers) | all `data/report/INFORME_*` / `REPORTE_*` are STALE (§3) |

**Superseded → treat as archive** (do not cite; kept for history): all `docs/*SUMMARY*.md` and `docs/*FINAL*.md` and `docs/PHASE_*`/`SOTA_*` (→ RESULTS_LEDGER); `data/report/INFORME_ALERTA_TEMPRANA_v2/v3/v4` (→ v5, itself stale §3); `REPORTE_TECNICO_*` v1/v2 (→ v3); `data/puc/*SUMMARY*.md`; session logs `docs/{27_12,29_12,first_work,previous_session}*.md`.

**Duplicates to collapse:** two Canvas API refs (keep lowercase); two FEATURE_CATALOGs (keep root).

---

## 3. ⚠️ STALE-DANGEROUS docs — numbers that must NOT be quoted

These present PUC ≥0.86 or UA ≥0.85 (optimistic / contaminated / label-leaky) as if current. **Superseded by `RESULTS_LEDGER.md`.** The highest-risk internal ones carry a banner; do not reuse any number below.

| Doc | Bad number | Why wrong |
|---|---|---|
| `ml_team_share/README.md` | UA **0.903** | KEEP-arm (51 active-zeros) |
| `README.md` (repo front) | "identify **81.8%** who fail" | old single-course/threshold artifact |
| `RESULTS_SUMMARY.md` | PUC **0.872** / 0.880 | non-nested + SMOTE |
| `docs/MODEL_RESULTS_REFERENCE.md` | UA **0.902** | grade-leaky |
| `docs/PIPELINE_OVERVIEW.md`, `docs/TECHNICAL_FEATURE_ENGINEERING.md` | UA 0.86 | KEEP-arm |
| `data/report/INFORME_ALERTA_TEMPRANA_v2–v5.md` | UA **0.90** | KEEP-arm (client reports) |
| `data/report/REPORTE_TECNICO_*_v2/v3.md`, `INFORME_MODELO_ALERTA_TEMPRANA.md` | UA 0.85–0.86 | 5-fold headlined over LOCO |
| `data/models/v4_optimized/README.md`, `data/analysis/BENCHMARK_RESULTS.md` | UA 0.86 / 0.88 | KEEP-arm / ensemble |
| `puc_analysis/BENCHMARK_PIPELINE.md`, `data/puc/sota_results/7courses*/BENCHMARK_REPORT*.md` | PUC **0.872**, oviedo 0.977 | non-nested; 0.977 = label-leak |
| `data/report/analysis/THRESHOLD_OPTIMIZATION_SUMMARY.md`, `ml_team_share/THRESHOLD_ANALYSIS_COMPLETE.md` | UA 0.902 / ≥0.85 | KEEP-arm |

---

## 4. Experiment ledger — everything that ran, and its verdict

Chronological; every row links its artifact. ✅ adopted · 🔬 tested→null (do not re-try) · 📊 measurement.

| # | Experiment | Verdict | Artifact |
|---|---|---|---|
| T1 | PUC clean rebuild: 3-level dedup + America/Santiago TZ + 30-min sessions; `interaction_seconds` excluded (unreliable) | ✅ adopted canonical | `scripts/puc_clean_rebuild.py`, `tier1_clean/cleaning_report.json`, `TIER1_RESULTS.md` |
| T1 | UA label remediation → **DROP-A** (drop 51 active-zeros, keep 86676) | ✅ adopted | `scripts/ua_remediate_labels.py` |
| T1 | Nested LOCO CV = honest headline (vs non-nested optimism) | ✅ adopted protocol | `tier1_clean/nested_cv_results.json` |
| T1 | CatBoost + HistGB added to zoo | ✅ CatBoost wins | `tier1_clean/catboost_results.json` |
| T1 | SHAP per-student (XGBoost TreeSHAP) | ✅ | `tier1_clean/shap_*` |
| T2 | Features-v2 (cross-course/thesis families) | 🔬 null — redundant | `TIER2_RESULTS.md §2` |
| T2 | Bake-off 10 configs → **CatBoost Bal-40 clean (C2)** | ✅ winner | `tier2_push/bakeoff_results.json` |
| T2 | Confirmatory PUC (nested, quotable) | 📊 PUC 0.78–0.84 | `tier2_push/confirmatory_results.json` |
| T2 | Train-only augmentation (extra negatives) | 🔬 net-neutral | `tier2_push/augment_ablation.json` |
| T2B | Calibrated CIs + stratified nested + honest HTML | ✅ | `tier2_push/{confirmatory_calibrated_ci,stratified_nested_results}.json` |
| T3 | Shared cross-institution pipeline (23 invariant feats) + pooling | 🔬 **NULL** (0.71 < single-inst) | `TIER3_RESULTS.md`, `tier3_pooled/confirmatory_results.json` |
| T3 | Leave-institution-out transfer (UA→PUC vs actas) | 📊 ~0.72 (cold-start prior) | same |
| q3 | percentile-rank vs z-norm | 📊 pct-rank modestly better (adopt for late weeks) | `q3_pctrank_results.json` |
| q4 | per-institution feature importance | 📊 Jaccard 0.18 (explains the null) | `q4_perinst_features.json` |
| q6–q9 | historical ~325-feature corpus vs 62 basics | 🔬 **null** (±0.001) | `q{6,7,8,9}_*.json` |
| q10 | mRMR selection | 🔬 null (loses to ExtraTrees) | `q10_mrmr.json` |
| q11 | survival / time-to-disengagement | 🔬 null (Δ−0.001; disengagement rare) | `q11_survival.json` |
| q12 | temporal cascade (wk2 risk → wk4 model …) | 🔬 null AUC (+0.005); risk feature ranks #2–3 but redundant | `q12_cascade.json` |
| audit | full results audit → the ledger | ✅ | `RESULTS_LEDGER.md` |
| audit | pipeline review (7 stages + roadmap) | ✅ | `PIPELINE_REVIEW.md` |

All `q*.json` live in `data/puc/sota_results/tier3_pooled/`.

---

## 5. Canonical artifacts (inputs & code)

- **Data (canonical inputs):** `data/puc/puc_clean_data.parquet` (1.77M rows, 7 PUC courses) · `data/ua_clean/ua_clean_data.parquet` (105k rows, 10 UA courses) · `data/puc/puc_grades_clean.parquet` · `data/page_views/student_enrollments.csv` (UA labels).
- **Feature matrices:** `tier1_clean/features/week_{w}_clean.parquet` (PUC) · `tier3_pooled/features/pooled_week_{w}.parquet` (shared) · `feature_schema.json` (the 62-base / 23-invariant sets) · `data/feature_selection/feature_rankings.parquet` (236 ranked).
- **OOF prediction vectors (persisted, leak-free):** `tier2_push/oof_calibrated_week_{w}.parquet` (PUC) · `tier3_pooled/oof_pooled_week_{w}.parquet`.
- **Pipeline code:** `scripts/{puc_clean_rebuild,ua_clean_rebuild}.py` (clean) → `scripts/common_features.py` (shared features) → `scripts/tier3_common.py` (harness: FS, models, LOCO) → `scripts/{puc_confirmatory_v2,g6_confirmatory}.py` (confirmatory).
- **Env:** `.venv-tier1/bin/python` (catboost, xgboost, sklearn, lifelines). `RANDOM_STATE=42`, seeds {42–46}.

---

## 6. Maintenance protocol — keep this file the SSOT

**This index is the arbiter; each fact has exactly one home.**
- **New metric** → `RESULTS_LEDGER.md` (+ `TIER{n}_RESULTS.md` for detail), then update §1 if it's a headline. Never introduce a metric that only lives in a summary/report.
- **New feature** → `FEATURE_CATALOG.md` only.
- **New experiment** → add a row to §4 with its artifact path (json/parquet), and to `EXPERIMENT_REGISTER.md`. No prose-only conclusions — every claim traces to a file.
- **New doc** → decide its class up front (authoritative / client-report / session-log / reference). Session logs and superseded report versions get an `archive/` prefix immediately.
- **Any doc quoting PUC ≥0.86 or UA ≥0.85 without the optimistic/contaminated caveat** is auto-STALE-DANGEROUS → banner it and add to §3.
- **Drift check (weekly / each session start):** `grep -rlE "0\.8[6-9]|0\.90|81\.8%" --include=*.md .` — any new hit that isn't caveated is drift. Confirm `CLAUDE.md` + memory still point here.

---

## 7. Open items (from `PIPELINE_REVIEW.md`, ranked by ROI)
1. **Official UA acta grades** (Option C) — highest lever; data/label, not modeling.
2. **Cluster/hierarchical bootstrap CIs** — current CIs too narrow (students span courses); re-run PUC 0.84 with persisted OOF.
3. Fix doc/code drifts (the `interaction_seconds` "82% null"; remediation script still emits A+ n=286 not DROP-A n=322).
4. **TabPFN v2** head-to-head (only untested model matching this ≤10k-row regime); **partial-pooling** hierarchical model; **SSL sequence model on PUC clickstream** (the one open modeling hypothesis).
- **Do NOT chase:** ARIMAX/SARIMAX (category error), dense/CNN NNs, deep domain-adaptation, more hand-engineered features, mRMR/survival/cascade — all tested or dominated.

> **Product note:** temporal cascade (§4 q12) doesn't help AUC but never hurts — worth it for a coherent monotonic risk-over-time UX for counselors.
