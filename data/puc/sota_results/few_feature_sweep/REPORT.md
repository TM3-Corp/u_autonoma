# PUC Few-Feature Sweep — binary_4.0 vs binary_4.5

**Generated:** 2026-06-29 · `scripts/puc_few_feature_sweep.py` · 800 evals, 9.8 min
**Method:** same feature engineering + SOTA composite ranking + grouped-by-course CV as
`puc_benchmark_sota.py` (the authoritative benchmark). Only the **number of top-ranked
features** (N) and the **grade threshold** vary. 7 benchmark courses (560 students),
percentile windowing fixed at 0.05. Per-fold feature ranking computed on TRAIN only
(leak-free). `StratifiedGroupKFold(groups=course_id)` ⇒ out-of-fold ROC-AUC is a
**cross-course (LOCO-style) generalization** estimate.

## 1. Does "2-5 features" suffice? — NO

Best-of-5-models ROC-AUC, with-assessment, by N (threshold 4.0):

| week | N=2 | N=3 | N=5 | N=8 | N=13 | N=21 | N=34 | full |
|------|-----|-----|-----|-----|------|------|------|------|
| 2    |0.681|0.709|0.748|0.739|0.742 |0.760 |0.779 |0.812 |
| 4    |0.673|0.686|0.737|0.777|0.806 |0.822 |0.827 |0.813 |
| 6    |0.744|0.769|0.798|0.814|0.809 |0.803 |0.841 |0.843 |
| 8    |0.741|0.787|0.837|0.848|0.825 |0.833 |0.847 |0.855 |
| full |0.632|0.650|0.671|0.768|0.739 |0.773 |0.806 |0.839 |

- At **N=2-5** AUC is **0.06-0.17 below** the full-feature model. No plateau at few features.
- **Optimal small-N** (smallest N within 0.015 AUC of full) is mostly **N=13-34**, often >34.
  Only week-8 reaches N=8. This matches the dedicated SOTA FS pipeline that settled on **33-40**.
- **Conclusion:** the real SOTA sweet spot is ~**13-34 features**, not 2-5. The "2-5 features
  work great" claim is **not reproducible** in this repo (Ignacio's source is external).

## 2. The 4.5 threshold trade-off — balance UP, discriminability DOWN

| week | 4.0 full AUC | 4.5 full AUC | Δ (4.0→4.5) |
|------|--------------|--------------|-------------|
| 2    | 0.812        | 0.649        | **-0.164**  |
| 4    | 0.813        | 0.674        | **-0.139**  |
| 6    | 0.843        | 0.710        | **-0.132**  |
| 8    | 0.855        | 0.729        | **-0.126**  |
| full | 0.839        | 0.768        | **-0.071**  |

- Prevalence (class balance): **4.0 = 7.2%** (imbalance ~12.7:1) → **4.5 = 20.4%** (~3.9:1).
  The 4.5 cut does reduce imbalance ~3×, as hoped.
- **But ROC-AUC collapses 0.07-0.16** across every cutoff week. The drop holds at full features
  AND at every N, so it is **intrinsic to the label change**, not a feature-count or tuning artifact.
- **Mechanism:** the 74 students in `[4.0, 4.5)` who flip to "reprobado" are *marginal-pass*
  students whose LMS behavior resembles passers, not failers. Adding them as positives injects
  hard-to-separate cases the model cannot discriminate ⇒ AUC falls. Class-balancing models
  (`*_balanced`) are already in the best-of-5 and do not recover it — the problem is **signal,
  not balance**.

## 3. Sanity check vs the artifact

Full-feature grouped-CV 4.0 AUCs here (0.81-0.86) match the artifact's published 0.83-0.87.
Mine run slightly lower because they are **untuned** (no optuna) and use **grouped/LOCO CV**
for every cell (the artifact's headline used phase-2 tuned models). Methodology reproduces the
ballpark ⇒ the pipeline is consistent.

## 4. Bottom line

- **Imbalance was not the bottleneck.** The 4.0 models are already strong (0.81-0.86 LOCO AUC).
- **Moving to 4.5 makes the predictor materially worse** (−0.07 to −0.16 AUC) in exchange for
  better nominal class balance — a bad trade for a sales/early-warning tool.
- **Recommended:** keep the **4.0** threshold (the official Chilean passing grade *and* the
  stronger classifier). Handle the low-prevalence precision behavior as already done in the
  artifact (operating-point + absolute counts), not by redefining failure.
- The SOTA few-feature story, if used, is **~13-34 features**, not 2-5.

Raw data: `sweep_results.json` (800 rows; per cell: roc_auc, recall, precision, f1/f2, mcc,
operating-point confusion matrices, per-course metrics, top features).
