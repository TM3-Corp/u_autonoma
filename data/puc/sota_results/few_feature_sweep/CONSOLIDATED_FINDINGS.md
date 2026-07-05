# PUC Modeling — Consolidated Findings (SOTA review, 4.5 threshold, calibration, framing)

**Date:** 2026-06-29 · target: PUC early-warning (7 courses, 560 students, Canvas LMS clickstream)
All experiments under leave-one-course-out (StratifiedGroupKFold) CV, per-fold leak-free
SOTA top-40 feature selection, with-assessment features.

## 1. Is our pipeline SOTA? — YES (as a discriminator)

A 9-agent SOTA review (Oviedo paper Riestra-González et al. 2021 C&E — the team's own
reference — + 2024-2026 Learning Analytics literature + adversarial verification) concluded:
tuned gradient-boosted trees on course-relative tabular features **is** the state of the art
for early LMS dropout prediction. Deep/temporal/GNN/transformer models would not beat it at
560 students / 7 courses / early weeks. Our feature engineering matches or exceeds the
published comparators; our cross-course (LOCO) evaluation is stronger than most.
The only genuine gaps were deployability plumbing (calibration; SMOTE reliance) — now closed.

## 2. Moving the pass cut 4.0 -> 4.5 is NOT worth it (fair, tuned, with CIs)

Earlier the 4.5 gap was a tuned-vs-untuned comparison with no significance test. Given a fair
shot (correct `scale_pos_weight` per threshold) + a construct-valid regression-then-threshold
arbiter (one model ranked once vs both cuts) + bootstrap CIs:

- The 4.0>4.5 AUC gap **shrank** (untuned 0.07-0.16 -> fair ~0.05-0.12) but **survived**:
  significant in 3/5 weeks (arbiter) and 4/5 (tuned binary); direction 4.0>4.5 in 10/10 cells.
- 4.5 binary is also badly calibrated (Brier 0.16-0.18 vs 0.06 at 4.0).
- **Smoking gun:** the 74 marginal [4.0,4.5) students have OOF predicted grade **4.89** — the
  model sees them as passing, sitting with the pass cohort (5.31), not the fail cohort (4.57).
  Their LMS behavior *is* that of passers => irreducible class-overlap (Bayes error). Tuning
  cannot fix it. An absolute 4.5 cut + course-relative features is also a transfer-validity threat.

**Conclusion:** keep the 4.0 binary. Imbalance at 7.3% was not the bottleneck.

## 3. SOTA risk-score gaps — CLOSED (target <4.0, LOCO CV)

| config | AUC | Brier | ECE |
|--------|-----|-------|-----|
| raw (uncalibrated)   | 0.798 | 0.063 | 0.048 |
| **Platt-calibrated** | 0.789 | 0.059 | **0.015** |
| SMOTE                | 0.768 | 0.091 | 0.085 |

- **Adopt Platt/sigmoid calibration:** ECE ~3x better (0.048->0.015), Brier better, AUC intact.
  A "70% risk" now means 70% (reliability bins line up). Mandatory for a number a human acts on.
- **Retire SMOTE:** strictly dominated — lower AUC AND worse calibration (over-estimates minority
  risk), exactly as the 2022-2026 risk-scoring consensus predicts.
- **Production config: XGBoost + scale_pos_weight + Platt calibration, no SMOTE.**

## 4. The construct-valid way to surface marginal students (instead of a 4.5 binary)

**Regression-on-grade** (one calibrated model, institution-configurable flag rate):

| week | AUC@4.0 | recall<4.0 @ flag 10/15/20% |
|------|---------|------------------------------|
| 6    | 0.792   | 0.37 / 0.49 / 0.61 |
| 8    | 0.802   | 0.41 / 0.51 / 0.56 |
| full | 0.816   | 0.41 / 0.49 / 0.51 |

Predicted grade is well-calibrated (wk8: pred 4.40->obs 4.62, 5.06->5.16, 5.42->5.46, 5.94->5.92).
Threshold-agnostic: the institution flags whatever rate it can staff; at a 20% review rate it
catches ~50-60% of true fails weeks before grades exist.

**Ordinal 3-tier (Frank-Hall: red=fail / amber=marginal / green=pass)** from the calibrated 4.0
and 4.5 boundaries. Concept validated — at wk8 the red tier is 62% truly-fail (high precision
for "intervene now"), amber concentrates the marginal cohort for human review. NOTE: assign tiers
by **capacity-based thresholds** (the flag-rate table above), not argmax — argmax under-flags at
this prevalence. The amber tier routes the irreducible-overlap cohort to review rather than
forcing a machine call.

## 5. Honest ceiling

LMS-clickstream early warning catches roughly **half** of eventual failures at a ~20% review
rate, consistently across every analysis. That is the real signal; the value proposition is
catching ~50-60% of at-risk students *weeks before any grade exists*, with a trustworthy
calibrated risk score — not a second, weaker binary at 4.5.

## Recommendation

1. Keep the **4.0** binary as the headline early-warning model.
2. Wire **Platt calibration** into the production model; drop SMOTE.
3. If surfacing marginal students is desired, ship **regression-on-grade** (calibrated expected
   grade + institution-set flag rate) and/or a **tiered red/amber/green** view with the marginal
   band routed to human review — never a second hard binary at 4.5.

Scripts: `puc_few_feature_sweep.py`, `puc_fair_threshold_compare.py`, `puc_calibration_smote.py`,
`puc_ordinal_regression_framing.py`. Data: `data/puc/sota_results/few_feature_sweep/*.json`.
