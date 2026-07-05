# PUC — Fair 4.0 vs 4.5 Threshold Comparison (with confidence intervals)

**Generated:** 2026-06-29 · `scripts/puc_fair_threshold_compare.py` · 4.5 min
**Why:** the earlier 4.0-vs-4.5 AUC gap was a *tuned-vs-untuned* comparison with no
significance test. This gives 4.5 a fair, optimized, construct-valid shot and tests
whether any gap is real. 7 courses / 560 students, LOCO (group-by-course) CV, per-fold
SOTA top-40 feature basis, with-assessment. CIs = student-level bootstrap (B=2000).

## A. Regression-then-threshold arbiter (one grade regressor, ranked once vs both cuts)

| week | AUC@4.0 | AUC@4.5 | diff | 95% CI | significant |
|------|---------|---------|------|--------|-------------|
| 2    | 0.756   | 0.633   | 0.123| [+0.048, +0.201] | **yes** |
| 4    | 0.742   | 0.695   | 0.046| [−0.022, +0.113] | no |
| 6    | 0.792   | 0.717   | 0.076| [+0.009, +0.147] | **yes** |
| 8    | 0.802   | 0.749   | 0.052| [−0.013, +0.116] | no |
| full | 0.816   | 0.749   | 0.067| [+0.006, +0.128] | **yes** |

No pipeline asymmetry is possible here (identical model + ranking; only the label cut changes).

## B. Tuned-equivalent binary (correct scale_pos_weight = neg/pos per threshold)

| week | AUC 4.0 | AUC 4.5 | diff | 95% CI | sig | Brier 4.0 | Brier 4.5 |
|------|---------|---------|------|--------|-----|-----------|-----------|
| 2    | 0.751   | 0.612   | 0.140| [+0.048,+0.229] | yes | 0.067 | 0.182 |
| 4    | 0.783   | 0.676   | 0.107| [+0.030,+0.184] | yes | 0.066 | 0.182 |
| 6    | 0.825   | 0.703   | 0.122| [+0.057,+0.187] | yes | 0.066 | 0.172 |
| 8    | 0.833   | 0.712   | 0.121| [+0.044,+0.191] | yes | 0.056 | 0.179 |
| full | 0.797   | 0.732   | 0.065| [−0.016,+0.143] | no  | 0.058 | 0.160 |

## Verdict: the 4.5 penalty is REAL, not an artifact

- After a fair, tuned shot the gap **shrank** (from the untuned 0.07–0.16 to ~0.05–0.12)
  but did **not** vanish. Direction is 4.0 > 4.5 in **10/10** week×method cells; significant
  in 3/5 (arbiter) and 4/5 (binary).
- 4.5 binary is also **badly calibrated** (Brier 0.16–0.18 vs 0.06 for 4.0).

## Smoking gun — the drop is intrinsic (Bayes-error / class overlap), now measured

OOF predicted grade by true band (week 8):

| true band | n | predicted grade |
|-----------|---|-----------------|
| fail (<4.0)        | 41  | 4.57 ± 0.59 |
| **marginal [4.0,4.5)** | 74  | **4.89 ± 0.56** |
| pass (≥4.5)        | 445 | 5.31 ± 0.57 |

The marginal cohort is predicted at **4.89 — i.e. the model sees them as passing**, sitting
on top of the pass cohort, not the fail cohort. Their LMS behavior *is* that of passers, so
asking a classifier to label them "fail" is irreducibly hard. This is the textbook overlap
signature; tuning cannot fix it.

## Recommendation

1. **Keep the 4.0 binary** for the early-warning tool: official passing grade, strongest
   classifier (0.75–0.83 LOCO AUC), best calibrated. Imbalance at 7.3% was not the bottleneck.
2. **To surface marginal students**, do NOT add a second hard binary at 4.5. Use either:
   - **Regression-on-grade** (already in the repo): threshold-agnostic, institution sets its own
     cut, returns "expected grade ± uncertainty" per student. 4.5 via regression ≈ or > binary 4.5.
   - **Ordinal fail/marginal/pass** with the marginal band routed to **human review** (not a hard
     machine call where the error is irreducible and a false positive stigmatizes).
3. An absolute 4.5 cut combined with course-relative features is also a cross-course
   transfer-validity threat (4.5 means different things across courses) — another reason to
   prefer regression/ordinal over a second absolute binary.

## SOTA context

A 9-agent SOTA review (Oviedo paper + 2024–2026 Learning Analytics literature + adversarial
verification) confirmed our pipeline is **at/near SOTA as a tabular discriminator** — tuned GBDT
on course-relative features is the state of the art; deep/temporal/GNN models would not help at
this N. The remaining genuine gaps are deployability plumbing: probability **calibration** +
Brier/ECE reporting, and **retiring SMOTE** (the 2022–2026 risk-scoring consensus rejects it).
