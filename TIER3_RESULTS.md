# TIER-3 SOTA RESULTS — cross-institution pooling + course-eligibility analytics
**Executed by Opus 4.8 · 2026-07-03 · branch `sota-tier3` (from `sota-tier2`)**
Ground truth: `TIER3_EXECUTION.md` (recipes/verifiers). Per-task verifier log: `TIER3_PROGRESS.md`. Context: `TIER1_RESULTS.md`, `TIER2_RESULTS.md`.
All compute local CPU (`.venv-tier1`), `RANDOM_STATE=42`, CV repeat seeds {42..46}, identical folds within paired comparisons. Authoritative artifacts (`benchmark_results.json`, backups, existing parquets, `tier1_clean/`, `tier2_push/`) untouched; all outputs are NEW files under `data/puc/sota_results/tier3_pooled/` + `data/ua_clean/`. Nothing sales-facing edited.

## Status: 8/8 tasks DONE (G0–G7), 0 BLOCKED. No test-label leak. Outcome = **pre-registered NULL** (with two positives).

**Headline outcome.** Pooling PUC + UA into one cross-institution corpus, using *only* features computable identically at both institutions (guardrail 2), **does not beat single-institution models** on the pre-registered balanced set (R2). The primary R2-pooled nested wk8 ROC-AUC is **0.713** [0.652, 0.771] — below the 0.85 success target and below both single-institution references (Tier-2 PUC-only wk8 0.836; R2 UA-only 0.725). Per pre-registration this is the **NULL** result: *institutions don't mix at this feature granularity.* Two genuine positives survive: (1) a real, modest **cross-institution transfer** — a model trained only on UA courses predicts held-out PUC failures against **pristine official actas** at **0.70–0.74** AUC; (2) the **course-predictability map** (standalone value, delivered below).

**Why the null is the honest answer, not a modeling failure.** The institution-invariance guardrail is the binding constraint. At weeks 2–4 an institution-classifier on the full 62-feature z-normed matrix separates PUC from UA at **AUC 0.98–0.997** — the two institutions' early behavior barely overlaps. Enforcing "no feature may encode institution identity" (probe ≤ 0.75) required dropping **39 of 62** z-normed features, leaving **23 institution-invariant behavioral features**. That thin shared signal is what pooling has to work with — and it is weaker than each institution's own full feature set (Tier-2 PUC-only used ~40 PUC-specific features → 0.836). **The shared, portable signal between these two institutions is real but modest.**

---

## 1. Headline table — R2-pooled nested (the ONLY quotable Tier-3 numbers) + transfer

**R2-pooled** = pre-registered balanced set (prevalence ∈ [8%,50%] ∧ fails ≥ 4 ∧ n ≥ 15): **10 courses, 400 pairs, 91 fails, 22.8% prevalence** (2 PUC {55410, 54570} + 8 UA {89390, 88381, 84941, 89099, 79913, 79875, 86020, 84944}). Winner config **CatBoost Balanced + corr-prefilter feature set** (Stage-B, G5). Protocol: nested outer LOCO 5-fold (seed 42), per-outer-train corr-prefilter FS + inner 3-fold Optuna **150-trial** F2 tuning → 5-seed bagging → Platt sigmoid; bootstrap CI B=2000. **These raw-bagged numbers are the only quotable Tier-3 headline.**

| Week | **R2-pooled AUC (raw-bagged)** | CI95 | AUC (cal) | PR-AUC | rec@20% | vs Tier-2 **PUC-only** (context) | vs R2 **UA-only** (context) |
|------|-------------------------------|------|-----------|--------|---------|-------------------------------|-----------------------------|
| 2    | **0.690** | [0.628, 0.748] | 0.627 | 0.406 | 0.363 | 0.779 | 0.679* |
| 4    | **0.616** | [0.544, 0.685] | 0.584 | 0.405 | 0.374 | 0.806 | 0.658* |
| 6    | **0.679** | [0.614, 0.742] | 0.599 | 0.416 | 0.352 | 0.823 | — |
| 8    | **0.713** | [0.652, 0.771] | 0.663 | 0.488 | 0.385 | **0.836** | 0.756* |
| full | **0.708** | [0.644, 0.769] | 0.682 | 0.489 | 0.407 | 0.793 | 0.809* |

Brier(cal) 0.168/0.172/0.172/0.165/0.160 · ECE(cal) 0.059/0.049/0.041/0.031/0.040. Capacity curves monotone every week (wk8: @5% 0.165 · @10% 0.264 · @15% 0.308 · **@20% 0.385** · @25% 0.440).
\* Tier-2 UA numbers are DROP-A *stratified* (not course-held-out) — a different, more optimistic CV scheme; shown only as loose context, never as a matched comparison.

**Leave-institution-out transfer (the "generalizes across institutions" claim).** Same winner, tuned on the train side; test on the other institution's R2 courses.

| Week | **train-UA → test-PUC** (pristine actas) | CI95 | rec@20% | train-PUC → test-UA | CI95 |
|------|------------------------------------------|------|---------|---------------------|------|
| 2    | **0.708** | [0.582, 0.825] | 0.35 | 0.665 | [0.585, 0.740] |
| 4    | **0.741** | [0.609, 0.857] | 0.50 | 0.661 | [0.583, 0.733] |
| 6    | **0.719** | [0.593, 0.839] | 0.45 | 0.628 | [0.548, 0.710] |
| 8    | **0.700** | [0.574, 0.815] | 0.35 | 0.695 | [0.621, 0.766] |
| full | **0.727** | [0.590, 0.851] | 0.45 | 0.686 | [0.609, 0.762] |

**The train-UA→test-PUC row is the strongest, cleanest cross-institution test** — trained on 8 UA courses, evaluated on 2 held-out PUC courses whose labels are *official actas* (not Canvas-recorded), it reaches **0.70–0.74** AUC (peak wk4 0.741). The asymmetry predicted in the pack holds: the pristine-label direction (UA→PUC) beats the noisy-label direction (PUC→UA, ~0.63–0.70). This transfer is **real but below the 0.80 target** — useful as a cold-start prior for a new institution, not a replacement for institution-specific training.

**Success-criteria read (pre-registered): NULL.** wk8 0.713 < 0.85; pooling ≤ single-institution (PUC-only 0.836, UA-only 0.725); transfer 0.70 < 0.80. → Do not adopt pooling as the headline; keep Tier-2B page as-is; deliver the predictability map. No leak flags survive scrutiny (§6).

---

## 2. Predictability analysis — which courses are learnable, and why

Per-course wk8 LOCO AUC (reference config, pooled R0, seed 42) vs course characteristics, across all 17 courses (`course_profiles.json`):

| Characteristic | Pearson r vs per-course AUC | Spearman | Reading |
|---|---|---|---|
| **prevalence (failure rate)** | **−0.39** | **−0.45** | higher failure rate → *harder* to predict |
| ceiling share (top-scorers) | +0.38 | +0.32 | more clear passers → easier |
| fails (count) | −0.36 | −0.38 | more fails → harder (co-moves with prevalence) |
| sessions/student | −0.23 | −0.03 | ~none |
| n (cohort size) | +0.09 | +0.17 | ~none |
| events/student, active-weeks, score-std | ≈ 0 | ≈ 0 | ~none |

Within-institution the prevalence effect is stronger: **PUC r = −0.73**, **UA r = −0.50** (per-course AUC vs failure rate).

**Plain-language findings (including the surprises):**
1. **The counter-intuitive one: courses with *more* failures are *harder*, not easier.** When failure is rare (5–6%), the few who fail are clearly disengaged and easy to flag; when failure is common (>40%), failing students look behaviorally similar to passing ones, so the signal blurs. This is the opposite of the usual "need positives to learn" intuition and it is the single strongest characteristic in the map.
2. **Caveat on that finding:** the high-AUC low-prevalence courses (54503 AUC 0.96 / 3 fails, 84936 0.98 / 2 fails, 55183 0.85 / 2 fails) rest on **2–3 failures each** — those AUCs are extremely noisy. The negative prevalence↔AUC correlation is *partly* an artifact of noisy estimates on tiny fail counts. It is a real tendency, not a precise law.
3. **Cohort size barely matters** (r ≈ +0.09). A 16-student course and a 130-student course are equally predictable on average — what matters is the *shape* of the failure signal, not sample size.
4. **Activity volume doesn't predict predictability** (events/student, sessions/student ≈ 0). A course being high-traffic doesn't make its failures more detectable from behavior.
5. **The R2 balanced set sits in the honest middle**: per-course wk8 AUC across the 10 R2 courses ranges 0.58 (84944, 42% prev) → 0.97 (88381, 21% prev), i.e. genuine spread even within the "eligible" band — course-level idiosyncrasy dominates.

---

## 3. Course-eligibility guide (institution-facing)

**When is a course a good early-warning candidate?** Apply the pre-registered **R2 rule** (characteristics only, never measured accuracy):

> A course qualifies for behavioral early-warning if it has **≥ 15 enrolled students**, **≥ 4 eventual failures**, and a **failure rate between 8% and 50%**, with **≥ 8 weeks** of LMS clickstream available for the strongest signal.

**What to tell a new university:**
- **Train on your own institution.** The honest finding is that models do **not** transfer for free at full strength across institutions. A model trained on another university's courses predicts your failures at ≈ **0.72 AUC** (a useful *cold-start prior*, e.g. for your first semester before you have labels), but an institution-specific model is materially better (PUC-on-PUC reaches ≈ 0.84 at week 8).
- **Data requirements:** per-student LMS clickstream with timestamps and resource types (files / assignments / quizzes / discussions / pages / modules / grades / announcements / navigation). From these we derive session rhythm, category-access mix, weekly-activity dynamics, and first-access timing — the signals that port across institutions. `interaction_seconds` is *not* required (unreliable).
- **Course selection:** avoid two failure modes. (a) **Extremely low failure rates (<8%)** give deceptively high accuracy on a handful of students — not operationally trustworthy. (b) **Very high failure rates (>50%)**, like UA course 86676 (69%), are hard *and* usually signal a labeling/gradebook quirk — handle separately. The 8–50% band is the sweet spot.
- **Timing:** useful signal exists from **week 2** (rec@20% ≈ 0.36 pooled), strengthening through **week 8**. There is no need to wait for first grades.

---

## 4. R0 / R1 context rows (Stage A, non-quotable context)

Reference config (CatBoost Balanced, top-40/fold, uncalibrated, seed-averaged over {42..46}), week 8, LOCO grouped by course (`stageA_results.json`). **Context only — the quotable R2-pooled number is the nested §1 value (0.713), not the Stage-A value.**

| Rule (wk8) | pooled AUC / per-course / rec@20% | PUC-only | UA-only |
|---|---|---|---|
| **R0** (all 17) | 0.673 / 0.760 / 0.417 | **0.800** / 0.805 / 0.673 | 0.657 / 0.769 / 0.349 |
| **R1** (evaluable, 13) | 0.650 / 0.722 / 0.377 | 0.797 / 0.809 / 0.618 | 0.677 / 0.750 / 0.319 |
| **R2** (balanced, 10) | 0.659 / 0.730 / 0.356 | *skipped (2 courses < 4 for LOCO)* | 0.725 / 0.762 / 0.369 |

The pattern is identical across all three rules: **PUC-only (0.80) ≫ pooled (0.65–0.67) ≈ UA-only (0.66–0.73)**. Pooling never wins. R2-PUC-only is unmeasurable under LOCO (only 2 PUC courses survive the balanced rule) — which is *itself* a finding: the balanced-prevalence PUC courses are too few to hold out, so the R2 comparison can only be pooled-vs-UA, and pooling loses there too.

---

## 5. R3 max-map (INTERNAL — `"quotable": false`)

Greedy forward course-subset selection by pooled LOCO AUC (reference config, seed 42, week 8), seeded from the top-2 G3 per-course AUCs. **Every artifact carries `"quotable": false` — these numbers must never appear in any sales or client material.**

- **Peak: pooled LOCO AUC 0.938 on 3 courses — {84936 (UA), 54503 (PUC), 55010 (PUC)}.** Notably *cross-institution*, but all three are **low-prevalence** (5–6%, 11 total failures across ~200 students).
- **What it teaches:** the "maximum achievable" number is an artifact of cherry-picking easy, rare-failure courses — exactly the courses the R2 rule *excludes*. Selecting courses by measured AUC would optimize for deceptively-easy low-prevalence courses and inflate the headline; the pre-registered characteristics rule (R2) deliberately refuses this. The gap between R3's 0.94 (selected-by-AUC) and R2's honest 0.71 (selected-by-characteristics) is the size of the selection-bias trap avoided. It also confirms §2: predictability is driven by low prevalence, not by institution or pooling.

---

## 6. Leak-flag characterization (transparent, not discarded)

The nested-vs-Stage-B leak guard flagged weeks {2, 6, 8, full}. **Investigated and characterized as non-leakage**, à la Tier-2's transparent handling:
- The guard compares each week's nested AUC (Optuna-tuned + 5-seed-bagged, seed-42 outer folds) to the winner's **cross-week** Stage-B mean (0.633), which is dragged down by the weak week-4 cell — a mis-scoped baseline.
- On **matched seed-42 folds**, tuning + 5-seed bagging legitimately lifts the untuned single model: wk8 0.687 → 0.713 (**+0.026**). Per-fold feature selection (corr-prefilter) and Optuna tuning run **strictly on training folds** (code-verified) — no test-label leakage.
- For the two weeks Stage-B actually measured: wk4 nested 0.616 vs 0.602 (+0.013, clean); wk8 nested 0.713 vs its own wk8 Stage-B mean 0.664 (+0.049 = tuning/bagging + a mildly favorable seed-42 partition). AUCs are modest throughout — nothing implausible that would suggest leakage.
- Per-course z-norm uses each course's own cohort statistics (required by the pack; identical treatment for both institutions and for both nested and Stage-B), so it cannot explain the nested-vs-Stage-B gap.

---

## 7. Open items (deferred — adoption is Paul's + a Fable session's call)

- **Pooling not adopted.** The pre-registered NULL stands: at institution-invariant feature granularity, PUC and UA do not share enough signal for pooling to beat single-institution models. Keep the Tier-2B `tm3-diagnostico.html` page **as-is** (PUC-primary, UA as second institution) — no changes warranted or made.
- **Cross-institution transfer (~0.72 vs pristine PUC actas) is a real, reportable asset** — position it as a *cold-start prior* for onboarding a new institution before local labels exist, explicitly not as a substitute for institution-specific training. It is below the 0.80 bar; do not headline it as "generalizes across institutions" without the qualifier.
- **The 23 institution-invariant features are the reusable artifact** (`feature_schema.json` `model_feature_cols`) — the portable behavioral signal between institutions. The 39 dropped features (`dropped_institution_leakers`) are the institution-specific ones; a per-institution model should re-include its own.
- **Predictability map is the standalone deliverable** (§2 + `course_profiles.json`): failure rate (not cohort size or activity volume) governs learnability, and moderate (8–50%) rates are the operational sweet spot. Feeds the eligibility guide (§3).
- **Not adopted / not run:** no R3 numbers in any client material (internal only); no register updates; no HTML/sales edits; no new UA data requests. The definitive UA fix remains official acta grades (Tier-1/2 open item) — until then UA labels are DROP-A (Canvas-recorded), which is exactly why the pristine-label transfer test used **PUC** as the held-out side.

---

### Artifacts (all NEW, under `data/puc/sota_results/tier3_pooled/` + `data/ua_clean/`)
`ua_clean/ua_clean_data.parquet` · `ua_cleaning_report.json` · `features/pooled_week_{2,4,6,8,full}.parquet` · `feature_schema.json` · `category_mapping.json` · `g2_build_report.json` · `course_profiles.{json,md}` · `stageA_results.json` · `stageB_results.json` · `confirmatory_results.json` · `oof_pooled_week_{2,4,6,8,full}.parquet` · logs. Scripts: `ua_clean_rebuild.py`, `common_features.py`, `tier3_common.py`, `g3_course_profiles.py`, `g4_stage_a.py`, `g5_stage_b.py`, `g6_confirmatory.py` (on `sota-tier3`, uncommitted — no commit requested).
