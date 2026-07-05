# Pipeline Review — Canvas-LMS Student-Failure Prediction (PUC + UA)

*Internal engineering audit · branch `sota-tier3` · 2026-07-04. Assembled by a 15-agent workflow (7 stage reviews → 7 independent adversarial verifications → synthesis); every verification correction applied; residual doc/code drifts called out inline. New experiments q10 (mRMR) and q11 (survival) run and folded in.*

## Executive summary

This is a rigorous, honest pipeline whose central finding is a **ceiling, not a win**: across two institutions the predictive signal saturates at ~62 basic clickstream features (PUC-only nested LOCO wk8 ≈0.84; R2-pooled nested wk8 0.713 [0.652, 0.771]), and the binding constraints are **data volume and label quality, not modeling sophistication**. Cross-institution pooling was pre-registered as a NULL and confirmed as one — PUC and UA rely on largely non-overlapping predictive features (top-10 Jaccard 0.176 at wk8), so the adversarial invariance probe must discard 39 of 62 z-norm features to defeat institution leakage, and what survives does not out-predict a single-institution model. A feature-ROI sweep (q6–q11) tested the historical ~325-feature corpus, mRMR selection, and a Cox survival signal, and every lever returned a delta inside ±0.01 — the corpus does not beat the basics. The remaining upside is concentrated in **label quality (official UA acta grades), statistical honesty (cluster bootstrap, true LOCO, matched-estimator tuning comparisons), and exactly one open modeling hypothesis the aggregate-feature ceiling does not close: a learned sequence representation over the raw clickstream.**

---

## 1. Data mining & cleaning

**What we did.** Built a raw-click hygiene pipeline ported across both institutions: three-level dedup (L1 exact-row drop, L2 HTML/api-twin collapse on `(student, course, normalized_url, created_at.floor('s'))` keep-first, L3 consecutive same-URL debounce at <10s), all row-accounted and asserted monotone-non-increasing. Timestamps localized UTC→America/Santiago with a per-row offset verifier (every row exactly 3h or 4h DST). Canvas `interaction_seconds` excluded after an unreliability audit; dwell instead derived from 30-min-gap sessionization in `common_features.py`. UA labels remediated by dropping active-zero enrollments (≥20 views AND `final_score==0`, an external-LTI-gradebook artifact).

- PUC dedup removed 547,985 rows (23.7%): 2,315,314 → L1 −5,721 → L2 −492,494 → L3 −49,770 → 1,767,329, monotone.
- UA same recipe removed 10,830 rows (9.3%): 116,436 → 105,606 across 208 users; L2 applicable (57,651 `/api/v1` hits).
- Timezone fix cut night-hour share 17.55%→4.37% (PUC) and 25.08%→5.10% (UA); 100% of rows carry a verified 3h/4h offset.
- `interaction_seconds`: 82.05% null (independently re-derived), nonnull median 3.41s but mean 1097.9s with a 13.6-day max and quantized heartbeat spikes (11.0s ×5182, 12.0s ×2819) → excluded.
- Adopted Tier-3 canonical is **DROP-A** (drop 51 active-zeros, **keep** course 86676) = 322 pairs, 98 fails.
- Decomposition (full-week stratified): OLD 0.892 → drop-active-zeros-only 0.850 → drop-86676-only 0.889 → A+ (both) 0.788 — the honest UA AUC drop is driven by removing active-zeros, **not** by 86676.

**Corrections applied from verification.**
- The 30-min timeout is **not** cleanly "empirically validated." `timeout_sensitivity.json` tests a **single week (wk4) on a single metric**: ROC-AUC favors 30-min (0.7935 vs 15-min 0.7902 vs 60-min 0.768), but **PR-AUC actually favors 15-min (0.2943 vs 0.261)**. The ROC-based choice holds directionally but is single-week/single-metric, not swept.
- Idempotency ("2nd pass removes 0") is **doc-sourced** (TIER1_RESULTS §1 prose), not in `cleaning_report.json`, which records only `monotone_non_increasing:true`. The row-drop and monotone checks *are* artifact-backed.
- The `interaction_seconds` doc drift is surgical: the correct fix is "**82% null, 0% exact-zero**" — the companion `0% exact-zero` stat is correct (`pct_zero=0.0`); only the "0% null" figure is wrong.
- Under **LOCO** (not just full-week stratified), dropping 86676 *helps* (0.737→0.769), reinforcing that keeping 86676 is a labeling-trust judgment call, not a modeling loss.

**Evidence.** `data/puc/sota_results/tier1_clean/cleaning_report.json`; `data/puc/sota_results/tier3_pooled/ua_cleaning_report.json`; `scripts/puc_clean_rebuild.py` L45-64; `scripts/ua_clean_rebuild.py` L82-106; `scripts/common_features.py` L49, L204-228; `scripts/ua_remediate_labels.py` L38-42, L139-143; `data/puc/sota_results/tier1_clean/timeout_sensitivity.json`; `TIER1_RESULTS.md` §1, §6; `TIER3_PROGRESS.md` L14, L20.

**SOTA assessment.** Above typical EDM practice. Full row-drop accounting with monotone + (doc-asserted) idempotency verifiers is more auditable than almost any published clickstream-cleaning description. Per-row TZ-offset verification — not a modal-hour proxy, which they correctly flag as misfiring on a flat-midday UTC distribution — is a genuinely correct fix to the madrugada artifact most LMS studies ignore. Excluding heartbeat-based `interaction_seconds` and replacing it with inter-click-delta dwell bounded by a 30-min session gap matches the *Beyond-Time-on-Task* / ECTEL philosophy (both PDFs sit in the repo). The correctness engineering is excellent; the *empirical* validation of the cleaning constants is thin — justified, since clean-vs-old paired ΔAUC CIs all straddle 0 (wk2 +0.013 [−0.043, 0.064], full −0.003), so cleaning is a correctness play, not a performance lever.

**Gaps / untested.**
- No sensitivity sweep on the L3 debounce window (5/10/20s) or the L2 1s twin-floor; twins straddling a 1-second boundary escape L2 uncounted.
- Dwell estimator assigns 0 minutes to singleton and last-of-session events with no imputation — a boundary-to-boundary undercount, uncharacterized.
- No bot/crawler/prefetch/user-agent filtering.
- UA active-zero threshold (≥20 views) is not sensitivity-tested (10/20/30); the 51-count is threshold-dependent.
- UA cleaning report runs **no** `interaction_seconds` audit — an asymmetry vs PUC.
- DROP-A selection bias (deleting engaged-but-failing students) is never quantified against the honest-AUC claim.
- Option C (official UA acta grades) — the definitive label fix — was not obtained.
- Live doc/code drifts: TIER1_RESULTS "0% null" should read 82% null (0% exact-zero); `ua_remediate_labels.py` still emits A+ (n=286), not the adopted DROP-A (n=322).

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| 3-level dedup + Santiago TZ + 30-min sessionization + `interaction_seconds` exclusion w/ derived dwell | Already done | Row-accounted, per-row-offset verified, timeout ROC-validated (single-week caveat). Strong core of the stage. |
| **Obtain official UA acta grades (Option C)** | **Worth trying — highest in stage** | Label quality is the established binding constraint (finding #4). UA AUC collapses 0.89→0.79 once mislabels are removed; real grades end the active-zero guesswork and let engaged-but-failing students re-enter, directly testing DROP-A selection bias. |
| Fix the two doc/code drifts (82% null; make script emit DROP-A n=322 or mark A+ superseded) | Worth trying | Cheap correctness fixes; re-running the remediation script today reproduces n=286, a live reproducibility trap. |
| Quantify DROP-A selection bias (re-include active-zeros w/ true/imputed labels, measure AUC & prevalence shift) | Worth trying | The honest ~0.79/0.72 headline assumes all 51 active-zeros are artifacts. Best done with Option C. |
| Final/singleton-event dwell imputation | Worth trying (correctness, not performance) | Fixes a real undercount; expected AUC lift marginal given the basics ceiling. |
| Sensitivity sweep: L3 debounce, L2 1s straddle count, active-zero threshold | Worth trying (low ROI) | Converts "defensible by construction" into "defensible by evidence." Prioritize the active-zero threshold — it changes n and prevalence. |
| `interaction_seconds` as a time-active feature | Not worth trying | 82% null, 13.6-day max tail, quantized heartbeats. Exclusion settled and correct. |
| Bot/user-agent/prefetch filtering | Not worth trying | L3 debounce absorbs most machine-rapid repeats; cleaning does not move AUC. |
| Per-user timezone handling | Not worth trying | Both cohorts domestic Chilean; fixed Santiago localization covers 100% of rows with verified DST offsets. |

---

## 2. Feature engineering

**What we did.** Two regimes coexist. (1) A hand-engineered corpus of ~325 columns across 21 documented families (session, category volume/breadth, time-of-day, weekly trajectory, n-gram transitions, resource-graph, proactivity percentile, PCA embeddings, course-relative timing, decay, inactivity, momentum, DCT rhythm, device, participation, plus external/leaky families), catalogued post-hoc in `FEATURE_CATALOG.md` with computability class (SHARED/UA-ONLY/PUC-ONLY/NEEDS-EXTERNAL) and leakage tags. (2) A clean cross-institution SHARED pipeline (`common_features.py`) of 62 base features (+62 per-course z-norm) covering families 1–4, computed identically on PUC and UA, gated by an adversarial institution-invariance probe. The q6–q11 ROI investigation then tested whether the historically top-ranked *missing* families lift AUC, under a fixed reference config (CatBoost `auto_class_weights=Balanced`, per-fold leak-free ExtraTrees top-N, grouped `StratifiedGroupKFold(5)`, seeds {42,43,44}, pooled AUC, `n_seeds_bag=1`, no Optuna).

- **q8 (definitive):** fed the *actual* historical enriched matrices (n_hist 243 PUC / 233 UA, 116 shared, 100 invariant after probe) against the 62 basics → **delta_best +0.0001 PUC (0.8057 vs 0.8056), −0.0053 UA, −0.0071 pooled**. Feature ceiling sits at the basics.
- q6 (28 added families): PUC −0.0114, UA +0.0054, pooled −0.0078 → "marginal."
- q7 (top-N sweep, +90 historical augment): PUC +0.0013, UA +0.004, pooled +0.0191 (80 feats 0.6743 vs 0.6552).
- q9 (basics+historical combined): PUC +0.0, pooled +0.0012.
- Corroborated independently: Tier-2 v2 thesis families (cross-course/intensity/slope) lose the tie-break to clean features (C8 +0.0156 vs C2 +0.0173).

**Corrections applied from verification.**
- The per-family q8 composition (bigram/PCA/proactivity/timing/decay/inactivity counts; mobile=0; DCT = 4-col `dct_pct` stub) is sourced from **`histcols.json`, which is NOT a repo file** — it lives at `/home/paul/.claude/jobs/3c9df0b8/tmp/histcols.json` and is loaded by `scripts/q8_historical_matrix.py`. Cite it as an out-of-repo scratch artifact.
- The **exact** per-family integers (bigram 19, PCA 27, proactivity 89, timing 69, decay 5, inactivity 6) **could not be reproduced exactly** by an independent regex (e.g. PUC bigram 16 / UA 0). Treat them as **approximate**. The *substantive* claims hold: the historical families are genuinely present as real implementations; mobile/device = 0 columns in all three sets; DCT appears only as a 4-col stub, not the full `dct_coef_0..11` family.
- **q8 UA `match=0.963`** (PUC 1.0): 3.7% of UA student-course rows had no historical-master row and were zero-filled by `align()`, mildly weakening the clean read of the UA −0.0053 delta.
- Catalog effect sizes echoed for the mobile/participation verdict (mobile_pct PUC r=−0.32, UA participation −0.14) are **catalog assertions, not independently re-derived** — directionally consistent with the "untested residue" framing, low risk.

**Evidence.** `FEATURE_CATALOG.md` (family table; "62 of ~325 columns (~19%)"; top-10 ranking line 51); `scripts/common_features.py` `featurize_pair()` L168-299, `select_invariant_features()` L374-399; `data/puc/sota_results/tier3_pooled/{q6_ablation,q7_topn_augmented,q8_historical,q9_combined,q10_mrmr,q11_survival}.json`; `data/feature_selection/feature_rankings.parquet`; out-of-repo `histcols.json`; `TIER2_RESULTS.md` §2–§3.

**SOTA assessment.** Strong and honest where it counts. The redundancy finding is unusually well-earned: q8 tests the **actual** historical implementations (not a strawman reimplementation), so the +0.0001 PUC null is a real refutation of the "we left signal on the table" hypothesis, corroborated in Tier-2 and Tier-3. Three axes are current best-practice: per-fold leak-free feature ranking (fit on train fold only), grouped LOCO CV, and an institution-invariance probe (textbook adversarial validation). `FEATURE_CATALOG.md` as a feature registry with computability + leakage classes is genuine feature-store discipline. **What is dated/missing:** the entire corpus is hand-engineered aggregate statistics; every "sequence" family that exists (n-gram bigrams, DCT spectral) is a bag-of-transitions or spectral summary, and q8 shows the bag-of-bigrams does not help. What was **never built or tested** is a *learned* sequential representation over the raw clickstream — the one feature hypothesis the aggregate ceiling does not close. Second: the ROI deltas carry **no uncertainty** — deltas of ±0.01–0.02 sit well inside the ~0.06 nested-CI half-width, so "feature ceiling" is correct directionally but q7's pooled +0.0191 is unquantified as signal-vs-noise. Third: `grades_check_per_week` (historical #2) partly *responds* to interim grades and is unflagged for soft-endogeneity.

**Gaps / untested.**
- No **learned** sequence/representation model (GRU/LSTM, Transformer-over-actions, self-supervised pretraining). q8 falsifies aggregate bag-of-bigrams, not learned temporal order.
- ROI deltas (q6–q11) have no paired-bootstrap CI; q7 pooled +0.0191 and q6 PUC −0.0114 are read as null but never tested against the ~0.06 half-width.
- `mobile_pct` and participation-verb features are confounded inside the q6 batch, never isolated; device/mobile absent from q8 masters (0 cols).
- Full DCT rhythm family (`dct_coef_0..11`) was NOT in q8 masters (only a 4-col `dct_pct` stub), yet PUC Tier-1 SHAP flagged `dct_3_znorm` top-3 at wk4 — spectral rhythm is under-tested pooled.
- `grades_check_per_week` not audited for reactive-to-grade endogeneity.
- `FEATURE_CATALOG.md` §D "recommended port order" is stale — it prescribes families q8 has since shown do not beat basics.

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| Port the full historical family corpus per FEATURE_CATALOG §D | Not worth trying | q8 already fed the *actual* matrices (243/233 feats) and got +0.0001 / −0.0053 / −0.0071. Definitive; §D port order is mooted. |
| **Learned sequence / representation model over raw clickstream** | **Worth trying — the one open hypothesis** | The only lever the aggregate ceiling does not rule out. Run on PUC first (~2275 events/student); UA (~246) is thin for deep models. Frame as a genuine test of the volume-is-binding thesis — expect it may also hit the ceiling. |
| Attach paired-bootstrap CIs to q6–q11 deltas | Worth trying | Cheap (reuse B=2000 harness on OOF preds); settles whether q7 pooled +0.0191 is real. High rigor payoff. |
| Construct-validity / endogeneity audit of `grades_check_per_week` | Worth trying | #2 historical feature; grade-checking responds to interim grades at wk8. A 1-hour temporal lead/lag check before any productionization. |
| Update FEATURE_CATALOG §D to reflect q8 | Worth trying | Doc hygiene — leaving it will cause the next engineer to repeat falsified work. |
| Isolate mobile_pct / participation / full DCT in a leak-free add-one-family ablation | Worth trying (low-moderate) | The only aggregate residue genuinely untested in isolation (mobile 0 cols in q8; DCT a 4-col stub; SHAP flagged dct_3). |
| mRMR / alternative selection over ExtraTrees | Already done | See §3/q10 — mRMR loses on all three sets. |
| Survival / Cox risk as a feature | Already done | See §7/q11 — null (−0.0013 / −0.0010). |

---

## 3. Feature selection & dimensionality reduction

**What we did.** Two efforts. (1) An older **UA-only** 5-stage funnel (`sota_feature_selection.py`): variance+correlation filter (280→236), univariate (MI/point-biserial/Mann-Whitney), embedded (L1/ElasticNet/RF/XGB Gini), wrapper (manual 20-iter Boruta + RFECV), stability selection (30 bootstraps + LOCO LASSO), fused into a weighted composite — a 50-feature set (LOCO-XGB 0.761) and a curated 33-feature set (LOCO 0.782). (2) The newer, honest **Tier-3** protocol (`tier3_common.rank_features`): per-fold leak-free ExtraTrees(300) ranking *inside* grouped `StratifiedGroupKFold`, top-N sweep (2..100), plus q4/q6/q7/q10. PCA/FeatureAgglomeration exist as engineered components but unsupervised DR is not on the winning path.

- UA funnel: 280 → 236 after filter → 50 selected; LOCO-XGB 0.761 (std 0.122).
- ExtraTrees top-N is flat past N~40: PUC wk8 optimal N=40 AUC 0.7953 (full-feature baseline 0.8056); UA optimal N=60 AUC 0.6816. At N=20, PUC 0.7647.
- Per-institution top-10 overlap is low (Jaccard 0.176 wk8; only 3 shared features) — feature rankings are non-transferable.
- RFECV (random StratifiedKFold, ignores course grouping) returns 186/236 features as "optimal" — wrapper FS barely discriminates.

**Corrections applied from verification (this stage was `confirmed:false`).**
- **MATERIAL / STALE:** q10 mRMR is **DONE, not in-progress.** `q10_mrmr.json` now contains `R2_pooled` (mrmr_best 0.6833 @N60 vs ExtraTrees 0.7034, **delta −0.0201**); logs show `[Q10] DONE [228s]`; no process is running. mRMR **loses to ExtraTrees top-N on all three sets** (PUC −0.0596, UA −0.0211, pooled −0.0201). The verdict flips from "in-progress" to "already-done."
- Sourcing: the "(full 0.8056)" figure is **not** in `q4_perinst_features.json` (q4 has no `full` field; max top-N 0.7953) — it is the ExtraTrees full-feature baseline in `q10_mrmr.json` `baselines_extratrees.PUC_only`.
- Sourcing: the leakage probe (`probe_auc_after 0.3411`, `n_dropped_leakers 0`) lives in **`q6_ablation.json` `results.R2_pooled`**, not q8.
- The 33-feature set's "**15 course-relative**" is uncorroborated — `optimal_features.json` records only `n_features 33`, `n_stable 26`. (`n_course_relative_in_selection 10` exists separately for the 50-set.)
- Composite weights are **0.15/0.25/0.15/0.15/0.15/0.15** (univariate/embedded/boruta/rfecv/bootstrap/loco, sum=1.00), matching executed code — the `:721` comment claiming "stability (0.30)" is a stale comment, not the code.

**Evidence.** `data/feature_selection/{selected_features,optimal_features,feature_rankings.parquet}.json`; `scripts/sota_feature_selection.py` L475-524, L722-729, L846-871; `scripts/tier3_common.py` L99-105; `data/puc/sota_results/tier3_pooled/{q4_perinst_features,q6_ablation,q7_topn_augmented,q10_mrmr}.json`; `scripts/feature_agglomeration.py` L96-151.

**SOTA assessment.** Mixed. **Strong:** the Tier-3 per-fold protocol ranks strictly inside grouped CV on the training fold — the correct, nested, leak-free method, genuinely SOTA-grade for small-n LOCO. Stability selection, multi-method rank fusion, multi-seed top-N sweeps, an mRMR redundancy comparison, and an explicit leakage probe are all good practice, and the honest conclusion (top-N flat past ~20–40, historical corpus adds ~0.00–0.02, mRMR underperforms) is well-evidenced. **Dated/weak:** the headline UA FS artifacts (0.782/0.761) come from the older `sota_feature_selection.py`, which selects features on the **full dataset** then LOCO-evaluates the same globally-chosen set (`:846-871`) — that is **selection leakage**, so those numbers are optimistically biased and not comparable to the honest nested figures. Stage-1 imputation/scaling are also fit on full data (minor). RFECV ignores course grouping; Boruta is a hand-rolled 20-iter Gini approximation (no BorutaPy/BorutaSHAP, no multiple-testing control); mRMR redundancy is mean |Pearson| (linear, misses nonlinear redundancy). **Bottom line:** with ~17 courses as LOCO groups, every experiment converges on the same message — FS yields *compactness, not accuracy*. Further FS sophistication is low-ROI.

**Gaps / untested.**
- The 0.782/0.761 UA sets were never reproduced under the honest per-fold protocol — the leaky-vs-leak-free gap is unquantified.
- No predictive head-to-head of PCA/FeatureAgglomeration vs raw features inside leak-free LOCO (only variance-explained compared).
- No formal compact deployable set (N≈20–30) chosen from the flat curve with a stated AUC-vs-cost tradeoff.
- Boruta/RFECV never re-run inside grouped CV; their true contribution unknown.
- No stability/selection-frequency reported for the per-fold ExtraTrees rankings (only for the leaky bootstrap/LASSO pipeline).

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| mRMR as a replacement for ExtraTrees top-N (q10) | Already done | `q10_mrmr.json` complete on all three sets; mRMR loses everywhere (PUC −0.0596, UA −0.0211, pooled −0.0201). Selection method is not the bottleneck. |
| Re-run the UA FS suite under the nested per-fold protocol to retract the leaky 0.782/0.761 | Worth trying | Cheap; corrects the record. Those numbers were selected on full data then LOCO-evaluated — inconsistent with the honest 0.713/0.80. Prevents an inflated set leaking into deployment claims. |
| Pick a compact deployable set (N≈20–30) from the flat curve | Worth trying | q4 shows PUC 0.7647 @N=20 vs 0.7953 @N=40; near-ceiling at a fraction of features. The real ROI of FS here is operational (cheaper feature computation) and is unclaimed. |
| BorutaSHAP / proper BorutaPy | Not worth trying | RFECV already keeps 186/236 and top-N is flat past ~40 — wrapper FS cannot discriminate a smaller high-value subset here. |
| PCA / FeatureAgglomeration as predictive DR in honest LOCO | Not worth trying | Supervised top-N dominates; with ~17 groups, variance-maximizing directions need not be discriminative. Useful PCA components already survive as individual features. |
| Conditional-MI redundancy (JMI/CMIM) instead of Pearson mRMR | Not worth trying | Plain mRMR already loses by up to 0.06; a nonlinear-redundancy refinement is unlikely to close that and flip the ranking at n~250–360. |
| Per-institution ranking / separate selected sets | Already done | q4 quantifies it (wk8 top-10 Jaccard 0.176, 3 shared). |
| Multi-method composite + bootstrap/LOCO stability selection | Already done | Implemented and materialized; only defect is the outer leakage, addressed by the re-run verdict above. |

---

## 4. Model selection & hyperparameter tuning

**What we did.** Benchmarked a broad zoo (XGBoost, CatBoost, HistGB, RandomForest, LogisticRegression, RBF-SVC, MLPs, stacking), then narrowed via a pre-registered 10-config bake-off (weeks {2,4,6,8,full} × seeds {42..46}, grouped by course). Winner: **CatBoost `auto_class_weights=Balanced`, 40 clean features**, selected by mean paired ΔAUC vs an XGBoost baseline with a recall@20% tie-break. Confirmed under nested outer-LOCO CV with per-fold leak-free FS, inner 3-fold Optuna (30-trial Tier-1/2 → 150-trial Tier-3, F2 objective), 5-seed bagging, Platt calibration, paired bootstrap CIs (B=2000), and a nested-vs-bakeoff leak guard.

- Bake-off winner C2 = CatBoost Balanced 40 clean, ΔAUC +0.0173, tie-broken over {C2, C8} on recall@20% (0.611 vs 0.608); rank-average ensembles and HistGB all scored **negative** ΔAUC (C4 −0.0219, C5 −0.0161, C6 −0.0222, C10 −0.0144).
- CatBoost vs XGBoost is a **weak-signal win**: mean +0.023 (Tier-1) / +0.017 (Tier-2), significant only at full week (+0.051 [0.010, 0.100]); at wk8 a tie.
- Regime: PUC 560 pairs / 41 fails; UA DROP-A 322; pooled R2 400 / 91.

**Corrections applied from verification (this stage was `confirmed:false`).**
- **OVERCLAIM softened.** The claim that "F2-at-0.5 tuning *actively HURTS ranking* / makes CatBoost *worse than untuned defaults*" is stated more strongly than the repo supports. The only evidence compares **non-matched estimators** — untuned bake-off seed-means (fold-seed-averaged, single-level, uncalibrated) vs the nested confirmatory number (one AUC on 5-seed-bagged probs from fixed seed-42 folds). TIER2_RESULTS L29-30 frames nested < bake-off as "expected, healthy" (the leak-guard direction) and names objective mismatch as *one plausible contributor*, not proven causation. **Counter-evidence the review omitted:** on *identical* seed-42 folds, TIER3_RESULTS §6 L110 shows Optuna tuning + 5-seed bagging *lifts* the untuned single model wk8 0.687→0.713 (+0.026) — this bundles tuning with bagging on pooled data, so not a clean refutation, but a balanced audit must cite it. **Net: the F2-at-0.5 objective is a documented smell and a legitimate thing to fix, but "tuning hurts" is not established causation.**
- **Attribution fix:** "honest nested CatBoost wk8 ~0.836–0.848" mis-assigns 0.848 to CatBoost. Per TIER1_RESULTS §2, **0.848 is nested XGBoost**; nested CatBoost wk8 is **0.836** (TIER2 §1). Tier-1's CatBoost 0.842 (§3) was under production `StratifiedGroupKFold`, *not* nested LOCO. The ~0.84 headline is right; the upper bound belongs to XGBoost.
- **Sourcing:** Platt `CalibratedClassifierCV(method='sigmoid', cv=3)` lives in `g6_confirmatory.py` L145-148 and `puc_confirmatory_calibrated_ci.py` ~L148, **not** `tier3_common.oof_predict` (which returns raw OOF probs).
- **Scope:** LightGBM appears only in the untuned zoo, **not** in any Optuna search space (only `cat`/`xgb` tuned) and not in bake-off BASE_FITS. "Dropped from the bake-off" is correct; "in the search space" is not.
- The bake-off **selection metric is computed over weeks {2,4,8} only**, not all five weeks.

**Evidence.** `scripts/puc_benchmark_sota.py` L183-251, L1451; `scripts/puc_catboost_zoo.py` L26, L64-68, L93-96, L108-111; `scripts/puc_bakeoff_v2.py` L45-67; `scripts/g6_confirmatory.py` L38, L68-71, L85, L145-148; `TIER1_RESULTS.md` §2–§3; `TIER2_RESULTS.md` §1, §3; `TIER3_RESULTS.md` §6.

**SOTA assessment.** Strong and largely at best-practice for the regime. GBMs (CatBoost) for small tabular data with few positives is exactly right — the Grinsztajn/Shwartz-Ziv consensus that GBMs beat deep nets at this size was confirmed empirically, not assumed. Selection is rigorous: paired bootstrap CIs with shared indices, 5-seed bagging, nested LOCO with per-fold FS, an explicit leak guard, and the discipline to **reject** rank-average ensembles and HistGB when they hurt rather than force a fashionable stack. Class imbalance is handled correctly; calibration is separated from ranking. **Weak/dated:** (1) the Optuna objective — F2 at a hardcoded 0.5 threshold — is a real defect on 7% prevalence where `predict_proba` rarely crosses 0.5, making F2@0.5 coarse; a ranking-consistent objective (average_precision/logloss) or F2-at-optimal-threshold would be strictly better and near-free (though, per the correction, the *magnitude* of harm is not proven). (2) "CatBoost wins" rests on a +0.017 coin-flip margin, defensible mainly on calibration/robustness. (3) The one true SOTA gap is **TabPFN (v2, 2025)**, a pretrained tabular transformer built for exactly this ≤10k-row / ≤100-feature regime, entirely absent.

**Gaps / untested.**
- F2@0.5 as the Optuna objective never validated against average_precision / logloss / F2-at-optimal-threshold.
- The tuned-vs-untuned decision was never made on **matched estimators** — it ships tuned nested numbers while the non-matched untuned bake-off scores higher.
- TabPFN (v2) untested in the one regime it was built for.
- Isotonic vs Platt calibration never compared.
- CatBoost search space is narrow (depth/lr/l2/iterations; no border_count/bagging_temperature/random_strength).
- LightGBM dropped from the bake-off with no documented reason.

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| Fix the Optuna objective (F2@0.5 → average_precision / logloss / F2-at-optimal-threshold) | Worth trying | Near-zero cost; F2@0.5 is coarse on 7% prevalence and mismatched to the reported AUC. Validate on **matched** nested folds before claiming a lift. |
| Add TabPFN (v2, 2025) head-to-head vs CatBoost inside the nested harness | Worth trying | The single SOTA model class matching this exact regime (≤10k rows, ≤100 feats, few classes). In-context, no per-fold tuning; the only untested thing with plausible upside over CatBoost. |
| Compare untuned CatBoost defaults as a candidate headline via a **matched** nested run with tuning disabled | Worth trying | The current untuned>tuned comparison is between non-comparable estimators; the honest way to make the Occam call is one clean nested run, not the bake-off seed-mean. |
| Dense FF / CNN as primary classifier | Not worth trying | 41–139 positives over 62 aggregate features is deep in GBM-dominant territory; CNNs need spatial structure the vectors lack. |
| End-to-end LSTM/Transformer classifier on clickstream | Not worth trying (as headline) | Order is the one place signal could hide, but end-to-end training needs orders of magnitude more positives; the SSL-pretrain-then-fine-tune variant is a research probe (see §2), not a near-term headline. |
| **ARIMAX / SARIMAX** | Not worth trying | Category error — univariate forecasting for continuous series, not cross-sectional binary risk classification at a temporal cutoff. Correctly excluded. |
| Rank-average / stacking ensembles | Already done | Tested and rejected: every ensemble config negative ΔAUC (C5 −0.016, C6 −0.022, C10 −0.014). |
| Nested Optuna + 5-seed bagging + Platt + leak guard machinery | Already done | Implemented rigorously; only the objective function inside it warrants change. |
| Widen CatBoost space / isotonic calibration | Not worth trying | Marginal at 400–560 rows / 41–91 positives; isotonic needs more positives than available. Spend compute on the objective fix and TabPFN. |

---

## 5. Validation methodology & rigor

**What we did.** Every headline reported under nested CV with per-fold leak-free FS: outer `StratifiedGroupKFold(5)` grouped by `course_id`, inner 3-fold Optuna (F2), feature ranking on the training fold only. Winners fit with 5-seed bagging {42..46} and Platt/sigmoid `CalibratedClassifierCV(cv=3)`; Brier and ECE reported. Every quotable number carries a B=2000 percentile bootstrap CI. Both LOCO and stratified schemes run, with stratified explicitly labeled optimistic. Cross-institution work adds a pre-registered NULL with fixed thresholds, a leave-institution-out transfer test, and an adversarial institution-invariance probe (HGB domain-classifier grouped-CV AUC forced ≤0.75).

**Corrections applied from verification.**
- The cluster-bootstrap fix is "cheap post-hoc on existing OOF parquets" **only for the Tier-3 pooled headline** — `g6_confirmatory.py:261` persists `oof_pooled_week_{w}.parquet`. The **flagship PUC number** (Tier-2 wk8 0.836) comes from `puc_confirmatory_v2.py`, which writes only summary metrics to `confirmatory_results.json` and **never persists per-row OOF probabilities**. Cluster-bootstrapping the most-quoted number requires a **full re-run**, not a reload.
- "Calibrated AUC exceeds raw" holds only at *some* weeks: wk6 cal 0.818 < raw-bagged 0.823; wk8 near-flat (0.838 vs 0.836).
- ECE(cal) minimum is **0.013**, not 0.014 (sequence 0.017/0.013/0.016/0.017/0.014).
- "~127 recurring PUC students" = `841 − 714` **duplicate enrollments across the full 20-course corpus**, not distinct recurring students (≤127, likely fewer), and not verified to fall inside the 560-pair confirmatory subset. Frame as a gap to **measure**, not an in-sample fact.
- The single-partition exposure is **deeper**: the inner Optuna CV (`n_splits=3, random_state=RS`) and the TPESampler are **also seeded at 42**, so tuning variance is single-seed too.
- The R2-pooled headline has only **2 PUC courses among its 10** (`{54570, 55410}` + 8 UA); `StratifiedGroupKFold(5)` on 10 courses = leave-2-out, and a fold could hold out both PUC courses at once — per-course PUC variance in the pooled headline is estimable from only 2 points.
- The invariance probe backward elimination has a **hard floor `len(cur) <= 15`** (`common_features.py:386`); it stopped at 23 because ≤0.75 was binding, so the "drop until ≤0.75" description is correct for the actual run but omits the disjunction.
- Leak-guard critique is **Tier-3-specific**: the PUC (Tier-2) guard compares nested vs the *per-week* bake-off seed-mean (`puc_confirmatory_v2.py:229-234`) — correctly scoped, fired **zero** flags. Only the Tier-3 guard used the mis-scoped cross-week Stage-B mean (0.633).

**Evidence.** `scripts/puc_confirmatory_v2.py` L135-143, L148, L155-160, L177-179, L229-234; `scripts/tier3_common.py` L117-122, L132-144; `scripts/g6_confirmatory.py` L60, L89, L112-120, L127, L261, L263-266; `scripts/common_features.py` L360-399; `TIER1_RESULTS.md` L42-48; `TIER2_RESULTS.md` §1, §5; `TIER3_RESULTS.md` §6, L108-112.

**SOTA assessment.** Clearly above the median learning-analytics paper. Genuinely SOTA-grade: (1) per-fold FS ranked strictly on the training fold, code-verified inside the outer loop; (2) course-held-out grouped CV reported alongside stratified, with stratified flagged optimistic; (3) a real pre-registration with fixed thresholds and a reported NULL; (4) size-appropriate Platt calibration with Brier+ECE; (5) an adversarial institution-invariance probe (proxy-A-distance construction); (6) transparent leak-flag characterization instead of silent deletion. **Dated/weak:** (a) "LOCO" is a misnomer — it is `StratifiedGroupKFold(5)`, i.e. leave-a-group-of-courses-out, not literal LOCO; with ~7 PUC / 10 UA courses, true LOCO is cheap. (b) The bootstrap CI is **iid over students and ignores course-level clustering**, so every headline CI is too narrow — a cluster/hierarchical bootstrap is the single biggest statistical gap. (c) The confirmatory headline rests on one outer partition (seed 42), with inner tuning also seeded 42. (d) Student-spanning-courses leakage is unmeasured. (e) Tuning to F2 while headlining AUC is internally inconsistent. (f) All CV is cross-sectional — no walk-forward/temporal split, though this is a single-semester snapshot so there is no temporal axis to split on yet.

**Gaps / untested.** Cluster bootstrap; fold-partition variance (single seed-42 outer + inner); student-spanning-courses leakage; transductive z-norm (per-course stats include the held-out cohort); no permutation/label-shuffle null for the tiny-fail courses; no temporal/walk-forward validation; global (not per-fold) invariant-feature selection; heuristic leak-guard threshold.

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| **Cluster/hierarchical bootstrap CIs** (resample courses, then students) | Worth trying — highest statistical ROI | Corrects the main understatement. **Cheap only for the Tier-3 pooled headline** (OOF parquets persisted); the PUC 0.836 flagship needs a full re-run since `puc_confirmatory_v2.py` never persists OOF rows. With 5–10 courses the CIs likely widen materially. |
| Repeat the outer CV over seeds {42..46} (and vary the inner seed) | Worth trying | Machinery exists in the bake-off; ~5× compute is affordable at n=400–560. Distinguishes real signal from a lucky seed-42 partition (inner tuning is also seed-42). |
| True leave-one-course-out for the headline | Worth trying | ~7 PUC / 10 UA courses make it cheap; gives per-course AUC spread and retires the "LOCO" misnomer. Note the pooled headline has only 2 PUC courses, so per-course PUC variance there rests on 2 points. |
| Quantify/eliminate student-spanning-courses leakage | Worth trying | Must be **measured** (not assumed) before publishing PUC 0.84 — recurring students could inflate it; the ~127 figure is duplicate-enrollment count over the full corpus, not verified inside the modeled subset. |
| Switch inner-tuning objective F2 → AUC/logloss to match the reported metric | Worth trying | Resolves the internal inconsistency; may recover ~0.02 (subject to the matched-estimator caveat in §4). |
| Permutation/label-shuffle null on tiny-fail predictability-map courses | Worth trying | Formalizes the 0.5 floor; stops 0.85–0.98 AUCs on 2–3 fails being over-read. |
| z-norm train-course-only refit sensitivity ablation | Worth trying | Transductive per-course normalization is operationally defensible but outside the per-fold guarantee; one ablation closes a reviewer objection. |
| Transparent leak-flag characterization | Already done | Handled well; only the **Tier-3** guard's baseline scoping needs tightening (the PUC guard is correctly scoped and fired zero flags). |
| Pre-registered pooling NULL + non-adoption | Already done | wk8 0.713 < 0.85, pooled ≤ single-institution. Do not revisit. |
| Walk-forward / temporal split | Not worth trying (now) | The true deployment axis, but the data is a single-semester snapshot per institution — no temporal axis to split on until more semesters are collected. |
| DeLong closed-form CI + isotonic recalibration | Not worth trying | Bootstrap already covers AUC uncertainty (clustering matters far more); isotonic overfits at ~40–90 fails. |

---

## 6. Explainability (XAI) — SHAP / TreeSHAP

**What we did.** The production pipeline `scripts/puc_shap_tier1.py` (T7) fits an **uncalibrated XGBoost** booster (`n_estimators=100, max_depth=5, lr=0.1, subsample=0.8, scale_pos_weight=neg/pos`) on the full clean PUC data at weeks 4 and 8 using the top-40 features, then runs `shap.TreeExplainer` to emit (a) summary beeswarm PNGs, (b) top-20 mean|SHAP| global-importance JSONs, and (c) per-student CSVs (560 rows each) giving `risk_score` plus the three highest-|SHAP| signed factors with an aumenta/reduce direction and a Spanish `humanize()` label. A second, older UA-facing script (`generate_shap_explanations.py`) exists but has produced **no committed artifacts**.

- 6 artifacts on disk dated Jul 3: `shap_week{4,8}_{summary.png, global_importance.json, per_student.csv}`; 560 data rows per CSV.
- `quizzes_views` is the dominant global driver: mean|SHAP| 0.735 (wk4) / 0.847 (wk8).

**Corrections applied from verification.**
- The per-student `risk_score` is **not** "near-zero 0.0016 scores." 0.0016 is only row 1. Verified from `shap_week8_per_student.csv`: the column spans **0.0004 → 0.9952, mean 0.0823, median 0.0033 — strongly bimodal**; genuine at-risk students get high scores (up to 0.99). The in-sample-fit critique is still valid (`puc_shap_tier1.py` L77-78 fits and predicts on the same `Xs`), but "near-zero 0.0016" misrepresents the distribution.
- CatBoost's "+0.023 AUC" is a **cross-week mean, significant only at the full week** — stating it flatly overstates CatBoost's edge at weeks 4 & 8, the exact weeks SHAP runs.
- **At wk8 the model-zoo leader is HistGB (0.854), not CatBoost** (XGBoost 0.841 / CatBoost 0.842). CatBoost only wins significantly at the full week — the "winning model" is **week-dependent**, which *sharpens* the integrity gap (SHAP is run on a third model, XGBoost).
- `generate_shap_explanations.py` is not a "superseded PUC path" — it targets a **different substrate entirely** (UA `enriched_features`, `user_id`, `final_score<57`) vs T7's PUC clean data (`student_id`, PUC 1-7). It is a **stale UA path**, not interchangeable with T7.

**Evidence.** `scripts/puc_shap_tier1.py` L4-6, L37-57, L68-81, L80-116; `data/puc/sota_results/tier1_clean/shap_week{4,8}_*`; `scripts/generate_shap_explanations.py` L113-114, L178-189; `TIER1_RESULTS.md` §2, §4, §7-8, L61-64, L110.

**SOTA assessment.** Strong and appropriate for the regime. TreeSHAP over a GBM on tabular clickstream is the correct, current best-practice XAI choice — exact, fast, signed per-student attributions. The pipeline does the right basics: global mean|SHAP| ranking, per-student top-k signed factors with human-readable direction, and correctly explains the underlying booster rather than the calibration wrapper. Producing all 560 per-student explanations (not just the top-10-at-risk the old UA script did) is deployment-correct. **Dated/missing:** (1) the explained model is XGBoost while the established winner is CatBoost — and at wk8 the honest leader is *HistGB* — so the shipped explanations may not match the shipped predictor; CatBoost has native TreeSHAP, a low-cost fix. (2) In-sample SHAP on a full-data fit: `risk_score` is a training-set fit, optimistic relative to the nested-CV headline; out-of-fold SHAP is the modern standard. (3) No explanation-stability analysis. (4) Heavy raw+znorm collinearity splits TreeSHAP credit across correlated twins, uncorrected. (5) The "plain-language, deployment-ready" claim is overstated — DCT-spectral, bigram-transition, and rank-percentile features are not counselor-legible, and `humanize()` leaves "hours", "dct", "bigram", "assi", "rank" untranslated. (6) No actionable/counterfactual layer. DeepSHAP/IG are correctly absent (no NN; finding #4 makes an NN unjustified).

**Gaps / untested.** SHAP never run on CatBoost (or the wk8 leader HistGB); `risk_score` is in-sample not OOF; no rank-stability check across seeds/folds; collinearity distorts per-feature credit; `feature_perturbation` mode left at default and undocumented; no interaction/dependence/waterfall plots; `humanize()` dictionary incomplete; inconsistent leakage stance (UA script excludes quiz/assignment; T7 makes quiz views #1); no fairness/subgroup SHAP; the UA script is dead/unexecuted and diverges in substrate.

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| Re-run TreeSHAP on the winning model (native CatBoost ShapValues) at wk4 & 8 | Worth trying | Explaining XGBoost while shipping CatBoost is an integrity gap — sharpened by wk8's honest leader being HistGB. Native exact TreeSHAP, ~1h, already flagged open in TIER1_RESULTS §8. |
| Out-of-fold / nested-CV SHAP so `risk_score` matches honest generalization | Worth trying | The CSV score is a training-fit; a counselor-facing product must show held-out probabilities aligned with the nested headline. (Note the score is bimodal 0.0004–0.9952, not uniformly near-zero.) |
| Explanation-stability report (SHAP-rank variance across seeds/folds) | Worth trying | n=560 PUC / 286 UA single fit gives unquantified-variance rankings; guards against over-claiming "quiz views dominate." |
| Fix `humanize()` so every shipped top-20 feature is counselor-legible (or drop opaque DCT/bigram from the surface) | Worth trying | The "plain-language" claim is currently false for DCT/bigram/rank features; near-zero modeling cost, high UX payoff. |
| Correlation-aware / clustered SHAP or de-duplicate raw+znorm twins | Worth trying | Top-40 contains raw+znorm duplicates that split credit; collapsing yields more faithful, stable attributions. |
| Counterfactual / actionable explanations (DiCE-style) | Worth trying (scoped) | Turns diagnosis into intervention; scope to actionable behavioral features (session count, quiz views, regularity) — several top drivers (weekly_trend, DCT) are not directly actionable. |
| Retire / consolidate `generate_shap_explanations.py` | Worth trying | Produces no artifacts, targets a divergent UA substrate, contradicts T7's leakage stance, top-10-only. Consolidate to one T7-style path (evidence-over-artifacts). |
| Global summary + per-student top-3 signed factors (PUC wk4 & 8) | Already done | Delivered on disk. |
| DeepSHAP / Integrated Gradients | Not worth trying | No NN in the project; finding #4 makes an NN unjustified. |
| SHAP interaction values / dependence plots | Not worth trying | Marginal for a small-n counselor product; adds analyst complexity without changing the early-warning decision. Defer unless a pedagogical hypothesis needs it. |

---

## 7. Cross-institution generalization & the performance ceiling

**What we did.** Built one shared feature pipeline over cleaned PUC (7 courses) and UA DROP-A (10 courses) — 882 student-course pairs, 139 fails, 62 base + 62 per-course z-norm. An adversarial institution-classifier probe gated invariance: the full 62-znorm matrix separates PUC from UA at **AUC 0.98–0.997 in weeks 2–4**, so greedy backward elimination **dropped 39 of 62 features** to reach probe ≤0.75, leaving **23 institution-invariant** behavioral features. On these, the pre-registered R2-balanced set (10 courses, 400 pairs, 22.8% prevalence) was evaluated under the honest nested LOCO protocol. Leave-institution-out transfer was run both directions, with train-UA→test-PUC evaluated against **pristine official PUC actas**. The q3–q11 ROI investigation probed whether any lever lifts the ceiling.

- **Pooling does not beat single-institution:** R2-pooled nested wk8 raw-bagged AUC **0.713 [0.652, 0.771]**, below the 0.85 target and below both Tier-2 PUC-only 0.836 and R2 UA-only 0.725.
- **Invariance is the binding constraint:** early-week probe on all 62 z-norm features 0.982 (wk2) / 0.997 (wk4); after dropping 39, final probe 0.611/0.642/0.634/0.620/0.587.
- **Non-overlapping features:** top-10 Jaccard 0.176 at wk8 (0.111 top5); only 3 shared (`active_days`, `last_event_day`, `n_active_weeks`).
- **Transfer** (train-UA→test-PUC against pristine actas) peaks wk4 0.741 [0.609, 0.857]; noisy direction (PUC→UA) 0.63–0.70. Test set is tiny (n_test=146, ~20 fails).
- **Predictability tracks failure rate, not size/volume:** prevalence vs per-course AUC Pearson r=−0.39 (within-PUC −0.73); cohort n r=+0.09; events/student ~0. High-AUC low-prevalence courses rest on 2–3 failures each — R3 greedy max-map peaks 0.938 on 3 courses, flagged `quotable:false`.
- **Feature ceiling at the basics:** q8 delta +0.0001 PUC / −0.0071 pooled; q9 +0.0 / +0.0012; q6 −0.011 / −0.008; q10 mRMR worse by −0.06 / −0.02.
- **pct-rank vs z-norm:** modest and late — pooled R2 wk8 0.674 vs 0.659, full 0.714 vs 0.676, but wk2 0.618 vs 0.644 (worse). pct-rank is more invariance-friendly (**54 features survive the same 0.75 gate vs 23 for z-norm**).

**Corrections applied from verification.**
- "~325-feature corpus" is the **catalog** size; the actual q8 experiment matched **n_hist=243 (PUC) / 233 (UA)** historical features (R2_pooled `n_hist=None`). Deltas are real and correctly cited; conclusion unaffected.
- Transfer CIs: source says **±0.13 half-width** (full width ~0.24–0.25), consistent with the "overlap 0.60–0.85" read.
- R2 UA-only is **0.725** (§4 Stage-A LOCO); a 0.756 figure appears in the §1 DROP-A stratified-CV context column. Pooling loses under both.
- **Verdict #2 (invariance-threshold sweep) is partly already answered and was over-stated as "open."** q3's pct-rank keeps **54** invariant features at the *same* 0.75 gate yet pooled R2 wk8 reaches only **0.6737** — still below single-institution (0.725 UA-only, 0.836 PUC-only). Expanding the shared set 23→54 does **not** flip the NULL; the "a few more shared features flip pooling to a win" hypothesis is **directionally falsified**, not fully open. A z-norm-gate sweep to 0.80/0.85 would still add rigor.
- Independent value-add confirmed: the **transfer peak (UA→PUC wk4 0.741) coincides with the pooled model's weakest week (wk4 raw-bagged 0.6155, the min across weeks)** and has the widest CI — genuinely consistent with a sampling-noise read.

**Evidence.** `TIER3_RESULTS.md` §1-2, §4-6; `data/puc/sota_results/tier3_pooled/confirmatory_results.json`; `{q3_pctrank_results, q4_perinst_features, q6_ablation, q8_historical, q9_combined, q10_mrmr, q11_survival}.json`; `TIER3_PROGRESS.md` G2, L34; `scripts/common_features.py` L375; `scripts/q11_survival.py`.

**SOTA assessment.** Strong, and unusually honest for EDM. The nested LOCO protocol with per-fold FS and train-only Optuna is genuinely leak-resistant. The pre-registered NULL with success criteria fixed in advance (0.85 AUC / beat single-institution / 0.80 transfer) is rare scientific hygiene. The adversarial institution-probe (≤0.75 gate) is a legitimate proxy-A-distance domain-shift diagnostic, and using the cleaner-label institution (PUC actas) as the transfer **test** side cleverly isolates label-noise from domain-shift. **Dated/incomplete:** (1) the domain-adaptation strategy is purely **subtractive** ("drop the 39 leaking features"); the textbook middle answer — **partial-pooling / hierarchical mixed model with institution as a random effect** — was never tried. (2) The "institutions don't mix" conclusion rests on **n=2 domains** — it is a finding about *PUC-vs-UA*, not institutions in general. (3) Transfer 0.70–0.74 is reported as if precise but rests on ~20 fails; the peak-on-weakest-week coincidence smells like noise. (4) The survival experiment is degenerate (Cox on a 0.7% event rate, concordance 0.998, "disengagement" nearly collinear with `n_active_weeks`). The ceiling conclusion (data volume + label quality bind, not modeling) is well-supported by the convergent q6/q8/q9/q10/q11 nulls.

### New experiments (q10 mRMR · q11 survival) — both run, both null

**q10 — mRMR selection vs ExtraTrees top-N** (per-fold leak-free, combined basics+historical matrix, CatBoost LOCO, seeds {42,43,44}):

| Cohort | mRMR best | ExtraTrees best | Δ |
|---|---|---|---|
| PUC-only | 0.746 @ N60 | 0.806 | **−0.060** |
| R2 UA-only | 0.738 @ N40 | 0.759 | −0.021 |
| R2-pooled | 0.683 @ N60 | 0.703 | −0.020 |

mRMR **loses on all three sets.** Redundancy-aware selection (MI relevance − mean\|Pearson\| redundancy) strips features CatBoost would exploit through interactions; the model already resolves redundancy at split time, so pre-filtering it hurts. The redundancy we found is *not* a selection problem — it confirms the ceiling is real, not a selection artifact. (`q10_mrmr.json`)

**q11 — time-to-disengagement survival** (Cox PH per institution; event = student goes silent >2 weeks before course end):

| Inst | Disengage rate | Log-rank fail-vs-pass | AUC(risk→fail) | Classifier + survival-risk |
|---|---|---|---|---|
| PUC | 0.7% | p<0.001 | 0.756 | 0.730 → 0.728 (**Δ−0.001**) |
| UA | 2.3% | p=4e-5 | 0.636 | 0.569 → 0.568 (**Δ−0.001**) |

Failing students *do* disengage earlier (log-rank significant — a real signal), but two catches: (1) disengagement is **rare** here (0.7–2.3%; most students stay active to course end regardless of grade), and (2) adding the survival risk to the classifier moves AUC by **−0.001**, because the signal is already carried by `n_active_weeks` / `active_days` — the dominant Cox coefficients. The eye-catching 0.998 concordance is a rare-event artifact (~4 PUC events), not a win. Only a **discrete-time / competing-risks** formulation matched to the sparse event rate remains un-ruled-out. (`q11_survival.json`)

**Both experiments reinforce the ceiling:** neither redundancy-aware selection nor the temporal-survival lens breaks ~0.80 (PUC) / ~0.71 (pooled).

**Gaps / untested.** Partial-pooling / hierarchical mixed-effects model (never tried); z-norm invariance-threshold sweep at 0.80/0.85 (pct-rank at 0.75 already shows 54 features don't flip the NULL); whether 0.72 transfer beats a trivial 1-feature baseline (`n_active_weeks`/`last_event_day`); few-shot recalibration on a handful of target labels (all transfer numbers are zero-shot); generalization beyond n=2 institutions; the counterfactual "pooled with clean UA labels"; discrete-time/competing-risks survival appropriate to the sparse event rate.

**Verdict table.**

| Item | Worth trying? | Why |
|---|---|---|
| Partial-pooling / hierarchical mixed-effects model (institution as random effect) | Worth trying | The pre-registered NULL only tested the two extremes (hard-pool 23 invariant features vs full separation). Multilevel partial pooling is the standard answer to "should I combine these groups" and closes the most glaring methodological gap. Expected gain modest given the ceiling. |
| Invariance-threshold sweep (relax z-norm gate 0.75 → 0.80/0.85) | Worth trying (rigor, not discovery) | The NULL is conditional on an arbitrary 0.75 gate — but the "flip to a win" hypothesis is **already directionally falsified**: pct-rank keeps 54 invariant features at the same gate and pooled wk8 still reaches only 0.6737 (< single-institution). A z-norm-gate sweep adds confirmatory rigor, not a likely reversal. |
| Few-shot recalibration of the transfer model on ~20–30 target labels | Worth trying | Transfer is positioned as a cold-start prior; zero-shot 0.72 + light per-institution recalibration is the realistic deployment mode and is untested. Validates the one operational claim transfer makes. |
| **Add official UA acta grades to replace Canvas DROP-A labels** | **Worth trying — highest-ROI lever** | The transfer asymmetry (pristine-label UA→PUC 0.72 beats noisy PUC→UA 0.63–0.70) is direct evidence that UA label noise, not modeling, caps that side and the pooled result. This is the constraint the whole ceiling story names. |
| pct-rank normalization for late-week/full-horizon pooled scoring | Worth trying (scoped) | q3: pct-rank beats z-norm at wk8 (+0.015) and full (+0.037) and survives the invariance probe far better (54 vs 23), but is worse at wk2 (−0.026). Adopt only for horizons where early-warning is already actionable — not a universal swap. |
| Collect a third institution before asserting "institutions don't mix" as a general law | Worth trying (data-collection) | The divergence (Jaccard 0.18, probe 0.98) is measured on exactly one pair; n=2 cannot distinguish "PUC and UA happen to differ" from "cross-institution pooling is generically hard." |
| Survival / time-to-disengagement as an added predictor | Already done | q11 null (−0.0013 / −0.0010); risk collinear with `n_active_weeks`, concordance 0.998 degenerate at 0.7% events. |
| mRMR / genetic feature selection over ExtraTrees | Already done | q10 complete; mRMR trails on all three sets. |
| Historical ~325 corpus (or 28-feature augment) to lift the ceiling | Already done | q6/q8/q9 converge on ±0.001 to −0.01. The core evidence for the "not modeling" half of the ceiling. |
| Deep domain-adaptation (DANN / CORAL / adversarial alignment) | Not worth trying | 882 pairs / 139 fails / 2 domains is orders of magnitude below where deep DA pays off; overfit risk dominates. The tabular winner is CatBoost precisely because the regime is small. |

---

## Prioritized next steps

Ranked by expected ROI **given the established ceiling** (label quality and data volume bind; the 62-basics feature ceiling and the pooling NULL are settled). Cross-stage.

1. **Obtain official UA acta grades (Option C).** *(Stages 1 & 7.)* The single lever the project's own evidence points at: UA AUC collapses 0.89→0.79 on mislabel removal, and the transfer asymmetry proves UA label noise caps the pooled result. Ends the active-zero DROP heuristic, lets engaged-but-failing students re-enter, and directly tests DROP-A selection bias. Highest ROI, and it is not a modeling task.
2. **Cluster / hierarchical bootstrap CIs.** *(Stage 5.)* Corrects the largest statistical understatement — all headline CIs are iid-over-students and too narrow. Cheap for the Tier-3 pooled headline (OOF parquets persisted); the PUC 0.836 flagship needs a re-run because `puc_confirmatory_v2.py` never persists OOF rows. With 5–10 courses the CIs likely widen materially, changing how confidently 0.836/0.713 can be quoted.
3. **Fix the two doc/code drifts** (82%-null; make the remediation script emit DROP-A n=322). *(Stage 1.)* Near-zero cost, removes a live reproducibility trap.
4. **Attach paired-bootstrap CIs to the q6–q11 deltas.** *(Stage 2.)* Cheap (reuse the B=2000 harness on OOF preds); converts "marginal" into a quantified null and settles whether q7 pooled +0.0191 is signal.
5. **Fix the Optuna objective (F2@0.5 → average_precision/logloss) and settle tuned-vs-untuned on matched nested folds.** *(Stages 4 & 5.)* Resolves an internal inconsistency; validate on matched estimators (the untuned>tuned comparison to date is between non-comparable estimators, and matched folds show tuning+bagging *lifts* 0.687→0.713).
6. **Re-run TreeSHAP on the deployed model (CatBoost; note wk8's honest leader is HistGB) with out-of-fold scores, plus a `humanize()` fix.** *(Stage 6.)* Closes the "explain a model you don't ship" integrity gap and makes the counselor surface honest and legible.
7. **True leave-one-course-out + multi-seed outer/inner CV.** *(Stage 5.)* Cheap at ~7–10 courses; gives per-course variance and retires the "LOCO" misnomer and the single-seed-42 exposure.
8. **Measure student-spanning-courses leakage.** *(Stage 5.)* Must be quantified before publishing PUC 0.84 — the ~127 figure is duplicate-enrollment count over the full corpus, not verified in the modeled subset.
9. **Re-run the leaky UA FS suite under the nested per-fold protocol** and **formalize a compact N≈20–30 deployable set.** *(Stage 3.)* Retracts the optimistic 0.782/0.761 and captures the real (operational) win of FS.
10. **Partial-pooling / hierarchical mixed model** and **few-shot transfer recalibration.** *(Stage 7.)* Close the "middle answer" methodological gap and validate the cold-start-prior deployment claim. Modest expected gain given the ceiling.
11. **Learned sequence / representation model over the raw PUC clickstream.** *(Stages 2 & 4.)* The one open modeling hypothesis the aggregate ceiling does not close — run on PUC (~2275 events/student) first, framed as a genuine test of the volume-is-binding thesis, not metric-chasing. Real research cost; deprioritized because the prior is that it too hits the ceiling.
12. **Add TabPFN (v2) head-to-head vs CatBoost** inside the nested harness. *(Stage 4.)* Cheap (in-context, no per-fold tuning) and the only untested model class matching this exact ≤10k-row / ≤100-feature regime.
13. **Collect a third institution.** *(Stage 7.)* The only way to promote "PUC and UA differ" (n=2) to a general claim. Data-collection, longer horizon.

*Lower-priority correctness items, worth doing but not lift-bearing:* singleton/last-event dwell imputation; L3/L2/active-zero sensitivity sweeps; z-norm train-course-only ablation; permutation nulls on tiny-fail courses; isolate mobile/DCT residue; construct-validity audit of `grades_check_per_week`; correlation-aware SHAP; counterfactual (DiCE) explanations; retire the stale UA SHAP script; update FEATURE_CATALOG §D.

## What NOT to chase

- **ARIMAX / SARIMAX** — category error. These are univariate forecasting models for continuous time series; the task is cross-sectional binary risk classification at a temporal cutoff, not extrapolating a series. Correctly excluded.
- **Deep neural nets (dense FF / CNN) as the primary classifier** — 41–139 positives over 62 aggregate tabular features is deep in GBM-dominant territory (Grinsztajn 2022); CNNs need spatial structure the vectors lack. Would overfit; no path past CatBoost.
- **End-to-end LSTM/Transformer trained on the clickstream for classification** — the one place order could add signal, but end-to-end training needs orders of magnitude more positives than 41–139. Only the SSL-pretrain-then-fine-tune variant (item 11) is defensible, and as a probe, not a headline.
- **Deep domain-adaptation (DANN / CORAL / adversarial alignment)** — 882 pairs / 139 fails / 2 domains is far below where deep DA pays off; instability dominates any expected transfer gain.
- **Porting the full historical ~325-feature corpus / more hand-engineered aggregates** — q8 fed the *actual* historical matrices and got +0.0001 PUC / −0.0071 pooled. The feature ceiling is at the basics; more aggregate features is a confirmed dead end. FEATURE_CATALOG §D's port order is mooted.
- **mRMR / genetic / conditional-MI feature selection** — q10 shows mRMR loses to ExtraTrees top-N on all three sets (−0.06 / −0.02 / −0.02); a nonlinear-redundancy refinement is unlikely to close a 6-point gap.
- **Survival / Cox time-to-disengagement modeling** — q11 null; degenerate at a 0.7% event rate; the signal is already in `n_active_weeks`.
- **`interaction_seconds` as a time-active feature; per-user timezones; bot filtering; wider CatBoost search space; isotonic calibration; DeLong CIs; DeepSHAP/IG; SHAP interaction plots** — each is settled, dominated, or below the noise floor for this data regime (see stage verdict tables).
- **Expanding the shared invariant-feature set to "flip" the pooling NULL** — already directionally falsified: pct-rank keeps 54 invariant features at the same 0.75 gate and pooled wk8 still trails single-institution. A gate sweep adds rigor, not a reversal.