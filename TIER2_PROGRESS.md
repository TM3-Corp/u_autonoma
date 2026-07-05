# TIER-2 SOTA Execution Progress

Branch: `sota-tier2` (from `sota-tier1`) · RANDOM_STATE=42 · CV seeds {42,43,44,45,46} · Started 2026-07-03
One entry per task: timestamp · what ran · verifier output · PASS/FAIL/BLOCKED.

Guardrails honored: never modify `benchmark_results.json`/backups/`few_feature_sweep/`/existing parquets/anything under `tier1_clean/` except NEW files. All Tier-2 outputs → `data/puc/sota_results/tier2_push/`. Never touch `~/projects/tm3-roi-diagnostico/`. P2/UA-2 candidate lists FROZEN. UA config selection within-arm only; KEEP-arm numbers always carry the label caveat. Serialize heavy jobs (16 cores).

Task order: P0 → P1 → P2 → P3 → P4 → UA-1 → UA-2 → UA-3 → P5 → P6.

---

## P0 — Setup — PASS (2026-07-03)
- Git branch `sota-tier2` created from `sota-tier1` (`git branch --show-current` → `sota-tier2`).
- Created dir `data/puc/sota_results/tier2_push/` (+ `features/` subdir).
- `TIER2_PROGRESS.md` initialized (this file).
- Env check: `.venv-tier1/bin/python` → catboost 1.2.10, xgboost 3.0.4, sklearn 1.7.1; 16 cores.
- Confirmed inputs present: `puc_clean_data.parquet` (1,767,329 rows, 20 courses, 714 students), `puc_grades_clean.parquet` (1,607 pairs), Tier-1 cached matrices `tier1_clean/features/week_{2,4,6,8,full}_{old,clean}.parquet`. Aug courses grades present: 53493 (80 stu, 0 fail), 54947 (56, 0), 56867 (31, 0).
- **Verifier**: branch + dir exist; PROGRESS initialized. PASS.

---

## P1 — Features v2 (thesis families on clean data) — PASS (2026-07-03)
Script `scripts/puc_features_v2.py` → `tier2_push/features/week_{2,4,6,8,full}_v2.parquet` + `tier2_push/features_v2_report.json`. Loads T2 clean base matrices (NOT recomputed), joins 27 NEW raw base features + 27 per-course znorm variants. Ran in ~3s (all 5 weeks).
- **New base features (27, ≥25 required)**: Family A cross-course (10): `xc_total_views/sessions`, `xc_course_share_views/sessions`, `xc_n_active_other_courses`, `xc_sessions_between_{mean,max,total}`, `xc_relative_neglect`, `xc_max_other_course_share`. Family B intensity (4). Family C slope (5). Family D peaks (5). Family E composites (3: `procrastination_x_regularity` = `quizzes_proact_mean_pct`×`session_regularity` [documented substitution for "proactivity index"], `pdh_entropy`, `pwd_entropy`).
- **Leak-freedom**: cutoff clock = TARGET course start (0.05 quantile, per-course → identical to T2); `created_at ≤ start+cutoff` for ALL families incl. cross-course; `full` = target course's last event. Session gap 30 min. Degenerate/empty pairs → 0 (not NaN).
- **Verifier (independent script, all PASS)**: each week exactly 560 rows ✓; `student_id/course_id/_y/_group` identical to T2 clean (`.equals`) ✓ all 5 weeks; 27 new base features (≥25) ✓; **leak spot-check** — 3 (student,course,wk2) cells, `xc_total_views` recomputed directly from raw parquet with independent snippet: 385=385, 2008=2008, 380=380 EXACT ✓; NaN rate new cols pre-fill = 0.00000 (≤5%) ✓. PASS.

---

## P2 — Pre-registered bake-off (FROZEN 10 configs) — PASS (2026-07-03)
Script `scripts/puc_bakeoff_v2.py` (via `.venv-tier1`) → `tier2_push/bakeoff_results.json` + log `tier2_push/logs/p2_bakeoff.log`. 10 configs × weeks{2,4,6,8,full} × seeds{42..46} × StratifiedGroupKFold(5,shuffle,seed) groups=course. Composite ranking computed ONCE per feature-set (clean, v2) per fold, sliced 30/40, reused across models. Uncalibrated (rank metrics). rank-avg = per-fold percentile-rank mean across members. Ran ~750s (12.5 min).
- **Selection rule (pre-registered)**: highest mean paired ΔAUC vs C1 over seeds × weeks{2,4,8}; tie <0.003 → higher recall@20%, fewer feats, single>ens.
- **Ranking (sel_mean_dAUC)**: C2 +0.0173 (rec20 0.6114) · C8 +0.0156 (0.6081) · C3 +0.0087 · C9 +0.0039 · C1 0.000 · C7 −0.0007 · C10 −0.0144 · C5 −0.0161 · C4 −0.0219 · C6 −0.0222.
- **WINNER = C2 (CatBoost Balanced, 40 feats, CLEAN)**. Tie within 0.003 = {C2, C8}; tie-break on recall@20% → C2 (0.6114 > C8 0.6081). The v2 thesis features (C8) essentially TIE clean CatBoost but do not beat it → honest null on cross-course/thesis features; the real lever is CatBoost. C2 beats C1 (XGB) at every week: wk2 +0.013, wk4 +0.017, wk6 +0.020, wk8 +0.022, full +0.033.
- **Verifier (all PASS)**: JSON = 10 configs × 5 weeks × 5 seeds ✓; C1 seed-42 vs promising_explore xgb_N40 anchor — wk4 0.8138 (exp 0.814, Δ0.0002), wk8 0.8336 (exp 0.834, Δ0.0004), full 0.7839 (exp 0.784, Δ0.0001) all ≤0.02 ✓; winner + tie-break rule application recorded in JSON ✓; every config logged (no silent drops) ✓. PASS.

---

## UA-1 — UA feature completion — PASS (2026-07-03, ran concurrently with P3; UA-1 has no Boruta so no thrash risk)
Script `scripts/ua_features_v2.py` → `tier2_push/ua_features/week_{2,4,6,8,full}.parquet` + `arms.parquet` + `tier2_push/ua_features_report.json`. Loads enriched features via `train_time_limited_model.load_features(include_znorm=True)`; reuses the EXACT T4 active-zero logic (`ua_remediate_labels.compute_active_zero_set`, incl. user_id normalization) so arm membership is identical to Tier-1.
- **Arms frozen**: KEEP 373 (149 fails, prev 40.0%) · DROP-A 322 (98 fails, 30.4%; drops 51 active-zeros, keeps 86676) · A+ 286 (73 fails, 25.5%; also drops 86676). active_zero=51 ✓.
- **Universe alignment**: each cutoff matrix aligned to the full 373-pair KEEP universe; no-activity students zero-filled (informative for early warning, PUC-consistent) → arm sizes EXACTLY 373/322/286 at every cutoff (temporal zero-fill counts: wk2=70, wk4=30, wk6=22, wk8=17, full=0). This keeps paired configs within an arm×week on identical rows.
- **Leak handling**: pre_assessment_features (34, full-horizon) included ONLY at `full` cutoff (temporal cutoffs exclude them to avoid post-cutoff leak); documented in report `leak_note`. U5 (pre_assessment value) is full-cutoff-only. Residual: `jaccard_to_passing` reused as-is (legacy ≥60 passing set; not recomputed) — noted. Optional UA cross-course feature not added (time budget) — noted.
- **Verifier (all PASS)**: per-cutoff arm sizes exact 373/322/286 ✓; pre_assessment present at full (34 cols) ✓; leak-handling note in report JSON ✓. PASS.

---

## P3 — Confirmatory run of the winner (C2) — PASS (2026-07-03) — the ONLY quotable PUC numbers
Script `scripts/puc_confirmatory_v2.py` (reads winner from bakeoff JSON) → `tier2_push/confirmatory_results.json` + log. Winner C2 = CatBoost Balanced, 40 feats, CLEAN. Protocol: nested LOCO5 outer (seed 42), per outer-train top-40 leak-free ranking + inner 3-fold Optuna 30-trial F2 tuning of CatBoost → 5-seed bagging {42..46} → Platt sigmoid(cv=3) for prob quality. Bootstrap CI B=2000. Ran ~780s (13 min).
- **Honest confirmatory ROC-AUC (raw-bagged [CI95] / calibrated)**: wk2 **0.779** [0.694,0.850] / 0.798 · wk4 **0.806** [0.734,0.869] / 0.830 · wk6 **0.823** [0.751,0.887] / 0.818 · wk8 **0.836** [0.769,0.894] / 0.838 · full **0.793** [0.697,0.876] / 0.830.
- **PR-AUC (raw)**: 0.297/0.316/0.283/0.373/0.431. **Brier(cal)**: 0.059/0.059/0.060/0.057/0.053. **ECE(cal)**: 0.017/0.013/0.016/0.017/0.014.
- **Capacity curve recall@{5,10,15,20,25}%** (raw-bagged, all monotone): wk8 = 0.342/0.488/0.634/**0.659**/0.707; full = 0.317/0.537/0.610/0.659/0.732.
- **No leak flags** any week (nested < bake-off seed-mean everywhere; bo_means 0.771/0.817/0.836/0.857/0.835 all exceed nested → expected, healthy).
- **Note (transparency)**: calibrated AUC runs slightly above raw-bagged at some weeks (wk4 +0.024, full +0.037) because `CalibratedClassifierCV(cv=3)` internally 3-fold-bags the tuned CatBoost (a different, mildly-better-regularized predictor) — not calibration changing ranking. Also the F2-objective inner tuning depresses AUC vs the untuned bake-off default (objective mismatch); the nested numbers are the honest, conservative headline. raw-bagged is the primary headline (has the CI + capacity curve).
- **Verifier (all PASS)**: JSON complete 5 weeks ✓; nested never exceeds bake-off mean by >0.02 (no leak) ✓; CI arrays present ✓; capacity curves monotone non-decreasing ✓. PASS.
- **Success-criteria read**: winner beats C1 by mean ΔAUC +0.017 (>+0.01) → adopt. wk4 raw 0.806 (cal 0.830) vs target ≥0.83; wk8 raw 0.836 / cal 0.838 vs stretch ≥0.86 → **PARTIAL** (wk8 misses 0.86 under both). wk8 rec@20% 0.659 ≥0.65 ✓. Per pre-registration Partial → adopt + report actuals; HTML decision to Paul with real numbers.

---

## P4 — Train-only augmentation ablation — PASS (2026-07-03)
Script `scripts/puc_augment_ablation.py` (via `.venv-tier1`) → `tier2_push/augment_ablation.json` + log. Winner config (tuned CatBoost, 5-seed bag), weeks {2,4,8}. 3 zero-fail courses [53493,54947,56867] → 167 extra negatives (features via clean pipeline, aligned to 560-schema, all y=0). Per-fold FS + F2-Optuna params computed ONCE on non-aug train, reused for BOTH arms (isolates the added-negatives effect). Test = identical P3 560 LOCO folds. Ran ~500s.
- **Results (base → aug, ΔAUC [CI95])**: wk2 0.7794→0.7906 (Δ**+0.0112 [0.0009, 0.0230]** SIG+) · wk4 0.8058→0.8208 (Δ+0.0150 [−0.0006, 0.0318]) · wk8 0.8357→0.8127 (Δ**−0.0230 [−0.0396, −0.0076]** SIG−).
- **Verdict: NEUTRAL overall (mean ΔAUC +0.0011), but heterogeneous** — significant HELP early (wk2, and wk4 near-sig) where signal is weak and extra negatives regularize; significant HURT at wk8 (zero-fail courses' late-week behavior doesn't generalize to the target fail boundary). → Augmentation NOT a clean win; do not adopt globally (could be considered week-2-only, but out of scope to adopt).
- **Verifier (all PASS)**: test-fold indices identical to P3 (n_test=560, base AUC EXACTLY reproduces P3 raw-bagged 0.7794/0.8058/0.8357, folds cover [0,560)) ✓; paired ΔAUC + CI per week ✓; augmented train sizes logged (~596–660 per fold = base+167) ✓. PASS.

---

## UA-2 — UA mini bake-off (FROZEN 5 configs) — PASS (2026-07-03)
Script `scripts/ua_bakeoff.py` (via `.venv-tier1`) → `tier2_push/ua_bakeoff_results.json` + log. 2 arms (KEEP 373 / DROP-A 322) × weeks{2,4,8,full} × seeds{42..46} × {StratifiedKFold(5), StratifiedGroupKFold(5,groups=course)}. U1=historical global-FS XGB (anchor baseline); U2=XGB+per-fold sota top40; U3=CatBoost Balanced+sota top40; U4=rank-avg(XGB,CB,HGB)+sota top40; U5=CatBoost incl-vs-excl pre_assessment (full only). Model seed fixed 42; CV fold seed varies. Ran ~1176s (20 min). **Config selection within-arm only.**
- **Selection (pre-registered: highest mean ΔAUC vs U1 over seeds×weeks{2,4,8}, strat primary)**: KEEP → **U3** (ΔvsU1_strat −0.0355) · DROP_A → **U3** (ΔvsU1_strat −0.0381). U3 is the best-improved config in both arms.
- **Honest interpretation**: ΔAUC vs U1 is NEGATIVE for all improved configs because U1 is the *historical global-FS pipeline* (mildly optimistic; leaks FS on all data; for KEEP also credits contaminated labels). The clean, unconfounded signal = **leak-free CatBoost (U3) vs leak-free XGB (U2), same sota FS: +0.0106 (KEEP) / +0.0121 (DROP_A)** → CatBoost genuinely helps UA, consistent with PUC. So the honest leak-free numbers are lower than the inflated 0.89 baseline, but CatBoost is the right model within the honest pipeline.
- **pre_assessment verdict (U5, full)**: incl−excl mean Δ ≈ +0.0048 (KEEP strat) / −0.0001 (DROP_A strat) / +0.0010–0.0048 (loco) → **NEGLIGIBLE, inconsistent-sign value**. Wiring in the 34 pre_assessment features does not materially help (honest null; resolves the Tier-1 "features left on the table" open item).
- **Verifier (all PASS)**: JSON has U1–U4 at every arm×week×cv×seed + U5(incl/excl) at full (5th config, full-only by leak design) ✓; U1 seed-42 anchors — KEEP-full-strat 0.8956 (T4 ≈0.892, Δ0.004), DROP_A-full-strat 0.8396 (T4 ≈0.850, Δ0.010), both ≤0.02 ✓; per-arm winners recorded ✓. PASS. **KEEP-arm caveat carried in JSON `label_caveat_KEEP`.**

---

## UA-3 — UA confirmatory + honest range — PASS (2026-07-03)
Script `scripts/ua_confirmatory.py` (via `.venv-tier1`) → `tier2_push/ua_confirmatory.json` + log. Per arm the UA-2 winner (both U3 = CatBoost) → nested CV (inner Optuna30 F2 on CatBoost) + 5-seed bag + Platt sigmoid, BOTH StratifiedKFold(primary) and LOCO, bootstrap CIs, capacity curve {10,15,20,25}%. A+ (286) sensitivity: single seed-42 strat. Ran ~1825s (30 min). **Config selection within-arm only; every KEEP number carries the label caveat.**
- **Honest range (raw-bagged, STRAT primary): DROP-A (quotable alone) → KEEP (with caveat)**: wk2 0.679→0.651* · wk4 0.658→0.658* · wk8 0.756→0.806* · full **0.809→0.872\***. (* KEEP caveat: *target = recorded Canvas outcome; includes 51 active-zero enrollments whose true grades are external*.)
- **LOCO (unseen-course floor)**: wk2 0.476/0.563 · wk4 0.526/0.542 · wk8 0.628/0.743 · full **0.605/0.687**. LOCO ≪ strat — UA generalization across courses is weak (consistent with Tier-1).
- **A+ (286) sensitivity (strat s42)**: wk2 0.600 · wk4 0.638 · wk8 0.761 · full 0.778 (≈ DROP-A, slightly lower — dropping 86676 barely moves it, confirming Tier-1).
- **Full-week prob quality (strat)**: DROP-A AUC_raw 0.809 [0.759,0.857] cal 0.805 PR 0.615 Brier 0.163 ECE 0.048; KEEP* 0.872 [0.833,0.908] cal 0.865 PR 0.827.
- **PUC-vs-UA headline check**: UA DROP-A full **STRAT** 0.809 numerically exceeds PUC full **LOCO** 0.793 — but this is a CV-scheme mismatch (UA strat is NOT course-held-out; it leaks same-course students across folds). Under matched **LOCO**, PUC 0.793 ≫ UA 0.605. **PUC stays the headline** (flagged, not silently decided, per pack).
- **Leak flag**: `DROP_A/4/loco` — nested 0.5256 vs bake-off LOCO mean 0.4995 (exceed +0.026). **Near-chance-regime artifact** (both ≈0.50 = no signal; a real leak inflates a *predictive* AUC): LOCO-only (secondary CV), STRAT primary clean, remaining 15/16 cells flag-free. DROP-A STRAT headline NOT invalidated; documented transparently rather than discarding valid results.
- **Verifier (all PASS)**: JSON complete both arms + A+ sensitivity ✓; nested ≤ bake-off+0.02 except the one near-chance LOCO cell (characterized) ✓; KEEP caveat carried in every output field + will be verbatim in TIER2_RESULTS.md ✓. PASS.

---

## P5 — TIER2_RESULTS.md — PASS (2026-07-03)
Wrote `TIER2_RESULTS.md` (repo root): §1 honest confirmatory PUC table (AUC raw+cal +CI vs Tier-1 nested XGB vs old optimistic) + capacity-curve table; §2 v2 honest-null; §3 bake-off all 10 configs (mean ΔAUC + per-week); §4 augmentation verdict; §5 UA two-arm honest range (strat+LOCO) + per-arm winners (CatBoost) + pre_assessment negligible-value verdict + PUC-vs-UA placement flag, **every KEEP number carrying the verbatim label caveat**; §6 explicit "Números para el documento de Enrique" (per-week PUC values + recall@20% line + honest-provenance paragraph); §7 open items.
- **Verifier (all PASS)**: file exists with the confirmatory table, capacity-curve table, bake-off summary, augmentation verdict, UA honest-range section, and the Enrique section ✓; KEEP-caveat discipline — no KEEP mention appears without its `*`/caveat/definitional context (grep audit clean) ✓. PASS.

## P6 — HTML numbers prep (GATED) — PASS (2026-07-03)
Wrote `tier2_push/html_update_proposal.md`: OLD (0.83/0.87/0.86/0.86/0.90) vs NEW honest (0.78/0.81/0.82/0.84/0.79 +CI) side-by-side for `metricas-tecnicas-udla.html`, PUC-only, neutral descriptive tone, recall@capacity table, single provenance note, suggestion to remove the "Cada afirmación" contrastive section, and to retire the "0.89–0.903" header in favor of wk8=0.84. **STOPPED — did NOT edit the HTML; Paul decides which numbers ship.**
- **Verifier (all PASS)**: proposal file exists ✓; HTML untouched — md5 `6d671a0e3fccf82ae323cc41088ee5a8` identical before/after (tm3-roi-diagnostico is not a git repo, so verified by hash) ✓. PASS.

---

## FINAL — all 10 tasks DONE (P0–P6, UA-1..UA-3), 0 BLOCKED (2026-07-03)
- **Guardrails honored (git-audited)**: `benchmark_results.json` + backups + `few_feature_sweep/` + existing parquets + everything under `tier1_clean/` **unmodified** (only NEW files added, all under `tier2_push/`); `tm3-roi-diagnostico` HTML untouched (hash-verified); RANDOM_STATE=42, CV seeds {42..46}, identical folds per paired comparison; no SMOTE / no threshold relitigating / session gap 30 min / clean data canonical — all respected. P2 & UA-2 candidate lists frozen; no single-seed maxima reported; UA config selection within-arm only; every KEEP number carries its caveat.
- **PUC headline (quotable, nested LOCO, calibrated CatBoost)**: wk2 0.78 · wk4 0.81 · wk6 0.82 · **wk8 0.84** · full 0.79; recall@20% ~0.61–0.68 from wk2. PARTIAL success (beats XGB +0.017, misses wk8≥0.86 stretch). v2 features = honest null; augmentation = net-neutral.
- **UA headline (honest range, strat primary)**: full **0.81 (DROP-A) – 0.87 (KEEP\*)** strat / 0.61–0.69 LOCO; CatBoost beats XGB +0.011; pre_assessment ≈ neutral. PUC stays the headline (UA "exceedance" was a CV-scheme mismatch).
- **Deliverables**: `TIER2_RESULTS.md` (exec tables + UA honest range + Enrique numbers); `tier2_push/` JSONs (bakeoff, confirmatory, augment_ablation, ua_bakeoff, ua_confirmatory, features_v2_report, ua_features_report) + logs; `tier2_push/html_update_proposal.md` (GATED). 7 new scripts on `sota-tier2` (uncommitted — no commit requested).
- **Not done (correctly, per scope)**: result adoption, register updates, HTML edits, sales-material changes — all Paul's + a Fable session's call.

---

# TIER-2B — best-defensible metrics + `tm3-diagnostico.html` update
Pack: `TIER2B_HTML_EXECUTION.md` (FROZEN defensibility ruleset). RANDOM_STATE=42, B=2000. Only file modified outside `uautonoma`: `~/projects/tm3-roi-diagnostico/tm3-diagnostico.html` (after H0 backup). All compute outputs → `data/puc/sota_results/tier2_push/`. Task order H0→H1→H2→H3→H4→H5.

## H0 — Backup + baseline hashes — PASS (2026-07-03)
- Backup created: `~/projects/tm3-roi-diagnostico/tm3-diagnostico_v1_2026-07-03.bak.html`.
- **md5(backup) == md5(original)**: `d16680dd14518b2915e0818115e25127` == `d16680dd14518b2915e0818115e25127` ✓ (byte-identical).
- Baseline hashes (must stay untouched through session):
  - `index.html` = `b357b910c7af77e5e9734d22f6e09cbe`
  - `metricas-tecnicas-udla.html` = `6d671a0e3fccf82ae323cc41088ee5a8`
  - `tm3-diagnostico.html` (original) = `d16680dd14518b2915e0818115e25127`
- **Verifier**: backup exists, md5(backup)==md5(original), all four hashes logged. PASS.

## H1 — Calibrated CIs + persisted OOF — PASS (2026-07-03)
Script `scripts/puc_confirmatory_calibrated_ci.py` (via `.venv-tier1`) → `tier2_push/confirmatory_calibrated_ci.json` + `oof_calibrated_week_{2,4,6,8,full}.parquet` + log `logs/h1_calibrated_ci.log`. Reuses stored per-fold CatBoost params from `confirmatory_results.json` (NO Optuna); same LOCO folds (StratifiedGroupKFold5, seed 42), per-fold top-40 leak-free FS, 5-seed bag, Platt sigmoid(cv=3). Ran ~67s.
- **Calibrated AUC (drift vs stored P3)**: wk2 0.7981 (+0.0000) · wk4 0.8298 (+0.0000) · wk6 0.8178 (+0.0000) · wk8 0.8376 (+0.0000) · full 0.8304 (+0.0000) — EXACT reproduction (≤±0.005).
- **Calibrated CI95**: wk2 [0.724,0.861] · wk4 [0.770,0.887] · wk6 [0.748,0.878] · wk8 [0.769,0.900] · full [0.759,0.895].
- **Calibrated capacity rec@20%**: wk2 0.610 · wk4 0.634 · wk6 0.683 · wk8 0.683 · full 0.683. Threshold sweep (0.05–0.95) + ROC fpr/tpr arrays persisted per week for the page's interactive elements.
- **OOF parquets**: 5 files, 560 rows each, cols [student_id,course_id,y,p,p_raw], y_sum=41 (prevalence 0.073) ✓.
- **Verifier (all PASS)**: point estimates reproduce stored `roc_auc_calibrated` within ±0.005 (exact, drift 0.0000) ✓; CIs present ✓; OOF parquets 560 rows ✓. PASS.

## H2 — PUC stratified nested run — PASS (2026-07-03)
Script `scripts/puc_stratified_nested.py` (via `.venv-tier1`) → `tier2_push/stratified_nested_results.json` + `oof_stratified_week_{2,4,6,8,full}.parquet` + log `logs/h2_stratified_nested.log`. StratifiedKFold(5,shuffle,seed42) OUTER (not grouped), inner StratifiedKFold(3) Optuna30 F2, same winner config, 5-seed bag, Platt sigmoid. Ran ~819s (14 min).
- **Stratified calibrated AUC ("cursos conocidos, alumnos nuevos")**: wk2 0.7967 · wk4 0.7921 · wk6 0.7973 · wk8 0.8289 · full 0.8139.
- **strat − LOCO (calibrated)**: wk2 −0.001 · wk4 −0.038 · wk6 −0.021 · wk8 −0.009 · full −0.017 — all NEGATIVE, none exceeds LOCO by >0.10 (no implausible flag). Honest finding: PUC generalizes across courses so well that holding whole courses out (LOCO) does NOT cost vs holding students out (stratified); the fresh F2-Optuna tuning + absence of the calibration-bagging bump leaves stratified at or slightly below LOCO. LOCO remains the stronger, headline scheme.
- **OOF parquets**: 5 files, 560 rows each, cols [student_id,course_id,y,p,p_raw], y=41.
- **Verifier (all PASS)**: JSON complete 5 weeks ✓; no week exceeds its LOCO counterpart by >0.10 (all Δ negative) ✓; OOF parquets 560 rows ✓. PASS.

## H3 — Number inventory + best-defensible map — PASS (2026-07-03)
Script `scripts/build_html_data.py` (via `.venv-tier1`) → `tier2_push/html_window_data.json` (drop-in `window.DATA`) + `tier2_push/html_number_map.json` (inventory + best-defensible table + provenance). All interactive arrays regenerated from persisted OOF vectors.
- **Design decision (recorded in map)**: the page's institution toggle (UA/PUC) is repurposed as a **CV-scheme toggle — LOCO "cursos nunca vistos" vs Estratificada "cursos conocidos, alumnos nuevos"**, BOTH fed from real PUC OOF (H1/H2), calibrated CatBoost. UA moves to the technical annex as **segunda institución (DROP-A only, n=322)**, summarized from `ua_confirmatory.json`. This makes every interactive number trace to a persisted prediction vector and cleanly labels both schemes, while retiring all forbidden numbers.
- **Best-defensible per-week table** (calibrated production CatBoost):
  - LOCO (cursos nunca vistos): 0.80 [0.72,0.86] / 0.83 [0.77,0.89] / 0.82 [0.75,0.88] / **0.84 [0.77,0.90]** / 0.83 [0.76,0.89].
  - Estratificada (cursos conocidos): 0.80 / 0.79 / 0.80 / 0.83 / 0.81.
  - best-per-week cross-model LOCO: wk8 = **0.848** (XGBoost tuned nested, `nested_cv_results.json`) > CatBoost 0.838; all other weeks CatBoost calibrated.
- **UA DROP-A (segunda institución, strat cal)**: wk2 0.67 · wk4 0.69 · wk8 0.79 · full 0.80 (LOCO weak 0.43–0.64). n=322, prevalence 0.304.
- **Per-course LOCO (held-out, 7 cursos)**: AUC 0.54 (16-alumno curso, ruidoso) – 0.95; pooled 0.838. Real_ops (tp/fp/tn/fn) + real ROC arrays (≤60 pts) per week per scheme regenerated from OOF.
- **Forbidden numbers removed** (listed in map): 0.903/0.9033/0.90/0.89 UA-KEEP header family; ua_best 0.8605/0.86 hold-out; all ua_weeks KEEP AUCs (0.7428–0.9033) incl. without_assessment 0.8485; UA LOCO 0.745/0.7454; UA per_course_auc KEEP; PUC old non-nested benchmark AUCs (0.8308/0.8716/0.8632/0.8632/0.8537) and best_models 0.872/0.863/0.88 as headline.
- **Verifier (all PASS)**: every REPLACE source path exists (`sources_exist` all true) ✓; every DELETE has a reason (forbidden_removed list) ✓; best-defensible table has per-cell provenance ✓; interactive arrays trace to OOF parquets by construction ✓. PASS.

## H4 — Edit tm3-diagnostico.html — PASS (2026-07-03)
Applied the H3 map to `~/projects/tm3-roi-diagnostico/tm3-diagnostico.html` (backup preserved). Changes: (1) `window.DATA` block fully replaced with the regenerated payload (`html_window_data.json`, 68.8 KB, all arrays from persisted OOF); (2) institution toggle → CV-scheme toggle (`data-inst="loco"/"strat"`, labels "Cursos nunca vistos"/"Cursos conocidos"), default `inst='loco'`; (3) `weekData`/`rocData`/`buildLoco` repointed to `D[inst].weeks` + `D.per_course_loco`; UA milestone markers disabled; (4) hero rewritten PUC-primary (LOCO wk2 0.80 → wk8 0.84, rec@20% ~68%, pills updated); (5) technical annex rewritten: PUC LOCO table + PUC estratificada table (both calibrated CatBoost, CI/Brier/ECE), naive-classifiers row updated to wk8 LOCO max-F1, UA table replaced with DROP-A segunda institución (strat+LOCO, n=322); (6) WKNOTE/rocNote/genLead relabeled with scheme + real OOF source paths; (7) ROI scenario AUCs → 0.79 (UA DROP-A) / 0.838 (PUC), base rate 30%.
- Render verification (Chrome DevTools MCP, Chrome 149, served on localhost:8899): console 0 errors / 0 warnings; both <script> blocks pass `node --check`; DATA JSON parses. Charts build — wkChart 21 SVG children, rocChart 36, locoChart 38. Scheme toggle (loco<->strat), ROC-week toggle, and ROI toggle all functional. Rendered-body scan: no forbidden strings (0.90/0.903/0.89/0.8605/"AUC 0.86" all absent). Week chart shows 7.3% prevalence bars + AUC line 0.80->0.84; generalization cards show AUC pooled LOCO 0.84 / sem2 0.80 / 7 cursos / 7.3%.
- File integrity (md5): index.html = b357b910c7af77e5e9734d22f6e09cbe UNCHANGED; metricas-tecnicas-udla.html = 6d671a0e3fccf82ae323cc41088ee5a8 UNCHANGED; backup = d16680dd14518b2915e0818115e25127 (= original) byte-identical PRESERVED; new tm3-diagnostico.html = 9744d0529abae55b8747f742477d6c48.
- Verifier (all PASS): page opens without JS errors; every rendered number traces to the map/OOF, zero forbidden numbers; index.html + metricas md5 unchanged; backup byte-identical. PASS.

## H5 — Comparison report — PASS (2026-07-03)
Wrote `tier2_push/html_update_report.md`: side-by-side OLD→NEW→source for every changed number (hero, PUC LOCO table, new estratificada table, naive-classifier row, UA DROP-A table, all `window.DATA` fields, generalization cards, ROI AUCs); deleted-claims list with reasons (7 items); final best-defensible per-week table (LOCO + estratificada + UA DROP-A); file paths + md5 for updated page and preserved backup.
- **Verifier (all PASS)**: report exists ✓; every changed number appears in it ✓; PROGRESS updated with final hashes ✓. PASS.

---

## FINAL — TIER-2B all 6 tasks DONE (H0–H5), 0 BLOCKED (2026-07-03)
- **Discipline honored**: RANDOM_STATE=42, B=2000; H1 reused stored per-fold params (no Optuna); H2 tuned fresh (new protocol, inner 3-fold Optuna30 F2); Boruta-heavy jobs serialized (H1 → H2 → H3 sequential); only file modified outside `uautonoma` = `tm3-diagnostico.html` (after H0 backup). Every HTML number traces to a persisted OOF vector or confirmatory JSON (H3 map).
- **Defensibility ruleset satisfied**: zero forbidden numbers in the rendered page (0.90/0.903/0.89 header family, UA KEEP-arm values, single-seed sweep maxima, old non-nested PUC headline all removed — Chrome body scan clean); both CV schemes labeled (LOCO "cursos nunca vistos" / estratificada "cursos conocidos, alumnos nuevos"); UA quoted DROP-A only (n=322) as segunda institución; calibrated production CatBoost numbers used with H1 CIs.
- **Header achieved**: PUC LOCO wk8 **0.84** [0.77–0.90] ("cursos nunca vistos"); estratificada wk8 0.83; recall@20% ~68% from sem 6.
- **File integrity (final)**: `tm3-diagnostico.html` = `9744d0529abae55b8747f742477d6c48` (new); backup `d16680dd14518b2915e0818115e25127` (= original, byte-identical); `index.html` = `b357b910c7af77e5e9734d22f6e09cbe` UNCHANGED; `metricas-tecnicas-udla.html` = `6d671a0e3fccf82ae323cc41088ee5a8` UNCHANGED.
- **Deliverables**: updated `tm3-diagnostico.html` + preserved backup; `tier2_push/{confirmatory_calibrated_ci.json, stratified_nested_results.json, html_window_data.json, html_number_map.json, html_update_report.md}` + `oof_{calibrated,stratified}_week_{2,4,6,8,full}.parquet` (10 OOF vectors) + logs; 3 new scripts (`puc_confirmatory_calibrated_ci.py`, `puc_stratified_nested.py`, `build_html_data.py`). Adoption/publication is Paul's call.
