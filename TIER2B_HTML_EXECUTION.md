# TIER-2B EXECUTION PACK — best-defensible metrics + `tm3-diagnostico.html` update
**Designed by Fable 5 · 2026-07-03 · to be executed by Opus 4.8 in a fresh session**
Prerequisites (DONE, do not redo): Tier-1 + Tier-2 (`TIER1_RESULTS.md`, `TIER2_RESULTS.md`, `TIER2_PROGRESS.md`, `tier2_push/`). Paul has EXPLICITLY opened the sales-materials gate for exactly ONE file: `~/projects/tm3-roi-diagnostico/tm3-diagnostico.html`. Everything else in that repo stays untouched (`index.html`, `metricas-tecnicas-udla.html` included).

## Mission

Assemble the **best defensible** per-week metrics from both deployments (PUC + UA) under the defensibility ruleset below, then update `tm3-diagnostico.html` with them — keeping a byte-identical backup of the old version for comparison. Audience: Enrique / any technical reviewer. Tone: neutral, descriptive (Paul's standing direction — no "se dijo/verdad" contrastive framing, no sales adjectives around the numbers).

## Defensibility ruleset (FROZEN — this defines "best defensible")

**Allowed (quotable):**
1. Best-per-week across models and deployments — the page's existing "mejor modelo en cada corte" framing. Per-cell provenance must be traceable to a confirmatory/verified JSON.
2. BOTH CV schemes, each labeled: **LOCO** (StratifiedGroupKFold by course) = "cursos nunca vistos"; **stratified** (StratifiedKFold) = "cursos conocidos, alumnos nuevos". Never mix them in one row without labels.
3. Calibrated-model numbers (the production artifact) once their CIs exist (H1).
4. UA numbers ONLY from the **DROP-A arm** (n=322). Cite as "segunda institución" with its one-line provenance.

**Forbidden (these die, no exceptions):**
- The `0.903 / 0.90 / ~0.89` header numbers (UA KEEP-arm, stratified, contaminated labels — 51 active-zero mislabels).
- Any UA KEEP-arm number, with or without caveat — this page is outward-facing; the caveat lives in internal docs.
- Single-seed bake-off maxima (P2/UA-2 sweeps are internal, never quotable).
- The old PUC non-nested tuned table (0.831/0.872/0.863/0.863/0.854) as a headline — superseded by nested. (It may appear ONLY inside the technical annex as "validación no anidada previa", if the annex already had it.)

**Sources of truth (exact paths):**
- PUC LOCO nested calibrated CatBoost: `tier2_push/confirmatory_results.json` (`roc_auc_calibrated` per week; CIs from H1).
- PUC LOCO nested tuned XGB (Tier-1): `tier1_clean/nested_cv_results.json` (`nested.roc_auc` — wk8 = 0.848, currently the best defensible wk8 LOCO cell).
- PUC stratified nested: NEW — H2 produces it.
- UA DROP-A nested: `tier2_push/ua_confirmatory.json`.
- Capacity curves: confirmatory JSONs (H1 regenerates from calibrated OOF).
- Calibration quality (ECE/Brier): confirmatory JSONs.

## Guardrails

1. Only file modified outside `uautonoma`: `~/projects/tm3-roi-diagnostico/tm3-diagnostico.html`. Before editing, create the backup copy (H0) — that repo is NOT git, so file-copy + md5 of both is the audit trail.
2. `index.html` and `metricas-tecnicas-udla.html`: verify untouched by md5 before/after the whole session.
3. All computation outputs → `data/puc/sota_results/tier2_push/` (new files) in the uautonoma repo, branch `sota-tier2`.
4. RANDOM_STATE=42; B=2000 bootstrap; reuse stored per-fold CatBoost params (`catboost_params_per_fold` in confirmatory JSON) — NO new Optuna in H1; H2 may tune (inner 3-fold, 30 trials) since it's a new protocol run.
5. Every number placed in the HTML must have a source cell in H3's mapping JSON. No number without provenance.
6. Append progress entries to `TIER2_PROGRESS.md` (same file, section "TIER-2B"), verifier-stamped.
7. 3 failed attempts → BLOCKED, move on. Serialize Boruta-heavy jobs.

---

## TASKS

### H0 — Backup + baseline hashes
Copy `tm3-diagnostico.html` → `tm3-diagnostico_v1_2026-07-03.bak.html` (same dir). Record md5 of: the backup, the original, `index.html`, `metricas-tecnicas-udla.html`.
**Verifier**: backup exists and md5(backup)==md5(original); all four hashes logged in PROGRESS.

### H1 — Calibrated CIs + persisted OOF (`scripts/puc_confirmatory_calibrated_ci.py`)
Re-run the P3 confirmatory calibrated arm per week {2,4,6,8,full}: same LOCO folds (seed 42), same per-fold top-40 selection, **reuse the stored per-fold tuned params** (no Optuna), 5-seed bagging, Platt (`sigmoid, cv=3`). THIS TIME persist the calibrated OOF probability vectors to `tier2_push/oof_calibrated_week_{w}.parquet` (student_id, course_id, y, p). Compute: `roc_auc_calibrated_ci95` + `pr_auc_calibrated_ci95` (bootstrap B=2000), calibrated capacity curve {5,10,15,20,25}%, and a threshold sweep table (thresholds 0.05–0.95 step 0.05: recall, precision, FPR) for the page's interactive elements.
Output: `tier2_push/confirmatory_calibrated_ci.json`.
**Verifier**: per-week point estimates reproduce the stored `roc_auc_calibrated` within ±0.005 (same pipeline, same seeds — should be near-exact; if drift >0.005, investigate before proceeding); CIs present; OOF parquets have 560 rows each.

### H2 — PUC stratified nested run (`scripts/puc_stratified_nested.py`)
Same winner config (CatBoost Balanced, top-40 per-fold, calibrated, 5-seed bagging) under **StratifiedKFold(5, shuffle, seed 42)** outer folds (NOT grouped), inner 3-fold Optuna 30 trials (F2), weeks {2,4,6,8,full}. Persist OOF vectors + CIs + capacity curve + threshold sweep, exactly like H1.
Output: `tier2_push/stratified_nested_results.json`.
This answers "alumnos nuevos en cursos conocidos" — the same (easier, honest) question the old UA 0.90 answered. Expected: above LOCO at wk6–8; whatever it is, it's measured.
**Verifier**: JSON complete 5 weeks; nested-vs-bakeoff leak rule N/A (new protocol) but flag if any week exceeds its LOCO counterpart by >0.10 (implausible → investigate); OOF parquets 560 rows.

### H3 — Number inventory + best-defensible mapping (`tier2_push/html_number_map.json`)
Parse `tm3-diagnostico.html` (~96 AUC mentions, JS data arrays for the threshold slider and operating curves). Produce a machine-readable inventory: every hardcoded metric/claim/array in the page → {current value, verdict: KEEP / REPLACE(new value + source path) / DELETE(reason)}. Then assemble the **best-defensible per-week table**: for each week and each labeled row (LOCO / estratificada), the max across allowed sources, with per-cell provenance. Regenerate ALL interactive-element data (ROC/fpr-tpr arrays, threshold sweeps, operating points) from the H1/H2 OOF parquets — the page's interactive data must correspond to a real persisted prediction vector, not synthetic curves.
**Verifier**: every REPLACE has a source path that exists; every DELETE has a reason; zero unmapped metric mentions remain (grep audit for `0\.\d\d` patterns in the HTML vs the map).

### H4 — Edit `tm3-diagnostico.html`
Apply the map. Structural directions:
- Header: retire 0.90/0.89; lead with the strongest defensible pair, e.g. week-8 LOCO **0.85** ("en cursos nunca vistos") and the best H2 stratified value ("con cursos conocidos"). Exact framing follows the measured values.
- Keep the page's design, layout, and interactivity intact; only data + the few text claims change.
- Both deployments may appear: PUC as primary; UA as "segunda institución" using DROP-A values only.
- Neutral descriptive tone throughout; single provenance sentence (the Tier-2 one: nested, LOCO/estratificada según fila, datos limpios, CatBoost calibrado, Shapley).
- Capacity framing for recall claims ("marcando al X% de mayor riesgo se detecta Y%"), replacing unanchored recall/precision pairs.
**Verifier**: page opens without JS errors (chrome-devtools MCP if available, else `node -e` syntax parse of extracted scripts); every number in the rendered page exists in the map; md5(original) unchanged for `index.html` + `metricas-tecnicas-udla.html`; backup still byte-identical.

### H5 — Comparison report for Paul (`tier2_push/html_update_report.md`)
Side-by-side: OLD value → NEW value → source, for every changed number; list of deleted claims with reasons; the final best-defensible table; links (file paths) to backup and new version. This is what Paul uses "to check the metrics".
**Verifier**: report exists; every changed number appears in it; PROGRESS updated with final hashes.

## Success criteria
- The page contains zero forbidden numbers; every metric maps to a persisted prediction vector or confirmatory JSON.
- Old copy preserved byte-identical for comparison.
- Best-defensible header achieved (expected: LOCO wk8 0.85; stratified wk6-8 possibly 0.86–0.88 — whatever H2 measures).
- Compute ≈ 1.5–2.5 h (H1 ~20 min, H2 ~60–90 min with Optuna, rest is assembly/editing).

## When done
Stop. Paul reviews `html_update_report.md` + the page; adoption/publication is his call.
