# FEATURE CATALOG — single source of truth for engineered features
**Created 2026-07-03.** Purpose: this repo accumulated ~325+ engineered features across notebooks, `scripts/calculate_*`, a separate branch, and 15+ component parquets — with no index. The Tier-3 shared pipeline was rebuilt from a short family list and unknowingly skipped several of the **historically top-ranked** families (e.g. `mobile_pct`, `grades_check_per_week` (#2), proactivity access-timing (#3–#6)). This catalog exists so that never happens silently again. **Before building any new feature pipeline, diff it against this file.**

## How to read this
- **Class** = cross-institution computability. PUC clean = `puc_clean_data.parquet`; UA clean = `ua_clean_data.parquet`.
  - **SHARED** — computable identically at both from raw clickstream columns present in both.
  - **UA-ONLY** / **PUC-ONLY** — needs a column present at only one institution.
  - **NEEDS-EXTERNAL** — needs data outside the clickstream (assignment `due_at`, official grades, Canvas Analytics/Modules API). Often also a leakage risk.
- **In T3?** = present in the Tier-3 shared pipeline `scripts/common_features.py` (62 features).
- **Importance** = rank/score in `data/feature_selection/feature_rankings.parquet` (UA, composite over 236 feats) where known.

## Shared raw columns (the computability substrate)
Both institutions' clean clickstream have: `created_at` (→hour/dow/week/session/timing), `controller`, `action`, category (`category` PUC / `resource_type` UA → 10 shared bins), `resource_id`, `user_agent`, `participated`, `http_method`, `interaction_seconds` (unreliable — excluded as a duration signal since Tier-1), `course_id`, student id.
- **PUC-only raw:** `contributed`, `context_type`, `semester`, `render_time`.
- **UA-only raw:** `session_id`, `referrer`, `bytes`, `microseconds`, `http_status`, `remote_ip`, `app_server`, `http_request` (full URL).
- **Untapped by every historical calculator:** `user_agent`, `controller`, `action`, `http_method`, `contributed`, `referrer` — behavioral signal nobody has mined yet (device, action-verb granularity, navigation source).

---

## A. Feature families (the corpus)

| # | Family | Example features | Measures | Class | In T3? | Importance / notes |
|---|--------|------------------|----------|-------|--------|--------------------|
| 1 | **Category volume/breadth** | `{cat}_views`, `{cat}_views_pct`, `{cat}_unique_resources`, `{cat}_time_min`, `content_vs_assessment_ratio`, **`grades_check_per_week`** | per-category access volume, breadth, dwell | SHARED | partial (cnt/share only) | `grades_check_per_week` **#2**; `assignments_views_znorm` **#1**; T3 lacks per-cat `unique_resources`/`_time_min` & the grade-check rate |
| 2 | **Time-of-day / day-of-week** | `pct_{night,morning,afternoon,evening}`, `pct_weekend`, `peak_hour`, `peak_day`, `hour_diversity`, `time_consistency`, `late_night_ratio`, `work_hours_ratio` | when in day/week they study | SHARED | partial | T3 has shares+entropy but not `time_consistency`, `peak_hour/day`, weekday×block cross |
| 3 | **Session** | `session_count`, `session_duration_{mean,std,median}`, `sessions_per_week`, `views_per_session`, `short/long_sessions_pct`, `session_regularity`, `session_density` | 30-min-gap session cadence | SHARED | mostly | UA `session_id` would make sessionization exact (currently gap-based both) |
| 4 | **Weekly trajectory** | `active_weeks_count`, `peak_week`, `early/late_semester_views`, `early_vs_late_ratio`, `avg_week_over_week_change`, `activity_consistency`, `engagement_pattern` | week-binned trend & momentum | SHARED | partial | `peak_week` **#8**; T3 has trend_slope/momentum but not peak_week/consistency |
| 5 | **N-gram / navigation transitions** | `bigram_{X}_to_{Y}` (top-15), `transition_entropy`, `self_loop_ratio`, `transition_diversity`, `dominant_transition` | Markov transitions between resource types within a session | SHARED | **NO** | pure `created_at`+category+session; fully portable |
| 6 | **Graph / resource-network** | `unique_resources`, `resource_coverage`, `resource_diversity`, `resources_vs_avg`, `category_diversity`, `resource_repetition_rate`, `resource_concentration_gini` | breadth/concentration of resource set | SHARED core | **NO** | measured predictive (PUC `resource_diversity` +0.30). ⚠ `jaccard_to_passing` & `access_cluster` = NEEDS-EXTERNAL (grades) and **leak** — exclude |
| 7 | **Proactivity / access-percentile** | per type {file,disc,quiz,assi,page,modu}: `{t}_mean_pct`, `{t}_access_rate`, `{t}_top25/top50_rate`, `{t}_hist_b1..5`; `overall_proactivity`, `dct_pct_0..3`, `download_*` | how early/completely each resource is accessed vs cohort | SHARED (except `download_*` = UA-only) | **NO** | **top family**: `assi_access_rate` **#4**, `quiz_median_pct` **#3**, `assi_median_pct` **#5**. 80 cols |
| 8 | **PCA resource embeddings** | `{cat}_pc1/pc2/pc3`, `{cat}_var_explained` | PCA over per-resource access matrices | SHARED | **NO** | `mods_var_explained` **#20**, `pages_pc1` **#30** |
| 9 | **Course-relative timing** | `first/last/median_access_pct`, `early_10/20/33_views_pct`, `{t}_timing_hist_b1..5`, `activity_bin_1..5`, `engagement_curve_slope`, `session_gap_{cv,median_hours,max_days}`, `longest_inactive_period_pct` | activity position normalized to course span | SHARED | **NO** (T3 z-norms but has no timing distributions) | 76–79 cols; `quiz_timing_hist_b3` **#6**, `page_timing_hist_b3` **#12**; 15 of the 33-feat optimal set |
| 10 | **Engagement decay / fade** | `engagement_decline_slope/r2`, `activity_fade_score`, `weeks_since_peak`, `first_half_vs_second_half_ratio`, `weekly_decline_rate` | how engagement erodes over term | SHARED | **NO** | PUC-script only; richer than T3 `trend_slope` |
| 11 | **Inactivity episodes** | `inactivity_episodes_gt{3,7}days`, `max_consecutive_inactive_days`, `recovery_time_after_gap_*`, `total_inactive_days` | gap/recovery structure | SHARED | **NO** (T3 has only max gap) | PUC-script only |
| 12 | **Early momentum** | `days_to_first_access`, `first_{3days,week}_views_pct`, `stall_in_first_2weeks`, `early_deceleration`, `days_until_10_views`, `early_activity_density` | speed of initial engagement | SHARED | partial | PUC-script only; beyond T3 `first_event_day` |
| 13 | **DCT / Fourier rhythm** | `dct_coef_0..11` (also PUC `dct_0..3_znorm`) | DCT of the 24×7 weekly activity vector | SHARED | **NO** | `created_at`-only; PUC SHAP shows `dct_3_znorm` top-3 at wk4 |
| 14 | **Device / browser** | `mobile_pct`, `mobile_accesses`, `desktop_accesses` | mobile-vs-desktop access mix | SHARED | **NO** | **in Ignacio's notebook, never productionized**; measured PUC −0.32*** |
| 15 | **Participation / action-verb** | `participation_rate`, active-`POST` share, `contributed` (PUC) | active vs passive interaction | SHARED | **NO** | measured UA participation −0.14* |
| 16 | **Cross-course engagement ratios** | `course_{session,views,time}_ratio` (+ weekly) | this course's share of the student's whole-LMS activity | SHARED-but-needs multi-course log | **NO** | Tier-2 "thesis" families found redundant for PUC; needs un-course-filtered feed |
| 17 | **Pre-assessment / deadline (SPLIT)** | shared: `first_{quiz,assignment}_access_days`, `quiz_access_pct`, revisits, `late_surge_ratio`, `activity_acceleration` · external: `activity_24/48/72h_before`, `preparation_intensity`, `n_deadlines` | cramming around due dates + assessment-access timing | SPLIT (access-timing SHARED; deadline-window NEEDS `due_at`) | **NO** | 34 cols; deadline half not portable |
| 18 | **Tardiness / submission** | `on_time/late/missing_rate`, `num_submissions` | assignment punctuality | NEEDS-EXTERNAL | **NO** | leaky — excluded from early-warning |
| 19 | **Partial grades / scores** | `current/final_score`, `avg/min/max_score`, `exam_N_score` | mid-course grades | NEEDS-EXTERNAL | **NO** | leakage — excluded |
| 20 | **Module completion** | `modules_completed`, `module_completion_rate`, `first_module_day` | LMS module progress | NEEDS-EXTERNAL (Modules API) | **NO** | UA-only |
| 21 | **Canvas activity levels** | `page_views_level`, `participations_level`, `total_activity_time` | Canvas' own bucketed activity | NEEDS-EXTERNAL (Analytics API) | **NO** | UA-only; the old `prediction_models.py` top predictors |

**Tier-3 `common_features.py` currently covers ~families 1–4 (partial) — 62 of ~325 columns (~19%).**

---

## B. Historical importance & validated sets (don't rediscover these)
- `data/feature_selection/feature_rankings.parquet` — 236 features scored (univariate + embedded + Boruta + RFECV + LOCO stability → `composite_score`). **Top 10:** assignments_views_znorm, grades_check_per_week, quiz_median_pct, assi_access_rate, assi_median_pct, quiz_timing_hist_b3, assi_std_pct, peak_week, discussions_unique_resources, total_views_znorm.
- `data/feature_selection/optimal_features.json` — **33-feature "Stable + Course-Relative" set, LOCO-AUC 0.782** (15 course-relative). The best-generalizing UA set.
- `data/engagement_dynamics/feature_importance.csv` — separate model; `total_participations` (0.725) then `dct_coef_1..9`, weekday×block, `first_module_day`.
- PUC (separate schema): `data/puc/sota_results/tier1_clean/shap_week{4,8}_global_importance.json` — quiz views, weekly trend, DCT spectral.

## C. Source-file map (where the code lives)
- Shared/prototype: `experiments/ignacio_experiments/features_extraction.ipynb` (incl. `calculate_device_features` → mobile_pct).
- Productionized calculators: `scripts/calculate_{session,time,weekly,category,proactivity,ngram,graph,pca,course_relative,pre_assessment}_features.py` (UA) + `scripts/puc_calculate_*` (PUC variants) + `scripts/puc_calculate_{engagement_decay,inactivity_episodes,early_momentum}.py`.
- UA Canvas-API pipeline: branch `origin/Ignacio_UA_inicial`; `scripts/engagement_dynamics_features.py` (+ `docs/ENGAGEMENT_FEATURES_DOCUMENTATION.md`), `early_warning_system.py`, `prediction_models.py`, `feature_agglomeration.py`.
- Materialized matrices: `data/enriched_features/*.parquet` (UA, `normalized_features.parquet` = 282-col master) + `data/enriched_features/cutoff_week_{2,4,6,8}/` snapshots; `data/puc/enriched_features/*` (PUC).
- Tier-3 shared pipeline: `scripts/common_features.py` → `data/puc/sota_results/tier3_pooled/feature_schema.json`.

## D. Recommended port order for a feature-complete shared pipeline
Highest ROI first (SHARED + historically top-ranked): **7 proactivity access-percentile → 9 course-relative timing histograms → 1 grades_check_per_week + per-cat unique_resources → 6 resource-diversity core → 13 DCT rhythm → 5 n-gram transitions → 8 PCA → 14 device/mobile → 15 participation → 10/11/12 decay/inactivity/momentum.** Skip 16–21 (external/leaky) for the pure-activity shared model. Re-run the institution-invariance probe (z-norm **and** percentile-rank) + drop leakers after each addition.

> Discoverability fix: keep this file current. Any PR that adds a feature family updates the table here; any new pipeline diffs against column D before training.
