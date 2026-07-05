# Feature Catalog

## Overview

| Metric | Value |
|--------|-------|
| Total Features | 280 |
| After Filtering | 236 |
| Optimal Selected | 33 |
| Target Variable | Binary (failed = final_score < 57%) |

**Legend:**
- **★** = Selected in optimal 33 features
- **znorm** = Z-score normalized per course
- **Stability:** Bootstrap % / LOCO folds (see Section 8)

---

## 1. Session Features (12 features)

**Script:** `scripts/calculate_session_features.py`

**Session Definition:** Gap ≥ 30 minutes between consecutive page views marks a new session.

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `session_count` | `count(sessions)` | Total number of learning sessions | |
| `session_count_znorm` | `z-score(session_count)` | Normalized session count | |
| `session_duration_mean` | `mean(session_durations)` | Average session length (minutes) | |
| `session_duration_std` | `std(session_durations)` | Variability of session duration | |
| `session_duration_median` | `median(session_durations)` | Median session length | ★ |
| `sessions_per_week` | `session_count / course_weeks` | Sessions per week of course | |
| `sessions_per_week_znorm` | `z-score(sessions_per_week)` | Normalized sessions/week | |
| `views_per_session` | `total_views / session_count` | Average page views per session | ★ |
| `short_sessions_pct` | `count(duration < 5min) / count(sessions) * 100` | % of sessions < 5 minutes | |
| `long_sessions_pct` | `count(duration > 30min) / count(sessions) * 100` | % of sessions > 30 minutes | |
| `session_regularity` | `1 - (std(gaps) / mean(gaps))` | Consistency of engagement (0-1) | ★ |
| `total_time_min` | `sum(session_durations)` | Total time spent in LMS | |

---

## 2. Category Engagement Features (45 features)

**Script:** `scripts/calculate_category_features.py`

**Categories:** files, discussions, quizzes, assignments, pages, modules, grades, announcements, home

### Per-Category Features (9 categories × 4 features = 36)

For each category `{cat}`:

| Feature | Formula | Description |
|---------|---------|-------------|
| `{cat}_views` | `count(views where resource_type == cat)` | Total views of category |
| `{cat}_views_pct` | `{cat}_views / total_views * 100` | % of total views |
| `{cat}_unique_resources` | `nunique(resource_id where type == cat)` | Unique resources visited |
| `{cat}_time_min` | `sum(interaction_seconds) / 60` | Time spent on category |

**Selected Category Features:**
| Feature | Selected |
|---------|----------|
| `announcements_views_pct` | ★ |
| `home_views_pct` | ★ |
| `discussions_unique_resources` | ★ |
| `grades_check_per_week` | ★ |

### Derived Category Features (9 features)

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `total_views` | `count(all page_views)` | Total LMS interactions | |
| `total_views_znorm` | `z-score(total_views)` | Normalized total views | |
| `content_vs_assessment_ratio` | `(files + pages + disc) / (quizzes + assignments)` | Learning vs assessment focus | |
| `grades_check_per_week` | `grades_views / course_weeks` | Grade checking frequency | ★ |
| `discussion_participation_rate` | `participated_disc / total_disc * 100` | Active discussion participation | |
| `download_count` | `count(file downloads)` | Total file downloads | |
| `download_count_znorm` | `z-score(download_count)` | Normalized download count | ★ |
| `unique_files_downloaded` | `nunique(downloaded files)` | Unique files downloaded | ★ |
| `download_rate` | `downloads / total_file_views` | Download propensity | |

---

## 3. Proactivity Features (PCT Rankings) (60+ features)

**Script:** `scripts/calculate_proactivity_features.py`

**PCT Concept:** For each resource, students are ranked by when they first accessed it.
- First student to access → PCT = 1.0
- Last student to access → PCT ≈ 1/N
- Never accessed → PCT = 0

### Per-Resource-Type PCT Features (6 types × 8 features = 48)

Resource types: `files`, `discussions`, `quizzes`, `assignments`, `pages`, `modules`

For each type with prefix `{p}` (file, disc, quiz, assi, page, modu):

| Feature | Formula | Description |
|---------|---------|-------------|
| `{p}_n_resources` | `count(resources of type)` | Resources of this type in course |
| `{p}_mean_pct` | `mean(PCT for all resources)` | Average proactivity percentile |
| `{p}_median_pct` | `median(PCT for all resources)` | Median proactivity |
| `{p}_std_pct` | `std(PCT values)` | Spread of access timing |
| `{p}_access_rate` | `count(PCT > 0) / n_resources` | % of resources accessed |
| `{p}_top25_rate` | `count(PCT >= 0.75) / n_resources` | % in top quartile |
| `{p}_top50_rate` | `count(PCT >= 0.50) / n_resources` | % in top half |

**Selected PCT Features:**
| Feature | Selected |
|---------|----------|
| `assi_std_pct` | ★ |

### PCT Histogram Features (6 types × 5 bins = 30)

For each resource type, 5-bin histogram of PCT distribution:

| Feature | PCT Range | Description |
|---------|-----------|-------------|
| `{p}_hist_b1` | PCT = 0 | Never accessed |
| `{p}_hist_b2` | 0 < PCT ≤ 0.25 | Bottom quartile |
| `{p}_hist_b3` | 0.25 < PCT ≤ 0.50 | Second quartile |
| `{p}_hist_b4` | 0.50 < PCT ≤ 0.75 | Third quartile |
| `{p}_hist_b5` | PCT > 0.75 | Top quartile (most proactive) |

### DCT Features (4)

| Feature | Formula | Description |
|---------|---------|-------------|
| `dct_pct_0` | DCT coefficient 0 | Overall level |
| `dct_pct_1` | DCT coefficient 1 | Linear trend |
| `dct_pct_2` | DCT coefficient 2 | Curvature |
| `dct_pct_3` | DCT coefficient 3 | Higher-order pattern |

### Global Proactivity (1)

| Feature | Formula | Description |
|---------|---------|-------------|
| `overall_proactivity` | `mean(non-zero type means)` | Overall proactivity score |

---

## 4. PCA Features (15 features)

**Script:** `scripts/calculate_pca_features.py`

**Concept:** For each learning material type, create a students × resources matrix of PCT values, then apply PCA for dimensionality reduction.

**Excluded:** quizzes, assignments (these are assessments, not learning materials)

### PCA Components by Type

| Type | Prefix | Components | Total Features |
|------|--------|------------|----------------|
| files | `files_` | 3 | `files_pc1`, `files_pc2`, `files_pc3` |
| discussions | `disc_` | 3 | `disc_pc1`, `disc_pc2`, `disc_pc3` |
| pages | `pages_` | 3 | `pages_pc1`, `pages_pc2`, `pages_pc3` |
| modules | `mods_` | 2 | `mods_pc1`, `mods_pc2` |

**Selected PCA Features:**
| Feature | Selected |
|---------|----------|
| `files_pc3` | ★ |
| `mods_var_explained` | ★ |

### Metadata Features (4)

| Feature | Description |
|---------|-------------|
| `{type}_n_resources` | Number of resources used in PCA |
| `{type}_var_explained` | Total variance explained by components |

---

## 5. Weekly Temporal Features (20 features)

**Script:** `scripts/calculate_weekly_features.py`

**Course Week:** Calculated from course start (5th percentile of activity timestamps).

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `total_views` | `sum(weekly views)` | Total views across weeks | |
| `total_sessions` | `sum(weekly sessions)` | Total sessions across weeks | |
| `active_weeks_count` | `count(weeks with views > 0)` | Weeks with any activity | |
| `active_weeks_count_znorm` | `z-score(active_weeks_count)` | Normalized active weeks | |
| `first_active_week` | `min(week with views > 0)` | First week of activity | |
| `last_active_week` | `max(week with views > 0)` | Last week of activity | |
| `peak_week` | `argmax(weekly_views)` | Week with most activity | ★ |
| `early_semester_views` | `sum(views in weeks 1-8)` | First half activity | |
| `early_semester_views_znorm` | `z-score(early_semester_views)` | Normalized early views | |
| `late_semester_views` | `sum(views in weeks 9-16)` | Second half activity | |
| `late_semester_views_znorm` | `z-score(late_semester_views)` | Normalized late views | |
| `early_vs_late_ratio` | `early_views / late_views` | Early vs late engagement | |
| `avg_week_over_week_change` | `mean((w[i+1] - w[i]) / w[i])` | Average weekly change | |
| `activity_consistency` | `std(weekly_views) / mean(weekly_views)` | Coefficient of variation | |
| `engagement_pattern` | Categorical (1=early, 2=middle, 3=late, 4=consistent) | Engagement pattern type | |

### First Access Timing (4)

| Feature | Description |
|---------|-------------|
| `quizzes_first_access_week` | Week of first quiz access |
| `assignments_first_access_week` | Week of first assignment access |
| `discussions_first_access_week` | Week of first discussion access |
| `grades_first_access_week` | Week of first grade check |

---

## 6. Time-of-Day Features (12 features)

**Script:** `scripts/calculate_time_features.py`

| Feature | Formula | Description |
|---------|---------|-------------|
| `pct_night` | `count(22:00-06:00) / total` | Night activity % |
| `pct_morning` | `count(06:00-12:00) / total` | Morning activity % |
| `pct_afternoon` | `count(12:00-18:00) / total` | Afternoon activity % |
| `pct_evening` | `count(18:00-22:00) / total` | Evening activity % |
| `pct_weekend` | `count(Sat-Sun) / total` | Weekend activity % |
| `peak_hour` | `mode(hour of activity)` | Most common hour |
| `peak_day` | `mode(day of week)` | Most common day |
| `hour_diversity` | `-Σ(p × log(p)) / log(24)` | Entropy of hour distribution |
| `time_consistency` | `1 - (std(hours) / 12)` | Consistency of study times |
| `late_night_ratio` | `count(00:00-04:00) / total` | Very late night activity |
| `work_hours_ratio` | `count(09:00-17:00) / total` | Business hours activity |

---

## 7. Course-Relative Features (77 features) ★ NEW

**Script:** `scripts/calculate_course_relative_features.py`

**Concept:** All temporal features normalized to 0-100% of course duration (from first to last student interaction), enabling fair comparison across courses with different lengths.

### Time Progression Features (5)

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `first_access_pct` | `(first_ts - course_start) / course_duration * 100` | When student started (0-100%) | |
| `last_access_pct` | `(last_ts - course_start) / course_duration * 100` | When student ended | ★ |
| `activity_span_pct` | `last_access_pct - first_access_pct` | Coverage of course duration | |
| `median_activity_pct` | `median(all activity timestamps as %)` | Central tendency | |
| `activity_std_pct` | `std(all activity timestamps as %)` | Spread of activity | |

### Early Pattern Features (7)

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `early_10_views_pct` | `count(ts <= 10%) / total * 100` | % of views in first 10% | ★ |
| `early_20_views_pct` | `count(ts <= 20%) / total * 100` | % of views in first 20% | |
| `early_33_views_pct` | `count(ts <= 33%) / total * 100` | % of views in first third | |
| `early_10_sessions` | `count(sessions starting <= 10%)` | Sessions in first 10% | |
| `early_20_sessions` | `count(sessions starting <= 20%)` | Sessions in first 20% | |
| `early_10_resource_types` | `nunique(resource_type where ts <= 10%)` | Resource diversity early | |
| `early_engagement_intensity` | `early_20_views / (total * 0.20)` | Relative early intensity | |

### Per-Resource Timing Features (48)

For each resource type (files, disc, assi, quiz, modu, page):

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `{p}_mean_access_pct` | `mean(first_access_ts as % of course)` | Mean first-access timing | `quiz_mean_access_pct` ★ |
| `{p}_median_access_pct` | `median(first_access_ts as %)` | Median first-access timing | |
| `{p}_std_access_pct` | `std(first_access_ts as %)` | Spread of access timing | `page_std_access_pct` ★ |
| `{p}_early_access_rate` | `count(accessed <= 20%) / n_resources` | % accessed early | |
| `{p}_timing_hist_b1` | `count(access <= 20%) / n_resources` | Accessed in 0-20% | |
| `{p}_timing_hist_b2` | `count(20% < access <= 40%) / n_resources` | Accessed in 20-40% | `file_timing_hist_b2` ★ |
| `{p}_timing_hist_b3` | `count(40% < access <= 60%) / n_resources` | Accessed in 40-60% | `quiz_timing_hist_b3` ★, `page_timing_hist_b3` ★, `disc_timing_hist_b3` ★ |
| `{p}_timing_hist_b4` | `count(60% < access <= 80%) / n_resources` | Accessed in 60-80% | `page_timing_hist_b4` ★, `disc_timing_hist_b4` ★, `quiz_timing_hist_b4` ★, `modu_hist_b4` ★ |
| `{p}_timing_hist_b5` | `count(access > 80%) / n_resources` | Accessed in 80-100% | `disc_timing_hist_b5` ★, `page_timing_hist_b5` ★ |

### Temporal Engagement Curve (7)

| Feature | Formula | Description |
|---------|---------|-------------|
| `activity_bin_1` | `count(ts in 0-20%) / total * 100` | Activity in first 20% |
| `activity_bin_2` | `count(ts in 20-40%) / total * 100` | Activity in 20-40% |
| `activity_bin_3` | `count(ts in 40-60%) / total * 100` | Activity in 40-60% |
| `activity_bin_4` | `count(ts in 60-80%) / total * 100` | Activity in 60-80% |
| `activity_bin_5` | `count(ts in 80-100%) / total * 100` | Activity in last 20% |
| `engagement_curve_slope` | `linear_regression(bins).slope` | Trend direction |
| `engagement_curve_trend` | Categorical (1=↑, 2=↓, 3=flat, 4=peak_middle) | Pattern classification |

### Enhanced Session Features (4)

| Feature | Formula | Description | Selected |
|---------|---------|-------------|----------|
| `session_gap_cv` | `std(gaps) / mean(gaps)` | Gap consistency | |
| `session_gap_median_hours` | `median(inter-session gaps)` | Typical gap between sessions | |
| `session_gap_median_hours_znorm` | `z-score(session_gap_median_hours)` | Normalized median gap | ★ |
| `session_gap_max_days` | `max(gap) / 86400` | Longest inactivity period | |
| `session_gap_max_days_znorm` | `z-score(session_gap_max_days)` | Normalized max gap | ★ |
| `longest_inactive_period_pct` | `max_gap / course_duration * 100` | Max gap as % of course | |

---

## Optimal Feature Set (33 Features)

The features selected by the SOTA pipeline, achieving LOCO AUC = 0.7819:

### By Category

| Category | Count | Features |
|----------|-------|----------|
| Time-Normalized Timing ★NEW | 15 | `quiz_timing_hist_b3`, `page_timing_hist_b3`, `quiz_mean_access_pct`, `disc_timing_hist_b5`, `session_gap_median_hours_znorm`, `disc_timing_hist_b3`, `session_gap_max_days_znorm`, `early_10_views_pct`, `last_access_pct`, `page_std_access_pct`, `file_timing_hist_b2`, `page_timing_hist_b4`, `disc_timing_hist_b4`, `quiz_timing_hist_b4`, `page_timing_hist_b5` |
| Resource Engagement | 7 | `grades_check_per_week`, `discussions_unique_resources`, `announcements_views_pct`, `home_views_pct`, `files_pc3`, `unique_files_downloaded`, `pages_time_min_znorm` |
| Assignment/Quiz Engagement | 4 | `assignments_views_znorm`, `assi_std_pct`, `quizzes_time_min_znorm`, `quizzes_time_min` |
| Session Patterns | 3 | `session_duration_median`, `views_per_session`, `session_regularity` |
| Temporal Patterns | 1 | `peak_week` |
| Other | 3 | `mods_var_explained`, `download_count_znorm`, `modu_hist_b4` |

### Key Finding

**45% of optimal features (15/33) are course-relative time-normalized features.**

These features capture WHEN students engage relative to the course timeline, enabling better cross-course generalization.

---

## Feature Categories Summary

| Category | Script | Features | In Optimal |
|----------|--------|----------|------------|
| Session | `calculate_session_features.py` | 12 | 3 |
| Category Engagement | `calculate_category_features.py` | 45 | 7 |
| Proactivity (PCT) | `calculate_proactivity_features.py` | 60+ | 1 |
| PCA | `calculate_pca_features.py` | 15 | 2 |
| Weekly Temporal | `calculate_weekly_features.py` | 20 | 1 |
| Time-of-Day | `calculate_time_features.py` | 12 | 0 |
| Course-Relative ★ | `calculate_course_relative_features.py` | 77 | 15 |
| Normalized (_znorm) | `normalize_features_per_course.py` | ~40 | 4 |
| **TOTAL** | | **280** | **33** |

---

## 8. Feature Stability Analysis

**Script:** `scripts/analyze_feature_stability.py`
**Experiment:** `experiments/2026-01-05_feature_stability/`

### Methodology

| Method | Description |
|--------|-------------|
| Bootstrap (100 samples) | Train on 80% random sample, record top 20 features |
| LOCO (10 folds) | Leave-one-course-out, record top 20 features |
| Stability Score | `(bootstrap_rate + loco_rate) / 2` |

### Most Stable Features

Features recommended for production (stability >= 0.50):

| Feature | Bootstrap % | LOCO Folds | Stability | Category |
|---------|-------------|------------|-----------|----------|
| `assi_access_rate` | 70% | 9/10 | 0.80 | Proactivity |
| `quiz_timing_hist_b3` | 62% | 8/10 | 0.71 | Course-Relative |
| `assi_mean_pct` | 53% | 8/10 | 0.67 | Proactivity |
| `quiz_median_pct` | 59% | 7/10 | 0.65 | Proactivity |
| `assi_median_pct` | 49% | 7/10 | 0.59 | Proactivity |
| `total_time_min_znorm` | 58% | 6/10 | 0.59 | Session |
| `grades_check_per_week` | 43% | 6/10 | 0.52 | Category |

### Unstable Features (Avoid in Production)

Features with high v4 importance but low cross-validation stability:

| Feature | v4 Rank | Bootstrap % | LOCO | Issue |
|---------|---------|-------------|------|-------|
| `page_n_resources` | #2 | 1% | 0/10 | Course-specific |
| `total_transitions` | #3 | 0% | 0/10 | N-gram overfitting |
| `transition_entropy` | #18 | 0% | 0/10 | N-gram overfitting |
| `modules_views` | #6 | 12% | 0/10 | Course-specific |
| `mods_n_resources` | #7 | 0% | 0/10 | Course-specific |

### Key Finding: Model Version Divergence

| Comparison | Overlap |
|------------|---------|
| v4 top 20 vs SOTA optimal 33 | **1 feature** (3%) |

**Conclusion:** N-gram and raw count features overfit to training data. Course-relative timing features (15/33 in SOTA) generalize better across courses.

---

*Last updated: 2026-01-05*
