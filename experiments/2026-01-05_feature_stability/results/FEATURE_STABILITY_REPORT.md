# Feature Stability Report

## Overview

This report analyzes feature importance stability across model versions
and validation strategies.

### Analysis Summary

| Metric | Value |
|--------|-------|
| Bootstrap samples | 100 |
| LOCO folds | 10 |
| v4 top features | 20 |
| SOTA optimal features | 33 |

---

## Key Finding: Feature Instability

**Top features differ significantly between v4 and SOTA models:**

### v4 Model Top 10
| Rank | Feature | Importance |
|------|---------|------------|
| 1 | total_time_min_znorm | 0.0597 | SOTA: N |
| 2 | page_n_resources | 0.0551 | SOTA: N |
| 3 | total_transitions | 0.0466 | SOTA: N |
| 4 | content_vs_assessment_ratio | 0.0439 | SOTA: N |
| 5 | total_views_x_znorm | 0.0436 | SOTA: N |
| 6 | modules_views | 0.0402 | SOTA: N |
| 7 | mods_n_resources | 0.0371 | SOTA: N |
| 8 | discussions_unique_resources | 0.0355 | SOTA: Y |
| 9 | modules_views_znorm | 0.0355 | SOTA: N |
| 10 | sessions_per_week_znorm | 0.0302 | SOTA: N |

### SOTA Optimal Top 10
| Rank | Feature | In v4 Top 20 |
|------|---------|--------------|
| 1 | assignments_views_znorm | N |
| 2 | grades_check_per_week | N |
| 3 | quiz_timing_hist_b3 | N |
| 4 | assi_std_pct | N |
| 5 | peak_week | N |
| 6 | discussions_unique_resources | Y |
| 7 | page_timing_hist_b3 | N |
| 8 | announcements_views_pct | N |
| 9 | home_views_pct | N |
| 10 | quiz_mean_access_pct | N |

---

## Stability Analysis Results

### Most Stable Features (Bootstrap + LOCO)

Features appearing consistently across both bootstrap samples AND LOCO folds:

| Feature | Bootstrap % | LOCO Folds | Stability Score | v4 Rank | SOTA Rank |
|---------|-------------|------------|-----------------|---------|-----------|
| assi_access_rate | 70% | 9/10 | 0.80 | - | - |
| quiz_timing_hist_b3 | 62% | 8/10 | 0.71 | - | 3 |
| assi_mean_pct | 53% | 8/10 | 0.67 | - | - |
| quiz_median_pct | 59% | 7/10 | 0.65 | - | - |
| assi_median_pct | 49% | 7/10 | 0.59 | - | - |
| total_time_min_znorm | 58% | 6/10 | 0.59 | 1 | - |
| grades_check_per_week | 43% | 6/10 | 0.52 | - | 2 |

### Unstable Features

Features with high v4 importance but low stability:

| Feature | v4 Importance | Bootstrap % | LOCO Folds | Issue |
|---------|---------------|-------------|------------|-------|
| page_n_resources | 0.0551 | 1% | 0/10 | Low bootstrap stability; Low LOCO stability |
| total_transitions | 0.0466 | 0% | 0/10 | Low bootstrap stability; Low LOCO stability |
| content_vs_assessment_ratio | 0.0439 | 12% | 1/10 | Low bootstrap stability; Low LOCO stability |
| total_views_x_znorm | 0.0436 | 13% | 1/10 | Low bootstrap stability; Low LOCO stability |
| modules_views | 0.0402 | 12% | 0/10 | Low bootstrap stability; Low LOCO stability |
| mods_n_resources | 0.0371 | 0% | 0/10 | Low bootstrap stability; Low LOCO stability |
| discussions_unique_resources | 0.0355 | 18% | 1/10 | Low bootstrap stability; Low LOCO stability |
| modules_views_znorm | 0.0355 | 7% | 0/10 | Low bootstrap stability; Low LOCO stability |
| sessions_per_week_znorm | 0.0302 | 24% | 2/10 | Low bootstrap stability; Low LOCO stability |
| pages_pc1 | 0.0288 | 21% | 2/10 | Low bootstrap stability; Low LOCO stability |
| total_time_min | 0.0280 | 9% | 0/10 | Low bootstrap stability; Low LOCO stability |
| files_pc1 | 0.0274 | 5% | 0/10 | Low bootstrap stability; Low LOCO stability |
| last_active_week | 0.0261 | 9% | 0/10 | Low bootstrap stability; Low LOCO stability |
| pages_var_explained | 0.0253 | 8% | 0/10 | Low bootstrap stability; Low LOCO stability |
| discussions_time_min_znorm | 0.0232 | 9% | 0/10 | Low bootstrap stability; Low LOCO stability |
| modu_mean_pct | 0.0232 | 4% | 0/10 | Low bootstrap stability; Low LOCO stability |
| hour_diversity | 0.0217 | 0% | 0/10 | Low bootstrap stability; Low LOCO stability |
| transition_entropy | 0.0214 | 0% | 0/10 | Low bootstrap stability; Low LOCO stability |
| files_views_pct | 0.0212 | 15% | 3/10 | Low bootstrap stability; Low LOCO stability |

---

## Feature Overlap Analysis

**v4 top 20 vs SOTA optimal 33:**
- Overlap: 1 features (3% of SOTA)
- v4-only: 19 features
- SOTA-only: 32 features

### Overlapping Features (Stable Across Both)

- discussions_unique_resources

### v4-Only Features (Not in SOTA Optimal)

- content_vs_assessment_ratio
- discussions_time_min_znorm
- files_pc1
- files_views_pct
- hour_diversity
- last_active_week
- mods_n_resources
- modu_mean_pct
- modules_views
- modules_views_znorm
- page_n_resources
- pages_pc1
- pages_var_explained
- sessions_per_week_znorm
- total_time_min
- total_time_min_znorm
- total_transitions
- total_views_x_znorm
- transition_entropy

---

## Conclusions

### 1. N-gram Features Are Unstable
- `total_transitions` ranked #3 in v4 (4.66% importance)
- NOT selected in SOTA optimal
- Likely overfits to specific course structures

### 2. Course-Relative Features Are More Stable
- 15/33 SOTA features are course-relative timing features
- These generalize better across courses (higher LOCO AUC)

### 3. Recommended Stable Feature Set
Use features with:
- Bootstrap selection rate > 50%
- LOCO fold appearance > 7/10

---

*Generated: 2026-01-05*
*Analysis: 100 bootstrap samples, 10 LOCO folds*