# Experiment: SOTA Feature Selection

**Date:** 2026-01-03
**Status:** Completed

---

## Objective

Apply state-of-the-art feature selection techniques to reduce 280 features to an optimal subset that maximizes cross-course generalization (LOCO AUC).

## Problem

- **280 features** extracted from Canvas LMS activity data
- **363 students** across 10 courses
- **Curse of dimensionality**: p ≈ n makes overfitting likely
- Need features that generalize to unseen courses

## Methods

### 5-Stage Feature Selection Pipeline

| Stage | Method | Purpose |
|-------|--------|---------|
| 1 | **Filter** | Remove near-constant features (variance threshold) and highly correlated features (r > 0.95) |
| 2 | **Univariate** | Rank by Mutual Information, point-biserial correlation, Mann-Whitney U |
| 3 | **Embedded** | LASSO, ElasticNet, Random Forest importance, XGBoost importance |
| 4 | **Wrapper** | Boruta (shadow features), RFECV (recursive elimination) |
| 5 | **Stability** | Bootstrap stability (>50% selection), LOCO stability |

### Validation Strategy

- **5-fold Stratified CV**: Within-sample performance
- **LOCO (Leave-One-Course-Out)**: Cross-course generalization

## Results

### Model Comparison

| Model | CV AUC | LOCO AUC | Features |
|-------|--------|----------|----------|
| Early Warning (baseline) | 0.8605 | 0.7454 | 40 |
| **New Optimal** | 0.8418 | **0.7708** | 33 |

**Improvement:** +3.4% LOCO AUC (better cross-course generalization)

### Optimal Feature Set (33 features)

```json
[
  "assignments_views_znorm",
  "grades_check_per_week",
  "quiz_timing_hist_b3",
  "assi_std_pct",
  "peak_week",
  "discussions_unique_resources",
  "page_timing_hist_b3",
  "announcements_views_pct",
  "home_views_pct",
  "quiz_mean_access_pct",
  "mods_var_explained",
  "session_duration_median",
  "disc_timing_hist_b5",
  "session_gap_median_hours_znorm",
  "views_per_session",
  "disc_timing_hist_b3",
  "session_gap_max_days_znorm",
  "quizzes_time_min_znorm",
  "early_10_views_pct",
  "last_access_pct",
  "download_count_znorm",
  "page_std_access_pct",
  "quizzes_time_min",
  "files_pc3",
  "session_regularity",
  "file_timing_hist_b2",
  "unique_files_downloaded",
  "page_timing_hist_b4",
  "modu_hist_b4",
  "disc_timing_hist_b4",
  "quiz_timing_hist_b4",
  "page_timing_hist_b5",
  "pages_time_min_znorm"
]
```

### Key Finding

**45% of optimal features (15/33) are course-relative time-normalized features.**

These features, introduced in the previous experiment (`2026-01-02_course_relative_features`), capture WHEN students engage relative to the course timeline, enabling better cross-course generalization.

## Files

| File | Description |
|------|-------------|
| `scripts/sota_feature_selection.py` | Main feature selection pipeline |
| `results/optimal_features.json` | Selected 33 features with metadata |
| `results/consolidated_summary.json` | Full analysis and comparison |
| `results/feature_rankings.parquet` | Complete ranking of all 280 features |

## Conclusions

1. **Curse of dimensionality addressed**: Reduced 280 → 33 features (88% reduction)
2. **Cross-course generalization improved**: +3.4% LOCO AUC
3. **Course-relative features validated**: 45% of optimal features use time normalization
4. **Trade-off identified**: Slight CV drop (-0.0187) for significant LOCO gain (+0.0254)

## Next Steps

- [ ] Retrain final model with optimal 33 features
- [ ] Generate SHAP explanations for interpretability
- [ ] Create early warning dashboard with risk scores

---

*Experiment conducted with Claude Code*
