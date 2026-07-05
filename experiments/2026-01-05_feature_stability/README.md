# Experiment: Feature Stability Analysis

**Date:** 2026-01-05
**Status:** Completed

---

## Objective

Validate whether feature importance findings are consistent across:
1. Bootstrap samples (sampling variability)
2. LOCO folds (cross-course generalization)
3. Model versions (v4 vs SOTA optimal)

---

## Motivation

User concern: "I've seen many different features between models, such as n-grams (files-files, quizzes-files, etc.) or module percentile features. Having top features vary so much makes findings inconsistent."

---

## Methodology

### Bootstrap Stability (100 samples)
```python
for i in range(100):
    sample = df.sample(frac=0.8, replace=True)
    model = XGBClassifier(n_estimators=100).fit(sample)
    record_top_20_features(model)
```
**Selection rate:** % of bootstraps where feature appeared in top 20

### LOCO Stability (10 folds)
```python
for course in courses:
    train = df[df.course_id != course]
    model = XGBClassifier(n_estimators=100).fit(train)
    record_top_20_features(model)
```
**LOCO folds:** Number of folds (out of 10) where feature appeared in top 20

### Stability Score
```
stability_score = (bootstrap_rate + loco_rate) / 2
```

---

## Key Results

### Critical Finding: Low Feature Overlap

| Comparison | Overlap |
|------------|---------|
| v4 top 20 vs SOTA optimal 33 | **1 feature** (3%) |

**Only overlapping feature:** `discussions_unique_resources`

### Most Stable Features (Stability Score >= 0.5)

| Feature | Bootstrap % | LOCO Folds | Stability |
|---------|-------------|------------|-----------|
| assi_access_rate | 70% | 9/10 | 0.80 |
| quiz_timing_hist_b3 | 62% | 8/10 | 0.71 |
| assi_mean_pct | 53% | 8/10 | 0.67 |
| quiz_median_pct | 59% | 7/10 | 0.65 |
| assi_median_pct | 49% | 7/10 | 0.59 |
| total_time_min_znorm | 58% | 6/10 | 0.59 |
| grades_check_per_week | 43% | 6/10 | 0.52 |

### Unstable Features (High v4 importance, Low stability)

| Feature | v4 Importance | Bootstrap % | LOCO |
|---------|---------------|-------------|------|
| page_n_resources | 5.51% | 1% | 0/10 |
| total_transitions | 4.66% | 0% | 0/10 |
| transition_entropy | 2.14% | 0% | 0/10 |
| modules_views | 4.02% | 12% | 0/10 |

---

## Conclusions

### 1. N-gram Features Are Completely Unstable
- `total_transitions`, `transition_entropy`: 0% bootstrap, 0/10 LOCO
- These overfit to specific course navigation structures
- **Recommendation:** Exclude from production model

### 2. Only 7 Features Are Truly Stable
- Use stability score >= 0.5 as threshold
- These should form the core of a robust model

### 3. Course-Relative Features More Stable
- 15/33 SOTA features are course-relative timing features
- These generalize better across courses

---

## Files

| File | Description |
|------|-------------|
| `results/bootstrap_stability.json` | 100-sample bootstrap results |
| `results/loco_stability.json` | 10-fold LOCO results |
| `results/v4_vs_sota_comparison.csv` | Feature set comparison |
| `results/FEATURE_STABILITY_REPORT.md` | Full report (copy in docs/) |

---

## Script

`scripts/analyze_feature_stability.py`

---

## Impact on Project

1. Validated user's concern about feature instability
2. Identified unreliable features (n-grams, module counts)
3. Established 7 stable features for production use
4. Updated feature catalog with stability metrics

---

*Analysis time: ~15 minutes (100 bootstraps + 10 LOCO folds)*
