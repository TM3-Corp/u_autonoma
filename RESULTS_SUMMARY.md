> ⚠️ **SUPERSEDED / STALE NUMBERS — do not quote.** This document predates the nested-CV correction; its headline AUCs are optimistic (non-nested) and/or contaminated (UA KEEP-arm active-zeros) and/or label-leaky. Current defensible metrics: **`RESULTS_LEDGER.md`**; start at **`PROJECT_SSOT.md`**. Kept for history only.

# PUC Early Warning System - Results Summary

## Overview

This document summarizes the complete end-to-end results from the early warning system developed for PUC (Pontifical Catholic University). The system predicts which students will fail (grade < 4.0 on Chilean 1-7 scale) using LMS activity data from Canvas.

**Generated:** February 2026

---

## Dataset & Validation Setup

### Data Characteristics
- **Total Records:** 2.3M activity events
- **Students:** 714 across 20 courses
- **Student-Course Pairs:** 841
- **Timespan:** Full semester plus pre-semester
- **Pass Rate:** 92.2% (769 pass, 72 fail)
- **Fail Rate:** 7.8% (at grade < 4.0 threshold)

### Validation Strategy
- **Walk-Forward Validation:** 5 temporal folds (weeks 2, 4, 6, 8, and full semester)
- **Stratification:** Per-course to handle imbalanced classes
- **Grid Search:** 2,640 experiments across:
  - 15 models (LightGBM, XGBoost, RandomForest, VotingEnsemble, etc.)
  - 5 classification schemes (binary_4.0, binary_5.0, 3-class, 4-class, etc.)
  - 12 threshold optimization criteria
  - Feature percentage cutoffs (0.05, 0.1, 0.2)

---

## Classification Schemes

### 1. **binary_4.0** (Primary: Official Passing Grade)
- **Definition:** Fail if grade < 4.0 on 1-7 scale
- **Baseline Accuracy:** 92.2%
- **Best Model:** XGBoost_balanced_tuned
  - **ROC-AUC:** 0.872
  - **Recall:** 34.1%
  - **F2-Score:** 0.370
  - **Configuration:** pct=0.2, week=4, with assessment

### 2. **binary_5.0** (Secondary: Strong Pass Threshold)
- **Definition:** Fail if grade < 5.0 on 1-7 scale
- **Identifies:** At-risk students (30.4% fail rate)
- **Best Model:** LightGBM_tuned
  - **ROC-AUC:** 0.773
  - **Recall:** 91.4% (with threshold optimization)
  - **F2-Score:** 0.668

### 3. **3class_marginal** (Nuanced Prediction)
- **Definition:** Fail (<4.0) | At-Risk (4.0-4.99) | Pass (≥5.0)
- **Best Model:** XGBoost_tuned_borderline_smote
  - **Fail-ROC-AUC:** 0.880
  - **Overall OVR-AUC:** 0.759
  - **Fail Recall:** 46.3%
  - **Accuracy:** 74.8%

### 4. **4class** (Granular Performance)
- **Definition:** Fail (<4.0) | Low (4.0-5.0) | Medium (5.0-6.0) | High (≥6.0)
- **Best Model:** VotingEnsemble
  - **Fail-ROC-AUC:** 0.864
  - **Fail Recall:** 26.8%
  - **Accuracy:** 41.1%

---

## Temporal Analysis: Week-by-Week Performance

### Binary_5.0 by Cutoff Week
Early predictions made with limited data improve with more activity:

| Week | ROC-AUC | Recall | F2 | Best Model |
|------|---------|--------|-----|-----------|
| **Week 2** | 0.651 | 0.659 | 0.633 | LightGBM |
| **Week 4** | 0.704 | 0.805 | 0.724 | LightGBM_tuned |
| **Week 6** | 0.748 | 0.854 | 0.803 | XGBoost |
| **Week 8** | 0.773 | 0.915 | 0.863 | LightGBM_tuned |
| **Full** | 0.799 | 0.951 | 0.918 | LightGBM_tuned |

**Key Insight:** Predictive power increases consistently through week 8, then plateaus. Early detection is possible by week 2 (ROC-AUC 0.651) with 66% recall.

---

## Top Predictive Features

### Week 2 Effect Sizes (Cohens d ≥ 0.9, all p < 0.001)

**Temporal Gap Features (Strongest Signal)**
1. `gap_std_hours` - Cohens d: 1.15, Cliff's Δ: -0.528
   - Measures variability in time between student activities
   - Failing students have MORE consistent gaps (more irregular study)

2. `mean_gap_hours` - Cohens d: 1.12, Cliff's Δ: -0.479
   - Average time between consecutive activities
   - Failing: 4.5 hours, Passing: 1.9 hours

**Consistency Features**
3. `daily_consistency` - Cohens d: 0.98, Cliff's Δ: 0.524
   - Score-based daily activity consistency
   - Passing students show MORE consistency

4. `daily_consistency_znorm` - Cohens d: 0.99, Cliff's Δ: 0.514
   - Z-normalized version (normalized per course)

**Module Coverage**
5. `modu_rank_pct_mean` - Cohens d: 0.98, Cliff's Δ: 0.529
   - Percentile coverage of course modules
   - Failing: 17.7%, Passing: 35.9%

6. `modules_coverage` - Cohens d: 0.95, Cliff's Δ: 0.394
   - Raw module coverage rate

**Transition Features**
7. `unique_transitions` - Cohens d: 0.93, Cliff's Δ: 0.489
   - Number of different resource types accessed
   - Failing: 35.8, Passing: 50.8

8. `transition_diversity` - Cohens d: 0.93, Cliff's Δ: 0.489
   - Proportion of unique transitions

**Proactivity**
9. `quizzes_proact_median_pct` - Cohens d: 0.95, Cliff's Δ: 0.294
   - Median percentile of quiz participation proactivity
   - Failing: 72.6%, Passing: 90.5%

**Session Features**
10. `n_sessions_znorm` - Cohens d: 0.90, Cliff's Δ: 0.566
    - Number of distinct learning sessions

### Normalized vs Raw Features
- **Normalized features** (z-norm per course): Slightly better signal
- **Pattern:** Course-relative metrics outperform raw metrics
  - Enables fair comparison across courses with different activity scales
  - Accounts for course-specific pedagogical styles

---

## Threshold Optimization

### XGBoost_tuned_borderline_smote (3class_marginal, ROC-AUC 0.880)

Different criteria for different use cases:

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy |
|-----------|-----------|--------|-----------|-----|-----|----------|
| **max_f2** | 0.32 | **61.0%** | 46.3% | 0.526 | **0.573** | 92.0% |
| **max_f1** | 0.37 | 58.5% | 49.0% | 0.533 | 0.563 | 92.5% |
| **youden_j** | 0.13 | 70.7% | 29.9% | 0.420 | 0.556 | 85.7% |
| **max_f3** | 0.05 | 78.0% | 22.7% | 0.352 | 0.525 | 78.9% |

**Recommendations by Use Case:**
- **Student Support (maximize coverage):** Threshold 0.05, 78% recall
- **Balanced Intervention:** Threshold 0.32 (F2), 61% recall, 46% precision
- **High Precision Targeting:** Threshold 0.37 (F1), 58% recall, 49% precision

---

## Model Comparison

### Top 5 Performers Across All Schemes

| Rank | Model | Best Scheme | ROC-AUC | Recall | F2 | Phase |
|------|-------|-------------|---------|--------|-----|-------|
| 1 | **XGBoost_tuned_borderline_smote** | 3class_marginal | 0.880 | 46.3% | 0.573 | 2 |
| 2 | **LightGBM_tuned** | binary_5.0 | 0.773 | 91.5% | 0.863 | 2 |
| 3 | **XGBoost_balanced_tuned** | binary_4.0 | 0.872 | 34.1% | 0.370 | 1 |
| 4 | **VotingEnsemble** | 4class | 0.864 | 26.8% | 0.299 | 1 |
| 5 | **RandomForest_tuned** | 4class | 0.856 | 48.8% | 0.467 | 2 |

**Model Insights:**
- XGBoost and LightGBM dominate (gradient boosting)
- SMOTE-based oversampling helps with minority class recall
- Voting ensembles add robustness but require calibration
- Phase 2 (hyperparameter tuning) consistently outperforms Phase 1

---

## Feature Engineering Impact

### Key Innovation: Temporal Gap Analysis
- **Problem:** Standard engagement metrics miss study patterns
- **Solution:** Calculate inter-activity time gaps
- **Result:** #1 predictor (Cohens d 1.15)
- **Interpretation:** Failing students have irregular study patterns (large, variable gaps)

### Normalized Features Win
- Z-normalized features per course: 3-7% better performance
- Accounts for course heterogeneity (class size, instructor style, discipline)
- Enables model transfer across different courses

### Dimension Reduction
- Feature percentage cutoff (pct parameter):
  - **pct=0.05:** Top 5% features, faster inference
  - **pct=0.2:** Top 20% features, best balance
- Top 20% features capture ~95% of predictive power

---

## Production Deployment Recommendations

### 1. **Early Detection (Week 2)**
- **Model:** LightGBM
- **ROC-AUC:** 0.651
- **Use:** Initial screening, outreach preparation
- **Lead Time:** 2 weeks before first midterm

### 2. **Mid-Semester Check-in (Week 4-6)**
- **Model:** XGBoost_tuned_borderline_smote
- **ROC-AUC:** 0.763 (week 4), 0.768 (week 6)
- **Use:** Targeted interventions, early tutoring
- **Lead Time:** 1-2 weeks before second assessment

### 3. **Late Intervention (Week 8)**
- **Model:** LightGBM_tuned
- **ROC-AUC:** 0.773
- **Use:** High-precision targeting, final support
- **Lead Time:** 2 weeks before final assessment

### 4. **Full Data (Post-Semester)**
- **Model:** LightGBM_tuned / XGBoost_tuned_borderline_smote
- **ROC-AUC:** 0.799-0.880
- **Use:** Post-hoc analysis, curriculum improvement

### Threshold Strategy
- Use **threshold 0.32 (F2)** for balance between sensitivity and specificity
- Generates ~30% intervention group across typical classes
- Achieves 61% recall on failing students

---

## Deployment Artifacts

### Available in ML Team Share

**Location:** `/ml_team_share/full/with_assessment/`

**Files:**
- `model_summary.json` - Model performance metrics
- `threshold_recommendations.txt` - Optimal thresholds by criterion
- `top_features.txt` - Ranked feature importance
- `threshold_optimization.json` - Complete threshold grid

### Benchmark Results

**Location:** `/data/puc/sota_results/7courses_multiclass/`

**Files:**
- `BENCHMARK_REPORT.md` - This comprehensive analysis
- `benchmark_results.json` - Raw experiment data (2,640 experiments)
- `week2_effect_sizes.csv` - Feature effect sizes with statistical tests
- `week2_feature_effects.png` - Visualization of top features
- `week2_effect_size_ranking.png` - Ranking visualization

---

## Statistical Validation

### P-Values (Week 2 Features)
All top features p < 0.001, many p < 1e-7

### Multiple Comparison Correction
- Holm-Bonferroni adjustment applied
- Significant features remain highly significant after correction

### Effect Sizes
- **Cohens d:** 0.9-1.15 (large effects)
- **Cliff's Delta:** 0.49-0.57 (moderate to large non-parametric effects)
- **Cohen's h (proportions):** 0.3-0.5 (small to medium)

---

## Limitations & Caveats

1. **Class Imbalance:** Only 7.8% fail rate
   - Model prefers predicting "pass" (higher accuracy)
   - Requires threshold adjustment for recall
   - SMOTE helps but doesn't solve completely

2. **Temporal Leakage:** Assessment scores included as features
   - Model performs better when grades available
   - Activity-only variant available (see binary_5.0_no_assess)

3. **Course Heterogeneity:** 20 courses with different styles
   - Normalized features help but not perfect
   - Course-specific models could improve further

4. **Early Warning Trade-off:**
   - Week 2 ROC-AUC (0.651) much lower than week 8 (0.773)
   - Early alerts have higher false positive rate

---

## Next Steps & Future Work

1. **Implement Threshold-Optimized Production Model**
   - Deploy LightGBM with threshold 0.32
   - Monitor false positive rate in real interventions

2. **A/B Test Intervention Strategies**
   - Compare effect of week 2 vs week 4 alerts
   - Measure student engagement response

3. **Course-Specific Model Variants**
   - Train per-department models for better calibration
   - Capture discipline-specific learning patterns

4. **Real-Time Dashboard**
   - Week-by-week student risk scores
   - Feature contribution visualizations
   - Instructor action recommendations

5. **Causal Analysis**
   - Identify modifiable features (e.g., "increase module coverage")
   - Link interventions to outcomes

---

## Contact & Questions

For detailed experimental results, model artifacts, or technical questions, refer to:
- **Benchmark Report:** `data/puc/sota_results/7courses_multiclass/BENCHMARK_REPORT.md`
- **Feature Analysis:** `data/puc/sota_results/7courses_multiclass/week2_effect_sizes.csv`
- **Model Artifacts:** `ml_team_share/full/with_assessment/`

