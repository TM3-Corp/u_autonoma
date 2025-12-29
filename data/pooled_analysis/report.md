# Pooled Binary Classification Report
## Student Failure Prediction Using Pure Activity Features

Generated: 2025-12-28 11:43

---

## Executive Summary

This analysis trained binary classifiers on **373 students** pooled across multiple courses
to predict academic failure (final grade < 57%) using **only pure activity features** (no grade leakage).

### Key Metrics
| Metric | Value |
|--------|-------|
| Total Students | 373 |
| Passing (≥57%) | 224 (60.1%) |
| Failing (<57%) | 149 (39.9%) |
| Best Model | Xgboost |
| Best ROC-AUC | 0.787 |
| Best Recall | 61.7% |

---

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 0.651 | 0.559 | 0.604 | 0.581 | 0.707 |
| Random Forest | 0.729 | 0.703 | 0.557 | 0.622 | 0.780 |
| Xgboost | 0.740 | 0.697 | 0.617 | 0.655 | 0.787 |

**Note:** Recall is prioritized as it measures the ability to catch at-risk students.

---

## Actionable Insights

The following insights are derived from statistical analysis of student activity patterns.
Only statistically significant findings (p < 0.05) with meaningful effect sizes (RR > 1.5 or < 0.67) are reported.

### 1. Sessions Per Week (RISK FACTOR)
⚠️ **Students with low sessions per week have 101% higher failure risk (failure rate: 53.2% vs 26.5%) (p < 0.001)**

- Relative Risk: **2.01x**
- Statistical significance: p = 0.0000
- Sample sizes: n=185 (high) vs n=188 (low)

### 2. Total Page Views (RISK FACTOR)
⚠️ **Students with low total page views have 93% higher failure risk (failure rate: 52.4% vs 27.2%) (p < 0.001)**

- Relative Risk: **1.93x**
- Statistical significance: p = 0.0000
- Sample sizes: n=184 (high) vs n=189 (low)

### 3. Total Number Of Study Sessions (RISK FACTOR)
⚠️ **Students with low total number of study sessions have 82% higher failure risk (failure rate: 51.3% vs 28.3%) (p < 0.001)**

- Relative Risk: **1.82x**
- Statistical significance: p = 0.0000
- Sample sizes: n=184 (high) vs n=189 (low)

### 4. Unique Active Hours (RISK FACTOR)
⚠️ **Students with low unique active hours have 82% higher failure risk (failure rate: 51.3% vs 28.3%) (p < 0.001)**

- Relative Risk: **1.82x**
- Statistical significance: p = 0.0000
- Sample sizes: n=184 (high) vs n=189 (low)

### 5. Weekend Studying (RISK FACTOR)
⚠️ **Students with low weekend studying have 81% higher failure risk (failure rate: 48.5% vs 26.7%) (p < 0.001)**

- Relative Risk: **1.81x**
- Statistical significance: p = 0.0000
- Sample sizes: n=146 (high) vs n=227 (low)

### 6. Studying In The Evening (6Pm-10Pm) (RISK FACTOR)
⚠️ **Students with low studying in the evening (6pm-10pm) have 76% higher failure risk (failure rate: 57.0% vs 32.4%) (p < 0.001)**

- Relative Risk: **1.76x**
- Statistical significance: p = 0.0000
- Sample sizes: n=259 (high) vs n=114 (low)

### 7. Average Gap Between Sessions (Days) (RISK FACTOR)
⚠️ **Students with high average gap between sessions (days) have 62% higher failure risk (failure rate: 49.5% vs 30.5%) (p < 0.001)**

- Relative Risk: **1.62x**
- Statistical significance: p = 0.0003
- Sample sizes: n=186 (high) vs n=187 (low)

### 8. Increasing Engagement Over Time (RISK FACTOR)
⚠️ **Students with low increasing engagement over time have 40% higher failure risk (failure rate: 46.5% vs 33.3%) (p < 0.05)**

- Relative Risk: **1.40x**
- Statistical significance: p = 0.0126
- Sample sizes: n=186 (high) vs n=187 (low)

---

## Top Predictive Features

### Feature Importance (from best model)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | session_count | 0.0870 |
| 2 | peak_ratio | 0.0414 |
| 3 | dct_coef_8 | 0.0400 |
| 4 | weekday_night_pct | 0.0329 |
| 5 | slope_std | 0.0328 |
| 6 | negative_slope_sum | 0.0292 |
| 7 | weekday_evening_pct | 0.0261 |
| 8 | first_access_day | 0.0250 |
| 9 | dct_coef_7 | 0.0250 |
| 10 | weekend_night_pct | 0.0247 |

### Logistic Regression Coefficients

| Feature | Coefficient | Odds Ratio | Interpretation |
|---------|-------------|------------|----------------|
| dct_coef_0 | -1.705 | 0.182 | 5.50x less likely to fail |
| first_access_day | -0.979 | 0.376 | 2.66x less likely to fail |
| dct_coef_1 | 0.931 | 2.538 | 2.54x more likely to fail |
| weekend_evening_pct | 0.884 | 2.420 | 2.42x more likely to fail |
| dct_coef_2 | -0.786 | 0.456 | 2.19x less likely to fail |
| access_time_pct | 0.767 | 2.152 | 2.15x more likely to fail |
| slope_std | -0.726 | 0.484 | 2.07x less likely to fail |
| session_gap_mean | 0.642 | 1.899 | 1.90x more likely to fail |
| positive_slope_sum | 0.595 | 1.814 | 1.81x more likely to fail |
| activity_span_days | -0.578 | 0.561 | 1.78x less likely to fail |

---

## Recommendations for Early Intervention

Based on the analysis, the following student profiles are at higher risk of failure:

### High-Risk Indicators
- **Sessions Per Week**: 26.5% failure rate vs 53.2%
- **Total Page Views**: 27.2% failure rate vs 52.4%
- **Total Number Of Study Sessions**: 28.3% failure rate vs 51.3%
- **Unique Active Hours**: 28.3% failure rate vs 51.3%
- **Weekend Studying**: 26.7% failure rate vs 48.5%

### Protective Factors

---

## Methodology Notes

1. **Data Leakage Prevention**: Features directly tied to grades (submissions, tardiness) were excluded
2. **Course-Agnostic Features**: All features were z-score normalized within each course
3. **Cross-Validation**: 5-fold stratified cross-validation was used for all model evaluation
4. **Class Imbalance**: Models used class weighting to handle the pass/fail imbalance
5. **Statistical Testing**: Chi-square or Fisher's exact test used for significance testing

---

## Files Generated

- `model_results.json`: Detailed model metrics and predictions
- `actionable_insights.json`: All generated insights with statistics
- `roc_curves.png`: ROC curves comparing all models
- `feature_importance.png`: Top features by importance
- `risk_factors.png`: Visualization of significant risk factors

---

*Report generated by pooled_binary_classifier.py*
