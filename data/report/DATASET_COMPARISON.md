# Dataset Comparison: PUC vs UA

## Overview

This report compares student grade data from two Chilean universities:
- **PUC** (Pontificia Universidad Católica de Chile)
- **UA** (Universidad Autónoma de Chile)

## Summary Statistics

| Metric | PUC | UA |
|--------|-----|-----|
| Total Enrollments | 1,648 | 373 |
| With Grades | 1,607 | 373 |
| Unique Students | 1,495 | 215 |
| Unique Courses | 24 | 10 |
| Grade Scale | 1-7 | 0-100% |
| Pass Threshold | >= 4.0 | >= 57.0% |
| Mean Grade | 5.35 | 58.15% |
| Std Dev | 1.07 | 34.38% |
| Median Grade | 5.50 | 70.59% |
| **Failure Rate** | **7.8%** | **39.9%** |

## Key Differences

1. **Sample Size**: PUC has 4.4x more enrollments than UA (1,648 vs 373)
2. **Failure Rate**: UA has significantly higher failure rate (39.9% vs 7.8%)
3. **Course Coverage**: PUC covers 24 courses, UA covers 10 courses
4. **Grade Variability**: UA shows higher variance (σ=34.38%) indicating more diverse outcomes

## Grade Distribution Comparison

When normalized to a 0-100 scale:
- PUC pass threshold (4.0 on 1-7 scale) ≈ 50%
- UA pass threshold is 57%

This means UA applies a **stricter passing standard** relative to their grade scale.

## PUC Course-Level Statistics

| Course | Students | Mean Grade | Failure Rate |
|--------|----------|------------|--------------|
| IIQ213Q-1 | 31 | 4.19 | 26.0% |
| IIQ2013-1 | 22 | 4.67 | 23.0% |
| IIQ2043-1 | 62 | 4.46 | 19.0% |
| IIC1253-1 | 107 | 4.61 | 18.0% |
| IIC2026-1 | 82 | 5.78 | 16.0% |
| IIC2233-2 | 100 | 5.14 | 15.0% |
| IIC2613-1 | 124 | 4.99 | 12.0% |
| IIQ2023-2 | 16 | 5.34 | 12.0% |
| IIC2213-1 | 92 | 5.33 | 8.0% |
| ICT2223-1 | 27 | 4.90 | 7.0% |
| ... | ... | ... | ... |
| ING1004-1 | 61 | 5.61 | 0.0% |
| ING1004-7 | 80 | 5.78 | 0.0% |
| ING2030-6 | 56 | 5.91 | 0.0% |

*Showing top 10 and bottom 3 courses by failure rate (24 total)*

## UA Course-Level Statistics

| Course ID | Students | Mean Grade | Failure Rate |
|-----------|----------|------------|--------------|
| 86676 | 40 | 38.81% | 72.0% |
| 84941 | 38 | 35.09% | 63.0% |
| 84944 | 40 | 56.18% | 45.0% |
| 79875 | 32 | 58.84% | 41.0% |
| 86020 | 51 | 59.07% | 37.0% |
| 84936 | 42 | 68.91% | 29.0% |
| 89099 | 35 | 61.10% | 29.0% |
| 88381 | 21 | 68.47% | 29.0% |
| 79913 | 41 | 65.44% | 27.0% |
| 89390 | 33 | 75.99% | 21.0% |

## Visualizations

![Summary Table](visualizations/dataset_summary_table.png)

![Grade Distributions](visualizations/dataset_grade_distributions.png)

![Normalized Comparison](visualizations/dataset_normalized_comparison.png)

![Failure Rates by Course](visualizations/dataset_failure_rates.png)

![Course Size Distribution](visualizations/dataset_course_sizes.png)

![Grade Boxplots by Course](visualizations/dataset_grade_boxplots.png)

## Implications for Model Development

### Class Imbalance
- **PUC**: Highly imbalanced (6.3% failures) - may require oversampling or class weights
- **UA**: More balanced (39.9% failures) - better for binary classification

### Generalization
- Models trained on one university may not transfer well due to:
  - Different pass thresholds
  - Different failure rates
  - Different course types and student populations

### Recommended Approach
1. Train separate models per university initially
2. Test cross-university transfer with domain adaptation
3. Consider normalizing features within each course to reduce institutional bias

---

*Generated: 2026-02-03 10:58*
