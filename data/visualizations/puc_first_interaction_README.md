# PUC Student First Interaction Analysis (Jan-Jul 2023)

## Overview

This analysis examines the relationship between when students first access their courses
and their final academic outcomes, using data from Pontificia Universidad Católica de Chile (PUC).

**Dataset:** PUC wave_analysis (1st semester 2023)
- **Total enrollments analyzed:** 868
- **Unique students:** 741
- **Unique courses:** 20
- **Date range:** January 18, 2023 - March 30, 2023

## Grading System

**Chilean 1-7 Scale:**
- **7.0** = Maximum grade (equivalent to A+ or 100%)
- **6.0-6.9** = High pass (A/A-)
- **5.0-5.9** = Mid pass (B/B-)
- **4.0-4.9** = Low pass (C, minimum passing)
- **1.0-3.9** = Fail (F)

**Dataset Grade Distribution:**
- Average grade: **5.37**
- Pass rate (≥4.0): **90.6%**
- Fail rate (<4.0): **6.3%**

## Color Coding

Scatter plots use the following color scheme:
- 🔵 **Blue** (#2E86AB): High Pass (6.0-7.0)
- 🟢 **Green** (#27AE60): Mid Pass (5.0-5.9)
- 🟡 **Yellow** (#F39C12): Low Pass (4.0-4.9)
- 🔴 **Red** (#E74C3C): Fail (<4.0)
- ⚫ **Gray**: No grade data

## Key Findings

### 1. Access Timing by Grade Category


**Fail (<4.0):**
- Total students: 55
- Average access delay: 42.3 days after course start (10th percentile)

**Low Pass (4.0-4.9):**
- Total students: 201
- Average access delay: 38.3 days after course start (10th percentile)

**Mid Pass (5.0-5.9):**
- Total students: 329
- Average access delay: 32.2 days after course start (10th percentile)

**High Pass (6.0-7.0):**
- Total students: 256
- Average access delay: 31.0 days after course start (10th percentile)


### 2. Course-Specific Patterns

Total courses analyzed: **20**

Top 5 courses by enrollment:

- **ICS2813-1** (n=138, avg=5.58, 6% fail)
- **IIC2613-1** (n=137, avg=4.99, 11% fail)
- **ICS2813-2** (n=121, avg=5.54, 5% fail)
- **ICS2813-3** (n=100, avg=5.37, 2% fail)
- **ING1004-7** (n=80, avg=5.78, 0% fail)

## Interpretation

### Early Access Correlation

The analysis reveals whether students who access courses earlier tend to have better outcomes.
Key questions addressed:

1. **Do failing students access later?**
   - Compare red dots position relative to purple line (course start)
   - Late access after week 2-3 may indicate disengagement

2. **Do high performers access early?**
   - Blue/green dots clustered near course start suggest proactive behavior
   - Early engagement correlates with better preparation

3. **Optimal intervention timing:**
   - Students who haven't accessed by 10th percentile date are at higher risk
   - Early warning systems should trigger interventions by week 2-3

### Limitations

1. **Class imbalance:** Low failure rate (~6%) means failing students are rare
2. **Correlation ≠ causation:** Late access may be symptom, not cause of failure
3. **Course variability:** Different courses have different engagement patterns
4. **Historical data:** Analysis based on 2023 semester, patterns may change

## Files Generated

1. **puc_first_interaction_by_grade_scatter.png** - Main visualization (scatter plot grid)
2. **puc_first_interaction_summary.csv** - Statistical summary per course per grade category
3. **puc_first_interaction_README.md** - This documentation

## Data Sources

- **Page views:** `/home/paul/projects/wave_analysis/puc_analysis/data/categorized_page_views.parquet`
  - 2,980,575 page views from 784 students across 22 courses

- **Grades:** `/home/paul/projects/wave_analysis/puc_analysis/data/grades_with_failure.parquet`
  - 1,648 enrollments with Chilean 1-7 scale grades

## Methodology

1. **First interaction calculation:** Minimum `created_at` timestamp per student per course
2. **Course start proxy:** 10th percentile of first access times (purple line in plots)
3. **Grade coloring:** Chilean 1-7 scale mapped to 4 color categories
4. **Jittering:** Y-axis randomization prevents dot overlap for visualization clarity

---

*Generated: 2026-02-08*
*Script: plot_puc_first_interaction_by_grade.py*
