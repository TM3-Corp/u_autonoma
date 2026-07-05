# PUC Cumulative Access Timeline Analysis (Jan-Jul 2023)

## Overview

This analysis examines the relationship between **when students first access their courses**
and their **final academic outcomes**, using a cumulative percentage visualization.

**Visualization Type:** Cumulative Access Timeline
- **X axis:** Date of first access
- **Y axis:** Cumulative % of students who have accessed (0-100%)
- **Each dot:** One student accessing at that specific time
- **Color:** Final grade (Chilean 1-7 scale)

**Dataset:** PUC wave_analysis (1st semester 2023)
- **Total enrollments:** 868
- **Unique students:** 741
- **Unique courses:** 20
- **Date range:** January 18, 2023 - March 30, 2023

## Key Finding

**Early accessors significantly outperform late accessors:**
- **Early 25% (first to access):** Average grade = **5.41**
- **Late 25% (last to access):** Average grade = **5.12**
- **Grade difference:** **0.29 points** on 1-7 scale

This is a **29.4%** relative difference in the grading scale.

## Interpretation

### What the Visualization Shows

1. **Bottom-left dots (early accessors, 0-25%):**
   - If mostly blue/green → early access correlates with success
   - These students access within days of course start

2. **Top-right dots (late accessors, 75-100%):**
   - If red/orange → late access correlates with failure
   - These students access weeks after course start

3. **Slope of the curve:**
   - **Steep:** Most students onboard quickly (engaged cohort)
   - **Flat:** Students trickle in slowly (disengaged cohort)

### Early Warning Implications

Students who haven't accessed by the **50% mark** (horizontal gray line) are at higher risk:
- They're accessing later than most of their peers
- Late access correlates with lower grades
- Intervention should happen BEFORE this threshold

## Data Source & Processing

**Page Views Data:**
```
/home/paul/projects/wave_analysis/puc_analysis/data/categorized_page_views.parquet
```
- 2,980,575 page views from 784 students

**Processing:**
1. Calculate first interaction: `MIN(created_at)` per student per course
2. Sort students by first interaction time (ascending)
3. Calculate cumulative percentage: student #i out of N = (i/N) × 100%
4. Plot each student as dot at (timestamp, cumulative_%)
5. Color by final grade

**Grades Data:**
```
/home/paul/projects/wave_analysis/puc_analysis/data/grades_with_failure.parquet
```
- 1,648 enrollments with Chilean 1-7 scale grades

## Files Generated

1. **puc_cumulative_access_by_grade.png** - Main visualization (cumulative timeline grid)
2. **puc_access_timing_analysis.csv** - Quartile analysis (early vs late accessors)
3. **puc_cumulative_access_README.md** - This documentation

## Color Coding

- 🔵 **Blue** (#2E86AB): High Pass (6.0-7.0)
- 🟢 **Green** (#27AE60): Mid Pass (5.0-5.9)
- 🟡 **Yellow** (#F39C12): Low Pass (4.0-4.9)
- 🔴 **Red** (#E74C3C): Fail (<4.0)
- ⚫ **Gray**: No grade data

## Limitations

1. **Correlation ≠ causation:** Late access may be symptom, not cause of poor performance
2. **Course variability:** Different courses have different onboarding patterns
3. **External factors:** Students may have valid reasons for late enrollment
4. **Historical data:** Based on 2023 semester, patterns may evolve

---

*Generated: 2026-02-08 15:34:11*
*Script: plot_puc_cumulative_access_by_grade.py*
