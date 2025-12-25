# Pure Activity Correlation Analysis Summary

## Universidad Autonoma de Chile - Canvas LMS
**Date:** December 2025
**Analysis:** Control de Gestion (Account 719)

---

## Overview

**Sample:** 186 students across 5 courses
**Features analyzed:** 14 pure activity features (no submission-based data leakage)

---

## Key Findings

### 1. CONSISTENTLY STRONG PREDICTORS

These features predict grades reliably across ALL courses:

| Feature | Avg Correlation | Direction | Interpretation |
|---------|-----------------|-----------|----------------|
| `unique_active_hours` | **+0.36** | Consistent + | More diverse engagement hours = better grades |
| `total_activity_time` | **+0.36** | Consistent + | More time spent = better grades |
| `avg_gap_hours` | **-0.35** | Consistent - | Longer gaps between sessions = worse grades |

### 2. YOUR HYPOTHESIS CONFIRMED

| Feature | Avg Correlation | Finding |
|---------|-----------------|---------|
| `gap_std_hours` | **-0.29** | High variance in inactivity time = at-risk indicator |

Students with irregular study patterns (high standard deviation in gaps between sessions) tend to have lower grades.

### 3. PREVIOUS FINDINGS INVALIDATED

| Feature | Previous Claim | Current Finding |
|---------|---------------|-----------------|
| `is_morning_studier` | "0% failure rate" | r = -0.07 (NO consistent effect) |
| `is_evening_studier` | "67% failure rate" | r = -0.05 (NO consistent effect) |

**Why the previous findings were wrong:**
- Only 5 morning studiers, 3 evening studiers with valid grades
- Many students with final_score=0 were incorrectly counted as "failures"
- The finding was a small-sample artifact, not a real pattern

---

## Detailed Per-Course Results

### Course 1: TALL DE COMPETENCIAS DIGITALES-P01 (n=50)
- Grade range: 27-91%, mean=72.4%, fail_rate=12%
- **Best predictors:**
  - `unique_active_hours`: r = +0.505 (STRONG)
  - `afternoon_activity`: r = +0.495 (MODERATE)
  - `activity_span_days`: r = +0.417 (MODERATE)
  - `gap_std_hours`: r = -0.382 (MODERATE)

### Course 2: TALL DE COMPETENCIAS DIGITALES-P02 (n=47)
- Grade range: 6-92%, mean=64.1%, fail_rate=32%
- **Best predictors:**
  - `avg_gap_hours`: r = -0.525 (STRONG)
  - `unique_active_hours`: r = +0.428 (MODERATE)
  - `total_activity_time`: r = +0.362 (MODERATE)

### Course 3: FUNDAMENTOS DE MACROECONOMIA-P03 (n=38)
- Grade range: 24-88%, mean=59.1%, fail_rate=42%
- **Best predictors:**
  - `avg_gap_hours`: r = -0.428 (MODERATE)
  - `gap_std_hours`: r = -0.416 (MODERATE)
  - `page_views_level`: r = +0.406 (MODERATE)

### Course 4: FUND DE BUSINESS ANALYTICS-P01 (n=36)
- Grade range: 15-82%, mean=43.1%, fail_rate=69%
- **Best predictors:**
  - `unique_active_hours`: r = +0.534 (STRONG)
  - `total_activity_time`: r = +0.457 (MODERATE)
  - `gap_std_hours`: r = -0.449 (MODERATE)
  - `avg_gap_hours`: r = -0.445 (MODERATE)

### Course 5: FUNDAMENTOS DE MICROECONOMIA-P01 (n=15)
- Grade range: 50-100%, mean=88.9%, fail_rate=7%
- **Best predictors:**
  - `total_activity_time`: r = +0.418 (MODERATE)
- Note: Low variance in grades limits correlation detection

---

## Features Ranked by Predictive Power

### Pure Activity Features (NO data leakage)

| Rank | Feature | Avg r | Consistency | Actionable? |
|------|---------|-------|-------------|-------------|
| 1 | `unique_active_hours` | +0.36 | YES | Monitor engagement diversity |
| 2 | `total_activity_time` | +0.36 | YES | Track total time in LMS |
| 3 | `avg_gap_hours` | -0.35 | YES | Alert on long inactivity |
| 4 | `gap_std_hours` | -0.29 | Mostly | Flag irregular patterns |
| 5 | `afternoon_activity` | +0.22 | Mixed | - |
| 6 | `page_views` | +0.21 | Mixed | Basic engagement metric |
| 7 | `evening_activity` | +0.19 | YES | - |
| 8 | `page_views_level` | +0.19 | Mixed | Canvas-computed level |
| 9 | `morning_activity` | +0.16 | Mixed | - |
| 10-14 | Others | <0.10 | - | Not useful |

### Submission-Related Features (potential data leakage)

| Feature | Avg r | Note |
|---------|-------|------|
| `missing` | -0.66 | Directly related to grades |
| `on_time` | +0.62 | Directly related to grades |
| `participations` | +0.59 | May include submission events |

---

## Recommended Early Warning Indicators

Based on this analysis, an early warning system should monitor:

1. **Total Activity Time** - Flag students below 25th percentile
2. **Average Gap Between Sessions** - Flag students with gaps > 72 hours
3. **Gap Variance** - Flag students with high irregularity in study patterns
4. **Unique Active Hours** - Flag students with < 10 unique hours of activity

### Risk Score Formula (proposed)

```
risk_score =
    - 0.36 * normalize(unique_active_hours)
    - 0.36 * normalize(total_activity_time)
    + 0.35 * normalize(avg_gap_hours)
    + 0.29 * normalize(gap_std_hours)
```

---

## Files Generated

| File | Description |
|------|-------------|
| `all_students_features.csv` | Raw data for 186 students, all features |
| `correlations_by_course.json` | Per-course correlation results |
| `average_correlations.json` | Cross-course average correlations |
| `correlation_heatmaps.png` | Visual: per-course heatmaps |
| `correlation_summary_heatmap.png` | Visual: cross-course comparison |

---

## Limitations

1. **Sample size:** 186 students is moderate but may miss weak effects
2. **Course variety:** All courses are from Control de Gestion program
3. **Temporal factors:** Data is from 2nd semester 2025 only
4. **LMS design:** Correlation strength depends on course design

---

## Next Steps

1. **Validate on POSTGRADO courses** - 17 courses with grades available
2. **Build predictive model** using only pure activity features
3. **Test early warning thresholds** on held-out data
4. **Analyze temporal patterns** - week-by-week activity changes

---

*Generated: December 2025*
