# Cumulative First Interaction Timeline Analysis

## Overview

This analysis shows **cumulative** first interaction patterns - the percentage of enrolled students who have accessed each course **at least once** by each date. The Y-axis starts at 0% and increases cumulatively as more students access the course for the first time.

## Data Source

- **Page Views**: `data/page_views/categorized_page_views.parquet`
  - Total records: 998,395
  - Unique users: 209
  - Unique courses with page views: 288
  - Date range: March 1 - December 30, 2025

- **Enrollments**: `data/page_views/student_enrollments.csv`
  - Total enrollments: 373
  - Courses with enrollment data: 10

- **Matched Courses**: 10 (courses with both page views AND enrollment data)

## Key Findings

### Course Engagement Speed

The analysis reveals how quickly students begin accessing their courses:

#### Fast Adopters (50% accessed within 2 weeks):
1. **TALLER DE COMP DIGITALES-P01**: 50% by day 5, 80% by day 9
2. **FUND. DE BUSINESS ANALYTICS-P01**: 50% by day 6, 80% by day 13
3. **TALL DE COMPETENCIAS DIGITALES-P02**: 50% by day 8, 80% by day 16
4. **FUND DE BUSINESS ANALYTICS-P01**: 50% by day 14, 80% by day 20

#### Slow Adopters (50% accessed after 3+ weeks):
1. **FUNDAMENTOS DE MACROECONOMÍA-P03**: 50% by day 39, 80% NOT reached (only 97.5% final)
2. **FUNDAMENTOS DE MICROECONOMÍA-P03**: 50% by day 31, 80% by day 51
3. **GESTIÓN DEL TALENTO-P01**: 50% by day 22, 80% by day 42
4. **FUNDAMENTOS DE MICROECONOMÍA-P01**: 50% by day 21, 80% by day 54 (only 86.8% final)

### Final Access Rates

| Course | Enrolled | Final % Accessed | Status |
|--------|----------|------------------|--------|
| TALLER DE COMP DIGITALES-P01 | 32 | 100.0% | ✅ Full participation |
| FUND. DE BUSINESS ANALYTICS-P01 | 41 | 100.0% | ✅ Full participation |
| TALL DE COMPETENCIAS DIGITALES-P02 | 51 | 100.0% | ✅ Full participation |
| FUND DE BUSINESS ANALYTICS-P01 | 40 | 100.0% | ✅ Full participation |
| MATEMÁTICAS PARA LOS NEGOCIOS-P01 | 21 | 100.0% | ✅ Full participation |
| FUNDAMENTOS DE MACROECONOMÍA-P03 | 40 | 97.5% | ⚠️ 1 never accessed |
| TALLER DE COMP DIGITALES-P01 | 35 | 97.1% | ⚠️ 1 never accessed |
| GESTIÓN DEL TALENTO-P01 | 33 | 97.0% | ⚠️ 1 never accessed |
| FUNDAMENTOS DE MICROECONOMÍA-P03 | 42 | 95.2% | ⚠️ 2 never accessed |
| FUNDAMENTOS DE MICROECONOMÍA-P01 | 38 | 86.8% | ❌ 5 never accessed |

### Pattern Analysis

**Three distinct engagement patterns emerge:**

1. **Rapid Engagement (Digital Skills courses)**
   - 50% access within first week
   - 80% access within 2-3 weeks
   - 100% final participation
   - Likely due to: Required early assignments, online nature of content

2. **Moderate Engagement (Business Analytics, Math)**
   - 50% access within 2 weeks
   - 80% access within 3 weeks
   - Near 100% final participation
   - Gradual but consistent student onboarding

3. **Delayed Engagement (Economics courses)**
   - 50% access takes 3-5 weeks
   - 80% access takes 7+ weeks (or never reached)
   - 87-97% final participation
   - Suggests: Late semester engagement, exam-driven access patterns

## Implications for Early Warning Models

### Why Early-Cutoff AUC is Low (0.55-0.65)

The cumulative first interaction data explains the low predictive power in early weeks:

1. **Weeks 2-4**: Only 30-60% of students have accessed most courses
   - Not enough signal to distinguish engaged vs disengaged
   - Many "future-engaged" students haven't started yet

2. **Week 6+**: 70-90% of students have accessed courses
   - Better signal for identifying never-access or very-late students
   - Explains why AUC improves in later weeks

### Optimal Intervention Timing

Based on this data:

- **Week 2-3**: Too early - many students legitimately haven't started
- **Week 4-5**: Optimal - can identify students who are significantly delayed
- **Week 6+**: Too late - patterns already established

### Course-Specific Strategies

**Fast-adoption courses** (Digital Skills):
- Can use early warning at Week 2-3
- Students who haven't accessed by Week 2 are true outliers

**Slow-adoption courses** (Economics):
- Must wait until Week 4-5 for reliable signals
- Earlier warnings would generate false alarms

## Files Generated

### Grid View
- **File**: `cumulative_first_interaction_timeline.png`
- **Format**: 4×3 grid (10 courses)
- **Shows**: Cumulative % of students accessed over time

### Individual Plots
- **Directory**: `cumulative_timelines_individual/`
- **Count**: 10 detailed plots (one per course)
- **Features**:
  - 50% and 100% reference lines
  - Clear cumulative progression from 0% to ~100%
  - Date-level granularity

### Summary Statistics
- **File**: `cumulative_first_interaction_summary.csv`
- **Columns**:
  - `course_id`: Course identifier
  - `course_name`: Full course name
  - `total_enrolled`: Number of enrolled students
  - `final_pct_accessed`: Final % who accessed at least once
  - `days_tracked`: Number of days with activity
  - `days_to_50pct`: Days to reach 50% first access
  - `days_to_80pct`: Days to reach 80% first access
  - `date_first_access`: First recorded access
  - `date_last_access`: Last recorded access

## Comparison with Previous Analysis

**Previous (Incorrect) Analysis:**
- Showed **daily** % accessed (non-cumulative)
- Spiky patterns with high variability
- Average daily access: 11-28%

**Current (Correct) Analysis:**
- Shows **cumulative** first interaction
- Smooth S-curve progression
- Final access: 87-100%

## Next Steps

1. **Correlate with Outcomes**: Compare access timing curves with final grades
2. **Identify Risk Signatures**: Students who never access OR access very late
3. **Build Temporal Features**: Use "days to first access" as predictive feature
4. **Course Clustering**: Group courses by engagement pattern for targeted models

## Technical Notes

- User IDs normalized from Canvas shard format (155100000000XXXXX → XXXXX)
- Timezone-naive datetime comparison to avoid inconsistencies
- Only courses with both page views AND enrollment records included
- Cumulative calculation: For each date, count students whose first access ≤ that date
