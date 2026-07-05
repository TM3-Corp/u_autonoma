# Course Activity Timeline Visualizations

## Overview

Generated visualizations showing daily student access patterns for 6 courses with available page view and enrollment data.

## Data Source

- **Page Views**: `data/page_views/categorized_page_views.parquet`
  - Total records: 998,395
  - Unique users: 209
  - Date range: March 1, 2025 - December 30, 2025 (10 months)
  - Courses with page views: 288

- **Course Metadata**: `data/postgrado_courses_with_grades.json`
  - Total courses: 23
  - Courses with both page views AND enrollment data: 6

## Key Findings

### Activity Patterns

The visualizations reveal several important engagement patterns:

1. **"GESTIÓN DEL TALENTO-P01"** shows the highest average daily access (27.7%)
   - Peak activity: 102.5% of enrolled students (indicates some students accessed multiple times or guests)
   - Consistent engagement over 128 days

2. **"TALL DE COMPETENCIAS DIGITALES"** courses (P01 and P02) show moderate engagement (20-23% average)
   - P02: 72.5% max, 23.1% average over 153 days
   - P01: 66.0% max, 20.8% average over 149 days

3. **"FUNDAMENTOS DE MICROECONOMÍA"** courses show lower average engagement (11-13%)
   - Despite lower averages, both courses have high peaks (76-90%)
   - Suggests concentrated activity around specific events (exams, assignments)
   - P03: 90.5% max, 12.8% average over 76 days
   - P01: 76.3% max, 11.5% average over 79 days

### Activity Variability

All courses show high standard deviations relative to their means, indicating:
- **Spiky engagement patterns** rather than consistent daily access
- Activity concentrated around key course events (assignments, exams)
- Low baseline engagement with periodic surges

### Max > 100% Explained

"GESTIÓN DEL TALENTO-P01" shows 102.5% max daily access, which can occur due to:
- Students accessing from multiple devices/sessions (counted as distinct page views)
- Course observers/auditors not in official enrollment
- Data collection capturing all access, not just enrolled students

## Files Generated

### Grid View
- **File**: `course_activity_timelines.png`
- **Format**: 4×2 grid (6 courses)
- **Shows**: Each course's daily activity timeline with enrollment context

### Individual Plots
- **Directory**: `individual_timelines/`
- **Count**: 6 detailed plots (one per course)
- **Features**:
  - Larger format for detailed analysis
  - Mean activity line for reference
  - Date-level granularity

### Summary Statistics
- **File**: `course_activity_summary.csv`
- **Columns**:
  - `course_id`: Canvas LMS course identifier
  - `course_name`: Full course name
  - `total_enrolled`: Number of enrolled students
  - `days_with_activity`: Days with at least one student access
  - `max_pct_accessed`: Peak daily access percentage
  - `mean_pct_accessed`: Average daily access percentage
  - `median_pct_accessed`: Median daily access percentage
  - `min_pct_accessed`: Minimum daily access percentage
  - `std_pct_accessed`: Standard deviation of daily access

## Usage for Analysis

These visualizations can inform:

1. **Early Warning Models**: Low early-semester engagement correlates with higher failure risk
2. **Course Design**: Identify courses needing better engagement strategies
3. **Intervention Timing**: Optimal timing for outreach based on typical activity patterns
4. **Resource Allocation**: Focus support on courses with consistently low engagement

## Limitations

- Only 6 of 23 courses with grades had matching page view data
- Data represents "second semester" 2025 courses (March-December)
- Percentages calculated against total enrollment, not active enrollment
- Does not account for course withdrawals or inactive students

## Next Steps

To expand this analysis:
1. Match more courses between page views and enrollment databases
2. Analyze weekly patterns (day-of-week effects)
3. Correlate activity patterns with grade outcomes
4. Identify "at-risk" activity signatures for early intervention
