# Discovery Scripts - Canvas LMS Course Analysis

This folder contains scripts for discovering and analyzing Canvas LMS courses to identify candidates for predictive modeling (Early Warning System).

---

## Quick Start

```bash
# 1. Analyze courses for LMS design metrics
python section7_refactor.py --campus-ids "176" --max-courses 100

# 2. Analyze courses for activity metrics
python activity_analysis.py --campus-ids "176" --max-courses 100

# 3. Deep statistical analysis of results
python deep_activity_analysis.py --input data/discovery/activity_analysis_latest.csv

# 4. Generate final combined report
python final_report_generator.py
```

---

## Scripts to Run (Execution Order)

### 1. `section7_refactor.py` - LMS Design Analysis

**Purpose:** Analyzes course structure and grade availability to calculate a "Prediction Potential Score".

**What it extracts:**
- Enrollment data and grades (current_score, final_score)
- Instructional design elements (assignments, quizzes, modules, files)
- Activity summaries (page views, participations)
- Calculates composite scores for prediction potential

**Usage:**
```bash
python section7_refactor.py --campus-ids "173,174,175,176" --max-courses 500 --workers 5
```

**Output:** `data/discovery/course_analysis_latest.csv`

---

### 2. `activity_analysis.py` - Student Activity Analysis

**Purpose:** Focuses on activity-based metrics from Canvas APIs for early warning without grades.

**What it extracts:**
- Tardiness breakdown (on_time, late, missing rates)
- Page views and participation levels (Canvas-computed 1-3)
- Recent student activity (last 7/30 days logins)
- Daily activity patterns
- Assignment-level statistics

**Usage:**
```bash
python activity_analysis.py --campus-ids "176" --max-courses 100 --workers 4
```

**Output:** `data/discovery/activity_analysis_latest.csv`

---

### 3. `deep_activity_analysis.py` - Statistical Deep Dive

**Purpose:** Performs comprehensive statistical analysis on activity data.

**What it does:**
- Correlation analysis between all metrics
- Campus comparisons with ANOVA tests
- Engagement pattern segmentation
- Clustering analysis (hierarchical)
- Risk identification
- Generates visualizations

**Usage:**
```bash
python deep_activity_analysis.py --input data/discovery/activity_analysis_latest.csv
```

**Output:** `data/discovery/deep_analysis/` (multiple files + visualizations)

---

### 4. `final_report_generator.py` - Combined Report

**Purpose:** Merges LMS design and activity analyses into a final comprehensive report.

**What it does:**
- Combines both CSV files
- Calculates combined score (50% design + 50% activity)
- Generates markdown report with all findings
- Creates visualizations
- Exports Top 50 courses ranking

**Usage:**
```bash
python final_report_generator.py
```

**Output:** `data/discovery/final_report/informe_completo_analisis.md`

---

### 5. `analyze_courses.py` - Correlation & Ranking

**Purpose:** Statistical analysis and correlation study of course discovery results.

**What it does:**
- Correlation matrix between all metrics
- Feature importance for prediction potential
- Campus/career comparisons
- Exportable rankings (CSV)
- Visualizations (heatmaps, distributions)

**Usage:**
```bash
python analyze_courses.py --input data/discovery/course_analysis_latest.csv --top 50
```

**Output:** `data/discovery/analysis/` (correlation matrix, rankings, plots)

---

## Support Modules (Not Run Directly)

### `canvas_client.py`
Thread-safe API client with rate limiting. Used by other scripts.
- Monitors `X-Rate-Limit-Remaining` header
- Adaptive delays based on quota (50-700)
- Bookmark-based pagination
- Retry logic with exponential backoff

### `course_analysis.py`
Course-level metrics extraction and scoring.
- `CourseMetrics` dataclass
- `analyze_course()` function
- Prediction potential scoring logic

### `career_analysis.py`
Career (sub-account) level aggregation.
- Aggregates course metrics by career
- Career recommendation logic (PRIORITIZE/MONITOR/SKIP)

### `batch_scanner.py`
Multi-threaded batch processing with progress tracking.
- Conservative threading (max 5 workers)
- Automatic quota monitoring
- Intermediate results saving

### `url_parser.py`
URL parsing utilities for page views analysis.
- Extracts course_id from Canvas URLs
- Categorizes page views by type
- Filters and aggregates page view data

### `__init__.py`
Module initialization with all exports.

---

## Data Flow

```
section7_refactor.py    activity_analysis.py
        │                       │
        ▼                       ▼
course_analysis_       activity_analysis_
   latest.csv              latest.csv
        │                       │
        └───────────┬───────────┘
                    ▼
         final_report_generator.py
                    │
                    ▼
            final_report/
         informe_completo.md
          top_50_combined.csv
```

---

## Key Configuration

All scripts read from `.env`:
```
CANVAS_API_URL=https://uautonoma.test.instructure.com
CANVAS_API_TOKEN=your_token_here
```

**Thresholds (from CLAUDE.md):**
- Pass threshold: 57% (Chilean 4.0/7.0 scale)
- Minimum students for prediction: 15
- Minimum grade variance: 10% std dev
- Ideal failure rate: 15-85%

---

## Campus IDs Reference

| ID | Campus |
|----|--------|
| 173-175 | Temuco |
| 176 | Providencia |
| 228-231 | San Miguel |

---

*Last updated: December 2025*
