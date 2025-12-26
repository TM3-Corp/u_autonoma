# Course 86676 Discovery Summary

**Course:** FUND DE BUSINESS ANALYTICS-P01
**Career:** Ing. en Control de Gestión (Account 719)
**Term:** Segundo Semestre 2025
**Date:** December 2025

---

## Course Overview

| Metric | Value |
|--------|-------|
| **Course ID** | 86676 |
| **Total Students** | 40 |
| **Pass Rate** | 27.5% (11/40) |
| **Grade Variance** | 26.76 (std dev) |
| **Assignments** | 24 (6 graded, 18 tracking) |
| **Modules** | 15 |
| **Pages** | 109 |
| **Files** | 125 |
| **Discussion Topics** | 86 |

---

## Data Extracted

### Files in `exploratory/data/courses/course_86676/`

| File | Records | Description |
|------|---------|-------------|
| `enrollments.parquet` | 40 | Student grades (current_score, final_score) |
| `assignments.parquet` | 24 | Assignment metadata |
| `assignment_groups.parquet` | 2 | Grade weight categories |
| `submissions.parquet` | 984 | Per-student submissions |
| `student_summaries.parquet` | 41 | Activity metrics (page_views, participations) |
| `modules.parquet` | 15 | Course structure |
| `pages.parquet` | 109 | Content pages |
| `files.parquet` | 125 | Course materials |
| `discussion_topics.parquet` | 86 | Discussion forums |
| `page_views.parquet` | 1,877 | Clickstream data (all 40 students) |
| `student_consolidated.csv` | 40 | All features merged (74 columns) |

---

## Feature Correlation Analysis

### Top Predictors of `final_score`

| Feature | Correlation | Category |
|---------|-------------|----------|
| `graded_count` | **0.976** | Submission |
| `avg_score` | **0.945** | Submission |
| `tardiness_missing` | **-0.857** | Behavior |
| `on_time` | **0.838** | Behavior |
| `submitted_count` | **0.838** | Submission |
| `participations` | **0.816** | Activity |
| `page_views` (Canvas) | **0.470** | Activity |
| `pv_ctrl_grades` | **0.431** | Clickstream |
| `pv_ctrl_assignment_groups` | **0.375** | Clickstream |
| `pv_ctrl_assignments` | **0.252** | Clickstream |

### Weak/No Correlation

| Feature | Correlation | Note |
|---------|-------------|------|
| `pv_total_views` | 0.099 | Raw clickstream count not predictive |
| `pv_total_interaction_min` | -0.031 | Time on site not predictive |
| `pv_unique_controllers` | 0.223 | Weak |

---

## Statistical Significance (t-test: PASS vs FAIL)

| Feature | PASS (n=11) | FAIL (n=29) | p-value | Significance |
|---------|-------------|-------------|---------|--------------|
| `participations` | 5.27 | 1.55 | 0.0000 | *** |
| `tardiness_missing` | 0.27 | 3.07 | 0.0000 | *** |
| `on_time` | 5.64 | 2.83 | 0.0000 | *** |
| `avg_score` | 75.10 | 51.83 | 0.0000 | *** |
| `page_views` (Canvas) | 902 | 595 | 0.0618 | * |
| `pv_total_views` (clickstream) | 48 | 47 | 0.9058 | - |

**Significance:** *** p<0.01, ** p<0.05, * p<0.1

---

## Key Findings

### 1. Submission Behavior is the Strongest Predictor

The most predictive features are related to **what students submit**, not how much they browse:

- `tardiness_missing` (r = -0.86): Missing assignments strongly predict failure
- `on_time` (r = 0.84): Timely submissions predict success
- `participations` (r = 0.82): Active participation predicts success

### 2. Canvas Summary > Raw Clickstream

Canvas's aggregated `page_views` metric (r = 0.47) is more predictive than raw clickstream `pv_total_views` (r = 0.10). The Canvas summary likely filters out noise (API calls, background requests).

### 3. Interaction Time is NOT Predictive

`pv_total_interaction_min` shows essentially zero correlation (-0.03). This is likely due to:
- Outliers leaving browser tabs open
- Passive viewing vs active engagement
- Quality matters more than quantity

### 4. Grade-Checking Behavior is Predictive

Students who check their grades (`pv_ctrl_grades`, r = 0.43) tend to perform better. This suggests engaged students monitor their progress.

---

## Recommended Features for Prediction Model

### Tier 1 (Strongest - Use These)

| Feature | Source | Correlation |
|---------|--------|-------------|
| `tardiness_missing` | student_summaries | -0.857 |
| `on_time` | student_summaries | 0.838 |
| `participations` | student_summaries | 0.816 |
| `page_views` | student_summaries | 0.470 |

### Tier 2 (Good - Consider These)

| Feature | Source | Correlation |
|---------|--------|-------------|
| `pv_ctrl_grades` | page_views | 0.431 |
| `pv_ctrl_assignment_groups` | page_views | 0.375 |
| `pv_ctrl_assignments` | page_views | 0.252 |
| `pv_participated_count` | page_views | 0.246 |

### Tier 3 (Weak - Avoid These)

| Feature | Source | Correlation |
|---------|--------|-------------|
| `pv_total_views` | page_views | 0.099 |
| `pv_total_interaction_min` | page_views | -0.031 |
| `pv_unique_controllers` | page_views | 0.223 |

---

## Consolidated CSV Schema (74 columns)

### Identity & Grades
- `user_id`, `enrollment_state`, `pass_fail`
- `current_score`, `final_score`, `current_grade`, `final_grade`
- `unposted_current_score`, `unposted_final_score`

### Canvas Activity Summary
- `page_views`, `page_views_level`
- `participations`, `participations_level`
- `on_time`, `tardiness_late`, `tardiness_missing`, `floating`

### Submission Aggregates
- `total_submissions`, `submitted_count`, `graded_count`
- `avg_score`, `min_score`, `max_score`, `std_score`
- `late_submissions`, `missing_submissions`, `excused_submissions`
- `avg_seconds_late`, `total_attempts`

### Per-Assignment Scores
- `score_TAREA 02`, `score_TAREA 03`, `score_TAREA 04`
- `score_Evaluación Regular 02`, `score_Presentacion Oral`

### Clickstream Stats (pv_*)
- `pv_total_views`, `pv_total_interaction_sec`, `pv_total_interaction_min`
- `pv_avg_interaction_sec`, `pv_participated_count`
- `pv_first_activity`, `pv_last_activity`, `pv_unique_controllers`

### Clickstream by Controller (pv_ctrl_*)
- `pv_ctrl_assignments`, `pv_ctrl_submissions`, `pv_ctrl_grades`
- `pv_ctrl_modules`, `pv_ctrl_discussion_topics`, `pv_ctrl_files`
- ... (32 controller types total)

---

## Prediction Model Results

### Model Performance (Leave-One-Out Cross-Validation)

| Task | Model | Metric | Value |
|------|-------|--------|-------|
| **Classification** | Logistic Regression | Accuracy | **85.0%** |
| | | Baseline (always FAIL) | 72.5% |
| | | Improvement | +17.2% |
| **Regression** | Ridge | R² Score | **0.691** |
| | | MAE | 10.6 points |

### Confusion Matrix

```
                 Predicted
                 FAIL  PASS
  Actual FAIL      26     3    (90% recall)
  Actual PASS       3     8    (73% recall)
```

### Feature Importance (Random Forest)

| Feature | Importance | Coefficient |
|---------|------------|-------------|
| `tardiness_missing` | 0.332 | -12.23 |
| `on_time` | 0.322 | +2.45 |
| `participations` | 0.228 | +7.36 |
| `page_views` | 0.118 | +2.39 |

### Model Files

| File | Description |
|------|-------------|
| `prediction_model.py` | Training script |
| `prediction_model.pkl` | Saved model (in course directory) |

### Usage

```bash
# Train and evaluate
python exploratory/discovery/prediction_model.py --course-dir exploratory/data/courses/course_86676

# Save model
python exploratory/discovery/prediction_model.py --course-dir exploratory/data/courses/course_86676 --save-model
```

---

## Next Steps

1. **Test early warning** using only pre-exam activity data
2. **Extract more courses** to validate findings across different subjects
3. **Compare with CLAUDE.md baseline** (Control de Gestión courses)
4. **Deploy real-time prediction** for incoming student data

---

*Last updated: December 2025*
