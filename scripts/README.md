# Scripts Guide

## Overview

This directory contains all Python scripts for the Early Warning System pipeline.

**Total Scripts:** ~60
**Key Categories:** Extraction, Processing, Feature Engineering, Modeling, Analysis, Reporting

---

## Execution Order

### 1. Data Extraction

| Script | Purpose | Output |
|--------|---------|--------|
| `extract_page_views_async.py` | Extract clickstream from Canvas API | `data/page_views/raw/*.json` |
| `fetch_enrollment_grades.py` | Fetch student grades | `data/model_courses_enrollments.json` |
| `fetch_complete_students.py` | Get complete student list | `data/page_views/student_enrollments.csv` |

### 2. Data Processing

| Script | Purpose | Output |
|--------|---------|--------|
| `categorize_page_views.py` | Parse URLs, extract resource types | `data/page_views/categorized_page_views.parquet` |

### 3. Feature Engineering

Run in this order:

| Script | Features | Output |
|--------|----------|--------|
| `calculate_session_features.py` | 12 session metrics | `data/enriched_features/session_features.parquet` |
| `calculate_category_features.py` | 45 category metrics | `data/enriched_features/category_features.parquet` |
| `calculate_proactivity_features.py` | 60+ PCT rankings | `data/enriched_features/proactivity_features.parquet` |
| `calculate_pca_features.py` | 15 PCA components | `data/enriched_features/pca_features.parquet` |
| `calculate_weekly_features.py` | 20 temporal features | `data/enriched_features/weekly_features.parquet` |
| `calculate_time_features.py` | 12 time-of-day features | `data/enriched_features/time_features.parquet` |
| `calculate_course_relative_features.py` | 77 time-normalized features | `data/enriched_features/course_relative_features.parquet` |
| `normalize_features_per_course.py` | Merge & normalize all | `data/enriched_features/normalized_features.parquet` |

### 4. Feature Selection & Modeling

| Script | Purpose | Output |
|--------|---------|--------|
| `sota_feature_selection.py` | 5-stage SOTA selection | `data/feature_selection/optimal_features.json` |
| `train_early_warning_model.py` | Train XGBoost model | `data/models/early_warning_model.pkl` |

### 5. Analysis

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_shap_explanations.py` | SHAP feature importance | `data/report/visualizations/shap_*.png` |
| `analyze_course_time_ranges.py` | Course duration analysis | Analysis output |

### 6. Reporting

| Script | Purpose | Output |
|--------|---------|--------|
| `generate_technical_report.py` | Generate Spanish report | `data/report/*.md` |
| `regenerate_visualizations.py` | Update all charts | `data/report/visualizations/*.png` |

---

## Quick Start

```bash
# Full pipeline (after data extraction)
python scripts/categorize_page_views.py
python scripts/calculate_session_features.py
python scripts/calculate_category_features.py
python scripts/calculate_proactivity_features.py
python scripts/calculate_pca_features.py
python scripts/calculate_weekly_features.py
python scripts/calculate_course_relative_features.py
python scripts/normalize_features_per_course.py
python scripts/sota_feature_selection.py
python scripts/train_early_warning_model.py
```

---

## Environment

```bash
# Required packages
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
pip install python-dotenv requests aiohttp  # For API extraction
```

---

## Configuration

### Model Courses

All feature scripts use these 10 courses:

```python
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
```

### Session Threshold

```python
SESSION_GAP_MINUTES = 30  # Gap >= 30 minutes = new session
```

### Target Variable

```python
FAILURE_THRESHOLD = 57  # final_score < 57% = failed
```

---

## Proposed Future Structure

```
scripts/
├── 01_extraction/
│   ├── extract_page_views_async.py
│   └── fetch_enrollment_grades.py
├── 02_processing/
│   └── categorize_page_views.py
├── 03_features/
│   ├── calculate_session_features.py
│   ├── calculate_category_features.py
│   ├── calculate_proactivity_features.py
│   ├── calculate_pca_features.py
│   ├── calculate_weekly_features.py
│   ├── calculate_course_relative_features.py
│   └── normalize_features_per_course.py
├── 04_modeling/
│   ├── sota_feature_selection.py
│   └── train_early_warning_model.py
├── 05_analysis/
│   └── generate_shap_explanations.py
└── 06_reporting/
    ├── generate_technical_report.py
    └── regenerate_visualizations.py
```

---

*Last updated: 2026-01-03*
