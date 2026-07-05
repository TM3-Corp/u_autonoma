# Data Science Pipeline Overview

## Early Warning System for Student Failure Prediction

**Universidad Autónoma de Chile - Canvas LMS Analytics**

---

## End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA EXTRACTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  Canvas LMS API                                                             │
│      │                                                                      │
│      ├── /api/v1/users/{id}/page_views    → Raw clickstream data           │
│      ├── /api/v1/courses/{id}/enrollments → Student grades & enrollment    │
│      └── /api/v1/courses/{id}/assignments → Course structure               │
│                                                                             │
│  Output: data/page_views/raw/*.json, data/enrollments.json                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA PROCESSING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Categorize Page Views                                                   │
│     - Parse URLs to extract resource_type, resource_id, course_id          │
│     - Categories: files, discussions, quizzes, assignments, pages,         │
│                   modules, grades, announcements, home                      │
│                                                                             │
│  2. Session Detection                                                       │
│     - Gap threshold: 30 minutes                                             │
│     - Consecutive views with gap < 30 min = same session                   │
│                                                                             │
│  Output: data/page_views/categorized_page_views.parquet                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE ENGINEERING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  6 Feature Scripts → 280 Total Features                                     │
│                                                                             │
│  ┌────────────────────────┬────────────────────────────────────────────┐  │
│  │ Script                 │ Output Features                            │  │
│  ├────────────────────────┼────────────────────────────────────────────┤  │
│  │ session_features.py    │ 12 session metrics                         │  │
│  │ category_features.py   │ 45 category engagement metrics             │  │
│  │ proactivity_features.py│ 60+ PCT rankings & histogram features      │  │
│  │ pca_features.py        │ 15 PCA components (learning materials)     │  │
│  │ weekly_features.py     │ 20 temporal patterns                       │  │
│  │ course_relative.py     │ 77 time-normalized features (★NEW)         │  │
│  └────────────────────────┴────────────────────────────────────────────┘  │
│                                                                             │
│  Output: data/enriched_features/*.parquet                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          FEATURE NORMALIZATION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  Per-Course Z-Score Normalization                                           │
│  - Raw counts/times: z-scored within each course                           │
│  - Percentages/ratios: kept as-is (already scale-invariant)               │
│                                                                             │
│  Formula: z = (x - median) / IQR                                           │
│                                                                             │
│  Output: data/enriched_features/normalized_features.parquet                │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FEATURE SELECTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│  SOTA 5-Stage Pipeline:                                                     │
│                                                                             │
│  Stage 1: Filter Methods                                                    │
│     - Variance threshold (remove near-constant)                            │
│     - Correlation filter (r > 0.95 → remove)                               │
│                                                                             │
│  Stage 2: Univariate Statistics                                            │
│     - Mutual Information                                                    │
│     - Point-biserial correlation                                            │
│     - Mann-Whitney U test                                                   │
│                                                                             │
│  Stage 3: Embedded Methods                                                  │
│     - LASSO (L1 regularization)                                            │
│     - ElasticNet                                                            │
│     - Random Forest importance                                              │
│     - XGBoost importance                                                    │
│                                                                             │
│  Stage 4: Wrapper Methods                                                   │
│     - Boruta (shadow feature comparison)                                   │
│     - RFECV (recursive elimination)                                        │
│                                                                             │
│  Stage 5: Stability Selection                                               │
│     - Bootstrap stability (>50% selection rate)                            │
│     - LOCO cross-validation stability                                       │
│                                                                             │
│  280 features → 33 optimal features                                         │
│  Output: data/feature_selection/optimal_features.json                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             MODEL TRAINING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Target Variable: Binary (failed = final_score < 57%)                       │
│                                                                             │
│  Model: XGBoost Classifier                                                  │
│     - max_depth: 4                                                          │
│     - n_estimators: 100                                                     │
│     - scale_pos_weight: auto (class imbalance)                             │
│                                                                             │
│  Validation Strategy: LOCO (Leave-One-Course-Out)                          │
│     - Train on 9 courses, test on 1                                        │
│     - Repeat for all 10 courses                                            │
│     - Report mean ± std of AUC across folds                                │
│                                                                             │
│  Why LOCO?                                                                  │
│     - Tests generalization to unseen courses                               │
│     - More realistic than random splits                                    │
│     - Accounts for course-level clustering                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FINAL RESULTS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  Optimal Model Performance:                                                 │
│                                                                             │
│  ┌──────────────────────┬───────────┬───────────┬─────────────┐           │
│  │ Metric               │ CV (5-fold)│ LOCO      │ Features    │           │
│  ├──────────────────────┼───────────┼───────────┼─────────────┤           │
│  │ Early Warning Model  │ 0.8605    │ 0.7454    │ 40          │           │
│  │ NEW Optimal Model    │ 0.8418    │ 0.7708    │ 33          │           │
│  └──────────────────────┴───────────┴───────────┴─────────────┘           │
│                                                                             │
│  Key Finding: Course-relative features improve cross-course                 │
│  generalization by +3.4% LOCO AUC                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Script Execution Order

### 1. Data Extraction (Run Once)

```bash
# Extract page views for all students in model courses
python scripts/extract_page_views_async.py

# Fetch enrollment data with grades
python scripts/fetch_enrollment_grades.py
```

### 2. Data Processing

```bash
# Categorize page views by resource type
python scripts/categorize_page_views.py
```

### 3. Feature Engineering (Run in Order)

```bash
# Session-based features
python scripts/calculate_session_features.py

# Category engagement features
python scripts/calculate_category_features.py

# Proactivity (PCT ranking) features
python scripts/calculate_proactivity_features.py

# PCA features from learning material engagement
python scripts/calculate_pca_features.py

# Weekly temporal features
python scripts/calculate_weekly_features.py

# Course-relative time-normalized features (NEW)
python scripts/calculate_course_relative_features.py
```

### 4. Feature Normalization

```bash
# Merge all features and apply per-course z-score normalization
python scripts/normalize_features_per_course.py
```

### 5. Feature Selection & Model Training

```bash
# Run SOTA feature selection pipeline
python scripts/sota_feature_selection.py

# Train early warning model with optimal features
python scripts/train_early_warning_model.py
```

---

## Data Files

### Raw Data
| File | Description |
|------|-------------|
| `data/page_views/raw/*.json` | Raw page view JSON from Canvas API |
| `data/enrollments.json` | Student enrollment records |
| `data/model_courses_enrollments.json` | Enrollments with grades for 10 model courses |

### Processed Data
| File | Description |
|------|-------------|
| `data/page_views/categorized_page_views.parquet` | Page views with resource categories |
| `data/page_views/student_enrollments.csv` | Student-course enrollment mapping |

### Feature Files
| File | Description |
|------|-------------|
| `data/enriched_features/session_features.parquet` | Session metrics |
| `data/enriched_features/category_features.parquet` | Category engagement |
| `data/enriched_features/proactivity_features.parquet` | PCT rankings |
| `data/enriched_features/pca_features.parquet` | PCA components |
| `data/enriched_features/weekly_features.parquet` | Temporal patterns |
| `data/enriched_features/course_relative_features.parquet` | Time-normalized features |
| `data/enriched_features/normalized_features.parquet` | All features merged & normalized |

### Model Outputs
| File | Description |
|------|-------------|
| `data/feature_selection/optimal_features.json` | 33 selected features |
| `data/feature_selection/feature_rankings.parquet` | Full feature ranking analysis |
| `data/models/early_warning_model.pkl` | Trained XGBoost model |

---

## Model Courses

| Course ID | Name | Students | Term |
|-----------|------|----------|------|
| 79875 | INGLÉS II-E04 | 44 | 2025-1 |
| 79913 | INGLÉS APLICADO II-E01 | 26 | 2025-1 |
| 84936 | FUNDAMENTOS DE MICROECONOMÍA-P03 | 41 | 2025-2 |
| 84941 | FUNDAMENTOS DE MICROECONOMÍA-P01 | 36 | 2025-2 |
| 84944 | FUND. DE MACROECONOMÍA Y POL. ECON-P04 | 50 | 2025-2 |
| 86020 | TALL DE COMPETENCIAS DIGITALES-P03 | 30 | 2025-2 |
| 86676 | TALLER PENSAMIENTO ANALÍTICO-P01 | 40 | 2025-2 |
| 88381 | DESARROLLO CARRERA PROFES II-E01 | 34 | 2025-2 |
| 89099 | ADMINISTRACIÓN Y LIDERAZGO-E01 | 30 | 2025-2 |
| 89390 | DESARROLLO CARRERA PROFES II-E03 | 32 | 2025-2 |

**Total: 363 students across 10 courses**

---

## Key Concepts

### Session Definition
A **session** is a continuous period of LMS activity. Sessions are separated by gaps of 30+ minutes.

### PCT (Percentile) Ranking
For each resource, students are ranked by when they first accessed it:
- First student to access → PCT = 1.0
- Last student to access → PCT ≈ 0
- Never accessed → PCT = 0

This captures **proactivity** - being among the first to engage.

### Course-Relative Time Normalization
All temporal features are expressed as 0-100% of the course's actual duration (from first to last student interaction), enabling fair comparison across courses with different lengths.

### LOCO (Leave-One-Course-Out) Cross-Validation
The gold standard for testing generalization:
- Train on N-1 courses
- Test on the held-out course
- Repeat for all N courses

This tests whether the model can predict failure in courses it has never seen.

---

*Last updated: 2026-01-03*
