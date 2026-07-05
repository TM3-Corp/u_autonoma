> ⚠️ **SUPERSEDED / STALE NUMBERS — do not quote.** This document predates the nested-CV correction; its headline AUCs are optimistic (non-nested) and/or contaminated (UA KEEP-arm active-zeros) and/or label-leaky. Current defensible metrics: **`RESULTS_LEDGER.md`**; start at **`PROJECT_SSOT.md`**. Kept for history only.

# Early Warning Models - Complete Methodology Documentation

## Table of Contents
1. [Overview](#overview)
2. [Data Sources](#data-sources)
3. [Feature Engineering Pipeline](#feature-engineering-pipeline)
4. [Feature Selection](#feature-selection)
5. [Model Training](#model-training)
6. [Evaluation Methodology](#evaluation-methodology)
7. [Results Summary](#results-summary)
8. [File Reference](#file-reference)

---

## Overview

This package contains early warning models for predicting student failure (final_score < 57%) using LMS engagement data from Canvas.

### Two Model Pipelines

| Pipeline | Script | Features Created | After Selection | Use Case |
|----------|--------|------------------|-----------------|----------|
| **Standard** | `train_time_limited_model.py` | 200+ | ~58 | Production models |
| **Multi-Model** | `optimize_multi_model.py` | 72 | 72 (no selection) | Model comparison |

### Best Results Summary (Pipeline Comparison)

| Week | Standard (XGBoost) | Multi-Model Best | Diff | Best Algorithm |
|------|-------------------|------------------|------|----------------|
| 2    | 0.743             | 0.758            | +1.5 | RandomForest   |
| 4    | 0.742             | 0.774            | +3.2 | LogisticReg    |
| 6    | 0.745             | 0.851            | +10.6| VotingEnsemble |
| 8    | 0.828             | 0.854            | +2.6 | VotingEnsemble |
| Full | 0.903             | 0.881            | -2.2 | VotingEnsemble |

**Key Finding:** Multi-Model pipeline outperforms for weeks 2-8 due to better algorithm selection. Standard pipeline wins for Full week due to more aggressive feature selection (200+ → 58 features).

---

## Data Sources

### Primary Data Files

| File | Description | Records |
|------|-------------|---------|
| `data/page_views/categorized_page_views.parquet` | Clickstream data with resource categories | ~500K |
| `data/page_views/student_enrollments.csv` | Enrollments with final grades | ~400 |
| `data/assignment_analytics.json` | Assignment metadata (due dates, points) | ~200 |

### Target Variable

```python
failed = (final_score < 57).astype(int)  # Chilean 4.0/7.0 scale equivalent
```

### Courses Used (10 total)

```python
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
```

---

## Feature Engineering Pipeline

### Pipeline 1: Multi-Model (72 Features)

Features calculated inline in `optimize_multi_model.py`:

#### 1. Session Features (6 raw)

```python
SESSION_GAP_MINUTES = 30  # Gap >= 30 min = new session

features = {
    'total_views': len(group),
    'n_sessions': session_starts.sum(),
    'sessions_per_week': n_sessions / total_span,
    'views_per_session': len(group) / max(n_sessions, 1),
}
```

#### 2. Category Features (14 raw, 7 categories)

Categories: `files`, `discussions`, `quizzes`, `assignments`, `modules`, `grades`, `pages`

```python
for cat in categories:
    features[f'{cat}_views'] = len(cat_data)
    features[f'{cat}_pct'] = len(cat_data) / len(group) * 100
```

#### 3. Time Features (6 raw)

```python
features['pct_morning'] = sum((hours >= 6) & (hours < 12)) / total * 100
features['pct_afternoon'] = sum((hours >= 12) & (hours < 18)) / total * 100
features['pct_evening'] = sum((hours >= 18) & (hours < 24)) / total * 100
features['pct_weekend'] = sum(days >= 5) / total * 100
features['unique_hours'] = hours.nunique()
features['unique_days'] = days.nunique()
```

#### 4. Trajectory Features (5 raw)

```python
features['active_weeks'] = weeks.nunique()
features['first_active_week'] = weeks.min()
features['weekly_trend'] = np.polyfit(range(len(weekly_counts)), weekly_counts.values, 1)[0]
features['dct_0'] = dct_coeffs[0]  # DCT coefficient (level)
features['dct_1'] = dct_coeffs[1]  # DCT coefficient (trend)
```

#### 5. Pre-Assessment Features (5 raw)

```python
features['activity_72h_before'] = activity_72h  # Activity in 72h before assignments
features['preparation_intensity'] = activity_72h / len(group)
features['quiz_access_count'] = len(quiz_views)
features['unique_quizzes'] = quiz_views['resource_id'].nunique()
features['assgn_access_count'] = len(assgn_views)
features['unique_assignments'] = assgn_views['resource_id'].nunique()
features['assessment_diversity'] = unique_quizzes + unique_assignments
```

#### 6. Z-Normalized Features (~36 additional)

For each raw feature, a z-score normalized version per course:

```python
for col in numeric_cols:
    mean, std = cdf[col].mean(), cdf[col].std()
    cdf[f'{col}_znorm'] = (cdf[col] - mean) / std if std > 0 else 0
```

**Total: ~36 raw + ~36 z-normalized = 72 features**

---

### Pipeline 2: Standard (200+ Features)

Features loaded from pre-calculated parquet files:

#### Feature Files Loaded

| File | Features | Description |
|------|----------|-------------|
| `session_features.parquet` | 12 | Session patterns |
| `category_features.parquet` | 45 | Per-category engagement |
| `proactivity_features.parquet` | 60+ | PCT rankings |
| `pca_features.parquet` | 15 | PCA of resource access |
| `weekly_features.parquet` | 20 | Weekly temporal |
| `ngram_features.parquet` | 20 | Navigation sequences |
| `graph_features.parquet` | 6 | Resource graph analysis |
| `time_features.parquet` | 12 | Time-of-day patterns |
| `normalized_features.parquet` | 40+ | Z-normalized variants |

**Total: ~200+ raw features**

#### Key Feature Categories

**PCT (Proactivity) Features:**
- For each resource, rank students by first access time
- First student: PCT = 1.0, Last: PCT ≈ 0, Never: PCT = 0
- Features: `{type}_mean_pct`, `{type}_access_rate`, `{type}_top25_rate`

**PCA Features:**
- Create students × resources matrix of PCT values
- Apply PCA per resource type (files, pages, modules, discussions)
- Extract 2-3 principal components per type

**N-gram Features:**
- Track navigation sequences (e.g., files → home → discussions)
- Calculate transition probabilities, entropy, diversity

---

## Feature Selection

### Standard Pipeline Only

Feature selection is applied in `train_time_limited_model.py`:

#### Step 1: Importance Threshold

```python
FEATURE_IMPORTANCE_THRESHOLD = 0.005  # 0.5%

# Train XGBoost, keep features with importance >= threshold
model.fit(X, y)
selected = [f for f, imp in importances.items() if imp >= threshold]
```

#### Step 2: Correlation Filtering

```python
CORRELATION_THRESHOLD = 0.85

# Remove highly correlated features (keep first)
corr_matrix = X[feature_cols].corr().abs()
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
```

#### Result

| Stage | Features |
|-------|----------|
| Raw loaded | 200+ |
| After importance filter | ~80 |
| After correlation filter | ~58 |

### Multi-Model Pipeline

**No feature selection applied** - all 72 features used for fair model comparison.

---

## Model Training

### Cross-Validation Setup

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

- **5-fold stratified**: Preserves class distribution in each fold
- **random_state=42**: Reproducible splits
- **OOF predictions**: Out-of-fold predictions for threshold optimization

### Models Tested (Multi-Model Pipeline)

| Model | Configuration |
|-------|---------------|
| XGBoost | `n_estimators=100, max_depth=5, learning_rate=0.1` |
| XGBoost_balanced | + `scale_pos_weight=1.5` |
| RandomForest | `n_estimators=100, max_depth=8` |
| RandomForest_balanced | + `class_weight='balanced'` |
| GradientBoosting | `n_estimators=100, max_depth=5` |
| SVM_RBF | `kernel='rbf', C=1.0` + CalibratedClassifierCV |
| SVM_balanced | + `class_weight='balanced'` |
| MLP | `hidden_layer_sizes=(64, 32)` |
| MLP_deep | `hidden_layer_sizes=(128, 64, 32)` |
| LogisticRegression | `C=1.0, max_iter=1000` |
| LogisticRegression_balanced | + `class_weight='balanced'` |
| VotingEnsemble | XGB + RF + MLP (soft voting) |
| StackingEnsemble | XGB + RF + MLP → LogisticRegression |

### Standard Pipeline Model

```python
XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    n_estimators=100,
    subsample=0.8,
    eval_metric='logloss',
    verbosity=0,
    random_state=42
)
```

---

## Evaluation Methodology

### Metrics Calculated

| Metric | Formula | Purpose |
|--------|---------|---------|
| ROC-AUC | Area under ROC curve | Overall discriminative ability |
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correctness |
| Recall (Sensitivity) | TP/(TP+FN) | At-risk student detection |
| Specificity | TN/(TN+FP) | False alarm rate |
| Precision | TP/(TP+FP) | Alert reliability |
| G-Mean | sqrt(Recall × Specificity) | **Balance metric (sweet spot)** |
| F1 | 2×Precision×Recall/(Precision+Recall) | Precision-recall balance |
| F2 | 5×Precision×Recall/(4×Precision+Recall) | Recall-weighted |

### Threshold Optimization

OOF predictions are evaluated at thresholds 0.15 to 0.65 (step 0.01):

```python
thresholds = np.arange(0.15, 0.65, 0.01)

for t in thresholds:
    y_pred = (y_pred_proba >= t).astype(int)
    # Calculate all metrics
```

### Optimal Threshold Strategies

| Strategy | Use Case | Selection |
|----------|----------|-----------|
| `max_g_mean` | Balance sensitivity/specificity | argmax(G-Mean) |
| `max_f2` | Prioritize recall | argmax(F2) |
| `acc_at_recall_75` | Best accuracy with ≥75% recall | argmax(Accuracy) where Recall ≥ 0.75 |
| `acc_at_recall_80` | Best accuracy with ≥80% recall | argmax(Accuracy) where Recall ≥ 0.80 |

---

## Results Summary

### Complete Experiment Table (10 experiments × 2 pipelines)

Both pipelines now have consistent naming with 10 experiments each (5 weeks × 2 assessment variants).

#### Standard Pipeline (XGBoost with Feature Selection)

| Experiment | ROC-AUC | Features | Samples |
|------------|---------|----------|---------|
| week_2_with_assessment | 0.743 | 58 | 303 |
| week_2_without_assessment | 0.740 | 64 | 303 |
| week_4_with_assessment | 0.742 | 54 | 303 |
| week_4_without_assessment | 0.742 | 69 | 303 |
| week_6_with_assessment | 0.745 | 58 | 303 |
| week_6_without_assessment | 0.736 | 60 | 303 |
| week_8_with_assessment | 0.828 | 61 | 303 |
| week_8_without_assessment | 0.833 | 64 | 303 |
| full_with_assessment | 0.903 | 59 | 303 |
| full_without_assessment | 0.848 | 61 | 303 |

#### Multi-Model Pipeline (Best Model per Experiment)

| Experiment | ROC-AUC | Best Model |
|------------|---------|------------|
| week_2_p10_with_assessment | 0.758 | RandomForest |
| week_2_p10_without_assessment | 0.751 | XGBoost |
| week_4_p20_with_assessment | 0.774 | LogisticRegression |
| week_4_p20_without_assessment | 0.758 | LogisticRegression |
| week_6_p20_with_assessment | 0.851 | VotingEnsemble |
| week_6_p20_without_assessment | 0.812 | LogisticRegression_balanced |
| week_8_p20_with_assessment | 0.854 | VotingEnsemble |
| week_8_p20_without_assessment | 0.806 | VotingEnsemble |
| week_full_p20_with_assessment | 0.881 | VotingEnsemble |
| week_full_p20_without_assessment | 0.829 | LogisticRegression |

### WITH vs WITHOUT Assessment

**Assessment features** include: quiz, quizzes, assignment, grade, score, submission

```python
ASSESSMENT_PATTERNS = ['quiz', 'quizzes', 'assi', 'assignment', 'grade', 'grad', 'score', 'submission']
```

| Week | Standard WITH | Standard WITHOUT | Multi-Model WITH | Multi-Model WITHOUT |
|------|--------------|------------------|------------------|---------------------|
| 2    | 0.743        | 0.740            | 0.758            | 0.751               |
| 4    | 0.742        | 0.742            | 0.774            | 0.758               |
| 6    | 0.745        | 0.736            | 0.851            | 0.812               |
| 8    | 0.828        | 0.833            | 0.854            | 0.806               |
| Full | 0.903        | 0.848            | 0.881            | 0.829               |

**Key Findings:**
1. Assessment features help most at Week 6 (Multi-Model: +3.9 points)
2. Multi-Model pipeline benefits more from assessment features across all weeks
3. Week 8 shows unusual pattern where Standard WITHOUT beats WITH (likely due to feature selection)

---

## File Reference

### Scripts

| File | Purpose |
|------|---------|
| `train_time_limited_model.py` | Standard pipeline (200+ → 58 features) |
| `optimize_multi_model.py` | Multi-model comparison (72 features, 13 models) |
| `optimize_all_thresholds.py` | Threshold optimization across weeks |
| `train_optimal_early_model.py` | WITH vs WITHOUT assessment comparison |

### Results Folders

```
ml_team_share/
├── week_2/
│   ├── with_assessment/
│   │   ├── model_summary.json      # Metrics + feature importance
│   │   ├── top_features.txt        # Human-readable ranking
│   │   ├── threshold_optimization.json
│   │   └── threshold_recommendations.txt
│   └── without_assessment/
│       └── ...
├── week_4/
├── week_6/
├── week_8/
├── full/
└── early_warning_models.zip        # Complete model artifacts
```

### Result Files

| File | Content |
|------|---------|
| `model_summary.json` | ROC-AUC, accuracy, precision, recall, F1, top features |
| `top_features.txt` | Feature ranking with importance scores |
| `threshold_optimization.json` | Metrics at all thresholds |
| `threshold_recommendations.txt` | Recommended thresholds per strategy |

---

## Reproducibility

### Requirements

```bash
pip install pandas numpy scikit-learn xgboost scipy
```

### Run Multi-Model Experiments

```bash
python scripts/optimize_multi_model.py
# Output: data/analysis/multi_model_optimization_results.json
```

### Run Standard Pipeline

```bash
python scripts/train_time_limited_model.py --cutoff 2 --with-assessment
# Output: data/analysis/time_cutoff_results.json
```

### Key Parameters

```python
# Reproducibility
random_state = 42
np.random.seed(42)

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Session definition
SESSION_GAP_MINUTES = 30

# Course start definition
course_start = df['created_at'].quantile(0.10)  # P10% or P20%
```

---

*Last updated: January 2026*
*Version: v2 (Multi-Model + Standard pipelines documented)*
