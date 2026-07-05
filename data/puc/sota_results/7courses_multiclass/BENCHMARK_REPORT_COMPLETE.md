# PUC SOTA Benchmark Report

Generated: 2026-02-13 18:51

Total experiments: 2640
- Phase 1: 1800 experiments
- Phase 2: 840 experiments

## Best Models by Classification Scheme

### binary_4.0

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | XGBoost_tuned | 0.2 | 4 | Yes | 0.872 | 0.341 | 0.370 | 0.932 |
| 2 | XGBoost_balanced_tuned | 0.2 | 4 | Yes | 0.872 | 0.341 | 0.370 | 0.932 |
| 3 | XGBoost | 0.2 | 6 | Yes | 0.863 | 0.195 | 0.223 | 0.929 |
| 4 | RandomForest_balanced_tuned | 0.2 | 8 | Yes | 0.863 | 0.415 | 0.417 | 0.916 |
| 5 | RandomForest_tuned | 0.2 | 8 | Yes | 0.863 | 0.415 | 0.417 | 0.916 |

### 4class

| Rank | Model | Pct | Week | Assess | Fail_ROC_AUC | ROC_AUC_OVR | Fail_Recall | Fail_F2 | Accuracy |
|------|-------|-----|------|--------|-------------|-------------|-------------|---------|----------|
| 1 | VotingEnsemble | 0.05 | full | Yes | 0.864 | 0.635 | 0.268 | 0.299 | 0.411 |
| 2 | XGBoost_tuned_smote | 0.05 | 8 | Yes | 0.855 | 0.638 | 0.415 | 0.429 | 0.420 |
| 3 | XGBoost_tuned_smote | 0.2 | 8 | Yes | 0.854 | 0.621 | 0.390 | 0.406 | 0.400 |
| 4 | XGBoost_balanced_tuned_smote | 0.2 | 8 | Yes | 0.854 | 0.621 | 0.390 | 0.406 | 0.400 |
| 5 | XGBoost_tuned_borderline_smote | 0.05 | 8 | Yes | 0.851 | 0.623 | 0.439 | 0.459 | 0.412 |

### 3class_marginal

| Rank | Model | Pct | Week | Assess | Fail_ROC_AUC | ROC_AUC_OVR | Fail_Recall | Fail_F2 | Accuracy |
|------|-------|-----|------|--------|-------------|-------------|-------------|---------|----------|
| 1 | XGBoost_tuned_borderline_smote | 0.2 | full | Yes | 0.880 | 0.759 | 0.463 | 0.463 | 0.748 |
| 2 | XGBoost_balanced_tuned_borderline_smote | 0.2 | full | Yes | 0.880 | 0.759 | 0.463 | 0.463 | 0.748 |
| 3 | XGBoost | 0.2 | 8 | Yes | 0.879 | 0.756 | 0.244 | 0.273 | 0.798 |
| 4 | XGBoost_balanced | 0.2 | 8 | Yes | 0.879 | 0.756 | 0.244 | 0.273 | 0.798 |
| 5 | XGBoost_tuned_smote | 0.05 | full | Yes | 0.870 | 0.740 | 0.390 | 0.392 | 0.732 |

## Threshold Optimization (Top Models)

### XGBoost_tuned_borderline_smote (pct=0.2, wk=full, 3class_marginal, Fail-ROC-AUC=0.880, phase=2)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.37 | 0.585 | 0.490 | 0.533 | 0.563 | 0.925 | 0.495 |
| max_f2 | 0.32 | 0.610 | 0.463 | 0.526 | 0.573 | 0.920 | 0.489 |
| max_f3 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| youden_j | 0.13 | 0.707 | 0.299 | 0.420 | 0.556 | 0.857 | 0.397 |
| mcc | 0.37 | 0.585 | 0.490 | 0.533 | 0.563 | 0.925 | 0.495 |
| g_mean | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| max_accuracy | 0.66 | 0.341 | 0.636 | 0.444 | 0.376 | 0.938 | 0.437 |
| cost_3x | 0.37 | 0.585 | 0.490 | 0.533 | 0.563 | 0.925 | 0.495 |
| cost_5x | 0.32 | 0.610 | 0.463 | 0.526 | 0.573 | 0.920 | 0.489 |
| recall_80 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| recall_85 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| recall_90 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |

### XGBoost_balanced_tuned_borderline_smote (pct=0.2, wk=full, 3class_marginal, Fail-ROC-AUC=0.880, phase=2)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.37 | 0.585 | 0.490 | 0.533 | 0.563 | 0.925 | 0.495 |
| max_f2 | 0.32 | 0.610 | 0.463 | 0.526 | 0.573 | 0.920 | 0.489 |
| max_f3 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| youden_j | 0.13 | 0.707 | 0.299 | 0.420 | 0.556 | 0.857 | 0.397 |
| mcc | 0.37 | 0.585 | 0.490 | 0.533 | 0.563 | 0.925 | 0.495 |
| g_mean | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| max_accuracy | 0.66 | 0.341 | 0.636 | 0.444 | 0.376 | 0.938 | 0.437 |
| cost_3x | 0.37 | 0.585 | 0.490 | 0.533 | 0.563 | 0.925 | 0.495 |
| cost_5x | 0.32 | 0.610 | 0.463 | 0.526 | 0.573 | 0.920 | 0.489 |
| recall_80 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| recall_85 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |
| recall_90 | 0.05 | 0.780 | 0.227 | 0.352 | 0.525 | 0.789 | 0.342 |

### XGBoost (pct=0.2, wk=8, 3class_marginal, Fail-ROC-AUC=0.879)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.17 | 0.488 | 0.455 | 0.471 | 0.481 | 0.920 | 0.427 |
| max_f2 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| max_f3 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| youden_j | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| mcc | 0.17 | 0.488 | 0.455 | 0.471 | 0.481 | 0.920 | 0.427 |
| g_mean | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| max_accuracy | 0.50 | 0.220 | 0.643 | 0.327 | 0.253 | 0.934 | 0.350 |
| cost_3x | 0.17 | 0.488 | 0.455 | 0.471 | 0.481 | 0.920 | 0.427 |
| cost_5x | 0.14 | 0.512 | 0.429 | 0.467 | 0.493 | 0.914 | 0.422 |
| recall_80 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| recall_85 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| recall_90 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |

### XGBoost_balanced (pct=0.2, wk=8, 3class_marginal, Fail-ROC-AUC=0.879)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.17 | 0.488 | 0.455 | 0.471 | 0.481 | 0.920 | 0.427 |
| max_f2 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| max_f3 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| youden_j | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| mcc | 0.17 | 0.488 | 0.455 | 0.471 | 0.481 | 0.920 | 0.427 |
| g_mean | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| max_accuracy | 0.50 | 0.220 | 0.643 | 0.327 | 0.253 | 0.934 | 0.350 |
| cost_3x | 0.17 | 0.488 | 0.455 | 0.471 | 0.481 | 0.920 | 0.427 |
| cost_5x | 0.14 | 0.512 | 0.429 | 0.467 | 0.493 | 0.914 | 0.422 |
| recall_80 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| recall_85 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |
| recall_90 | 0.05 | 0.659 | 0.303 | 0.415 | 0.534 | 0.864 | 0.384 |

### XGBoost_balanced_tuned (pct=0.2, wk=4, binary_4.0, ROC-AUC=0.872, phase=2)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.47 | 0.366 | 0.577 | 0.448 | 0.395 | 0.934 | 0.427 |
| max_f2 | 0.12 | 0.512 | 0.389 | 0.442 | 0.482 | 0.905 | 0.396 |
| max_f3 | 0.05 | 0.537 | 0.297 | 0.383 | 0.462 | 0.873 | 0.336 |
| youden_j | 0.12 | 0.512 | 0.389 | 0.442 | 0.482 | 0.905 | 0.396 |
| mcc | 0.47 | 0.366 | 0.577 | 0.448 | 0.395 | 0.934 | 0.427 |
| g_mean | 0.05 | 0.537 | 0.297 | 0.383 | 0.462 | 0.873 | 0.336 |
| max_accuracy | 0.47 | 0.366 | 0.577 | 0.448 | 0.395 | 0.934 | 0.427 |
| cost_3x | 0.47 | 0.366 | 0.577 | 0.448 | 0.395 | 0.934 | 0.427 |
| cost_5x | 0.12 | 0.512 | 0.389 | 0.442 | 0.482 | 0.905 | 0.396 |
| recall_80 | 0.05 | 0.537 | 0.297 | 0.383 | 0.462 | 0.873 | 0.336 |
| recall_85 | 0.05 | 0.537 | 0.297 | 0.383 | 0.462 | 0.873 | 0.336 |
| recall_90 | 0.05 | 0.537 | 0.297 | 0.383 | 0.462 | 0.873 | 0.336 |

## Per-Course Recall (Fail Class)

### binary_4.0
Model: **XGBoost_balanced_tuned** (pct=0.2, wk=4, ROC-AUC=0.872)

| Course ID | Students | Failures | Recall | Precision |
|-----------|----------|----------|--------|-----------|
| 54503 | 51 | 3 | 0.333 | 0.500 |
| 54529 | 131 | 8 | 0.125 | 1.000 |
| 54570 | 22 | 5 | 0.000 | 0.000 |
| 54581 | 16 | 2 | 0.500 | 1.000 |
| 55010 | 117 | 6 | 0.333 | 0.667 |
| 55183 | 99 | 2 | 0.000 | 0.000 |
| 55410 | 124 | 15 | 0.600 | 0.529 |

### 4class
Model: **VotingEnsemble** (pct=0.05, wk=full, ROC-AUC=0.864)

| Course ID | Students | Failures | Recall | Precision |
|-----------|----------|----------|--------|-----------|
| 54503 | 51 | 3 | 0.667 | 1.000 |
| 54529 | 131 | 8 | 0.125 | 0.250 |
| 54570 | 22 | 5 | 0.200 | 1.000 |
| 54581 | 16 | 2 | 0.500 | 1.000 |
| 55010 | 117 | 6 | 0.167 | 0.500 |
| 55183 | 99 | 2 | 0.000 | 0.000 |
| 55410 | 124 | 15 | 0.133 | 0.667 |

### 3class_marginal
Model: **XGBoost_tuned_borderline_smote** (pct=0.2, wk=full, ROC-AUC=0.880)

| Course ID | Students | Failures | Recall | Precision |
|-----------|----------|----------|--------|-----------|
| 54503 | 51 | 3 | 0.667 | 0.667 |
| 54529 | 131 | 8 | 0.375 | 0.250 |
| 54570 | 22 | 5 | 0.600 | 1.000 |
| 54581 | 16 | 2 | 0.500 | 0.333 |
| 55010 | 117 | 6 | 0.333 | 0.400 |
| 55183 | 99 | 2 | 0.000 | 0.000 |
| 55410 | 124 | 15 | 0.400 | 0.600 |


## Feature Importance Consensus

Top features across best 5 models (by frequency in top-10):

| Feature | Appearances (out of 5) |
|---------|----------------------|
| quizzes_views | 4 |
| modules_proact_std_pct_znorm | 3 |
| total_views_znorm | 3 |
| modules_coverage | 3 |
| daily_consistency_znorm | 2 |
| external_tools_time_min_znorm | 2 |
| modules_views | 2 |
| quizzes_proact_std_pct_znorm | 2 |
| content_vs_assessment_ratio_znorm | 2 |
| modules_unique | 2 |
| dct_3_znorm | 2 |
| discussions_proact_top50_rate_znorm | 2 |
| n_sessions_znorm | 2 |
| total_views | 2 |
| active_weeks_znorm | 2 |

## Phase 1 vs Phase 2 Comparison (binary_4.0)

**Phase 1**: XGBoost — ROC-AUC=0.863, Recall=0.195, Precision=0.533, F2=0.223
**Phase 2**: XGBoost_tuned — ROC-AUC=0.872, Recall=0.341, Precision=0.560, F2=0.370

## Performance by Cutoff Week

| Week | Mean ROC-AUC | Max ROC-AUC | Mean Recall | Max Recall |
|------|-------------|-------------|-------------|------------|
| 2 | 0.743 | 0.828 | 0.164 | 0.512 |
| 4 | 0.769 | 0.852 | 0.129 | 0.415 |
| 6 | 0.770 | 0.863 | 0.143 | 0.415 |
| 8 | 0.779 | 0.851 | 0.139 | 0.439 |
| full | 0.764 | 0.846 | 0.176 | 0.488 |

## Classification Scheme Comparison

| Scheme | N_experiments | Best Metric | Best Model |
|--------|--------------|-------------|------------|
| binary_4.0 | 873 | ROC-AUC=0.872 | XGBoost_balanced_tuned |
| 4class | 867 | ROC-AUC-OVR=0.662 | RandomForest_balanced_tuned_borderline_smote |
| 3class_marginal | 900 | ROC-AUC-OVR=0.771 | RandomForest_balanced_tuned_smote |

## Deployment Recommendations

**Aggressive** (maximize recall): SVM_RBF at t=0.05 — Recall=100.0%, Precision=9.2%

**Recall-focused** (max F2): RandomForest_balanced_tuned_smote at t=0.30 — Recall=70.7%, Precision=34.5%, F2=0.585

**Balanced** (Youden's J): VotingEnsemble at t=0.10 — Recall=78.0%, Precision=28.6%

**Conservative** (max MCC): RandomForest_balanced at t=0.30 — Recall=41.5%, Precision=70.8%
