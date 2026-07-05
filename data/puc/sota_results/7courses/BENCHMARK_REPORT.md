# PUC SOTA Benchmark Report

Generated: 2026-02-11 04:29

Total experiments: 4473
- Phase 1: 4200 experiments
- Phase 2: 273 experiments

## Best Models by Classification Scheme

### binary_4.0

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | XGBoost | 0.2 | 6 | Yes | 0.863 | 0.195 | 0.223 | 0.929 |
| 2 | RandomForest_balanced_tuned | 0.2 | 8 | Yes | 0.863 | 0.366 | 0.379 | 0.920 |
| 3 | RandomForest_tuned | 0.2 | 8 | Yes | 0.863 | 0.366 | 0.379 | 0.920 |
| 4 | XGBoost_balanced_tuned | 0.2 | 4 | Yes | 0.860 | 0.244 | 0.276 | 0.932 |
| 5 | XGBoost_tuned | 0.2 | 4 | Yes | 0.860 | 0.244 | 0.276 | 0.932 |

### binary_5.0

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | VotingEnsemble | 0.05 | full | Yes | 0.734 | 0.463 | 0.485 | 0.691 |
| 2 | XGBoost | 0.1 | full | Yes | 0.731 | 0.438 | 0.465 | 0.698 |
| 3 | RandomForest | 0.05 | full | Yes | 0.730 | 0.419 | 0.446 | 0.689 |
| 4 | XGBoost | 0.05 | full | Yes | 0.729 | 0.478 | 0.498 | 0.696 |
| 5 | XGBoost_balanced | 0.05 | full | Yes | 0.728 | 0.562 | 0.563 | 0.686 |

### 3class

| Rank | Model | Pct | Week | Assess | ROC_AUC_OVR | Recall | Accuracy |
|------|-------|-----|------|--------|---------|--------|----------|
| 1 | RandomForest_balanced | 0.1 | full | Yes | 0.714 | nan | 0.686 |
| 2 | RandomForest_balanced | 0.05 | full | Yes | 0.711 | nan | 0.664 |
| 3 | LightGBM_balanced | 0.2 | full | Yes | 0.710 | nan | 0.673 |
| 4 | LightGBM_balanced | 0.05 | full | Yes | 0.706 | nan | 0.664 |
| 5 | LightGBM_balanced | 0.1 | full | Yes | 0.705 | nan | 0.662 |

### 4class

| Rank | Model | Pct | Week | Assess | ROC_AUC_OVR | Recall | Accuracy |
|------|-------|-----|------|--------|---------|--------|----------|
| 1 | RandomForest_balanced | 0.2 | full | No | 0.650 | nan | 0.420 |
| 2 | XGBoost | 0.1 | 8 | Yes | 0.648 | nan | 0.409 |
| 3 | XGBoost_balanced | 0.1 | 8 | Yes | 0.648 | nan | 0.409 |
| 4 | RandomForest_balanced | 0.05 | full | No | 0.648 | nan | 0.420 |
| 5 | RandomForest_balanced | 0.05 | full | Yes | 0.647 | nan | 0.434 |

### oviedo_at_risk

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | XGBoost | 0.1 | full | Yes | 0.977 | 0.417 | 0.455 | 0.984 |
| 2 | XGBoost | 0.05 | full | Yes | 0.976 | 0.417 | 0.455 | 0.984 |
| 3 | XGBoost | 0.15 | full | Yes | 0.976 | 0.417 | 0.455 | 0.984 |
| 4 | XGBoost | 0.2 | full | Yes | 0.972 | 0.417 | 0.446 | 0.982 |
| 5 | RandomForest_balanced | 0.05 | full | Yes | 0.964 | 0.083 | 0.102 | 0.980 |

### oviedo_pass_fail

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | VotingEnsemble | 0.05 | full | Yes | 0.734 | 0.463 | 0.485 | 0.691 |
| 2 | XGBoost | 0.1 | full | Yes | 0.731 | 0.438 | 0.465 | 0.698 |
| 3 | RandomForest | 0.05 | full | Yes | 0.730 | 0.419 | 0.446 | 0.689 |
| 4 | XGBoost | 0.05 | full | Yes | 0.729 | 0.478 | 0.498 | 0.696 |
| 5 | XGBoost_balanced | 0.05 | full | Yes | 0.728 | 0.562 | 0.563 | 0.686 |

### oviedo_excellent

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | RandomForest_balanced | 0.2 | full | Yes | 0.770 | 0.265 | 0.302 | 0.736 |
| 2 | RandomForest_balanced | 0.15 | full | Yes | 0.763 | 0.271 | 0.309 | 0.738 |
| 3 | LogisticRegression_balanced | 0.2 | full | No | 0.762 | 0.482 | 0.502 | 0.742 |
| 4 | VotingEnsemble | 0.05 | full | Yes | 0.760 | 0.361 | 0.402 | 0.760 |
| 5 | XGBoost | 0.1 | full | Yes | 0.759 | 0.386 | 0.425 | 0.764 |

## Threshold Optimization (Top Binary Models)

### XGBoost (pct=0.1, wk=full, oviedo_at_risk, ROC-AUC=0.977)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.25 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| max_f2 | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| max_f3 | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| youden_j | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| mcc | 0.74 | 0.333 | 1.000 | 0.500 | 0.385 | 0.986 | 0.573 |
| g_mean | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| max_accuracy | 0.74 | 0.333 | 1.000 | 0.500 | 0.385 | 0.986 | 0.573 |
| cost_3x | 0.25 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| cost_5x | 0.25 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| recall_80 | 0.05 | 0.583 | 0.350 | 0.438 | 0.515 | 0.968 | 0.437 |
| recall_85 | 0.05 | 0.583 | 0.350 | 0.438 | 0.515 | 0.968 | 0.437 |
| recall_90 | 0.05 | 0.583 | 0.350 | 0.438 | 0.515 | 0.968 | 0.437 |

### XGBoost (pct=0.05, wk=full, oviedo_at_risk, ROC-AUC=0.976)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.62 | 0.417 | 0.833 | 0.556 | 0.463 | 0.986 | 0.583 |
| max_f2 | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| max_f3 | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| youden_j | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| mcc | 0.62 | 0.417 | 0.833 | 0.556 | 0.463 | 0.986 | 0.583 |
| g_mean | 0.06 | 0.583 | 0.389 | 0.467 | 0.530 | 0.971 | 0.462 |
| max_accuracy | 0.62 | 0.417 | 0.833 | 0.556 | 0.463 | 0.986 | 0.583 |
| cost_3x | 0.24 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| cost_5x | 0.24 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| recall_80 | 0.05 | 0.583 | 0.318 | 0.412 | 0.500 | 0.964 | 0.414 |
| recall_85 | 0.05 | 0.583 | 0.318 | 0.412 | 0.500 | 0.964 | 0.414 |
| recall_90 | 0.05 | 0.583 | 0.318 | 0.412 | 0.500 | 0.964 | 0.414 |

### XGBoost (pct=0.15, wk=full, oviedo_at_risk, ROC-AUC=0.976)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.32 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| max_f2 | 0.06 | 0.583 | 0.368 | 0.452 | 0.522 | 0.970 | 0.449 |
| max_f3 | 0.06 | 0.583 | 0.368 | 0.452 | 0.522 | 0.970 | 0.449 |
| youden_j | 0.06 | 0.583 | 0.368 | 0.452 | 0.522 | 0.970 | 0.449 |
| mcc | 0.32 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| g_mean | 0.06 | 0.583 | 0.368 | 0.452 | 0.522 | 0.970 | 0.449 |
| max_accuracy | 0.43 | 0.417 | 0.714 | 0.526 | 0.455 | 0.984 | 0.538 |
| cost_3x | 0.32 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| cost_5x | 0.32 | 0.500 | 0.600 | 0.545 | 0.517 | 0.982 | 0.539 |
| recall_80 | 0.05 | 0.583 | 0.350 | 0.438 | 0.515 | 0.968 | 0.437 |
| recall_85 | 0.05 | 0.583 | 0.350 | 0.438 | 0.515 | 0.968 | 0.437 |
| recall_90 | 0.05 | 0.583 | 0.350 | 0.438 | 0.515 | 0.968 | 0.437 |

## Per-Course Recall (Best binary_4.0 Model)

Model: **XGBoost** (pct=0.2, wk=6, ROC-AUC=0.863)

| Course ID | Students | Failures | Recall | Precision |
|-----------|----------|----------|--------|-----------|
| 54503 | 51 | 3 | 0.333 | 1.000 |
| 54529 | 131 | 8 | 0.250 | 0.400 |
| 54570 | 22 | 5 | 0.000 | 0.000 |
| 54581 | 16 | 2 | 0.000 | 0.000 |
| 55010 | 117 | 6 | 0.000 | 0.000 |
| 55183 | 99 | 2 | 0.000 | 0.000 |
| 55410 | 124 | 15 | 0.333 | 0.556 |

## Feature Importance Consensus

Top features across best 5 models (by frequency in top-10):

| Feature | Appearances (out of 5) |
|---------|----------------------|
| n_sessions | 5 |
| active_weeks | 5 |
| quizzes_coverage | 5 |
| overall_proactivity_znorm | 5 |
| quizzes_views | 5 |
| quizzes_views_znorm | 5 |
| quizzes_proact_std_pct_znorm | 4 |
| modules_unique_znorm | 4 |
| daily_consistency_znorm | 4 |
| early_late_ratio_znorm | 3 |
| last_active_week | 2 |
| files_proact_std_pct_znorm | 1 |
| total_time_min_znorm | 1 |
| modules_proact_mean_pct_znorm | 1 |

## Phase 1 vs Phase 2 Comparison (binary_4.0)

**Phase 1**: XGBoost — ROC-AUC=0.863, Recall=0.195, Precision=0.533, F2=0.223
**Phase 2**: RandomForest_balanced_tuned — ROC-AUC=0.863, Recall=0.366, Precision=0.441, F2=0.379

## Performance by Cutoff Week

| Week | Mean ROC-AUC | Max ROC-AUC | Mean Recall | Max Recall |
|------|-------------|-------------|-------------|------------|
| 2 | 0.673 | 0.866 | 0.245 | 0.660 |
| 4 | 0.697 | 0.866 | 0.253 | 0.723 |
| 6 | 0.696 | 0.872 | 0.271 | 0.753 |
| 8 | 0.717 | 0.897 | 0.280 | 0.788 |
| full | 0.738 | 0.977 | 0.325 | 0.704 |

## Classification Scheme Comparison

| Scheme | N_experiments | Best Metric | Best Model |
|--------|--------------|-------------|------------|
| binary_4.0 | 873 | ROC-AUC=0.863 | XGBoost |
| binary_5.0 | 600 | ROC-AUC=0.734 | VotingEnsemble |
| 3class | 600 | ROC-AUC-OVR=0.714 | RandomForest_balanced |
| 4class | 600 | ROC-AUC-OVR=0.650 | RandomForest_balanced |
| oviedo_at_risk | 600 | ROC-AUC=0.977 | XGBoost |
| oviedo_pass_fail | 600 | ROC-AUC=0.734 | VotingEnsemble |
| oviedo_excellent | 600 | ROC-AUC=0.770 | RandomForest_balanced |

## Deployment Recommendations

**Aggressive** (maximize recall): SVM_balanced at t=0.08 — Recall=100.0%, Precision=36.6%

**Recall-focused** (max F2): LightGBM_balanced at t=0.05 — Recall=95.1%, Precision=43.5%, F2=0.768

**Balanced** (Youden's J): RandomForest at t=0.09 — Recall=83.3%, Precision=20.0%

**Conservative** (max MCC): RandomForest at t=0.29 — Recall=58.3%, Precision=87.5%
