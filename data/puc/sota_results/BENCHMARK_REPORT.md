# PUC SOTA Benchmark Report

Generated: 2026-02-10 20:12

Total experiments: 4446
- Phase 1: 4200 experiments
- Phase 2: 246 experiments

## Best Models by Classification Scheme

### binary_4.0

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | LightGBM_tuned | 0.15 | 6 | Yes | 0.837 | 0.145 | 0.169 | 0.935 |
| 2 | XGBoost | 0.15 | 4 | No | 0.836 | 0.073 | 0.088 | 0.935 |
| 3 | XGBoost_balanced_tuned | 0.2 | 2 | No | 0.833 | 0.364 | 0.339 | 0.893 |
| 4 | XGBoost_balanced | 0.15 | 6 | Yes | 0.831 | 0.127 | 0.149 | 0.933 |
| 5 | XGBoost_balanced | 0.15 | 4 | No | 0.830 | 0.109 | 0.128 | 0.932 |

### binary_5.0

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | LightGBM | 0.1 | full | No | 0.697 | 0.457 | 0.472 | 0.717 |
| 2 | SVM_balanced | 0.15 | full | No | 0.697 | 0.137 | 0.160 | 0.698 |
| 3 | RandomForest | 0.05 | full | No | 0.695 | 0.312 | 0.342 | 0.711 |
| 4 | LightGBM_balanced | 0.1 | full | No | 0.692 | 0.488 | 0.489 | 0.690 |
| 5 | RandomForest_balanced | 0.05 | full | No | 0.692 | 0.340 | 0.363 | 0.697 |

### 3class

| Rank | Model | Pct | Week | Assess | ROC_AUC_OVR | Recall | Accuracy |
|------|-------|-----|------|--------|---------|--------|----------|
| 1 | LightGBM_balanced | 0.2 | 8 | No | 0.681 | nan | 0.643 |
| 2 | XGBoost_balanced | 0.2 | 8 | No | 0.674 | nan | 0.666 |
| 3 | XGBoost | 0.2 | 8 | No | 0.674 | nan | 0.666 |
| 4 | RandomForest_balanced | 0.2 | 8 | No | 0.673 | nan | 0.673 |
| 5 | RandomForest_balanced | 0.2 | 6 | Yes | 0.668 | nan | 0.672 |

### 4class

| Rank | Model | Pct | Week | Assess | ROC_AUC_OVR | Recall | Accuracy |
|------|-------|-----|------|--------|---------|--------|----------|
| 1 | LightGBM_balanced | 0.05 | full | Yes | 0.651 | nan | 0.449 |
| 2 | LightGBM_balanced | 0.15 | full | Yes | 0.647 | nan | 0.448 |
| 3 | XGBoost | 0.15 | full | Yes | 0.646 | nan | 0.453 |
| 4 | XGBoost_balanced | 0.15 | full | Yes | 0.646 | nan | 0.453 |
| 5 | RandomForest_balanced | 0.05 | full | Yes | 0.645 | nan | 0.442 |

### oviedo_at_risk

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | XGBoost | 0.05 | full | No | 0.949 | 0.267 | 0.308 | 0.986 |
| 2 | XGBoost | 0.1 | full | No | 0.942 | 0.267 | 0.308 | 0.986 |
| 3 | XGBoost | 0.2 | full | No | 0.936 | 0.200 | 0.231 | 0.983 |
| 4 | XGBoost | 0.15 | full | No | 0.935 | 0.200 | 0.231 | 0.983 |
| 5 | XGBoost_balanced | 0.05 | full | No | 0.934 | 0.333 | 0.373 | 0.986 |

### oviedo_pass_fail

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | LightGBM | 0.1 | full | No | 0.697 | 0.457 | 0.472 | 0.717 |
| 2 | SVM_balanced | 0.15 | full | No | 0.697 | 0.137 | 0.160 | 0.698 |
| 3 | RandomForest | 0.05 | full | No | 0.695 | 0.312 | 0.342 | 0.711 |
| 4 | LightGBM_balanced | 0.1 | full | No | 0.692 | 0.488 | 0.489 | 0.690 |
| 5 | RandomForest_balanced | 0.05 | full | No | 0.692 | 0.340 | 0.363 | 0.697 |

### oviedo_excellent

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | LightGBM | 0.05 | full | No | 0.702 | 0.414 | 0.434 | 0.713 |
| 2 | XGBoost | 0.05 | full | No | 0.696 | 0.332 | 0.361 | 0.716 |
| 3 | RandomForest | 0.05 | full | No | 0.694 | 0.238 | 0.273 | 0.730 |
| 4 | LightGBM_balanced | 0.05 | full | No | 0.694 | 0.496 | 0.493 | 0.683 |
| 5 | XGBoost | 0.1 | full | No | 0.693 | 0.348 | 0.377 | 0.721 |

## Threshold Optimization (Top Binary Models)

### XGBoost (pct=0.05, wk=full, oviedo_at_risk, ROC-AUC=0.949)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.18 | 0.467 | 0.636 | 0.538 | 0.493 | 0.986 | 0.538 |
| max_f2 | 0.06 | 0.667 | 0.370 | 0.476 | 0.575 | 0.974 | 0.485 |
| max_f3 | 0.06 | 0.667 | 0.370 | 0.476 | 0.575 | 0.974 | 0.485 |
| youden_j | 0.06 | 0.667 | 0.370 | 0.476 | 0.575 | 0.974 | 0.485 |
| mcc | 0.18 | 0.467 | 0.636 | 0.538 | 0.493 | 0.986 | 0.538 |
| g_mean | 0.06 | 0.667 | 0.370 | 0.476 | 0.575 | 0.974 | 0.485 |
| max_accuracy | 0.56 | 0.267 | 1.000 | 0.421 | 0.312 | 0.987 | 0.513 |
| cost_3x | 0.14 | 0.533 | 0.533 | 0.533 | 0.533 | 0.983 | 0.525 |
| cost_5x | 0.06 | 0.667 | 0.370 | 0.476 | 0.575 | 0.974 | 0.485 |
| recall_80 | 0.05 | 0.667 | 0.345 | 0.455 | 0.562 | 0.971 | 0.467 |
| recall_85 | 0.05 | 0.667 | 0.345 | 0.455 | 0.562 | 0.971 | 0.467 |
| recall_90 | 0.05 | 0.667 | 0.345 | 0.455 | 0.562 | 0.971 | 0.467 |

### XGBoost (pct=0.1, wk=full, oviedo_at_risk, ROC-AUC=0.942)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| max_f2 | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| max_f3 | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| youden_j | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| mcc | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| g_mean | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| max_accuracy | 0.19 | 0.400 | 0.667 | 0.500 | 0.435 | 0.986 | 0.510 |
| cost_3x | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| cost_5x | 0.09 | 0.667 | 0.435 | 0.526 | 0.602 | 0.979 | 0.528 |
| recall_80 | 0.05 | 0.667 | 0.323 | 0.435 | 0.549 | 0.969 | 0.450 |
| recall_85 | 0.05 | 0.667 | 0.323 | 0.435 | 0.549 | 0.969 | 0.450 |
| recall_90 | 0.05 | 0.667 | 0.323 | 0.435 | 0.549 | 0.969 | 0.450 |

### XGBoost (pct=0.2, wk=full, oviedo_at_risk, ROC-AUC=0.936)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| max_f2 | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| max_f3 | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| youden_j | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| mcc | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| g_mean | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| max_accuracy | 0.62 | 0.200 | 1.000 | 0.333 | 0.238 | 0.986 | 0.444 |
| cost_3x | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| cost_5x | 0.07 | 0.600 | 0.409 | 0.486 | 0.549 | 0.977 | 0.484 |
| recall_80 | 0.05 | 0.600 | 0.375 | 0.462 | 0.536 | 0.975 | 0.463 |
| recall_85 | 0.05 | 0.600 | 0.375 | 0.462 | 0.536 | 0.975 | 0.463 |
| recall_90 | 0.05 | 0.600 | 0.375 | 0.462 | 0.536 | 0.975 | 0.463 |

## Per-Course Recall (Best binary_4.0 Model)

Model: **LightGBM_tuned** (pct=0.15, wk=6, ROC-AUC=0.837)

| Course ID | Students | Failures | Recall | Precision |
|-----------|----------|----------|--------|-----------|
| 53190 | 13 | 3 | 0.000 | 0.000 |
| 53201 | 18 | 3 | 0.000 | 0.000 |
| 53205 | 11 | 1 | 0.000 | 0.000 |
| 53260 | 3 | 0 | N/A (0 fail) | 0.000 |
| 53319 | 17 | 1 | 1.000 | 1.000 |
| 53493 | 80 | 0 | N/A (0 fail) | 0.000 |
| 54360 | 1 | 0 | N/A (0 fail) | 0.000 |
| 54503 | 51 | 3 | 0.000 | 0.000 |
| 54529 | 131 | 8 | 0.125 | 1.000 |
| 54570 | 22 | 5 | 0.200 | 1.000 |
| 54581 | 16 | 2 | 0.000 | 0.000 |
| 54947 | 56 | 0 | N/A (0 fail) | 0.000 |
| 55010 | 117 | 6 | 0.000 | 0.000 |
| 55183 | 99 | 2 | 0.000 | 0.000 |
| 55410 | 124 | 15 | 0.333 | 0.385 |
| 56019 | 24 | 3 | 0.000 | 0.000 |
| 56867 | 31 | 0 | N/A (0 fail) | 0.000 |
| 57586 | 3 | 0 | N/A (0 fail) | 0.000 |
| 57587 | 1 | 0 | N/A (0 fail) | 0.000 |
| 59036 | 23 | 3 | 0.000 | 0.000 |

## Feature Importance Consensus

Top features across best 5 models (by frequency in top-10):

| Feature | Appearances (out of 5) |
|---------|----------------------|
| weeks_since_last | 5 |
| n_sessions | 5 |
| n_sessions_znorm | 5 |
| last_active_week_znorm | 5 |
| overall_proactivity_znorm | 5 |
| files_proact_std_pct_znorm | 5 |
| discussions_proact_std_pct_znorm | 3 |
| dct_0_znorm | 3 |
| dct_2 | 3 |
| navigation_time_min | 2 |
| discussions_proact_std_pct | 2 |
| early_late_ratio_znorm | 1 |
| dct_2_znorm | 1 |
| discussions_proact_top50_rate_znorm | 1 |
| resource_coverage_rate | 1 |

## Phase 1 vs Phase 2 Comparison (binary_4.0)

**Phase 1**: XGBoost — ROC-AUC=0.836, Recall=0.073, Precision=0.500, F2=0.088
**Phase 2**: LightGBM_tuned — ROC-AUC=0.837, Recall=0.145, Precision=0.500, F2=0.169

## Performance by Cutoff Week

| Week | Mean ROC-AUC | Max ROC-AUC | Mean Recall | Max Recall |
|------|-------------|-------------|-------------|------------|
| 2 | 0.637 | 0.825 | 0.196 | 0.629 |
| 4 | 0.647 | 0.836 | 0.198 | 0.672 |
| 6 | 0.654 | 0.833 | 0.216 | 0.648 |
| 8 | 0.667 | 0.840 | 0.225 | 0.688 |
| full | 0.698 | 0.949 | 0.270 | 0.719 |

## Classification Scheme Comparison

| Scheme | N_experiments | Best Metric | Best Model |
|--------|--------------|-------------|------------|
| binary_4.0 | 846 | ROC-AUC=0.837 | LightGBM_tuned |
| binary_5.0 | 600 | ROC-AUC=0.697 | LightGBM |
| 3class | 600 | ROC-AUC-OVR=0.681 | LightGBM_balanced |
| 4class | 600 | ROC-AUC-OVR=0.651 | LightGBM_balanced |
| oviedo_at_risk | 600 | ROC-AUC=0.949 | XGBoost |
| oviedo_pass_fail | 600 | ROC-AUC=0.697 | LightGBM |
| oviedo_excellent | 600 | ROC-AUC=0.702 | LightGBM |

## Deployment Recommendations

**Aggressive** (maximize recall): RandomForest at t=0.07 — Recall=100.0%, Precision=30.5%

**Recall-focused** (max F2): LogisticRegression_balanced at t=0.12 — Recall=94.1%, Precision=36.0%, F2=0.712

**Balanced** (Youden's J): RandomForest at t=0.07 — Recall=80.0%, Precision=14.5%

**Conservative** (max MCC): VotingEnsemble at t=0.37 — Recall=33.3%, Precision=100.0%
