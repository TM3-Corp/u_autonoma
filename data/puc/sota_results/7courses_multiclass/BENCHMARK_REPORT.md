# PUC SOTA Benchmark Report

Generated: 2026-06-29 17:14

Total experiments: 26
- Phase 1: 26 experiments

## Best Models by Classification Scheme

### binary_4.5

| Rank | Model | Pct | Week | Assess | ROC_AUC | Recall | F2 | Accuracy |
|------|-------|-----|------|--------|---------|--------|------|----------|
| 1 | GradientBoosting | 0.05 | 4 | No | 0.695 | 0.304 | 0.328 | 0.787 |
| 2 | XGBoost_balanced | 0.05 | 4 | No | 0.689 | 0.348 | 0.354 | 0.750 |
| 3 | RandomForest | 0.05 | 4 | No | 0.687 | 0.261 | 0.285 | 0.784 |
| 4 | XGBoost_balanced | 0.05 | 4 | Yes | 0.679 | 0.391 | 0.388 | 0.741 |
| 5 | GradientBoosting | 0.05 | 4 | Yes | 0.679 | 0.261 | 0.279 | 0.764 |

## Threshold Optimization (Top Models)

### GradientBoosting (pct=0.05, wk=4, binary_4.5, ROC-AUC=0.695)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| max_f2 | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| max_f3 | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| youden_j | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| mcc | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| g_mean | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| max_accuracy | 0.92 | 0.070 | 0.727 | 0.127 | 0.085 | 0.804 | 0.183 |
| cost_3x | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| cost_5x | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| recall_80 | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| recall_85 | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |
| recall_90 | 0.05 | 0.748 | 0.325 | 0.453 | 0.593 | 0.629 | 0.280 |

### XGBoost_balanced (pct=0.05, wk=4, binary_4.5, ROC-AUC=0.689)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| max_f2 | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| max_f3 | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| youden_j | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| mcc | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| g_mean | 0.09 | 0.722 | 0.295 | 0.419 | 0.560 | 0.589 | 0.224 |
| max_accuracy | 0.94 | 0.017 | 0.500 | 0.034 | 0.022 | 0.795 | 0.062 |
| cost_3x | 0.30 | 0.513 | 0.345 | 0.413 | 0.468 | 0.700 | 0.229 |
| cost_5x | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| recall_80 | 0.06 | 0.809 | 0.284 | 0.421 | 0.591 | 0.543 | 0.232 |
| recall_85 | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |
| recall_90 | 0.05 | 0.826 | 0.285 | 0.424 | 0.599 | 0.539 | 0.240 |

### RandomForest (pct=0.05, wk=4, binary_4.5, ROC-AUC=0.687)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.28 | 0.548 | 0.344 | 0.423 | 0.490 | 0.693 | 0.240 |
| max_f2 | 0.11 | 0.913 | 0.248 | 0.390 | 0.595 | 0.414 | 0.186 |
| max_f3 | 0.06 | 0.974 | 0.220 | 0.360 | 0.579 | 0.287 | 0.117 |
| youden_j | 0.28 | 0.548 | 0.344 | 0.423 | 0.490 | 0.693 | 0.240 |
| mcc | 0.36 | 0.435 | 0.403 | 0.418 | 0.428 | 0.752 | 0.261 |
| g_mean | 0.28 | 0.548 | 0.344 | 0.423 | 0.490 | 0.693 | 0.240 |
| max_accuracy | 0.67 | 0.043 | 1.000 | 0.083 | 0.054 | 0.804 | 0.187 |
| cost_3x | 0.36 | 0.435 | 0.403 | 0.418 | 0.428 | 0.752 | 0.261 |
| cost_5x | 0.17 | 0.783 | 0.273 | 0.404 | 0.570 | 0.527 | 0.200 |
| recall_80 | 0.15 | 0.817 | 0.264 | 0.399 | 0.576 | 0.495 | 0.192 |
| recall_85 | 0.13 | 0.852 | 0.254 | 0.391 | 0.579 | 0.455 | 0.179 |
| recall_90 | 0.11 | 0.913 | 0.248 | 0.390 | 0.595 | 0.414 | 0.186 |

### XGBoost_balanced (pct=0.05, wk=4, binary_4.5, ROC-AUC=0.679)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.07 | 0.809 | 0.292 | 0.429 | 0.597 | 0.557 | 0.245 |
| max_f2 | 0.07 | 0.809 | 0.292 | 0.429 | 0.597 | 0.557 | 0.245 |
| max_f3 | 0.06 | 0.817 | 0.283 | 0.421 | 0.593 | 0.537 | 0.232 |
| youden_j | 0.07 | 0.809 | 0.292 | 0.429 | 0.597 | 0.557 | 0.245 |
| mcc | 0.46 | 0.435 | 0.391 | 0.412 | 0.425 | 0.745 | 0.250 |
| g_mean | 0.07 | 0.809 | 0.292 | 0.429 | 0.597 | 0.557 | 0.245 |
| max_accuracy | 0.95 | 0.000 | 0.000 | 0.000 | 0.000 | 0.791 | -0.030 |
| cost_3x | 0.46 | 0.435 | 0.391 | 0.412 | 0.425 | 0.745 | 0.250 |
| cost_5x | 0.07 | 0.809 | 0.292 | 0.429 | 0.597 | 0.557 | 0.245 |
| recall_80 | 0.07 | 0.809 | 0.292 | 0.429 | 0.597 | 0.557 | 0.245 |
| recall_85 | 0.05 | 0.826 | 0.274 | 0.411 | 0.589 | 0.514 | 0.216 |
| recall_90 | 0.05 | 0.826 | 0.274 | 0.411 | 0.589 | 0.514 | 0.216 |

### GradientBoosting (pct=0.05, wk=4, binary_4.5, ROC-AUC=0.679)

| Criterion | Threshold | Recall | Precision | F1 | F2 | Accuracy | MCC |
|-----------|-----------|--------|-----------|-----|-----|----------|-----|
| max_f1 | 0.08 | 0.591 | 0.319 | 0.415 | 0.505 | 0.657 | 0.221 |
| max_f2 | 0.05 | 0.652 | 0.300 | 0.411 | 0.528 | 0.616 | 0.210 |
| max_f3 | 0.05 | 0.652 | 0.300 | 0.411 | 0.528 | 0.616 | 0.210 |
| youden_j | 0.08 | 0.591 | 0.319 | 0.415 | 0.505 | 0.657 | 0.221 |
| mcc | 0.39 | 0.339 | 0.406 | 0.370 | 0.351 | 0.762 | 0.226 |
| g_mean | 0.08 | 0.591 | 0.319 | 0.415 | 0.505 | 0.657 | 0.221 |
| max_accuracy | 0.93 | 0.070 | 0.615 | 0.125 | 0.085 | 0.800 | 0.156 |
| cost_3x | 0.09 | 0.574 | 0.324 | 0.414 | 0.497 | 0.666 | 0.221 |
| cost_5x | 0.05 | 0.652 | 0.300 | 0.411 | 0.528 | 0.616 | 0.210 |
| recall_80 | 0.05 | 0.652 | 0.300 | 0.411 | 0.528 | 0.616 | 0.210 |
| recall_85 | 0.05 | 0.652 | 0.300 | 0.411 | 0.528 | 0.616 | 0.210 |
| recall_90 | 0.05 | 0.652 | 0.300 | 0.411 | 0.528 | 0.616 | 0.210 |

## Per-Course Recall (Fail Class)


## Feature Importance Consensus

Top features across best 5 models (by frequency in top-10):

| Feature | Appearances (out of 5) |
|---------|----------------------|
| n_sessions_znorm | 5 |
| daily_consistency_znorm | 5 |
| early_late_ratio | 5 |
| morning_pct | 4 |
| bigram_discussions_to_discussions_znorm | 4 |
| modules_proact_mean_pct | 4 |
| dct_1 | 3 |
| morning_pct_znorm | 3 |
| modules_proact_top50_rate_znorm | 2 |
| modules_coverage | 2 |
| dct_2 | 1 |
| discussions_time_min_znorm | 1 |
| unique_transitions | 1 |
| resource_coverage_rate | 1 |
| day_entropy_znorm | 1 |

## Performance by Cutoff Week

| Week | Mean ROC-AUC | Max ROC-AUC | Mean Recall | Max Recall |
|------|-------------|-------------|-------------|------------|
| 4 | 0.642 | 0.695 | 0.241 | 0.530 |

## Classification Scheme Comparison

| Scheme | N_experiments | Best Metric | Best Model |
|--------|--------------|-------------|------------|
| binary_4.5 | 26 | ROC-AUC=0.695 | GradientBoosting |

## Deployment Recommendations

**Aggressive** (maximize recall): SVM_RBF at t=0.06 — Recall=100.0%, Precision=20.6%

**Recall-focused** (max F2): XGBoost_balanced at t=0.05 — Recall=82.6%, Precision=28.5%, F2=0.599

**Balanced** (Youden's J): GradientBoosting at t=0.05 — Recall=74.8%, Precision=32.5%

**Conservative** (max MCC): GradientBoosting at t=0.05 — Recall=74.8%, Precision=32.5%
