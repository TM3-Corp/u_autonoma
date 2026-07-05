# Multi-Model Benchmark Results

Generated: 2026-02-03 02:16:38

## Overview

| Config | Samples | Features | Failure Rate | Best AUC | Best Model |
|--------|---------|----------|--------------|----------|------------|
| week_2_with_assessment | 303 | 396 | 39.6% | 0.740 | XGBoost_balanced |
| week_2_without_assessment | 303 | 300 | 39.6% | 0.730 | XGBoost |
| week_4_with_assessment | 343 | 396 | 38.5% | 0.742 | VotingEnsemble |
| week_4_without_assessment | 343 | 300 | 38.5% | 0.740 | XGBoost |
| week_6_with_assessment | 351 | 396 | 38.7% | 0.745 | VotingEnsemble |
| week_6_without_assessment | 351 | 300 | 38.7% | 0.720 | GradientBoosting |
| week_8_with_assessment | 356 | 396 | 38.8% | 0.821 | XGBoost |
| week_8_without_assessment | 356 | 300 | 38.8% | 0.792 | VotingEnsemble |
| week_full_with_assessment | 373 | 244 | 39.9% | 0.880 | VotingEnsemble |
| week_full_without_assessment | 373 | 187 | 39.9% | 0.842 | XGBoost_balanced |

## WITH vs WITHOUT Assessment Comparison

| Cutoff | WITH AUC | WITHOUT AUC | Delta | Best Model (WITH) |
|--------|----------|-------------|-------|-------------------|
| Week 2 | 0.740 | 0.730 | +0.010 | XGBoost_balanced |
| Week 4 | 0.742 | 0.740 | +0.002 | VotingEnsemble |
| Week 6 | 0.745 | 0.720 | +0.025 | VotingEnsemble |
| Week 8 | 0.821 | 0.792 | +0.029 | XGBoost |
| Week full | 0.880 | 0.842 | +0.038 | VotingEnsemble |

## Best Accuracy at 80% Recall

| Config | Model | Threshold | Accuracy | Recall | Specificity |
|--------|-------|-----------|----------|--------|-------------|
| week_2_with_assessment | RandomForest_balanced | 0.32 | 64.7% | 81.7% | 53.6% |
| week_2_without_assessment | RandomForest | 0.32 | 64.4% | 80.0% | 54.1% |
| week_4_with_assessment | VotingEnsemble | 0.24 | 66.8% | 80.3% | 58.3% |
| week_4_without_assessment | XGBoost | 0.17 | 66.8% | 83.3% | 56.4% |
| week_6_with_assessment | VotingEnsemble | 0.23 | 67.0% | 80.9% | 58.1% |
| week_6_without_assessment | RandomForest | 0.30 | 60.7% | 80.1% | 48.4% |
| week_8_with_assessment | GradientBoosting | 0.11 | 74.2% | 81.9% | 69.3% |
| week_8_without_assessment | VotingEnsemble | 0.24 | 72.2% | 80.4% | 67.0% |
| week_full_with_assessment | VotingEnsemble | 0.34 | 81.5% | 81.2% | 81.7% |
| week_full_without_assessment | MLP | 0.11 | 75.6% | 81.9% | 71.4% |

## Model Rankings (by G-Mean at Full Data WITH Assessment)

| Rank | Model | ROC-AUC | G-Mean | Recall | Accuracy |
|------|-------|---------|--------|--------|----------|
| 1 | VotingEnsemble | 0.880 | 0.815 | 83.2% | 81.2% |
| 2 | StackingEnsemble | 0.867 | 0.807 | 85.9% | 79.9% |
| 3 | XGBoost_balanced | 0.868 | 0.795 | 77.9% | 79.9% |
| 4 | MLP_deep | 0.859 | 0.793 | 80.5% | 79.1% |
| 5 | XGBoost | 0.873 | 0.792 | 82.6% | 78.6% |
| 6 | GradientBoosting | 0.862 | 0.782 | 80.5% | 77.7% |
| 7 | MLP | 0.857 | 0.781 | 69.8% | 80.4% |
| 8 | LogisticRegression | 0.833 | 0.775 | 76.5% | 77.7% |
| 9 | LogisticRegression_balanced | 0.833 | 0.775 | 76.5% | 77.7% |
| 10 | RandomForest_balanced | 0.848 | 0.772 | 71.8% | 78.6% |