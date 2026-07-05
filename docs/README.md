# Documentation Index

## Early Warning System - Canvas LMS Analytics

**Universidad Autónoma de Chile**

---

## Quick Links

| Document | Description |
|----------|-------------|
| [PIPELINE_OVERVIEW.md](./PIPELINE_OVERVIEW.md) | End-to-end data flow diagram |
| [FEATURE_CATALOG.md](./03_feature_engineering/FEATURE_CATALOG.md) | Complete feature reference (280 features) |
| [../CLAUDE.md](../CLAUDE.md) | Canvas API reference |
| [../scripts/README.md](../scripts/README.md) | Script execution guide |

---

## Documentation Structure

```
docs/
├── README.md                           # This file
├── PIPELINE_OVERVIEW.md                # End-to-end data flow
│
├── 01_data_extraction/                 # Canvas API & raw data
│   └── (see ../CLAUDE.md)
│
├── 02_data_processing/                 # Clickstream processing
│   ├── PAGE_VIEW_PROCESSING.md         # URL parsing
│   └── SESSION_DETECTION.md            # 30-min gap algorithm
│
├── 03_feature_engineering/             # Feature creation
│   └── FEATURE_CATALOG.md              # ★ Complete feature reference
│
├── 04_model_training/                  # Model documentation
│   ├── FEATURE_SELECTION.md            # SOTA selection pipeline
│   └── MODEL_EVALUATION.md             # LOCO CV methodology
│
└── 05_results/                         # Analysis results
    ├── MODEL_PERFORMANCE.md            # Final metrics
    ├── KEY_FINDINGS.md                 # Insights summary
    └── FEATURE_STABILITY_REPORT.md     # ★ Stability analysis
```

---

## Key Concepts

### Target Variable
- **Failed:** `final_score < 57%` (Chilean grading: < 4.0 on 1-7 scale)
- Binary classification: Predict which students will fail

### Validation Strategy
- **5-fold Stratified CV:** Within-sample performance
- **LOCO (Leave-One-Course-Out):** Cross-course generalization (gold standard)

### Model Performance
| Metric | Value |
|--------|-------|
| LOCO AUC | 0.7708 |
| CV AUC | 0.8418 |
| Optimal Features | 33 |

---

## Experiments

Chronological experiment log:

| Date | Experiment | Key Finding |
|------|------------|-------------|
| 2026-01-02 | Course-relative features | 77 new time-normalized features |
| 2026-01-03 | SOTA feature selection | 280 → 33 features, +3.4% LOCO |
| 2026-01-05 | Feature stability analysis | Only 7 stable features, 3% v4/SOTA overlap |

See `experiments/` directory for detailed experiment logs.

---

## Data Files

### Raw Data
- `data/page_views/raw/*.json` - Canvas API responses
- `data/model_courses_enrollments.json` - Student grades

### Processed Data
- `data/page_views/categorized_page_views.parquet` - Parsed clickstream
- `data/enriched_features/normalized_features.parquet` - All 280 features

### Model Outputs
- `data/feature_selection/optimal_features.json` - Selected 33 features
- `data/models/early_warning_model.pkl` - Trained XGBoost model

---

*Last updated: 2026-01-05*
