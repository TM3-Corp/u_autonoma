# PUC Enhanced Early Warning System

Implementation of advanced early warning features based on:
- **Nguyen 2020**: IQR_ITT data-driven session thresholds
- **Oviedo (Beyond Time on Task)**: Workload dynamics and peak/slope features
- **UA Feature Engineering**: Course-relative time normalization

## Current Results

| Approach | Recall | F1 | Notes |
|----------|--------|-----|-------|
| Basic features | 0.40 | 0.39 | 78 features |
| Enhanced + threshold tuning | 0.63 | 0.46 | Better inactivity features |
| **Target** | **0.80+** | 0.60+ | With full features + IQR_ITT |

## Pipeline Components

### Feature Engineering Scripts

| Script | Features | Description |
|--------|----------|-------------|
| `implement_iqr_itt.py` | ~12 | Data-driven session thresholds (Nguyen 2020) |
| `calculate_inactivity_features.py` | ~20 | Enhanced inactivity metrics (gaps, CV, quartiles) |
| `calculate_workload_dynamics.py` | ~35 | Peak/slope/variability features (Oviedo paper) |
| `calculate_course_relative_features.py` | ~71 | Time normalization (0-100% of course) |
| `calculate_time_features.py` | ~11 | Time-of-day patterns |
| `feature_selection.py` | - | Correlation, MI, RFECV, SHAP-based selection |
| `train_comprehensive_model.py` | - | XGBoost, LightGBM, Ensemble training |

### Key Feature Categories

1. **IQR_ITT Session Features**
   - Adaptive session thresholds per course
   - Outlier detection using individual + resource + temporal context
   - Cleaned time-on-task estimates

2. **Enhanced Inactivity Features**
   - `max_inactivity_hours`, `min_inactivity_hours`
   - `inactivity_cv` (regularity indicator)
   - `longest_inactive_pct` (as % of course)
   - `consecutive_inactive_weeks`

3. **Workload Dynamics (Oviedo Paper)**
   - Peak features: `low/medium/high_intensity_peaks`, `max_peak_intensity`
   - Slope features: `max_positive_slope`, `max_negative_slope` (r=-0.70 with outcomes)
   - Variability: `activity_cv`, `consistency_score`
   - Distribution: `cramming_indicator`, `front_loaded/back_loaded`

4. **Course-Relative Time Normalization**
   - `first_access_pct`, `activity_span_pct`
   - `early_10/20/33_views_pct`
   - Per-resource timing histograms (5 bins × 6 resource types)
   - `engagement_curve_slope`, `engagement_curve_trend`

## Usage

### Run Full Pipeline
```bash
cd puc_analysis/scripts
python run_full_pipeline.py
```

### Run Individual Steps
```bash
# Feature engineering
python implement_iqr_itt.py
python calculate_inactivity_features.py
python calculate_workload_dynamics.py
python calculate_course_relative_features.py
python calculate_time_features.py

# Feature selection
python feature_selection.py

# Model training
python train_comprehensive_model.py
```

### Skip Steps
```bash
# Only train (use existing features)
python run_full_pipeline.py --only-train

# Skip feature engineering
python run_full_pipeline.py --skip-features

# Skip feature selection
python run_full_pipeline.py --skip-selection
```

## Directory Structure

```
puc_analysis/
├── scripts/
│   ├── config.py                           # Configuration
│   ├── run_full_pipeline.py               # Main runner
│   ├── implement_iqr_itt.py               # IQR_ITT sessions
│   ├── calculate_inactivity_features.py   # Inactivity metrics
│   ├── calculate_workload_dynamics.py     # Peak/slope features
│   ├── calculate_course_relative_features.py  # Time normalization
│   ├── calculate_time_features.py         # Time-of-day
│   ├── feature_selection.py               # Feature selection
│   └── train_comprehensive_model.py       # Model training
├── data/
│   ├── raw/                               # Raw input data
│   ├── page_views/                        # Processed page views
│   ├── enriched_features/                 # Feature parquet files
│   ├── feature_selection/                 # Selection results
│   ├── models/                            # Trained models
│   └── results/                           # Training results
└── notebooks/                             # Exploratory notebooks
```

## Memory Considerations (WSL 8GB)

The pipeline is designed for memory efficiency:
- Processes one feature type at a time
- Saves intermediate parquet files
- Uses float32 where possible
- Clears memory between steps

## Key References

1. **Nguyen 2020**: IQR_ITT methodology for outlier detection
2. **Oviedo et al.**: Workload dynamics, peak/slope features (ECTEL 2022)
3. **UA Feature Engineering**: 13 calculate_*.py scripts with 217 features

## Expected Outcomes

| Feature Set | Expected Recall | Rationale |
|-------------|-----------------|-----------|
| Current (78) | 0.63 | Baseline |
| + Inactivity + Course-relative | 0.72 | Critical timing features |
| + Workload dynamics | 0.75 | Oviedo paper patterns |
| + Pre-assessment | 0.78 | Deadline behavior |
| + IQR_ITT + Selection | **0.80+** | Remove noise |
