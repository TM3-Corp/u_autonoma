# Early Warning System - Universidad Autónoma de Chile

A predictive analytics system to identify at-risk students using Canvas LMS activity data, enabling early intervention before academic failure.

## Project Overview

**Objective:** Predict which students will fail courses (grade < 4.0 on Chilean 1-7 scale) using only LMS activity data available in the first 2-3 weeks of a semester.

**Key Finding:** Using behavioral patterns alone, we can identify **81.8% of students who will fail** early enough for intervention.

## Key Insights

| Finding | Impact |
|---------|--------|
| Morning studiers | **0% failure rate** |
| Evening studiers | **67% failure rate** |
| Low engagement (Q1) | 87.5% failure rate |
| Early module access | 2x better outcomes |

## Project Structure

```
u_autonoma/
├── data/
│   ├── early_warning/         # Processed features and visualizations
│   ├── raw/                   # Raw API extracts
│   └── processed/             # Cleaned datasets
├── docs/
│   ├── early_warning_findings.md
│   └── canvas_resource_tracking_analysis.md
├── notebooks/
│   ├── 01_eda_canvas_data.ipynb
│   └── 02_early_warning_visualization.ipynb
├── scripts/
│   ├── early_warning_system.py    # Main prediction system
│   ├── prediction_models.py       # Model training
│   └── utils/                     # Helper utilities
└── CLAUDE.md                      # API documentation
```

## Getting Started

### Prerequisites
- Python 3.8+
- Canvas LMS API access (test environment)

### Installation

```bash
# Clone the repository
git clone https://github.com/TM3-Corp/u_autonoma.git
cd u_autonoma

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter requests
```

### Running the Analysis

```bash
# Extract features and train models
python scripts/early_warning_system.py

# View visualizations
jupyter notebook notebooks/02_early_warning_visualization.ipynb
```

## Branch Structure

| Branch | Purpose |
|--------|---------|
| `main` | Production-ready code (protected) |
| `develop` | Integration branch for features |
| `feature/eda-*` | Exploratory data analysis work |

## Contributing

1. Create a feature branch from `develop`
2. Make your changes
3. Submit a pull request to `develop`
4. After review, changes will be merged to `main`

## Team

- **Lead:** Paul
- **Intern:** Vicente - Exploratory Data Analysis
- **Intern:** Sebastian - Exploratory Data Analysis

## License

Private - Universidad Autónoma de Chile / TM3 Corp

---

*Early Warning System v1.0 - December 2025*
