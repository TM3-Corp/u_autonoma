# Career Potential Score (CPS)

**Purpose:** Rank careers for early failure prediction and student intervention.

**Last Updated:** 2025-12-26

---

## Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `extract_career_data.py` | Fetch raw data from Canvas API | `python extract_career_data.py --career-id 248` |
| `analyze_cps.py` | Compute CPS from Parquet data | `python analyze_cps.py --career-id 248 --update-report` |

**Workflow:**
```bash
# Step 1: Extract raw data (no analysis)
python exploratory/discovery/extract_career_data.py --career-id 719

# Step 2: Analyze and compute CPS
python exploratory/discovery/analyze_cps.py --career-id 719 --update-report

# Analyze all careers
python exploratory/discovery/analyze_cps.py --all --update-report
```

---

## CPS Formula

```
CPS = (HP_Score × 0.30) + (Quality_Score × 0.25) + (Coverage_Score × 0.20) +
      (Data_Score × 0.15) + (Diversity_Score × 0.10)
```

## Analysis Configuration

- **Minimum students per course:** 16
- **Minimum students with grades:** 10
- **Pass threshold:** 57% (Chilean 4.0 scale)
- **Grade variance threshold:** >15%
- **Pass rate range:** 20-80%
- **Target terms:** 336 (2nd Sem 2025), 322 (1st Sem 2025)

## Component Definitions

| Component           | Weight | What it Measures                           |
| ------------------- | ------ | ------------------------------------------ |
| **HP Score**        | 30%    | High-potential course quantity + density   |
| **Quality Score**   | 25%    | Course tier distribution                   |
| **Coverage Score**  | 20%    | % of students in analyzable courses        |
| **Data Score**      | 15%    | Grade availability and completeness        |
| **Diversity Score** | 10%    | Pass rate balance and grade variance       |

### Course Tier Definitions

| Tier                 | Criteria                                                          | ML Value                 |
| -------------------- | ----------------------------------------------------------------- | ------------------------ |
| **HIGH POTENTIAL**   | Grades ≥10 students, Variance >15, Pass rate 20-80%, Assignments ≥5, Activity data | Ideal for training       |
| **MEDIUM POTENTIAL** | Grades ≥10 students, Variance >15, Assignments ≥3                 | Usable with caveats      |
| **LOW**              | Variance ≤15 OR few assignments                                   | Limited predictive value |
| **SKIP**             | No grade data available (<10 students with grades)                | Cannot use               |

### Diversity Score Components

Measures grade distribution quality for ML training:

| Factor | Weight | Description |
|--------|--------|-------------|
| Pass Balance | 50% | Ideal at 50% pass rate (100 points), decreases toward extremes |
| Variance Quality | 30% | Higher variance = better prediction potential |
| Cross-Course Diversity | 20% | Standard deviation of pass rates across courses |

---

## Career Ranking

| Rank | ID  | Career                           | CPS  | HIGH | MED | LOW | SKIP | Total | Analyzable Students |
| ---- | --- | -------------------------------- | ---- | ---- | --- | --- | ---- | ----- | :-----------------: |
| 1    | 248 | Ingeniería Civil Informática     | 48.1 | 5    | 6   | 22  | 24   | 57    |    437/1920 (23%)   |
| 2    | 719 | Ingeniería en Control de Gestión | 33.3 | 2    | 0   | 9   | 15   | 26    |     91/897 (10%)    |

*Run `python analyze_cps.py --all --update-report` to refresh all careers.*

---

## CPS Component Scores

| ID  | Career                           |   HP | Quality | Coverage | Data | Diversity |  CPS |
| --- | -------------------------------- | ---: | ------: | -------: | ---: | --------: | ---: |
| 248 | Ingeniería Civil Informática     | 52.5 |    32.5 |     38.2 | 78.7 |      47.6 | 48.1 |
| 719 | Ingeniería en Control de Gestión | 27.6 |    17.8 |     27.7 | 76.9 |      35.2 | 33.3 |

*Run `python analyze_cps.py --all --update-report` to refresh all careers.*

---

## Quality Metrics

| ID  | Career                           | Avg Variance | Avg Pass Rate | Pass Rate Std | Courses w/ Grades |
| --- | -------------------------------- | :----------: | :-----------: | :-----------: | :---------------: |
| 248 | Ingeniería Civil Informática     |     16.5     |      54%      |     0.28      |      33/57        |
| 719 | Ingeniería en Control de Gestión |      7.3     |      30%      |     0.31      |      11/26        |

*Run `python analyze_cps.py --all --update-report` to refresh all careers.*

---

## Top Courses by Career

### 248 - Ingeniería Civil Informática (CPS: 48.1)

| Course ID | Name                               | Students | Variance | Pass Rate |
| --------- | ---------------------------------- | :------: | :------: | :-------: |
| 81891     | Ciencia Materiales para Electr-P01 |    62    |   32.6   |    68%    |
| 87302     | Álgebra Lineal-P01                 |    44    |   32.2   |    66%    |
| 86667     | Cálculo I-P01                      |    50    |   26.3   |    72%    |
| 81890     | Programación Orientada a Objet-P04 |    31    |   16.7   |    77%    |
| 86668     | Álgebra-P01                        |    50    |   15.2   |    20%    |

### 719 - Ingeniería en Control de Gestión (CPS: 33.3)

| Course ID | Name                               | Students | Variance | Pass Rate |
| --------- | ---------------------------------- | :------: | :------: | :-------: |
| 86676     | FUND DE BUSINESS ANALYTICS-P01     |    40    |   26.4   |    28%    |
| 86020     | TALL DE COMPETENCIAS DIGITALES-P02 |    51    |   24.7   |    63%    |

*Run `python analyze_cps.py --all --update-report` to refresh all careers.*

---
