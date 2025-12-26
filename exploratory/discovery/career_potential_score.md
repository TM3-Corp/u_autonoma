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

## Consolidated Data by Career

### Evaluations & Course Structure

| ID  | Career                           | Courses | With Grades | Assignments | Modules | With Activity |
|:---:|----------------------------------|:-------:|:-----------:|:-----------:|:-------:|:-------------:|
| 248 | Ingeniería Civil Informática     |   57    |     33      |     531     |   606   |      57       |
| 719 | Ingeniería en Control de Gestión |   26    |     11      |     373     |   416   |      26       |

### Grade Metrics

| ID  | Career                           | Students | With Grades | Avg Grade | Avg Variance | Avg Pass Rate |
|:---:|----------------------------------|:--------:|:-----------:|:---------:|:------------:|:-------------:|
| 248 | Ingeniería Civil Informática     |   1920   |    1133     |   32.7%   |     11.3     |      61%      |
| 719 | Ingeniería en Control de Gestión |    897   |     414     |   21.6%   |      6.6     |      64%      |

### Course Tier Distribution

| ID  | Career                           | HIGH | MEDIUM | LOW | SKIP | Analyzable % |
|:---:|----------------------------------|:----:|:------:|:---:|:----:|:------------:|
| 248 | Ingeniería Civil Informática     |  5   |   6    | 22  |  24  |     19%      |
| 719 | Ingeniería en Control de Gestión |  2   |   0    |  9  |  15  |      8%      |

*Run `python analyze_cps.py --all --update-report` to refresh all careers.*

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

## Career Details

### 248 - Ingeniería Civil Informática (CPS: 48.1)

**Summary:**
- **57 courses** total, **33 with grades** (58%)
- **531 assignments** across all courses
- **1920 students**, 1133 with grades (59%)
- **5 HIGH + 6 MEDIUM** potential courses (19% analyzable)

**Top Courses (HIGH & MEDIUM Potential):**

| Course ID | Name                               | Students | Grades | Assignments | Variance | Pass Rate | Tier   |
|:---------:|------------------------------------|:--------:|:------:|:-----------:|:--------:|:---------:|:------:|
| 81891     | Ciencia Materiales para Electr-P01 |    62    |   62   |     18      |   32.6   |    68%    | HIGH   |
| 87302     | Álgebra Lineal-P01                 |    44    |   44   |     10      |   32.2   |    66%    | HIGH   |
| 87314     | Cálculo II-P01                     |    37    |   37   |      3      |   29.8   |    78%    | MEDIUM |
| 81872     | Arquitectura de Software-P01       |    36    |   36   |     14      |   28.4   |    97%    | MEDIUM |
| 86667     | Cálculo I-P01                      |    50    |   50   |     30      |   26.3   |    72%    | HIGH   |
| 87303     | Física Mecánica-P01                |    23    |   23   |      3      |   21.2   |     9%    | MEDIUM |
| 81868     | Bases de Datos II-P04              |    26    |   26   |      5      |   18.9   |     8%    | MEDIUM |
| 81875     | Desarrollo de Aplicaciones Web-P01 |    26    |   26   |      9      |   18.2   |    88%    | MEDIUM |
| 81874     | Bases de Datos II-P01              |    52    |   52   |      6      |   17.2   |     8%    | MEDIUM |
| 81890     | Programación Orientada a Objet-P04 |    31    |   31   |     10      |   16.7   |    77%    | HIGH   |
| 86668     | Álgebra-P01                        |    50    |   50   |     30      |   15.2   |    20%    | HIGH   |

---

### 719 - Ingeniería en Control de Gestión (CPS: 33.3)

**Summary:**
- **26 courses** total, **11 with grades** (42%)
- **373 assignments** across all courses
- **897 students**, 414 with grades (46%)
- **2 HIGH + 0 MEDIUM** potential courses (8% analyzable)

**Top Courses (HIGH & MEDIUM Potential):**

| Course ID | Name                               | Students | Grades | Assignments | Variance | Pass Rate | Tier |
|:---------:|------------------------------------|:--------:|:------:|:-----------:|:--------:|:---------:|:----:|
| 86676     | FUND DE BUSINESS ANALYTICS-P01     |    40    |   40   |     24      |   26.4   |    28%    | HIGH |
| 86020     | TALL DE COMPETENCIAS DIGITALES-P02 |    51    |   51   |     18      |   24.7   |    63%    | HIGH |

*Run `python analyze_cps.py --all --update-report` to refresh all careers.*

---
