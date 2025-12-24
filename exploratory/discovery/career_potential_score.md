# Career Potential Score (CPS) - Intervention Focused

**Purpose:** Rank careers for early failure prediction and student intervention.

**Last Updated:** 2025-12-24

---

## CPS Formula (Intervention-Focused)

```
CPS = (HP_Score × 0.25) + (Quality_Score × 0.20) + (Coverage_Score × 0.20) +
      (Data_Score × 0.15) + (Intervention_Score × 0.20)
```

### Key Change: Intervention Score

Replaces Diversity Score. Optimizes for **high failure rates** (more students to help)
while maintaining trainability (need some passes for ML model).

| Pass Rate | Failure Rate | Intervention Factor | Reason |
|:---------:|:------------:|:-------------------:|--------|
| 20-40% | **60-80%** | 100 (max) | Sweet spot: high intervention, trainable |
| 15-20% | 80-85% | 70-100 | Very high failure, slightly reduced trainability |
| 40-85% | 15-60% | 40-100 | Moderate failure, decreasing intervention value |
| >85% | <15% | 0-40 | Too few failures to help |
| <15% | >85% | 0-40 | Too few passes to train model |

---

## Career Ranking

| Rank | ID | Career | CPS | HIGH | MED | LOW | SKIP | Intervention Potential |
|------|----|--------|-----|------|-----|-----|------|:----------------------:|
| 1 | 248 | Ingeniería Civil Informática | 57.1 | 5 | 9 | 17 | 22 | 376 students |
| 2 | 263 | Ingeniería Civil Industrial | 53.7 | 5 | 3 | 4 | 29 | 208 students |
| 3 | 251 | Ingeniería Civil Química | 53.3 | 2 | 1 | 4 | 5 | 120 students |
| 4 | 311 | Ingeniería Comercial | 51.0 | 4 | 8 | 22 | 31 | 354 students |
| 5 | 260 | Nutrición y Dietética | 48.4 | 3 | 2 | 4 | 31 | 172 students |
| 6 | 262 | Química y Farmacia | 48.1 | 3 | 1 | 17 | 18 | 143 students |
| 7 | 719 | Ingeniería en Control de Gestión | 46.3 | 2 | 1 | 7 | 14 | 104 students |
| 8 | 250 | Aud. e Ing En Control de Gest. | 40.0 | 1 | 1 | 2 | 11 | 37 students |
| 9 | 249 | Medicina | 38.2 | 1 | 1 | 2 | 16 | 14 students |
| 10 | 254 | Kinesiología | 36.2 | 3 | 0 | 5 | 25 | 107 students |
| 11 | 247 | Psicología | 35.2 | 0 | 4 | 16 | 45 | 144 students |
| 12 | 253 | Derecho | 12.6 | 0 | 0 | 17 | 60 | 0 students |

---

## CPS Component Scores

| ID | Career | HP | Quality | Coverage | Data | Intervention | CPS |
|----|--------|---:|--------:|---------:|-----:|-------------:|----:|
| 248 | Ingeniería Civil Informática | 57.1 | 44.8 | 43.7 | 79.6 | 65.9 | 57.1 |
| 263 | Ingeniería Civil Industrial | 50.3 | 41.8 | 36.3 | 71.7 | 73.5 | 53.7 |
| 251 | Ingeniería Civil Química | 36.5 | 46.9 | 49.4 | 78.6 | 65.5 | 53.3 |
| 311 | Ingeniería Comercial | 51.1 | 41.0 | 34.5 | 77.6 | 57.6 | 51.0 |
| 260 | Nutrición y Dietética | 38.7 | 37.7 | 32.3 | 62.8 | 76.6 | 48.4 |
| 262 | Química y Farmacia | 36.4 | 39.3 | 28.9 | 80.3 | 66.4 | 48.1 |
| 719 | Ingeniería en Control de Gestión | 31.5 | 39.3 | 33.1 | 76.7 | 62.1 | 46.3 |
| 250 | Aud. e Ing En Control de Gest. | 23.6 | 37.9 | 26.4 | 58.4 | 62.2 | 40.0 |
| 249 | Medicina | 22.4 | 36.0 | 29.2 | 67.5 | 47.0 | 38.2 |
| 254 | Kinesiología | 34.5 | 27.4 | 28.9 | 69.7 | 29.4 | 36.2 |
| 247 | Psicología | 21.9 | 23.9 | 24.6 | 65.6 | 51.1 | 35.2 |
| 253 | Derecho | 0.0 | 11.5 | 0.0 | 68.8 | 0.0 | 12.6 |

---

## Quality & Intervention Metrics

| ID | Career | Variance | Pass Rate | Failure Rate | Analyzable | Potential Failures |
|----|--------|:--------:|:---------:|:------------:|:----------:|:------------------:|
| 248 | Ingeniería Civil Informática | 10.8 | 31% | **69%** | 546 | **376** |
| 263 | Ingeniería Civil Industrial | 18.1 | 34% | **66%** | 319 | **208** |
| 251 | Ingeniería Civil Química | 14.5 | 19% | **81%** | 148 | **120** |
| 311 | Ingeniería Comercial | 9.0 | 18% | **82%** | 432 | **354** |
| 260 | Nutrición y Dietética | 21.1 | 27% | **73%** | 237 | **172** |
| 262 | Química y Farmacia | 11.3 | 26% | **74%** | 194 | **143** |
| 719 | Ingeniería en Control de Gestión | 7.2 | 26% | **74%** | 141 | **104** |
| 250 | Aud. e Ing En Control de Gest. | 7.3 | 37% | **63%** | 59 | **37** |
| 249 | Medicina | 24.6 | 87% | **13%** | 112 | **14** |
| 254 | Kinesiología | 10.0 | 12% | **88%** | 122 | **107** |
| 247 | Psicología | 10.3 | 15% | **85%** | 171 | **144** |
| 253 | Derecho | 0.0 | 0% | **100%** | 0 | **0** |

---

## Top Courses by Career

### 248 - Ingeniería Civil Informática (CPS: 57.1)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 81891 | Ciencia Materiales para Electr-P01 | 62 | 32.6 | 68% | **20** |
| 87302 | Álgebra Lineal-P01 | 44 | 32.2 | 66% | **15** |
| 86667 | Cálculo I-P01 | 50 | 26.3 | 72% | **14** |
| 81890 | Programación Orientada a Objet-P04 | 31 | 16.7 | 77% | **7** |
| 86668 | Álgebra-P01 | 50 | 15.2 | 20% | **40** |

### 263 - Ingeniería Civil Industrial (CPS: 53.7)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 87299 | ECUACIONES DIFERENCIALES-P01 | 21 | 35.1 | 48% | **11** |
| 81837 | ÁLGEBRA-P01 | 55 | 26.7 | 35% | **36** |
| 87294 | FÍSICA MECÁNICA-P01 | 26 | 18.6 | 62% | **10** |
| 81848 | ÁLGEBRA-P04 | 54 | 15.7 | 70% | **16** |
| 81843 | CÁLCULO I-P04 | 53 | 15.7 | 70% | **16** |

### 251 - Ingeniería Civil Química (CPS: 53.3)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 81879 | QUÍMICA ORGÁNICA-P01 | 40 | 28.5 | 62% | **15** |
| 81887 | PROGRAMACIÓN-P07 | 51 | 16.8 | 37% | **32** |

### 311 - Ingeniería Comercial (CPS: 51.0)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 86321 | ECONOMETRÍA-P03 | 32 | 35.1 | 44% | **18** |
| 86025 | TALL DE COMPETENCIAS DIGITALES-P05 | 26 | 35.1 | 50% | **13** |
| 86022 | TALL DE COMPETENCIAS DIGITALES-P04 | 44 | 28.6 | 61% | **17** |
| 86021 | TALL DE COMPETENCIAS DIGITALES-P03 | 50 | 23.5 | 64% | **18** |

### 260 - Nutrición y Dietética (CPS: 48.4)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 80686 | BROMAT Y BIOQUÍM DE LOS ALIMEN-P01 | 35 | 27.4 | 69% | **11** |
| 80694 | MICROBIOLOGÍA Y PARASITOLOGÍA-P06 | 39 | 22.1 | 74% | **10** |
| 80699 | MICROBIOLOGÍA Y PARASITOLOGÍA-P11 | 56 | 17.4 | 64% | **19** |

### 262 - Química y Farmacia (CPS: 48.1)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 83168 | MICROBIOLOGÍA Y PARASITOLOGÍA-P32 | 56 | 16.7 | 39% | **34** |
| 83182 | MICROBIOLOGÍA Y PARASITOLOGÍA-P26 | 54 | 16.1 | 26% | **40** |
| 83194 | SALUD PÚBLICA-P01 | 37 | 12.9 | 57% | **16** |

### 719 - Ingeniería en Control de Gestión (CPS: 46.3)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 86676 | FUND DE BUSINESS ANALYTICS-P01 | 40 | 26.4 | 28% | **29** |
| 86020 | TALL DE COMPETENCIAS DIGITALES-P02 | 51 | 24.7 | 63% | **19** |

### 250 - Aud. e Ing En Control de Gest. (CPS: 40.0)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 82610 | CTRL GEST. DRLLO. ORGA.-P01 | 29 | 18.9 | 48% | **14** |

### 249 - Medicina (CPS: 38.2)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 82198 | NEUROCIENCIAS-P01 | 56 | 29.5 | 48% | **29** |

### 254 - Kinesiología (CPS: 36.2)

| Course ID | Name | Students | Variance | Pass Rate | Failures |
|-----------|------|:--------:|:--------:|:---------:|:--------:|
| 80667 | BIOFÍSICA-P06 | 35 | 30.5 | 54% | **16** |
| 83844 | SALUD FAMILIAR Y COMUNITARIA-P01 | 59 | 20.8 | 20% | **47** |
| 83100 | METOD. DE LA INVEST.-P02 | 28 | 19.8 | 21% | **22** |

### 247 - Psicología (CPS: 35.2)

*No HIGH potential courses found.*

### 253 - Derecho (CPS: 12.6)

*No HIGH potential courses found.*

---

## Analysis Configuration

- **Minimum students per course:** 20
- **Pass threshold:** 57% (Chilean 4.0 scale)
- **Sweet spot pass rate:** 20-40% (60-80% failure)
- **Target terms:** 336 (2nd Sem 2025), 322 (1st Sem 2025)

## Component Definitions

| Component | Weight | What it Measures |
|-----------|--------|------------------|
| **HP Score** | 25% | High-potential course quantity + density |
| **Quality Score** | 20% | Course tier distribution |
| **Coverage Score** | 20% | % of students in analyzable courses |
| **Data Score** | 15% | Grade availability and completeness |
| **Intervention Score** | 20% | Failure rate optimization for intervention |

### Course Tier Definitions

| Tier | Criteria | ML Value |
|------|----------|----------|
| **HIGH POTENTIAL** | Variance > 10, Pass rate 20-80%, Assignments >= 5 | Ideal for training |
| **MEDIUM POTENTIAL** | Variance > 10, Assignments >= 3 | Usable with caveats |
| **LOW** | Variance <= 10 OR few assignments | Limited predictive value |
| **SKIP** | No grade data available | Cannot use |