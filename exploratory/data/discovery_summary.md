# PREGRADO Course Discovery Summary

**Date:** December 2024
**Branch:** feature/eda-vicente

---

## High-Potential Courses Found

| Course ID | Name | Account ID | Career | Students | Variance | Pass Rate | Assignments |
|-----------|------|------------|--------|----------|----------|-----------|-------------|
| 81891 | Ciencia Materiales para Electr-P01 | 248 | Ingeniería Civil Informática | 62 | 32.6 | 68% | 18 |
| 82198 | NEUROCIENCIAS-P01 | 249 | Medicina | 56 | 29.5 | 48% | 8 |
| 81837 | ÁLGEBRA-P01 | 263 | Ingeniería Civil Industrial | 55 | 26.7 | 35% | 28 |
| 83844 | SALUD FAMILIAR Y COMUNITARIA-P01 | 254 | Kinesiología | 59 | 20.8 | 20% | 7 |
| 80699 | MICROBIOLOGÍA Y PARASITOLOGÍA-P11 | 260 | Nutrición y Dietética | 56 | 17.4 | 64% | 28 |
| 83168 | MICROBIOLOGÍA Y PARASITOLOGÍA-P32 | 262 | Química y Farmacia | 56 | 16.7 | 39% | 28 |

---

## Medium-Potential Courses

| Course ID | Name | Account ID | Career | Students | Variance | Pass Rate | Assignments |
|-----------|------|------------|--------|----------|----------|-----------|-------------|
| 82202 | PATOLOGÍA Y ANAT PATOLÓGICA-P01 | 249 | Medicina | 56 | 58.1 | 100%* | 42 |
| 80693 | QUÍMICA ORGÁNICA-P05 | 260 | Nutrición y Dietética | 55 | 15.6 | 0%* | 24 |
| 81888 | ÁLGEBRA-P07 | 251 | Ingeniería Civil Química | 57 | 14.9 | 0%* | 27 |

*Pass rate outside ideal 20-80% range

---

## Account ID to Career Mapping (Providencia)

| Account ID | Career |
|------------|--------|
| 244 | Odontología |
| 245 | Publicidad Profesional y Comunicación Integral |
| 246 | Administración Pública |
| 247 | Psicología |
| 248 | Ingeniería Civil Informática |
| 249 | Medicina |
| 250 | Aud. e Ing En Control de Gest. |
| 251 | Ingeniería Civil Química |
| 252 | Periodismo |
| 253 | Derecho |
| 254 | Kinesiología |
| 255 | Ingeniería Comercial |
| 256 | Obstetricia y Puericultura |
| 257 | Enfermería |
| 258 | Fonoaudiología |
| 259 | Terapia Ocupacional |
| 260 | Nutrición y Dietética |
| 261 | Formación General |
| 262 | Química y Farmacia |
| 263 | Ingeniería Civil Industrial |
| 311 | Ingeniería Comercial |
| 730 | Ingeniería Civil Industrial |

---

## Discovery Statistics

### Providencia Scan (Account 176)

- **Sub-accounts scanned:** 41 programs
- **Total candidate courses:** 741 (with 20+ students)
- **Courses analyzed:** 50 (top by student count)
- **High potential:** 6 courses
- **Medium potential:** 3 courses

### Original Pregrado Scan (Psicología + Derecho)

- **Sub-accounts scanned:** 3 (730, 247, 253)
- **Total candidate courses:** 100 (with 15+ students)
- **High potential:** 1 course
- **Medium potential:** 2 courses

---

## Selection Criteria

A "high-potential" course must have:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Students | 20+ | Statistical significance for ML |
| Grade Variance | > 10 (std dev) | Enough variation to predict |
| Pass Rate | 20-80% | Class diversity for training |
| Assignments | 5+ | Good LMS design indicator |
| Term | 336 or 322 | Current or recent semester |

---

## Recommended Courses for Prediction Modeling

### Tier 1 (Best Candidates)

1. **ÁLGEBRA-P01** (81837) - Ingeniería Civil Industrial
   - 35% pass rate (ideal class balance)
   - 28 assignments (rich LMS data)
   - High variance (26.7)

2. **NEUROCIENCIAS-P01** (82198) - Medicina
   - 48% pass rate (near-perfect balance)
   - 8 assignments
   - High variance (29.5)

3. **MICROBIOLOGÍA Y PARASITOLOGÍA-P32** (83168) - Química y Farmacia
   - 39% pass rate
   - 28 assignments
   - Good variance (16.7)

### Tier 2 (Good Candidates)

4. **Ciencia Materiales para Electr-P01** (81891) - Ing. Civil Informática
5. **SALUD FAMILIAR Y COMUNITARIA-P01** (83844) - Kinesiología
6. **MICROBIOLOGÍA Y PARASITOLOGÍA-P11** (80699) - Nutrición y Dietética

---

## Output Files

| File | Description |
|------|-------------|
| `pregrado_discovery_results.csv` | Original scan (100 courses from Psicología, Derecho) |
| `providencia_discovery_results.csv` | Providencia scan (50 courses analyzed) |
| `discovery_summary.md` | This summary document |

---

## Next Steps

1. Extract page views for Tier 1 courses
2. Build feature matrices combining grades + activity
3. Train prediction models
4. Compare results with Control de Gestión baseline
