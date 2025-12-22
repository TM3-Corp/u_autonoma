# Phase 1: Baseline Model Training Results

**Date:** December 2025
**Checkpoint:** 1 of 4

---

## Summary

Trained prediction models on 7 Control de Gestión courses with reliable grade data.

| Metric | Value |
|--------|-------|
| Total courses | 7 |
| Total students | 302 |
| Completed courses (with failures) | 2 |
| Ongoing courses (all passing) | 5 |

---

## Key Finding: Class Diversity Issue

**Completed courses** (FUNDAMENTOS DE MICROECONOMÍA) have real class diversity:
- P03: 30 pass / 11 fail (27% failure rate)
- P01: 14 pass / 22 fail (61% failure rate)

**Ongoing courses** using `current_score` have **no failing students**:
- All students score 80%+ on completed work
- 100% pass rate on current_score
- Cannot train classification models (no class diversity)

**Implication:** For failure prediction, we need:
1. Completed courses (with final grades), OR
2. External grades from university for ongoing courses

---

## Per-Course Results

### Completed Courses (Use final_score)

| Course | Students | Target | Failure Rate | RF R² |
|--------|----------|--------|--------------|-------|
| FUNDAMENTOS DE MICROECONOMÍA-P03 | 41 | 70.6% avg | 27% | 1.000 |
| FUNDAMENTOS DE MICROECONOMÍA-P01 | 36 | 37.0% avg | 61% | 1.000 |

These courses have real predictive value for failure detection.

### Ongoing Courses (Use current_score)

| Course | Students | Target | Pass Rate | RF R² |
|--------|----------|--------|-----------|-------|
| TALL DE COMPETENCIAS DIGITALES-P01 | 48 | 86.0% avg | 100% | 0.376 |
| TALL DE COMPETENCIAS DIGITALES-P02 | 47 | 83.6% avg | 100% | 0.141 |
| FUND DE BUSINESS ANALYTICS-P01 | 23 | 96.4% avg | 100% | -4.709 |
| GESTIÓN DEL TALENTO-P01 | 40 | 80.3% avg | 100% | 0.539 |
| PENSAMIENTO MATEMÁTICO-P03 | 17 | 93.6% avg | 100% | 0.990 |

These courses can only be used for regression (predicting grade), not classification (pass/fail).

---

## Model Performance

### Aggregated Results

| Model Type | Avg R² | Avg F1 | Notes |
|------------|--------|--------|-------|
| ALL-DATA | -0.095 | 1.000 | F1=1.0 only from 2 completed courses |
| ACTIVITY-ONLY | 0.423 | 1.000 | Better R² due to activity features |

### Completed Courses (Classification Possible)

For FUNDAMENTOS DE MICROECONOMÍA courses:
- **ALL-DATA Model:** Perfect R² = 1.0 (likely overfitting on small sample)
- **ACTIVITY-ONLY Model:** High R² (activity correlates strongly with grades)
- **Classification F1:** 1.0 (perfect on train data)

**Note:** R² = 1.0 with small datasets (36-41 students) suggests overfitting. Need more data for reliable generalization.

---

## Data Extracted

Raw data saved to `/data/baseline/`:

| File | Records |
|------|---------|
| course_84936_raw.json | 41 enrollments, 82 submissions |
| course_84941_raw.json | 36 enrollments, 36 submissions |
| course_86005_raw.json | 50 enrollments, 867 submissions |
| course_86020_raw.json | 51 enrollments, 884 submissions |
| course_86676_raw.json | 40 enrollments, 984 submissions |
| course_86689_raw.json | 40 enrollments, 1320 submissions |
| course_76755_raw.json | 44 enrollments, 945 submissions |
| baseline_models_results.json | All model results |

---

## Pagination Verification

All extractions used bookmark-based pagination correctly:
- Enrollments: 1 page each (< 100 records)
- Submissions: Up to 14 pages (verified no truncation)
- Student summaries: 1 page each

No pagination truncation detected.

---

## Recommendations for Phase 2

1. **Focus extraction on activity data** - Available for all 21 courses
2. **Request external grades** for 16 courses without Canvas grades (Phase 3)
3. **For ongoing courses with Canvas grades**, wait for course completion OR use external final grades
4. **Expand to POSTGRADO** for more completed courses with class diversity

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/utils/pagination.py` | Robust Canvas API pagination |
| `scripts/train_baseline_models.py` | Baseline model training script |
| `data/baseline/*.json` | Raw course data |
| `data/baseline/baseline_models_results.json` | Model results |
| `docs/phase1_baseline_results.md` | This documentation |

---

## Next Steps

- [x] Phase 1.1: Create pagination utility
- [x] Phase 1.2: Extract baseline course data
- [x] Phase 1.3: Verify data quality
- [x] Phase 1.4: Train baseline models
- [x] Phase 1.5: Document results (this file)
- [ ] Phase 2: Extract activity from all 21 courses
- [ ] Phase 3: Prepare university data request
- [ ] Phase 4: Build integration framework

---

*Checkpoint 1 Complete - December 2025*
