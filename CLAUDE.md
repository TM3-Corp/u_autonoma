# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

> ## 🧭 SINGLE SOURCE OF TRUTH → read **[`PROJECT_SSOT.md`](PROJECT_SSOT.md)** FIRST
> Authoritative index: headline metrics, the doc-map (one canonical doc per topic), the full experiment ledger, and which of the ~98 repo docs are stale. **Defensible numbers: PUC ROC-AUC ~0.80 (wk2) → 0.84–0.85 (wk8); UA ~0.81 (full, within-institution); nested-LOCO calibrated.** Any repo number ≥0.86 PUC / ≥0.85 UA is optimistic/contaminated — do not quote it.


## Project Overview

Canvas LMS data extraction and student failure prediction for Universidad Autónoma de Chile. Predicts which students will fail (grade < 4.0 on Chilean 1-7 scale, or < 57% on percentage scale) using LMS activity data.

## Setup

```bash
# Create and configure environment
cp .env.example .env
# Add CANVAS_API_URL and CANVAS_API_TOKEN to .env

# Install dependencies
pip install -r requirements.txt
```

## API Configuration

Credentials stored in `.env` (not committed):

```python
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}
```

**Environment**: Test (`uautonoma.test.instructure.com`)

## Account Hierarchy

Token has access to:
- **PREGRADO** (Account 46) → Providencia (Account 176) → 3,393 courses
  - Ing. en Control de Gestión (Account 719) - primary test target
- **POSTGRADO** (Account 42) → 66 sub-accounts, 1,000+ courses

## Test Courses

| Course ID | Name | Students | Grade Scale |
|-----------|------|----------|-------------|
| 76755 | PENSAMIENTO MATEMÁTICO-P03 | 44 | Chilean 1-7 |
| 86005 | TALL DE COMPETENCIAS DIGITALES-P01 | 50 | Percentage |
| 86676 | TALLER PENSAMIENTO ANALÍTICO-P01 | 40 | Good variance |
| 84936 | FUNDAMENTOS DE MICROECONOMÍA-P03 | 41 | Near-perfect prediction |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/prediction_models.py` | Train regression + classification models |
| `scripts/extract_course_data.py` | Bulk data extraction from Canvas API |
| `notebooks/exploration.ipynb` | Data exploration and API testing |

## Prediction Models

> ⚠️ *Legacy setup context (early exploratory run) — NOT current performance. Current metrics: `PROJECT_SSOT.md` §1 / `RESULTS_LEDGER.md`. The F1=1.000 below is a small-sample artifact.*

Two model types trained on 6 courses (258 students):

| Model | Avg R² | Avg F1 | Use Case |
|-------|--------|--------|----------|
| All-Data (21 features) | 0.756 | 1.000 | Maximum accuracy with grade data |
| Activity-Only (5 features) | 0.491 | 0.933 | Early warning before any grades exist |

Top activity predictors: `participations_level` (0.36), `participations` (0.35), `total_activity_time` (0.15)

## Data Available via Canvas API

| Data | API | Key Fields |
|------|-----|------------|
| Course grades | Enrollments | `current_score`, `final_score` |
| Assignment grades | Submissions | `score`, `submitted_at`, `graded_at` |
| Activity metrics | Student Summaries | `page_views`, `participations`, `tardiness_breakdown` |
| Clickstream | Page Views | `url`, `action`, `interaction_seconds` |
| Grade history | Gradebook History | `previous_grade`, `new_grade` |
| Assignment stats | Course Assignments Analytics | `median`, `quartiles`, `tardiness_breakdown` |
| Department analytics | Account Analytics | Activity by category, grade distribution |

Full API reference with request/response examples: `docs/canvas-api-reference.md`

## Important Notes

- Page Views API has **no course_id filter** — must parse URLs post-fetch
- "Libro de Calificaciones" is an external LTI tool — grades there are NOT accessible via Canvas API
- Different courses use different grade scales (1-7 Chilean vs 0-100%)
- Always paginate (max 100 records per request)
- Canvas has rate limits — implement delays for bulk extraction
- 25/31 Control de Gestión courses lacked sufficient grade data
- Best prediction results come from courses with 20-80% pass rates
