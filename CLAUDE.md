# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Early warning system to predict student failure in courses at Universidad Autónoma de Chile using Canvas LMS activity data. Goal: identify at-risk students BEFORE first exams when intervention can still help.

**Failure threshold:** < 4.0 on Chilean 1-7 scale (= < 57% on percentage scale)

---

## Development Commands

```bash
# Environment setup
python -m venv venv
source venv/bin/activate
pip install pandas numpy scikit-learn matplotlib seaborn jupyter requests python-dotenv

# Run early warning system
python scripts/early_warning_system.py

# Run baseline model training
python scripts/train_baseline_models.py

# Run correlation analysis
python scripts/correlation_analysis.py

# Start Jupyter notebooks
jupyter notebook notebooks/

# Test pagination utility
python scripts/utils/pagination.py
```

---

## Code Architecture

### Core Scripts (`scripts/`)
- `config.py` - API configuration, account IDs, course lists (loads from `.env`)
- `early_warning_system.py` - Main `EarlyWarningSystem` class for prediction
- `prediction_models.py` - ALL-DATA vs ACTIVITY-ONLY model comparison
- `train_baseline_models.py` - Model training pipeline
- `utils/pagination.py` - **Critical**: Canvas bookmark-based pagination

### Key Class: EarlyWarningSystem
Located in `scripts/early_warning_system.py`, provides:
- `extract_comprehensive_features()` - Pulls from Canvas APIs
- `calculate_early_access_scores()` - Ranks students by module timing
- `train_early_warning_models()` - Trains classifiers (LogReg, RF, GradBoost)
- `generate_risk_report()` - Outputs risk levels (Low/Medium/High/Critical)

### Pagination Warning
Canvas uses **bookmark-based pagination**, NOT page numbers. Always use:
```python
from utils.pagination import paginate_canvas
results = paginate_canvas(url, headers, params={...})
```

### Feature Sets
**ACTIVITY-ONLY** (early warning before grades):
- `page_views`, `participations`, `total_activity_time`
- `morning_activity`, `evening_activity` (time patterns)
- `early_access_score` (module access timing)

**ALL-DATA** (includes grades):
- Activity features + `avg_score`, `submission_rate`, etc.

---

## Branch Workflow

| Branch | Purpose |
|--------|---------|
| `main` | Protected - merged via PR only |
| `develop` | Integration branch |
| `feature/eda-{name}` | Individual EDA work branches |

Claude Code hooks in `.claude/hooks/check-git-branch.sh` enforce restrictions based on `git config user.allowed-branch`.

---

## Data Flow

1. **Extract**: Canvas APIs → raw JSON
2. **Transform**: Feature engineering in `early_warning_system.py`
3. **Save**: `data/early_warning/student_features.csv`
4. **Train**: Models → `data/early_warning/model_results.json`
5. **Visualize**: Notebooks → `data/early_warning/viz_*.png`

---

## API Configuration

### Environment: TEST (as of December 2025)

**Credentials are stored in `.env` file (not committed to Git).**

```python
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')

# Standard headers for all requests
headers = {'Authorization': f'Bearer {API_TOKEN}'}
```

**Setup:** Copy `.env.example` to `.env` and add your credentials.

### Account Hierarchy (Full Access Discovered)
```
Universidad Autónoma de Chile (UA) [Account ID: 1]
├── PREGRADO [Account ID: 46]
│   └── Providencia [Account ID: 176] (Sede) - 43 careers, 3,393 courses
│       ├── Ing. en Control de Gestión [Account ID: 719] - 97 courses, 32 with students
│       ├── Ing. en Control de Gestión Alt [Account ID: 718] - 20 active courses
│       └── Ingeniería Civil Industrial [Account ID: 730] - 18 active courses
│
└── POSTGRADO [Account ID: 42] - 66 sub-accounts, 1000+ active courses
    ├── Campus Virtual [401] - 29 active courses
    ├── Providencia [402] - 97 active courses
    ├── Temuco [405] - 79 active courses
    ├── Talca [404] - 62 active courses
    ├── Magíster en Psicología Clínica [554] - 77 active courses
    ├── Especialidad en Medicina de Urgenci [747] - 70 active courses
    ├── Especialidad Médica en Medicina Int [743] - 65 active courses
    ├── Magíster en dirección de personas [463] - 61 active courses
    ├── Especialidad en Medicina Familiar [745] - 59 active courses
    ├── Magíster en Dirección de Empresas [551] - 58 active courses
    └── ... and 50+ more Masters/Specialty programs
```

**Key Discovery:** Token has access to MUCH more than Control de Gestión - entire POSTGRADO with 1000+ courses!

### Test Courses (Verified with Data)
| Course ID | Name | Term | Students | Has Grades |
|-----------|------|------|----------|------------|
| 76755 | PENSAMIENTO MATEMÁTICO-P03 | 322 (1st Sem 2025) | 44 | ✅ Chilean 1-7 scale |
| 86005 | TALL DE COMPETENCIAS DIGITALES-P01 | 336 (2nd Sem 2025) | 50 | ✅ Percentage scale |
| 86676 | TALLER PENSAMIENTO ANALÍTICO-P01 | 336 | 40 | ✅ Good grade variance |
| 84936 | FUNDAMENTOS DE MICROECONOMÍA-P03 | 352 | 41 | ✅ Near-perfect prediction |
| 84941 | FUNDAMENTOS DE MICROECONOMÍA-P01 | 352 | 36 | ✅ Near-perfect prediction |

---

## Verified API Endpoints

### 1. ENROLLMENTS API ✅ (Primary Grade Source)

**Purpose:** Get aggregate course grades (current_score, final_score) for all students.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/enrollments
```

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `type[]` | `StudentEnrollment` | Filter to students only |
| `per_page` | `100` | Pagination (max 100) |
| `include[]` | `grades` | Include grade data |
| `include[]` | `total_scores` | Include total scores |

**Example Request:**
```python
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

response = requests.get(
    f'{API_URL}/api/v1/courses/86005/enrollments',
    headers=headers,
    params={
        'type[]': 'StudentEnrollment',
        'per_page': 100,
        'include[]': ['grades', 'total_scores']
    }
)

enrollments = response.json()
for e in enrollments:
    grades = e.get('grades', {})
    print(f"User {e['user_id']}: current={grades.get('current_score')}, final={grades.get('final_score')}")
```

**Response Fields:**
```json
{
  "user_id": 117656,
  "course_id": 86005,
  "enrollment_state": "active",
  "grades": {
    "current_score": 79.07,      // Running grade (completed work only)
    "final_score": 46.65,        // Final grade (including zeros for missing)
    "current_grade": null,       // Letter grade (if configured)
    "final_grade": null
  }
}
```

**Verified Results (Course 86005):**
- 50 students enrolled
- 48 with current_score (72% - 95.7%)
- 48 with final_score (18.5% - 60.6%)
- 47/48 students below 57% (potential failures)

---

### 2. SUBMISSIONS API ✅ (Per-Assignment Grades)

**Purpose:** Get individual assignment scores for each student.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/students/submissions
```

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `student_ids[]` | `all` | Get all students |
| `per_page` | `100` | Pagination |
| `include[]` | `assignment` | Include assignment details |
| `include[]` | `submission_history` | Include submission history |

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/students/submissions',
    headers=headers,
    params={
        'student_ids[]': 'all',
        'per_page': 100
    }
)

submissions = response.json()
for s in submissions:
    if s.get('score') is not None:
        print(f"User {s['user_id']}, Assignment {s['assignment_id']}: score={s['score']}")
```

**Response Fields:**
```json
{
  "user_id": 88268,
  "assignment_id": 465607,
  "score": 92.0,
  "grade": "92",
  "submitted_at": "2025-09-15T14:30:00Z",
  "graded_at": "2025-09-16T10:00:00Z",
  "workflow_state": "graded"  // or "submitted", "unsubmitted", "pending_review"
}
```

**Verified Results (Course 86005):**
- 3000+ total submissions
- 1460+ with scores
- Assignment types: "formativa" (quizzes), "sumativa" (exams)

---

### 3. ASSIGNMENTS API ✅ (Assignment Metadata)

**Purpose:** Get all assignments with due dates, points, and types.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/assignments
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/assignments',
    headers=headers,
    params={'per_page': 50, 'order_by': 'due_at'}
)

assignments = response.json()
for a in assignments:
    print(f"{a['id']}: {a['name']} | {a['points_possible']}pts | Due: {a['due_at']}")
```

**Response Fields:**
```json
{
  "id": 465607,
  "name": "Evaluación sumativa semana 3",
  "points_possible": 100.0,
  "due_at": "2025-11-06T02:59:59Z",
  "grading_type": "points",
  "submission_types": ["online_text_entry", "online_upload"],
  "assignment_group_id": 150309
}
```

**Assignment Types Identified:**
- `Evaluación formativa` = Formative assessments (quizzes, low-stakes)
- `Evaluación sumativa` = Summative assessments (exams, high-stakes)

---

### 4. ASSIGNMENT GROUPS API ✅ (Grade Weights)

**Purpose:** Get assignment categories and their weight in final grade.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/assignment_groups
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/assignment_groups',
    headers=headers
)

groups = response.json()
for g in groups:
    print(f"{g['id']}: {g['name']} ({g['group_weight']}%)")
```

**Verified Results (Course 86005):**
```
Semana 3:  6% weight (Evaluación sumativa)
Semana 6:  6% weight (Evaluación sumativa)
Semana 8: 35% weight (Evaluación sumativa) ← Main exam!
Semana 10: 6% weight (Evaluación sumativa)
```

---

### 5. STUDENT SUMMARIES API ✅ (Activity Metrics)

**Purpose:** Get engagement metrics (page views, participations, tardiness).

**Endpoint:**
```
GET /api/v1/courses/{course_id}/analytics/student_summaries
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/student_summaries',
    headers=headers,
    params={'per_page': 100}
)

summaries = response.json()
for s in summaries:
    print(f"User {s['id']}: views={s['page_views']}, participations={s['participations']}")
    tb = s.get('tardiness_breakdown', {})
    print(f"  Tardiness: on_time={tb.get('on_time')}, late={tb.get('late')}, missing={tb.get('missing')}")
```

**Response Fields:**
```json
{
  "id": 117656,
  "page_views": 1672,
  "page_views_level": 3,
  "participations": 13,
  "participations_level": 2,
  "tardiness_breakdown": {
    "on_time": 10,
    "late": 0,
    "missing": 4,
    "floating": 0
  }
}
```

**Verified Results (Course 86005):**
- 51 students with activity data
- Total page views: 34,887
- All 51 have participations > 0
- Tardiness data available for all

---

### 6. GRADEBOOK HISTORY API ✅ (Grade Changes)

**Purpose:** Track historical grade changes with timestamps.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/gradebook_history/days
GET /api/v1/courses/{course_id}/gradebook_history/feed
```

**Example Request:**
```python
# Get days with grading activity
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/gradebook_history/days',
    headers=headers,
    params={'per_page': 10}
)

days = response.json()
for day in days:
    print(f"Date: {day['date']}, Graders: {len(day.get('graders', []))}")

# Get detailed feed
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/gradebook_history/feed',
    headers=headers,
    params={'per_page': 50}
)

feed = response.json()
for entry in feed:
    print(f"User {entry['user_id']}: {entry['previous_grade']} → {entry['grade']}")
```

---

### 7. PAGE VIEWS API ✅ (Detailed Clickstream)

**Purpose:** Get detailed user activity (URLs visited, time spent).

**Endpoint:**
```
GET /api/v1/users/{user_id}/page_views
```

**⚠️ Important Limitations:**
- Only filters by `start_time` and `end_time`
- **NO course_id filter** - must filter post-fetch by parsing URLs
- Data persists even after student dis-enrollment

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/users/117656/page_views',
    headers=headers,
    params={
        'start_time': '2025-08-01T00:00:00Z',
        'end_time': '2025-12-01T00:00:00Z',
        'per_page': 100
    }
)

page_views = response.json()
for pv in page_views:
    print(f"URL: {pv['url']}, Action: {pv['action']}, Time: {pv['interaction_seconds']}s")
```

**Filtering by Course (Post-Fetch):**
```python
import re

def extract_course_id(url):
    """Extract course_id from Canvas URL"""
    match = re.search(r"/courses/(\d+)", url)
    return int(match.group(1)) if match else -1

# Filter page views for specific course
course_views = [pv for pv in page_views if extract_course_id(pv['url']) == 86005]
```

---

### 8. EXTERNAL TOOLS API ✅ (LTI Tools)

**Purpose:** List external tools (including "Libro de Calificaciones").

**Endpoint:**
```
GET /api/v1/accounts/{account_id}/external_tools
GET /api/v1/courses/{course_id}/external_tools
```

**Key Finding - "Libro de Calificaciones":**
```json
{
  "id": 644,
  "name": "Libro de Calificaciones",
  "url": "https://uautonoma.ltigb.entornosdeformacion.com/launch",
  "consumer_key": "canvas",
  "privacy_level": "public"
}
```

**⚠️ Limitation:** This is an external LTI tool. Grades stored in this system are NOT accessible via Canvas API. Data lives on external server `ltigb.entornosdeformacion.com`.

---

### 9. CUSTOM GRADEBOOK COLUMNS API ✅ (Tested - No Data)

**Purpose:** Get custom columns added to gradebook.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/custom_gradebook_columns
```

**Verified Result:** No custom columns defined in test courses.

---

### 10. COURSE ACTIVITY API ✅ (Daily Aggregates)

**Purpose:** Get daily page views and participations for entire course.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/analytics/activity
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/activity',
    headers=headers
)

activity = response.json()
# Returns list of daily activity
# [{'date': '2025-10-31', 'views': 78, 'participations': 2}, ...]
```

**Verified Result:** 120 days of activity data for course 86005.

---

### 11. USER ACTIVITY IN COURSE API ✅ (Per-User Detail)

**Purpose:** Get hourly page views breakdown for specific student.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/users/117656/activity',
    headers=headers
)

# Returns dict with 'page_views' and 'participations' broken down by hour
```

---

### 12. QUIZZES API ✅ (Quiz Metadata)

**Purpose:** Get quiz details (separate from assignments).

**Endpoint:**
```
GET /api/v1/courses/{course_id}/quizzes
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/quizzes',
    headers=headers,
    params={'per_page': 50}
)

quizzes = response.json()
for q in quizzes:
    print(f"{q['id']}: {q['title']} ({q['points_possible']} pts)")
```

**Verified Result:** 10 quizzes found (formative evaluations).

---

### 13. MODULES API ✅ (Course Structure)

**Purpose:** Get course module structure.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/modules
```

**Verified Result:** 4 modules (Inicio, Unidad 1, Unidad 2, etc.)

---

### 14. ENROLLMENT TERMS API ✅ (Academic Periods)

**Purpose:** Get all enrollment terms/semesters.

**Endpoint:**
```
GET /api/v1/accounts/{account_id}/terms
```

**Example Request:**
```python
response = requests.get(
    f'{API_URL}/api/v1/accounts/1/terms',
    headers=headers,
    params={'per_page': 50}
)

terms = response.json().get('enrollment_terms', [])
for t in terms:
    print(f"{t['id']}: {t['name']}")
```

**Verified Result:** 20 terms available.

---

### 15. GRAPHQL API ✅ (Alternative Query Method)

**Purpose:** Flexible queries combining multiple data types.

**Endpoint:**
```
POST /api/graphql
```

**Example Request:**
```python
query = """
query {
  course(id: "86005") {
    name
    enrollmentsConnection(first: 10) {
      nodes {
        user {
          _id
          name
        }
        grades {
          currentScore
          finalScore
        }
      }
    }
  }
}
"""

response = requests.post(
    f'{API_URL}/api/graphql',
    headers=headers,
    json={'query': query}
)

data = response.json()
enrollments = data['data']['course']['enrollmentsConnection']['nodes']
for e in enrollments:
    print(f"{e['user']['name']}: {e['grades']}")
```

**Verified Result:** Works! Returns grades with `currentScore` and `finalScore`.

---

### 16. USERS IN COURSE API ✅ (Alternative to Enrollments)

**Purpose:** List users with optional enrollment data.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/users
```

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `enrollment_type[]` | `student` | Filter to students |
| `include[]` | `enrollments` | Include enrollment details |

**Note:** This is an alternative way to get enrollment data. Use Enrollments API for direct grade access.

---

### 17. DEPARTMENT ANALYTICS APIs ✅ (NEW - Account-Level Aggregates)

**Purpose:** Get aggregated analytics at the account/program level.

#### 17a. Department Activity by Category

**Endpoint:**
```
GET /api/v1/accounts/{account_id}/analytics/terms/{term_id}/activity
GET /api/v1/accounts/{account_id}/analytics/current/activity
GET /api/v1/accounts/{account_id}/analytics/completed/activity
```

**Returns:**
```json
{
  "by_date": [{"date": "2025-11-24", "views": 2709, "participations": 19}],
  "by_category": [
    {"category": "announcements", "views": 4559},
    {"category": "assignments", "views": 6273},
    {"category": "discussions", "views": 24549},
    {"category": "files", "views": 15450},
    {"category": "grades", "views": 515},
    {"category": "modules", "views": 5587},
    {"category": "pages", "views": 2555},
    {"category": "quizzes", "views": 6342}
  ]
}
```

**Use Case:** Understand HOW students engage (content vs assessments vs grade-checking).

#### 17b. Department Grade Distribution

**Endpoint:**
```
GET /api/v1/accounts/{account_id}/analytics/terms/{term_id}/grades
```

**Returns:** Grade distribution binned 0-100:
```json
{"0": 834, "57": 12, "70": 45, "85": 89, "100": 23}
```

**Use Case:** Quick overview of grade distribution across all courses in an account.

#### 17c. Department Statistics

**Endpoint:**
```
GET /api/v1/accounts/{account_id}/analytics/terms/{term_id}/statistics
```

**Returns:**
```json
{
  "courses": 29,
  "teachers": 29,
  "students": 159,
  "discussion_topics": 2720,
  "attachments": 3272,
  "assignments": 397
}
```

---

### 18. COURSE ASSIGNMENTS ANALYTICS ✅ (NEW - Assignment Statistics)

**Purpose:** Get per-assignment statistics with quartiles and tardiness breakdown.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/analytics/assignments
```

**Parameters:**
| Parameter | Value | Description |
|-----------|-------|-------------|
| `async` | `true/false` | Enable async processing for large courses |

**Returns:**
```json
{
  "assignment_id": 475337,
  "title": "TAREA 01",
  "due_at": "2025-08-22T03:59:59Z",
  "points_possible": 80.0,
  "max_score": 80.0,
  "min_score": 45.0,
  "first_quartile": 65.0,
  "median": 72.0,
  "third_quartile": 78.0,
  "tardiness_breakdown": {
    "missing": 0.425,
    "late": 0.0,
    "on_time": 0.575,
    "total": 40
  }
}
```

**Use Case:** More efficient than fetching all submissions - gives statistical summary directly.

---

### 19. USER ASSIGNMENTS ANALYTICS ✅ (NEW - Per-Student Assignment Status)

**Purpose:** Get assignment data for a specific student with submission status.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/analytics/users/{student_id}/assignments
```

**Returns:**
```json
{
  "assignment_id": 481087,
  "title": "TAREA 02",
  "points_possible": 80.0,
  "due_at": "2025-09-26T02:59:00Z",
  "status": "on_time",
  "excused": false,
  "submission": {
    "posted_at": "2025-10-01T20:09:10Z",
    "score": 79.0,
    "submitted_at": "2025-09-26T02:23:50Z"
  }
}
```

**Use Case:** Track individual student progress and submission patterns.

---

### 20. RECENT STUDENTS API ✅ (NEW - Activity Recency)

**Purpose:** Get students ordered by last login time.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/recent_students
```

**Returns:**
```json
{
  "id": 117656,
  "name": "Student Name",
  "last_login": "2025-12-18T17:08:47Z"
}
```

**Use Case:** Quickly identify inactive students for early intervention.

---

### 21. USER COMMUNICATION API ✅ (Tested - Limited Data)

**Purpose:** Get message counts between instructor and student.

**Endpoint:**
```
GET /api/v1/courses/{course_id}/analytics/users/{student_id}/communication
```

**Note:** Returns empty `{}` if no messaging occurred. Data availability depends on course communication patterns.

---

### 22. USER/BULK PROGRESS API ⚠️ (Requires Module Completion)

**Purpose:** Get module completion progress for students.

**Endpoints:**
```
GET /api/v1/courses/{course_id}/users/{user_id}/progress
GET /api/v1/courses/{course_id}/bulk_user_progress
```

**Limitation:** Only works for courses with module completion requirements enabled. Returns error otherwise:
```json
{"error": {"message": "no progress available because this course is not module based"}}
```

---

## APIs with Limitations

| API | Limitation | Workaround |
|-----|------------|------------|
| Page Views | No `course_id` filter parameter | Filter post-fetch by parsing URL |
| External Tool (LTI) "Libro de Calificaciones" | Grades stored externally on `ltigb.entornosdeformacion.com` | Use Canvas native grades instead |
| Custom Gradebook Columns | No data in test courses | Check if used in production |
| Courses Listing | Some courses not visible in `/courses` | Access directly by course ID |

---

## Data Summary

### What We Can Extract

| Data Type | API | Fields | Status |
|-----------|-----|--------|--------|
| **Course grades** | Enrollments | `current_score`, `final_score`, `current_grade`, `final_grade` | ✅ Verified |
| **Assignment grades** | Submissions | `score`, `grade`, `submitted_at`, `graded_at` | ✅ Verified |
| **Assignment metadata** | Assignments | `name`, `points_possible`, `due_at`, `grading_type` | ✅ Verified |
| **Assignment statistics** | Course Assignments Analytics | `min_score`, `max_score`, `median`, `quartiles`, `tardiness_breakdown` | ✅ NEW |
| **Grade weights** | Assignment Groups | `group_weight`, `name` | ✅ Verified |
| **Activity metrics** | Student Summaries | `page_views`, `participations`, `tardiness_breakdown` | ✅ Verified |
| **Activity by category** | Department Activity | `by_category` (announcements, assignments, files, etc.) | ✅ NEW |
| **Daily activity** | Course Activity | `date`, `views`, `participations` | ✅ Verified |
| **User hourly activity** | User Activity | Page views by hour | ✅ Verified |
| **Grade history** | Gradebook History | `previous_grade`, `new_grade`, timestamps | ✅ Verified |
| **Clickstream** | Page Views | `url`, `action`, `interaction_seconds` | ✅ Verified |
| **Quiz data** | Quizzes | `title`, `points_possible`, quiz metadata | ✅ Verified |
| **Course structure** | Modules | `name`, module items | ✅ Verified |
| **Academic terms** | Enrollment Terms | `name`, `start_at`, `end_at` | ✅ Verified |
| **GraphQL grades** | GraphQL | `currentScore`, `finalScore` via query | ✅ Verified |
| **Department grades** | Department Analytics | Grade distribution binned 0-100 | ✅ NEW |
| **Department stats** | Department Statistics | `courses`, `students`, `teachers`, `assignments` counts | ✅ NEW |
| **Recent logins** | Recent Students | `last_login` per student | ✅ NEW |
| **User assignment status** | User Assignments Analytics | `status` (on_time/late/missing), `submission` | ✅ NEW |

### Sample Data Available (Course 86005)

```
Students:           50
With grades:        48
With activity:      51
Assignments:        17 (7 exams + 10 formative)
Submissions:        3000+
Scores available:   1460+

Grade range:        18.5% - 60.6% (final_score)
Failures (<57%):    47 students
```

---

## Prediction Model Feasibility

### Target Variable
- **Primary:** `final_score` from Enrollments API
- **Alternative:** First exam score from Submissions API

### Feature Sources
1. **Activity features:** `page_views`, `participations`, `tardiness_breakdown`
2. **Early grades:** Formative assessment scores
3. **Engagement patterns:** Page views over time (from Page Views API)

### Model Approach
1. **Regression:** Predict final grade percentage
2. **Classification:** Convert to PASS/FAIL at 57% threshold
3. **Early warning:** Predict first exam from pre-exam activity

---

## Prediction Models (Implemented)

### Script Location
`/home/paul/projects/uautonoma/scripts/prediction_models.py`

### Two Model Types

**1. ALL-DATA Model** - Uses all available features including grades/scores:
```python
ALL_DATA_FEATURES = [
    'page_views', 'participations', 'total_activity_time',
    'page_views_level', 'participations_level',
    'on_time', 'late', 'missing', 'floating',
    'on_time_rate', 'late_rate', 'missing_rate',
    'num_submissions', 'submission_rate',
    'avg_score', 'min_score', 'max_score', 'score_std',
    'first_score', 'num_graded', 'num_scores'
]
```

**2. ACTIVITY-ONLY Model** - Uses only pure activity features (NO grades/submissions):
```python
ACTIVITY_ONLY_FEATURES = [
    'page_views',           # Total page views in course
    'participations',       # Total participations count
    'total_activity_time',  # Total time spent in course
    'page_views_level',     # Canvas-computed engagement level (1-3)
    'participations_level', # Canvas-computed participation level (1-3)
]
```

**Why Activity-Only?** Enables early warning BEFORE any grades exist. Submission-based features (`on_time_rate`, `missing_rate`) are excluded because they track assignment interactions, not pure engagement.

### Model Results Summary (6 courses, 258 students)

| Metric | All-Data Model | Activity-Only Model |
|--------|----------------|---------------------|
| **Avg R² (Regression)** | 0.756 | 0.491 |
| **Avg F1 (Classification)** | 1.000 | 0.933 |

### Key Findings

1. **Activity-only prediction works!** R² = 0.49 means ~49% of grade variance explained by pure activity
2. **Classification is strong**: F1 = 0.93 for pass/fail using only activity features
3. **Top Activity Predictors (Random Forest importance)**:
   - `participations_level`: 0.36 (strongest)
   - `participations`: 0.35
   - `total_activity_time`: 0.15
   - `page_views`: 0.13

### Per-Course Results

| Course | Students | Pass Rate | All-Data R² | Activity-Only R² | Activity F1 |
|--------|----------|-----------|-------------|------------------|-------------|
| FUNDAMENTOS DE MICROECONOMÍA-P03 | 41 | 73% | 0.999 | 0.977 | 1.00 |
| FUNDAMENTOS DE MICROECONOMÍA-P01 | 36 | 39% | 0.999 | 0.993 | 1.00 |
| TALLER PENSAMIENTO ANALÍTICO-P01 | 40 | 23% | 0.998 | 0.751 | 0.80 |
| FORMACIÓN INTEGRAL II-P01 | 51 | 0% | 0.704 | 0.422 | N/A* |
| TALL DE COMPETENCIAS DIGITALES-P01 | 50 | 2% | 0.367 | 0.323 | N/A* |
| INTROD A LA ING EN CONTROL DE GEST-P01 | 40 | 0% | 0.945 | -0.11 | N/A* |

*N/A = Insufficient class diversity (all/most students fail)

### Results File
`/home/paul/projects/uautonoma/data/prediction_models_results.json`

---

## Code Examples

### Complete Data Extraction Script

```python
import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

def get_course_data(course_id):
    """Extract all data for a course"""
    data = {'course_id': course_id}

    # 1. Enrollments (grades)
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/enrollments',
                     headers=headers,
                     params={'type[]': 'StudentEnrollment', 'per_page': 100, 'include[]': 'grades'})
    data['enrollments'] = r.json() if r.status_code == 200 else []

    # 2. Assignments
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/assignments',
                     headers=headers, params={'per_page': 100})
    data['assignments'] = r.json() if r.status_code == 200 else []

    # 3. Submissions (paginated)
    submissions = []
    page = 1
    while True:
        r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/students/submissions',
                        headers=headers,
                        params={'student_ids[]': 'all', 'per_page': 100, 'page': page})
        if r.status_code != 200 or not r.json():
            break
        submissions.extend(r.json())
        if len(r.json()) < 100:
            break
        page += 1
    data['submissions'] = submissions

    # 4. Student summaries (activity)
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries',
                     headers=headers, params={'per_page': 100})
    data['student_summaries'] = r.json() if r.status_code == 200 else []

    return data

# Extract data
course_data = get_course_data(86005)
print(f"Enrollments: {len(course_data['enrollments'])}")
print(f"Assignments: {len(course_data['assignments'])}")
print(f"Submissions: {len(course_data['submissions'])}")
print(f"Activity: {len(course_data['student_summaries'])}")
```

---

## Important Notes

1. **Course Access:** Some courses are accessible via direct ID but don't appear in `/courses` listing
2. **Pagination:** Always paginate - max 100 records per request
3. **Rate Limiting:** Canvas has rate limits - implement delays for bulk extraction
4. **Grade Scales:** Different courses may use different scales (1-7 Chilean vs 0-100%)
5. **Test vs Production:** Current credentials are for TEST environment

---

## Next Steps

### Completed ✅
1. [x] Extract complete data from Control de Gestión courses
2. [x] Build feature matrix combining grades + activity
3. [x] Train prediction models (regression + classification)
4. [x] Compare all-data vs activity-only models
5. [x] Validate early warning capability (Activity-only F1 = 0.93!)

### Next Actions
1. [ ] Expand to POSTGRADO courses (1000+ available) for more training data
2. [ ] Test models on held-out courses for true generalization
3. [ ] Build temporal features (activity by week) for time-series prediction
4. [ ] Create production pipeline for real-time early warning
5. [ ] Investigate why some courses have poor activity-only prediction (low variance?)

### Data Quality Notes
- 25/31 Control de Gestión courses lacked sufficient grade data
- Many courses have 0% or 100% pass rates (no class diversity)
- Best results come from courses with 20-80% pass rates

---

*Last updated: December 2025*
*Environment: TEST (uautonoma.test.instructure.com)*
