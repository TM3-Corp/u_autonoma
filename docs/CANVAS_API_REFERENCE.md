# Canvas LMS API Reference

**Universidad Autónoma de Chile - Early Warning System Project**

*Last updated: December 2025*

---

## Quick Start

### Setup

1. Copy `.env.example` to `.env`
2. Add your Canvas API credentials

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}
```

### Environment

- **Current:** TEST (`uautonoma.test.instructure.com`)
- **Pagination:** Max 100 records per request
- **Rate Limiting:** Implement delays for bulk extraction

---

## Account Hierarchy

```
Universidad Autónoma de Chile (UA) [Account ID: 1]
├── PREGRADO [Account ID: 46]
│   └── Providencia [Account ID: 176] - 43 careers, 3,393 courses
│       ├── Ing. en Control de Gestión [Account ID: 719] - 97 courses
│       ├── Ing. en Control de Gestión Alt [Account ID: 718] - 20 courses
│       └── Ingeniería Civil Industrial [Account ID: 730] - 18 courses
│
└── POSTGRADO [Account ID: 42] - 66 sub-accounts, 1000+ courses
    ├── Campus Virtual [401] - 29 courses
    ├── Providencia [402] - 97 courses
    ├── Temuco [405] - 79 courses
    └── ... 50+ more programs
```

---

## API Endpoints Reference

### 1. ENROLLMENTS API - Get Student Grades

**Purpose:** Get aggregate course grades for all students.

```
GET /api/v1/courses/{course_id}/enrollments
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `type[]` | `StudentEnrollment` | Filter to students only |
| `per_page` | `100` | Pagination (max 100) |
| `include[]` | `grades` | Include grade data |
| `include[]` | `total_scores` | Include total scores |

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/enrollments',
    headers=headers,
    params={
        'type[]': 'StudentEnrollment',
        'per_page': 100,
        'include[]': ['grades', 'total_scores']
    }
)

for e in response.json():
    grades = e.get('grades', {})
    print(f"User {e['user_id']}: current={grades.get('current_score')}, final={grades.get('final_score')}")
```

**Response:**
```json
{
  "user_id": 117656,
  "course_id": 86005,
  "enrollment_state": "active",
  "grades": {
    "current_score": 79.07,
    "final_score": 46.65,
    "current_grade": null,
    "final_grade": null
  }
}
```

**Key Fields:**
- `current_score`: Running grade (completed work only)
- `final_score`: Final grade (including zeros for missing work)

---

### 2. SUBMISSIONS API - Per-Assignment Grades

**Purpose:** Get individual assignment scores for each student.

```
GET /api/v1/courses/{course_id}/students/submissions
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `student_ids[]` | `all` | Get all students |
| `per_page` | `100` | Pagination |
| `include[]` | `assignment` | Include assignment details |

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/students/submissions',
    headers=headers,
    params={'student_ids[]': 'all', 'per_page': 100}
)

for s in response.json():
    if s.get('score') is not None:
        print(f"User {s['user_id']}, Assignment {s['assignment_id']}: {s['score']}")
```

**Response:**
```json
{
  "user_id": 88268,
  "assignment_id": 465607,
  "score": 92.0,
  "grade": "92",
  "submitted_at": "2025-09-15T14:30:00Z",
  "graded_at": "2025-09-16T10:00:00Z",
  "workflow_state": "graded"
}
```

**workflow_state values:** `graded`, `submitted`, `unsubmitted`, `pending_review`

---

### 3. ASSIGNMENTS API - Assignment Metadata

**Purpose:** Get all assignments with due dates and points.

```
GET /api/v1/courses/{course_id}/assignments
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/assignments',
    headers=headers,
    params={'per_page': 50, 'order_by': 'due_at'}
)

for a in response.json():
    print(f"{a['id']}: {a['name']} | {a['points_possible']}pts | Due: {a['due_at']}")
```

**Response:**
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

**Assignment Types at UA:**
- `Evaluación formativa` = Formative (quizzes, low-stakes)
- `Evaluación sumativa` = Summative (exams, high-stakes)

---

### 4. ASSIGNMENT GROUPS API - Grade Weights

**Purpose:** Get assignment categories and their weight in final grade.

```
GET /api/v1/courses/{course_id}/assignment_groups
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/assignment_groups',
    headers=headers
)

for g in response.json():
    print(f"{g['id']}: {g['name']} ({g['group_weight']}%)")
```

**Sample Output:**
```
Semana 3:  6% weight
Semana 6:  6% weight
Semana 8: 35% weight  <-- Main exam
Semana 10: 6% weight
```

---

### 5. STUDENT SUMMARIES API - Activity Metrics

**Purpose:** Get engagement metrics (page views, participations, tardiness).

```
GET /api/v1/courses/{course_id}/analytics/student_summaries
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/student_summaries',
    headers=headers,
    params={'per_page': 100}
)

for s in response.json():
    print(f"User {s['id']}: views={s['page_views']}, participations={s['participations']}")
    tb = s.get('tardiness_breakdown', {})
    print(f"  on_time={tb.get('on_time')}, late={tb.get('late')}, missing={tb.get('missing')}")
```

**Response:**
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

**Key Fields:**
- `page_views_level`: Canvas engagement level (1-3)
- `participations_level`: Canvas participation level (1-3)
- `tardiness_breakdown`: Assignment submission timing

---

### 6. COURSE ACTIVITY API - Daily Aggregates

**Purpose:** Get daily page views and participations for entire course.

```
GET /api/v1/courses/{course_id}/analytics/activity
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/activity',
    headers=headers
)

# Returns list of daily activity
for day in response.json():
    print(f"{day['date']}: views={day['views']}, participations={day['participations']}")
```

---

### 7. USER ACTIVITY API - Per-Student Hourly Detail

**Purpose:** Get hourly page views breakdown for specific student.

```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/users/117656/activity',
    headers=headers
)

data = response.json()
# data['page_views'] = dict with hourly timestamps
# data['participations'] = list of participation events
```

**Use Case:** Calculate time-of-day patterns, activity gaps, session duration.

---

### 8. PAGE VIEWS API - Detailed Clickstream

**Purpose:** Get detailed user activity (URLs visited, time spent).

```
GET /api/v1/users/{user_id}/page_views
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `start_time` | ISO timestamp | Start of time range |
| `end_time` | ISO timestamp | End of time range |
| `per_page` | `100` | Pagination |

**Important:** No `course_id` filter - must filter post-fetch by parsing URLs.

**Example:**
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

# Filter by course
import re
def extract_course_id(url):
    match = re.search(r"/courses/(\d+)", url)
    return int(match.group(1)) if match else -1

course_views = [pv for pv in response.json() if extract_course_id(pv['url']) == 86005]
```

**Response:**
```json
{
  "url": "/courses/86005/assignments/465607",
  "action": "assignments",
  "interaction_seconds": 45,
  "created_at": "2025-10-15T14:30:00Z"
}
```

---

### 9. GRADEBOOK HISTORY API - Grade Changes

**Purpose:** Track historical grade changes with timestamps.

```
GET /api/v1/courses/{course_id}/gradebook_history/days
GET /api/v1/courses/{course_id}/gradebook_history/feed
```

**Example:**
```python
# Get detailed feed
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/gradebook_history/feed',
    headers=headers,
    params={'per_page': 50}
)

for entry in response.json():
    print(f"User {entry['user_id']}: {entry['previous_grade']} -> {entry['grade']}")
```

---

### 10. QUIZZES API - Quiz Metadata

**Purpose:** Get quiz details (separate from assignments).

```
GET /api/v1/courses/{course_id}/quizzes
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/quizzes',
    headers=headers,
    params={'per_page': 50}
)

for q in response.json():
    print(f"{q['id']}: {q['title']} ({q['points_possible']} pts)")
```

---

### 11. MODULES API - Course Structure

**Purpose:** Get course module structure.

```
GET /api/v1/courses/{course_id}/modules
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/modules',
    headers=headers
)

for m in response.json():
    print(f"{m['id']}: {m['name']} ({m['items_count']} items)")
```

---

### 12. ENROLLMENT TERMS API - Academic Periods

**Purpose:** Get all enrollment terms/semesters.

```
GET /api/v1/accounts/{account_id}/terms
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/accounts/1/terms',
    headers=headers,
    params={'per_page': 50}
)

for t in response.json().get('enrollment_terms', []):
    print(f"{t['id']}: {t['name']}")
```

**Key Terms:**
- `322` = 1st Semester 2025
- `336` = 2nd Semester 2025
- `352` = Another term

---

### 13. COURSE ASSIGNMENTS ANALYTICS - Assignment Statistics

**Purpose:** Get per-assignment statistics with quartiles and tardiness.

```
GET /api/v1/courses/{course_id}/analytics/assignments
```

**Example:**
```python
response = requests.get(
    f'{API_URL}/api/v1/courses/86005/analytics/assignments',
    headers=headers
)

for a in response.json():
    print(f"{a['title']}: median={a.get('median')}, missing={a['tardiness_breakdown']['missing']:.0%}")
```

**Response:**
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

### 14. USER ASSIGNMENTS ANALYTICS - Per-Student Status

**Purpose:** Get assignment data for a specific student with submission status.

```
GET /api/v1/courses/{course_id}/analytics/users/{student_id}/assignments
```

**Response:**
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

**status values:** `on_time`, `late`, `missing`

---

### 15. RECENT STUDENTS API - Activity Recency

**Purpose:** Get students ordered by last login time.

```
GET /api/v1/courses/{course_id}/recent_students
```

**Response:**
```json
{
  "id": 117656,
  "name": "Student Name",
  "last_login": "2025-12-18T17:08:47Z"
}
```

**Use Case:** Quickly identify inactive students for early intervention.

---

### 16. DEPARTMENT ANALYTICS APIs - Account-Level Aggregates

**Purpose:** Get aggregated analytics at the account/program level.

#### Activity by Category
```
GET /api/v1/accounts/{account_id}/analytics/terms/{term_id}/activity
GET /api/v1/accounts/{account_id}/analytics/current/activity
```

**Response:**
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

#### Grade Distribution
```
GET /api/v1/accounts/{account_id}/analytics/terms/{term_id}/grades
```

**Response:**
```json
{"0": 834, "57": 12, "70": 45, "85": 89, "100": 23}
```

#### Department Statistics
```
GET /api/v1/accounts/{account_id}/analytics/terms/{term_id}/statistics
```

**Response:**
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

### 17. GRAPHQL API - Flexible Queries

**Purpose:** Combine multiple data types in a single query.

```
POST /api/graphql
```

**Example:**
```python
query = """
query {
  course(id: "86005") {
    name
    enrollmentsConnection(first: 10) {
      nodes {
        user { _id, name }
        grades { currentScore, finalScore }
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
for e in data['data']['course']['enrollmentsConnection']['nodes']:
    print(f"{e['user']['name']}: {e['grades']}")
```

---

## Data Summary Table

| Data Type | API | Key Fields |
|-----------|-----|------------|
| Course grades | Enrollments | `current_score`, `final_score` |
| Assignment grades | Submissions | `score`, `submitted_at`, `graded_at` |
| Assignment metadata | Assignments | `name`, `points_possible`, `due_at` |
| Assignment statistics | Course Assignments Analytics | `min`, `max`, `median`, `quartiles` |
| Grade weights | Assignment Groups | `group_weight` |
| Activity metrics | Student Summaries | `page_views`, `participations`, `tardiness_breakdown` |
| Activity by category | Department Activity | `by_category` breakdown |
| Daily activity | Course Activity | `date`, `views`, `participations` |
| Hourly activity | User Activity | Page views by hour |
| Grade history | Gradebook History | `previous_grade`, `new_grade` |
| Clickstream | Page Views | `url`, `action`, `interaction_seconds` |
| Quiz data | Quizzes | `title`, `points_possible` |
| Course structure | Modules | `name`, module items |
| Recent logins | Recent Students | `last_login` |

---

## Complete Data Extraction Script

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}


def get_course_data(course_id):
    """Extract all data for a course."""
    data = {'course_id': course_id}

    # 1. Enrollments (grades)
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 100, 'include[]': 'grades'}
    )
    data['enrollments'] = r.json() if r.status_code == 200 else []

    # 2. Assignments
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/assignments',
        headers=headers,
        params={'per_page': 100}
    )
    data['assignments'] = r.json() if r.status_code == 200 else []

    # 3. Submissions (paginated)
    submissions = []
    page = 1
    while True:
        r = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/students/submissions',
            headers=headers,
            params={'student_ids[]': 'all', 'per_page': 100, 'page': page}
        )
        if r.status_code != 200 or not r.json():
            break
        submissions.extend(r.json())
        if len(r.json()) < 100:
            break
        page += 1
    data['submissions'] = submissions

    # 4. Student summaries (activity)
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries',
        headers=headers,
        params={'per_page': 100}
    )
    data['student_summaries'] = r.json() if r.status_code == 200 else []

    # 5. Assignment analytics (statistics)
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/assignments',
        headers=headers
    )
    data['assignment_analytics'] = r.json() if r.status_code == 200 else []

    return data


# Usage
if __name__ == '__main__':
    course_data = get_course_data(86005)
    print(f"Enrollments: {len(course_data['enrollments'])}")
    print(f"Assignments: {len(course_data['assignments'])}")
    print(f"Submissions: {len(course_data['submissions'])}")
    print(f"Activity summaries: {len(course_data['student_summaries'])}")
    print(f"Assignment analytics: {len(course_data['assignment_analytics'])}")
```

---

## Known Limitations

| API | Limitation | Workaround |
|-----|------------|------------|
| Page Views | No `course_id` filter | Filter post-fetch by parsing URL |
| "Libro de Calificaciones" | External LTI tool, grades not accessible | Use Canvas native grades |
| User/Bulk Progress | Requires module completion enabled | Check if course is module-based |
| User Communication | Returns empty if no messaging | N/A |

---

## Test Courses (Verified)

| Course ID | Name | Students | Has Grades |
|-----------|------|----------|------------|
| 86005 | TALL DE COMPETENCIAS DIGITALES-P01 | 50 | Yes |
| 86020 | TALL DE COMPETENCIAS DIGITALES-P02 | 47 | Yes |
| 84944 | FUND MACROECONOMIA-P03 | 38 | Yes |
| 86676 | FUND BUSINESS ANALYTICS-P01 | 36 | Yes |
| 84941 | FUND MICROECONOMIA-P01 | 15 | Yes |

---

## Important Notes

1. **Chilean Grading:** Passing threshold is 57% (equivalent to 4.0 on 1-7 scale)
2. **Pagination:** Always paginate - max 100 records per request
3. **Rate Limiting:** Implement `time.sleep(0.3)` between bulk API calls
4. **Course Access:** Some courses accessible by ID but don't appear in `/courses` listing

---

*Document generated for U. Autónoma Early Warning System Project*
*December 2025*
