# Canvas API Reference

Quick reference for Canvas LMS endpoints used in this project.

**Base URL:** Stored in `.env` as `CANVAS_API_URL`
**Auth:** `Authorization: Bearer {CANVAS_API_TOKEN}`

---

## Accounts

### Get Account Details
```
GET /api/v1/accounts/{account_id}
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `account_id` | Yes | Account/career ID |

**Response:**
```json
{
  "id": 248,
  "name": "Ingeniería Civil Informática",
  "parent_account_id": 176
}
```

**Used in:** `extract_career_data.py:69` - Get career name

---

### Get Account Courses
```
GET /api/v1/accounts/{account_id}/courses
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `account_id` | Yes | Account/career ID |
| `per_page` | No | Results per page (max 100) |
| `enrollment_term_id` | No | Filter by term |
| `with_enrollments` | No | Only courses with enrollments |
| `include[]` | No | `total_students`, `term` |

**Response:**
```json
{
  "id": 86005,
  "name": "TALL DE COMPETENCIAS DIGITALES-P01",
  "account_id": 719,
  "enrollment_term_id": 336,
  "total_students": 50,
  "term": {
    "id": 336,
    "name": "2do Semestre 2025"
  }
}
```

**Used in:** `extract_career_data.py:92` - List courses for a career

---

## Courses

### Get Enrollments (Grades)
```
GET /api/v1/courses/{course_id}/enrollments
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `course_id` | Yes | Course ID |
| `type[]` | No | `StudentEnrollment`, `TeacherEnrollment` |
| `per_page` | No | Results per page (max 100) |
| `include[]` | No | `grades`, `total_scores` |

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

| Grade Field | Description |
|-------------|-------------|
| `current_score` | Running average (completed work only) |
| `final_score` | Includes zeros for missing work |

**Used in:** `extract_career_data.py:135` - Get student grades for pass/fail calculation

---

### Get Assignments
```
GET /api/v1/courses/{course_id}/assignments
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `course_id` | Yes | Course ID |
| `per_page` | No | Results per page (max 100) |

**Response:**
```json
{
  "id": 465607,
  "name": "Evaluación sumativa semana 3",
  "points_possible": 100.0,
  "due_at": "2025-11-06T02:59:59Z",
  "grading_type": "points"
}
```

**Used in:** `extract_career_data.py:158` - Count assignments per course

---

### Get Modules
```
GET /api/v1/courses/{course_id}/modules
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `course_id` | Yes | Course ID |
| `per_page` | No | Results per page (max 100) |

**Response:**
```json
{
  "id": 12345,
  "name": "Unidad 1",
  "position": 1,
  "items_count": 5
}
```

**Used in:** `extract_career_data.py:166` - Count modules per course

---

### Get Student Summaries (Activity)
```
GET /api/v1/courses/{course_id}/analytics/student_summaries
```

| Parameter | Required | Description |
|-----------|----------|-------------|
| `course_id` | Yes | Course ID |
| `per_page` | No | Results per page (max 100) |

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

| Field | Description |
|-------|-------------|
| `page_views` | Total page views in course |
| `participations` | Total participations (submissions, discussions) |
| `page_views_level` | Canvas engagement level (1-3) |
| `tardiness_breakdown` | On-time vs late vs missing submissions |

**Used in:** `extract_career_data.py:174` - Check activity data exists

---

## Documented but Not Yet Used

These endpoints are documented in `docs/data_access_discovery.md`:

| Endpoint | Purpose |
|----------|---------|
| `GET /users/{id}/page_views` | Detailed clickstream |
| `GET /courses/{id}/students/submissions` | Per-assignment scores |
| `GET /courses/{id}/gradebook_history/feed` | Grade changes over time |
| `GET /courses/{id}/analytics/activity` | Daily course activity |

---

## Pagination

Canvas uses two pagination styles:

**Page-based:** `?page=1&per_page=100`
**Bookmark-based:** Follow `Link` header with `rel="next"`

```python
# Bookmark pagination example
link_header = response.headers.get('Link', '')
match = re.search(r'<([^>]+)>; rel="next"', link_header)
next_url = match.group(1) if match else None
```

---

## Rate Limiting

Check `X-Rate-Limit-Remaining` header. If < 100, slow down requests.

```python
remaining = int(response.headers.get('X-Rate-Limit-Remaining', 700))
if remaining < 100:
    time.sleep(10)
```

---

*Add new endpoints following the format above.*
