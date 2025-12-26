# Course Data Extractor

**Script:** `extract_course_data.py`

Extracts all available data from a Canvas LMS course and saves to Parquet files for analysis.

---

## Usage

```bash
# Basic extraction
python exploratory/discovery/extract_course_data.py --course-id 86676

# Include clickstream page views (slower, requires per-student API calls)
python exploratory/discovery/extract_course_data.py --course-id 86676 --include-page-views

# Custom date range for page views
python exploratory/discovery/extract_course_data.py --course-id 86676 --include-page-views \
    --start-date 2025-08-01 --end-date 2025-12-31

# Custom output directory
python exploratory/discovery/extract_course_data.py --course-id 86676 --output-dir data/courses
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--course-id` | int | required | Canvas course ID to extract |
| `--output-dir` | str | `exploratory/data/courses` | Output directory |
| `--start-date` | str | `2025-08-01` | Start date for page views (YYYY-MM-DD) |
| `--end-date` | str | `2025-12-31` | End date for page views (YYYY-MM-DD) |
| `--include-page-views` | flag | False | Extract detailed clickstream data |

---

## Output Structure

```
{output_dir}/course_{id}/
├── enrollments.parquet        # Student grades
├── assignments.parquet        # Assignment metadata
├── assignment_groups.parquet  # Grade weight categories
├── submissions.parquet        # Per-student submissions
├── student_summaries.parquet  # Activity metrics
├── modules.parquet            # Course structure
├── quizzes.parquet            # Quiz metadata
├── pages.parquet              # Content pages
├── files.parquet              # Course materials
├── discussion_topics.parquet  # Discussion forums
├── page_views.parquet         # Clickstream (optional)
└── course_info.parquet        # Course metadata
```

---

## Data Categories

The extracted data falls into two categories based on `course_analysis.md`:

### Instructional Design (Radiografía)
Data about course structure and content quality:
- `modules.parquet` - Course organization
- `assignments.parquet` - Learning activities
- `quizzes.parquet` - Assessments
- `pages.parquet` - Content pages
- `files.parquet` - Materials/resources
- `discussion_topics.parquet` - Interaction opportunities

### Prediction Potential
Data for building ML prediction models:
- `enrollments.parquet` - Student grades (target variable)
- `submissions.parquet` - Assignment completion patterns
- `student_summaries.parquet` - Engagement metrics
- `page_views.parquet` - Detailed clickstream (optional)

---

## Parquet File Schemas

### enrollments.parquet
**Purpose:** Student enrollment records with grades. Primary source for prediction target variable.

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Unique Canvas user identifier |
| `course_id` | int | Course ID |
| `enrollment_state` | str | Status: `active`, `invited`, `completed` |
| `created_at` | str | Enrollment creation timestamp |
| `updated_at` | str | Last update timestamp |
| `current_score` | float | Running grade (completed work only, 0-100) |
| `final_score` | float | Grade including zeros for missing work (0-100) |
| `current_grade` | str | Letter grade if configured (e.g., "A", "B+") |
| `final_grade` | str | Final letter grade if configured |
| `unposted_current_score` | float | Score including unposted grades |
| `unposted_final_score` | float | Final score including unposted grades |

**Key fields for prediction:**
- `final_score` - Target variable for failure prediction (< 57% = fail in Chilean scale)
- `current_score` - Early indicator of performance

---

### assignments.parquet
**Purpose:** Assignment metadata including due dates, points, and types.

| Column | Type | Description |
|--------|------|-------------|
| `assignment_id` | int | Unique assignment identifier |
| `course_id` | int | Course ID |
| `name` | str | Assignment name |
| `description` | str | Assignment description (truncated to 500 chars) |
| `points_possible` | float | Maximum points for the assignment |
| `due_at` | str | Due date (ISO 8601 timestamp) |
| `unlock_at` | str | When assignment becomes available |
| `lock_at` | str | When assignment closes |
| `grading_type` | str | `points`, `percent`, `letter_grade`, `pass_fail` |
| `submission_types` | str | Comma-separated: `online_text_entry`, `online_upload`, etc. |
| `assignment_group_id` | int | Links to assignment_groups for grade weights |
| `position` | int | Order in assignment list |
| `published` | bool | Whether assignment is visible to students |
| `has_submitted_submissions` | bool | Whether any student has submitted |
| `created_at` | str | Creation timestamp |
| `updated_at` | str | Last update timestamp |

---

### assignment_groups.parquet
**Purpose:** Grade weight categories (e.g., "Exams 40%", "Homework 30%").

| Column | Type | Description |
|--------|------|-------------|
| `group_id` | int | Unique group identifier |
| `course_id` | int | Course ID |
| `name` | str | Group name (e.g., "Tareas", "Evaluaciones") |
| `position` | int | Display order |
| `group_weight` | float | Percentage weight in final grade (0-100) |
| `rules` | str | Drop/never_drop rules as string |

---

### submissions.parquet
**Purpose:** Individual assignment submissions per student. Shows completion patterns and timing.

| Column | Type | Description |
|--------|------|-------------|
| `submission_id` | int | Unique submission identifier |
| `user_id` | int | Student ID |
| `assignment_id` | int | Assignment ID |
| `course_id` | int | Course ID |
| `score` | float | Points earned (null if not graded) |
| `grade` | str | Grade as string |
| `submitted_at` | str | Submission timestamp (null if not submitted) |
| `graded_at` | str | When graded (null if not graded) |
| `workflow_state` | str | `submitted`, `graded`, `unsubmitted`, `pending_review` |
| `late` | bool | Whether submitted after due date |
| `missing` | bool | Whether assignment is missing |
| `excused` | bool | Whether student is excused |
| `attempt` | int | Submission attempt number |
| `seconds_late` | int | Seconds past due date (0 if on time) |
| `grade_matches_current_submission` | bool | Grade reflects latest submission |

**Key fields for prediction:**
- `late`, `missing` - Early warning indicators
- `seconds_late` - Procrastination patterns
- `workflow_state` - Completion status

---

### student_summaries.parquet
**Purpose:** Aggregated activity metrics from Canvas Analytics. Primary engagement features for prediction.

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Student ID |
| `course_id` | int | Course ID |
| `page_views` | int | Total page views in the course |
| `page_views_level` | int | Canvas engagement level (1=low, 2=medium, 3=high) |
| `participations` | int | Total participations (submissions, discussions, etc.) |
| `participations_level` | int | Canvas participation level (0-3) |
| `on_time` | int | Assignments submitted on time |
| `late` | int | Assignments submitted late |
| `missing` | int | Assignments not submitted |
| `floating` | int | Assignments with no due date |

**Key fields for prediction:**
- `page_views` - Primary engagement indicator
- `participations` - Active learning indicator
- `missing` - Strong failure predictor
- `on_time` / `late` ratio - Time management indicator

---

### modules.parquet
**Purpose:** Course structure/organization.

| Column | Type | Description |
|--------|------|-------------|
| `module_id` | int | Unique module identifier |
| `course_id` | int | Course ID |
| `name` | str | Module name (e.g., "Unidad 1: Introducción") |
| `position` | int | Display order |
| `items_count` | int | Number of items in module |
| `state` | str | Module state |
| `unlock_at` | str | When module becomes available |
| `published` | bool | Whether module is visible |

---

### quizzes.parquet
**Purpose:** Quiz/assessment metadata.

| Column | Type | Description |
|--------|------|-------------|
| `quiz_id` | int | Unique quiz identifier |
| `course_id` | int | Course ID |
| `title` | str | Quiz title |
| `quiz_type` | str | `practice_quiz`, `assignment`, `graded_survey`, `survey` |
| `points_possible` | float | Total points |
| `time_limit` | int | Time limit in minutes (null if unlimited) |
| `question_count` | int | Number of questions |
| `due_at` | str | Due date |
| `unlock_at` | str | Available from |
| `lock_at` | str | Available until |
| `published` | bool | Whether visible to students |
| `allowed_attempts` | int | Number of attempts allowed (-1 = unlimited) |

---

### pages.parquet
**Purpose:** Content pages (instructional materials, resources).

| Column | Type | Description |
|--------|------|-------------|
| `page_id` | int | Unique page identifier |
| `course_id` | int | Course ID |
| `url` | str | Page URL slug |
| `title` | str | Page title |
| `created_at` | str | Creation timestamp |
| `updated_at` | str | Last update timestamp |
| `published` | bool | Whether visible to students |
| `front_page` | bool | Whether this is the course front page |
| `locked_for_user` | bool | Whether locked for current user |

---

### files.parquet
**Purpose:** Course files/materials (PDFs, slides, documents).

| Column | Type | Description |
|--------|------|-------------|
| `file_id` | int | Unique file identifier |
| `course_id` | int | Course ID |
| `folder_id` | int | Parent folder ID |
| `display_name` | str | File name shown to users |
| `filename` | str | Actual filename |
| `content_type` | str | MIME type (e.g., `application/pdf`) |
| `size` | int | File size in bytes |
| `created_at` | str | Upload timestamp |
| `updated_at` | str | Last update timestamp |
| `locked` | bool | Whether file is locked |
| `hidden` | bool | Whether file is hidden |

---

### discussion_topics.parquet
**Purpose:** Discussion forums and announcements.

| Column | Type | Description |
|--------|------|-------------|
| `topic_id` | int | Unique topic identifier |
| `course_id` | int | Course ID |
| `title` | str | Discussion title |
| `message` | str | Topic content (truncated to 500 chars) |
| `discussion_type` | str | `side_comment`, `threaded` |
| `posted_at` | str | When posted |
| `delayed_post_at` | str | Scheduled post time |
| `last_reply_at` | str | Most recent reply timestamp |
| `discussion_subentry_count` | int | Number of replies |
| `published` | bool | Whether visible |
| `locked` | bool | Whether closed for replies |
| `pinned` | bool | Whether pinned to top |
| `assignment_id` | int | Linked assignment (if graded discussion) |

---

### page_views.parquet (Optional)
**Purpose:** Detailed clickstream data. Requires `--include-page-views` flag.

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int | Student ID |
| `course_id` | int | Course ID |
| `url` | str | Page URL visited |
| `context_type` | str | `Course`, `User`, etc. |
| `asset_type` | str | `module`, `assignment`, `page`, `file`, etc. |
| `controller` | str | Canvas controller (e.g., `context_modules`) |
| `action` | str | Action performed (e.g., `show`, `index`) |
| `interaction_seconds` | float | Time spent on page |
| `created_at` | str | Timestamp of page view |
| `participated` | bool | Whether this was a participation event |
| `user_agent` | str | Browser/device info (truncated) |

**Note:** Page views extraction is slow (requires per-student API calls) and should only be used when detailed clickstream analysis is needed.

---

### course_info.parquet
**Purpose:** Course metadata and extraction summary.

| Column | Type | Description |
|--------|------|-------------|
| `course_id` | int | Course ID |
| `course_name` | str | Course name |
| `account_id` | int | Parent account (career) ID |
| `term_id` | int | Enrollment term ID |
| `term_name` | str | Term name (e.g., "Segundo Semestre 2025") |
| `total_students` | int | Number of enrolled students |
| `extraction_date` | str | When data was extracted |
| `start_date` | str | Page views start date |
| `end_date` | str | Page views end date |
| `n_enrollments` | int | Records extracted |
| `n_assignments` | int | Records extracted |
| `n_submissions` | int | Records extracted |
| `n_student_summaries` | int | Records extracted |
| `n_modules` | int | Records extracted |
| `n_quizzes` | int | Records extracted |
| `n_pages` | int | Records extracted |
| `n_files` | int | Records extracted |
| `n_discussion_topics` | int | Records extracted |
| `n_page_views` | int | Records extracted (if applicable) |

---

## Example: Loading Data in Python

```python
import pandas as pd

# Load all data for a course
course_id = 86676
base_path = f'exploratory/data/courses/course_{course_id}'

enrollments = pd.read_parquet(f'{base_path}/enrollments.parquet')
summaries = pd.read_parquet(f'{base_path}/student_summaries.parquet')
submissions = pd.read_parquet(f'{base_path}/submissions.parquet')

# Merge for analysis
df = enrollments.merge(summaries, on=['user_id', 'course_id'], how='left')

# Identify at-risk students (low engagement + missing assignments)
at_risk = df[
    (df['page_views'] < 500) &
    (df['missing'] > 3) &
    (df['final_score'] < 57)
]
print(f"At-risk students: {len(at_risk)}")
```

---

## API Endpoints Used

| Endpoint | Data |
|----------|------|
| `GET /api/v1/courses/{id}` | Course metadata |
| `GET /api/v1/courses/{id}/enrollments` | Student grades |
| `GET /api/v1/courses/{id}/assignments` | Assignment metadata |
| `GET /api/v1/courses/{id}/assignment_groups` | Grade weights |
| `GET /api/v1/courses/{id}/students/submissions` | Submission data |
| `GET /api/v1/courses/{id}/analytics/student_summaries` | Activity metrics |
| `GET /api/v1/courses/{id}/modules` | Course structure |
| `GET /api/v1/courses/{id}/quizzes` | Quiz metadata |
| `GET /api/v1/courses/{id}/pages` | Content pages |
| `GET /api/v1/courses/{id}/files` | Course files |
| `GET /api/v1/courses/{id}/discussion_topics` | Discussions |
| `GET /api/v1/users/{id}/page_views` | Clickstream (optional) |

---

## Rate Limiting

The script implements adaptive rate limiting based on `X-Rate-Limit-Remaining` header:

| Remaining Quota | Delay |
|-----------------|-------|
| < 50 | 30 seconds |
| 50-99 | 10 seconds |
| 100-199 | 5 seconds |
| 200-299 | 2 seconds |
| 300-499 | 1 second |
| 500+ | 0.3 seconds |

---

*Last updated: December 2025*
