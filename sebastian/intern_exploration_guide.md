# Canvas Data Exploration Guide for Interns

## Universidad Autónoma de Chile - LMS Analytics Project

**Objective:** Identify high-potential courses for early failure prediction analysis and extract detailed user activity data.

**Prerequisites:**
- Python 3.8+
- API credentials in `.env` file (copy from `.env.example`)
- Familiarity with Canvas LMS concepts

---

## Stage 1: Course Discovery

### Goal
Find courses with **high analytical potential** based on:
1. Active enrollment (current semester)
2. Available grade data with variance
3. Good LMS design (resources, activities, student engagement)

### Account Hierarchy

```
Universidad Autónoma de Chile [Account 1]
├── PREGRADO [Account 46] ← Undergraduate programs
│   ├── Providencia [176] - 43 careers, 3,393 courses
│   │   ├── Ing. Control de Gestión [719] - ALREADY ANALYZED ✓
│   │   ├── Ing. Control de Gestión Alt [718]
│   │   ├── Ingeniería Civil Industrial [730]
│   │   ├── Psicología [247]
│   │   ├── Derecho [253]
│   │   ├── Medicina [249]
│   │   └── ... (43 total careers)
│   ├── Temuco [173]
│   ├── Talca [174]
│   ├── San Miguel [175]
│   └── Campus Virtual [360]
│
└── POSTGRADO [Account 42] ← Graduate programs (1000+ active courses)
    ├── Campus Virtual [401] - 29 active courses
    ├── Providencia [402] - 97 active courses
    ├── Temuco [405] - 79 active courses
    ├── Talca [404] - 62 active courses
    ├── Magíster en Psicología Clínica [554] - 77 courses
    ├── Magíster en Dirección de Empresas [551] - 58 courses
    └── ... (66 sub-accounts total)
```

**Reference:** See `docs/data_access_discovery.md` for complete hierarchy and statistics.

### Key Enrollment Terms (2025)

| Term ID | Name | Notes |
|---------|------|-------|
| **336** | Segundo Semestre 2025 - Periodo Completo | Current semester (active) |
| **322** | Primer Semestre 2025 - Periodo Completo | Past semester (may have grades) |
| **352** | Segundo Semestre 2025 - Bimestral | Shorter period courses |

**Important:** Courses from past terms may not have grades accessible. Focus on terms 336 (current) and 322 (recent past).

---

### Step 1.1: List Sub-Accounts to Explore

```python
import requests
import os
from dotenv import load_dotenv
load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

def get_sub_accounts(account_id):
    """Get all sub-accounts for a parent account"""
    r = requests.get(
        f'{API_URL}/api/v1/accounts/{account_id}/sub_accounts',
        headers=headers,
        params={'per_page': 100}
    )
    return r.json() if r.status_code == 200 else []

# Pregrado sub-accounts (excluding Control de Gestión 719)
pregrado_subs = get_sub_accounts(46)
print("PREGRADO Sub-accounts:")
for acc in pregrado_subs:
    print(f"  {acc['id']}: {acc['name']}")

# Postgrado sub-accounts
postgrado_subs = get_sub_accounts(42)
print(f"\nPOSTGRADO Sub-accounts: {len(postgrado_subs)} programs")
for acc in postgrado_subs[:10]:
    print(f"  {acc['id']}: {acc['name']}")
```

### Step 1.2: Get Courses from an Account

```python
def get_courses_with_students(account_id, enrollment_term_id=None, min_students=10):
    """
    Get courses from an account with minimum student count.

    Args:
        account_id: Canvas account ID
        enrollment_term_id: Filter by term (336=2nd Sem 2025, 322=1st Sem 2025)
        min_students: Minimum enrolled students

    Returns:
        List of course dictionaries with student counts
    """
    params = {
        'per_page': 100,
        'include[]': ['total_students', 'term'],
        'with_enrollments': True
    }
    if enrollment_term_id:
        params['enrollment_term_id'] = enrollment_term_id

    courses = []
    url = f'{API_URL}/api/v1/accounts/{account_id}/courses'

    while url:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            break

        for course in r.json():
            if course.get('total_students', 0) >= min_students:
                courses.append({
                    'id': course['id'],
                    'name': course['name'],
                    'students': course.get('total_students', 0),
                    'term': course.get('term', {}).get('name', 'Unknown'),
                    'term_id': course.get('enrollment_term_id')
                })

        # Pagination
        url = r.links.get('next', {}).get('url')
        params = {}  # URL already contains params

    return sorted(courses, key=lambda x: x['students'], reverse=True)

# Example: Get courses from Providencia (176) for current semester
# EXCLUDE account 719 (Control de Gestión) - already analyzed
courses = get_courses_with_students(176, enrollment_term_id=336, min_students=20)
print(f"Found {len(courses)} courses with 20+ students")
for c in courses[:10]:
    print(f"  {c['id']}: {c['name'][:50]} ({c['students']} students)")
```

### Step 1.3: Check Course Quality (Grades + LMS Design)

```python
def analyze_course_potential(course_id):
    """
    Analyze a course for analytical potential.

    Returns dict with:
    - has_grades: Whether students have grades
    - grade_variance: Standard deviation of grades (higher = better for prediction)
    - n_students: Number of students with grades
    - n_resources: Number of modules/assignments (LMS design)
    - pass_rate: Percentage passing (< 57% threshold)
    """
    import numpy as np

    result = {
        'course_id': course_id,
        'has_grades': False,
        'grade_variance': 0,
        'n_students': 0,
        'n_assignments': 0,
        'n_modules': 0,
        'pass_rate': None,
        'recommendation': 'SKIP'
    }

    # 1. Get enrollments with grades
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 100, 'include[]': 'grades'}
    )
    if r.status_code != 200:
        return result

    enrollments = r.json()
    grades = [e['grades'].get('final_score') for e in enrollments
              if e.get('grades', {}).get('final_score') is not None]

    if len(grades) >= 10:
        result['has_grades'] = True
        result['n_students'] = len(grades)
        result['grade_variance'] = np.std(grades)
        result['pass_rate'] = sum(1 for g in grades if g >= 57) / len(grades)

    # 2. Count assignments (LMS design indicator)
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/assignments',
                     headers=headers, params={'per_page': 100})
    if r.status_code == 200:
        result['n_assignments'] = len(r.json())

    # 3. Count modules
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/modules',
                     headers=headers, params={'per_page': 100})
    if r.status_code == 200:
        result['n_modules'] = len(r.json())

    # 4. Recommendation logic
    if result['has_grades'] and result['grade_variance'] > 10:
        if result['n_assignments'] >= 5 and 0.2 <= result['pass_rate'] <= 0.8:
            result['recommendation'] = 'HIGH POTENTIAL'
        elif result['n_assignments'] >= 3:
            result['recommendation'] = 'MEDIUM POTENTIAL'
        else:
            result['recommendation'] = 'LOW - Few assignments'
    elif result['has_grades']:
        result['recommendation'] = 'LOW - Low grade variance'
    else:
        result['recommendation'] = 'SKIP - No grades'

    return result

# Example: Analyze a specific course
analysis = analyze_course_potential(86005)
print(f"Course {analysis['course_id']}: {analysis['recommendation']}")
print(f"  Students: {analysis['n_students']}, Grade StdDev: {analysis['grade_variance']:.1f}")
print(f"  Assignments: {analysis['n_assignments']}, Modules: {analysis['n_modules']}")
print(f"  Pass Rate: {analysis['pass_rate']:.1%}" if analysis['pass_rate'] else "  Pass Rate: N/A")
```

### Step 1.4: Bulk Scan for High-Potential Courses

```python
from tqdm import tqdm
import time

def scan_account_for_potential(account_id, term_id=336, exclude_accounts=None):
    """
    Scan all courses in an account for analytical potential.

    Args:
        account_id: Parent account to scan
        term_id: Enrollment term to filter
        exclude_accounts: List of sub-account IDs to skip (e.g., [719])
    """
    exclude_accounts = exclude_accounts or []

    # Get all courses
    courses = get_courses_with_students(account_id, term_id, min_students=15)
    print(f"Found {len(courses)} courses with 15+ students")

    results = []
    for course in tqdm(courses, desc="Analyzing courses"):
        # Skip excluded accounts (checked via course endpoint)
        analysis = analyze_course_potential(course['id'])
        analysis['course_name'] = course['name']
        results.append(analysis)
        time.sleep(0.5)  # Rate limiting

    # Sort by potential
    high_potential = [r for r in results if 'HIGH' in r['recommendation']]
    medium_potential = [r for r in results if 'MEDIUM' in r['recommendation']]

    print(f"\nHIGH POTENTIAL: {len(high_potential)} courses")
    for r in high_potential[:10]:
        print(f"  {r['course_id']}: {r['course_name'][:40]}")
        print(f"    Students={r['n_students']}, Variance={r['grade_variance']:.1f}, Pass={r['pass_rate']:.0%}")

    print(f"\nMEDIUM POTENTIAL: {len(medium_potential)} courses")
    for r in medium_potential[:5]:
        print(f"  {r['course_id']}: {r['course_name'][:40]}")

    return results

# Scan Providencia (176) excluding Control de Gestión (719)
# results = scan_account_for_potential(176, term_id=336, exclude_accounts=[719])
```

### Your Task: Stage 1 Deliverable

1. **Scan Pregrado** (Account 46, all sub-accounts except 719):
   - Focus on terms 336 (current) and 322 (recent past)
   - Identify top 5 high-potential courses
   - Document: course_id, name, students, grade_variance, pass_rate

2. **Scan Postgrado** (Account 42):
   - Start with larger programs (Providencia 402, Campus Virtual 401)
   - Identify top 5 high-potential courses
   - Document same metrics

3. **Select 2 courses for Stage 2:**
   - 1 from Pregrado (not Control de Gestión)
   - 1 from Postgrado
   - Criteria: highest grade variance, 20+ students, 5+ assignments

---

## Stage 2: Page Views ETL Pipeline

### Goal
Extract detailed clickstream data (page views) from Canvas API for students in selected high-potential courses.

### API Endpoint

```
GET /api/v1/users/{user_id}/page_views
```

**Parameters:**
| Parameter | Description |
|-----------|-------------|
| `start_time` | ISO 8601 timestamp (e.g., `2025-08-01T00:00:00Z`) |
| `end_time` | ISO 8601 timestamp |
| `per_page` | Records per page (max 100) |

**Response Fields:**
```json
{
  "url": "/courses/86005/modules",
  "context_type": "Course",
  "asset_type": "module",
  "controller": "context_modules",
  "action": "index",
  "interaction_seconds": 45.2,
  "created_at": "2025-10-15T14:30:00Z",
  "user_agent": "Mozilla/5.0...",
  "remote_ip": "192.168.1.1"
}
```

**Reference:** See `docs/canvas_resource_tracking_analysis.md` for detailed API capabilities.

### Step 2.1: Get Student List for a Course

```python
def get_course_students(course_id):
    """Get all student user IDs for a course"""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 100}
    )
    if r.status_code != 200:
        return []

    return [e['user_id'] for e in r.json()]

students = get_course_students(86005)
print(f"Found {len(students)} students")
```

### Step 2.2: Page Views ETL with Rate Limiting

**Critical:** Canvas API has rate limits. The `X-Rate-Limit-Remaining` header tells you remaining quota.

```python
import concurrent.futures
import threading
import time
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from tqdm import tqdm
import re

# Configuration
OUTPUT_DIR = 'data/page_views'
PROCESSED_FILE = f'{OUTPUT_DIR}/processed_student_ids.txt'
MAX_WORKERS = 5  # Conservative threading
MAX_RETRIES = 3
BASE_DELAY = 2  # Seconds

# Thread-safe state
file_lock = threading.Lock()
processed_lock = threading.Lock()
rate_limit_lock = threading.Lock()
current_delay = 0.5  # Dynamic delay based on rate limit

def calculate_delay(remaining_quota):
    """
    Calculate delay based on remaining API quota.
    More aggressive backing off as quota decreases.
    """
    if remaining_quota < 10:
        return 30  # Critical - almost no quota
    elif remaining_quota < 50:
        return 10  # Very low
    elif remaining_quota < 100:
        return 5   # Low
    elif remaining_quota < 200:
        return 2   # Moderate
    elif remaining_quota < 300:
        return 1   # Healthy
    else:
        return 0.5  # Abundant

def load_processed_ids():
    """Load already processed student IDs"""
    try:
        with open(PROCESSED_FILE, 'r') as f:
            return set(int(line.strip()) for line in f if line.strip())
    except FileNotFoundError:
        return set()

def save_processed_id(student_id):
    """Thread-safe save of processed ID"""
    with processed_lock:
        with open(PROCESSED_FILE, 'a') as f:
            f.write(f"{student_id}\n")

def extract_course_id(url):
    """Extract course_id from Canvas URL"""
    match = re.search(r"/courses/(\d+)", url)
    return int(match.group(1)) if match else None

def fetch_page_views(student_id, course_id, start_time, end_time):
    """
    Fetch all page views for a student within time range.
    Filters to specific course_id.

    Returns: List of page view records or None if failed
    """
    global current_delay

    all_views = []
    url = f'{API_URL}/api/v1/users/{student_id}/page_views'
    params = {
        'start_time': start_time,
        'end_time': end_time,
        'per_page': 100
    }

    retries = 0
    while url and retries < MAX_RETRIES:
        try:
            time.sleep(current_delay)
            r = requests.get(url, headers=headers, params=params)

            # Update delay based on rate limit
            remaining = int(r.headers.get('X-Rate-Limit-Remaining', 700))
            with rate_limit_lock:
                current_delay = calculate_delay(remaining)

            if r.status_code == 403:
                # Rate limited - back off significantly
                print(f"Rate limited for user {student_id}, waiting 60s...")
                time.sleep(60)
                retries += 1
                continue

            if r.status_code != 200:
                print(f"Error {r.status_code} for user {student_id}")
                return None

            # Filter page views to our course
            page_data = r.json()
            for pv in page_data:
                pv_course = extract_course_id(pv.get('url', ''))
                if pv_course == course_id:
                    all_views.append({
                        'user_id': student_id,
                        'course_id': course_id,
                        'url': pv.get('url'),
                        'context_type': pv.get('context_type'),
                        'asset_type': pv.get('asset_type'),
                        'controller': pv.get('controller'),
                        'action': pv.get('action'),
                        'interaction_seconds': pv.get('interaction_seconds'),
                        'created_at': pv.get('created_at'),
                        'participated': pv.get('participated', False)
                    })

            # Pagination
            url = r.links.get('next', {}).get('url')
            params = {}  # URL contains params
            retries = 0  # Reset retries on success

        except Exception as e:
            print(f"Exception for user {student_id}: {e}")
            retries += 1
            time.sleep(BASE_DELAY * (2 ** retries))

    return all_views

def save_to_parquet(views, student_id, output_dir):
    """Save page views to parquet file (one per student)"""
    if not views:
        return

    import pandas as pd
    df = pd.DataFrame(views)

    with file_lock:
        output_path = f"{output_dir}/student_{student_id}.parquet"
        df.to_parquet(output_path, engine='pyarrow', index=False)

def process_student(args):
    """Worker function for thread pool"""
    student_id, course_id, start_time, end_time, output_dir = args

    views = fetch_page_views(student_id, course_id, start_time, end_time)

    if views is not None:
        save_to_parquet(views, student_id, output_dir)
        save_processed_id(student_id)
        return (student_id, len(views), 'success')
    else:
        return (student_id, 0, 'failed')
```

### Step 2.3: Run the ETL Pipeline

```python
import os

def run_page_views_etl(course_id, start_date='2025-08-01', end_date='2025-12-31'):
    """
    Run complete ETL pipeline for a course.

    Args:
        course_id: Canvas course ID
        start_date: Start of time range (YYYY-MM-DD)
        end_date: End of time range (YYYY-MM-DD)
    """
    global OUTPUT_DIR, PROCESSED_FILE

    # Setup output directory
    OUTPUT_DIR = f'data/page_views/course_{course_id}'
    PROCESSED_FILE = f'{OUTPUT_DIR}/processed_student_ids.txt'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Time range
    start_time = f"{start_date}T00:00:00Z"
    end_time = f"{end_date}T23:59:59Z"

    # Get students
    students = get_course_students(course_id)
    print(f"Course {course_id}: {len(students)} students")

    # Filter already processed
    processed = load_processed_ids()
    to_process = [s for s in students if s not in processed]
    print(f"Already processed: {len(processed)}, remaining: {len(to_process)}")

    if not to_process:
        print("All students already processed!")
        return

    # Prepare work items
    work_items = [
        (student_id, course_id, start_time, end_time, OUTPUT_DIR)
        for student_id in to_process
    ]

    # Run with thread pool
    results = {'success': 0, 'failed': 0, 'total_views': 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = list(tqdm(
            executor.map(process_student, work_items),
            total=len(work_items),
            desc="Extracting page views"
        ))

        for student_id, n_views, status in futures:
            results[status] = results.get(status, 0) + 1
            results['total_views'] += n_views

    print(f"\nETL Complete!")
    print(f"  Success: {results['success']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Total page views: {results['total_views']}")
    print(f"  Output: {OUTPUT_DIR}/")

# Example usage:
# run_page_views_etl(86005, start_date='2025-08-01', end_date='2025-12-20')
```

### Step 2.4: Combine Parquet Files

```python
import pandas as pd
import glob

def combine_parquet_files(course_id):
    """Combine individual student parquet files into one"""
    pattern = f'data/page_views/course_{course_id}/student_*.parquet'
    files = glob.glob(pattern)

    if not files:
        print(f"No parquet files found for course {course_id}")
        return None

    dfs = [pd.read_parquet(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)

    output_path = f'data/page_views/course_{course_id}_combined.parquet'
    combined.to_parquet(output_path, engine='pyarrow', index=False)

    print(f"Combined {len(files)} files into {output_path}")
    print(f"Total records: {len(combined)}")
    print(f"Unique users: {combined['user_id'].nunique()}")
    print(f"Date range: {combined['created_at'].min()} to {combined['created_at'].max()}")

    return combined

# df = combine_parquet_files(86005)
```

### Rate Limit Reference

| Remaining Quota | Delay (seconds) | Status |
|-----------------|-----------------|--------|
| 700+ | 0.5 | Abundant |
| 300-699 | 1 | Healthy |
| 200-299 | 2 | Moderate |
| 100-199 | 5 | Low |
| 50-99 | 10 | Very Low |
| < 50 | 30 | Critical |

### Your Task: Stage 2 Deliverable

1. **Select 2 courses from Stage 1** (1 Pregrado, 1 Postgrado)
2. **Run ETL pipeline** for each course
3. **Document results:**
   - Number of students processed
   - Total page views extracted
   - Date range of activity
   - Any errors or rate limit issues
4. **Create combined parquet file** for analysis

---

## Output Files Structure

```
data/
├── page_views/
│   ├── course_86005/
│   │   ├── student_117656.parquet
│   │   ├── student_117657.parquet
│   │   ├── ...
│   │   └── processed_student_ids.txt
│   ├── course_86005_combined.parquet
│   └── course_XXXXX_combined.parquet
```

---

## Reference Documents

| Document | Content |
|----------|---------|
| `docs/data_access_discovery.md` | Full account hierarchy, API endpoints, statistics |
| `docs/canvas_resource_tracking_analysis.md` | Page views API details, module tracking |
| `docs/control_de_gestion_courses.md` | Example course scan results |
| `docs/early_warning_findings.md` | Model results and key features |
| `CLAUDE.md` | Complete API reference and code examples |

---

## Tips and Gotchas

1. **Page Views API has NO course filter** - you must filter post-fetch by parsing URLs
2. **Rate limits are per-token** - multiple scripts share the same quota
3. **Parquet is preferred** over CSV for large datasets (faster, smaller, typed)
4. **Always use `processed_student_ids.txt`** to enable resumable ETL
5. **Check enrollment_term_id** - old courses may not have accessible grades
6. **Pass rate between 20-80%** is ideal for prediction models

---

*Last updated: December 2025*
