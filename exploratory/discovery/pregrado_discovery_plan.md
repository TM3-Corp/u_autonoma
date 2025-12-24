# PREGRADO Course Discovery Plan

**Author:** Vicente
**Date:** December 2025
**Branch:** feature/eda-vicente

---

## Objective

Find new high-potential courses for early failure prediction analysis in PREGRADO (Account 46), excluding the already-analyzed Control de Gestión program (Account 719).

---

## User Preferences

| Setting     | Value                                               |
| ----------- | --------------------------------------------------- |
| Focus Area  | PREGRADO (Account 46)                               |
| Excluded    | Account 719 (Control de Gestión - already analyzed) |
| Term Filter | Both current (336) and recent past (322)            |
| Scan Depth  | Quick scan (~50 courses, top 3 sub-accounts)        |

---

## Course Selection Criteria

A "high-potential" course must have:

| Criterion      | Threshold      | Why                             |
| -------------- | -------------- | ------------------------------- |
| Students       | 20+            | Statistical significance for ML |
| Grade Variance | > 10 (std dev) | Enough variation to predict     |
| Pass Rate      | 20-80%         | Class diversity for training    |
| Assignments    | 5+             | Good LMS design indicator       |
| Term           | 336 or 322     | Current or recent semester      |

---

## Target Sub-Accounts

Focus on exploring these PREGRADO sub-accounts (update after discovery):

| Priority | Account ID | Program                     | Notes              |
| -------- | ---------- | --------------------------- | ------------------ |
| 1        | 730        | Ingeniería Civil Industrial | ~18 active courses |
| 2        | 247        | Psicología                  | Large program      |
| 3        | 253        | Derecho                     | Large program      |

**Skip:** Account 719 (Ing. en Control de Gestión) - already has 7 courses deeply analyzed.

---

## Rate Limit Management

Canvas API uses a quota system. Monitor the `X-Rate-Limit-Remaining` header.

### Adaptive Delay Strategy

| Remaining Quota | Delay | Status                      |
| --------------- | ----- | --------------------------- |
| 700+            | 0.5s  | Abundant - proceed normally |
| 500-699         | 1s    | Healthy                     |
| 300-499         | 2s    | Moderate - slow down        |
| 200-299         | 5s    | Low - significant slowdown  |
| 100-199         | 10s   | Very Low - caution          |
| 50-99           | 30s   | Critical - nearly paused    |
| < 50            | 60s   | Emergency - long wait       |

### If You Get Rate Limited (403 Response)

1. Stop immediately
2. Wait 60 seconds
3. Resume with longer delays
4. Consider running at off-peak hours

---

## How to Use the Notebook

### Prerequisites

1. **Environment setup:**

   ```bash
   cd /home/vicho1950/u_autonoma
   source venv/bin/activate  # If using virtual environment
   ```

2. **Verify credentials:**
   - Check `.env` file exists with `CANVAS_API_URL` and `CANVAS_API_TOKEN`
   - Copy from `.env.example` if needed

### Running the Notebook

```bash
jupyter notebook exploratory/01_pregrado_discovery.ipynb
```

### Step-by-Step Execution

#### Section 1: Setup & API Connection

- Run cells to load credentials
- Verify "Connected as: [your name]" appears
- Check rate limit shows 700+ remaining

#### Section 2: Rate Limit Management

- Just run to define helper functions
- No API calls made here

#### Section 3: Discover PREGRADO Sub-Accounts

- Lists all programs under PREGRADO
- Note which sub-accounts look promising
- Already excludes Account 719

#### Section 4: Scan Sub-Accounts for Courses

- **IMPORTANT:** Update `TARGET_SUB_ACCOUNTS` list based on Section 3 results
- Scans for courses with 15+ students
- May take 1-2 minutes per sub-account

#### Section 5: Analyze Course Potential

- **WARNING:** Makes ~3 API calls per course
- For 50 courses = ~150 API calls
- Watch the progress bar and rate limit messages
- High-potential courses print immediately when found

#### Section 6: Select Top 5 Courses

- Review the ranked results
- Courses marked "HIGH POTENTIAL" are best candidates

#### Section 7: Save Results

- Exports to `data/pregrado_discovery_results.csv`
- Can be used for future reference

#### Section 8: Summary

- Fill in the top 5 courses table manually
- Document any issues encountered

---

## Expected Deliverables

1. **Jupyter Notebook:** `exploratory/01_pregrado_discovery.ipynb` (with outputs)
2. **Data Output:** `data/pregrado_discovery_results.csv`
3. **Summary:** Top 5 courses documented in Section 8

---

## Troubleshooting

### "No candidate courses found"

- Check if `TARGET_SUB_ACCOUNTS` is populated
- Verify the account IDs exist
- Try different sub-accounts from Section 3

### "Rate limited" messages

- Increase delays in `calculate_delay()` function
- Wait 5-10 minutes before resuming
- Consider running fewer courses

### "No grades" for all courses

- Some programs may use external grading (LTI)
- Try different sub-accounts
- Check if term_id filter is correct

### API connection fails

- Verify `.env` file has correct credentials
- Check network connectivity
- Test with: `curl -H "Authorization: Bearer $TOKEN" $API_URL/api/v1/users/self`

---

## Next Steps (After Discovery)

1. **Select 1-2 best courses** from your top 5
2. **Extract page views** using code from `intern_exploration_guide.md` (Stage 2)
3. **Document findings** in project docs
4. **Consider POSTGRADO** - 1,122 courses available for future exploration

---

## Key Reference Files

| File                                      | Purpose                                |
| ----------------------------------------- | -------------------------------------- |
| `exploratory/intern_exploration_guide.md` | Complete code examples, Page Views ETL |
| `CLAUDE.md`                               | Full API documentation                 |
| `docs/data_access_discovery.md`           | Account hierarchy details              |
| `scripts/utils/pagination.py`             | Reusable pagination utilities          |
| `scripts/config.py`                       | API configuration constants            |

---

## API Quick Reference

### Get Sub-Accounts

```python
GET /api/v1/accounts/{account_id}/sub_accounts
params: {'per_page': 100}
```

### Get Courses from Account

```python
GET /api/v1/accounts/{account_id}/courses
params: {
    'per_page': 100,
    'include[]': ['total_students', 'term'],
    'enrollment_term_id': 336,  # Optional filter
    'with_enrollments': True
}
```

### Get Enrollments with Grades

```python
GET /api/v1/courses/{course_id}/enrollments
params: {
    'type[]': 'StudentEnrollment',
    'per_page': 100,
    'include[]': 'grades'
}
```

### Get Assignments

```python
GET /api/v1/courses/{course_id}/assignments
params: {'per_page': 100}
```

### Get Modules

```python
GET /api/v1/courses/{course_id}/modules
params: {'per_page': 100}
```

---

_Last updated: December 2024_
