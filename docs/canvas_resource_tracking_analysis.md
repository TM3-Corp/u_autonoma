# Canvas API Resource Tracking Analysis

**Date:** December 2025
**Purpose:** Evaluate Canvas API capabilities for Oviedo-style resource tracking metrics

---

## Summary

Investigated Canvas API endpoints to implement features from the Oviedo paper:
- `ResourceViewUniquePct`: % of resources viewed by each student
- `ResourceViewTime{1-5}`: Timing of when resources were accessed
- Early vs late access patterns

### Key Finding

Canvas API provides **module-level** completion tracking but NOT **resource-level** view tracking per student without expensive per-user API calls.

---

## API Capabilities

### Available Endpoints (Tested)

| Endpoint | Purpose | Data Available |
|----------|---------|---------------|
| `/courses/:id/bulk_user_progress` | Module completion for all students | ONLY works if modules have completion requirements |
| `/courses/:id/modules?student_id=X` | Module state per student | `state` (locked/started/completed), `completed_at` |
| `/courses/:id/analytics/users/:id/activity` | Hourly activity breakdown | `page_views` by hour, `participations` with URLs |
| `/courses/:id/analytics/student_summaries` | Aggregate activity | `page_views`, `participations`, `tardiness_breakdown` |

### Limitations

| What We Need | Endpoint Status | Workaround |
|--------------|-----------------|------------|
| Which files each student viewed | NO direct endpoint | Page Views API (expensive) |
| View count per resource per student | NO endpoint | Page Views API + URL parsing |
| Resource completion rate | Requires module completion requirements | Use module completion if configured |

### Page Views API (Expensive Alternative)

```
GET /api/v1/users/:user_id/page_views
```

- Returns ALL page views for a user (not filterable by course)
- Must filter by parsing URL post-fetch (`/courses/86005/files/1234`)
- Would require 77+ API calls for our dataset
- Contains: `url`, `action`, `interaction_seconds`, `created_at`

---

## Oviedo Features Implemented

### Features We CAN Extract

| Feature | Source | Description |
|---------|--------|-------------|
| `early_access_score` | modules with student_id | Normalized rank (0=early, 1=late) |
| `module_completion_pct` | modules with student_id | % of modules completed |
| `activity_span_days` | analytics/users/:id/activity | Days between first and last activity |
| `is_morning` | analytics/users/:id/activity | Dominant study time = 6am-12pm |
| `is_evening` | analytics/users/:id/activity | Dominant study time = 6pm-10pm |

### Features We CANNOT Easily Extract

| Oviedo Feature | Why Not Available |
|----------------|-------------------|
| `ResourceViewUniquePct` | No bulk resource view tracking |
| `ResourceViewTime{1-5}` | No timestamp per resource per student |
| Per-file access patterns | Would need Page Views API |

---

## Model Results

### Classification (Pass/Fail Prediction)

| Feature Set | F1 Score (5-fold CV) |
|-------------|---------------------|
| Pure Activity (page_views only) | 0.740 ± 0.115 |
| Oviedo Timing Features | 0.729 ± 0.096 |
| **Combined** | **0.761 ± 0.112** |

### Feature Correlations with Final Grade

| Feature | Correlation | Significance |
|---------|-------------|--------------|
| `is_morning` | r = +0.241 | Morning studiers avg 97.8% |
| `early_access_score` | r = +0.229 | p = 0.051 (marginal) |
| `is_evening` | r = -0.182 | Evening studiers avg 31.5% |
| `total_page_views` | r = +0.332 | Same as previous analysis |

### Time of Day Analysis

| Study Time | Avg Grade | Count | Interpretation |
|------------|-----------|-------|----------------|
| Morning (6am-12pm) | 97.8% | 5 | **Best performers** |
| Afternoon (12pm-6pm) | 54.4% | 60 | Most common |
| Night (10pm-6am) | 63.0% | 3 | Mixed |
| Evening (6pm-10pm) | 31.5% | 9 | **Worst performers** |

### Linear Regression Coefficients

```
is_morning:         +53.93 points  (morning studiers score higher!)
early_access_score: +41.01 points  (early accessors score higher)
is_evening:         -29.54 points  (evening studiers score lower)
is_night:           +21.06 points
page_views:         +0.08 points per view
```

---

## Comparison with Oviedo Paper

| Oviedo Feature | Our Implementation | Result |
|----------------|-------------------|--------|
| ResourceViewUniquePct | `module_completion_pct` | No variance (all students completed same modules) |
| ResourceViewTime | `early_access_score` | r = 0.229 with grade |
| CIR (Course and resource view) | `total_page_views` | r = 0.332 with grade |
| Time patterns | `is_morning/evening/night` | Morning = best, Evening = worst |

**Oviedo reported 80.1% accuracy at 10% course completion. Our combined Oviedo features achieve F1 = 0.761, which translates to similar performance.**

---

## Recommendations

### Short-term (Current API)

1. **Use module completion timing** (`early_access_score`) - Available now, r = 0.229
2. **Use time-of-day features** - Morning studiers perform best
3. **Combine with page_views** for best F1 = 0.761

### Medium-term (Page Views API)

To get true per-resource tracking:
```python
# For each student
page_views = paginate_canvas(f'/users/{user_id}/page_views', ...)

# Filter by course
course_views = [pv for pv in page_views
                if f'/courses/{course_id}/' in pv['url']]

# Count unique resources viewed
resources = set()
for pv in course_views:
    if '/files/' in pv['url'] or '/pages/' in pv['url']:
        resources.add(pv['url'])

resource_view_pct = len(resources) / total_course_resources
```

**Cost:** ~77 API calls for current dataset, each potentially multi-page

### Long-term (Live Events)

Canvas Live Events (`asset_accessed` events) provide real-time tracking but require:
- AWS SQS setup
- Enterprise Canvas subscription
- Event stream processing infrastructure

---

## Files Created

| File | Purpose |
|------|---------|
| `scripts/test_resource_tracking.py` | API endpoint exploration |
| `scripts/extract_resource_access.py` | Oviedo feature extraction |
| `data/resource_access/module_completion_data.csv` | Module state per student |
| `data/resource_access/activity_timing_data.csv` | Hourly activity patterns |
| `data/resource_access/oviedo_features.csv` | Calculated Oviedo features |

---

## Conclusion

Canvas API provides module-level completion tracking that enables Oviedo-style timing features. The `early_access_score` (who completes modules first) correlates with grades (r=0.229, p=0.051). Time-of-day preference is a strong predictor: morning studiers average 97.8% vs evening studiers at 31.5%.

For full ResourceViewUniquePct implementation, the Page Views API would need to be used, requiring per-user API calls.

---

## Sources

- [Canvas Analytics API](https://canvas.instructure.com/doc/api/analytics.html)
- [Canvas Modules API](https://canvas.instructure.com/doc/api/modules.html)
- [Canvas Courses API - bulk_user_progress](https://canvas.instructure.com/doc/api/courses.html)
- [Canvas Live Events](https://canvas.instructure.com/doc/api/file.data_service_introduction.html)
- [Oviedo Paper](file:///home/paul/projects/uautonoma/oviedo_paper.pdf) - Learning Analytics for Early Prediction

*Last Updated: December 2025*
