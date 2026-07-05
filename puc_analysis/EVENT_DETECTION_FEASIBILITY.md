# Event-Anchored Features: Feasibility Analysis

**Date**: February 2026
**Dataset**: PUC 7 benchmark courses (560 student-course pairs, 1.46M page views)
**Context**: The event-anchored ML architecture needs assessment dates. This analysis tests whether we can detect them from clickstream data alone, and evaluates TSMixer as an alternative temporal model.

---

## 1. Assessment Date Detection from Activity Spikes

### Problem

We don't have explicit assessment dates in the LMS. The `calendar_events` table mixes personal events, classes, and holidays. We need a clean signal for evaluation dates.

### Method

Detect assessment days from daily quiz+assignment page view spikes:

```
daily_views = page_views.groupby([course_id, date, category]).count()
assessment_views = daily_views[category in ('quizzes', 'assignments')]
spike_threshold = mean + 2 * std  (per course)
assessment_dates = days where assessment_views > spike_threshold
```

### Results

The method works with high confidence. Across 7 courses, it detects 5-11 events each — matching expected semester assessment frequency (~1 every 1-2 weeks).

| Course | Students | Spikes Detected | Day Pattern | Dominant Type |
|--------|----------|-----------------|-------------|---------------|
| 54503  | 51       | 11              | Wednesdays (weekly)      | quizzes + assignments |
| 54529  | 131      | 6               | Fridays (biweekly)       | quizzes |
| 54570  | 22       | 6               | Irregular                | assignments |
| 54581  | 16       | 7               | End-of-semester cluster  | assignments |
| 55010  | 117      | 9               | Mon/Fri mix              | quizzes |
| 55183  | 99       | 5               | Wednesdays               | mixed |
| 55410  | 124      | 10              | Tuesdays (weekly)        | quizzes (>95%) |

### Validation: Spike Days vs Normal Days (Course 54529)

| Metric | Spike Days | Normal Days | Ratio |
|--------|-----------|-------------|-------|
| Total page views | 9,253 | 1,417 | 6.5x |
| Quiz views | 1,402 | 10.5 | **133x** |
| Assignment views | 955 | 11.3 | **85x** |
| Quiz+assignment share | 25.5% | 0.9% | 28x |
| Student participation | 120-128/131 | ~40-60/131 | ~2x |

The 133x ratio for quiz views on spike days vs normal days leaves no ambiguity — these are real assessment events.

### Additional Confirmation Signals

- `action='submission'`: 26,861 rows across 7 courses (direct submission events)
- `participated=True`: 10,643 rows (Canvas participation flag, matches `http_method=post`)
- `action='create'`: 520 rows (new submission creation)
- URL patterns: `/quizzes/.../take`, `/assignments/.../submissions/new`

### Conclusion

**Assessment date detection from spikes is reliable and sufficient for the event-anchored architecture.** No explicit calendar data needed. The detected dates can populate `assessment_schedules` automatically.

---

## 1b. Two-Layer Detection: Confirmed vs Inferred Assessments

### Problem

Not all evaluations live in Canvas. A student may prepare using LMS resources for a paper exam, oral presentation, or assignment submitted outside the platform. Activity spikes without matching quiz/assignment spikes may indicate these "off-Canvas" evaluations.

### Method: Two detection layers

1. **Confirmed** (high confidence): dates with quiz submission spikes (`action='submission'`) OR quiz+assignment page view spikes. The evaluation IS in Canvas.
2. **Inferred** (medium confidence): total activity spikes with NO matching submission or assessment-view spike within ±1 day. The evaluation likely exists but is NOT in Canvas — students used the LMS to prepare.

### Key finding: submissions are quiz-only

- All 26,861 `action='submission'` events are quiz submissions. Assignment submissions are NOT captured as submission events.
- `action='create'` on assignments (520 rows) is the closest proxy for assignment hand-ins.
- This means submission-based detection is strong for quizzes but blind to assignments. The activity-based layer fills this gap.

### Results per course

| Course | Students | Quizzes | Assignments* | Submissions | Confirmed | Inferred |
|--------|----------|---------|-------------|-------------|-----------|----------|
| 54503  | 51       | 15      | 17          | 2,148       | 11        | 1        |
| 54529  | 131      | 7       | 9           | 3,155       | 6         | 3        |
| 55010  | 117      | 29      | 28          | 5,146       | 9         | 3        |
| 55183  | 99       | 75      | 75          | 14,534      | 5         | 2        |
| 55410  | 124      | 14      | 2           | 1,725       | 10        | 0        |
| 54570  | 22       | 3       | 24          | 67          | 7         | 2        |
| 54581  | 16       | 6       | 54          | 86          | 10        | 3        |

*\*Assignment counts include quiz-shell echoes. See Section 1c for corrected counts.*

### Course instrumentation gradient

The courses split naturally into three levels:

**Well-instrumented** (all spikes explained by Canvas assessments):
- **55410**: 10 confirmed, 0 inferred — weekly Tuesday quizzes, every spike has a matching submission burst
- **54503**: 11 confirmed, 1 inferred — almost fully instrumented

**Partially instrumented** (most evaluations in Canvas, some outside):
- **54529**: 6 confirmed, 3 inferred — biweekly quizzes + 3 unexplained activity bursts
- **55010**: 9 confirmed, 3 inferred — rich assessment structure + 3 outside events
- **55183**: 5 confirmed, 2 inferred — 75 quizzes/75 assignments but most are small; 2 big outside-Canvas events

**Poorly instrumented** (few submissions despite heavy activity):
- **54570**: 3 quizzes, only 67 submissions — heavy assignment course, evaluations mostly offline
- **54581**: 6 quizzes, only 86 submissions — 54 assignments but almost no digital submission

### Implication

The well-instrumented courses (55410, 54503) serve as **ground truth** for validating the activity-based detection. If inferred events from poorly-instrumented courses (54570, 54581) show the same preparation wave patterns as confirmed events in well-instrumented courses, we can trust the inference.

### Visualization

Per-course timeline plots showing confirmed (green) and inferred (red) assessment dates overlaid on daily activity:

`data/puc/sota_results/7courses_multiclass/assessment_detection_by_course.png`

Script: `scripts/puc_plot_assessment_detection.py`

---

## 1c. Data Quality Corrections

### Issue 1: `action='create'` on assignments = student submission

Every `create` event on assignments hits `/courses/NNN/assignments/NNN/submissions` with `http_method=post`. This is the Canvas page view entry for a student submitting an assignment file/text. Since the dataset only contains per-student page views (collected via the Canvas Student Page Views API), there are no teacher/TA actions present.

**Conclusion**: `create` on assignments = student submission. Equivalent to `submission` on quizzes. No correction needed.

### Issue 2: Quiz-assignment double counting (66.9% of assignment views)

Canvas creates an "assignment shell" for every quiz. When a student takes a quiz, Canvas logs both quiz page views AND assignment API calls for the shell object.

**Evidence**:
- **81.8%** of quiz peak dates coincide with assignment peak dates
- **Jaccard similarity 0.70-0.96** between quiz and assignment student sets on overlapping days
- **76-94%** of assignment `(student, minute)` pairs overlap with quiz events
- 43% of assignment views are `/api/v1/` calls (programmatic metadata fetches, not user actions)
- **66.9% of all assignment views (74,549 / 111,444) are quiz-shell echoes**

**Root cause**: When Canvas loads a quiz page, its JavaScript fires both the quiz request AND an assignment API call for the shell object within milliseconds. This is a causal, deterministic behavior — not a statistical coincidence.

**Detection method**: Temporal co-occurrence. For each assignment resource_id, compute the fraction of its `(student_id, 10-second-time-bucket)` pairs that also appear in quiz page views for the same course. The distribution is perfectly bimodal:
- **Genuine assignments**: 0-8% overlap with quiz timestamps
- **Quiz shells**: 50-100% overlap with quiz timestamps
- **Gap**: 0.41 (any threshold between 0.10 and 0.49 gives identical classification)

This exploits the causal mechanism rather than relying on statistical heuristics like date matching or Jaccard similarity.

**Alternative (with API access)**: The Canvas Assignments API includes `submission_types: ["online_quiz"]` and a `quiz_id` field for shell assignments. This would be ground truth but requires API access to the institution's Canvas instance.

**Correction applied**: `identify_quiz_shell_assignments()` in `puc_plot_calendar_heatmap.py` uses temporal co-occurrence with threshold 0.15.

**Impact on features**: `assignments_views` and `assignments_pct` in the SOTA benchmark are inflated by ~67%. Feature engineering pipelines should apply similar deduplication when computing assignment-related features.

### Issue 3: Course 54581 inflated assignment count

54 unique assignment resource_ids detected, but:
- **33 genuine** (viewed by ≥30% of 16 students)
- **13 borderline** (3-4 students)
- **8 suspect** (≤2 students, sub-1-day lifespan, includes admin/grading-settings URLs)
- **9 assignments appeared on July 13** (last day of semester) — likely bulk grade-book entries

**Correction applied**: Minimum student coverage filter (≥25% of enrolled students) removes admin items, drafts, and grade-book-only entries from heatmap markers.

### Corrected evaluation counts

After applying both corrections (quiz-shell exclusion + low-coverage filter):

| Course | Quizzes | Genuine Assignments | Shell Excluded | Low Coverage Excluded | Original Assignments |
|--------|--------:|--------------------:|---------------:|----------------------:|---------------------:|
| 54503  | 14      | 4                   | 13             | 0                     | 17                   |
| 54529  | 6       | 1                   | 6              | 2                     | 9                    |
| 55010  | 28      | 0                   | 28             | 0                     | 28                   |
| 55183  | 74      | 1                   | 74             | 0                     | 75                   |
| 55410  | 13      | 0                   | 1              | 1                     | 2                    |
| 54570  | 3       | 22                  | 0              | 2                     | 24                   |
| 54581  | 5       | 37                  | 1              | 16                    | 54                   |

**Key observations**:
- Quiz-heavy courses (55010, 55183, 55410) had near-100% shell contamination in assignments — almost every "assignment" was a quiz echo
- Assignment-heavy course 54570 had zero shells (no quiz overlap), confirming the detection is precise
- Course 54581 dropped from 54 to 37 assignments (1 shell + 16 low-coverage items removed)
- Total across 7 courses: 123 shell assignments removed, 21 low-coverage items filtered

---

## 2. Activity Slope Analysis (Trending Periods)

### Problem

Can we detect the pre-assessment preparation ramp-up from course-level activity slopes? This would give us the temporal "wave" shape without needing dates first.

### Method

```
daily_activity = page_views.groupby([course_id, date]).count()
rolling_mean = daily_activity.rolling(3).mean()
slope = rolling_mean.diff()
trending_up = consecutive days where slope > mean(slope) + std(slope)
```

### Results (Course 54529)

- **9 spike days** detected (mean + 2*std threshold on daily activity)
- **16 trending-up periods** detected (slope > 1 std above mean)
- Trending-up periods **precede** the spike days — they capture the ramp-up
- The ramp is dramatic: quiz views go from ~10/day to 1,400/day on assessment day
- Spike days are isolated single-day bursts (none within 3 days of each other), consistent with periodic scheduled assessments

### Cross-Validation

Spike days detected from total activity include all assessment spikes but also produce 3 additional false positives (likely exam prep or end-of-semester activity). Using category-filtered detection (quiz+assignment views only) eliminates false positives.

### Implication for Feature Engineering

The slope analysis provides the `prep_ramp_slope` feature from the event-anchored plan:
1. Detect assessment event from spike
2. Look backwards 72h at the activity slope leading into it
3. Characterize the preparation wave shape (ramp slope, peak position, symmetry)

This works without any external data — the events are fully detectable from clickstream behavior.

---

## 3. TSMixer / Temporal Model Feasibility

### Problem

Can we use TSMixer to predict student activity ("this user will have this much activity in the next hour") based on their past behavior and classmate behavior?

### Hourly Resolution: Not Viable

| Metric | Value |
|--------|-------|
| Median non-zero hours per student (first 3 weeks) | 27 / 504 |
| Hourly density | **5.4%** |
| Students with >= 30 active hours | 42% |
| Students with >= 50 active hours | 9% |

At 95% zeros, TSMixer would learn "predict zero" and achieve high accuracy by doing nothing useful. The mixing operations across time steps would be dominated by zero-to-zero transitions.

### Daily Resolution: Viable

| Metric | Value |
|--------|-------|
| Median active days per student (first 3 weeks) | 13 / 21 |
| Daily density | **62%** |
| Students with >= 7 active days | 93% |

At daily resolution, the data is dense enough for temporal models.

### Course-Level Aggregation Helps

For course 54529 (131 students), aggregating across all students fills **77% of hourly bins** and ~100% of daily bins. Course-level patterns can serve as a context channel even if individual students are sparse.

### Recommended Architecture (If Pursuing TSMixer)

```
Input: (21 days × K channels) per student
K channels:
  - daily_page_views
  - daily_sessions
  - daily_unique_resources
  - daily_interaction_time_min
  - daily_quiz_assignment_views
  + course_aggregate_views (context)
  + course_aggregate_sessions (context)

Shape: 500 students × 21 time steps × 7 channels
```

This gives 62% density, which is reasonable for channel-mixing architectures.

### But: Hand-Crafted Features Likely Win

The current pipeline already captures temporal dynamics through:
- `weekly_std`, `weekly_mean` (trajectory shape)
- `dct_0`, `dct_1` (DCT coefficients = level + trend)
- `max_gap_hours`, `mean_gap_hours`, `gap_std_hours` (inactivity patterns)
- `daily_consistency` (regularity)
- `active_weeks` (engagement breadth)

These features achieve ROC-AUC 0.831-0.872 with XGBoost. With only 500 samples, a temporal deep learning model is unlikely to discover patterns that these hand-crafted features miss.

### Recommended Validation Before Investing

Run a quick experiment:
1. Create the (21, 5) daily tensor for each student
2. Train a 1D-CNN or small LSTM on it (simple, fast)
3. Compare ROC-AUC against XGBoost on 280 hand-crafted features
4. If the temporal model matches or beats XGBoost → invest in TSMixer
5. If not → the hand-crafted features already capture the temporal signal

Estimated effort: 1 day. Avoids a 2-week TSMixer investment on insufficient data.

---

## Summary of Recommendations

| Question | Answer | Confidence |
|----------|--------|------------|
| Can we detect assessment dates from clickstream? | **Yes** — spike detection on quiz+assignment views, 5-11 events per course | High |
| Can we detect pre-assessment ramp-ups? | **Yes** — slope analysis on 3-day rolling mean finds trending periods before spikes | High |
| Is TSMixer viable at hourly resolution? | **No** — 95% sparsity, model would learn "predict zero" | High |
| Is TSMixer viable at daily resolution? | **Maybe** — 62% density is workable, but 500 samples is low | Medium |
| Should we invest in TSMixer now? | **Not yet** — run a 1-day 1D-CNN experiment first to check if temporal modeling adds value over hand-crafted features | — |

### Immediate Actions for Event-Anchored Architecture

1. **Use spike detection** to auto-populate `assessment_schedules` — no manual data entry needed
2. **Compute event-anchored features** (Categories A-C) using detected dates
3. **Skip TSMixer** for now — the event-anchored features on top of XGBoost is the higher-ROI path
4. **Revisit temporal models** when a fourth tenant provides >5,000 samples or GPU budget becomes available
