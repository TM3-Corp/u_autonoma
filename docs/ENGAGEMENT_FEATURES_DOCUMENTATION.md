# Engagement Dynamics Features: Complete Documentation

This document explains how each engagement dynamics feature is created from raw Canvas LMS API data.

---

## Table of Contents

1. [Data Sources (Canvas API Endpoints)](#1-data-sources-canvas-api-endpoints)
2. [Feature Categories Overview](#2-feature-categories-overview)
3. [Session Regularity Features](#3-session-regularity-features)
4. [Time Block Features](#4-time-block-features)
5. [DCT Coefficient Features](#5-dct-coefficient-features)
6. [Engagement Trajectory Features](#6-engagement-trajectory-features)
7. [Workload Dynamics Features](#7-workload-dynamics-features)
8. [Time-to-Access Features](#8-time-to-access-features)
9. [Raw Aggregate Features](#9-raw-aggregate-features)
10. [Teacher/TA Features](#10-teacherta-features)
11. [Normalization](#11-normalization)
12. [Feature Agglomeration](#12-feature-agglomeration)

---

## 1. Data Sources (Canvas API Endpoints)

### Primary Endpoints Used

| Endpoint | Purpose | Key Fields | Rate Limit |
|----------|---------|------------|------------|
| `GET /api/v1/courses/{id}/enrollments` | Student list + grades | `user_id`, `grades.final_score`, `total_activity_time` | 100/page |
| `GET /api/v1/courses/{id}/analytics/student_summaries` | Aggregate activity | `page_views`, `participations`, `tardiness_breakdown` | 100/page |
| `GET /api/v1/courses/{id}/analytics/users/{id}/activity` | Hourly activity | `page_views` (dict), `participations` (list) | Per-student |
| `GET /api/v1/courses/{id}/modules?student_id={id}` | Module progress | `state`, `completed_at` | Per-student |
| `GET /api/v1/users/{id}/page_views` | Detailed clickstream | `url`, `created_at`, `interaction_seconds` | Per-user |
| `GET /api/v1/courses/{id}` | Course metadata | `start_at`, `end_at`, `created_at` | Once |

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           CANVAS LMS API                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│   Enrollments   │      │  User Activity API  │      │  Student        │
│   API           │      │  (Hourly Data)      │      │  Summaries API  │
└─────────────────┘      └─────────────────────┘      └─────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────┐
│ • user_id       │      │ • page_views dict   │      │ • page_views    │
│ • final_score   │      │   {timestamp: count}│      │ • participations│
│ • activity_time │      │ • participations[]  │      │ • tardiness     │
└─────────────────┘      └─────────────────────┘      └─────────────────┘
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     TIMESTAMP PARSING         │
                    │  Convert ISO strings to       │
                    │  datetime objects             │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     FEATURE CALCULATION       │
                    │  54 engagement features       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     NORMALIZATION             │
                    │  Within-course z-scores       │
                    └───────────────────────────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     FEATURE AGGLOMERATION     │
                    │  54 features → 8 clusters     │
                    └───────────────────────────────┘
```

---

## 2. Feature Categories Overview

| Category | Count | Source API | Description |
|----------|-------|------------|-------------|
| Session Regularity | 7 | User Activity | Study session patterns and gaps |
| Time Blocks | 11 | User Activity | When students study (time-of-day/week) |
| DCT Coefficients | 12 | User Activity | Periodic activity patterns (Fourier-like) |
| Engagement Trajectory | 6 | User Activity | How engagement changes over time |
| Workload Dynamics | 10 | User Activity | Peaks, slopes, variability in activity |
| Time-to-Access | 4 | User Activity + Modules | Procrastination indicators |
| Raw Aggregates | 4 | Student Summaries | Total counts and spans |
| **Total** | **54** | | |

---

## 3. Session Regularity Features

### Source API

```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
```

### Raw Data Structure

```json
{
  "page_views": {
    "2025-09-15T14:00:00Z": 3,
    "2025-09-15T15:00:00Z": 2,
    "2025-09-16T10:00:00Z": 5,
    "2025-09-18T20:00:00Z": 1
  },
  "participations": [...]
}
```

### Transformation Process

1. **Parse timestamps**: Convert ISO strings to datetime objects
2. **Expand counts**: If timestamp has count=3, create 3 entries for that hour
3. **Sort chronologically**: Order all activity timestamps
4. **Calculate gaps**: Compute time difference between consecutive activities
5. **Identify sessions**: Gap > 60 minutes = new session

### Feature Definitions

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `session_count` | Count of session starts (gaps ≥ 60 min) | Total study sessions in course |
| `session_gap_min` | `min(inter_session_gaps)` | Shortest break between sessions (hours) |
| `session_gap_max` | `max(inter_session_gaps)` | Longest break between sessions (hours) |
| `session_gap_mean` | `mean(inter_session_gaps)` | Average break duration (hours) |
| `session_gap_std` | `std(inter_session_gaps)` | Variability in breaks (hours) |
| `session_regularity` | `1 - (gap_std / gap_mean)` | Consistency score (0-1, higher = more regular) |
| `sessions_per_week` | `session_count / course_weeks` | Average weekly session frequency |

### Example Calculation

```python
# Raw timestamps from API (sorted)
timestamps = [
    "2025-09-15T14:00:00Z",  # Session 1 start
    "2025-09-15T15:00:00Z",  # Same session (gap < 60 min)
    "2025-09-16T10:00:00Z",  # Session 2 start (gap = 19 hours)
    "2025-09-18T20:00:00Z",  # Session 3 start (gap = 58 hours)
]

# Calculate gaps in hours
gaps = [1.0, 19.0, 58.0]

# Inter-session gaps (only gaps >= 60 min)
inter_session_gaps = [19.0, 58.0]

# Features
session_count = 3  # Three sessions identified
session_gap_min = 19.0
session_gap_max = 58.0
session_gap_mean = 38.5
session_gap_std = 27.58
session_regularity = 1 - (27.58 / 38.5) = 0.284
```

### Predictive Interpretation

- **Higher `session_count`** → More frequent engagement → Better grades (r = +0.33)
- **Lower `session_gap_mean`** → Shorter breaks → Better grades (r = -0.33)
- **Lower `session_regularity`** → More erratic study patterns → Worse grades (r = -0.28)

---

## 4. Time Block Features

### Source API

Same as Session Regularity:
```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
```

### Transformation Process

1. **Parse timestamps**: Get datetime from each page_view entry
2. **Classify by day type**: Monday-Friday = weekday, Saturday-Sunday = weekend
3. **Classify by time of day**:
   - Morning: 6:00 - 11:59
   - Afternoon: 12:00 - 17:59
   - Evening: 18:00 - 23:59
   - Night: 0:00 - 5:59
4. **Count activities per block**: Sum page views in each of 8 blocks
5. **Calculate proportions**: Divide by total activity
6. **Track weekly consistency**: Calculate SD of proportions across weeks

### Feature Definitions

| Feature | Formula | Range | Interpretation |
|---------|---------|-------|----------------|
| `weekday_morning_pct` | Views(Mon-Fri, 6-12) / Total | 0-1 | % of activity on weekday mornings |
| `weekday_afternoon_pct` | Views(Mon-Fri, 12-18) / Total | 0-1 | % of activity on weekday afternoons |
| `weekday_evening_pct` | Views(Mon-Fri, 18-24) / Total | 0-1 | % of activity on weekday evenings |
| `weekday_night_pct` | Views(Mon-Fri, 0-6) / Total | 0-1 | % of activity on weekday nights |
| `weekend_morning_pct` | Views(Sat-Sun, 6-12) / Total | 0-1 | % of activity on weekend mornings |
| `weekend_afternoon_pct` | Views(Sat-Sun, 12-18) / Total | 0-1 | % of activity on weekend afternoons |
| `weekend_evening_pct` | Views(Sat-Sun, 18-24) / Total | 0-1 | % of activity on weekend evenings |
| `weekend_night_pct` | Views(Sat-Sun, 0-6) / Total | 0-1 | % of activity on weekend nights |
| `weekday_morning_sd` | SD of weekly weekday_morning_pct | ≥0 | Consistency of morning study habit |
| `weekday_afternoon_sd` | SD of weekly weekday_afternoon_pct | ≥0 | Consistency of afternoon study habit |
| `weekend_total_sd` | SD of weekly total weekend activity | ≥0 | Consistency of weekend engagement |

### Example Calculation

```python
# Activity counts by block
blocks = {
    'weekday_morning': 45,
    'weekday_afternoon': 30,
    'weekday_evening': 60,
    'weekday_night': 5,
    'weekend_morning': 10,
    'weekend_afternoon': 20,
    'weekend_evening': 25,
    'weekend_night': 5,
}

total = 200

# Percentages
weekday_morning_pct = 45 / 200 = 0.225  # 22.5%
weekday_evening_pct = 60 / 200 = 0.300  # 30.0% (dominant)
weekend_evening_pct = 25 / 200 = 0.125  # 12.5%
```

### Predictive Interpretation

- **Higher `weekend_afternoon_pct`** → More weekend studying → Better grades (r = +0.24)
- **Higher `weekday_evening_pct`** → Evening study preference → Better grades (r = +0.17)
- **Higher `weekday_afternoon_pct`** → Daytime studying → Worse grades (r = -0.17)

---

## 5. DCT Coefficient Features

### Source API

Same as above, but processed into a 168-dimensional weekly activity vector.

### Background: What is DCT?

The **Discrete Cosine Transform (DCT)** decomposes a signal into frequency components. For our 168-slot weekly activity vector (24 hours × 7 days), DCT captures:

- **Low-frequency coefficients (0-3)**: Overall activity level and broad patterns
- **Mid-frequency coefficients (4-7)**: Daily rhythms (circadian patterns)
- **Higher-frequency coefficients (8-11)**: Finer temporal variations

### Transformation Process

1. **Build 168-slot vector**: Create array where index = (day_of_week × 24) + hour
2. **Normalize**: Divide by total to get proportions
3. **Apply DCT**: Use scipy.fftpack.dct with orthonormal normalization
4. **Keep first 12 coefficients**: These capture 80%+ of variance

```python
from scipy.fftpack import dct

# Build weekly activity vector (168 slots)
weekly_vector = np.zeros(168)
for timestamp in timestamps:
    day = timestamp.weekday()  # 0=Monday
    hour = timestamp.hour
    slot = day * 24 + hour
    weekly_vector[slot] += 1

# Normalize
weekly_vector = weekly_vector / weekly_vector.sum()

# Apply DCT
dct_coeffs = dct(weekly_vector, norm='ortho')

# Take first 12 coefficients
features = dct_coeffs[:12]
```

### Feature Definitions

| Feature | Interpretation |
|---------|----------------|
| `dct_coef_0` | **DC component** - Overall activity level (always positive) |
| `dct_coef_1` | Captures weekly trend (early vs late week activity) |
| `dct_coef_2` | Captures mid-week patterns |
| `dct_coef_3` | Captures ~2.3 day periodicity |
| `dct_coef_4` | Captures ~1.75 day periodicity |
| `dct_coef_5` | **Daily rhythm** - Strongest circadian pattern |
| `dct_coef_6` | Sub-daily variations |
| `dct_coef_7` | ~12-hour patterns (morning vs evening) |
| `dct_coef_8-11` | Higher-frequency variations |

### Why DCT?

DCT compresses 168 dimensions into 12 while preserving:
- **Circadian rhythms**: When students prefer to study
- **Weekly patterns**: Weekday vs weekend differences
- **Regularity**: Consistent patterns have stronger low-frequency components

### Predictive Interpretation

- **`dct_coef_5`** has strongest correlation (r = -0.38): Students with irregular daily rhythms (high absolute value) tend to have lower grades
- **`dct_coef_0`** (DC component) correlates with overall engagement

---

## 6. Engagement Trajectory Features

### Source API

```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
```

### Concept

These features capture how engagement **changes over time** during the course, not just total engagement.

### Transformation Process

1. **Group activity by week**: Count page views per ISO week number
2. **Create weekly time series**: `[week1_count, week2_count, ..., weekN_count]`
3. **Calculate derivatives**: Velocity (1st derivative), acceleration (2nd derivative)
4. **Measure consistency**: Coefficient of variation, trend reversals

### Feature Definitions

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `engagement_velocity` | Linear regression slope of weekly counts | Positive = increasing engagement |
| `engagement_acceleration` | 2nd derivative (rate of velocity change) | Positive = engagement speeding up |
| `weekly_cv` | `std(weekly_counts) / mean(weekly_counts)` | Variability (lower = more stable) |
| `trend_reversals` | Count of week-to-week direction changes | Higher = erratic engagement |
| `early_engagement_ratio` | `sum(weeks 1-3) / sum(all weeks)` | Front-loading vs procrastination |
| `late_surge` | `sum(final 2 weeks) / mean(prior weeks)` | Last-minute cramming indicator |

### Example Calculation

```python
# Weekly page view counts
weekly_counts = [50, 45, 60, 55, 80, 100, 120, 150]  # 8 weeks

# Velocity (linear regression slope)
x = np.arange(8)
slope, intercept = np.polyfit(x, weekly_counts, 1)
engagement_velocity = slope  # ≈ +14.3 (increasing)

# Acceleration (2nd-order polynomial)
coeffs = np.polyfit(x, weekly_counts, 2)
engagement_acceleration = 2 * coeffs[0]  # ≈ +1.2

# Coefficient of variation
weekly_cv = np.std(weekly_counts) / np.mean(weekly_counts)  # ≈ 0.39

# Trend reversals
diffs = np.diff(weekly_counts)  # [-5, 15, -5, 25, 20, 20, 30]
reversals = sum(diffs[i] * diffs[i-1] < 0 for i in range(1, len(diffs)))  # = 2

# Early engagement ratio
early_engagement_ratio = sum(weekly_counts[:3]) / sum(weekly_counts)  # = 155/660 = 0.23

# Late surge
late_surge = sum(weekly_counts[-2:]) / np.mean(weekly_counts[:-2])  # = 270/65 = 4.15
```

### Predictive Interpretation

- **Higher `weekly_cv`** → More variable engagement → Better grades (r = +0.25)
  - *Counter-intuitive: May indicate responsive engagement to course demands*
- **Higher `trend_reversals`** → More erratic → Better grades (r = +0.24)
  - *May indicate active adjustment to course rhythm*

---

## 7. Workload Dynamics Features

### Source API

```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
```

### Concept (from "Beyond Time on Task" paper)

These features capture **intensity variations** in engagement:
- **Peaks**: Weeks with unusually high activity
- **Slopes**: Week-to-week changes in activity
- **Range**: Difference between highest and lowest activity weeks

### Transformation Process

1. **Group by week**: Same as trajectory features
2. **Calculate course mean**: Average weekly activity
3. **Identify peaks**: Weeks exceeding thresholds (1.25x, 1.5x, 2x mean)
4. **Calculate slopes**: Differences between consecutive weeks

### Feature Definitions

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `peak_count_type1` | Count of weeks > 1.25 × mean | Low-intensity peaks |
| `peak_count_type2` | Count of weeks > 1.50 × mean | Medium-intensity peaks |
| `peak_count_type3` | Count of weeks > 2.00 × mean | High-intensity peaks |
| `peak_ratio` | `type3_count / (type1_count + 0.01)` | Proportion of high vs low peaks |
| `max_positive_slope` | `max(weekly_diffs)` | Largest week-to-week increase |
| `max_negative_slope` | `min(weekly_diffs)` | Largest week-to-week decrease |
| `slope_std` | `std(weekly_diffs)` | Variability of week-to-week changes |
| `positive_slope_sum` | Sum of all positive slopes | Total upward momentum |
| `negative_slope_sum` | Sum of all negative slopes | Total downward momentum |
| `weekly_range` | `max(weekly_counts) - min(weekly_counts)` | Activity spread |

### Example Calculation

```python
weekly_counts = [50, 45, 60, 55, 80, 100, 120, 150]
mean_count = 82.5

# Peak detection
threshold_1 = 1.25 * mean_count = 103.1
threshold_2 = 1.50 * mean_count = 123.8
threshold_3 = 2.00 * mean_count = 165.0

peak_count_type1 = 2  # Weeks with 100, 120
peak_count_type2 = 1  # Week with 150 (exceeds 1.5x but not 2x)
peak_count_type3 = 0  # No weeks exceed 2x

# Slopes
slopes = np.diff(weekly_counts)  # [-5, 15, -5, 25, 20, 20, 30]
max_positive_slope = 30
max_negative_slope = -5
slope_std = 14.4

weekly_range = 150 - 45 = 105
```

### Predictive Interpretation

- **Higher `slope_std`** → More variable engagement → Better grades (r = +0.36)
- **Higher `weekly_range`** → Wider activity spread → Better grades (r = +0.35)
- **Higher `max_positive_slope`** → Bigger increases → Better grades (r = +0.31)

---

## 8. Time-to-Access Features

### Source APIs

```
GET /api/v1/courses/{course_id}/analytics/users/{user_id}/activity
GET /api/v1/courses/{course_id}/modules?student_id={user_id}
GET /api/v1/courses/{course_id}  (for course start date)
```

### Concept (from Oviedo paper)

These features capture **procrastination patterns**:
- How early/late students start engaging with the course
- How quickly they access course modules

### Raw Data Structure

**Course Info:**
```json
{
  "id": 86005,
  "start_at": "2025-07-21T00:00:00Z",
  "end_at": "2025-12-15T23:59:59Z"
}
```

**Module Progress:**
```json
[
  {
    "id": 12345,
    "name": "Module 1: Introduction",
    "state": "completed",
    "completed_at": "2025-08-05T14:30:00Z"
  },
  {
    "id": 12346,
    "name": "Module 2: Fundamentals",
    "state": "completed",
    "completed_at": "2025-08-20T10:00:00Z"
  }
]
```

### Feature Definitions

| Feature | Formula | Interpretation |
|---------|---------|----------------|
| `first_access_day` | Days from course_start to first page_view | Early start vs late start |
| `first_module_day` | Days from course_start to first module completion | Module engagement timing |
| `first_assignment_day` | Days from course_start to first assignment interaction | Assignment engagement timing |
| `access_time_pct` | Geometric mean of (first N access days / course_duration) | Overall access earliness (0-1) |

### Example Calculation

```python
course_start = datetime(2025, 7, 21)
course_end = datetime(2025, 12, 15)
course_duration = (course_end - course_start).days  # 147 days

first_activity = datetime(2025, 8, 1)  # First page view
first_module = datetime(2025, 8, 5)    # First module completed

first_access_day = (first_activity - course_start).days  # 11 days
first_module_day = (first_module - course_start).days    # 15 days

# Access time percentage (using first 5 access timestamps)
first_5_days = [11, 12, 15, 18, 22]  # Days from course start
proportions = [d / 147 for d in first_5_days]  # [0.075, 0.082, 0.102, 0.122, 0.150]
access_time_pct = np.exp(np.mean(np.log(proportions)))  # Geometric mean ≈ 0.102
```

### Predictive Interpretation

- **Lower `first_access_day`** → Earlier start → Better grades (r = -0.12)
- **Lower `first_module_day`** → Earlier module completion → Better grades (r = -0.18)
- **Lower `access_time_pct`** → Earlier overall access pattern → Better grades (r = -0.13)

---

## 9. Raw Aggregate Features

### Source API

```
GET /api/v1/courses/{course_id}/analytics/student_summaries
```

### Raw Data Structure

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

### Feature Definitions

| Feature | Source Field | Interpretation |
|---------|--------------|----------------|
| `total_page_views` | `page_views` | Total course views (clicks) |
| `total_participations` | `participations` | Total participations (forum posts, quiz attempts, etc.) |
| `activity_span_days` | Calculated from timestamps | Days between first and last activity |
| `unique_active_hours` | Calculated from timestamps | Number of distinct hours with activity |

### Predictive Interpretation

- **Higher `total_participations`** → More active engagement → Better grades (r = +0.40)
  - **This is the strongest predictor in the entire feature set (72.5% RF importance)**
- **Higher `total_page_views`** → More viewing → Better grades (r = +0.26)
- **Higher `unique_active_hours`** → More spread engagement → Better grades (r = +0.33)

---

## 10. Teacher/TA Features

### Source APIs

Teachers don't appear in `/analytics` endpoints, so we use:

```
GET /api/v1/courses/{course_id}/enrollments?type[]=TeacherEnrollment
GET /api/v1/courses/{course_id}/enrollments?type[]=TaEnrollment
GET /api/v1/users/{user_id}/page_views?start_time={}&end_time={}
```

### Transformation Process

1. **Get teacher/TA enrollments**: Filter by enrollment type
2. **Fetch page views**: Use page_views API (not analytics)
3. **Filter by course**: Parse URLs to keep only `/courses/{course_id}/` views
4. **Calculate same features**: Apply same feature calculations as students

### Limitations

- **No participation data**: Teachers don't have `participations` metric
- **No tardiness data**: Teachers don't submit assignments
- **URL filtering required**: Page views API returns all activity, must filter by course URL
- **Higher API cost**: One additional API call per teacher

### Available Features for Teachers

All session, time block, trajectory, and workload features can be calculated. Raw aggregates are limited to `total_page_views`, `activity_span_days`, and `unique_active_hours`.

---

## 11. Normalization

### Within-Course Z-Score Normalization

To compare students across different courses (which have different baseline activity levels), features are normalized:

```python
for course_id in unique_courses:
    course_mask = df['course_id'] == course_id
    course_data = df.loc[course_mask]

    for feature in features_to_normalize:
        mean_val = course_data[feature].mean()
        std_val = course_data[feature].std()

        if std_val > 0:
            df.loc[course_mask, f'{feature}_norm'] = (course_data[feature] - mean_val) / std_val
        else:
            df.loc[course_mask, f'{feature}_norm'] = 0
```

### Features Normalized

```python
features_to_normalize = [
    'session_count', 'session_gap_mean', 'session_regularity', 'sessions_per_week',
    'engagement_velocity', 'weekly_cv', 'early_engagement_ratio', 'late_surge',
    'peak_count_type1', 'peak_count_type2', 'peak_count_type3',
    'max_positive_slope', 'max_negative_slope', 'weekly_range',
    'first_access_day', 'total_page_views', 'total_participations'
]
```

### Why Normalize?

- Course A might have 100 page views average, Course B might have 1000
- Raw `total_page_views = 150` in Course A is above average
- Raw `total_page_views = 150` in Course B is well below average
- Normalized score of +1.0 means "1 standard deviation above course average" in both cases

---

## 12. Feature Agglomeration

### Purpose

Reduce 54 features to 6-8 interpretable aggregates for:
- Simpler models
- Reduced overfitting risk
- More interpretable results

### Method: Ward's Hierarchical Clustering

Based on Oviedo et al. approach:

```python
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature Agglomeration
fa = FeatureAgglomeration(n_clusters=8, linkage='ward')
X_reduced = fa.fit_transform(X_scaled)
```

### Resulting Clusters (8)

| Cluster | Name | Features Included |
|---------|------|-------------------|
| 0 | WORKLOAD_DYNAMICS | session_count, sessions_per_week, trend_reversals, max_positive_slope, slope_std, ... |
| 1 | DCT_COEFFICIENTS | weekday_afternoon_pct, dct_coef_1, dct_coef_5, dct_coef_10, dct_coef_11, ... |
| 2 | TIME_BLOCKS | weekday_morning_pct, weekday_morning_sd, dct_coef_3, dct_coef_4, peak_count_type3, ... |
| 3 | DCT_COEFFICIENTS | weekday_evening_pct, weekend_night_pct, dct_coef_7, dct_coef_8, dct_coef_9, ... |
| 4 | TIME_BLOCKS | weekend_morning_pct, weekend_afternoon_pct, weekend_evening_pct, weekend_total_sd, ... |
| 5 | TIME_TO_ACCESS | session_gap_min, session_regularity, weekday_night_pct, early_engagement_ratio, ... |
| 6 | SESSION_REGULARITY | session_gap_max, session_gap_mean, session_gap_std |
| 7 | ENGAGEMENT_TRAJECTORY | weekday_afternoon_sd, dct_coef_0, weekly_cv, late_surge |

### Trade-off

- **Original features (54)**: R² = 0.55 for grade prediction
- **Aggregated features (8)**: R² drops but maintains interpretability
- **Recommendation**: Use original features for prediction, aggregated for interpretation

---

## Appendix: Complete Feature List

| # | Feature | Category | Correlation | Description |
|---|---------|----------|-------------|-------------|
| 1 | session_count | Session | +0.33 | Total study sessions |
| 2 | session_gap_min | Session | -0.24 | Shortest break (hours) |
| 3 | session_gap_max | Session | -0.05 | Longest break (hours) |
| 4 | session_gap_mean | Session | -0.33 | Average break (hours) |
| 5 | session_gap_std | Session | -0.14 | Break variability |
| 6 | session_regularity | Session | -0.28 | Consistency score |
| 7 | sessions_per_week | Session | +0.33 | Weekly frequency |
| 8 | weekday_morning_pct | Time Block | +0.06 | % morning activity |
| 9 | weekday_afternoon_pct | Time Block | -0.17 | % afternoon activity |
| 10 | weekday_evening_pct | Time Block | +0.17 | % evening activity |
| 11 | weekday_night_pct | Time Block | -0.07 | % night activity |
| 12 | weekend_morning_pct | Time Block | +0.08 | % weekend morning |
| 13 | weekend_afternoon_pct | Time Block | +0.24 | % weekend afternoon |
| 14 | weekend_evening_pct | Time Block | +0.15 | % weekend evening |
| 15 | weekend_night_pct | Time Block | -0.03 | % weekend night |
| 16 | weekday_morning_sd | Time Block | +0.12 | Morning consistency |
| 17 | weekday_afternoon_sd | Time Block | +0.09 | Afternoon consistency |
| 18 | weekend_total_sd | Time Block | +0.08 | Weekend consistency |
| 19-30 | dct_coef_0 to _11 | DCT | varies | Periodic patterns |
| 31 | engagement_velocity | Trajectory | +0.01 | Trend slope |
| 32 | engagement_acceleration | Trajectory | -0.03 | Trend curvature |
| 33 | weekly_cv | Trajectory | +0.25 | Weekly variability |
| 34 | trend_reversals | Trajectory | +0.24 | Direction changes |
| 35 | early_engagement_ratio | Trajectory | -0.02 | Front-loading |
| 36 | late_surge | Trajectory | +0.07 | Last-minute cramming |
| 37-39 | peak_count_type1-3 | Workload | +0.05 to +0.19 | Peak counts |
| 40 | peak_ratio | Workload | +0.18 | High/low peak ratio |
| 41 | max_positive_slope | Workload | +0.31 | Biggest increase |
| 42 | max_negative_slope | Workload | -0.31 | Biggest decrease |
| 43 | slope_std | Workload | +0.36 | Slope variability |
| 44 | positive_slope_sum | Workload | +0.27 | Total increases |
| 45 | negative_slope_sum | Workload | -0.27 | Total decreases |
| 46 | weekly_range | Workload | +0.35 | Activity spread |
| 47 | first_access_day | Access | -0.12 | Days to first access |
| 48 | first_module_day | Access | -0.18 | Days to first module |
| 49 | first_assignment_day | Access | N/A | Days to first assignment |
| 50 | access_time_pct | Access | -0.13 | Overall access earliness |
| 51 | total_page_views | Aggregate | +0.26 | Total views |
| 52 | total_participations | Aggregate | **+0.40** | Total participations |
| 53 | activity_span_days | Aggregate | +0.20 | Engagement duration |
| 54 | unique_active_hours | Aggregate | +0.33 | Distinct active hours |

---

*Document generated: December 2025*
*Based on: Oviedo et al., ECTEL-2022, Beyond Time on Task papers*
*Implementation: scripts/engagement_dynamics_features.py*
