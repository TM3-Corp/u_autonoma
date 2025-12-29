#!/usr/bin/env python3
"""
Engagement Dynamics Feature Engineering for Student Success Prediction.

Creates advanced engagement dynamics features from Canvas LMS data including:
- Session regularity features (gap statistics)
- Time block aggregation (8 blocks + 12 DCT coefficients)
- Engagement trajectory features (velocity, acceleration, consistency)
- Workload dynamics features (peaks, slopes)
- Time-to-access features (procrastination indicators)
- Teacher/TA activity features

Based on research from:
- Oviedo et al. (Feature Agglomeration approach)
- ECTEL 2022 (Session-based tactics)
- Beyond Time on Task (Workload dynamics)
"""

import os
import re
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from dotenv import load_dotenv
import requests

# Optional imports with fallbacks
try:
    from scipy.fftpack import dct
    HAS_DCT = True
except ImportError:
    HAS_DCT = False
    print("Warning: scipy not available, DCT features will be skipped")

try:
    from scipy import stats
    HAS_STATS = True
except ImportError:
    HAS_STATS = False

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Test courses with grade data (Control de Gestión - optimal for analysis)
TEST_COURSES = [
    # Original courses
    {'id': 86005, 'name': 'TALL DE COMPETENCIAS DIGITALES-P01'},
    {'id': 86676, 'name': 'FUND DE BUSINESS ANALYTICS-P01'},
    {'id': 84936, 'name': 'FUNDAMENTOS DE MICROECONOMÍA-P03'},
    {'id': 84941, 'name': 'FUNDAMENTOS DE MICROECONOMÍA-P01'},
    # Additional Control de Gestión courses with good data
    {'id': 84944, 'name': 'FUNDAMENTOS DE MACROECONOMÍA-P03'},
    {'id': 86020, 'name': 'TALL DE COMPETENCIAS DIGITALES-P02'},
    {'id': 79804, 'name': 'FUNDAMENTOS TRIBUTARIOS-P01'},
    {'id': 79875, 'name': 'TALLER DE COMP DIGITALES-P01'},
    {'id': 79913, 'name': 'FUND. DE BUSINESS ANALYTICS-P01'},
    {'id': 88381, 'name': 'MATEMÁTICAS PARA LOS NEGOCIOS-P01'},
    {'id': 89099, 'name': 'TALLER DE COMP DIGITALES-P01'},
    {'id': 89390, 'name': 'GESTIÓN DEL TALENTO-P01'},
    {'id': 89736, 'name': 'FUNDAMENTOS DE MACROECONOMÍA-P01'},
]

# Session threshold in hours
SESSION_GAP_THRESHOLD = 1.0  # 60 minutes = new session


@dataclass
class EngagementFeatures:
    """All engagement dynamics features for a student."""
    # Identifiers
    course_id: int = 0
    user_id: int = 0
    user_role: str = 'student'  # student, teacher, ta

    # Target variable
    final_score: float = None
    failed: int = None

    # === SESSION REGULARITY FEATURES ===
    session_count: int = 0
    session_gap_min: float = 0.0
    session_gap_max: float = 0.0
    session_gap_mean: float = 0.0
    session_gap_std: float = 0.0
    session_regularity: float = 0.0  # 1 - (std/mean)
    sessions_per_week: float = 0.0

    # === TIME BLOCK FEATURES (8 blocks) ===
    weekday_morning_pct: float = 0.0   # Mon-Fri 6-12
    weekday_afternoon_pct: float = 0.0  # Mon-Fri 12-18
    weekday_evening_pct: float = 0.0   # Mon-Fri 18-24
    weekday_night_pct: float = 0.0     # Mon-Fri 0-6
    weekend_morning_pct: float = 0.0   # Sat-Sun 6-12
    weekend_afternoon_pct: float = 0.0  # Sat-Sun 12-18
    weekend_evening_pct: float = 0.0   # Sat-Sun 18-24
    weekend_night_pct: float = 0.0     # Sat-Sun 0-6

    # Time block consistency (SD across weeks)
    weekday_morning_sd: float = 0.0
    weekday_afternoon_sd: float = 0.0
    weekend_total_sd: float = 0.0

    # === DCT COEFFICIENTS (12 features) ===
    dct_coef_0: float = 0.0  # DC component (overall activity level)
    dct_coef_1: float = 0.0
    dct_coef_2: float = 0.0
    dct_coef_3: float = 0.0
    dct_coef_4: float = 0.0
    dct_coef_5: float = 0.0
    dct_coef_6: float = 0.0
    dct_coef_7: float = 0.0
    dct_coef_8: float = 0.0
    dct_coef_9: float = 0.0
    dct_coef_10: float = 0.0
    dct_coef_11: float = 0.0

    # === ENGAGEMENT TRAJECTORY FEATURES ===
    engagement_velocity: float = 0.0  # Slope of weekly activity
    engagement_acceleration: float = 0.0  # 2nd derivative
    weekly_cv: float = 0.0  # Coefficient of variation
    trend_reversals: int = 0  # Count of direction changes
    early_engagement_ratio: float = 0.0  # (weeks 1-3) / total
    late_surge: float = 0.0  # (final 2 weeks) / mean of prior

    # === WORKLOAD DYNAMICS FEATURES ===
    peak_count_type1: int = 0  # 1.25x mean
    peak_count_type2: int = 0  # 1.50x mean
    peak_count_type3: int = 0  # 2.00x mean
    peak_ratio: float = 0.0  # type3 / type1
    max_positive_slope: float = 0.0
    max_negative_slope: float = 0.0
    slope_std: float = 0.0
    positive_slope_sum: float = 0.0
    negative_slope_sum: float = 0.0
    weekly_range: float = 0.0  # max - min weekly activity

    # === TIME-TO-ACCESS FEATURES ===
    first_access_day: float = 0.0  # Days from course start
    first_module_day: float = 0.0
    first_assignment_day: float = 0.0
    access_time_pct: float = 0.0  # Geometric mean of first N accesses / course duration

    # === RAW AGGREGATES (for normalization) ===
    total_page_views: int = 0
    total_participations: int = 0
    activity_span_days: int = 0
    unique_active_hours: int = 0


def paginate(url: str, params: dict = None, max_pages: int = 20) -> List[dict]:
    """Paginate through Canvas API results."""
    all_results = []
    params = params or {}
    params['per_page'] = 100
    page = 1

    while page <= max_pages:
        params['page'] = page
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        all_results.extend(data)
        if len(data) < 100:
            break
        page += 1
        time.sleep(0.2)

    return all_results


def get_course_info(course_id: int) -> dict:
    """Get course start/end dates."""
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}', headers=headers)
    if r.status_code != 200:
        return {}
    return r.json()


def get_enrollments(course_id: int, enrollment_type: str = 'StudentEnrollment') -> List[dict]:
    """Get enrollments with grades."""
    return paginate(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        params={
            'type[]': enrollment_type,
            'include[]': ['grades', 'total_activity_time']
        }
    )


def get_student_summaries(course_id: int) -> List[dict]:
    """Get student activity summaries."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries',
        headers=headers,
        params={'per_page': 100}
    )
    if r.status_code != 200:
        return []
    return r.json()


def get_user_activity(course_id: int, user_id: int) -> dict:
    """Get hourly activity for a student."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/users/{user_id}/activity',
        headers=headers
    )
    if r.status_code != 200:
        return {}
    return r.json()


def get_user_page_views(user_id: int, start_time: str, end_time: str, course_id: int = None) -> List[dict]:
    """Get page views for a user (used for teachers/TAs)."""
    page_views = paginate(
        f'{API_URL}/api/v1/users/{user_id}/page_views',
        params={'start_time': start_time, 'end_time': end_time},
        max_pages=10
    )

    # Filter to course if specified
    if course_id and page_views:
        pattern = f'/courses/{course_id}'
        page_views = [pv for pv in page_views if pattern in pv.get('url', '')]

    return page_views


def get_module_progress(course_id: int, user_id: int) -> List[dict]:
    """Get module completion data for a student."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/modules',
        headers=headers,
        params={'student_id': user_id, 'per_page': 50}
    )
    if r.status_code != 200:
        return []
    return r.json()


def parse_timestamps(page_views_dict: Dict[str, int]) -> List[datetime]:
    """Parse page view timestamps into datetime objects."""
    timestamps = []
    for ts_str, count in page_views_dict.items():
        if count <= 0:
            continue
        try:
            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            # Add multiple entries for count > 1
            timestamps.extend([dt] * count)
        except (ValueError, TypeError):
            continue
    timestamps.sort()
    return timestamps


def calculate_session_features(timestamps: List[datetime], course_weeks: float) -> Dict[str, float]:
    """Calculate session regularity features from timestamps."""
    features = {
        'session_count': 0,
        'session_gap_min': 0.0,
        'session_gap_max': 0.0,
        'session_gap_mean': 0.0,
        'session_gap_std': 0.0,
        'session_regularity': 0.0,
        'sessions_per_week': 0.0,
    }

    if len(timestamps) < 2:
        return features

    # Calculate gaps in hours
    gaps = []
    for i in range(1, len(timestamps)):
        gap_hours = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
        gaps.append(gap_hours)

    # Identify session boundaries (gap > threshold = new session)
    session_starts = [0]  # First timestamp starts first session
    for i, gap in enumerate(gaps):
        if gap >= SESSION_GAP_THRESHOLD:
            session_starts.append(i + 1)

    features['session_count'] = len(session_starts)

    # Inter-session gaps (gaps between sessions, not within)
    inter_session_gaps = [g for g in gaps if g >= SESSION_GAP_THRESHOLD]

    if inter_session_gaps:
        features['session_gap_min'] = min(inter_session_gaps)
        features['session_gap_max'] = max(inter_session_gaps)
        features['session_gap_mean'] = np.mean(inter_session_gaps)
        features['session_gap_std'] = np.std(inter_session_gaps) if len(inter_session_gaps) > 1 else 0

        if features['session_gap_mean'] > 0:
            features['session_regularity'] = 1 - (features['session_gap_std'] / features['session_gap_mean'])
            features['session_regularity'] = max(0, min(1, features['session_regularity']))

    if course_weeks > 0:
        features['sessions_per_week'] = features['session_count'] / course_weeks

    return features


def calculate_time_block_features(timestamps: List[datetime]) -> Dict[str, float]:
    """Calculate 8 time block features + consistency measures."""
    features = {
        'weekday_morning_pct': 0.0,
        'weekday_afternoon_pct': 0.0,
        'weekday_evening_pct': 0.0,
        'weekday_night_pct': 0.0,
        'weekend_morning_pct': 0.0,
        'weekend_afternoon_pct': 0.0,
        'weekend_evening_pct': 0.0,
        'weekend_night_pct': 0.0,
        'weekday_morning_sd': 0.0,
        'weekday_afternoon_sd': 0.0,
        'weekend_total_sd': 0.0,
    }

    if not timestamps:
        return features

    # Count by block
    blocks = defaultdict(int)
    weekly_blocks = defaultdict(lambda: defaultdict(int))  # week -> block -> count

    for ts in timestamps:
        is_weekend = ts.weekday() >= 5
        hour = ts.hour

        # Determine time of day
        if 6 <= hour < 12:
            time_slot = 'morning'
        elif 12 <= hour < 18:
            time_slot = 'afternoon'
        elif 18 <= hour < 24:
            time_slot = 'evening'
        else:
            time_slot = 'night'

        day_type = 'weekend' if is_weekend else 'weekday'
        block = f'{day_type}_{time_slot}'
        blocks[block] += 1

        # Track by week for consistency calculation
        week_num = ts.isocalendar()[1]
        weekly_blocks[week_num][block] += 1

    total = sum(blocks.values())
    if total == 0:
        return features

    # Calculate percentages
    for block, count in blocks.items():
        features[f'{block}_pct'] = count / total

    # Calculate weekly consistency (SD of proportions across weeks)
    if len(weekly_blocks) >= 2:
        weekly_totals = {w: sum(blocks.values()) for w, blocks in weekly_blocks.items()}

        # Weekday morning consistency
        wm_props = []
        for week, total_week in weekly_totals.items():
            if total_week > 0:
                wm_props.append(weekly_blocks[week].get('weekday_morning', 0) / total_week)
        if len(wm_props) >= 2:
            features['weekday_morning_sd'] = np.std(wm_props)

        # Weekday afternoon consistency
        wa_props = []
        for week, total_week in weekly_totals.items():
            if total_week > 0:
                wa_props.append(weekly_blocks[week].get('weekday_afternoon', 0) / total_week)
        if len(wa_props) >= 2:
            features['weekday_afternoon_sd'] = np.std(wa_props)

        # Weekend total consistency
        we_props = []
        for week, total_week in weekly_totals.items():
            if total_week > 0:
                weekend_count = (weekly_blocks[week].get('weekend_morning', 0) +
                               weekly_blocks[week].get('weekend_afternoon', 0) +
                               weekly_blocks[week].get('weekend_evening', 0) +
                               weekly_blocks[week].get('weekend_night', 0))
                we_props.append(weekend_count / total_week)
        if len(we_props) >= 2:
            features['weekend_total_sd'] = np.std(we_props)

    return features


def calculate_dct_features(timestamps: List[datetime]) -> Dict[str, float]:
    """Calculate DCT coefficients from weekly activity pattern."""
    features = {f'dct_coef_{i}': 0.0 for i in range(12)}

    if not HAS_DCT or not timestamps:
        return features

    # Build 168-slot weekly activity vector
    weekly_vector = np.zeros(168)

    for ts in timestamps:
        # Map to 0-167 (hour of week)
        day_of_week = ts.weekday()  # 0=Monday
        hour = ts.hour
        slot = day_of_week * 24 + hour
        weekly_vector[slot] += 1

    # Normalize
    total = weekly_vector.sum()
    if total > 0:
        weekly_vector = weekly_vector / total

    # Apply DCT
    dct_coeffs = dct(weekly_vector, norm='ortho')

    # Keep first 12 coefficients
    for i in range(min(12, len(dct_coeffs))):
        features[f'dct_coef_{i}'] = float(dct_coeffs[i])

    return features


def calculate_trajectory_features(timestamps: List[datetime]) -> Dict[str, float]:
    """Calculate engagement trajectory features."""
    features = {
        'engagement_velocity': 0.0,
        'engagement_acceleration': 0.0,
        'weekly_cv': 0.0,
        'trend_reversals': 0,
        'early_engagement_ratio': 0.0,
        'late_surge': 0.0,
    }

    if len(timestamps) < 2:
        return features

    # Group by week
    weekly_counts = defaultdict(int)
    for ts in timestamps:
        week_num = ts.isocalendar()[1]
        weekly_counts[week_num] += 1

    if len(weekly_counts) < 2:
        return features

    # Sort weeks and get counts
    sorted_weeks = sorted(weekly_counts.keys())
    counts = [weekly_counts[w] for w in sorted_weeks]

    # Velocity (linear regression slope)
    x = np.arange(len(counts))
    if len(counts) >= 2:
        slope, intercept = np.polyfit(x, counts, 1)
        features['engagement_velocity'] = slope

        # Acceleration (2nd derivative via 2nd-order polynomial)
        if len(counts) >= 3:
            coeffs = np.polyfit(x, counts, 2)
            features['engagement_acceleration'] = 2 * coeffs[0]

    # Coefficient of variation
    mean_count = np.mean(counts)
    if mean_count > 0:
        features['weekly_cv'] = np.std(counts) / mean_count

    # Trend reversals
    if len(counts) >= 3:
        diffs = np.diff(counts)
        reversals = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i-1] < 0)
        features['trend_reversals'] = reversals

    # Early engagement ratio (first 3 weeks / total)
    early_count = sum(counts[:3]) if len(counts) >= 3 else sum(counts)
    total_count = sum(counts)
    if total_count > 0:
        features['early_engagement_ratio'] = early_count / total_count

    # Late surge (last 2 weeks / mean of prior)
    if len(counts) >= 4:
        late_count = sum(counts[-2:])
        prior_mean = np.mean(counts[:-2])
        if prior_mean > 0:
            features['late_surge'] = late_count / (2 * prior_mean)

    return features


def calculate_workload_dynamics(timestamps: List[datetime]) -> Dict[str, float]:
    """Calculate workload dynamics features (peaks, slopes)."""
    features = {
        'peak_count_type1': 0,
        'peak_count_type2': 0,
        'peak_count_type3': 0,
        'peak_ratio': 0.0,
        'max_positive_slope': 0.0,
        'max_negative_slope': 0.0,
        'slope_std': 0.0,
        'positive_slope_sum': 0.0,
        'negative_slope_sum': 0.0,
        'weekly_range': 0.0,
    }

    if len(timestamps) < 2:
        return features

    # Group by week
    weekly_counts = defaultdict(int)
    for ts in timestamps:
        week_num = ts.isocalendar()[1]
        weekly_counts[week_num] += 1

    if len(weekly_counts) < 2:
        return features

    sorted_weeks = sorted(weekly_counts.keys())
    counts = [weekly_counts[w] for w in sorted_weeks]

    mean_count = np.mean(counts)

    # Peak analysis
    if mean_count > 0:
        for count in counts:
            if count > 2.0 * mean_count:
                features['peak_count_type3'] += 1
            elif count > 1.5 * mean_count:
                features['peak_count_type2'] += 1
            elif count > 1.25 * mean_count:
                features['peak_count_type1'] += 1

        if features['peak_count_type1'] > 0:
            features['peak_ratio'] = features['peak_count_type3'] / features['peak_count_type1']

    # Slope features
    slopes = np.diff(counts)
    if len(slopes) > 0:
        positive_slopes = [s for s in slopes if s > 0]
        negative_slopes = [s for s in slopes if s < 0]

        features['max_positive_slope'] = max(slopes) if slopes.size > 0 else 0
        features['max_negative_slope'] = min(slopes) if slopes.size > 0 else 0
        features['slope_std'] = np.std(slopes)
        features['positive_slope_sum'] = sum(positive_slopes) if positive_slopes else 0
        features['negative_slope_sum'] = sum(negative_slopes) if negative_slopes else 0

    features['weekly_range'] = max(counts) - min(counts) if counts else 0

    return features


def calculate_time_to_access_features(
    timestamps: List[datetime],
    course_start: datetime,
    course_end: datetime,
    modules: List[dict]
) -> Dict[str, float]:
    """Calculate time-to-access features."""
    features = {
        'first_access_day': 0.0,
        'first_module_day': 0.0,
        'first_assignment_day': 0.0,
        'access_time_pct': 0.0,
    }

    if not timestamps or not course_start:
        return features

    course_duration = (course_end - course_start).days if course_end else 120
    if course_duration <= 0:
        course_duration = 120

    # First access day
    first_ts = min(timestamps)
    features['first_access_day'] = max(0, (first_ts - course_start).days)

    # First module completion
    if modules:
        completed_dates = []
        for m in modules:
            completed_at = m.get('completed_at')
            if completed_at:
                try:
                    dt = datetime.fromisoformat(completed_at.replace('Z', '+00:00'))
                    completed_dates.append(dt)
                except (ValueError, TypeError):
                    pass

        if completed_dates:
            first_module = min(completed_dates)
            features['first_module_day'] = max(0, (first_module - course_start).days)

    # Access time percentage (geometric mean of first N access times / course duration)
    if len(timestamps) >= 5 and course_duration > 0:
        first_n_days = [(ts - course_start).days / course_duration for ts in timestamps[:5]]
        first_n_days = [max(0.001, min(1, d)) for d in first_n_days]  # Clip to (0, 1]
        features['access_time_pct'] = np.exp(np.mean(np.log(first_n_days)))

    return features


def extract_student_features(
    course_id: int,
    user_id: int,
    enrollment: dict,
    summary: dict,
    course_start: datetime,
    course_end: datetime
) -> EngagementFeatures:
    """Extract all engagement features for a student."""
    features = EngagementFeatures()
    features.course_id = course_id
    features.user_id = user_id
    features.user_role = 'student'

    # Target variable
    grades = enrollment.get('grades', {})
    final_score = grades.get('final_score')
    if final_score is not None:
        features.final_score = final_score
        features.failed = 1 if final_score < 57 else 0

    # Get hourly activity
    activity = get_user_activity(course_id, user_id)
    page_views_dict = activity.get('page_views', {})
    timestamps = parse_timestamps(page_views_dict)

    # Raw aggregates
    features.total_page_views = summary.get('page_views', 0) if summary else 0
    features.total_participations = summary.get('participations', 0) if summary else 0

    if timestamps:
        features.activity_span_days = (max(timestamps) - min(timestamps)).days
        features.unique_active_hours = len(set(ts.replace(minute=0, second=0, microsecond=0) for ts in timestamps))

    # Calculate course duration in weeks
    course_weeks = ((course_end - course_start).days / 7) if (course_start and course_end) else 15
    course_weeks = max(1, course_weeks)

    # Session regularity features
    session_features = calculate_session_features(timestamps, course_weeks)
    for k, v in session_features.items():
        setattr(features, k, v)

    # Time block features
    time_block_features = calculate_time_block_features(timestamps)
    for k, v in time_block_features.items():
        setattr(features, k, v)

    # DCT features
    dct_features = calculate_dct_features(timestamps)
    for k, v in dct_features.items():
        setattr(features, k, v)

    # Trajectory features
    trajectory_features = calculate_trajectory_features(timestamps)
    for k, v in trajectory_features.items():
        setattr(features, k, v)

    # Workload dynamics
    workload_features = calculate_workload_dynamics(timestamps)
    for k, v in workload_features.items():
        setattr(features, k, v)

    # Time-to-access features
    modules = get_module_progress(course_id, user_id)
    access_features = calculate_time_to_access_features(timestamps, course_start, course_end, modules)
    for k, v in access_features.items():
        setattr(features, k, v)

    return features


def extract_teacher_features(
    course_id: int,
    user_id: int,
    enrollment: dict,
    course_start: datetime,
    course_end: datetime
) -> EngagementFeatures:
    """Extract engagement features for a teacher/TA."""
    features = EngagementFeatures()
    features.course_id = course_id
    features.user_id = user_id

    enrollment_type = enrollment.get('type', '')
    if 'Teacher' in enrollment_type:
        features.user_role = 'teacher'
    elif 'Ta' in enrollment_type:
        features.user_role = 'ta'
    else:
        features.user_role = 'other'

    # Use page views API (no analytics for teachers)
    start_str = course_start.isoformat() if course_start else '2025-01-01T00:00:00Z'
    end_str = course_end.isoformat() if course_end else '2025-12-31T23:59:59Z'

    page_views = get_user_page_views(user_id, start_str, end_str, course_id)

    # Convert page views to timestamps
    timestamps = []
    for pv in page_views:
        created_at = pv.get('created_at')
        if created_at:
            try:
                dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                timestamps.append(dt)
            except (ValueError, TypeError):
                pass
    timestamps.sort()

    features.total_page_views = len(page_views)

    if timestamps:
        features.activity_span_days = (max(timestamps) - min(timestamps)).days
        features.unique_active_hours = len(set(ts.replace(minute=0, second=0, microsecond=0) for ts in timestamps))

    course_weeks = ((course_end - course_start).days / 7) if (course_start and course_end) else 15
    course_weeks = max(1, course_weeks)

    # Calculate same features as students
    session_features = calculate_session_features(timestamps, course_weeks)
    for k, v in session_features.items():
        setattr(features, k, v)

    time_block_features = calculate_time_block_features(timestamps)
    for k, v in time_block_features.items():
        setattr(features, k, v)

    trajectory_features = calculate_trajectory_features(timestamps)
    for k, v in trajectory_features.items():
        setattr(features, k, v)

    return features


def normalize_features(df: pd.DataFrame, features_to_normalize: List[str]) -> pd.DataFrame:
    """Create within-course normalized versions of features."""
    df_normalized = df.copy()

    for course_id in df['course_id'].unique():
        mask = df['course_id'] == course_id
        course_df = df.loc[mask]

        for feat in features_to_normalize:
            if feat in df.columns:
                values = course_df[feat]
                mean_val = values.mean()
                std_val = values.std()

                if std_val > 0:
                    df_normalized.loc[mask, f'{feat}_norm'] = (values - mean_val) / std_val
                else:
                    df_normalized.loc[mask, f'{feat}_norm'] = 0

    return df_normalized


def extract_course_features(course: dict, include_teachers: bool = True) -> Tuple[List[dict], List[dict]]:
    """Extract all features for a course."""
    course_id = course['id']
    course_name = course['name']

    print(f"\n{'='*60}")
    print(f"Extracting features for: {course_name} ({course_id})")
    print(f"{'='*60}")

    # Get course info
    course_info = get_course_info(course_id)
    start_at = course_info.get('start_at') or course_info.get('created_at')
    end_at = course_info.get('end_at')

    course_start = None
    course_end = None

    if start_at:
        try:
            course_start = datetime.fromisoformat(start_at.replace('Z', '+00:00'))
        except ValueError:
            pass

    if end_at:
        try:
            course_end = datetime.fromisoformat(end_at.replace('Z', '+00:00'))
        except ValueError:
            pass

    if not course_start:
        course_start = datetime.now().astimezone() - timedelta(days=120)
    if not course_end:
        course_end = datetime.now().astimezone()

    print(f"  Course period: {course_start.date()} to {course_end.date()}")

    # Get student enrollments and summaries
    enrollments = get_enrollments(course_id, 'StudentEnrollment')
    summaries = get_student_summaries(course_id)
    summaries_dict = {s['id']: s for s in summaries}

    print(f"  Student enrollments: {len(enrollments)}")
    print(f"  Activity summaries: {len(summaries)}")

    # Extract student features
    student_features = []
    for i, enrollment in enumerate(enrollments):
        user_id = enrollment['user_id']
        summary = summaries_dict.get(user_id)

        features = extract_student_features(
            course_id, user_id, enrollment, summary, course_start, course_end
        )
        student_features.append(asdict(features))

        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(enrollments)} students...")
        time.sleep(0.3)  # Rate limiting

    print(f"  Extracted features for {len(student_features)} students")

    # Extract teacher features
    teacher_features = []
    if include_teachers:
        teacher_enrollments = get_enrollments(course_id, 'TeacherEnrollment')
        ta_enrollments = get_enrollments(course_id, 'TaEnrollment')
        all_instructor_enrollments = teacher_enrollments + ta_enrollments

        print(f"  Teachers/TAs: {len(all_instructor_enrollments)}")

        for enrollment in all_instructor_enrollments:
            user_id = enrollment['user_id']
            features = extract_teacher_features(course_id, user_id, enrollment, course_start, course_end)
            teacher_features.append(asdict(features))
            time.sleep(0.3)

        print(f"  Extracted features for {len(teacher_features)} teachers/TAs")

    return student_features, teacher_features


def main():
    """Main extraction pipeline."""
    print("\n" + "=" * 70)
    print("ENGAGEMENT DYNAMICS FEATURE EXTRACTION")
    print("=" * 70)

    all_student_features = []
    all_teacher_features = []

    for course in TEST_COURSES:
        student_features, teacher_features = extract_course_features(course, include_teachers=True)
        all_student_features.extend(student_features)
        all_teacher_features.extend(teacher_features)

    # Create DataFrames
    df_students = pd.DataFrame(all_student_features)
    df_teachers = pd.DataFrame(all_teacher_features)

    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total students: {len(df_students)}")
    print(f"Total teachers/TAs: {len(df_teachers)}")

    # Normalize features
    features_to_normalize = [
        'session_count', 'session_gap_mean', 'session_regularity', 'sessions_per_week',
        'engagement_velocity', 'weekly_cv', 'early_engagement_ratio', 'late_surge',
        'peak_count_type1', 'peak_count_type2', 'peak_count_type3',
        'max_positive_slope', 'max_negative_slope', 'weekly_range',
        'first_access_day', 'total_page_views', 'total_participations'
    ]

    df_students = normalize_features(df_students, features_to_normalize)

    # Save results
    output_dir = 'data/engagement_dynamics'
    os.makedirs(output_dir, exist_ok=True)

    df_students.to_csv(f'{output_dir}/student_features.csv', index=False)
    df_teachers.to_csv(f'{output_dir}/teacher_features.csv', index=False)

    print(f"\nSaved to {output_dir}/")
    print(f"  - student_features.csv ({len(df_students)} rows, {len(df_students.columns)} columns)")
    print(f"  - teacher_features.csv ({len(df_teachers)} rows, {len(df_teachers.columns)} columns)")

    # Quick correlation analysis
    if 'final_score' in df_students.columns and df_students['final_score'].notna().sum() >= 10:
        print(f"\n{'='*70}")
        print("TOP CORRELATIONS WITH FINAL SCORE")
        print("=" * 70)

        feature_cols = [c for c in df_students.columns
                       if c not in ['course_id', 'user_id', 'user_role', 'final_score', 'failed']
                       and not c.endswith('_norm')]

        correlations = {}
        for col in feature_cols:
            valid = df_students[[col, 'final_score']].dropna()
            if len(valid) >= 10:
                corr = valid[col].corr(valid['final_score'])
                if not np.isnan(corr):
                    correlations[col] = corr

        sorted_corrs = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\n{'Feature':<30} {'Correlation':>12}")
        print("-" * 44)
        for feat, corr in sorted_corrs[:15]:
            print(f"{feat:<30} {corr:>+12.3f}")

        # Save correlations
        with open(f'{output_dir}/feature_correlations.json', 'w') as f:
            json.dump(correlations, f, indent=2)
        print(f"\nSaved correlations to {output_dir}/feature_correlations.json")

    return df_students, df_teachers


if __name__ == '__main__':
    main()
