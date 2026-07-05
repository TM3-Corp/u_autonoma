#!/usr/bin/env python3
"""
Calculate Course-Relative Time Normalization Features.

This script normalizes all temporal features to 0-100% of the course's actual
activity span (from first to last student interaction), rather than using
term start dates which can be unreliable.

Key Features:
1. Time Progression (5): When student started/ended as % of course
2. Early Patterns (7): Activity in first 10%, 20%, 33% of course
3. Per-Resource Timing (48): When each resource type was first accessed, with histogram binning
4. Temporal Engagement Curve (7): 5-bin activity distribution + trend
5. Enhanced Session Features (4): Gap statistics relative to course duration

Output:
    data/enriched_features/course_relative_features.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from scipy import stats as scipy_stats

# Paths
BASE_DIR = Path(__file__).parent.parent
INPUT_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
OUTPUT_FILE = BASE_DIR / "data/enriched_features/course_relative_features.parquet"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Model courses
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Resource types for per-resource features
RESOURCE_TYPES = ['files', 'discussions', 'assignments', 'quizzes', 'modules', 'pages']

# Session gap threshold (30 minutes)
SESSION_GAP_MINUTES = 30


def calculate_course_boundaries(df):
    """
    Calculate course start/end from actual student activity.

    Returns:
        Dict[course_id] -> {'start': datetime, 'end': datetime, 'duration': timedelta}
    """
    boundaries = {}

    for course_id in df['course_id'].unique():
        course_data = df[df['course_id'] == course_id]

        if len(course_data) == 0:
            continue

        start = course_data['created_at'].min()
        end = course_data['created_at'].max()
        duration = end - start

        # Avoid division by zero (minimum 1 hour)
        if duration.total_seconds() < 3600:
            duration = timedelta(hours=1)

        boundaries[course_id] = {
            'start': start,
            'end': end,
            'duration': duration,
            'duration_days': duration.total_seconds() / 86400
        }

    return boundaries


def calculate_relative_time(timestamp, course_start, course_duration):
    """Convert timestamp to percentage of course duration (0-100)."""
    elapsed = timestamp - course_start
    pct = (elapsed.total_seconds() / course_duration.total_seconds()) * 100
    return max(0, min(100, pct))  # Clamp to 0-100


def calculate_time_progression_features(df_user, course_start, course_duration):
    """
    Calculate time progression features (5 features).

    Features:
        - first_access_pct: When student first accessed (0-100%)
        - last_access_pct: When student last accessed
        - activity_span_pct: Coverage of course duration
        - median_activity_pct: Central tendency of activity
        - activity_std_pct: Spread of activity over time
    """
    features = {
        'first_access_pct': 0.0,
        'last_access_pct': 0.0,
        'activity_span_pct': 0.0,
        'median_activity_pct': 0.0,
        'activity_std_pct': 0.0,
    }

    if len(df_user) == 0:
        return features

    timestamps = df_user['created_at']

    # Convert all to relative percentages
    relative_pcts = [
        calculate_relative_time(ts, course_start, course_duration)
        for ts in timestamps
    ]

    if len(relative_pcts) == 0:
        return features

    features['first_access_pct'] = min(relative_pcts)
    features['last_access_pct'] = max(relative_pcts)
    features['activity_span_pct'] = features['last_access_pct'] - features['first_access_pct']
    features['median_activity_pct'] = float(np.median(relative_pcts))
    features['activity_std_pct'] = float(np.std(relative_pcts)) if len(relative_pcts) > 1 else 0.0

    return features


def calculate_early_pattern_features(df_user, course_start, course_duration, relative_pcts=None):
    """
    Calculate early pattern features (7 features).

    Features at 10%, 20%, 33% thresholds:
        - early_X_views_pct: % of views in first X% of course
        - early_X_sessions: Sessions in first X%
        - early_10_resource_types: Unique resource types accessed early
        - early_engagement_intensity: Relative early intensity
    """
    features = {
        'early_10_views_pct': 0.0,
        'early_20_views_pct': 0.0,
        'early_33_views_pct': 0.0,
        'early_10_sessions': 0,
        'early_20_sessions': 0,
        'early_10_resource_types': 0,
        'early_engagement_intensity': 0.0,
    }

    if len(df_user) == 0:
        return features

    # Calculate relative percentages if not provided
    if relative_pcts is None:
        relative_pcts = [
            calculate_relative_time(ts, course_start, course_duration)
            for ts in df_user['created_at']
        ]

    total_views = len(relative_pcts)

    # Views in early periods
    early_10 = [p for p in relative_pcts if p <= 10]
    early_20 = [p for p in relative_pcts if p <= 20]
    early_33 = [p for p in relative_pcts if p <= 33]

    features['early_10_views_pct'] = len(early_10) / total_views * 100 if total_views > 0 else 0
    features['early_20_views_pct'] = len(early_20) / total_views * 100 if total_views > 0 else 0
    features['early_33_views_pct'] = len(early_33) / total_views * 100 if total_views > 0 else 0

    # Sessions in early periods (using 30-min gap)
    timestamps = df_user['created_at'].sort_values()
    relative_pcts_sorted = [
        calculate_relative_time(ts, course_start, course_duration)
        for ts in timestamps
    ]

    # Count sessions in early periods
    early_10_sessions = 0
    early_20_sessions = 0
    current_session_start_pct = None

    for i, (ts, pct) in enumerate(zip(timestamps, relative_pcts_sorted)):
        if i == 0:
            current_session_start_pct = pct
            if pct <= 10:
                early_10_sessions = 1
            if pct <= 20:
                early_20_sessions = 1
        else:
            prev_ts = timestamps.iloc[i-1]
            gap_minutes = (ts - prev_ts).total_seconds() / 60

            if gap_minutes >= SESSION_GAP_MINUTES:
                # New session
                current_session_start_pct = pct
                if pct <= 10:
                    early_10_sessions += 1
                if pct <= 20:
                    early_20_sessions += 1

    features['early_10_sessions'] = early_10_sessions
    features['early_20_sessions'] = early_20_sessions

    # Resource types in first 10%
    if 'resource_type' in df_user.columns:
        df_user_copy = df_user.copy()
        df_user_copy['rel_pct'] = relative_pcts
        early_resources = df_user_copy[df_user_copy['rel_pct'] <= 10]['resource_type'].nunique()
        features['early_10_resource_types'] = early_resources

    # Early engagement intensity: (views in first 20%) / (expected 20% of views)
    # If student has uniform activity, this would be 1.0
    expected_early = total_views * 0.20
    if expected_early > 0:
        features['early_engagement_intensity'] = len(early_20) / expected_early

    return features


def calculate_resource_time_features(df_user, course_start, course_duration):
    """
    Calculate per-resource timing features (48 features).

    For each resource type:
        - {type}_mean_access_pct: Mean first-access time as % of course
        - {type}_median_access_pct: Median first-access time
        - {type}_std_access_pct: Spread of access timing
        - {type}_early_access_rate: % accessed in first 20%
        - {type}_timing_hist_b{1-5}: 5-bin histogram (0-20%, 20-40%, etc.)
    """
    features = {}

    for resource_type in RESOURCE_TYPES:
        prefix = resource_type[:4]  # file, disc, assi, quiz, modu, page

        # Initialize with zeros
        features[f'{prefix}_mean_access_pct'] = 0.0
        features[f'{prefix}_median_access_pct'] = 0.0
        features[f'{prefix}_std_access_pct'] = 0.0
        features[f'{prefix}_early_access_rate'] = 0.0
        for i in range(1, 6):
            features[f'{prefix}_timing_hist_b{i}'] = 0.0

        # Filter to this resource type
        if 'resource_type' not in df_user.columns:
            continue

        type_data = df_user[df_user['resource_type'] == resource_type]

        if len(type_data) == 0:
            continue

        # Get first access time per resource
        if 'resource_id' in type_data.columns:
            first_access_per_resource = type_data.groupby('resource_id')['created_at'].min()
        else:
            # If no resource_id, treat all as one resource
            first_access_per_resource = pd.Series([type_data['created_at'].min()])

        if len(first_access_per_resource) == 0:
            continue

        # Convert to relative percentages
        access_pcts = [
            calculate_relative_time(ts, course_start, course_duration)
            for ts in first_access_per_resource
        ]

        n_resources = len(access_pcts)

        # Statistics
        features[f'{prefix}_mean_access_pct'] = float(np.mean(access_pcts))
        features[f'{prefix}_median_access_pct'] = float(np.median(access_pcts))
        features[f'{prefix}_std_access_pct'] = float(np.std(access_pcts)) if n_resources > 1 else 0.0

        # Early access rate (% accessed in first 20% of course)
        early_access = sum(1 for p in access_pcts if p <= 20)
        features[f'{prefix}_early_access_rate'] = early_access / n_resources if n_resources > 0 else 0.0

        # 5-bin histogram (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
        access_array = np.array(access_pcts)
        features[f'{prefix}_timing_hist_b1'] = np.sum(access_array <= 20) / n_resources
        features[f'{prefix}_timing_hist_b2'] = np.sum((access_array > 20) & (access_array <= 40)) / n_resources
        features[f'{prefix}_timing_hist_b3'] = np.sum((access_array > 40) & (access_array <= 60)) / n_resources
        features[f'{prefix}_timing_hist_b4'] = np.sum((access_array > 60) & (access_array <= 80)) / n_resources
        features[f'{prefix}_timing_hist_b5'] = np.sum(access_array > 80) / n_resources

    return features


def calculate_temporal_curve_features(df_user, course_start, course_duration):
    """
    Calculate temporal engagement curve features (7 features).

    Features:
        - activity_bin_{1-5}: % of views in each 20% time slice
        - engagement_curve_slope: Linear trend
        - engagement_curve_trend: Categorical (1=increasing, 2=decreasing, 3=flat, 4=peak_middle)
    """
    features = {
        'activity_bin_1': 0.0,
        'activity_bin_2': 0.0,
        'activity_bin_3': 0.0,
        'activity_bin_4': 0.0,
        'activity_bin_5': 0.0,
        'engagement_curve_slope': 0.0,
        'engagement_curve_trend': 3,  # Default: flat
    }

    if len(df_user) == 0:
        return features

    # Calculate relative percentages
    relative_pcts = [
        calculate_relative_time(ts, course_start, course_duration)
        for ts in df_user['created_at']
    ]

    total_views = len(relative_pcts)

    # Count views in each bin
    bin_counts = [0, 0, 0, 0, 0]
    for pct in relative_pcts:
        if pct <= 20:
            bin_counts[0] += 1
        elif pct <= 40:
            bin_counts[1] += 1
        elif pct <= 60:
            bin_counts[2] += 1
        elif pct <= 80:
            bin_counts[3] += 1
        else:
            bin_counts[4] += 1

    # Convert to percentages
    for i in range(5):
        features[f'activity_bin_{i+1}'] = bin_counts[i] / total_views * 100 if total_views > 0 else 0.0

    # Calculate slope (linear regression of bin percentages)
    if total_views > 0:
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([features[f'activity_bin_{i}'] for i in range(1, 6)])

        if np.std(y) > 0:
            slope, _, _, _, _ = scipy_stats.linregress(x, y)
            features['engagement_curve_slope'] = float(slope)

            # Determine trend
            if slope > 5:  # Significant increase
                features['engagement_curve_trend'] = 1  # increasing
            elif slope < -5:  # Significant decrease
                features['engagement_curve_trend'] = 2  # decreasing
            elif y[2] > max(y[0], y[4]) * 1.2:  # Middle peak
                features['engagement_curve_trend'] = 4  # peak_middle
            else:
                features['engagement_curve_trend'] = 3  # flat

    return features


def calculate_enhanced_session_features(df_user, course_start, course_duration):
    """
    Calculate enhanced session features (4 features).

    Features:
        - session_gap_cv: Coefficient of variation (std/mean)
        - session_gap_median_hours: Median gap between sessions
        - session_gap_max_days: Longest gap in days
        - longest_inactive_period_pct: Longest gap as % of course duration
    """
    features = {
        'session_gap_cv': 0.0,
        'session_gap_median_hours': 0.0,
        'session_gap_max_days': 0.0,
        'longest_inactive_period_pct': 0.0,
    }

    if len(df_user) < 2:
        return features

    # Sort by timestamp
    timestamps = df_user['created_at'].sort_values()

    # Calculate gaps between consecutive activities
    gaps_seconds = []
    for i in range(1, len(timestamps)):
        gap = (timestamps.iloc[i] - timestamps.iloc[i-1]).total_seconds()
        gaps_seconds.append(gap)

    if len(gaps_seconds) == 0:
        return features

    # Identify session boundaries (gaps >= 30 minutes)
    session_gaps_hours = [g / 3600 for g in gaps_seconds if g >= SESSION_GAP_MINUTES * 60]

    if len(session_gaps_hours) > 0:
        gap_mean = np.mean(session_gaps_hours)
        gap_std = np.std(session_gaps_hours) if len(session_gaps_hours) > 1 else 0

        features['session_gap_cv'] = gap_std / gap_mean if gap_mean > 0 else 0.0
        features['session_gap_median_hours'] = float(np.median(session_gaps_hours))

    # Max gap (in days)
    max_gap_seconds = max(gaps_seconds)
    features['session_gap_max_days'] = max_gap_seconds / 86400

    # Longest inactive period as % of course duration
    course_duration_seconds = course_duration.total_seconds()
    features['longest_inactive_period_pct'] = (max_gap_seconds / course_duration_seconds * 100) if course_duration_seconds > 0 else 0.0

    return features


def process_student(df_user, course_start, course_duration):
    """Process all features for a single student."""
    all_features = {}

    # Time progression features (5)
    all_features.update(calculate_time_progression_features(df_user, course_start, course_duration))

    # Early pattern features (7)
    all_features.update(calculate_early_pattern_features(df_user, course_start, course_duration))

    # Per-resource time features (48)
    all_features.update(calculate_resource_time_features(df_user, course_start, course_duration))

    # Temporal curve features (7)
    all_features.update(calculate_temporal_curve_features(df_user, course_start, course_duration))

    # Enhanced session features (4)
    all_features.update(calculate_enhanced_session_features(df_user, course_start, course_duration))

    return all_features


def main():
    print("=" * 70)
    print("Course-Relative Time Normalization Features")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet(INPUT_FILE)
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)

    # Convert to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Filter to model courses
    df = df[df['course_id'].isin(MODEL_COURSES)]

    print(f"  Page views: {len(df):,} records")
    print(f"  Enrollments: {len(df_enroll)} records")
    print(f"  Courses: {df['course_id'].nunique()}")

    # Calculate course boundaries from actual activity
    print("\nCalculating course boundaries...")
    boundaries = calculate_course_boundaries(df)

    for course_id, bounds in boundaries.items():
        print(f"  Course {course_id}: {bounds['duration_days']:.1f} days "
              f"({bounds['start'].strftime('%Y-%m-%d')} to {bounds['end'].strftime('%Y-%m-%d')})")

    # Process each student
    print("\nProcessing students...")

    # Determine user column
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'

    results = []
    total_students = len(df_enroll)

    for idx, (_, enrollment) in enumerate(df_enroll.iterrows()):
        user_id = enrollment['user_id']
        course_id = enrollment['course_id']

        if course_id not in boundaries:
            continue

        bounds = boundaries[course_id]

        # Get user's page views
        df_user = df[(df[user_col] == user_id) & (df['course_id'] == course_id)]

        # Calculate all features
        features = process_student(df_user, bounds['start'], bounds['duration'])

        # Add identifiers
        features['user_id'] = user_id
        features['course_id'] = course_id

        results.append(features)

        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{total_students} students...")

    # Create DataFrame
    df_features = pd.DataFrame(results)

    # Reorder columns (identifiers first)
    cols = ['user_id', 'course_id'] + [c for c in df_features.columns if c not in ['user_id', 'course_id']]
    df_features = df_features[cols]

    # Save
    df_features.to_parquet(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 70}")
    print("OUTPUT SUMMARY")
    print("=" * 70)
    print(f"  Records: {len(df_features)}")
    print(f"  Features: {len(df_features.columns) - 2}")  # Exclude user_id, course_id
    print(f"  Output: {OUTPUT_FILE}")

    # Print feature groups
    feature_groups = {
        'Time Progression': [c for c in df_features.columns if 'access_pct' in c or 'span_pct' in c or 'std_pct' in c],
        'Early Patterns': [c for c in df_features.columns if c.startswith('early_')],
        'Resource Timing': [c for c in df_features.columns if '_timing_hist_' in c or '_early_access_rate' in c],
        'Temporal Curve': [c for c in df_features.columns if 'activity_bin_' in c or 'curve_' in c],
        'Session Enhanced': [c for c in df_features.columns if 'session_gap_' in c or 'inactive_period' in c],
    }

    print(f"\nFeature Groups:")
    for group, features in feature_groups.items():
        print(f"  {group}: {len(features)} features")

    # Show sample statistics
    print(f"\nSample Statistics (mean ± std):")
    key_features = ['first_access_pct', 'early_20_views_pct', 'engagement_curve_slope',
                    'longest_inactive_period_pct', 'session_gap_cv']
    for feat in key_features:
        if feat in df_features.columns:
            mean_val = df_features[feat].mean()
            std_val = df_features[feat].std()
            print(f"  {feat}: {mean_val:.2f} ± {std_val:.2f}")


if __name__ == "__main__":
    main()
