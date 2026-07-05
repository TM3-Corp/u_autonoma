#!/usr/bin/env python3
"""
Calculate course-relative timing features for PUC data.

Normalize all timestamps to 0-100% of course duration.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/course_relative_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

RESOURCE_TYPES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules']

def calculate_course_relative_features(df_user, course_start, course_end):
    """Calculate features relative to course timeline (0-100%)."""

    course_duration = (course_end - course_start).total_seconds()
    if course_duration <= 0:
        course_duration = 1  # Avoid division by zero

    # Calculate relative position (0-100%) for each access
    df_user['relative_time'] = (df_user['created_at'] - course_start).dt.total_seconds() / course_duration * 100
    df_user['relative_time'] = df_user['relative_time'].clip(0, 100)

    features = {}

    # Overall timing features
    features['first_access_pct'] = df_user['relative_time'].min()
    features['last_access_pct'] = df_user['relative_time'].max()
    features['mean_access_pct'] = df_user['relative_time'].mean()
    features['median_access_pct'] = df_user['relative_time'].median()
    features['std_access_pct'] = df_user['relative_time'].std()

    # Early/mid/late distribution (0-33%, 33-66%, 66-100%)
    features['early_activity_pct'] = (df_user['relative_time'] <= 33).sum() / len(df_user)
    features['mid_activity_pct'] = ((df_user['relative_time'] > 33) & (df_user['relative_time'] <= 66)).sum() / len(df_user)
    features['late_activity_pct'] = (df_user['relative_time'] > 66).sum() / len(df_user)

    # Per-resource-type timing
    for rtype in RESOURCE_TYPES:
        rtype_data = df_user[df_user['category'] == rtype]

        if len(rtype_data) > 0:
            prefix = rtype[:4]
            features[f'{prefix}_first_access_pct'] = rtype_data['relative_time'].min()
            features[f'{prefix}_mean_access_pct'] = rtype_data['relative_time'].mean()
            features[f'{prefix}_early_pct'] = (rtype_data['relative_time'] <= 33).sum() / len(rtype_data)
        else:
            prefix = rtype[:4]
            features[f'{prefix}_first_access_pct'] = 0
            features[f'{prefix}_mean_access_pct'] = 0
            features[f'{prefix}_early_pct'] = 0

    # Histogram of access times (5 bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
    for i in range(5):
        bin_start = i * 20
        bin_end = (i + 1) * 20
        features[f'timing_hist_b{i}'] = ((df_user['relative_time'] >= bin_start) &
                                          (df_user['relative_time'] < bin_end)).sum() / len(df_user)

    return pd.Series(features)

print("\nCalculating course-relative features...")

all_features = []

for course_id in df['course_id'].unique():
    df_course = df[df['course_id'] == course_id]

    # Course boundaries
    course_start = df_course['course_start'].iloc[0]
    course_end = df_course['created_at'].max()

    # Process each student
    for student_id in df_course['student_id'].unique():
        df_user = df_course[df_course['student_id'] == student_id].copy()

        user_features = calculate_course_relative_features(df_user, course_start, course_end)
        user_features['student_id'] = student_id
        user_features['course_id'] = course_id

        all_features.append(user_features)

features = pd.DataFrame(all_features)

print(f"\nGenerated {len(features)} enrollment-level feature sets")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
