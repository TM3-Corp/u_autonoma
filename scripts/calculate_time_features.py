#!/usr/bin/env python3
"""
Time-of-Day Feature Extraction

Extracts temporal patterns from page views:
- pct_night_activity: % of activity during night hours (22:00-06:00)
- pct_weekend_activity: % of activity on weekends
- pct_morning_activity: % during morning (06:00-12:00)
- pct_afternoon_activity: % during afternoon (12:00-18:00)
- pct_evening_activity: % during evening (18:00-22:00)
- peak_hour: Most common hour of activity
- study_regularity: How consistent study times are

Based on SOTA: Time-of-day engagement patterns as predictive features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

DATA_DIR = Path('/home/paul/projects/uautonoma/data')
PAGE_VIEWS_DIR = DATA_DIR / 'page_views'
ENRICHED_DIR = DATA_DIR / 'enriched_features'

# Target courses
TARGET_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]


def calculate_time_features(df_user):
    """Calculate time-of-day features for a user's page views."""
    if len(df_user) < 5:
        return None

    # Extract hour and day of week
    hours = df_user['created_at'].dt.hour
    days = df_user['created_at'].dt.dayofweek  # 0=Monday, 6=Sunday

    features = {}

    # Time of day percentages
    total = len(df_user)
    features['pct_night'] = ((hours >= 22) | (hours < 6)).sum() / total
    features['pct_morning'] = ((hours >= 6) & (hours < 12)).sum() / total
    features['pct_afternoon'] = ((hours >= 12) & (hours < 18)).sum() / total
    features['pct_evening'] = ((hours >= 18) & (hours < 22)).sum() / total

    # Weekend vs weekday
    features['pct_weekend'] = (days >= 5).sum() / total

    # Peak hour (most common)
    hour_counts = hours.value_counts()
    features['peak_hour'] = hour_counts.idxmax()

    # Peak day of week
    day_counts = days.value_counts()
    features['peak_day'] = day_counts.idxmax()

    # Hour diversity (entropy-like)
    hour_dist = hours.value_counts(normalize=True)
    features['hour_diversity'] = -sum(p * np.log2(p + 1e-10) for p in hour_dist) / np.log2(24)

    # Study regularity (coefficient of variation of hours)
    if hours.std() > 0:
        features['time_consistency'] = 1 - (hours.std() / 12)  # Normalize by half-day
    else:
        features['time_consistency'] = 1.0

    # Late night study indicator (strong predictor of struggle)
    features['late_night_ratio'] = ((hours >= 0) & (hours < 4)).sum() / total

    # Working hours ratio (9-17)
    features['work_hours_ratio'] = ((hours >= 9) & (hours < 17)).sum() / total

    return features


def main():
    print('=' * 60)
    print('TIME-OF-DAY FEATURE EXTRACTION')
    print('=' * 60)
    print()

    # Load categorized page views
    pv_path = PAGE_VIEWS_DIR / 'categorized_page_views.parquet'
    if not pv_path.exists():
        print(f'ERROR: {pv_path} not found')
        return

    print('Loading page views...')
    df = pd.read_parquet(pv_path)

    # Use source_user_id for matching
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'
    df['user_id'] = df[user_col]

    # Filter to target courses
    df = df[df['course_id'].isin(TARGET_COURSES)].copy()

    print(f'  Loaded {len(df):,} page views (target courses)')
    print(f'  Unique students: {df["user_id"].nunique()}')
    print(f'  Unique courses: {df["course_id"].nunique()}')

    # Parse timestamps
    df['created_at'] = pd.to_datetime(df['created_at'])

    print()

    # Calculate features per user-course
    print('Calculating time-of-day features...')
    all_features = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        features = calculate_time_features(group)
        if features:
            features['user_id'] = user_id
            features['course_id'] = course_id
            all_features.append(features)

    features_df = pd.DataFrame(all_features)
    print(f'  Created {len(features_df)} rows, {len(features_df.columns)} columns')

    # Save
    output_path = ENRICHED_DIR / 'time_features.parquet'
    features_df.to_parquet(output_path, index=False)
    print(f'\nSaved to: {output_path}')

    # Summary stats
    print('\nFeature Summary:')
    for col in ['pct_night', 'pct_weekend', 'pct_morning', 'late_night_ratio', 'work_hours_ratio']:
        if col in features_df.columns:
            print(f'  {col}: mean={features_df[col].mean():.3f}, std={features_df[col].std():.3f}')


if __name__ == '__main__':
    main()
