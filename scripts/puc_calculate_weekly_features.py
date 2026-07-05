#!/usr/bin/env python3
"""
Calculate weekly activity features from PUC page views.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/weekly_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_weekly_features(df_user):
    """Calculate weekly activity features."""
    weeks = df_user.groupby('week_number_from_start').size()

    if len(weeks) == 0:
        return pd.Series({
            'active_weeks_count': 0,
            'first_active_week': 0,
            'last_active_week': 0,
            'peak_week': 0,
            'peak_week_views': 0,
            'early_late_ratio': 0,
            'activity_consistency': 0,
            'weeks_since_last_activity': 0,
            'avg_views_per_week': 0,
            'week_activity_std': 0,
            'early_weeks_views': 0,
            'mid_weeks_views': 0,
            'late_weeks_views': 0,
            'early_engagement_rate': 0,
            'late_engagement_rate': 0,
            'engagement_trend': 0
        })

    features = {
        'active_weeks_count': len(weeks),
        'first_active_week': weeks.index.min(),
        'last_active_week': weeks.index.max(),
        'peak_week': weeks.idxmax(),
        'peak_week_views': weeks.max(),
        'avg_views_per_week': weeks.mean(),
        'week_activity_std': weeks.std() if len(weeks) > 1 else 0,
    }

    # Early vs late activity (first 3 weeks vs rest)
    early_views = df_user[df_user['week_number_from_start'] <= 3]['week_number_from_start'].count()
    late_views = df_user[df_user['week_number_from_start'] > 3]['week_number_from_start'].count()

    features['early_late_ratio'] = early_views / late_views if late_views > 0 else 0

    # Activity consistency (inverse of CV)
    if weeks.mean() > 0:
        features['activity_consistency'] = 1 - (weeks.std() / weeks.mean())
    else:
        features['activity_consistency'] = 0

    # Time since last activity
    max_week = df_user['week_number_from_start'].max()
    features['weeks_since_last_activity'] = max_week - features['last_active_week']

    # Split into thirds
    total_weeks = max_week
    third = total_weeks / 3

    features['early_weeks_views'] = len(df_user[df_user['week_number_from_start'] <= third])
    features['mid_weeks_views'] = len(df_user[(df_user['week_number_from_start'] > third) &
                                                (df_user['week_number_from_start'] <= 2*third)])
    features['late_weeks_views'] = len(df_user[df_user['week_number_from_start'] > 2*third])

    # Engagement rates per third
    total_views = len(df_user)
    features['early_engagement_rate'] = features['early_weeks_views'] / total_views if total_views > 0 else 0
    features['late_engagement_rate'] = features['late_weeks_views'] / total_views if total_views > 0 else 0

    # Engagement trend (late/early)
    features['engagement_trend'] = features['late_engagement_rate'] / features['early_engagement_rate'] if features['early_engagement_rate'] > 0 else 0

    return pd.Series(features)

print("\nCalculating weekly features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(calculate_weekly_features, include_groups=False).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
