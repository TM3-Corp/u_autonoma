#!/usr/bin/env python3
"""
Calculate time-of-day features from PUC page views.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/time_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_time_features(df_user):
    """Calculate time-of-day and day-of-week features."""
    if len(df_user) == 0:
        return pd.Series({
            'morning_pct': 0, 'afternoon_pct': 0, 'evening_pct': 0, 'night_pct': 0,
            'weekend_pct': 0, 'work_hours_ratio': 0, 'late_night_ratio': 0,
            'peak_hour': 0, 'peak_day': 0, 'hour_diversity': 0,
            'time_consistency': 0, 'day_diversity': 0
        })

    total = len(df_user)

    # Hour-based features (using 'hour' column)
    hours = df_user['hour']

    morning = ((hours >= 6) & (hours < 12)).sum() / total  # 6am-12pm
    afternoon = ((hours >= 12) & (hours < 18)).sum() / total  # 12pm-6pm
    evening = ((hours >= 18) & (hours < 24)).sum() / total  # 6pm-12am
    night = (hours < 6).sum() / total  # 12am-6am

    work_hours = ((hours >= 9) & (hours < 17)).sum() / total  # 9am-5pm
    late_night = ((hours >= 0) & (hours < 4)).sum() / total  # 12am-4am

    # Peak hour and diversity
    hour_counts = hours.value_counts()
    peak_hour = hour_counts.idxmax() if len(hour_counts) > 0 else 0
    hour_diversity = hour_counts.nunique() / 24  # Normalized

    # Time consistency (inverse of hour dispersion)
    time_consistency = 1 - (hours.std() / 12) if hours.std() > 0 else 0

    # Day-of-week features (using 'day_of_week' column)
    days = df_user['day_of_week']
    weekend = (days >= 5).sum() / total  # Saturday=5, Sunday=6

    # Peak day and diversity
    day_counts = days.value_counts()
    peak_day = day_counts.idxmax() if len(day_counts) > 0 else 0
    day_diversity = day_counts.nunique() / 7

    features = {
        'morning_pct': morning,
        'afternoon_pct': afternoon,
        'evening_pct': evening,
        'night_pct': night,
        'weekend_pct': weekend,
        'work_hours_ratio': work_hours,
        'late_night_ratio': late_night,
        'peak_hour': peak_hour,
        'peak_day': peak_day,
        'hour_diversity': hour_diversity,
        'time_consistency': max(0, time_consistency),
        'day_diversity': day_diversity
    }

    return pd.Series(features)

print("\nCalculating time features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(calculate_time_features, include_groups=False).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
