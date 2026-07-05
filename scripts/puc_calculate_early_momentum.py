#!/usr/bin/env python3
"""
Calculate early momentum features from PUC page views.

Early momentum = student behavior in first days/weeks of course.
Fast engagement often correlates with success.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/early_momentum_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_early_momentum(df_user):
    """Calculate early engagement metrics for a user's page views."""
    if len(df_user) == 0:
        return pd.Series({
            'days_to_first_access': np.nan,
            'first_3days_views': 0,
            'first_3days_views_pct': 0.0,
            'first_week_views': 0,
            'first_week_views_pct': 0.0,
            'first_week_activity_rate': 0.0,
            'stall_in_first_2weeks': 0,
            'early_deceleration': 1.0,
            'first_day_views': 0,
            'first_day_time_minutes': 0.0,
            'days_until_10_views': np.nan,
            'early_activity_density': 0.0,
        })

    # Sort by timestamp
    df_user = df_user.sort_values('created_at')
    timestamps = pd.to_datetime(df_user['created_at'])

    # Get course start date
    if 'course_start' in df_user.columns:
        course_start = pd.to_datetime(df_user['course_start'].iloc[0])
    else:
        # Use first access as proxy
        course_start = timestamps.min()

    # Days to first access
    first_access = timestamps.min()
    days_to_first = (first_access - course_start).days

    # Total views
    total_views = len(df_user)

    # First 3 days features
    first_3days_end = course_start + pd.Timedelta(days=3)
    first_3days_df = df_user[timestamps <= first_3days_end]
    first_3days_views = len(first_3days_df)
    first_3days_pct = (first_3days_views / total_views) * 100 if total_views > 0 else 0.0

    # First week features
    first_week_end = course_start + pd.Timedelta(days=7)
    first_week_df = df_user[timestamps <= first_week_end]
    first_week_views = len(first_week_df)
    first_week_pct = (first_week_views / total_views) * 100 if total_views > 0 else 0.0

    # Calculate average weekly views for comparison
    course_duration = (timestamps.max() - course_start).days
    course_weeks = max(course_duration / 7, 1)
    avg_weekly_views = total_views / course_weeks
    first_week_rate = first_week_views / avg_weekly_views if avg_weekly_views > 0 else 0.0

    # Check for stalls in first 2 weeks (gap > 5 days)
    first_2weeks_end = course_start + pd.Timedelta(days=14)
    first_2weeks_df = df_user[timestamps <= first_2weeks_end]

    stall_flag = 0
    if len(first_2weeks_df) > 1:
        first_2weeks_gaps = timestamps[timestamps <= first_2weeks_end].diff()
        max_gap_days = first_2weeks_gaps.max().days if not first_2weeks_gaps.empty else 0
        stall_flag = 1 if max_gap_days > 5 else 0

    # Early deceleration: Week 2 views / Week 1 views
    week1_end = course_start + pd.Timedelta(days=7)
    week2_end = course_start + pd.Timedelta(days=14)

    week1_views = len(df_user[(timestamps > course_start) & (timestamps <= week1_end)])
    week2_views = len(df_user[(timestamps > week1_end) & (timestamps <= week2_end)])

    early_decel = week2_views / week1_views if week1_views > 0 else 1.0

    # First day metrics
    first_day_end = course_start + pd.Timedelta(days=1)
    first_day_df = df_user[timestamps <= first_day_end]
    first_day_views = len(first_day_df)
    first_day_time = first_day_df['interaction_seconds'].sum() / 60 if 'interaction_seconds' in df_user.columns else 0.0

    # Days until 10 views
    if total_views >= 10:
        tenth_view_time = timestamps.iloc[9]
        days_until_10 = (tenth_view_time - course_start).days
    else:
        days_until_10 = np.nan

    # Early activity density (views per day in first week)
    early_density = first_week_views / 7 if course_duration >= 7 else first_week_views

    features = {
        'days_to_first_access': days_to_first,
        'first_3days_views': first_3days_views,
        'first_3days_views_pct': first_3days_pct,
        'first_week_views': first_week_views,
        'first_week_views_pct': first_week_pct,
        'first_week_activity_rate': first_week_rate,
        'stall_in_first_2weeks': stall_flag,
        'early_deceleration': early_decel,
        'first_day_views': first_day_views,
        'first_day_time_minutes': first_day_time,
        'days_until_10_views': days_until_10,
        'early_activity_density': early_density,
    }

    return pd.Series(features)

print("\nCalculating early momentum features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(
    calculate_early_momentum,
    include_groups=False
).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")
print(f"Feature columns: {[c for c in features.columns if c not in ['student_id', 'course_id']]}")

# Show statistics
print("\nFeature statistics:")
for col in features.columns:
    if col not in ['student_id', 'course_id']:
        non_null = features[col].dropna()
        if len(non_null) > 0:
            print(f"  {col}: mean={non_null.mean():.2f}, std={non_null.std():.2f}, max={non_null.max():.2f}")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
print(f"  Features: {len([c for c in features.columns if c not in ['student_id', 'course_id']])}")
