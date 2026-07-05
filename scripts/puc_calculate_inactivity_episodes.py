#!/usr/bin/env python3
"""
Calculate inactivity episode features from PUC page views.

Inactivity episodes = gaps in student engagement that may signal risk.
Features track consecutive days without activity and recovery patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/inactivity_episode_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_inactivity_episodes(df_user):
    """Calculate inactivity episode metrics for a user's page views."""
    if len(df_user) < 2:
        return pd.Series({
            'inactivity_episodes_gt3days': 0,
            'inactivity_episodes_gt7days': 0,
            'max_consecutive_inactive_days': 0,
            'total_inactive_days': 0,
            'inactive_days_pct': 0.0,
            'recovery_time_after_gap_mean': 0.0,
            'recovery_time_after_gap_median': 0.0,
            'inactivity_episode_count': 0,
            'avg_inactivity_gap_days': 0.0,
        })

    # Sort by timestamp
    df_user = df_user.sort_values('created_at')
    timestamps = pd.to_datetime(df_user['created_at'])

    # Get course duration
    course_start = timestamps.min()
    course_end = timestamps.max()
    course_duration_days = max((course_end - course_start).days, 1)

    # Calculate gaps between consecutive activities (in days)
    gaps_seconds = timestamps.diff().dt.total_seconds()
    gaps_days = gaps_seconds / (24 * 3600)
    gaps_days = gaps_days.dropna()

    if len(gaps_days) == 0:
        return pd.Series({
            'inactivity_episodes_gt3days': 0,
            'inactivity_episodes_gt7days': 0,
            'max_consecutive_inactive_days': 0,
            'total_inactive_days': 0,
            'inactive_days_pct': 0.0,
            'recovery_time_after_gap_mean': 0.0,
            'recovery_time_after_gap_median': 0.0,
            'inactivity_episode_count': 0,
            'avg_inactivity_gap_days': 0.0,
        })

    # Count episodes by threshold
    episodes_gt3days = (gaps_days > 3).sum()
    episodes_gt7days = (gaps_days > 7).sum()

    # Maximum consecutive inactive days
    max_gap = gaps_days.max()

    # Total inactive days (sum of all gaps > 1 day)
    total_inactive = gaps_days[gaps_days > 1].sum()

    # Percentage of course duration spent inactive
    inactive_pct = (total_inactive / course_duration_days) * 100

    # Recovery time after gaps > 5 days
    # (How many days after a long gap until they return to activity)
    recovery_times = []
    gaps_gt5 = gaps_days[gaps_days > 5]

    if len(gaps_gt5) > 0:
        # For each gap > 5 days, measure time until next activity
        for idx in gaps_gt5.index:
            # Find next activity after this gap
            next_activities = gaps_days.loc[idx:].iloc[1:]  # Skip the gap itself
            if len(next_activities) > 0:
                # Recovery time = time until activity resumes (next gap if < 1 day, means active)
                next_gap = next_activities.iloc[0]
                # If next gap is small (<1 day), they recovered quickly
                recovery_times.append(next_gap if next_gap < 1 else 1)

    recovery_mean = np.mean(recovery_times) if recovery_times else 0.0
    recovery_median = np.median(recovery_times) if recovery_times else 0.0

    # Count all inactivity episodes (gaps > 2 days)
    inactivity_episodes = (gaps_days > 2).sum()

    # Average gap size for inactivity episodes
    avg_gap = gaps_days[gaps_days > 2].mean() if (gaps_days > 2).any() else 0.0

    features = {
        'inactivity_episodes_gt3days': episodes_gt3days,
        'inactivity_episodes_gt7days': episodes_gt7days,
        'max_consecutive_inactive_days': max_gap,
        'total_inactive_days': total_inactive,
        'inactive_days_pct': inactive_pct,
        'recovery_time_after_gap_mean': recovery_mean,
        'recovery_time_after_gap_median': recovery_median,
        'inactivity_episode_count': inactivity_episodes,
        'avg_inactivity_gap_days': avg_gap,
    }

    return pd.Series(features)

print("\nCalculating inactivity episode features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(
    calculate_inactivity_episodes,
    include_groups=False
).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")
print(f"Feature columns: {[c for c in features.columns if c not in ['student_id', 'course_id']]}")

# Show statistics
print("\nFeature statistics:")
for col in features.columns:
    if col not in ['student_id', 'course_id']:
        print(f"  {col}: mean={features[col].mean():.2f}, std={features[col].std():.2f}, max={features[col].max():.2f}")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
print(f"  Features: {len([c for c in features.columns if c not in ['student_id', 'course_id']])}")
