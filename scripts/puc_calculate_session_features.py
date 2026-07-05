#!/usr/bin/env python3
"""
Calculate session-based features from PUC page views.

Session definition: Gap >= 30 minutes between consecutive page views.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/session_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SESSION_GAP_MINUTES = 30

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_sessions(df_user):
    """Calculate session metrics for a user's page views."""
    if len(df_user) < 2:
        return pd.Series({
            'session_count': 1,
            'session_duration_mean': 0,
            'session_duration_std': 0,
            'session_duration_median': 0,
            'sessions_per_week': 0,
            'views_per_session': len(df_user),
            'short_sessions_pct': 0,
            'long_sessions_pct': 0,
            'session_regularity': 0,
            'session_density': 0,
            'session_spread_days': 1,
            'total_views': len(df_user),
            'total_time_min': df_user['interaction_seconds'].sum() / 60
        })

    # Sort by timestamp
    df_user = df_user.sort_values('created_at')
    timestamps = pd.to_datetime(df_user['created_at'])

    # Calculate gaps between consecutive views
    gaps = timestamps.diff().dt.total_seconds() / 60  # in minutes

    # Identify session boundaries (gap >= threshold)
    session_starts = gaps >= SESSION_GAP_MINUTES
    session_starts.iloc[0] = True  # First view starts first session

    # Assign session IDs
    session_ids = session_starts.cumsum()

    # Calculate session metrics
    sessions = []
    for sid, group in df_user.groupby(session_ids):
        ts = pd.to_datetime(group['created_at'])
        duration = (ts.max() - ts.min()).total_seconds() / 60  # minutes
        sessions.append({
            'duration_min': duration,
            'views': len(group),
        })

    sessions_df = pd.DataFrame(sessions)

    # Calculate features
    n_sessions = len(sessions_df)
    durations = sessions_df['duration_min']
    views_per_session = sessions_df['views']

    # Course span in weeks
    total_span = (timestamps.max() - timestamps.min()).days / 7
    total_span = max(total_span, 1)  # Avoid division by zero

    # Calculate unique active days
    unique_days = pd.to_datetime(timestamps.dt.date).nunique()

    # Calculate mean values
    duration_mean = durations.mean()
    views_mean = views_per_session.mean()

    # Session density: clicks per minute
    session_density = views_mean / duration_mean if duration_mean > 0 else 0

    # Session regularity (consistency of inter-session gaps)
    if n_sessions > 1:
        session_starts_times = []
        current_group = df_user.groupby(session_ids)
        for sid, group in current_group:
            session_starts_times.append(group['created_at'].min())
        session_gaps = pd.Series(session_starts_times).diff().dt.total_seconds() / 60
        session_gaps = session_gaps.dropna()
        if len(session_gaps) > 0 and session_gaps.mean() > 0:
            regularity = 1 - (session_gaps.std() / session_gaps.mean())
        else:
            regularity = 0
    else:
        regularity = 0

    features = {
        'session_count': n_sessions,
        'session_duration_mean': duration_mean,
        'session_duration_std': durations.std() if n_sessions > 1 else 0,
        'session_duration_median': durations.median(),
        'sessions_per_week': n_sessions / total_span,
        'views_per_session': views_mean,
        'short_sessions_pct': (durations < 5).sum() / n_sessions,
        'long_sessions_pct': (durations > 30).sum() / n_sessions,
        'session_regularity': max(0, regularity),
        'session_density': session_density,
        'session_spread_days': unique_days,
        'total_views': len(df_user),
        'total_time_min': df_user['interaction_seconds'].sum() / 60
    }

    return pd.Series(features)

print("\nCalculating session features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(calculate_sessions, include_groups=False).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")
print(f"Feature columns: {[c for c in features.columns if c not in ['student_id', 'course_id']]}")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
print(f"  Features: {len([c for c in features.columns if c not in ['student_id', 'course_id']])}")
