#!/usr/bin/env python3
"""
Calculate session-based features from page views.

Session definition: Gap >= 30 minutes between consecutive page views.

Features generated per student per course:
- session_count: Total number of sessions
- session_duration_mean: Average session duration (minutes)
- session_duration_std: Variability of session duration
- session_duration_median: Median session duration
- sessions_per_week: Sessions per week of course
- views_per_session: Average page views per session
- short_sessions_pct: % of sessions < 5 minutes
- long_sessions_pct: % of sessions > 30 minutes
- session_regularity: 1 - (std/mean) of inter-session gaps
- session_density: Clicks per minute (views_per_session / session_duration_mean)
- session_spread_days: Number of unique days with activity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta

INPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/categorized_page_views.parquet')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/session_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Session threshold: 30 minutes
SESSION_GAP_MINUTES = 30

# Our courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]


def calculate_sessions(df_user):
    """Calculate session metrics for a user's page views."""
    if len(df_user) < 2:
        return None

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
            'session_id': sid,
            'start': ts.min(),
            'end': ts.max(),
            'duration_min': duration,
            'views': len(group),
        })

    if not sessions:
        return None

    sessions_df = pd.DataFrame(sessions)

    # Calculate inter-session gaps
    session_gaps = sessions_df['start'].diff().dt.total_seconds() / 60
    session_gaps = session_gaps[session_gaps >= SESSION_GAP_MINUTES]

    # Calculate features
    n_sessions = len(sessions_df)
    durations = sessions_df['duration_min']
    views_per_session = sessions_df['views']

    # Course span in weeks
    total_span = (timestamps.max() - timestamps.min()).days / 7
    total_span = max(total_span, 1)  # Avoid division by zero

    # Calculate unique active days
    unique_days = pd.to_datetime(timestamps.dt.date).nunique()

    # Calculate mean values for density
    duration_mean = durations.mean()
    views_mean = views_per_session.mean()

    # Session density: clicks per minute (avoid div by zero)
    session_density = views_mean / duration_mean if duration_mean > 0 else 0

    features = {
        'session_count': n_sessions,
        'session_duration_mean': duration_mean,
        'session_duration_std': durations.std() if n_sessions > 1 else 0,
        'session_duration_median': durations.median(),
        'sessions_per_week': n_sessions / total_span,
        'views_per_session': views_mean,
        'short_sessions_pct': (durations < 5).sum() / n_sessions * 100,
        'long_sessions_pct': (durations > 30).sum() / n_sessions * 100,
        'total_views': len(df_user),
        'total_time_min': durations.sum(),
        'session_density': session_density,
        'session_spread_days': unique_days,
    }

    # Session regularity (based on inter-session gaps)
    if len(session_gaps) > 1:
        gap_mean = session_gaps.mean()
        gap_std = session_gaps.std()
        features['session_regularity'] = max(0, 1 - (gap_std / gap_mean)) if gap_mean > 0 else 0
    else:
        features['session_regularity'] = 0

    return features


def main():
    print('Loading categorized page views...')
    df = pd.read_parquet(INPUT_FILE)
    print(f'Loaded {len(df)} page views')

    # Get user_id column
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'

    # Filter to our courses
    df = df[df['course_id'].isin(COURSES)].copy()
    print(f'Filtered to our courses: {len(df)} page views')
    print()

    # Calculate features per user per course
    print('Calculating session features...')
    all_features = []

    for (user_id, course_id), group in df.groupby([user_col, 'course_id']):
        features = calculate_sessions(group)
        if features:
            features['user_id'] = user_id
            features['course_id'] = course_id
            all_features.append(features)

    features_df = pd.DataFrame(all_features)
    print(f'Generated features for {len(features_df)} user-course pairs')
    print()

    # Summary stats
    print('Feature summary:')
    print(features_df[['session_count', 'session_duration_mean', 'sessions_per_week', 'views_per_session', 'session_density', 'session_spread_days']].describe())
    print()

    # Save
    print(f'Saving to {OUTPUT_FILE}...')
    features_df.to_parquet(OUTPUT_FILE, index=False)
    print('Done!')


if __name__ == '__main__':
    main()
