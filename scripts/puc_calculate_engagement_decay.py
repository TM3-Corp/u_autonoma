#!/usr/bin/env python3
"""
Calculate engagement decay features from PUC page views.

Engagement decay = trend in activity level over time.
Failing students often show declining engagement as course progresses.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/engagement_decay_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_engagement_decay(df_user):
    """Calculate engagement trend metrics for a user's page views."""
    if len(df_user) < 7:  # Need at least a week of data for trend
        return pd.Series({
            'engagement_decline_slope': 0.0,
            'engagement_decline_r2': 0.0,
            'engagement_polynomial_slope': 0.0,
            'first_half_vs_second_half_ratio': 1.0,
            'weekly_decline_rate': 0.0,
            'activity_fade_score': 1.0,
            'early_vs_late_ratio': 1.0,
            'peak_activity_week': 0,
            'weeks_since_peak': 0,
        })

    # Sort by timestamp
    df_user = df_user.sort_values('created_at')
    timestamps = pd.to_datetime(df_user['created_at'])

    # Calculate daily view counts
    daily_views = df_user.groupby(timestamps.dt.date).size()
    daily_dates = pd.to_datetime(daily_views.index)

    # Create continuous date range (fill missing days with 0)
    date_range = pd.date_range(daily_dates.min(), daily_dates.max(), freq='D')
    daily_views_continuous = daily_views.reindex(date_range, fill_value=0)

    # X-axis: days from start
    days_from_start = np.arange(len(daily_views_continuous))
    views = daily_views_continuous.values

    # Linear regression for decline slope
    if len(days_from_start) > 1 and views.std() > 0:
        slope, intercept, r_value, p_value, std_err = stats.linregress(days_from_start, views)
        r2 = r_value ** 2
    else:
        slope = 0.0
        r2 = 0.0

    # Polynomial regression (quadratic) for non-linear decay
    if len(days_from_start) > 2:
        try:
            poly_coeffs = np.polyfit(days_from_start, views, deg=2)
            # Derivative at midpoint gives trend
            midpoint = len(days_from_start) / 2
            poly_slope = 2 * poly_coeffs[0] * midpoint + poly_coeffs[1]
        except:
            poly_slope = 0.0
    else:
        poly_slope = 0.0

    # First half vs second half ratio
    midpoint_idx = len(df_user) // 2
    first_half_views = len(df_user[:midpoint_idx])
    second_half_views = len(df_user[midpoint_idx:])
    half_ratio = first_half_views / max(second_half_views, 1)

    # Weekly decline rate
    weekly_views = df_user.groupby(df_user['created_at'].dt.to_period('W')).size()
    if len(weekly_views) > 1:
        weekly_changes = weekly_views.pct_change().dropna()
        weekly_decline = weekly_changes.mean() * 100  # Percentage
    else:
        weekly_decline = 0.0

    # Activity fade score: recent activity / early activity
    # Recent = last 25%, early = first 25%
    n_total = len(df_user)
    early_quarter = n_total // 4
    late_quarter = n_total - (n_total // 4)

    early_activity = len(df_user[:early_quarter])
    late_activity = len(df_user[late_quarter:])
    fade_score = late_activity / max(early_activity, 1)

    # Early vs late ratio (first 33% vs last 33%)
    early_third = n_total // 3
    late_third = n_total - (n_total // 3)
    early_vs_late = len(df_user[:early_third]) / max(len(df_user[late_third:]), 1)

    # Peak activity week and weeks since peak
    if len(weekly_views) > 0:
        peak_period = weekly_views.idxmax()
        last_period = weekly_views.index[-1]
        # Get week numbers by converting to ordinal
        peak_week = peak_period.to_timestamp().isocalendar()[1]
        last_week = last_period.to_timestamp().isocalendar()[1]
        weeks_since_peak = max(0, len(weekly_views) - list(weekly_views.index).index(peak_period) - 1)
    else:
        peak_week = 0
        weeks_since_peak = 0

    features = {
        'engagement_decline_slope': slope,
        'engagement_decline_r2': r2,
        'engagement_polynomial_slope': poly_slope,
        'first_half_vs_second_half_ratio': half_ratio,
        'weekly_decline_rate': weekly_decline,
        'activity_fade_score': fade_score,
        'early_vs_late_ratio': early_vs_late,
        'peak_activity_week': peak_week,
        'weeks_since_peak': weeks_since_peak,
    }

    return pd.Series(features)

print("\nCalculating engagement decay features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(
    calculate_engagement_decay,
    include_groups=False
).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")
print(f"Feature columns: {[c for c in features.columns if c not in ['student_id', 'course_id']]}")

# Show statistics
print("\nFeature statistics:")
for col in features.columns:
    if col not in ['student_id', 'course_id']:
        print(f"  {col}: mean={features[col].mean():.2f}, std={features[col].std():.2f}")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
print(f"  Features: {len([c for c in features.columns if c not in ['student_id', 'course_id']])}")
