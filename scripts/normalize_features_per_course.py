#!/usr/bin/env python3
"""
Per-Course Feature Normalization

Z-scores features within each course to remove course-specific scale differences.
This helps with cross-course generalization (LOCO validation).

Features normalized:
- Raw counts (total_views, session_count, etc.)
- Time metrics (total_time_min, session_duration_*)
- Category counts (files_views, discussions_views, etc.)

Features NOT normalized (already scale-invariant):
- PCT rankings (already 0-1)
- Percentages (*_pct, *_rate)
- PCA components (already standardized)
- Ratios
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path('/home/paul/projects/uautonoma/data')
ENRICHED_DIR = DATA_DIR / 'enriched_features'

# Features that should be z-scored per course
FEATURES_TO_NORMALIZE = [
    # Session features (raw counts/times)
    'session_count', 'total_views', 'total_time_min',
    'session_duration_mean', 'session_duration_std', 'session_duration_median',
    'views_per_session', 'sessions_per_week',

    # Category counts (raw)
    'files_views', 'discussions_views', 'pages_views', 'modules_views',
    'announcements_views', 'home_views', 'other_views',
    'files_unique_resources', 'discussions_unique_resources',
    'pages_unique_resources', 'modules_unique_resources',
    'files_time_min', 'discussions_time_min', 'pages_time_min',

    # Weekly activity (raw counts)
    'early_semester_views', 'late_semester_views',
    'active_weeks_count', 'peak_week',
]

# Features that should NOT be normalized (already normalized or ratios)
SKIP_PATTERNS = [
    '_pct',      # Percentages (including course-relative _access_pct, _views_pct)
    '_rate',     # Rates
    '_ratio',    # Ratios
    '_pc',       # PCA components
    'pct_',      # PCT rankings
    'mean_pct', 'median_pct', 'std_pct',  # PCT aggregates
    'hist_b',    # Histogram bins (proportions)
    'dct_',      # DCT coefficients (normalized)
    'regularity', # Already a ratio
    'consistency', # Coefficient of variation
    'activity_bin_',  # Temporal curve bins (already percentages)
    'curve_slope',    # Engagement curve slope (relative)
    'curve_trend',    # Categorical
    '_cv',            # Coefficient of variation
]


def should_normalize(col):
    """Check if a column should be z-scored."""
    col_lower = col.lower()

    # Skip if matches skip patterns
    for pattern in SKIP_PATTERNS:
        if pattern in col_lower:
            return False

    # Normalize if in explicit list
    if col in FEATURES_TO_NORMALIZE:
        return True

    # Normalize raw count/time features not in skip list
    if any(x in col_lower for x in ['_views', '_count', '_time', 'session_', '_unique']):
        for pattern in SKIP_PATTERNS:
            if pattern in col_lower:
                return False
        return True

    return False


def normalize_per_course(df, feature_cols):
    """Z-score features within each course."""
    df_normalized = df.copy()
    normalized_cols = []

    for col in feature_cols:
        if should_normalize(col):
            # Z-score within course
            df_normalized[f'{col}_znorm'] = df.groupby('course_id')[col].transform(
                lambda x: (x - x.median()) / (x.quantile(0.75) - x.quantile(0.25) + 1e-6)
            )
            normalized_cols.append(f'{col}_znorm')

    return df_normalized, normalized_cols


def main():
    print('=' * 60)
    print('PER-COURSE FEATURE NORMALIZATION')
    print('=' * 60)
    print()

    # Load all feature files
    session_df = pd.read_parquet(ENRICHED_DIR / 'session_features.parquet')
    category_df = pd.read_parquet(ENRICHED_DIR / 'category_features.parquet')
    proact_df = pd.read_parquet(ENRICHED_DIR / 'proactivity_features.parquet')
    pca_df = pd.read_parquet(ENRICHED_DIR / 'pca_features.parquet')
    weekly_df = pd.read_parquet(ENRICHED_DIR / 'weekly_features.parquet')

    # Load course-relative features (time normalized to 0-100% of course duration)
    course_rel_path = ENRICHED_DIR / 'course_relative_features.parquet'
    if course_rel_path.exists():
        course_rel_df = pd.read_parquet(course_rel_path)
        print(f'Loaded course-relative features: {len(course_rel_df.columns) - 2} features')
    else:
        course_rel_df = None
        print('Course-relative features not found, skipping...')

    # Merge all
    merged = session_df.merge(category_df, on=['user_id', 'course_id'], how='outer')
    merged = merged.merge(proact_df, on=['user_id', 'course_id'], how='left')
    merged = merged.merge(pca_df, on=['user_id', 'course_id'], how='left')

    # Weekly (select non-assessment columns)
    weekly_cols = ['user_id', 'course_id', 'active_weeks_count', 'first_active_week',
                   'last_active_week', 'peak_week', 'early_semester_views', 'late_semester_views',
                   'early_vs_late_ratio', 'avg_week_over_week_change', 'activity_consistency',
                   'engagement_pattern']
    weekly_cols = [c for c in weekly_cols if c in weekly_df.columns]
    merged = merged.merge(weekly_df[weekly_cols], on=['user_id', 'course_id'], how='left')

    # Merge course-relative features
    if course_rel_df is not None:
        merged = merged.merge(course_rel_df, on=['user_id', 'course_id'], how='left')

    print(f'Total samples: {len(merged)}')
    print(f'Total features: {len(merged.columns) - 2}')  # -2 for user_id, course_id
    print(f'Unique courses: {merged["course_id"].nunique()}')
    print()

    # Get feature columns
    feature_cols = [c for c in merged.columns if c not in ['user_id', 'course_id']]

    # Identify which to normalize
    to_normalize = [c for c in feature_cols if should_normalize(c)]
    to_keep = [c for c in feature_cols if not should_normalize(c)]

    print(f'Features to normalize (z-score per course): {len(to_normalize)}')
    for col in to_normalize[:10]:
        print(f'  - {col}')
    if len(to_normalize) > 10:
        print(f'  ... and {len(to_normalize) - 10} more')
    print()

    print(f'Features to keep as-is: {len(to_keep)}')
    print()

    # Normalize
    merged_normalized, new_cols = normalize_per_course(merged, feature_cols)

    print(f'New normalized columns created: {len(new_cols)}')

    # Save
    output_path = ENRICHED_DIR / 'normalized_features.parquet'
    merged_normalized.to_parquet(output_path, index=False)
    print(f'\nSaved to: {output_path}')

    # Show stats for a sample column
    if 'total_views_znorm' in merged_normalized.columns:
        print('\nSample: total_views normalization by course')
        for course_id in merged_normalized['course_id'].unique()[:3]:
            mask = merged_normalized['course_id'] == course_id
            raw = merged_normalized.loc[mask, 'total_views']
            norm = merged_normalized.loc[mask, 'total_views_znorm']
            print(f'  Course {course_id}: raw [{raw.min():.0f}-{raw.max():.0f}] -> norm [{norm.min():.2f}-{norm.max():.2f}]')


if __name__ == '__main__':
    main()
