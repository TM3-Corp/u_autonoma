#!/usr/bin/env python3
"""
Calculate Proactivity Features using PCT (Percentile) Ranking.

For each resource in a course, students are ranked by WHEN they first accessed it:
- First student to access = PCT 1.0
- Last student to access = PCT near 0
- Students who never accessed = 0

This captures "proactivity" - being among the first to engage with course materials.

Dimensionality Reduction:
Since courses have variable numbers of resources, we aggregate to fixed-size features:
1. Aggregate Statistics: mean, median, std of PCT per resource type
2. Distribution Features: % in top 25%, % never accessed
3. Histogram Features: 5-bin distribution of PCT values
4. DCT Coefficients: If resources ordered temporally (optional)

Output Features per student per course per resource type:
- {type}_mean_pct: Average proactivity percentile
- {type}_access_rate: % of resources accessed
- {type}_top25_rate: % of resources where student was in top 25%
- {type}_histogram_b{1-5}: Distribution across 5 PCT bins
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fftpack import dct
from collections import defaultdict

INPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/categorized_page_views.parquet')
ENROLLMENTS_FILE = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/proactivity_features.parquet')
OUTPUT_DETAILED = Path('/home/paul/projects/uautonoma/data/enriched_features/proactivity_detailed.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Resource types with resource_id available (can do per-resource ranking)
RANKABLE_TYPES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules']

# All resource types (for general access features)
ALL_TYPES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules',
             'announcements', 'home']


def calculate_pct_rankings(df_course, enrolled_users):
    """
    Calculate PCT rankings for all resources in a course.

    Returns: Dict[resource_type][resource_id][user_id] = PCT value (0-1)
    """
    rankings = defaultdict(lambda: defaultdict(dict))

    user_col = 'source_user_id' if 'source_user_id' in df_course.columns else 'user_id'

    for resource_type in RANKABLE_TYPES:
        type_data = df_course[df_course['resource_type'] == resource_type]

        if len(type_data) == 0:
            continue

        # Get unique resources
        resources = type_data['resource_id'].dropna().unique()

        for resource_id in resources:
            resource_data = type_data[type_data['resource_id'] == resource_id]

            # Get first access time per user
            first_access = resource_data.groupby(user_col)['created_at'].min()

            if len(first_access) == 0:
                continue

            # Convert to datetime and sort
            first_access = pd.to_datetime(first_access).sort_values()

            # Rank users (1 = first, N = last)
            n_accessors = len(first_access)

            for rank, (user_id, _) in enumerate(first_access.items(), 1):
                # PCT: First accessor gets 1.0, last gets 1/N
                pct = (n_accessors - rank + 1) / n_accessors
                rankings[resource_type][resource_id][user_id] = pct

            # Users who never accessed get 0
            for user_id in enrolled_users:
                if user_id not in rankings[resource_type][resource_id]:
                    rankings[resource_type][resource_id][user_id] = 0.0

    return rankings


def aggregate_pct_features(rankings, user_id, resource_type):
    """
    Aggregate PCT values for a user across all resources of a type.

    Returns fixed-size feature dict regardless of number of resources.
    """
    features = {}
    prefix = resource_type[:4]  # Short prefix (file, disc, quiz, etc.)

    # Collect all PCT values for this user
    pct_values = []
    resources = rankings.get(resource_type, {})
    n_resources = len(resources)

    for resource_id, user_pcts in resources.items():
        pct = user_pcts.get(user_id, 0.0)
        pct_values.append(pct)

    if n_resources == 0:
        # No resources of this type in course
        features[f'{prefix}_n_resources'] = 0
        features[f'{prefix}_mean_pct'] = 0.0
        features[f'{prefix}_median_pct'] = 0.0
        features[f'{prefix}_std_pct'] = 0.0
        features[f'{prefix}_access_rate'] = 0.0
        features[f'{prefix}_top25_rate'] = 0.0
        features[f'{prefix}_top50_rate'] = 0.0
        # Histogram bins
        for i in range(5):
            features[f'{prefix}_hist_b{i+1}'] = 0.0
        return features

    pct_array = np.array(pct_values)

    # Basic statistics
    features[f'{prefix}_n_resources'] = n_resources
    features[f'{prefix}_mean_pct'] = float(np.mean(pct_array))
    features[f'{prefix}_median_pct'] = float(np.median(pct_array))
    features[f'{prefix}_std_pct'] = float(np.std(pct_array))

    # Access rate (% of resources accessed = PCT > 0)
    n_accessed = np.sum(pct_array > 0)
    features[f'{prefix}_access_rate'] = n_accessed / n_resources

    # Top quartile rates (how often in top 25%, top 50%)
    features[f'{prefix}_top25_rate'] = np.sum(pct_array >= 0.75) / n_resources
    features[f'{prefix}_top50_rate'] = np.sum(pct_array >= 0.50) / n_resources

    # Histogram features (5 bins: 0, 0-25%, 25-50%, 50-75%, 75-100%)
    # Bin 1: PCT = 0 (never accessed)
    # Bin 2: 0 < PCT <= 0.25 (bottom quartile)
    # Bin 3: 0.25 < PCT <= 0.50 (second quartile)
    # Bin 4: 0.50 < PCT <= 0.75 (third quartile)
    # Bin 5: 0.75 < PCT <= 1.0 (top quartile)
    features[f'{prefix}_hist_b1'] = np.sum(pct_array == 0) / n_resources  # Never accessed
    features[f'{prefix}_hist_b2'] = np.sum((pct_array > 0) & (pct_array <= 0.25)) / n_resources
    features[f'{prefix}_hist_b3'] = np.sum((pct_array > 0.25) & (pct_array <= 0.50)) / n_resources
    features[f'{prefix}_hist_b4'] = np.sum((pct_array > 0.50) & (pct_array <= 0.75)) / n_resources
    features[f'{prefix}_hist_b5'] = np.sum(pct_array > 0.75) / n_resources  # Top quartile

    return features


def calculate_download_features(df_course, user_id):
    """Calculate download-specific features (files with /download in URL)."""
    user_col = 'source_user_id' if 'source_user_id' in df_course.columns else 'user_id'

    # Filter to files with download action
    downloads = df_course[
        (df_course['resource_type'] == 'files') &
        (df_course['http_request'].str.contains('/download', case=False, na=False))
    ]

    user_downloads = downloads[downloads[user_col] == user_id]
    total_downloads = len(downloads[user_col].unique())

    features = {
        'download_count': len(user_downloads),
        'unique_files_downloaded': user_downloads['resource_id'].nunique(),
        'download_rate': len(user_downloads) / total_downloads if total_downloads > 0 else 0
    }

    return features


def calculate_dct_features(pct_values, n_coeffs=4):
    """
    Apply DCT to PCT sequence for additional pattern features.

    This captures the "shape" of proactivity across resources.
    """
    features = {f'dct_pct_{i}': 0.0 for i in range(n_coeffs)}

    if len(pct_values) < 4:
        return features

    # Normalize to avoid DC dominance
    pct_array = np.array(pct_values)
    if pct_array.sum() > 0:
        pct_array = pct_array / pct_array.sum()

    # Apply DCT
    try:
        coeffs = dct(pct_array, norm='ortho')
        for i in range(min(n_coeffs, len(coeffs))):
            features[f'dct_pct_{i}'] = float(coeffs[i])
    except:
        pass

    return features


def main():
    print('=' * 60)
    print('Calculating Proactivity Features (PCT Ranking)')
    print('=' * 60)
    print()

    # Load data
    print('Loading data...')
    df = pd.read_parquet(INPUT_FILE)
    print(f'  Page views: {len(df):,}')

    enrollments = pd.read_csv(ENROLLMENTS_FILE)
    print(f'  Enrollments: {len(enrollments)}')
    print()

    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'

    # Filter to our courses
    df = df[df['course_id'].isin(COURSES)].copy()
    print(f'Filtered to {len(COURSES)} courses: {len(df):,} page views')
    print()

    # Calculate features per course
    all_features = []
    all_detailed = []

    for course_id in COURSES:
        print(f'Processing course {course_id}...')

        # Get course data
        df_course = df[df['course_id'] == course_id]

        # Get enrolled users
        course_enrollments = enrollments[enrollments['course_id'] == course_id]
        enrolled_users = set(course_enrollments['user_id'].unique())

        if len(enrolled_users) == 0:
            print(f'  No enrollments, skipping')
            continue

        # Calculate PCT rankings for all resources
        rankings = calculate_pct_rankings(df_course, enrolled_users)

        # Report resource counts
        for rtype in RANKABLE_TYPES:
            n_resources = len(rankings.get(rtype, {}))
            if n_resources > 0:
                print(f'  {rtype}: {n_resources} resources')

        # Calculate features per user
        for user_id in enrolled_users:
            user_features = {
                'user_id': user_id,
                'course_id': course_id
            }

            # Aggregate PCT features per resource type
            all_pct_values = []
            for rtype in RANKABLE_TYPES:
                type_features = aggregate_pct_features(rankings, user_id, rtype)
                user_features.update(type_features)

                # Collect PCT values for global DCT
                for resource_id, user_pcts in rankings.get(rtype, {}).items():
                    pct = user_pcts.get(user_id, 0.0)
                    all_pct_values.append(pct)

            # Download-specific features
            download_features = calculate_download_features(df_course, user_id)
            user_features.update(download_features)

            # Global DCT on all PCT values (captures overall proactivity pattern)
            dct_features = calculate_dct_features(all_pct_values, n_coeffs=4)
            user_features.update(dct_features)

            # Overall proactivity score (mean of means)
            mean_pcts = [user_features.get(f'{rtype[:4]}_mean_pct', 0) for rtype in RANKABLE_TYPES]
            user_features['overall_proactivity'] = np.mean([m for m in mean_pcts if m > 0]) if any(m > 0 for m in mean_pcts) else 0

            all_features.append(user_features)

            # Store detailed PCT values for analysis
            for rtype in RANKABLE_TYPES:
                for resource_id, user_pcts in rankings.get(rtype, {}).items():
                    pct = user_pcts.get(user_id, 0.0)
                    all_detailed.append({
                        'user_id': user_id,
                        'course_id': course_id,
                        'resource_type': rtype,
                        'resource_id': resource_id,
                        'pct': pct
                    })

        print()

    # Create DataFrames
    features_df = pd.DataFrame(all_features)
    print(f'Generated features for {len(features_df)} user-course pairs')
    print(f'Total features: {len(features_df.columns)}')
    print()

    # Summary statistics
    print('=== Proactivity Summary ===')
    for rtype in RANKABLE_TYPES:
        prefix = rtype[:4]
        col = f'{prefix}_mean_pct'
        if col in features_df.columns:
            mean_val = features_df[col].mean()
            std_val = features_df[col].std()
            print(f'{rtype:12s}: mean_pct={mean_val:.3f} +/- {std_val:.3f}')

    print()
    overall_mean = features_df['overall_proactivity'].mean()
    overall_std = features_df['overall_proactivity'].std()
    print(f'Overall proactivity: {overall_mean:.3f} +/- {overall_std:.3f}')
    print()

    # Access rate summary
    print('=== Access Rate Summary ===')
    for rtype in RANKABLE_TYPES:
        prefix = rtype[:4]
        col = f'{prefix}_access_rate'
        if col in features_df.columns:
            mean_val = features_df[col].mean()
            print(f'{rtype:12s}: {mean_val*100:.1f}% of resources accessed on average')
    print()

    # Save
    print(f'Saving features to {OUTPUT_FILE}...')
    features_df.to_parquet(OUTPUT_FILE, index=False)

    if all_detailed:
        detailed_df = pd.DataFrame(all_detailed)
        print(f'Saving detailed PCT values to {OUTPUT_DETAILED}...')
        print(f'  {len(detailed_df):,} resource-level records')
        detailed_df.to_parquet(OUTPUT_DETAILED, index=False)

    print()
    print('Done!')

    # Preview features
    print()
    print('Sample features:')
    sample_cols = ['user_id', 'course_id', 'file_mean_pct', 'disc_mean_pct',
                   'quiz_mean_pct', 'overall_proactivity']
    sample_cols = [c for c in sample_cols if c in features_df.columns]
    print(features_df[sample_cols].head(5).to_string())


if __name__ == '__main__':
    main()
