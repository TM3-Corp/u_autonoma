#!/usr/bin/env python3
"""
Calculate Proactivity Features using PCT (Percentile) Ranking for PUC data.

For each resource in a course, students are ranked by WHEN they first accessed it:
- First student to access = PCT 1.0
- Last student to access = PCT near 0
- Students who never accessed = 0
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/proactivity_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Resource types to track
RESOURCE_TYPES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules']

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_pct_rankings(df_course, enrolled_users):
    """Calculate PCT rankings for all resources in a course."""
    rankings = defaultdict(lambda: defaultdict(dict))

    for resource_type in RESOURCE_TYPES:
        type_data = df_course[df_course['category'] == resource_type]

        if len(type_data) == 0:
            continue

        # Get unique resources
        resources = type_data['resource_id'].dropna().unique()

        for resource_id in resources:
            resource_data = type_data[type_data['resource_id'] == resource_id]

            # Get first access time per user
            first_access = resource_data.groupby('student_id')['created_at'].min()

            if len(first_access) == 0:
                continue

            # Sort by time
            first_access = first_access.sort_values()

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
    """Aggregate PCT values for a user across all resources of a type."""
    features = {}
    prefix = resource_type[:4]  # files->file, discussions->disc, etc.

    # Collect all PCT values for this user
    pct_values = []
    resources = rankings.get(resource_type, {})
    n_resources = len(resources)

    for resource_id, user_pcts in resources.items():
        pct = user_pcts.get(user_id, 0.0)
        pct_values.append(pct)

    if n_resources == 0:
        # No resources of this type in course
        return {
            f'{prefix}_mean_pct': 0.0,
            f'{prefix}_median_pct': 0.0,
            f'{prefix}_std_pct': 0.0,
            f'{prefix}_access_rate': 0.0,
            f'{prefix}_top25_rate': 0.0,
            f'{prefix}_hist_b0': 0.0,
            f'{prefix}_hist_b1': 0.0,
            f'{prefix}_hist_b2': 0.0,
            f'{prefix}_hist_b3': 0.0,
            f'{prefix}_hist_b4': 0.0,
        }

    pct_array = np.array(pct_values)

    # Basic statistics
    features[f'{prefix}_mean_pct'] = float(np.mean(pct_array))
    features[f'{prefix}_median_pct'] = float(np.median(pct_array))
    features[f'{prefix}_std_pct'] = float(np.std(pct_array))

    # Access rate (% of resources accessed = PCT > 0)
    n_accessed = np.sum(pct_array > 0)
    features[f'{prefix}_access_rate'] = n_accessed / n_resources

    # Top quartile rate (how often in top 25%)
    features[f'{prefix}_top25_rate'] = np.sum(pct_array >= 0.75) / n_resources

    # Histogram features (5 bins: 0, 0-25%, 25-50%, 50-75%, 75-100%)
    features[f'{prefix}_hist_b0'] = np.sum(pct_array == 0) / n_resources  # Never accessed
    features[f'{prefix}_hist_b1'] = np.sum((pct_array > 0) & (pct_array <= 0.25)) / n_resources
    features[f'{prefix}_hist_b2'] = np.sum((pct_array > 0.25) & (pct_array <= 0.50)) / n_resources
    features[f'{prefix}_hist_b3'] = np.sum((pct_array > 0.50) & (pct_array <= 0.75)) / n_resources
    features[f'{prefix}_hist_b4'] = np.sum(pct_array > 0.75) / n_resources  # Top quartile

    return features

print("\nCalculating proactivity features per course...")

all_features = []

for course_id in df['course_id'].unique():
    print(f"  Processing course {course_id}...")

    df_course = df[df['course_id'] == course_id]
    enrolled_users = df_course['student_id'].unique()

    # Calculate PCT rankings for this course
    rankings = calculate_pct_rankings(df_course, enrolled_users)

    # Generate features for each student
    for user_id in enrolled_users:
        user_features = {'student_id': user_id, 'course_id': course_id}

        # Aggregate features for each resource type
        for rtype in RESOURCE_TYPES:
            type_features = aggregate_pct_features(rankings, user_id, rtype)
            user_features.update(type_features)

        all_features.append(user_features)

features = pd.DataFrame(all_features)

print(f"\nGenerated {len(features)} enrollment-level feature sets")
print(f"Feature columns: {len([c for c in features.columns if c not in ['student_id', 'course_id']])}")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
