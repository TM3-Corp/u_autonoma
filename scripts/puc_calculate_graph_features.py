#!/usr/bin/env python3
"""
Calculate graph-based features (resource coverage, diversity) for PUC data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/graph_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

def calculate_graph_features(df_user, df_course):
    """Calculate resource coverage and diversity features."""

    # Total unique resources in course
    total_resources = df_course['resource_id'].nunique()

    # Resources accessed by this user
    user_resources = df_user['resource_id'].nunique()

    # Coverage rate
    coverage_rate = user_resources / total_resources if total_resources > 0 else 0

    # Category diversity (how many different categories accessed)
    categories_accessed = df_user['category'].nunique()
    total_categories = df_course['category'].nunique()
    category_diversity = categories_accessed / total_categories if total_categories > 0 else 0

    # Depth: avg visits per unique resource
    depth = len(df_user) / user_resources if user_resources > 0 else 0

    # Breadth: unique resources per session (using hour blocks as proxy)
    resources_per_hour = df_user.groupby(df_user['created_at'].dt.floor('H'))['resource_id'].nunique()
    breadth = resources_per_hour.mean() if len(resources_per_hour) > 0 else 0

    # Resource repetition rate (% of views that are revisits)
    total_views = len(df_user)
    unique_views = user_resources
    repetition_rate = (total_views - unique_views) / total_views if total_views > 0 else 0

    # Resource concentration (Gini coefficient of visit distribution)
    resource_counts = df_user['resource_id'].value_counts().values
    if len(resource_counts) > 1:
        # Gini calculation
        sorted_counts = np.sort(resource_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_counts)) / (n * cumsum[-1]) - (n + 1) / n
    else:
        gini = 0

    return pd.Series({
        'resource_coverage_rate': coverage_rate,
        'category_diversity': category_diversity,
        'resource_depth': depth,
        'resource_breadth': breadth,
        'resource_repetition_rate': repetition_rate,
        'resource_concentration_gini': gini
    })

print("\nCalculating graph features...")

all_features = []

for course_id in df['course_id'].unique():
    df_course = df[df['course_id'] == course_id]

    for student_id in df_course['student_id'].unique():
        df_user = df_course[df_course['student_id'] == student_id]

        user_features = calculate_graph_features(df_user, df_course)
        user_features['student_id'] = student_id
        user_features['course_id'] = course_id

        all_features.append(user_features)

features = pd.DataFrame(all_features)

print(f"\nGenerated {len(features)} enrollment-level feature sets")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
