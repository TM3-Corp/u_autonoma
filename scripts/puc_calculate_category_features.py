#!/usr/bin/env python3
"""
Calculate category-based features from PUC page views.

Features per category (files, discussions, quizzes, assignments, pages, modules, grades, announcements).
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/category_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

# Categories of interest (matching U Autonoma naming)
CATEGORIES = [
    'files', 'discussions', 'quizzes', 'assignments',
    'pages', 'modules', 'grades', 'announcements', 'navigation'
]

def calculate_category_features(df_user):
    """Calculate category-based features for a user."""
    total_views = len(df_user)

    features = {}

    # Per-category metrics
    for cat in CATEGORIES:
        cat_data = df_user[df_user['category'] == cat]

        features[f'{cat}_views'] = len(cat_data)
        features[f'{cat}_views_pct'] = len(cat_data) / total_views if total_views > 0 else 0
        features[f'{cat}_unique_resources'] = cat_data['resource_id'].nunique()
        features[f'{cat}_time_min'] = cat_data['interaction_seconds'].sum() / 60

    # Derived ratios
    content_views = features['files_views'] + features['pages_views'] + features['modules_views']
    assessment_views = features['quizzes_views'] + features['assignments_views']

    features['content_vs_assessment_ratio'] = content_views / assessment_views if assessment_views > 0 else 0
    features['grades_check_per_week'] = features['grades_views'] / (df_user['week_number_from_start'].max() or 1)
    features['discussions_vs_files_ratio'] = features['discussions_views'] / features['files_views'] if features['files_views'] > 0 else 0

    return pd.Series(features)

print("\nCalculating category features...")

# Group by student_id and course_id
features = df.groupby(['student_id', 'course_id']).apply(calculate_category_features, include_groups=False).reset_index()

print(f"Generated {len(features)} enrollment-level feature sets")
print(f"Feature columns: {len([c for c in features.columns if c not in ['student_id', 'course_id']])}")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
