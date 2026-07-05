#!/usr/bin/env python3
"""
Calculate features by resource category.

Features per category (files, discussions, quizzes, assignments, pages, modules, grades):
- {cat}_views: Total views of this category
- {cat}_views_pct: % of total views
- {cat}_unique_resources: Unique resources visited
- {cat}_time_spent: Estimated time (sum of interaction_seconds)

Derived features:
- content_vs_assessment_ratio: (files + pages + discussions) / (quizzes + assignments)
- grades_check_frequency: visits to /grades per week
- discussion_participation_rate: discussions with participated=True / total discussions
"""

import pandas as pd
import numpy as np
from pathlib import Path

INPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/categorized_page_views.parquet')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/category_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

CATEGORIES = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules', 'grades', 'announcements', 'home']
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]


def calculate_category_features(df_user):
    """Calculate category-based features for a user's page views."""
    if len(df_user) == 0:
        return None

    total_views = len(df_user)

    # Parse timestamps
    timestamps = pd.to_datetime(df_user['created_at'])
    span_weeks = max((timestamps.max() - timestamps.min()).days / 7, 1)

    # Get interaction_seconds column if available
    time_col = 'interaction_seconds' if 'interaction_seconds' in df_user.columns else None

    features = {}

    for cat in CATEGORIES:
        cat_views = df_user[df_user['resource_type'] == cat]

        features[f'{cat}_views'] = len(cat_views)
        features[f'{cat}_views_pct'] = len(cat_views) / total_views * 100 if total_views > 0 else 0

        # Unique resources (where resource_id is available)
        unique_resources = cat_views['resource_id'].dropna().nunique()
        features[f'{cat}_unique_resources'] = unique_resources

        # Time spent (if available)
        if time_col and time_col in cat_views.columns:
            time_spent = pd.to_numeric(cat_views[time_col], errors='coerce').sum()
            features[f'{cat}_time_min'] = time_spent / 60  # Convert to minutes
        else:
            features[f'{cat}_time_min'] = 0

    # Derived features
    content_views = features['files_views'] + features['pages_views'] + features['discussions_views']
    assessment_views = features['quizzes_views'] + features['assignments_views']

    features['content_vs_assessment_ratio'] = content_views / assessment_views if assessment_views > 0 else content_views

    # Grades check frequency (per week)
    features['grades_check_per_week'] = features['grades_views'] / span_weeks

    # Discussion participation rate
    disc_views = df_user[df_user['resource_type'] == 'discussions']
    if len(disc_views) > 0 and 'participated' in disc_views.columns:
        participated = disc_views['participated'].fillna(False).astype(bool).sum()
        features['discussion_participation_rate'] = participated / len(disc_views) * 100
    else:
        features['discussion_participation_rate'] = 0

    # Total views
    features['total_views'] = total_views

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
    print('Calculating category features...')
    all_features = []

    for (user_id, course_id), group in df.groupby([user_col, 'course_id']):
        features = calculate_category_features(group)
        if features:
            features['user_id'] = user_id
            features['course_id'] = course_id
            all_features.append(features)

    features_df = pd.DataFrame(all_features)
    print(f'Generated features for {len(features_df)} user-course pairs')
    print()

    # Summary stats
    print('Feature summary:')
    cols = [c for c in features_df.columns if c.endswith('_views') and not c.startswith('total')]
    print(features_df[cols].describe())
    print()

    # Save
    print(f'Saving to {OUTPUT_FILE}...')
    features_df.to_parquet(OUTPUT_FILE, index=False)
    print('Done!')


if __name__ == '__main__':
    main()
