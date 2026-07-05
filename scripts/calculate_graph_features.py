#!/usr/bin/env python3
"""
Graph-Based Feature Extraction

Builds a bipartite student-resource graph and extracts features:
- Resource coverage (% of course resources accessed)
- Jaccard similarity to passing students' resources
- Access pattern clustering

Based on SOTA: "Graph-Based Features (Bipartite Student-Resource)"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from collections import defaultdict

DATA_DIR = Path('/home/paul/projects/uautonoma/data')
ENRICHED_DIR = DATA_DIR / 'enriched_features'
PAGE_VIEWS_DIR = DATA_DIR / 'page_views'


def calculate_jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def main():
    print('=' * 60)
    print('GRAPH-BASED FEATURE EXTRACTION')
    print('=' * 60)
    print()

    # Load categorized page views (raw events with resource_id)
    pv_path = PAGE_VIEWS_DIR / 'categorized_page_views.parquet'
    if not pv_path.exists():
        print(f'ERROR: {pv_path} not found')
        return

    print('Loading page views data...')
    df = pd.read_parquet(pv_path)

    # Filter out rows without valid course_id or resource_id
    df = df[df['course_id'].notna() & df['resource_id'].notna()]
    df['course_id'] = df['course_id'].astype(int)

    # Use source_user_id for matching with session features
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'
    df['user_id'] = df[user_col]

    # Filter to target courses (same as session features)
    TARGET_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
    df = df[df['course_id'].isin(TARGET_COURSES)].copy()

    print(f'  Loaded {len(df):,} page view events (target courses)')
    print(f'  Unique students: {df["user_id"].nunique()}')
    print(f'  Unique courses: {df["course_id"].nunique()}')
    print(f'  Unique resources: {df["resource_id"].nunique()}')

    # Load enrollments for pass/fail labels
    enrollments_path = PAGE_VIEWS_DIR / 'student_enrollments.csv'
    if enrollments_path.exists():
        enrollments = pd.read_csv(enrollments_path)
        enrollments['failed'] = enrollments['final_score'] < 57
        print(f'  Loaded {len(enrollments)} enrollment records')
        print(f'  Target courses: {enrollments["course_id"].nunique()}')

        # Filter page views to only target courses with enrollments
        target_courses = enrollments['course_id'].unique()
        df = df[df['course_id'].isin(target_courses)]
        print(f'  Filtered to {len(df):,} events in target courses')
    else:
        print('  WARNING: No enrollments found, skipping similarity features')
        enrollments = None

    print()

    # Step 1: Build resource access sets per student-course
    print('Building resource access graph...')
    student_resources = defaultdict(set)  # (user_id, course_id) -> set of resource_ids
    course_resources = defaultdict(set)   # course_id -> set of all resource_ids
    course_students = defaultdict(set)    # course_id -> set of user_ids

    for _, row in df.iterrows():
        key = (row['user_id'], row['course_id'])
        student_resources[key].add(row['resource_id'])
        course_resources[row['course_id']].add(row['resource_id'])
        course_students[row['course_id']].add(row['user_id'])

    print(f'  {len(student_resources)} student-course pairs')
    print(f'  {len(course_resources)} courses')

    # Step 2: Get passing students' resources per course (for similarity)
    passing_resources = {}  # course_id -> union of all passing students' resources
    if enrollments is not None:
        passing_enrollments = enrollments[enrollments['failed'] == False]
        for course_id in course_resources.keys():
            passing_users = passing_enrollments[
                passing_enrollments['course_id'] == course_id
            ]['user_id'].tolist()

            course_passing_resources = set()
            for user_id in passing_users:
                key = (user_id, course_id)
                if key in student_resources:
                    course_passing_resources.update(student_resources[key])

            passing_resources[course_id] = course_passing_resources

        print(f'  Computed passing student resources for {len(passing_resources)} courses')

    print()

    # Step 3: Calculate features per student-course
    print('Calculating graph features...')
    features_list = []

    for (user_id, course_id), resources in student_resources.items():
        features = {
            'user_id': user_id,
            'course_id': course_id,
        }

        # Resource coverage
        course_total = len(course_resources[course_id])
        features['resource_coverage'] = len(resources) / course_total if course_total > 0 else 0

        # Unique resources accessed
        features['unique_resources'] = len(resources)

        # Resources per student average in course
        n_students = len(course_students[course_id])
        avg_resources = sum(
            len(student_resources[(u, course_id)])
            for u in course_students[course_id]
        ) / n_students if n_students > 0 else 0
        features['resources_vs_avg'] = len(resources) / avg_resources if avg_resources > 0 else 0

        # Jaccard similarity to passing students
        if course_id in passing_resources:
            passing_res = passing_resources[course_id]
            features['jaccard_to_passing'] = calculate_jaccard_similarity(resources, passing_res)
        else:
            features['jaccard_to_passing'] = 0.0

        # Resource diversity by type (if type info available in resource_id)
        # For now, use unique count as proxy
        features['resource_diversity'] = len(resources) / (course_total + 1)

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f'  Created {len(features_df)} rows, {len(features_df.columns)} columns')

    # Step 4: Add clustering features (cluster students by access patterns)
    print('\nClustering students by access patterns...')

    for course_id in course_resources.keys():
        course_mask = features_df['course_id'] == course_id
        if course_mask.sum() < 5:  # Need minimum students for clustering
            features_df.loc[course_mask, 'access_cluster'] = 0
            continue

        # Create feature matrix for clustering
        cluster_features = features_df.loc[course_mask, [
            'resource_coverage', 'unique_resources', 'resources_vs_avg'
        ]].fillna(0)

        if len(cluster_features) >= 3:
            n_clusters = min(3, len(cluster_features))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(cluster_features)
            features_df.loc[course_mask, 'access_cluster'] = clusters
        else:
            features_df.loc[course_mask, 'access_cluster'] = 0

    print(f'  Assigned cluster labels')

    # Step 5: Save
    output_path = ENRICHED_DIR / 'graph_features.parquet'
    features_df.to_parquet(output_path, index=False)
    print(f'\nSaved to: {output_path}')

    # Summary stats
    print('\nFeature Summary:')
    for col in ['resource_coverage', 'jaccard_to_passing', 'resources_vs_avg']:
        if col in features_df.columns:
            print(f'  {col}: mean={features_df[col].mean():.3f}, std={features_df[col].std():.3f}')


if __name__ == '__main__':
    main()
