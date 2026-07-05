#!/usr/bin/env python3
"""
Calculate PCA features from PCT matrices per resource type.

For each course and LEARNING MATERIAL type (files, discussions, pages, modules):
1. Create PCT matrix (students x resources)
2. Apply StandardScaler
3. Fit PCA and extract fixed number of components
4. Store as comparable features across courses

EXCLUDES: quizzes, assignments (these are assessments, not learning materials)

This enables true "early warning" prediction BEFORE any grades exist.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

INPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/proactivity_detailed.parquet')
ENROLLMENTS_FILE = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/enriched_features/pca_features.parquet')
TSNE_DIR = Path('/home/paul/projects/uautonoma/data/report/visualizations/tsne')
TSNE_DIR.mkdir(parents=True, exist_ok=True)

# Learning materials only (EXCLUDE assessments: quizzes, assignments)
RESOURCE_TYPES = ['files', 'discussions', 'pages', 'modules']

# Fixed components per type
COMPONENTS = {
    'files': 3,
    'discussions': 3,
    'pages': 3,
    'modules': 2
}

# Short prefixes for feature names
PREFIXES = {
    'files': 'files',
    'discussions': 'disc',
    'pages': 'pages',
    'modules': 'mods'
}


def create_zero_features(user_ids, n_components, prefix):
    """Create zero features when no resources exist."""
    features = {'user_id': list(user_ids)}
    for i in range(n_components):
        features[f'{prefix}_pc{i+1}'] = [0.0] * len(user_ids)
    features[f'{prefix}_n_resources'] = [0] * len(user_ids)
    features[f'{prefix}_var_explained'] = [0.0] * len(user_ids)
    return pd.DataFrame(features)


def extract_pca_features(df_type, enrolled_users, n_components, prefix):
    """
    Extract PCA features from a (students x resources) PCT matrix.

    Args:
        df_type: DataFrame with user_id, resource_id, pct
        enrolled_users: Set of enrolled user_ids
        n_components: Number of PCA components to extract
        prefix: Feature name prefix

    Returns: DataFrame with user_id and PC features
    """
    if len(df_type) == 0 or df_type['resource_id'].isna().all():
        return create_zero_features(enrolled_users, n_components, prefix)

    # Create pivot matrix (students x resources)
    pivot = df_type.pivot_table(
        index='user_id',
        columns='resource_id',
        values='pct',
        aggfunc='first'
    )

    # Ensure all enrolled users are in the index
    missing_users = set(enrolled_users) - set(pivot.index)
    if missing_users:
        missing_df = pd.DataFrame(index=list(missing_users), columns=pivot.columns)
        pivot = pd.concat([pivot, missing_df])

    # Fill NaN with 0 (never accessed)
    pivot = pivot.fillna(0)

    # Reindex to match enrolled users order
    pivot = pivot.reindex(enrolled_users)

    n_students, n_resources = pivot.shape

    if n_resources == 0:
        return create_zero_features(enrolled_users, n_components, prefix)

    # Determine actual components to extract
    actual_components = min(n_components, n_students - 1, n_resources)
    if actual_components < 1:
        return create_zero_features(enrolled_users, n_components, prefix)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot.values)

    # Apply PCA
    pca = PCA(n_components=actual_components)
    X_pca = pca.fit_transform(X_scaled)

    # Build feature DataFrame
    features = {'user_id': list(pivot.index)}

    for i in range(n_components):
        col_name = f'{prefix}_pc{i+1}'
        if i < actual_components:
            features[col_name] = X_pca[:, i]
        else:
            features[col_name] = [0.0] * len(pivot.index)  # Pad if fewer components

    # Metadata
    features[f'{prefix}_n_resources'] = [n_resources] * len(pivot.index)
    features[f'{prefix}_var_explained'] = [sum(pca.explained_variance_ratio_)] * len(pivot.index)

    return pd.DataFrame(features), pivot, pca


def create_tsne_visualization(pivot, enrollments, resource_type, course_id, output_dir):
    """Create t-SNE visualization colored by outcome."""
    if pivot is None or len(pivot) < 10:
        return

    # Get outcomes for students
    outcomes = enrollments.set_index('user_id')['failed'].reindex(pivot.index)

    # Filter to students with known outcomes
    valid_mask = outcomes.notna()
    if valid_mask.sum() < 10:
        return

    X = pivot.loc[valid_mask].values
    y = outcomes.loc[valid_mask].values

    # Apply t-SNE
    try:
        tsne = TSNE(n_components=2, perplexity=min(30, len(X) - 1), random_state=42)
        X_tsne = tsne.fit_transform(X)
    except Exception as e:
        print(f'    t-SNE failed for {resource_type}: {e}')
        return

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot by outcome
    colors = ['green' if not failed else 'red' for failed in y]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.6, s=50)

    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_title(f't-SNE: {resource_type.capitalize()} Engagement Patterns\nCourse {course_id}')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='Aprobados'),
        Patch(facecolor='red', alpha=0.6, label='Reprobados')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # Save
    output_path = output_dir / f'tsne_{resource_type}_course_{course_id}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print('=' * 70)
    print('PCA Features for Learning Materials (Excluding Assessments)')
    print('=' * 70)
    print()

    # Load data
    print('Loading data...')
    df = pd.read_parquet(INPUT_FILE)
    print(f'  Proactivity detailed: {len(df):,} records')

    enrollments = pd.read_csv(ENROLLMENTS_FILE)
    enrollments['failed'] = enrollments['final_score'] < 57
    print(f'  Enrollments: {len(enrollments)} records')
    print()

    # Filter to learning materials only
    df = df[df['resource_type'].isin(RESOURCE_TYPES)]
    print(f'Filtered to learning materials: {len(df):,} records')
    print(f'Resource types: {df["resource_type"].unique().tolist()}')
    print()

    # Get unique courses
    courses = df['course_id'].unique()
    print(f'Processing {len(courses)} courses...')
    print()

    all_features = []
    tsne_count = 0

    for course_id in courses:
        print(f'Course {course_id}:')
        df_course = df[df['course_id'] == course_id]

        # Get enrolled students
        course_enrollments = enrollments[enrollments['course_id'] == course_id]
        enrolled_users = course_enrollments['user_id'].unique()

        if len(enrolled_users) == 0:
            print(f'  No enrollments, skipping')
            continue

        course_features = pd.DataFrame({'user_id': enrolled_users})
        course_features['course_id'] = course_id

        for rtype in RESOURCE_TYPES:
            df_type = df_course[df_course['resource_type'] == rtype]
            prefix = PREFIXES[rtype]
            n_components = COMPONENTS[rtype]

            n_resources = df_type['resource_id'].nunique() if len(df_type) > 0 else 0
            print(f'  {rtype}: {n_resources} resources', end='')

            result = extract_pca_features(df_type, enrolled_users, n_components, prefix)

            if isinstance(result, tuple):
                type_features, pivot, pca = result
                var_explained = sum(pca.explained_variance_ratio_) * 100
                print(f' -> {len(pca.components_)} PCs ({var_explained:.1f}% variance)')

                # Create t-SNE visualization
                create_tsne_visualization(pivot, course_enrollments, rtype, course_id, TSNE_DIR)
                tsne_count += 1
            else:
                type_features = result
                print(f' -> 0 PCs (no data)')

            # Merge with course features
            course_features = course_features.merge(
                type_features,
                on='user_id',
                how='left'
            )

        all_features.append(course_features)
        print()

    # Combine all courses
    result_df = pd.concat(all_features, ignore_index=True)

    # Fill any remaining NaN with 0
    result_df = result_df.fillna(0)

    print('=' * 70)
    print('Summary')
    print('=' * 70)
    print(f'Total records: {len(result_df)}')
    print(f'Total features: {len(result_df.columns)}')
    print(f't-SNE visualizations created: {tsne_count}')
    print()

    # Feature columns
    feature_cols = [c for c in result_df.columns if c not in ['user_id', 'course_id']]
    print('PCA Features:')
    for col in sorted(feature_cols):
        if '_pc' in col:
            mean_val = result_df[col].mean()
            std_val = result_df[col].std()
            print(f'  {col}: mean={mean_val:.3f}, std={std_val:.3f}')

    print()
    print('Variance Explained:')
    for rtype in RESOURCE_TYPES:
        prefix = PREFIXES[rtype]
        col = f'{prefix}_var_explained'
        if col in result_df.columns:
            mean_var = result_df[col].mean() * 100
            print(f'  {rtype}: {mean_var:.1f}% average')

    # Save
    print()
    print(f'Saving to {OUTPUT_FILE}...')
    result_df.to_parquet(OUTPUT_FILE, index=False)

    print(f't-SNE visualizations saved to {TSNE_DIR}/')
    print()
    print('Done!')


if __name__ == '__main__':
    main()
