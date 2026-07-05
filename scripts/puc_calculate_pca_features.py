#!/usr/bin/env python3
"""
Calculate PCA features from resource access matrices for PUC data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/pca_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

RESOURCE_TYPES = ['files', 'discussions', 'pages', 'modules']

all_features = []

for course_id in df['course_id'].unique():
    print(f"  Processing course {course_id}...")

    df_course = df[df['course_id'] == course_id]
    enrolled_users = sorted(df_course['student_id'].unique())

    course_features = {}

    for rtype in RESOURCE_TYPES:
        type_data = df_course[df_course['category'] == rtype]

        if len(type_data) == 0:
            # No resources of this type
            for user_id in enrolled_users:
                if user_id not in course_features:
                    course_features[user_id] = {'student_id': user_id, 'course_id': course_id}
                course_features[user_id][f'{rtype[:4]}_pc1'] = 0
                course_features[user_id][f'{rtype[:4]}_pc2'] = 0
                course_features[user_id][f'{rtype[:4]}_pc3'] = 0
                course_features[user_id][f'{rtype[:4]}_var_explained'] = 0
            continue

        # Create student × resource matrix (count of views)
        matrix = type_data.pivot_table(
            index='student_id',
            columns='resource_id',
            values='created_at',
            aggfunc='count',
            fill_value=0
        )

        # Ensure all enrolled students are in the matrix
        for user_id in enrolled_users:
            if user_id not in matrix.index:
                matrix.loc[user_id] = 0

        matrix = matrix.sort_index()

        # Apply PCA (3 components)
        n_components = min(3, min(matrix.shape) - 1)
        if n_components < 1:
            # Not enough data for PCA
            for user_id in enrolled_users:
                if user_id not in course_features:
                    course_features[user_id] = {'student_id': user_id, 'course_id': course_id}
                course_features[user_id][f'{rtype[:4]}_pc1'] = 0
                course_features[user_id][f'{rtype[:4]}_pc2'] = 0
                course_features[user_id][f'{rtype[:4]}_pc3'] = 0
                course_features[user_id][f'{rtype[:4]}_var_explained'] = 0
            continue

        # Standardize
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(matrix.values)

        # PCA
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(matrix_scaled)

        # Store features
        for idx, user_id in enumerate(matrix.index):
            if user_id not in course_features:
                course_features[user_id] = {'student_id': user_id, 'course_id': course_id}

            course_features[user_id][f'{rtype[:4]}_pc1'] = components[idx, 0] if n_components >= 1 else 0
            course_features[user_id][f'{rtype[:4]}_pc2'] = components[idx, 1] if n_components >= 2 else 0
            course_features[user_id][f'{rtype[:4]}_pc3'] = components[idx, 2] if n_components >= 3 else 0
            course_features[user_id][f'{rtype[:4]}_var_explained'] = pca.explained_variance_ratio_.sum()

    all_features.extend(course_features.values())

features = pd.DataFrame(all_features)

print(f"\nGenerated {len(features)} enrollment-level feature sets")

# Save
features.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved to: {OUTPUT_FILE}")
print(f"  Shape: {features.shape}")
