#!/usr/bin/env python3
"""
Merge and normalize all PUC feature sets.

Per-course z-score normalization for count/time features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

FEATURES_DIR = Path('data/puc/enriched_features')
OUTPUT_FILE = Path('data/puc/enriched_features/normalized_features.parquet')

print("Loading all feature sets...")

# Load all feature files
feature_files = [
    'session_features.parquet',
    'category_features.parquet',
    'weekly_features.parquet',
    'time_features.parquet',
    'proactivity_features.parquet',
    'pca_features.parquet',
    'course_relative_features.parquet',
    'ngram_features.parquet',
    'graph_features.parquet'
]

dfs = []
for fname in feature_files:
    fpath = FEATURES_DIR / fname
    if fpath.exists():
        df = pd.read_parquet(fpath)
        print(f"  {fname}: {df.shape}")
        dfs.append(df)
    else:
        print(f"  ⚠ {fname}: NOT FOUND")

# Merge all features on student_id and course_id
print("\nMerging features...")
merged = dfs[0]
for df in dfs[1:]:
    merged = merged.merge(df, on=['student_id', 'course_id'], how='outer')

print(f"Merged shape: {merged.shape}")
print(f"Total features: {len(merged.columns) - 2}")

# Add grades
print("\nAdding grades...")
grades_file = Path('data/puc/puc_merged_data.parquet')
grades_df = pd.read_parquet(grades_file)[['student_id', 'course_id', 'grade', 'failed']].drop_duplicates()

merged = merged.merge(grades_df, on=['student_id', 'course_id'], how='left')

print(f"Final shape with grades: {merged.shape}")

# Normalize features per course (z-score)
print("\nNormalizing features per course...")

# Identify numeric columns to normalize (exclude IDs and target variables)
exclude_cols = ['student_id', 'course_id', 'grade', 'failed']
numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
normalize_cols = [c for c in numeric_cols if c not in exclude_cols]

print(f"Normalizing {len(normalize_cols)} numeric features")

# Per-course z-score normalization
for col in normalize_cols:
    # Group by course and normalize
    merged[col] = merged.groupby('course_id')[col].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-10)
    )

# Fill any remaining NaNs with 0
merged = merged.fillna(0)

# Save normalized features
merged.to_parquet(OUTPUT_FILE, index=False)
print(f"\n✓ Saved normalized features to: {OUTPUT_FILE}")
print(f"  Shape: {merged.shape}")
print(f"  Enrollments: {len(merged)}")
print(f"  Features: {len(normalize_cols)}")

# Generate summary
summary = {
    'total_enrollments': len(merged),
    'total_features': len(normalize_cols),
    'features_by_category': {
        'session': len([c for c in normalize_cols if 'session' in c or 'total_' in c]),
        'category': len([c for c in normalize_cols if any(cat in c for cat in ['files', 'disc', 'quiz', 'assi', 'page', 'modu', 'grad', 'anno'])]),
        'weekly': len([c for c in normalize_cols if 'week' in c]),
        'time': len([c for c in normalize_cols if any(t in c for t in ['morning', 'afternoon', 'evening', 'night', 'hour', 'day'])]),
        'proactivity': len([c for c in normalize_cols if '_pct' in c or '_hist_' in c or 'access_rate' in c]),
        'pca': len([c for c in normalize_cols if '_pc' in c]),
        'course_relative': len([c for c in normalize_cols if 'relative' in c or 'timing' in c]),
        'ngram': len([c for c in normalize_cols if 'transition' in c]),
        'graph': len([c for c in normalize_cols if 'coverage' in c or 'diversity' in c or 'depth' in c or 'breadth' in c])
    },
    'grade_stats': {
        'mean': float(merged['grade'].mean()),
        'std': float(merged['grade'].std()),
        'failure_rate': float(merged['failed'].mean())
    }
}

summary_file = Path('data/puc/enriched_features/feature_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*60)
print("FEATURE ENGINEERING SUMMARY")
print("="*60)
print(f"Total enrollments:     {summary['total_enrollments']}")
print(f"Total features:        {summary['total_features']}")
print(f"\nFeatures by category:")
for cat, count in summary['features_by_category'].items():
    print(f"  {cat:20s}: {count:3d}")
print(f"\nGrade statistics:")
print(f"  Mean:                {summary['grade_stats']['mean']:.2f}")
print(f"  Std:                 {summary['grade_stats']['std']:.2f}")
print(f"  Failure rate:        {summary['grade_stats']['failure_rate']*100:.1f}%")
print("="*60)
print("\n✓ Phase 2 complete: Feature engineering successful")
