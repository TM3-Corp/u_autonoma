#!/usr/bin/env python3
"""
Merge all SOTA features (Phase 1 + Phase 2) with existing PUC features.

This combines:
- Phase 1: Inactivity episodes, engagement decay, early momentum
- Phase 2: N-gram features
- Existing: Normalized features from multi-class model

Applies course-relative (z-score) normalization to all new features.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Input files
EXISTING_FEATURES = Path('data/puc/enriched_features/normalized_features_multiclass.parquet')
INACTIVITY_FEATURES = Path('data/puc/enriched_features/inactivity_episode_features.parquet')
DECAY_FEATURES = Path('data/puc/enriched_features/engagement_decay_features.parquet')
MOMENTUM_FEATURES = Path('data/puc/enriched_features/early_momentum_features.parquet')
NGRAM_FEATURES = Path('data/puc/enriched_features/ngram_features.parquet')

# Output
OUTPUT_FILE = Path('data/puc/enriched_features/all_features_sota.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

print("="*60)
print("MERGING SOTA FEATURES")
print("="*60)

# Load existing features
print("\n1. Loading existing normalized features...")
df_existing = pd.read_parquet(EXISTING_FEATURES)
print(f"   Existing features: {df_existing.shape}")

# Load Phase 1 features
print("\n2. Loading Phase 1 features (inactivity, decay, momentum)...")
df_inactivity = pd.read_parquet(INACTIVITY_FEATURES)
df_decay = pd.read_parquet(DECAY_FEATURES)
df_momentum = pd.read_parquet(MOMENTUM_FEATURES)

print(f"   Inactivity: {df_inactivity.shape}")
print(f"   Decay: {df_decay.shape}")
print(f"   Momentum: {df_momentum.shape}")

# Load Phase 2 features
print("\n3. Loading Phase 2 features (N-grams)...")
df_ngram = pd.read_parquet(NGRAM_FEATURES)
print(f"   N-grams: {df_ngram.shape}")

# Merge all features
print("\n4. Merging all features...")
df_merged = df_existing.copy()

# Merge on student_id and course_id
for df_feat, name in [(df_inactivity, 'inactivity'),
                       (df_decay, 'decay'),
                       (df_momentum, 'momentum'),
                       (df_ngram, 'ngram')]:

    df_merged = df_merged.merge(df_feat, on=['student_id', 'course_id'], how='left')
    print(f"   After merging {name}: {df_merged.shape}")

print(f"\n   Total features (including IDs): {df_merged.shape[1]}")
print(f"   Total students: {df_merged['student_id'].nunique()}")

# Apply course-relative z-score normalization to NEW features only
print("\n5. Applying course-relative normalization to new features...")

# Get list of new feature columns (exclude existing normalized features)
exclude_cols = ['student_id', 'course_id', 'grade_category']
existing_feature_cols = [c for c in df_existing.columns if c not in exclude_cols]

# New features = all features except existing ones
new_feature_cols = [c for c in df_merged.columns
                    if c not in existing_feature_cols and c not in exclude_cols]

print(f"   New features to normalize: {len(new_feature_cols)}")

# Normalize each new feature per course (z-score)
for col in new_feature_cols:
    if df_merged[col].dtype in [np.float64, np.int64]:
        # Course-relative z-score
        df_merged[f'{col}_znorm'] = df_merged.groupby('course_id')[col].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
        )

# Get list of all normalized features
normalized_cols = [c for c in df_merged.columns if c.endswith('_znorm')]
print(f"   Total normalized features: {len(normalized_cols)}")

# Save merged features
print(f"\n6. Saving merged features...")
df_merged.to_parquet(OUTPUT_FILE, index=False)
print(f"   ✓ Saved to: {OUTPUT_FILE}")
print(f"   Shape: {df_merged.shape}")

# Summary statistics
print("\n7. Feature Summary:")
print(f"   Total columns: {len(df_merged.columns)}")
print(f"   ID columns: {len(exclude_cols)}")
print(f"   Existing features: {len(existing_feature_cols)}")
print(f"   New raw features: {len(new_feature_cols)}")
print(f"   New normalized features: {len([c for c in normalized_cols if c.replace('_znorm', '') in new_feature_cols])}")

# Show sample of new features
print("\n8. Sample of new SOTA features (first 10):")
new_feat_sample = new_feature_cols[:10]
for feat in new_feat_sample:
    if feat in df_merged.columns:
        print(f"   {feat}: mean={df_merged[feat].mean():.3f}, std={df_merged[feat].std():.3f}")

print("\n" + "="*60)
print("SOTA FEATURE MERGING COMPLETE")
print("="*60)
print(f"\nNext steps:")
print(f"  1. Run feature selection: scripts/puc_sota_feature_selection.py")
print(f"  2. Train SOTA model: scripts/puc_train_multiclass_sota.py")
print("="*60)
