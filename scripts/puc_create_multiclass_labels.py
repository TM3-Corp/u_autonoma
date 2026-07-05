#!/usr/bin/env python3
"""
Phase 3: Create multi-class labels (4 classes) and update normalized features dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "puc"
FEATURES_FILE = DATA_DIR / "enriched_features" / "normalized_features.parquet"
OUTPUT_FILE = DATA_DIR / "enriched_features" / "normalized_features_multiclass.parquet"
DIST_FILE = DATA_DIR / "multiclass_distribution.json"

# Grade class definitions
GRADE_CLASSES = {
    0: {'name': 'EXCELLENT', 'range': (6.0, 7.0)},
    1: {'name': 'GOOD', 'range': (5.0, 5.9)},
    2: {'name': 'MARGINAL', 'range': (4.0, 4.9)},
    3: {'name': 'FAIL', 'range': (0.0, 3.9)}
}


def assign_grade_class(grade):
    """Assign grade to one of 4 classes."""
    if grade >= 6.0:
        return 0  # EXCELLENT
    elif grade >= 5.0:
        return 1  # GOOD
    elif grade >= 4.0:
        return 2  # MARGINAL
    else:
        return 3  # FAIL


def main():
    print("="*60)
    print("Phase 3: Create Multi-Class Labels")
    print("="*60)

    # Load features
    print("\nLoading features...")
    df = pd.read_parquet(FEATURES_FILE)
    print(f"Loaded {len(df)} student enrollments")

    # Assign grade classes
    print("\nAssigning grade classes...")
    df['grade_class'] = df['grade'].apply(assign_grade_class)
    df['class_label'] = df['grade_class'].map(lambda x: GRADE_CLASSES[x]['name'])

    # Verify class distribution
    print("\nClass Distribution:")
    distribution = {}
    for class_id in range(4):
        count = (df['grade_class'] == class_id).sum()
        pct = count / len(df) * 100
        class_info = GRADE_CLASSES[class_id]

        print(f"  Class {class_id} ({class_info['name']:9s}): {count:3d} students ({pct:5.1f}%) - Grade {class_info['range'][0]:.1f}-{class_info['range'][1]:.1f}")

        distribution[class_info['name']] = {
            'class_id': class_id,
            'count': int(count),
            'percentage': float(pct),
            'grade_range': class_info['range']
        }

    # Verify expected distribution
    expected = {'EXCELLENT': 256, 'GOOD': 329, 'MARGINAL': 201, 'FAIL': 82}
    print("\nVerification:")
    all_match = True
    for class_name, expected_count in expected.items():
        actual_count = distribution[class_name]['count']
        match = "✓" if actual_count == expected_count else "✗"
        print(f"  {match} {class_name:9s}: Expected {expected_count:3d}, Got {actual_count:3d}")
        if actual_count != expected_count:
            all_match = False

    if all_match:
        print("\n✓ All class counts match expected distribution!")
    else:
        print("\n✗ Warning: Class distribution differs from expected!")

    # Check for missing values
    missing = df[['grade', 'grade_class', 'class_label']].isnull().sum()
    print("\nMissing values:")
    print(missing)

    if missing.any():
        print("\n✗ Warning: Found missing values!")
    else:
        print("\n✓ No missing values in grade columns")

    # Save updated dataset
    print(f"\nSaving multi-class dataset...")
    df.to_parquet(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")

    # Save distribution summary
    with open(DIST_FILE, 'w') as f:
        json.dump({
            'total_students': len(df),
            'class_distribution': distribution,
            'imbalance_ratio': float(distribution['GOOD']['count'] / distribution['FAIL']['count'])
        }, f, indent=2)
    print(f"Saved: {DIST_FILE}")

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics by Class:")
    print("="*60)
    for class_id in range(4):
        class_name = GRADE_CLASSES[class_id]['name']
        class_df = df[df['grade_class'] == class_id]
        print(f"\n{class_name} (n={len(class_df)}):")
        print(f"  Grade range: {class_df['grade'].min():.2f} - {class_df['grade'].max():.2f}")
        print(f"  Grade mean:  {class_df['grade'].mean():.2f} ± {class_df['grade'].std():.2f}")
        print(f"  Grade median: {class_df['grade'].median():.2f}")

    print("\n✓ Phase 3 complete!")


if __name__ == "__main__":
    main()
