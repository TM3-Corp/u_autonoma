#!/usr/bin/env python3
"""
Cleanup and standardize multi-model optimization results.

This script:
1. Renames legacy experiments to consistent naming format
2. Adds missing 'with_assessment' metadata
3. Removes legacy experiments with wrong percentile settings
"""

import json
from pathlib import Path

# Paths
INPUT_FILE = Path("data/analysis/multi_model_optimization_results.json")
OUTPUT_FILE = INPUT_FILE  # Overwrite in place
BACKUP_FILE = INPUT_FILE.with_suffix(".json.bak")

# Rename mappings: old_name -> new_name
RENAMES = {
    'week_2_p10': 'week_2_p10_with_assessment',
    'week_4_p20': 'week_4_p20_with_assessment',
    'week_6_p20': 'week_6_p20_with_assessment',
}

# Legacy experiments to remove (wrong percentile settings)
REMOVE = [
    'week_full_p15',              # Wrong percentile
    'week_8_p10_with_assessment', # Wrong percentile
    'week_8_p10_without_assessment',  # Wrong percentile
]

def main():
    print("=" * 60)
    print("Multi-Model Results Cleanup")
    print("=" * 60)

    # Load data
    with open(INPUT_FILE) as f:
        data = json.load(f)

    print(f"\nOriginal experiments ({len(data)}):")
    for exp in sorted(data.keys()):
        has_meta = 'with_assessment' in data[exp]
        print(f"  {exp} {'✓' if has_meta else '✗'} (metadata: {has_meta})")

    # Create backup
    with open(BACKUP_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\nBackup created: {BACKUP_FILE}")

    # Step 1: Remove legacy experiments
    print(f"\n--- Step 1: Remove legacy experiments ---")
    for exp_name in REMOVE:
        if exp_name in data:
            del data[exp_name]
            print(f"  Removed: {exp_name}")
        else:
            print(f"  Already gone: {exp_name}")

    # Step 2: Rename experiments
    print(f"\n--- Step 2: Rename experiments ---")
    for old_name, new_name in RENAMES.items():
        if old_name in data:
            exp_data = data.pop(old_name)
            # Add missing metadata
            if 'with_assessment' not in exp_data:
                exp_data['with_assessment'] = True
            data[new_name] = exp_data
            print(f"  {old_name} -> {new_name}")
        else:
            print(f"  Skip (not found): {old_name}")

    # Step 3: Ensure all experiments have 'with_assessment' metadata
    print(f"\n--- Step 3: Add missing metadata ---")
    for exp_name, exp_data in data.items():
        if 'with_assessment' not in exp_data:
            # Infer from name
            if 'without_assessment' in exp_name:
                exp_data['with_assessment'] = False
            elif 'with_assessment' in exp_name:
                exp_data['with_assessment'] = True
            else:
                print(f"  WARNING: Cannot infer for {exp_name}")
                continue
            print(f"  Added metadata to: {exp_name} = {exp_data['with_assessment']}")

    # Save cleaned data
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(data, f, indent=2)

    # Final summary
    print(f"\n" + "=" * 60)
    print(f"Cleaned experiments ({len(data)}):")
    for exp in sorted(data.keys()):
        with_assessment = data[exp].get('with_assessment', 'MISSING')
        print(f"  {exp}: with_assessment={with_assessment}")

    print(f"\nSaved to: {OUTPUT_FILE}")

    # Verify expected count
    if len(data) == 10:
        print("\n✅ SUCCESS: Exactly 10 experiments as expected")
    else:
        print(f"\n⚠️  WARNING: Expected 10 experiments, got {len(data)}")

if __name__ == "__main__":
    main()
