#!/usr/bin/env python3
"""
Phase 1: Prepare PUC data for early warning pipeline.

Merges page views with grades, calculates course start dates,
and creates unified dataset matching U Autonoma format.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

# Paths
PUC_DATA_DIR = Path('/home/paul/projects/wave_analysis/puc_analysis/data')
OUTPUT_DIR = Path('data/puc')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading PUC data...")

# Load page views (already categorized)
pv = pd.read_parquet(PUC_DATA_DIR / 'categorized_page_views.parquet')
print(f"Loaded {len(pv):,} page views")

# Load grades
grades = pd.read_parquet(PUC_DATA_DIR / 'grades_with_failure.parquet')
print(f"Loaded {len(grades):,} grade records")

# Create course code column (e.g., "ICS1513-1")
grades['course_code'] = grades['Sigla']

# Rename columns to match expected format
grades = grades.rename(columns={
    'user_lms_id': 'student_id',
    'course_lms_id': 'course_id'
})

print("\nMerging data...")

# Merge page views with grades
merged = pv.merge(
    grades[['student_id', 'course_id', 'course_code', 'grade', 'failed']],
    on=['student_id', 'course_id'],
    how='inner'
)

print(f"Merged dataset: {len(merged):,} page views")
print(f"Unique students: {merged['student_id'].nunique()}")
print(f"Unique courses: {merged['course_id'].nunique()}")
print(f"Unique enrollments: {merged.groupby(['student_id', 'course_id']).ngroups}")

# Calculate course start dates (5th percentile of first activity per student)
print("\nCalculating course start dates...")

course_starts = {}
for course_id in merged['course_id'].unique():
    course_data = merged[merged['course_id'] == course_id]

    # Get first activity per student
    first_activities = course_data.groupby('student_id')['created_at'].min()

    # 5th percentile
    course_start = first_activities.quantile(0.05)
    course_starts[course_id] = course_start

    print(f"  Course {course_id}: {course_start.date()}")

# Add course start and recalculate week_number relative to course start
merged['course_start'] = merged['course_id'].map(course_starts)
merged['days_since_start'] = (merged['created_at'] - merged['course_start']).dt.total_seconds() / 86400
merged['week_number_from_start'] = (merged['days_since_start'] / 7).astype(int) + 1

# Filter to positive weeks only (after course start)
merged = merged[merged['week_number_from_start'] > 0].copy()

print(f"\nFiltered to {len(merged):,} page views after course start")

# Add resource_id (use hash of controller+action as proxy)
merged['resource_id'] = (merged['controller'].astype(str) + '_' + merged['action'].astype(str)).apply(hash).abs()

# Reorder columns to match U Autonoma format
final_columns = [
    'student_id',
    'course_id',
    'course_code',
    'created_at',
    'category',
    'controller',
    'action',
    'resource_id',
    'interaction_seconds',
    'participated',
    'date',
    'week_number_from_start',
    'hour',
    'day_of_week',
    'course_start',
    'days_since_start',
    'grade',
    'failed'
]

merged = merged[final_columns]

# Save merged data
output_path = OUTPUT_DIR / 'puc_merged_data.parquet'
merged.to_parquet(output_path, index=False)
print(f"\nSaved merged data to: {output_path}")

# Generate summary statistics
summary = {
    'total_page_views': len(merged),
    'total_students': int(merged['student_id'].nunique()),
    'total_courses': int(merged['course_id'].nunique()),
    'total_enrollments': int(merged.groupby(['student_id', 'course_id']).ngroups),
    'date_range': {
        'start': str(merged['created_at'].min()),
        'end': str(merged['created_at'].max())
    },
    'grade_stats': {
        'mean': float(merged.groupby(['student_id', 'course_id'])['grade'].first().mean()),
        'std': float(merged.groupby(['student_id', 'course_id'])['grade'].first().std()),
        'min': float(merged.groupby(['student_id', 'course_id'])['grade'].first().min()),
        'max': float(merged.groupby(['student_id', 'course_id'])['grade'].first().max())
    },
    'failure_rate': float(merged.groupby(['student_id', 'course_id'])['failed'].first().mean()),
    'categories': merged['category'].unique().tolist(),
    'course_starts': {int(k): str(v) for k, v in course_starts.items()}
}

summary_path = OUTPUT_DIR / 'puc_data_summary.json'
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved to: {summary_path}")
print("\n" + "="*60)
print("DATA SUMMARY")
print("="*60)
print(f"Total page views:      {summary['total_page_views']:,}")
print(f"Total students:        {summary['total_students']:,}")
print(f"Total courses:         {summary['total_courses']:,}")
print(f"Total enrollments:     {summary['total_enrollments']:,}")
print(f"Date range:            {summary['date_range']['start'][:10]} to {summary['date_range']['end'][:10]}")
print(f"\nGrade statistics:")
print(f"  Mean:                {summary['grade_stats']['mean']:.2f}")
print(f"  Std:                 {summary['grade_stats']['std']:.2f}")
print(f"  Range:               {summary['grade_stats']['min']:.1f} - {summary['grade_stats']['max']:.1f}")
print(f"\nFailure rate (<4.0):   {summary['failure_rate']*100:.1f}%")
print(f"Categories:            {len(summary['categories'])}")
print("="*60)
print("\n✓ Phase 1 complete: Data preparation successful")
