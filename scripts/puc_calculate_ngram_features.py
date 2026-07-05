#!/usr/bin/env python3
"""
Session N-gram Feature Extraction for PUC data.

Extracts sequential pattern features from student clickstream data.
Captures the ORDER of resource access, which reveals learning strategies.

Features:
- Top N bigram counts (e.g., modules->files, home->assignments)
- Transition entropy (diversity of navigation patterns)
- Self-loop ratio (e.g., files->files indicates deep reading)
- Transition diversity

Adapted from U. Autonoma SOTA feature engineering.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import entropy

INPUT_FILE = Path('data/puc/puc_merged_data.parquet')
OUTPUT_FILE = Path('data/puc/enriched_features/ngram_features.parquet')
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SESSION_GAP_MINUTES = 30

# Map PUC categories to resource types
CATEGORY_MAPPING = {
    'announcements': 'announcements',
    'assignments': 'assignments',
    'discussion_topics': 'discussions',
    'discussion_entries': 'discussions',
    'files': 'files',
    'modules': 'modules',
    'quizzes': 'quizzes',
    'wiki_pages': 'pages',
    'grades': 'grades',
    'courses': 'home',
}

print("Loading PUC merged data...")
df = pd.read_parquet(INPUT_FILE)
print(f"Loaded {len(df):,} page views")

# Map category to resource_type
df['resource_type'] = df['category'].map(CATEGORY_MAPPING).fillna('other')

# Convert timestamps
df['created_at'] = pd.to_datetime(df['created_at'])

# Create session IDs (30-min gap threshold)
print("Creating session IDs (30-min gap threshold)...")
df = df.sort_values(['student_id', 'course_id', 'created_at'])
df['time_gap'] = df.groupby(['student_id', 'course_id'])['created_at'].diff()
df['new_session'] = (df['time_gap'] > pd.Timedelta(minutes=SESSION_GAP_MINUTES)) | df['time_gap'].isna()
df['session_id'] = df.groupby(['student_id', 'course_id'])['new_session'].cumsum()

print(f"  Total sessions: {df['session_id'].max():,}")


def extract_session_bigrams(session_events):
    """Extract bigrams (consecutive pairs) from a session."""
    if len(session_events) < 2:
        return []

    # Sort by timestamp
    sorted_events = session_events.sort_values('created_at')
    types = sorted_events['resource_type'].tolist()

    # Create bigrams
    bigrams = []
    for i in range(len(types) - 1):
        bigram = f"{types[i]}->{types[i+1]}"
        bigrams.append(bigram)

    return bigrams


def calculate_transition_entropy(bigram_counts, n_types=10):
    """Calculate entropy of transition distribution."""
    if not bigram_counts:
        return 0.0

    counts = list(bigram_counts.values())
    if sum(counts) == 0:
        return 0.0

    # Normalize to probabilities
    total = sum(counts)
    probs = [c / total for c in counts]

    # Calculate entropy (normalized by max possible)
    max_entropy = np.log2(n_types * n_types)  # Max possible transitions
    return entropy(probs, base=2) / max_entropy if max_entropy > 0 else 0.0


def calculate_self_loop_ratio(bigram_counts):
    """Calculate ratio of self-loops (same->same transitions)."""
    if not bigram_counts:
        return 0.0

    total = sum(bigram_counts.values())
    if total == 0:
        return 0.0

    self_loops = sum(count for bigram, count in bigram_counts.items()
                     if bigram.split('->')[0] == bigram.split('->')[1])

    return self_loops / total


print('\nExtracting session bigrams...')
student_course_bigrams = {}

grouped = df.groupby(['student_id', 'course_id'])
for (student_id, course_id), user_course_df in grouped:
    # Extract bigrams from each session
    all_bigrams = []
    for session_id, session_df in user_course_df.groupby('session_id'):
        session_bigrams = extract_session_bigrams(session_df)
        all_bigrams.extend(session_bigrams)

    student_course_bigrams[(student_id, course_id)] = Counter(all_bigrams)

print(f'  Processed {len(student_course_bigrams)} student-course pairs')

# Identify top bigrams
print('Identifying top bigrams...')
total_counts = Counter()
for counts in student_course_bigrams.values():
    total_counts.update(counts)

top_bigrams = [bigram for bigram, _ in total_counts.most_common(15)]
print(f'  Top 15 bigrams:')
for i, bigram in enumerate(top_bigrams, 1):
    print(f'    {i:2d}. {bigram}: {total_counts[bigram]:,}')

# Create feature dataframe
print('\nCreating feature matrix...')
features_list = []

RESOURCE_TYPES = list(set(CATEGORY_MAPPING.values()))

for (student_id, course_id), bigram_counts in student_course_bigrams.items():
    features = {
        'student_id': student_id,
        'course_id': course_id,
    }

    # Top bigram counts
    for bigram in top_bigrams:
        safe_name = bigram.replace('->', '_to_').replace(' ', '_')
        features[f'bigram_{safe_name}'] = bigram_counts.get(bigram, 0)

    # Aggregate features
    features['transition_entropy'] = calculate_transition_entropy(bigram_counts, len(RESOURCE_TYPES))
    features['self_loop_ratio'] = calculate_self_loop_ratio(bigram_counts)
    features['total_transitions'] = sum(bigram_counts.values())
    features['unique_transitions'] = len(bigram_counts)

    # Transition diversity (unique transitions / possible)
    features['transition_diversity'] = len(bigram_counts) / (len(RESOURCE_TYPES) ** 2) if len(RESOURCE_TYPES) > 0 else 0

    features_list.append(features)

features_df = pd.DataFrame(features_list)
print(f'  Created {len(features_df)} rows, {len(features_df.columns)} columns')

# Save
features_df.to_parquet(OUTPUT_FILE, index=False)
print(f'\n✓ Saved to: {OUTPUT_FILE}')

# Summary stats
print('\nFeature Summary:')
numeric_cols = features_df.select_dtypes(include=[np.number]).columns
for col in numeric_cols[:10]:
    print(f'  {col}: mean={features_df[col].mean():.3f}, std={features_df[col].std():.3f}')
