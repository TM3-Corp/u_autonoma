#!/usr/bin/env python3
"""
Session N-gram Feature Extraction

Extracts sequential pattern features from student clickstream data.
Captures the ORDER of resource access within sessions, which reveals
learning strategies (structured vs. scattered).

Features:
- Top N bigram counts (e.g., modules->files, home->assignments)
- Transition entropy (diversity of navigation patterns)
- Self-loop ratio (e.g., files->files indicates deep reading)
- Most common transition pattern

Based on SOTA: "From Weekly Buckets to Sequential Pattern Mining"
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.stats import entropy

DATA_DIR = Path('/home/paul/projects/uautonoma/data')
PAGE_VIEWS_DIR = DATA_DIR / 'page_views'
ENRICHED_DIR = DATA_DIR / 'enriched_features'

# Resource types to track
RESOURCE_TYPES = ['files', 'discussions', 'pages', 'modules', 'announcements',
                  'home', 'quizzes', 'assignments', 'grades', 'other']


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


def get_top_bigrams(all_bigram_counts, top_n=15):
    """Get the top N most common bigrams across all students."""
    total_counts = Counter()
    for counts in all_bigram_counts.values():
        total_counts.update(counts)

    return [bigram for bigram, _ in total_counts.most_common(top_n)]


def main():
    print('=' * 60)
    print('SESSION N-GRAM FEATURE EXTRACTION')
    print('=' * 60)
    print()

    # Load categorized page views
    pv_path = PAGE_VIEWS_DIR / 'categorized_page_views.parquet'
    if not pv_path.exists():
        print(f'ERROR: {pv_path} not found')
        return

    print('Loading page views...')
    df = pd.read_parquet(pv_path)

    # Use source_user_id for matching with other feature files
    user_col = 'source_user_id' if 'source_user_id' in df.columns else 'user_id'
    df['user_id'] = df[user_col]

    # Filter to target courses
    TARGET_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
    df = df[df['course_id'].isin(TARGET_COURSES)].copy()

    print(f'  Loaded {len(df):,} page views (target courses)')
    print(f'  Unique students: {df["user_id"].nunique()}')
    print(f'  Unique courses: {df["course_id"].nunique()}')

    # Convert timestamps
    df['created_at'] = pd.to_datetime(df['created_at'])

    # Ensure resource_type exists
    if 'resource_type' not in df.columns:
        print('ERROR: resource_type column not found')
        return

    # Check if session_id exists, otherwise create sessions
    if 'session_id' not in df.columns:
        print('Creating session IDs (30-min gap threshold)...')
        df = df.sort_values(['user_id', 'course_id', 'created_at'])
        df['time_gap'] = df.groupby(['user_id', 'course_id'])['created_at'].diff()
        df['new_session'] = df['time_gap'] > pd.Timedelta(minutes=30)
        df['session_id'] = df.groupby(['user_id', 'course_id'])['new_session'].cumsum()

    print()

    # Step 1: Extract bigrams per student-course
    print('Extracting session bigrams...')
    student_course_bigrams = {}

    grouped = df.groupby(['user_id', 'course_id'])
    for (user_id, course_id), user_course_df in grouped:
        # Extract bigrams from each session
        all_bigrams = []
        for session_id, session_df in user_course_df.groupby('session_id'):
            session_bigrams = extract_session_bigrams(session_df)
            all_bigrams.extend(session_bigrams)

        student_course_bigrams[(user_id, course_id)] = Counter(all_bigrams)

    print(f'  Processed {len(student_course_bigrams)} student-course pairs')

    # Step 2: Identify top bigrams
    print('Identifying top bigrams...')
    top_bigrams = get_top_bigrams(student_course_bigrams, top_n=15)
    print(f'  Top 15 bigrams:')
    total_counts = Counter()
    for counts in student_course_bigrams.values():
        total_counts.update(counts)
    for i, bigram in enumerate(top_bigrams, 1):
        print(f'    {i:2d}. {bigram}: {total_counts[bigram]:,}')
    print()

    # Step 3: Create feature dataframe
    print('Creating feature matrix...')
    features_list = []

    for (user_id, course_id), bigram_counts in student_course_bigrams.items():
        features = {
            'user_id': user_id,
            'course_id': course_id,
        }

        # Top bigram counts
        for bigram in top_bigrams:
            safe_name = bigram.replace('->', '_to_').replace(' ', '_')
            features[f'bigram_{safe_name}'] = bigram_counts.get(bigram, 0)

        # Aggregate features
        features['transition_entropy'] = calculate_transition_entropy(bigram_counts)
        features['self_loop_ratio'] = calculate_self_loop_ratio(bigram_counts)
        features['total_transitions'] = sum(bigram_counts.values())

        # Most common transition
        if bigram_counts:
            most_common = bigram_counts.most_common(1)[0][0]
            features['dominant_transition'] = most_common
        else:
            features['dominant_transition'] = 'none'

        # Transition diversity (unique transitions / possible)
        features['transition_diversity'] = len(bigram_counts) / (len(RESOURCE_TYPES) ** 2)

        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    print(f'  Created {len(features_df)} rows, {len(features_df.columns)} columns')

    # Step 4: Save
    output_path = ENRICHED_DIR / 'ngram_features.parquet'
    features_df.to_parquet(output_path, index=False)
    print(f'\nSaved to: {output_path}')

    # Summary stats
    print('\nFeature Summary:')
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols[:5]:
        print(f'  {col}: mean={features_df[col].mean():.3f}, std={features_df[col].std():.3f}')


if __name__ == '__main__':
    main()
