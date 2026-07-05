#!/usr/bin/env python3
"""
Calculate ALL features with a time cutoff.

This script recalculates all 203 features using only page views
from the first N weeks of each course.

Usage:
    python calculate_features_with_cutoff.py --cutoff 4
    python calculate_features_with_cutoff.py --all  # Run all cutoffs (2, 4, 6, 8)

Output:
    data/enriched_features/cutoff_week_{N}/
        - session_features.parquet
        - category_features.parquet
        - proactivity_features.parquet
        - pca_features.parquet
        - weekly_features.parquet
        - ngram_features.parquet
        - graph_features.parquet
        - time_features.parquet
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from collections import Counter
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
PAGE_VIEWS_FILE = BASE_DIR / "data/page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = BASE_DIR / "data/page_views/student_enrollments.csv"
TIME_RANGES_FILE = BASE_DIR / "data/analysis/course_time_ranges.json"

# Model courses
COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Session threshold
SESSION_GAP_MINUTES = 30


def normalize_user_id(user_id):
    """Normalize Canvas user_id to short format."""
    # Canvas shard format: 155100000000XXXXX -> XXXXX
    if user_id > 10000000000:
        return user_id % 10000000000
    return user_id


def load_data():
    """Load and filter page views."""
    print("Loading data...")
    df = pd.read_parquet(PAGE_VIEWS_FILE)
    df['created_at'] = pd.to_datetime(df['created_at'])
    df = df[df['course_id'].isin(COURSES)].copy()

    # Normalize user_id to short format (matching enrollments)
    df['user_id'] = df['user_id'].apply(normalize_user_id)

    # Load enrollments
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)

    print(f"  Loaded {len(df):,} page views for {len(COURSES)} courses")
    print(f"  Unique users: {df['user_id'].nunique()}")
    return df, df_enroll


def get_course_starts(df):
    """Calculate course start dates (5th percentile)."""
    course_starts = {}
    for course_id in COURSES:
        df_course = df[df['course_id'] == course_id]
        if len(df_course) > 0:
            course_starts[course_id] = df_course['created_at'].quantile(0.05)
    return course_starts


def filter_by_cutoff(df, course_starts, cutoff_weeks):
    """Filter page views to only include first N weeks."""
    filtered = []
    for course_id in COURSES:
        if course_id not in course_starts:
            continue
        start = course_starts[course_id]
        cutoff_date = start + timedelta(weeks=cutoff_weeks)
        df_course = df[(df['course_id'] == course_id) & (df['created_at'] <= cutoff_date)]
        filtered.append(df_course)

    df_filtered = pd.concat(filtered, ignore_index=True)
    print(f"  Filtered to {len(df_filtered):,} page views (cutoff: {cutoff_weeks} weeks)")
    return df_filtered


# =============================================================================
# FEATURE CALCULATION FUNCTIONS
# =============================================================================

def calc_session_features(df, course_starts):
    """Calculate session-based features."""
    print("  Calculating session features...")
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        if len(group) < 2:
            continue

        group = group.sort_values('created_at')
        timestamps = pd.to_datetime(group['created_at'])

        # Calculate gaps
        gaps = timestamps.diff().dt.total_seconds() / 60
        session_starts = gaps >= SESSION_GAP_MINUTES
        session_starts.iloc[0] = True
        session_ids = session_starts.cumsum()

        # Session metrics
        sessions = []
        for sid in session_ids.unique():
            mask = session_ids == sid
            ts = timestamps[mask]
            duration = (ts.max() - ts.min()).total_seconds() / 60
            sessions.append({'duration': duration, 'views': mask.sum()})

        if not sessions:
            continue

        durations = [s['duration'] for s in sessions]
        views = [s['views'] for s in sessions]
        n_sessions = len(sessions)

        # Course span
        course_start = course_starts.get(course_id, timestamps.min())
        total_span = max((timestamps.max() - course_start).days / 7, 1)

        # Session gaps for regularity
        session_times = []
        for sid in session_ids.unique():
            session_times.append(timestamps[session_ids == sid].min())
        session_times = pd.Series(session_times)
        session_gaps = session_times.diff().dt.total_seconds() / 60
        session_gaps = session_gaps[session_gaps >= SESSION_GAP_MINUTES]

        regularity = 0
        if len(session_gaps) > 1:
            gap_mean = session_gaps.mean()
            gap_std = session_gaps.std()
            if gap_mean > 0:
                regularity = max(0, 1 - (gap_std / gap_mean))

        results.append({
            'user_id': user_id,
            'course_id': course_id,
            'session_count': n_sessions,
            'session_duration_mean': np.mean(durations),
            'session_duration_std': np.std(durations) if n_sessions > 1 else 0,
            'session_duration_median': np.median(durations),
            'sessions_per_week': n_sessions / total_span,
            'views_per_session': np.mean(views),
            'short_sessions_pct': sum(1 for d in durations if d < 5) / n_sessions * 100,
            'long_sessions_pct': sum(1 for d in durations if d > 30) / n_sessions * 100,
            'total_views': len(group),
            'total_time_min': sum(durations),
            'session_regularity': regularity,
        })

    return pd.DataFrame(results)


def calc_category_features(df):
    """Calculate category-based features."""
    print("  Calculating category features...")
    results = []

    categories = ['files', 'discussions', 'quizzes', 'assignments', 'pages',
                  'modules', 'grades', 'announcements', 'home']

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        total = len(group)
        if total == 0:
            continue

        features = {'user_id': user_id, 'course_id': course_id, 'total_views': total}

        for cat in categories:
            cat_data = group[group['resource_type'] == cat]
            views = len(cat_data)
            features[f'{cat}_views'] = views
            features[f'{cat}_views_pct'] = views / total * 100 if total > 0 else 0
            features[f'{cat}_unique_resources'] = cat_data['resource_id'].nunique() if 'resource_id' in cat_data.columns else views
            features[f'{cat}_time_min'] = cat_data['interaction_seconds'].sum() / 60 if 'interaction_seconds' in cat_data.columns else 0

        # Derived features
        content = features.get('files_views', 0) + features.get('pages_views', 0) + features.get('discussions_views', 0)
        assess = features.get('quizzes_views', 0) + features.get('assignments_views', 0)
        features['content_vs_assessment_ratio'] = content / assess if assess > 0 else content

        disc_views = features.get('discussions_views', 0)
        disc_participated = len(group[(group['resource_type'] == 'discussions') & (group.get('participated', False) == True)])
        features['discussion_participation_rate'] = disc_participated / disc_views if disc_views > 0 else 0

        # Grades check frequency
        grades_views = features.get('grades_views', 0)
        timestamps = pd.to_datetime(group['created_at'])
        span_weeks = max((timestamps.max() - timestamps.min()).days / 7, 1)
        features['grades_check_per_week'] = grades_views / span_weeks

        results.append(features)

    return pd.DataFrame(results)


def calc_proactivity_features(df, course_starts):
    """Calculate proactivity (PCT) features."""
    print("  Calculating proactivity features...")
    results = []

    resource_types = ['files', 'discussions', 'quizzes', 'assignments', 'pages', 'modules']

    for course_id in df['course_id'].unique():
        df_course = df[df['course_id'] == course_id]
        course_users = df_course['user_id'].unique()

        # Calculate PCT rankings for each resource type
        pct_data = {uid: {} for uid in course_users}

        for rtype in resource_types:
            df_type = df_course[df_course['resource_type'] == rtype]
            if len(df_type) == 0:
                continue

            # Get first access time per user per resource
            first_access = df_type.groupby(['user_id', 'resource_id'])['created_at'].min().reset_index()

            for resource_id in first_access['resource_id'].unique():
                res_data = first_access[first_access['resource_id'] == resource_id].sort_values('created_at')
                n_accessors = len(res_data)

                for rank, (_, row) in enumerate(res_data.iterrows(), 1):
                    pct = (n_accessors - rank + 1) / n_accessors
                    if row['user_id'] in pct_data:
                        if rtype not in pct_data[row['user_id']]:
                            pct_data[row['user_id']][rtype] = []
                        pct_data[row['user_id']][rtype].append(pct)

        # Calculate features per user
        for user_id in course_users:
            user_pct = pct_data.get(user_id, {})
            features = {'user_id': user_id, 'course_id': course_id}

            all_pcts = []
            for rtype in resource_types:
                prefix = rtype[:4] if rtype != 'discussions' else 'disc'
                pcts = user_pct.get(rtype, [])

                features[f'{prefix}_n_resources'] = len(pcts)
                features[f'{prefix}_mean_pct'] = np.mean(pcts) if pcts else 0
                features[f'{prefix}_median_pct'] = np.median(pcts) if pcts else 0
                features[f'{prefix}_std_pct'] = np.std(pcts) if len(pcts) > 1 else 0
                features[f'{prefix}_access_rate'] = len(pcts) / max(1, len(pcts))  # Simplified
                features[f'{prefix}_top25_rate'] = sum(1 for p in pcts if p >= 0.75) / max(1, len(pcts))
                features[f'{prefix}_top50_rate'] = sum(1 for p in pcts if p >= 0.50) / max(1, len(pcts))

                # Histogram bins
                if pcts:
                    pct_array = np.array(pcts)
                    n = len(pcts)
                    features[f'{prefix}_hist_b1'] = np.sum(pct_array == 0) / n
                    features[f'{prefix}_hist_b2'] = np.sum((pct_array > 0) & (pct_array <= 0.25)) / n
                    features[f'{prefix}_hist_b3'] = np.sum((pct_array > 0.25) & (pct_array <= 0.50)) / n
                    features[f'{prefix}_hist_b4'] = np.sum((pct_array > 0.50) & (pct_array <= 0.75)) / n
                    features[f'{prefix}_hist_b5'] = np.sum(pct_array > 0.75) / n
                else:
                    for b in range(1, 6):
                        features[f'{prefix}_hist_b{b}'] = 0

                all_pcts.extend(pcts)

            # Overall proactivity
            features['overall_proactivity'] = np.mean(all_pcts) if all_pcts else 0

            # DCT coefficients
            if len(all_pcts) >= 4:
                try:
                    pct_array = np.array(all_pcts[:20])  # Limit for stability
                    pct_norm = pct_array / (pct_array.sum() + 1e-10)
                    coeffs = dct(pct_norm, norm='ortho')
                    for i in range(min(4, len(coeffs))):
                        features[f'dct_pct_{i}'] = coeffs[i]
                except:
                    for i in range(4):
                        features[f'dct_pct_{i}'] = 0
            else:
                for i in range(4):
                    features[f'dct_pct_{i}'] = 0

            # Download features (simplified)
            df_user = df_course[df_course['user_id'] == user_id]
            file_views = len(df_user[df_user['resource_type'] == 'files'])
            features['download_count'] = file_views  # Simplified
            features['download_rate'] = 1.0 if file_views > 0 else 0
            features['unique_files_downloaded'] = df_user[df_user['resource_type'] == 'files']['resource_id'].nunique() if 'resource_id' in df_user.columns else file_views

            results.append(features)

    return pd.DataFrame(results)


def calc_pca_features(df, course_starts):
    """Calculate PCA features."""
    print("  Calculating PCA features...")
    results = []

    resource_types = {'files': 3, 'discussions': 3, 'pages': 3, 'modules': 2}

    for course_id in df['course_id'].unique():
        df_course = df[df['course_id'] == course_id]
        course_users = df_course['user_id'].unique()

        user_features = {uid: {'user_id': uid, 'course_id': course_id} for uid in course_users}

        for rtype, n_components in resource_types.items():
            prefix = rtype[:4] if rtype != 'discussions' else 'disc'
            if rtype == 'modules':
                prefix = 'mods'

            df_type = df_course[df_course['resource_type'] == rtype]

            if len(df_type) == 0:
                for uid in course_users:
                    for i in range(1, n_components + 1):
                        user_features[uid][f'{prefix}_pc{i}'] = 0
                    user_features[uid][f'{prefix}_var_explained'] = 0
                    user_features[uid][f'{prefix}_n_resources'] = 0
                continue

            # Create user-resource matrix
            resource_ids = df_type['resource_id'].unique() if 'resource_id' in df_type.columns else [0]
            n_resources = len(resource_ids)

            for uid in course_users:
                user_features[uid][f'{prefix}_n_resources'] = n_resources

            if n_resources < 2:
                for uid in course_users:
                    for i in range(1, n_components + 1):
                        user_features[uid][f'{prefix}_pc{i}'] = 0
                    user_features[uid][f'{prefix}_var_explained'] = 0
                continue

            # Build matrix
            matrix = []
            user_order = []
            for uid in course_users:
                user_type = df_type[df_type['user_id'] == uid]
                row = [len(user_type[user_type.get('resource_id', 0) == rid]) for rid in resource_ids] if 'resource_id' in df_type.columns else [len(user_type)]
                matrix.append(row)
                user_order.append(uid)

            X = np.array(matrix)

            # Apply PCA
            if X.shape[0] >= n_components and X.shape[1] >= n_components:
                try:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    actual_components = min(n_components, X.shape[0], X.shape[1])
                    pca = PCA(n_components=actual_components)
                    X_pca = pca.fit_transform(X_scaled)
                    var_explained = sum(pca.explained_variance_ratio_)

                    for i, uid in enumerate(user_order):
                        for j in range(actual_components):
                            user_features[uid][f'{prefix}_pc{j+1}'] = X_pca[i, j]
                        user_features[uid][f'{prefix}_var_explained'] = var_explained
                except:
                    for uid in course_users:
                        for i in range(1, n_components + 1):
                            user_features[uid][f'{prefix}_pc{i}'] = 0
                        user_features[uid][f'{prefix}_var_explained'] = 0
            else:
                for uid in course_users:
                    for i in range(1, n_components + 1):
                        user_features[uid][f'{prefix}_pc{i}'] = 0
                    user_features[uid][f'{prefix}_var_explained'] = 0

        results.extend(user_features.values())

    return pd.DataFrame(results)


def calc_weekly_features(df, course_starts):
    """Calculate weekly activity features."""
    print("  Calculating weekly features...")
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        if len(group) < 2:
            continue

        timestamps = pd.to_datetime(group['created_at'])
        course_start = course_starts.get(course_id, timestamps.min())

        # Calculate week for each view
        weeks = ((timestamps - course_start).dt.days // 7 + 1).clip(lower=1)
        weekly_counts = weeks.value_counts().sort_index()

        active_weeks = weekly_counts.index.tolist()

        features = {
            'user_id': user_id,
            'course_id': course_id,
            'active_weeks_count': len(active_weeks),
            'first_active_week': min(active_weeks) if active_weeks else 0,
            'last_active_week': max(active_weeks) if active_weeks else 0,
            'peak_week': weekly_counts.idxmax() if len(weekly_counts) > 0 else 0,
            'total_views': len(group),
            'total_sessions': len(group),  # Simplified
        }

        # Early vs late activity
        mid_week = (max(active_weeks) + min(active_weeks)) / 2 if active_weeks else 1
        early_views = sum(weekly_counts.get(w, 0) for w in active_weeks if w <= mid_week)
        late_views = sum(weekly_counts.get(w, 0) for w in active_weeks if w > mid_week)

        features['early_semester_views'] = early_views
        features['late_semester_views'] = late_views
        features['early_vs_late_ratio'] = early_views / late_views if late_views > 0 else early_views

        # Week-over-week change
        if len(weekly_counts) > 1:
            changes = weekly_counts.diff().dropna()
            features['avg_week_over_week_change'] = changes.mean()
        else:
            features['avg_week_over_week_change'] = 0

        # Activity consistency
        if len(weekly_counts) > 1:
            features['activity_consistency'] = weekly_counts.std() / weekly_counts.mean() if weekly_counts.mean() > 0 else 0
        else:
            features['activity_consistency'] = 0

        # Engagement pattern
        if features['early_vs_late_ratio'] > 1.5:
            pattern = 1  # Early focused
        elif features['early_vs_late_ratio'] < 0.67:
            pattern = 3  # Late focused
        elif features['activity_consistency'] < 0.5:
            pattern = 4  # Consistent
        else:
            pattern = 2  # Variable
        features['engagement_pattern'] = pattern

        # First access week for specific resources
        for rtype in ['assignments', 'quizzes', 'discussions', 'grades']:
            type_data = group[group['resource_type'] == rtype]
            if len(type_data) > 0:
                first_ts = pd.to_datetime(type_data['created_at']).min()
                week = max(1, ((first_ts - course_start).days // 7) + 1)
                features[f'{rtype}_first_access_week'] = week
            else:
                features[f'{rtype}_first_access_week'] = 0

        results.append(features)

    return pd.DataFrame(results)


def calc_ngram_features(df):
    """Calculate navigation pattern (n-gram) features."""
    print("  Calculating n-gram features...")
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        if len(group) < 2:
            continue

        group = group.sort_values('created_at')
        types = group['resource_type'].tolist()

        # Bigrams
        bigrams = [(types[i], types[i+1]) for i in range(len(types)-1)]
        bigram_counts = Counter(bigrams)
        total_transitions = len(bigrams)

        features = {
            'user_id': user_id,
            'course_id': course_id,
            'total_transitions': total_transitions,
        }

        # Common bigrams
        common_bigrams = [
            ('files', 'files'), ('files', 'home'), ('home', 'home'),
            ('home', 'assignments'), ('home', 'modules'),
            ('assignments', 'assignments'), ('assignments', 'files'),
            ('assignments', 'home'), ('assignments', 'quizzes'),
            ('quizzes', 'quizzes'), ('discussions', 'discussions'),
            ('modules', 'modules'), ('modules', 'files'),
            ('other', 'assignments'), ('pages', 'pages'),
        ]

        for bg in common_bigrams:
            key = f'bigram_{bg[0]}_to_{bg[1]}'
            features[key] = bigram_counts.get(bg, 0) / total_transitions if total_transitions > 0 else 0

        # Self-loop ratio
        self_loops = sum(1 for bg in bigrams if bg[0] == bg[1])
        features['self_loop_ratio'] = self_loops / total_transitions if total_transitions > 0 else 0

        # Transition diversity
        features['transition_diversity'] = len(set(bigrams))

        # Transition entropy
        if total_transitions > 0:
            probs = [c / total_transitions for c in bigram_counts.values()]
            features['transition_entropy'] = -sum(p * np.log2(p + 1e-10) for p in probs)
        else:
            features['transition_entropy'] = 0

        # Dominant transition
        if bigram_counts:
            dominant = bigram_counts.most_common(1)[0][0]
            features['dominant_transition'] = hash(dominant) % 1000  # Encoded
        else:
            features['dominant_transition'] = 0

        results.append(features)

    return pd.DataFrame(results)


def calc_graph_features(df, df_enroll):
    """Calculate graph-based features."""
    print("  Calculating graph features...")
    results = []

    for course_id in df['course_id'].unique():
        df_course = df[df['course_id'] == course_id]
        course_users = df_course['user_id'].unique()

        # Get passing students for this course
        course_enroll = df_enroll[df_enroll['course_id'] == course_id]
        passing_users = set(course_enroll[course_enroll['final_score'] >= 60]['user_id'].tolist())

        # Resources accessed by passing students
        passing_resources = set()
        for uid in passing_users:
            user_resources = df_course[df_course['user_id'] == uid]
            if 'resource_id' in user_resources.columns:
                passing_resources.update(user_resources['resource_id'].unique())

        # Total resources in course
        total_resources = df_course['resource_id'].nunique() if 'resource_id' in df_course.columns else len(df_course)

        # Average resources per user
        avg_resources = df_course.groupby('user_id')['resource_id'].nunique().mean() if 'resource_id' in df_course.columns else len(df_course) / len(course_users)

        for user_id in course_users:
            user_data = df_course[df_course['user_id'] == user_id]

            user_resources = set(user_data['resource_id'].unique()) if 'resource_id' in user_data.columns else set()
            n_resources = len(user_resources) if user_resources else len(user_data)

            features = {
                'user_id': user_id,
                'course_id': course_id,
                'unique_resources': n_resources,
                'resource_coverage': n_resources / total_resources if total_resources > 0 else 0,
                'resources_vs_avg': n_resources / avg_resources if avg_resources > 0 else 1,
            }

            # Resource diversity (by type)
            type_counts = user_data['resource_type'].value_counts()
            if len(type_counts) > 0:
                probs = type_counts / type_counts.sum()
                features['resource_diversity'] = -sum(p * np.log2(p + 1e-10) for p in probs)
            else:
                features['resource_diversity'] = 0

            # Jaccard similarity to passing students
            if passing_resources and user_resources:
                intersection = len(user_resources & passing_resources)
                union = len(user_resources | passing_resources)
                features['jaccard_to_passing'] = intersection / union if union > 0 else 0
            else:
                features['jaccard_to_passing'] = 0

            # Cluster assignment (simplified - based on coverage)
            if features['resource_coverage'] > 0.7:
                cluster = 2  # High engagement
            elif features['resource_coverage'] > 0.3:
                cluster = 1  # Medium
            else:
                cluster = 0  # Low
            features['access_cluster'] = cluster

            results.append(features)

    return pd.DataFrame(results)


def calc_time_features(df):
    """Calculate time-of-day features."""
    print("  Calculating time features...")
    results = []

    for (user_id, course_id), group in df.groupby(['user_id', 'course_id']):
        timestamps = pd.to_datetime(group['created_at'])
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofweek

        total = len(hours)
        if total == 0:
            continue

        features = {
            'user_id': user_id,
            'course_id': course_id,
            'pct_morning': sum((hours >= 6) & (hours < 12)) / total * 100,
            'pct_afternoon': sum((hours >= 12) & (hours < 18)) / total * 100,
            'pct_evening': sum((hours >= 18) & (hours < 24)) / total * 100,
            'pct_night': sum((hours >= 0) & (hours < 6)) / total * 100,
            'pct_weekend': sum(days >= 5) / total * 100,
            'work_hours_ratio': sum((hours >= 9) & (hours < 18) & (days < 5)) / total * 100,
            'late_night_ratio': sum((hours >= 0) & (hours < 6)) / total * 100,
            'peak_hour': hours.mode().iloc[0] if len(hours.mode()) > 0 else 12,
            'peak_day': days.mode().iloc[0] if len(days.mode()) > 0 else 0,
        }

        # Hour diversity
        hour_counts = hours.value_counts()
        if len(hour_counts) > 0:
            probs = hour_counts / hour_counts.sum()
            features['hour_diversity'] = -sum(p * np.log2(p + 1e-10) for p in probs)
        else:
            features['hour_diversity'] = 0

        # Time consistency
        if len(hours) > 1:
            features['time_consistency'] = 1 - (hours.std() / 12) if hours.std() < 12 else 0
        else:
            features['time_consistency'] = 0

        results.append(features)

    return pd.DataFrame(results)


def process_cutoff(df_full, df_enroll, cutoff_weeks, course_starts):
    """Process all features for a single cutoff."""
    print(f"\n{'='*60}")
    print(f"Processing cutoff: {cutoff_weeks} weeks")
    print(f"{'='*60}")

    # Filter data
    df = filter_by_cutoff(df_full, course_starts, cutoff_weeks)

    if len(df) == 0:
        print("  No data after filtering!")
        return

    # Output directory
    output_dir = BASE_DIR / f"data/enriched_features/cutoff_week_{cutoff_weeks}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate all features
    dfs = {}

    dfs['session'] = calc_session_features(df, course_starts)
    dfs['category'] = calc_category_features(df)
    dfs['proactivity'] = calc_proactivity_features(df, course_starts)
    dfs['pca'] = calc_pca_features(df, course_starts)
    dfs['weekly'] = calc_weekly_features(df, course_starts)
    dfs['ngram'] = calc_ngram_features(df)
    dfs['graph'] = calc_graph_features(df, df_enroll)
    dfs['time'] = calc_time_features(df)

    # Save
    for name, df_feat in dfs.items():
        if df_feat is not None and len(df_feat) > 0:
            output_file = output_dir / f"{name}_features.parquet"
            df_feat.to_parquet(output_file, index=False)
            print(f"  Saved {name}_features.parquet ({len(df_feat)} rows, {len(df_feat.columns)} cols)")

    # Summary
    print(f"\n  Summary for cutoff {cutoff_weeks} weeks:")
    print(f"    Page views: {len(df):,}")
    print(f"    Students: {df['user_id'].nunique()}")
    print(f"    Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Calculate features with time cutoff')
    parser.add_argument('--cutoff', type=int, help='Cutoff in weeks (2, 4, 6, 8)')
    parser.add_argument('--all', action='store_true', help='Run all cutoffs')
    args = parser.parse_args()

    # Load data
    df_full, df_enroll = load_data()
    course_starts = get_course_starts(df_full)

    # Process cutoffs
    if args.all:
        cutoffs = [2, 4, 6, 8]
    elif args.cutoff:
        cutoffs = [args.cutoff]
    else:
        print("Please specify --cutoff N or --all")
        return

    for cutoff in cutoffs:
        process_cutoff(df_full, df_enroll, cutoff, course_starts)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == "__main__":
    main()
