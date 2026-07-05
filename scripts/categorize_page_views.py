#!/usr/bin/env python3
"""
Categorize page views by resource type.

Categories:
- files: /files/{id}, /items/{id}
- discussions: /discussion_topics/{id}
- announcements: /announcements/{id}
- quizzes: /quizzes/{id}
- assignments: /assignments/{id}
- pages: /pages/{slug}
- modules: /modules/{id}
- grades: /grades
- home: / (course home)
- other: everything else
"""

import re
import pandas as pd
from pathlib import Path

INPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/all_page_views.parquet')
OUTPUT_FILE = Path('/home/paul/projects/uautonoma/data/page_views/categorized_page_views.parquet')


def extract_course_id(url):
    """Extract course_id from Canvas URL."""
    if not url or pd.isna(url):
        return None
    match = re.search(r'/courses/(\d+)', str(url))
    return int(match.group(1)) if match else None


def categorize_url(url):
    """Categorize a Canvas URL by resource type."""
    if not url or pd.isna(url):
        return 'other', None

    url = str(url)

    # Files
    if re.search(r'/files/(\d+)', url):
        match = re.search(r'/files/(\d+)', url)
        return 'files', int(match.group(1))

    # Items (also files in some contexts)
    if re.search(r'/items/(\d+)', url):
        match = re.search(r'/items/(\d+)', url)
        return 'files', int(match.group(1))

    # Discussion topics
    if re.search(r'/discussion_topics/(\d+)', url):
        match = re.search(r'/discussion_topics/(\d+)', url)
        return 'discussions', int(match.group(1))

    # Announcements (with optional ID)
    ann_match = re.search(r'/announcements/(\d+)', url)
    if ann_match:
        return 'announcements', int(ann_match.group(1))
    if re.search(r'/announcements(/|$|\?)', url):
        return 'announcements', None

    # Quizzes
    if re.search(r'/quizzes/(\d+)', url):
        match = re.search(r'/quizzes/(\d+)', url)
        return 'quizzes', int(match.group(1))

    # Assignments
    if re.search(r'/assignments/(\d+)', url):
        match = re.search(r'/assignments/(\d+)', url)
        return 'assignments', int(match.group(1))

    # Pages
    if re.search(r'/pages/([^/?]+)', url):
        match = re.search(r'/pages/([^/?]+)', url)
        return 'pages', match.group(1)

    # Modules
    if re.search(r'/modules(/|\?|$)', url) or re.search(r'/modules/(\d+)', url):
        match = re.search(r'/modules/(\d+)', url)
        return 'modules', int(match.group(1)) if match else None

    # Grades
    if re.search(r'/grades(/|$|\?)', url):
        return 'grades', None

    # Course home
    if re.match(r'^/courses/\d+/?$', url) or re.search(r'/courses/\d+\?', url):
        return 'home', None

    # API calls (exclude from analysis but categorize)
    if '/api/' in url:
        return 'api', None

    return 'other', None


def main():
    print('Loading page views...')
    df = pd.read_parquet(INPUT_FILE)
    print(f'Loaded {len(df)} page views')
    print()

    # Get URL column (may be 'http_request' or 'url')
    url_col = 'http_request' if 'http_request' in df.columns else 'url'

    # Extract course_id from URLs
    print('Extracting course IDs...')
    df['course_id'] = df[url_col].apply(extract_course_id)

    # Categorize URLs
    print('Categorizing URLs...')
    categorized = df[url_col].apply(categorize_url)
    df['resource_type'] = categorized.apply(lambda x: x[0])
    df['resource_id'] = categorized.apply(lambda x: str(x[1]) if x[1] is not None else None)

    # Reclassify discussions that are actually announcements
    print('Reclassifying announcements...')
    referrer_col = 'referrer' if 'referrer' in df.columns else None

    n_before = (df['resource_type'] == 'announcements').sum()

    # Rule 1: URLs with only_announcements=1 (API listing calls for announcements)
    rule1_mask = (
        df['resource_type'].isin(['discussions', 'api'])
        & df[url_col].fillna('').str.contains('only_announcements=1', regex=False)
    )
    df.loc[rule1_mask, 'resource_type'] = 'announcements'
    n_rule1 = rule1_mask.sum()
    print(f'  Rule 1 (only_announcements=1 param): {n_rule1:,} rows reclassified')

    # Rule 2: Views where referrer contains /announcements
    if referrer_col:
        rule2_mask = (
            (df['resource_type'] == 'discussions')
            & df[referrer_col].fillna('').str.contains('/announcements', regex=False)
        )
        df.loc[rule2_mask, 'resource_type'] = 'announcements'
        n_rule2 = rule2_mask.sum()
        print(f'  Rule 2 (referrer /announcements): {n_rule2:,} rows reclassified')
    else:
        print('  Rule 2 skipped (no referrer column)')

    n_after = (df['resource_type'] == 'announcements').sum()
    n_with_id = df.loc[df['resource_type'] == 'announcements', 'resource_id'].notna().sum()
    print(f'  Announcements: {n_before:,} → {n_after:,} (+{n_after - n_before:,})')
    print(f'  Announcements with resource_id: {n_with_id:,}')

    # Summary
    print()
    print('Category distribution:')
    print(df['resource_type'].value_counts())
    print()

    # Filter to only course-related views (exclude API calls for feature generation)
    df_filtered = df[df['resource_type'] != 'api'].copy()
    print(f'After filtering API calls: {len(df_filtered)} page views')

    # Save
    print()
    print(f'Saving to {OUTPUT_FILE}...')
    df_filtered.to_parquet(OUTPUT_FILE, index=False)
    print('Done!')


if __name__ == '__main__':
    main()
