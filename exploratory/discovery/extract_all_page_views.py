#!/usr/bin/env python3
"""
Extract ALL page views for a specific student (no course filter).

Usage:
    python extract_all_page_views.py --user-id 117462
    python extract_all_page_views.py --user-id 117462 --start-date 2025-08-01 --end-date 2025-12-31
    python extract_all_page_views.py --user-id 117462 --output-dir data/page_views
"""

import argparse
import os
import re
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}


def extract_course_id_from_url(url: str) -> int:
    """Extract course_id from a Canvas URL."""
    match = re.search(r'/courses/(\d+)', url)
    return int(match.group(1)) if match else -1


def safe_request(url: str, params: Optional[Dict] = None) -> Tuple[Optional[List], Optional[str]]:
    """
    Make a rate-limited request to the Canvas API.

    Returns:
        Tuple of (data, next_url) where next_url is from Link header
    """
    try:
        r = requests.get(url, headers=headers, params=params, timeout=30)

        if r.status_code == 401:
            print("Error: Unauthorized. Check your API token.")
            return None, None

        if r.status_code == 404:
            print(f"Error: Resource not found at {url}")
            return None, None

        if r.status_code != 200:
            print(f"Error {r.status_code}: {r.text[:200]}")
            return None, None

        # Adaptive rate limiting
        remaining = float(r.headers.get('X-Rate-Limit-Remaining', 500))
        if remaining < 50:
            time.sleep(30)
        elif remaining < 100:
            time.sleep(10)
        elif remaining < 200:
            time.sleep(5)
        elif remaining < 300:
            time.sleep(2)
        elif remaining < 500:
            time.sleep(1)
        else:
            time.sleep(0.3)

        # Get next page URL from Link header
        link_header = r.headers.get('Link', '')
        match = re.search(r'<([^>]+)>; rel="next"', link_header)
        next_url = match.group(1) if match else None

        return r.json(), next_url

    except requests.exceptions.Timeout:
        print(f"Timeout requesting {url}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


def get_all_page_views(user_id: int, start_date: str, end_date: str) -> List[Dict]:
    """
    Fetch ALL page views for a student (no course filter).

    Args:
        user_id: Canvas user ID
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        List of all page view records
    """
    print(f"Fetching ALL page views for student {user_id}...")
    print(f"Date range: {start_date} to {end_date}")

    all_views = []
    url = f'{API_URL}/api/v1/users/{user_id}/page_views'
    params = {
        'start_time': f'{start_date}T00:00:00Z',
        'end_time': f'{end_date}T23:59:59Z',
        'per_page': 100
    }

    page_num = 1

    while url:
        # Only use params on first request
        data, next_url = safe_request(url, params if page_num == 1 else None)

        if data is None:
            break

        if not data:
            break

        # Add all page views (no filtering)
        for pv in data:
            all_views.append({
                'user_id': user_id,
                'course_id': extract_course_id_from_url(pv.get('url', '')),
                'url': pv.get('url'),
                'context_type': pv.get('context_type'),
                'asset_type': pv.get('asset_type'),
                'controller': pv.get('controller'),
                'action': pv.get('action'),
                'interaction_seconds': pv.get('interaction_seconds'),
                'created_at': pv.get('created_at'),
                'participated': pv.get('participated', False),
                'user_agent': pv.get('user_agent', '')[:200] if pv.get('user_agent') else None
            })

        print(f"  Page {page_num}: {len(data)} views (cumulative: {len(all_views)})")

        url = next_url
        page_num += 1

    print(f"\nTotal page views: {len(all_views)}")

    return all_views


def save_to_parquet(data: List[Dict], filepath: str) -> bool:
    """Save data to Parquet file."""
    if not data:
        print("No data to save")
        return False

    df = pd.DataFrame(data)
    df.to_parquet(filepath, index=False)
    print(f"Saved {len(data)} records to {filepath}")
    return True


def print_summary(data: List[Dict]):
    """Print summary statistics."""
    if not data:
        print("No data to summarize")
        return

    df = pd.DataFrame(data)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    print(f"\nDate range: {df['created_at'].min()} to {df['created_at'].max()}")

    total_time = df['interaction_seconds'].sum()
    print(f"Total interaction time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    print("\nCourse breakdown:")
    course_counts = df['course_id'].value_counts()
    print(course_counts.to_string())

    print("\nController breakdown:")
    print(df['controller'].value_counts().to_string())

    print("\nAction breakdown (top 10):")
    print(df['action'].value_counts().head(10).to_string())

    participated = df['participated'].sum()
    print(f"\nParticipation events: {participated}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ALL page views for a specific student (no course filter)'
    )
    parser.add_argument('--user-id', type=int, required=True,
                        help='Canvas user ID')
    parser.add_argument('--start-date', type=str, default='2025-08-01',
                        help='Start date (YYYY-MM-DD), default: 2025-08-01')
    parser.add_argument('--end-date', type=str, default='2025-12-31',
                        help='End date (YYYY-MM-DD), default: 2025-12-31')
    parser.add_argument('--output-dir', type=str, default='exploratory/data/page_views',
                        help='Output directory for parquet file')

    args = parser.parse_args()

    # Validate connection
    print("=" * 60)
    print("CANVAS ALL PAGE VIEWS EXTRACTOR")
    print("=" * 60)

    r = requests.get(f'{API_URL}/api/v1/users/self', headers=headers)
    if r.status_code != 200:
        print(f"Failed to connect to Canvas API: {r.status_code}")
        return

    user = r.json()
    print(f"Connected as: {user.get('name', 'Unknown')}")
    print()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Fetch ALL page views
    page_views = get_all_page_views(
        user_id=args.user_id,
        start_date=args.start_date,
        end_date=args.end_date
    )

    # Save to parquet
    output_file = f"{args.output_dir}/student_{args.user_id}_all.parquet"
    save_to_parquet(page_views, output_file)

    # Print summary
    print_summary(page_views)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
