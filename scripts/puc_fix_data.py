#!/usr/bin/env python3
"""
PUC Data Fix Script
===================
Fixes critical data bugs in PUC page views:
1. Extracts real resource IDs from URLs (replacing synthetic hash-based IDs)
2. Applies hybrid URL+controller category mapping
3. Filters out grade=NaN students (withdrawals)
4. Computes temporal columns (week_number, hour, day_of_week)

Input:
  - /home/paul/projects/wave_analysis/Canvas_Files/filtered_page_views.parquet
  - /home/paul/projects/wave_analysis/Canvas_Files/students_grades_processed_with_sigla.csv

Output:
  - data/puc/puc_fixed_data.parquet
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PV_FILE = Path("/home/paul/projects/wave_analysis/Canvas_Files/filtered_page_views.parquet")
GRADES_FILE = Path("/home/paul/projects/wave_analysis/Canvas_Files/students_grades_processed_with_sigla.csv")
OUTPUT_DIR = BASE_DIR / "data" / "puc"
OUTPUT_FILE = OUTPUT_DIR / "puc_fixed_data.parquet"


# ── URL-based resource ID extraction ──────────────────────────────────────
# Patterns ordered by specificity (most specific first)
URL_PATTERNS: list[tuple[str, str, int | None]] = [
    # (regex, category, group_index_for_id)  — None means no numeric ID
    (r"/discussion_topics/(\d+)", "discussions", 1),
    (r"/quizzes/(\d+)", "quizzes", 1),
    (r"/assignments/(\d+)", "assignments", 1),
    (r"/files/(\d+)", "files", 1),
    (r"/modules/items/(\d+)", "modules", 1),
    (r"/modules/(\d+)", "modules", 1),
    (r"/pages/([^/?#]+)", "pages", 1),  # slug, not numeric
    (r"/grades(?:/|$|\?)", "grades", None),
    (r"/announcements(?:/|$|\?)", "announcements", None),
]

# Controller -> category fallback (when URL doesn't match specific patterns)
CONTROLLER_CATEGORY_MAP: dict[str, str] = {
    "discussion_topics": "discussions",
    "discussion_topics_api": "discussions",
    "discussion_entries": "discussions",
    "quizzes": "quizzes",
    "quizzes/quizzes": "quizzes",
    "quiz_submissions": "quizzes",
    "quizzes/quiz_submissions": "quizzes",
    "assignments": "assignments",
    "assignments_api": "assignments",
    "submissions": "assignments",
    "submissions_api": "assignments",
    "files": "files",
    "files_api": "files",
    "context_module_items_api": "modules",
    "context_modules_api": "modules",
    "context_modules": "modules",
    "wiki_pages": "pages",
    "wiki_pages_api": "pages",
    "gradebooks": "grades",
    "gradebook_history_api": "grades",
    "announcements": "announcements",
    "courses": "navigation",
    "tabs": "navigation",
    "enrollments_api": "navigation",
    "users": "navigation",
    "external_tools": "external_tools",
    "external_tools_api": "external_tools",
    "groups": "navigation",
}


def extract_resource_info(url: str) -> tuple[str | None, str | None]:
    """Extract (category, resource_id) from a Canvas URL.

    Returns (None, None) if no pattern matches — caller should use controller fallback.
    """
    if not isinstance(url, str):
        return None, None
    for pattern, category, group_idx in URL_PATTERNS:
        m = re.search(pattern, url)
        if m:
            rid = m.group(group_idx) if group_idx is not None else None
            return category, rid
    return None, None


def categorize_row(url: str, controller: str) -> tuple[str, str | None]:
    """Hybrid URL + controller categorization.

    1. Try URL patterns first (most reliable)
    2. Fall back to controller mapping
    3. Default to 'other'
    """
    cat_url, rid = extract_resource_info(url)
    if cat_url is not None:
        return cat_url, rid

    # Controller fallback
    cat_ctrl = CONTROLLER_CATEGORY_MAP.get(str(controller), "other")
    return cat_ctrl, None


def main() -> None:
    print("=" * 70)
    print("PUC Data Fix Pipeline")
    print("=" * 70)

    # ── 1. Load raw page views ─────────────────────────────────────────
    print("\n[1/7] Loading raw page views...")
    df_pv = pd.read_parquet(PV_FILE)
    print(f"  Loaded {len(df_pv):,} rows, {df_pv['student_id'].nunique()} students, "
          f"{df_pv['course_id'].nunique()} courses")

    # ── 2. Load grades ─────────────────────────────────────────────────
    print("\n[2/7] Loading grades...")
    df_grades = pd.read_csv(GRADES_FILE)
    df_grades = df_grades.rename(columns={
        "user_lms_id": "student_id",
        "course_lms_id": "course_id",
    })
    print(f"  Loaded {len(df_grades):,} grade records, {df_grades['course_id'].nunique()} courses")

    # Filter out NaN grades (withdrawals / incomplete)
    n_before = len(df_grades)
    df_grades = df_grades.dropna(subset=["grade"])
    print(f"  Dropped {n_before - len(df_grades)} rows with NaN grades -> {len(df_grades):,} remaining")

    # Compute failure labels
    df_grades["failed"] = (df_grades["grade"] < 4.0).astype(int)
    print(f"  Fail rate (< 4.0): {df_grades['failed'].mean():.1%} ({df_grades['failed'].sum()}/{len(df_grades)})")

    # ── 3. Filter PVs to courses with grades ───────────────────────────
    print("\n[3/7] Filtering page views to courses with grades...")
    grade_courses = set(df_grades["course_id"].unique())
    grade_students = set(df_grades["student_id"].unique())

    # Only keep PVs for (student, course) pairs that have grades
    df_pv = df_pv.merge(
        df_grades[["student_id", "course_id"]].drop_duplicates(),
        on=["student_id", "course_id"],
        how="inner",
    )
    print(f"  Filtered to {len(df_pv):,} rows, "
          f"{df_pv['student_id'].nunique()} students, "
          f"{df_pv['course_id'].nunique()} courses")

    # ── 4. Extract real resource IDs from URLs ─────────────────────────
    print("\n[4/7] Extracting resource IDs from URLs (vectorized)...")

    # Vectorized URL pattern matching
    url_series = df_pv["url"].fillna("")
    ctrl_series = df_pv["controller"].fillna("")

    # Try URL patterns first (vectorized with str.extract for each pattern)
    df_pv["category"] = None
    df_pv["resource_id"] = None

    # Apply URL patterns in order
    unmatched = df_pv["category"].isna()
    for pattern, category, group_idx in URL_PATTERNS:
        if not unmatched.any():
            break
        matches = url_series[unmatched].str.extract(f"({pattern})", expand=True)
        matched_mask = matches.iloc[:, 0].notna()
        if matched_mask.any():
            matched_idx = matched_mask[matched_mask].index
            df_pv.loc[matched_idx, "category"] = category
            if group_idx is not None:
                # The group is at column index group_idx (0-based in the full match, 1-based in original)
                df_pv.loc[matched_idx, "resource_id"] = matches.loc[matched_idx].iloc[:, group_idx].values
            unmatched = df_pv["category"].isna()

    # Controller fallback for unmatched rows
    n_url_matched = (~unmatched).sum()
    if unmatched.any():
        df_pv.loc[unmatched, "category"] = ctrl_series[unmatched].map(CONTROLLER_CATEGORY_MAP).fillna("other")

    n_ctrl_matched = ((~df_pv["category"].isin(["other", None])) & unmatched).sum()
    n_other = (df_pv["category"] == "other").sum()

    print(f"  URL-matched: {n_url_matched:,} ({n_url_matched/len(df_pv):.1%})")
    print(f"  Controller-matched: {(len(df_pv) - n_url_matched - n_other):,}")
    print(f"  Unmatched ('other'): {n_other:,} ({n_other/len(df_pv):.1%})")

    # ── 5. Report category distribution ────────────────────────────────
    print("\n[5/7] Category distribution:")
    cat_counts = df_pv["category"].value_counts()
    for cat, count in cat_counts.items():
        n_unique_res = df_pv.loc[df_pv["category"] == cat, "resource_id"].dropna().nunique()
        print(f"  {cat:20s}: {count:>10,} views, {n_unique_res:>6,} unique resources")

    # Report resource_id coverage
    has_rid = df_pv["resource_id"].notna().sum()
    print(f"\n  Resource ID coverage: {has_rid:,}/{len(df_pv):,} ({has_rid/len(df_pv):.1%})")

    # Compare with old synthetic approach
    old_synthetic = (df_pv["controller"].astype(str) + "_" + df_pv["action"].astype(str)).nunique()
    new_real = df_pv["resource_id"].dropna().nunique()
    print(f"  Old synthetic resource_ids: ~{old_synthetic}")
    print(f"  New real resource_ids: {new_real}")

    # ── 6. Compute temporal columns ────────────────────────────────────
    print("\n[6/7] Computing temporal columns...")
    df_pv["created_at"] = pd.to_datetime(df_pv["created_at"], utc=True)
    df_pv["hour"] = df_pv["created_at"].dt.hour
    df_pv["day_of_week"] = df_pv["created_at"].dt.dayofweek  # 0=Mon, 6=Sun

    # Course start dates (5th percentile of all activity)
    course_starts = df_pv.groupby("course_id")["created_at"].quantile(0.05)
    course_starts.name = "course_start"
    df_pv = df_pv.merge(course_starts, on="course_id", how="left")

    # Week number relative to course start
    df_pv["days_since_start"] = (
        df_pv["created_at"] - df_pv["course_start"]
    ).dt.total_seconds() / 86400
    df_pv["week_number"] = (df_pv["days_since_start"] / 7).astype(int) + 1

    # Filter out activity before course start (negative weeks)
    n_negative = (df_pv["week_number"] <= 0).sum()
    df_pv = df_pv[df_pv["week_number"] > 0].copy()
    print(f"  Dropped {n_negative:,} rows with week_number <= 0")
    print(f"  Week range: {df_pv['week_number'].min()} to {df_pv['week_number'].max()}")

    # ── 7. Merge grades and save ───────────────────────────────────────
    print("\n[7/7] Merging grades and saving...")

    # Keep grade info for later use (one row per student-course)
    grade_cols = ["student_id", "course_id", "grade", "failed", "Code", "Sigla"]
    grade_cols = [c for c in grade_cols if c in df_grades.columns]
    df_grade_info = df_grades[grade_cols].drop_duplicates(subset=["student_id", "course_id"])

    # Save page views with categories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_pv.to_parquet(OUTPUT_FILE, index=False)
    print(f"  Saved page views: {OUTPUT_FILE}")
    print(f"  Final shape: {df_pv.shape}")

    # Also save grade lookup separately
    grade_output = OUTPUT_DIR / "puc_grades_clean.parquet"
    df_grade_info.to_parquet(grade_output, index=False)
    print(f"  Saved grades: {grade_output}")

    # ── Final report ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    n_pairs = df_pv.groupby(["student_id", "course_id"]).ngroups
    print(f"  Total page views:          {len(df_pv):>12,}")
    print(f"  Unique students:           {df_pv['student_id'].nunique():>12,}")
    print(f"  Unique courses:            {df_pv['course_id'].nunique():>12,}")
    print(f"  (student, course) pairs:   {n_pairs:>12,}")
    print(f"  Unique real resource_ids:  {df_pv['resource_id'].dropna().nunique():>12,}")
    print(f"  Category 'other' pct:      {(df_pv['category'] == 'other').mean():>11.1%}")
    print(f"  Grades: {len(df_grade_info)} pairs, "
          f"fail rate {df_grade_info['failed'].mean():.1%}")

    # Per-category resource_id sanity check
    print("\n  Per-category unique resource counts:")
    for cat in ["files", "discussions", "quizzes", "assignments", "pages", "modules"]:
        n = df_pv.loc[df_pv["category"] == cat, "resource_id"].dropna().nunique()
        print(f"    {cat:15s}: {n:>6,}")


if __name__ == "__main__":
    main()
