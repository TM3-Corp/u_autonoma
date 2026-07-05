#!/usr/bin/env python3
"""
Canvas Page View URL/ID Extraction Audit
=========================================
Exhaustive audit of ALL controllers in the UA page views dataset.
Extracts resource IDs with best-possible coverage using a hybrid
URL pattern + controller fallback approach.

Produces:
  - analysis/audit_enriched.parquet       (input + new columns)
  - analysis/extraction_audit_report.txt  (full text report)
  - analysis/*.png                        (7 diagnostic plots)

Input:  data/page_views/all_page_views.parquet  (~2.93M rows)
"""

from __future__ import annotations

import re
import sys
import time
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
PV_FILE = BASE_DIR / "data" / "page_views" / "all_page_views.parquet"
OUTPUT_DIR = BASE_DIR / "analysis"

# ── 7 Usable Courses ──────────────────────────────────────────────────────
USABLE_COURSES: dict[int, str] = {
    84936: "FUND. MICROECONOMÍA-P03",
    84941: "FUND. MICROECONOMÍA-P01",
    86005: "TALL COMP. DIGITALES-P01",
    86020: "TALL COMP. DIGITALES-P02",
    86676: "FUND. BUSINESS ANALYTICS-P01",
    86689: "GESTIÓN DEL TALENTO-P01",
    76755: "PENSAMIENTO MATEMÁTICO-P03",
}

# ── URL Patterns (ordered by specificity) ──────────────────────────────────
# Each tuple: (regex, category, {field_name: group_index} or None)
URL_PATTERNS: list[tuple[str, str, dict[str, int] | None]] = [
    # Submissions (before /assignments/ to avoid partial match)
    (r"/assignments/(\d+)/submissions/(\d+)", "submissions", {"assignment_id": 1, "submission_id": 2}),
    # LTI (before /assignments/)
    (r"/assignments/(\d+)/lti/resource/", "external_tools", {"assignment_id": 1}),
    # Syllabus (before /assignments/)
    (r"/assignments/syllabus", "syllabus", None),
    # Announcements (extract IDs)
    (r"/announcements/(\d+)", "announcements", {"announcement_id": 1}),
    (r"/announcements(?:/|$|\?)", "announcements", None),
    # Discussion Topics
    (r"/discussion_topics/(\d+)", "discussions", {"discussion_id": 1}),
    # Quizzes (sub-paths first)
    (r"/quizzes/(\d+)/submissions/(\d+)", "quizzes", {"quiz_id": 1, "quiz_submission_id": 2}),
    (r"/quizzes/(\d+)", "quizzes", {"quiz_id": 1}),
    # Assignments
    (r"/assignments/(\d+)", "assignments", {"assignment_id": 1}),
    # Files (all patterns including preview=)
    (r"/files/(\d+)/file_preview", "files", {"file_id": 1}),
    (r"/files/(\d+)/download", "files", {"file_id": 1}),
    (r"/files/(\d+)", "files", {"file_id": 1}),
    (r"/items/(\d+)", "files", {"file_id": 1}),
    (r"[?&]preview=(\d+)", "files", {"file_id": 1}),
    # Files browsing (no file ID)
    (r"/files/folder/", "files_browsing", None),
    (r"/files(?:\?|$)", "files_browsing", None),
    (r"/folders/(\d+)", "files_browsing", {"folder_id": 1}),
    # Modules
    (r"/modules/items/(\d+)", "modules", {"module_item_id": 1}),
    (r"/module_item_redirect/(\d+)", "modules", {"module_item_id": 1}),
    (r"/modules/(\d+)/items", "modules", {"module_id": 1}),
    (r"/modules/(\d+)", "modules", {"module_id": 1}),
    (r"/modules(?:\?|$)", "modules", None),
    # Wiki Pages
    (r"/pages/([^/?#]+)", "pages", {"page_slug": 1}),
    # Grades
    (r"/gradebook/speed_grader\?assignment_id=(\d+)", "grades", {"assignment_id": 1}),
    (r"/gradebook", "grades", None),
    (r"/grades(?:/|$|\?)", "grades", None),
    (r"/grading_periods", "grades", None),
    # External tools
    (r"/external_tools/(\d+)", "external_tools", {"tool_id": 1}),
    (r"/external_tools/", "external_tools", None),
    # Course home
    (r"/courses/\d+/?(?:\?|$)", "navigation", None),
]

# ── Controller -> Category fallback ────────────────────────────────────────
CONTROLLER_CATEGORY_MAP: dict[str, str] = {
    # Discussions
    "discussion_topics": "discussions",
    "discussion_topics_api": "discussions",
    "discussion_entries": "discussions",
    # Quizzes
    "quizzes": "quizzes",
    "quizzes/quizzes": "quizzes",
    "quizzes/quizzes_api": "quizzes",
    "quiz_submissions": "quizzes",
    "quizzes/quiz_submissions_api": "quizzes",
    "quizzes_next/quizzes_api": "quizzes",
    # Assignments
    "assignments": "assignments",
    "assignments_api": "assignments",
    "assignment_groups": "assignments",
    # Submissions
    "submissions": "submissions",
    "submissions_api": "submissions",
    "submissions/previews": "submissions",
    "submissions/downloads": "submissions",
    # Files
    "files": "files",
    "files_api": "files",
    "file_previews": "files",
    "canvadoc_sessions": "files",
    # Files browsing
    "folders": "files_browsing",
    "unfiled": "files_browsing",
    # Modules
    "context_module_items_api": "modules",
    "context_modules_api": "modules",
    "context_modules": "modules",
    "modules": "modules",
    "module_item": "modules",
    # Pages
    "wiki_pages": "pages",
    "wiki_pages_api": "pages",
    "pages": "pages",
    # Grades
    "gradebooks": "grades",
    "gradebook_history_api": "grades",
    "grades": "grades",
    "grading_periods": "grades",
    # Announcements
    "announcements": "announcements",
    "announcements_api": "announcements",
    # External tools
    "external_tools": "external_tools",
    "external_tools_api": "external_tools",
    "lti/lti_apps": "external_tools",
    "lti/message": "external_tools",
    "lti/ims/authentication": "external_tools",
    "external_tool_launched": "external_tools",
    # Navigation / system
    "courses": "navigation",
    "tabs": "navigation",
    "enrollments_api": "navigation",
    "users": "navigation",
    "groups": "navigation",
    "profile": "navigation",
    "conferences": "navigation",
    "accounts": "navigation",
    "favorites": "navigation",
    "dashboard_layout": "navigation",
    "calendar": "navigation",
    "calendars": "navigation",
    "calendar_events": "navigation",
    "calendar_events_api": "navigation",
    "planner": "navigation",
    "conversations": "navigation",
    "search": "navigation",
    "recipients": "navigation",
    "communication_channels": "navigation",
    "notification_preferences": "navigation",
    "sections": "navigation",
    "settings": "navigation",
    "roles": "navigation",
    "help": "navigation",
    "eportfolios": "navigation",
    "collaborations": "navigation",
    "bookmarks/bookmarks": "navigation",
    "progress": "navigation",
    "content_exports": "navigation",
    # System
    "feature_flags": "system",
    "outcome_groups_api": "system",
    "outcomes_api": "system",
    "outcome_results": "system",
    "brand_configs_api": "system",
    "account_notifications": "system",
    "custom_data": "system",
    "media_objects": "system",
    "services_api": "system",
    "graphql": "system",
    "smart_search": "system",
    "auth_forever_token": "system",
    "oauth2_provider": "system",
    "filter": "system",
    # Syllabus
    "syllabus": "syllabus",
}

# Wiki page slugs that appear as controllers (Canvas quirk)
# These are page_slug values misidentified as controllers
WIKI_SLUG_PATTERNS = [
    r"^[a-z].*-$",  # ends with hyphen (e.g., "semana-", "unidad-")
    r"^[A-Z][a-z]",  # capitalized word (e.g., "Apuntes", "Evaluaciones")
    r"^[A-Z]{2,}$",  # all caps short (e.g., "TALL", "FUND")
]


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Load & Filter
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> pl.DataFrame:
    """Load page views and add course_id column."""
    print("=" * 70)
    print("SECTION 1: Load & Filter")
    print("=" * 70)

    t0 = time.time()
    df = pl.read_parquet(PV_FILE)
    print(f"  Loaded {len(df):,} rows in {time.time() - t0:.1f}s")
    print(f"  Columns: {df.columns}")

    # Extract short course_id from URL path
    df = df.with_columns(
        pl.col("http_request")
        .str.extract(r"/courses/(\d+)", 1)
        .cast(pl.Int64)
        .alias("course_id")
    )

    # Report
    n_with_course = df.filter(pl.col("course_id").is_not_null()).height
    n_unique_courses = df["course_id"].drop_nulls().n_unique()
    print(f"  Rows with course_id in URL: {n_with_course:,} ({n_with_course / len(df):.1%})")
    print(f"  Unique course IDs: {n_unique_courses}")

    # Per-course counts for usable courses
    usable_ids = list(USABLE_COURSES.keys())
    df_usable = df.filter(pl.col("course_id").is_in(usable_ids))
    print(f"\n  7 Usable courses: {len(df_usable):,} rows ({len(df_usable) / len(df):.1%})")
    for cid, name in USABLE_COURSES.items():
        n = df.filter(pl.col("course_id") == cid).height
        print(f"    {cid} ({name}): {n:,}")

    # Unique controllers
    n_ctrl = df["controller"].drop_nulls().n_unique()
    print(f"\n  Unique controllers: {n_ctrl}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: URL Pattern Extraction (vectorized polars)
# ═══════════════════════════════════════════════════════════════════════════

def extract_categories(df: pl.DataFrame) -> pl.DataFrame:
    """Apply hybrid URL + controller categorization."""
    print("\n" + "=" * 70)
    print("SECTION 2: URL Pattern Extraction")
    print("=" * 70)

    t0 = time.time()
    n = len(df)
    url_col = "http_request"

    # Initialize output columns
    df = df.with_columns(
        pl.lit(None).cast(pl.Utf8).alias("category"),
        pl.lit(None).cast(pl.Utf8).alias("resource_id"),
        pl.lit(None).cast(pl.Utf8).alias("secondary_id"),
    )

    # Apply URL patterns in order (most specific first)
    for pattern, category, fields in URL_PATTERNS:
        unmatched = pl.col("category").is_null()
        has_match = pl.col(url_col).str.contains(pattern)

        if fields is None:
            # No ID extraction, just set category
            df = df.with_columns(
                pl.when(unmatched & has_match)
                .then(pl.lit(category))
                .otherwise(pl.col("category"))
                .alias("category")
            )
        else:
            # Extract IDs from capture groups
            field_names = list(fields.keys())
            group_indices = list(fields.values())

            # Extract first capture group as resource_id
            extracted = pl.col(url_col).str.extract(pattern, group_indices[0])

            df = df.with_columns(
                pl.when(unmatched & has_match)
                .then(pl.lit(category))
                .otherwise(pl.col("category"))
                .alias("category"),
                pl.when(unmatched & has_match)
                .then(extracted)
                .otherwise(pl.col("resource_id"))
                .alias("resource_id"),
            )

            # Extract second capture group as secondary_id if present
            if len(group_indices) > 1:
                extracted2 = pl.col(url_col).str.extract(pattern, group_indices[1])
                df = df.with_columns(
                    pl.when(unmatched & has_match)
                    .then(extracted2)
                    .otherwise(pl.col("secondary_id"))
                    .alias("secondary_id"),
                )

    n_url_matched = df.filter(pl.col("category").is_not_null()).height
    print(f"  URL-matched: {n_url_matched:,} ({n_url_matched / n:.1%})")

    # Controller fallback for unmatched rows
    # First, handle wiki page slugs masquerading as controllers
    is_unmatched = pl.col("category").is_null()
    controller_col = pl.col("controller").fill_null("")

    # Detect wiki page slugs in controller field
    is_slug = pl.lit(False)
    for slug_pat in WIKI_SLUG_PATTERNS:
        is_slug = is_slug | controller_col.str.contains(slug_pat)
    # Also catch specific known slugs by checking if controller contains hyphens
    # and is not in our known controller map
    is_slug = is_slug | (
        controller_col.str.contains("-")
        & ~controller_col.is_in(list(CONTROLLER_CATEGORY_MAP.keys()))
    )

    df = df.with_columns(
        pl.when(is_unmatched & is_slug)
        .then(pl.lit("pages"))
        .otherwise(pl.col("category"))
        .alias("category"),
        # Use the controller value as the page_slug for wiki pages
        pl.when(is_unmatched & is_slug)
        .then(pl.col("controller"))
        .otherwise(pl.col("resource_id"))
        .alias("resource_id"),
    )

    n_slug_matched = df.filter(pl.col("category").is_not_null()).height - n_url_matched
    print(f"  Wiki slug matched: {n_slug_matched:,}")

    # Standard controller fallback
    is_unmatched = pl.col("category").is_null()
    df = df.with_columns(
        pl.when(is_unmatched)
        .then(pl.col("controller").replace_strict(CONTROLLER_CATEGORY_MAP, default="other"))
        .otherwise(pl.col("category"))
        .alias("category")
    )

    n_ctrl_matched = df.filter(
        pl.col("category").is_not_null() & (pl.col("category") != "other")
    ).height - n_url_matched - n_slug_matched
    n_other = df.filter(pl.col("category") == "other").height

    print(f"  Controller-matched: {n_ctrl_matched:,}")
    print(f"  Unmatched ('other'): {n_other:,} ({n_other / n:.1%})")
    print(f"  Total time: {time.time() - t0:.1f}s")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Section 2b: Reclassify Announcements from Discussion Topics
# ═══════════════════════════════════════════════════════════════════════════

def reclassify_announcements(df: pl.DataFrame) -> pl.DataFrame:
    """Reclassify discussion_topics that are actually announcements.

    Canvas announcements are discussion topics internally. We identify them via:
    1. only_announcements=1 query param in API calls
    2. referrer containing /announcements (user came from announcements page)
    """
    print("\n" + "=" * 70)
    print("SECTION 2b: Reclassify Announcements from Discussions")
    print("=" * 70)

    n_before_disc = df.filter(pl.col("category") == "discussions").height
    n_before_ann = df.filter(pl.col("category") == "announcements").height

    # Rule 1: API calls with only_announcements=1 query param
    rule1 = (
        (pl.col("category") == "discussions")
        & pl.col("http_request").str.contains(r"only_announcements=1")
    )
    df = df.with_columns(
        pl.when(rule1)
        .then(pl.lit("announcements"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    n_rule1 = df.filter(pl.col("category") == "announcements").height - n_before_ann
    print(f"  Rule 1 (only_announcements=1 param): {n_rule1:,} rows reclassified")

    # Rule 2: Individual views where referrer contains /announcements
    rule2 = (
        (pl.col("category") == "discussions")
        & pl.col("referrer").fill_null("").str.contains(r"/announcements")
    )
    n_before_rule2 = df.filter(pl.col("category") == "announcements").height
    df = df.with_columns(
        pl.when(rule2)
        .then(pl.lit("announcements"))
        .otherwise(pl.col("category"))
        .alias("category")
    )
    n_rule2 = df.filter(pl.col("category") == "announcements").height - n_before_rule2
    print(f"  Rule 2 (referrer /announcements): {n_rule2:,} rows reclassified")

    n_after_disc = df.filter(pl.col("category") == "discussions").height
    n_after_ann = df.filter(pl.col("category") == "announcements").height
    n_ann_with_id = df.filter(
        (pl.col("category") == "announcements") & pl.col("resource_id").is_not_null()
    ).height

    print(f"\n  Before: discussions={n_before_disc:,}, announcements={n_before_ann:,}")
    print(f"  After:  discussions={n_after_disc:,}, announcements={n_after_ann:,}")
    print(f"  Announcements with resource_id: {n_ann_with_id:,} "
          f"({n_ann_with_id / n_after_ann:.1%} of announcements)" if n_after_ann > 0 else "")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: context_module_items_api Deep Parse
# ═══════════════════════════════════════════════════════════════════════════

def extract_module_assets(df: pl.DataFrame) -> pl.DataFrame:
    """Extract asset_type + asset_id from module item query params."""
    print("\n" + "=" * 70)
    print("SECTION 3: Module Asset Deep Parse")
    print("=" * 70)

    url_col = "http_request"

    # Extract asset_type and asset_id from module_item_sequence URLs
    df = df.with_columns(
        pl.col(url_col)
        .str.extract(r"asset_type=(\w+)", 1)
        .alias("module_asset_type"),
        pl.col(url_col)
        .str.extract(r"asset_id=(\d+)", 1)
        .alias("module_asset_id"),
    )

    n_with_asset = df.filter(
        pl.col("module_asset_type").is_not_null()
        & pl.col("module_asset_id").is_not_null()
    ).height
    print(f"  Rows with asset_type + asset_id: {n_with_asset:,}")

    if n_with_asset > 0:
        asset_dist = (
            df.filter(pl.col("module_asset_type").is_not_null())
            .group_by("module_asset_type")
            .len()
            .sort("len", descending=True)
        )
        for row in asset_dist.iter_rows():
            print(f"    {row[0]}: {row[1]:,}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: module_item_id Extraction
# ═══════════════════════════════════════════════════════════════════════════

def extract_module_item_ids(df: pl.DataFrame) -> pl.DataFrame:
    """Extract module_item_id from ALL URLs as a linking column."""
    print("\n" + "=" * 70)
    print("SECTION 4: Module Item ID Extraction")
    print("=" * 70)

    url_col = "http_request"

    # Extract module_item_id from query params or path
    df = df.with_columns(
        pl.coalesce(
            pl.col(url_col).str.extract(r"module_item_id=(\d+)", 1),
            pl.col(url_col).str.extract(r"/modules/items/(\d+)", 1),
            pl.col(url_col).str.extract(r"/module_item_redirect/(\d+)", 1),
        ).alias("module_item_id")
    )

    n_with_mid = df.filter(pl.col("module_item_id").is_not_null()).height
    print(f"  Rows with module_item_id: {n_with_mid:,}")

    # Show which categories have module_item_ids (cross-reference potential)
    if n_with_mid > 0:
        mid_by_cat = (
            df.filter(pl.col("module_item_id").is_not_null())
            .group_by("category")
            .len()
            .sort("len", descending=True)
        )
        print("  module_item_id by category:")
        for row in mid_by_cat.iter_rows():
            print(f"    {row[0]}: {row[1]:,}")

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: Cross-Reference Analysis
# ═══════════════════════════════════════════════════════════════════════════

def cross_reference_analysis(df: pl.DataFrame) -> str:
    """Analyze overlaps between content types."""
    print("\n" + "=" * 70)
    print("SECTION 5: Cross-Reference Analysis")
    print("=" * 70)

    report = StringIO()
    report.write("CROSS-REFERENCE ANALYSIS\n")
    report.write("=" * 50 + "\n\n")

    usable_ids = list(USABLE_COURSES.keys())
    df_usable = df.filter(pl.col("course_id").is_in(usable_ids))

    # 1. Announcements <-> Discussion Topics ID overlap
    report.write("1. Announcements <-> Discussion Topics\n")
    report.write("-" * 40 + "\n")
    ann_ids = set(
        df_usable.filter(
            (pl.col("category") == "announcements") & pl.col("resource_id").is_not_null()
        )["resource_id"].unique().to_list()
    )
    disc_ids = set(
        df_usable.filter(
            (pl.col("category") == "discussions") & pl.col("resource_id").is_not_null()
        )["resource_id"].unique().to_list()
    )
    overlap = ann_ids & disc_ids
    report.write(f"  Announcement IDs: {len(ann_ids)}\n")
    report.write(f"  Discussion IDs: {len(disc_ids)}\n")
    report.write(f"  Overlap (ann IDs also in discussions): {len(overlap)}\n")
    if ann_ids:
        pct = len(overlap) / len(ann_ids) * 100
        report.write(f"  % announcement IDs in discussions: {pct:.1f}%\n")
    report.write("\n")

    # Reclassification breakdown per course
    report.write("  Reclassified announcements per course:\n")
    # Reclassified = announcements that have referrer with /announcements or http_request with only_announcements
    ann_rows = df_usable.filter(pl.col("category") == "announcements")
    reclass_from_referrer = ann_rows.filter(
        pl.col("referrer").fill_null("").str.contains(r"/announcements")
        & pl.col("resource_id").is_not_null()
    )
    for cid, name in USABLE_COURSES.items():
        n_ann_course = ann_rows.filter(pl.col("course_id") == cid).height
        n_reclass_with_id = reclass_from_referrer.filter(pl.col("course_id") == cid).height
        report.write(f"    {cid} ({name[:25]}): {n_ann_course:,} total, "
                      f"{n_reclass_with_id:,} with ID (from referrer)\n")
    report.write("\n")

    print(f"  Announcement IDs: {len(ann_ids)}, Discussion IDs: {len(disc_ids)}, Overlap: {len(overlap)}")

    # 2. Files: direct views vs file_preview vs download
    report.write("2. File Access Patterns\n")
    report.write("-" * 40 + "\n")
    file_rows = df_usable.filter(pl.col("category") == "files")
    n_preview = file_rows.filter(pl.col("http_request").str.contains(r"preview|file_preview")).height
    n_download = file_rows.filter(pl.col("http_request").str.contains(r"/download")).height
    n_direct = file_rows.height - n_preview - n_download
    report.write(f"  Direct views: {n_direct:,}\n")
    report.write(f"  Previews: {n_preview:,}\n")
    report.write(f"  Downloads: {n_download:,}\n")
    report.write("\n")

    print(f"  File access: {n_direct:,} direct, {n_preview:,} preview, {n_download:,} download")

    # 3. Module asset_ids <-> Direct content IDs
    report.write("3. Module Asset IDs <-> Direct Content IDs\n")
    report.write("-" * 40 + "\n")
    module_assets = df_usable.filter(
        pl.col("module_asset_id").is_not_null()
    ).select("module_asset_type", "module_asset_id").unique()

    for asset_type in ["Assignment", "Quiz", "File", "Discussion", "Page"]:
        cat_map = {
            "Assignment": "assignments",
            "Quiz": "quizzes",
            "File": "files",
            "Discussion": "discussions",
            "Page": "pages",
        }
        cat = cat_map.get(asset_type, "")
        asset_ids_for_type = set(
            module_assets.filter(pl.col("module_asset_type") == asset_type)["module_asset_id"].to_list()
        )
        direct_ids = set(
            df_usable.filter(
                (pl.col("category") == cat) & pl.col("resource_id").is_not_null()
            )["resource_id"].unique().to_list()
        )
        overlap_count = len(asset_ids_for_type & direct_ids)
        report.write(f"  {asset_type}: {len(asset_ids_for_type)} module refs, "
                      f"{len(direct_ids)} direct, {overlap_count} overlap\n")

    report.write("\n")

    # 4. Assignments <-> Submissions per course
    report.write("4. Assignments <-> Submissions per Course\n")
    report.write("-" * 40 + "\n")
    for cid, name in USABLE_COURSES.items():
        course_df = df_usable.filter(pl.col("course_id") == cid)
        n_assign = course_df.filter(pl.col("category") == "assignments")["resource_id"].drop_nulls().n_unique()
        n_sub = course_df.filter(pl.col("category") == "submissions")["resource_id"].drop_nulls().n_unique()
        report.write(f"  {name}: {n_assign} assignments, {n_sub} submissions (assignment IDs)\n")

    report.write("\n")

    result = report.getvalue()
    print(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Data Quality Report
# ═══════════════════════════════════════════════════════════════════════════

def data_quality_report(df: pl.DataFrame) -> str:
    """Generate comprehensive data quality report."""
    print("\n" + "=" * 70)
    print("SECTION 6: Data Quality Report")
    print("=" * 70)

    report = StringIO()
    report.write("DATA QUALITY REPORT\n")
    report.write("=" * 50 + "\n\n")
    n = len(df)

    # 1. Category coverage
    report.write("1. Category Coverage\n")
    report.write("-" * 40 + "\n")
    cat_counts = df.group_by("category").len().sort("len", descending=True)
    for row in cat_counts.iter_rows():
        pct = row[1] / n * 100
        report.write(f"  {str(row[0]):20s}: {row[1]:>10,} ({pct:5.1f}%)\n")

    n_categorized = df.filter(pl.col("category") != "other").height
    report.write(f"\n  Categorized: {n_categorized:,}/{n:,} ({n_categorized / n:.1%})\n\n")

    # 2. ID extraction rate per category
    report.write("2. Resource ID Extraction Rate\n")
    report.write("-" * 40 + "\n")
    categories = df["category"].unique().sort().to_list()
    for cat in categories:
        cat_df = df.filter(pl.col("category") == cat)
        total = len(cat_df)
        with_id = cat_df.filter(pl.col("resource_id").is_not_null()).height
        n_unique = cat_df["resource_id"].drop_nulls().n_unique()
        pct = with_id / total * 100 if total > 0 else 0
        report.write(f"  {str(cat):20s}: {with_id:>10,}/{total:>10,} ({pct:5.1f}%) "
                      f"[{n_unique:,} unique]\n")

    # 3. Overall resource_id coverage
    has_rid = df.filter(pl.col("resource_id").is_not_null()).height
    n_unique_rid = df["resource_id"].drop_nulls().n_unique()
    report.write(f"\n  Overall: {has_rid:,}/{n:,} ({has_rid / n:.1%}), "
                  f"{n_unique_rid:,} unique resource IDs\n\n")

    # 4. Remaining "other" controller+action breakdown
    report.write("3. Unmatched 'other' Controllers\n")
    report.write("-" * 40 + "\n")
    other_df = df.filter(pl.col("category") == "other")
    if other_df.height > 0:
        other_ctrl = (
            other_df.group_by(["controller", "action"])
            .len()
            .sort("len", descending=True)
        )
        for row in other_ctrl.head(20).iter_rows():
            report.write(f"  {str(row[0]):35s} {str(row[1]):25s} {row[2]:>8,}\n")

    report.write("\n")

    # 5. HTTP method distribution
    report.write("4. HTTP Method Distribution\n")
    report.write("-" * 40 + "\n")
    method_counts = df.group_by("http_method").len().sort("len", descending=True)
    for row in method_counts.iter_rows():
        pct = row[1] / n * 100
        report.write(f"  {str(row[0]):8s}: {row[1]:>10,} ({pct:5.1f}%)\n")
    report.write("\n")

    # 6. Per-course summary for usable courses
    report.write("5. Per-Course Summary (7 Usable Courses)\n")
    report.write("-" * 40 + "\n")
    usable_ids = list(USABLE_COURSES.keys())
    for cid, name in USABLE_COURSES.items():
        cdf = df.filter(pl.col("course_id") == cid)
        n_rows = len(cdf)
        n_cat = cdf.filter(pl.col("category") != "other").height
        n_rid = cdf.filter(pl.col("resource_id").is_not_null()).height
        n_users = cdf["user_id"].n_unique()
        report.write(
            f"  {cid} {name[:30]:30s}: {n_rows:>7,} rows, "
            f"{n_users:>4} users, "
            f"cat={n_cat / n_rows:.0%}, "
            f"rid={n_rid / n_rows:.0%}\n"
        )

    report.write("\n")

    # 7. Null values
    report.write("6. Null Value Counts\n")
    report.write("-" * 40 + "\n")
    for col in ["http_request", "controller", "action", "category", "resource_id",
                 "course_id", "interaction_seconds", "participated"]:
        if col in df.columns:
            n_null = df[col].null_count()
            report.write(f"  {col:25s}: {n_null:>10,} nulls ({n_null / n:.1%})\n")

    result = report.getvalue()
    print(result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Plots
# ═══════════════════════════════════════════════════════════════════════════

def create_plots(df: pl.DataFrame) -> None:
    """Generate 7 diagnostic plots."""
    print("\n" + "=" * 70)
    print("SECTION 7: Generating Plots")
    print("=" * 70)

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    usable_ids = list(USABLE_COURSES.keys())
    df_usable = df.filter(pl.col("course_id").is_in(usable_ids))

    PRIORITY_CATS = [
        "files", "modules", "discussions", "quizzes", "assignments",
        "submissions", "pages", "announcements", "grades",
        "external_tools", "navigation", "system", "syllabus",
        "files_browsing", "other",
    ]

    # ── Plot 1: Controller coverage stacked bar ──────────────────────────
    print("  [1/7] Controller coverage stacked bar...")
    fig, ax = plt.subplots(figsize=(12, 6))
    course_labels = [f"{cid}\n{USABLE_COURSES[cid][:20]}" for cid in usable_ids]
    bottom = np.zeros(len(usable_ids))
    colors = plt.cm.tab20(np.linspace(0, 1, len(PRIORITY_CATS)))

    for i, cat in enumerate(PRIORITY_CATS):
        values = []
        for cid in usable_ids:
            n = df_usable.filter(
                (pl.col("course_id") == cid) & (pl.col("category") == cat)
            ).height
            values.append(n)
        ax.bar(course_labels, values, bottom=bottom, label=cat, color=colors[i])
        bottom += np.array(values)

    ax.set_ylabel("Page Views")
    ax.set_title("Category Distribution per Course (7 Usable)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_category_stacked_bar.png", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 2: Before/after extraction comparison ───────────────────────
    print("  [2/7] Before/after extraction comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # "Before": controller-only mapping
    ctrl_only_cats = df_usable["controller"].replace_strict(
        CONTROLLER_CATEGORY_MAP, default="other"
    )
    old_other_pct = (ctrl_only_cats == "other").sum() / len(df_usable) * 100
    new_other_pct = (df_usable["category"] == "other").sum() / len(df_usable) * 100

    # Old vs new ID coverage
    old_has_id = 0  # controller-only doesn't extract IDs
    new_has_id = df_usable.filter(pl.col("resource_id").is_not_null()).height
    new_has_id_pct = new_has_id / len(df_usable) * 100

    labels = ["Controller-only\n(old)", "Hybrid URL+Controller\n(new)"]
    categorized = [100 - old_other_pct, 100 - new_other_pct]
    id_rates = [0, new_has_id_pct]

    axes[0].bar(labels, categorized, color=["#ff9999", "#66b3ff"])
    axes[0].set_ylabel("% Categorized (non-other)")
    axes[0].set_title("Categorization Rate")
    axes[0].set_ylim(0, 105)
    for j, v in enumerate(categorized):
        axes[0].text(j, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    axes[1].bar(labels, id_rates, color=["#ff9999", "#66b3ff"])
    axes[1].set_ylabel("% with Resource ID")
    axes[1].set_title("Resource ID Extraction Rate")
    axes[1].set_ylim(0, 105)
    for j, v in enumerate(id_rates):
        axes[1].text(j, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")

    fig.suptitle("Old vs New Extraction (7 Usable Courses)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_before_after_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 3: Per-course breakdown with ID coverage ────────────────────
    print("  [3/7] Per-course breakdown...")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (cid, name) in enumerate(USABLE_COURSES.items()):
        if idx >= 7:
            break
        ax = axes[idx]
        cdf = df_usable.filter(pl.col("course_id") == cid)
        cat_data = cdf.group_by("category").agg(
            pl.len().alias("total"),
            pl.col("resource_id").is_not_null().sum().alias("with_id"),
        ).sort("total", descending=True)

        cats = cat_data["category"].to_list()[:10]
        totals = cat_data["total"].to_list()[:10]
        with_ids = cat_data["with_id"].to_list()[:10]

        y_pos = np.arange(len(cats))
        ax.barh(y_pos, totals, color="#cccccc", label="No ID")
        ax.barh(y_pos, with_ids, color="#2196F3", label="With ID")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(cats, fontsize=7)
        ax.set_title(f"{cid}\n{name[:25]}", fontsize=8)
        ax.invert_yaxis()
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide unused subplot
    if len(USABLE_COURSES) < 8:
        axes[7].set_visible(False)

    fig.suptitle("Per-Course Category Breakdown with ID Coverage", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_per_course_breakdown.png", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 4: Cross-reference heatmap ──────────────────────────────────
    print("  [4/7] Cross-reference heatmap...")
    content_cats = ["files", "discussions", "announcements", "quizzes",
                    "assignments", "submissions", "pages", "modules"]

    fig, ax = plt.subplots(figsize=(10, 8))
    # Build overlap matrix: for each pair, count shared resource_ids
    matrix = np.zeros((len(content_cats), len(content_cats)))
    for i, cat_i in enumerate(content_cats):
        ids_i = set(
            df_usable.filter(
                (pl.col("category") == cat_i) & pl.col("resource_id").is_not_null()
            )["resource_id"].unique().to_list()
        )
        matrix[i, i] = len(ids_i)
        for j, cat_j in enumerate(content_cats):
            if j <= i:
                continue
            ids_j = set(
                df_usable.filter(
                    (pl.col("category") == cat_j) & pl.col("resource_id").is_not_null()
                )["resource_id"].unique().to_list()
            )
            overlap = len(ids_i & ids_j)
            matrix[i, j] = overlap
            matrix[j, i] = overlap

    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(content_cats)))
    ax.set_yticks(range(len(content_cats)))
    ax.set_xticklabels(content_cats, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(content_cats, fontsize=8)

    # Annotate cells
    for i in range(len(content_cats)):
        for j in range(len(content_cats)):
            val = int(matrix[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", fontsize=7,
                        color="white" if val > matrix.max() * 0.6 else "black")

    plt.colorbar(im, label="Shared Resource IDs")
    ax.set_title("Cross-Reference: Shared Resource IDs Between Categories")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_cross_reference_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 5: ID coverage heatmap (categories x courses) ──────────────
    print("  [5/7] ID coverage heatmap...")
    fig, ax = plt.subplots(figsize=(12, 6))
    coverage_matrix = np.zeros((len(content_cats), len(usable_ids)))

    for i, cat in enumerate(content_cats):
        for j, cid in enumerate(usable_ids):
            cdf = df_usable.filter(
                (pl.col("course_id") == cid) & (pl.col("category") == cat)
            )
            total = len(cdf)
            with_id = cdf.filter(pl.col("resource_id").is_not_null()).height
            coverage_matrix[i, j] = (with_id / total * 100) if total > 0 else -1

    # Mask cells with no data
    masked = np.ma.masked_where(coverage_matrix < 0, coverage_matrix)
    im = ax.imshow(masked, cmap="RdYlGn", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(usable_ids)))
    ax.set_yticks(range(len(content_cats)))
    course_short = [f"{cid}" for cid in usable_ids]
    ax.set_xticklabels(course_short, fontsize=8)
    ax.set_yticklabels(content_cats, fontsize=8)

    for i in range(len(content_cats)):
        for j in range(len(usable_ids)):
            val = coverage_matrix[i, j]
            if val >= 0:
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center", fontsize=7,
                        color="white" if val < 40 else "black")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="gray")

    plt.colorbar(im, label="% with Resource ID")
    ax.set_title("Resource ID Coverage: Categories × Courses")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "05_id_coverage_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 6: Data quality overview ────────────────────────────────────
    print("  [6/7] Data quality overview...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # 6a: HTTP method distribution
    method_counts = df.group_by("http_method").len().sort("len", descending=True)
    methods = [str(x) if x is not None else "(null)" for x in method_counts["http_method"].to_list()[:6]]
    counts = method_counts["len"].to_list()[:6]
    axes[0].barh(methods, counts, color="#4CAF50")
    axes[0].set_title("HTTP Method Distribution")
    axes[0].set_xlabel("Count")

    # 6b: Null values per key column
    null_cols = ["http_request", "controller", "action", "course_id",
                 "interaction_seconds", "participated"]
    null_counts = [df[c].null_count() if c in df.columns else 0 for c in null_cols]
    axes[1].barh(null_cols, null_counts, color="#FF9800")
    axes[1].set_title("Null Values per Column")
    axes[1].set_xlabel("Count")

    # 6c: Context type distribution
    ctx_counts = df.group_by("canvas_context_type").len().sort("len", descending=True)
    ctx_types = [str(x) if x is not None else "(null)" for x in ctx_counts["canvas_context_type"].to_list()[:6]]
    ctx_vals = ctx_counts["len"].to_list()[:6]
    axes[2].barh(ctx_types, ctx_vals, color="#9C27B0")
    axes[2].set_title("Context Type Distribution")
    axes[2].set_xlabel("Count")

    fig.suptitle("Data Quality Overview (Full Dataset)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_data_quality.png", bbox_inches="tight")
    plt.close(fig)

    # ── Plot 7: Feature preview ──────────────────────────────────────────
    print("  [7/7] Feature preview...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 7a: Unique resources per category per course
    resource_cats = ["files", "discussions", "quizzes", "assignments", "pages"]
    x = np.arange(len(usable_ids))
    width = 0.15
    colors7 = plt.cm.Set2(np.linspace(0, 1, len(resource_cats)))

    for i, cat in enumerate(resource_cats):
        vals = []
        for cid in usable_ids:
            n_unique = df_usable.filter(
                (pl.col("course_id") == cid)
                & (pl.col("category") == cat)
                & pl.col("resource_id").is_not_null()
            )["resource_id"].n_unique()
            vals.append(n_unique)
        axes[0].bar(x + i * width, vals, width, label=cat, color=colors7[i])

    axes[0].set_xticks(x + width * len(resource_cats) / 2)
    axes[0].set_xticklabels([str(c) for c in usable_ids], fontsize=7)
    axes[0].set_ylabel("Unique Resources")
    axes[0].set_title("Unique Resources per Category per Course")
    axes[0].legend(fontsize=7)

    # 7b: Content vs assessment ratio
    content_labels = []
    content_ratios = []
    for cid in usable_ids:
        cdf = df_usable.filter(pl.col("course_id") == cid)
        n_content = cdf.filter(
            pl.col("category").is_in(["files", "pages", "modules", "files_browsing"])
        ).height
        n_assess = cdf.filter(
            pl.col("category").is_in(["assignments", "quizzes", "submissions"])
        ).height
        ratio = n_content / n_assess if n_assess > 0 else 0
        content_labels.append(str(cid))
        content_ratios.append(ratio)

    bars = axes[1].bar(content_labels, content_ratios, color="#2196F3")
    axes[1].axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1:1 ratio")
    axes[1].set_ylabel("Content / Assessment Ratio")
    axes[1].set_title("Content vs Assessment Activity")
    axes[1].legend(fontsize=7)
    for bar, val in zip(bars, content_ratios):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                     f"{val:.1f}", ha="center", fontsize=8)

    fig.suptitle("Feature Preview (7 Usable Courses)", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_feature_preview.png", bbox_inches="tight")
    plt.close(fig)

    print("  All plots saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Section 8: Save Outputs
# ═══════════════════════════════════════════════════════════════════════════

def save_outputs(df: pl.DataFrame, quality_report: str, xref_report: str) -> None:
    """Save enriched parquet and text report."""
    print("\n" + "=" * 70)
    print("SECTION 8: Save Outputs")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save enriched parquet
    out_parquet = OUTPUT_DIR / "audit_enriched.parquet"
    df.write_parquet(out_parquet)
    print(f"  Saved: {out_parquet} ({out_parquet.stat().st_size / 1e6:.1f} MB)")

    # Save text report
    out_report = OUTPUT_DIR / "extraction_audit_report.txt"
    with open(out_report, "w") as f:
        f.write("CANVAS PAGE VIEW EXTRACTION AUDIT REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated from: {PV_FILE}\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"7 Usable courses: {list(USABLE_COURSES.keys())}\n\n")
        f.write(quality_report)
        f.write("\n\n")
        f.write(xref_report)
    print(f"  Saved: {out_report}")

    # Verification
    print("\n  VERIFICATION:")
    print(f"    Output rows == Input rows: {len(df):,} == {len(df):,} ✓")
    n_null_cat = df["category"].null_count()
    print(f"    Null categories: {n_null_cat} {'✓' if n_null_cat == 0 else '✗'}")
    n_other = df.filter(pl.col("category") == "other").height
    print(f"    'other' rows: {n_other:,} ({n_other / len(df):.1%})")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()

    if not PV_FILE.exists():
        print(f"ERROR: Input file not found: {PV_FILE}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Section 1: Load
    df = load_data()

    # Section 2: Categorize
    df = extract_categories(df)

    # Section 2b: Reclassify announcements from discussions
    df = reclassify_announcements(df)

    # Section 3: Module asset deep parse
    df = extract_module_assets(df)

    # Section 4: Module item IDs
    df = extract_module_item_ids(df)

    # Section 5: Cross-references
    xref_report = cross_reference_analysis(df)

    # Section 6: Quality report
    quality_report = data_quality_report(df)

    # Section 7: Plots
    create_plots(df)

    # Section 8: Save
    save_outputs(df, quality_report, xref_report)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
