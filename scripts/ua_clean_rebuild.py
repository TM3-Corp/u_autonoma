#!/usr/bin/env python3
"""G1 (Tier-3) — UA clickstream hygiene: apply the Tier-1 PUC recipe to UA.

Input : data/page_views/categorized_page_views.parquet (raw UA page views)
Output: data/ua_clean/ua_clean_data.parquet
        data/puc/sota_results/tier3_pooled/ua_cleaning_report.json

Recipe (ported from scripts/puc_clean_rebuild.py, adapted to UA columns):
  - Filter to the 10 model courses; normalize user_id (raw page-view ids are the
    big form `1551...` → `% 1e10` to match student_enrollments.csv, exactly as
    ua_remediate_labels.normalize_user_id).
  - UA has no `url`; use `http_request` (the request path) as the URL surface.
  - created_at is a naive string → parse as UTC (Canvas timestamps are UTC).
  - L1 exact duplicate rows.
  - L2 HTML/api twin: probe for `/api/v1` twins in http_request first. Present
    (~57k) → applicable; dedup on (user,course,normalized_url,created_at@1s).
  - L3 rapid same-URL (<10s) debounce within (user,course), keep first of run.
  - Timezone → America/Santiago local columns (hour_local/dow_local); keep UTC.

Idempotent by construction (re-running on the output removes 0 rows). Every drop
counted. Per-row UTC−local offset must be exactly 3 or 4h for 100% of rows.
"""
import argparse, json, re
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO / "data/page_views/categorized_page_views.parquet"
DEFAULT_OUT = REPO / "data/ua_clean/ua_clean_data.parquet"
DEFAULT_REPORT = REPO / "data/puc/sota_results/tier3_pooled/ua_cleaning_report.json"

MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]
DEBOUNCE_SECONDS = 10

# strip a leading /api/v1 that sits right after an optional scheme://host
_API_RE = re.compile(r'^(https?://[^/]+)?/api/v1(?=/|\?|$)')


def normalize_user_id(uid):
    return uid % 10_000_000_000 if uid > 10_000_000_000 else uid


def normalize_url(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.replace(_API_RE, lambda m: m.group(1) or '', regex=True)


def load_raw(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["course_id"].notna()].copy()
    df["course_id"] = df["course_id"].astype("int64")
    df = df[df["course_id"].isin(MODEL_COURSES)].copy()
    df["user_id"] = df["user_id"].map(normalize_user_id).astype("int64")
    # created_at: naive string → UTC tz-aware
    if not pd.api.types.is_datetime64_any_dtype(df["created_at"]):
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    elif df["created_at"].dt.tz is None:
        df["created_at"] = df["created_at"].dt.tz_localize("UTC")
    # URL surface = http_request
    df["url"] = df["http_request"].fillna("").astype(str)
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    n0 = len(df)
    report["rows_input"] = n0
    report["model_courses"] = MODEL_COURSES

    # probe: are there /api/v1 twins in the URL surface? (decides L2 applicability)
    api_hits = int(df["url"].str.contains("/api/v1", na=False).sum())
    l2_applicable = api_hits > 0
    report["L2_probe"] = {"api_v1_url_hits": api_hits, "applicable": l2_applicable}

    # L1 — exact duplicate rows across all columns
    df1 = df.drop_duplicates()
    n1 = len(df1)
    report["L1_exact_duplicates"] = {"removed": n0 - n1, "remaining": n1}

    df1 = df1.copy()
    df1["normalized_url"] = normalize_url(df1["url"])

    # L2 — HTML+API twin: dedup on (user, course, normalized_url, created_at@1s)
    if l2_applicable:
        ts_1s = df1["created_at"].dt.floor("s")
        dup = pd.DataFrame({
            "u": df1["user_id"], "c": df1["course_id"],
            "n": df1["normalized_url"], "t": ts_1s,
        }).duplicated(keep="first")
        df2 = df1.loc[~dup].copy()
    else:
        report["L2_note"] = "L2 not applicable (no /api/v1 twins) — skipped"
        df2 = df1.copy()
    n2 = len(df2)
    report["L2_html_api_twin"] = {"removed": n1 - n2, "remaining": n2}

    # L3 — rapid same-URL repeats within (user, course): same url as previous AND
    #      delta < DEBOUNCE_SECONDS. Keep FIRST of each run.
    df2 = df2.sort_values(["user_id", "course_id", "created_at"], kind="mergesort")
    grp = [df2["user_id"], df2["course_id"]]
    prev_url = df2["normalized_url"].groupby(grp).shift(1)
    prev_time = df2["created_at"].groupby(grp).shift(1)
    delta = (df2["created_at"] - prev_time).dt.total_seconds()
    drop_mask = (df2["normalized_url"] == prev_url) & (delta < DEBOUNCE_SECONDS) & delta.notna()
    df3 = df2.loc[~drop_mask].copy()
    n3 = len(df3)
    report["L3_rapid_repeats"] = {"removed": n2 - n3, "remaining": n3}

    # Timezone — America/Santiago local columns (keep UTC created_at)
    df3["created_at_local"] = df3["created_at"].dt.tz_convert("America/Santiago")
    df3["hour_local"] = df3["created_at_local"].dt.hour.astype("int32")
    df3["dow_local"] = df3["created_at_local"].dt.dayofweek.astype("int32")

    off = ((df3["created_at"].dt.tz_localize(None) - df3["created_at_local"].dt.tz_localize(None))
           .dt.total_seconds() / 3600).round().astype(int)
    off_counts = {int(k): int(v) for k, v in off.value_counts().items()}
    utc_h = df3["created_at"].dt.hour
    report["timezone"] = {
        "per_row_offset_hours_counts": off_counts,
        "all_rows_offset_3_or_4": bool(set(off_counts.keys()) <= {3, 4}),
        "night_share_utc": round(float(((utc_h >= 0) & (utc_h < 6)).mean()), 4),
        "night_share_local": round(float(((df3["hour_local"] >= 0) & (df3["hour_local"] < 6)).mean()), 4),
    }

    report["rows_output"] = n3
    report["total_removed"] = n0 - n3
    report["monotone_non_increasing"] = bool(n0 >= n1 >= n2 >= n3)
    report["per_course_rows_output"] = {int(k): int(v) for k, v in df3["course_id"].value_counts().items()}
    report["n_users_output"] = int(df3["user_id"].nunique())
    return df3, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--no-write", action="store_true", help="idempotency check: don't write outputs")
    ap.add_argument("--from-clean", action="store_true",
                    help="idempotency check: input is already-clean parquet")
    args = ap.parse_args()

    print(f"[G1] reading {args.input}", flush=True)
    if args.from_clean:
        df = pd.read_parquet(args.input)
        df = df.drop(columns=[c for c in ["normalized_url", "created_at_local", "hour_local", "dow_local"]
                              if c in df.columns], errors="ignore")
        # already filtered/normalized; ensure url present
        if "url" not in df.columns:
            df["url"] = df["http_request"].fillna("").astype(str)
    else:
        df = load_raw(args.input)

    df_clean, report = clean(df)
    print(json.dumps(report, indent=2), flush=True)

    if not args.no_write:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(args.output, index=False)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[G1] wrote {args.output} ({len(df_clean)} rows) and {args.report}", flush=True)
    return report


if __name__ == "__main__":
    main()
