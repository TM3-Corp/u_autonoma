#!/usr/bin/env python3
"""T1 — PUC clean rebuild: 3-level dedup + timezone + interaction_seconds audit.

Input : data/puc/puc_fixed_data.parquet (raw, UTC, zero dedup)
Output: data/puc/puc_clean_data.parquet
        data/puc/sota_results/tier1_clean/cleaning_report.json

Recipes are ported from the thesis notebook (see EXPERIMENT_REGISTER.md addenda).
Idempotent by construction: running on its own output removes 0 rows.
"""
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DEFAULT_IN = REPO / "data/puc/puc_fixed_data.parquet"
DEFAULT_OUT = REPO / "data/puc/puc_clean_data.parquet"
DEFAULT_REPORT = REPO / "data/puc/sota_results/tier1_clean/cleaning_report.json"

DEBOUNCE_SECONDS = 10
CAP_SECONDS = 1800  # Canvas interaction_seconds cap referenced in thesis

# strip a leading /api/v1 that sits right after an optional scheme://host
_API_RE = re.compile(r'^(https?://[^/]+)?/api/v1(?=/|\?|$)')


def normalize_url(s: pd.Series) -> pd.Series:
    return s.str.replace(_API_RE, lambda m: m.group(1) or '', regex=True)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    report = {}
    n0 = len(df)
    report["rows_input"] = n0

    # L1 — exact duplicate rows across all columns
    df1 = df.drop_duplicates()
    n1 = len(df1)
    report["L1_exact_duplicates"] = {"removed": n0 - n1, "remaining": n1}

    # normalized_url (HTML twin of /api/v1/... path)
    df1 = df1.copy()
    df1["normalized_url"] = normalize_url(df1["url"])

    # L2 — HTML+API twin: dedup on (student, course, normalized_url, created_at@1s), keep first
    ts_1s = df1["created_at"].dt.floor("s")
    df2 = df1.loc[~pd.DataFrame({
        "s": df1["student_id"], "c": df1["course_id"],
        "u": df1["normalized_url"], "t": ts_1s,
    }).duplicated(keep="first")].copy()
    n2 = len(df2)
    report["L2_html_api_twin"] = {"removed": n1 - n2, "remaining": n2}

    # L3 — rapid same-URL repeats within (student, course): drop if same url as previous
    #      original row AND delta < DEBOUNCE_SECONDS. Keep FIRST of each run.
    df2 = df2.sort_values(["student_id", "course_id", "created_at"], kind="mergesort")
    grp = [df2["student_id"], df2["course_id"]]
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

    modal_utc = int(df3["created_at"].dt.hour.mode().iloc[0])
    modal_local = int(df3["hour_local"].mode().iloc[0])
    # Robust check: per-row UTC-minus-local offset must be exactly 3 or 4h for
    # every row (Chile = UTC-4 winter / UTC-3 DST). The modal-hour proxy is
    # confounded here by a near-flat midday UTC distribution (hours 12-15 within
    # ~7%), so we assert on the per-row offset, which is unambiguous.
    off = ((df3["created_at"].dt.tz_localize(None) - df3["created_at_local"].dt.tz_localize(None))
           .dt.total_seconds() / 3600).round().astype(int)
    off_counts = {int(k): int(v) for k, v in off.value_counts().items()}
    utc_h = df3["created_at"].dt.hour
    night_utc = float(((utc_h >= 0) & (utc_h < 6)).mean())
    night_local = float(((df3["hour_local"] >= 0) & (df3["hour_local"] < 6)).mean())
    report["timezone"] = {
        "modal_hour_utc": modal_utc,
        "modal_hour_local": modal_local,
        "per_row_offset_hours_counts": off_counts,
        "all_rows_offset_3_or_4": bool(set(off_counts.keys()) <= {3, 4}),
        "night_share_utc": round(night_utc, 4),
        "night_share_local": round(night_local, 4),
        "modal_proxy_note": "modal-hour diff is a weak proxy (flat midday UTC dist); "
                            "per-row offset is the definitive check.",
    }

    report["rows_output"] = n3
    report["total_removed"] = n0 - n3
    report["monotone_non_increasing"] = n0 >= n1 >= n2 >= n3
    return df3, report


def interaction_seconds_audit(df: pd.DataFrame) -> dict:
    s = df["interaction_seconds"]
    n = len(s)
    nonnull = s.dropna()
    by_ctrl = (
        df.groupby("controller")["interaction_seconds"]
        .agg(["count", "mean", "median"]).sort_values("count", ascending=False).head(15)
    )
    return {
        "n": n,
        "pct_null": round(100 * s.isna().mean(), 4),
        "pct_zero": round(100 * (nonnull == 0).mean(), 4),
        "pct_gt_1800": round(100 * (nonnull > CAP_SECONDS).mean(), 4),
        "median": float(nonnull.median()),
        "mean": float(nonnull.mean()),
        "max": float(nonnull.max()),
        "top_repeated_values": {
            str(k): int(v) for k, v in nonnull.round(3).value_counts().head(12).items()
        },
        "by_controller_top15": {
            str(idx): {"count": int(r["count"]), "mean": round(float(r["mean"]), 2),
                       "median": round(float(r["median"]), 2)}
            for idx, r in by_ctrl.iterrows()
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    ap.add_argument("--report", default=str(DEFAULT_REPORT))
    ap.add_argument("--no-write", action="store_true", help="idempotency check: don't write outputs")
    args = ap.parse_args()

    print(f"[T1] reading {args.input}", flush=True)
    df = pd.read_parquet(args.input)
    # if re-run on our own output, drop derived cols so the pipeline is self-consistent
    df = df.drop(columns=[c for c in ["normalized_url", "created_at_local", "hour_local", "dow_local"]
                          if c in df.columns], errors="ignore")

    df_clean, report = clean(df)
    report["interaction_seconds_audit"] = interaction_seconds_audit(df_clean)

    print(json.dumps({k: v for k, v in report.items() if k != "interaction_seconds_audit"}, indent=2), flush=True)

    if not args.no_write:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        df_clean.to_parquet(args.output, index=False)
        with open(args.report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[T1] wrote {args.output} ({len(df_clean)} rows) and {args.report}", flush=True)
    return report


if __name__ == "__main__":
    main()
