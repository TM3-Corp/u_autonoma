#!/usr/bin/env python3
"""Assessment Detection Visualization.

For each of the 7 benchmark courses, generates a plot showing:
- Daily total activity (gray area)
- Daily quiz+assignment activity (overlaid)
- Submission-based assessment dates (high confidence, green markers)
- Activity-based inferred assessment dates (orange markers)
- Trending-up periods (shaded regions before spikes)

Two detection methods:
  1. Submission-driven: dates with quiz submission spikes (the evaluation IS in Canvas)
  2. Activity-driven: total activity spikes with no matching submission spike
     (the evaluation likely exists but is NOT in Canvas)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
OUTPUT_DIR = Path("data/puc/sota_results/7courses_multiclass")
SPIKE_STD = 2.0  # threshold: mean + N*std
TREND_STD = 1.0  # slope threshold for trending-up detection
ROLLING_WINDOW = 3  # days for rolling mean
CLUSTER_DAYS = 3  # merge spikes within N days into one event


def load_data() -> pd.DataFrame:
    df = pd.read_parquet("data/puc/puc_fixed_data.parquet")
    df = df[df["course_id"].isin(COURSE_IDS)].copy()
    df["date"] = pd.to_datetime(df["created_at"]).dt.date
    return df


def detect_submission_spikes(df_course: pd.DataFrame) -> tuple[list, pd.Series]:
    """Detect assessment dates from quiz submission spikes."""
    # Quiz submissions: action='submission' and category='quizzes'
    subs = df_course[
        (df_course["action"] == "submission") & (df_course["category"] == "quizzes")
    ]
    if len(subs) < 5:
        return [], pd.Series(dtype=float)

    daily_subs = subs.groupby("date").size()
    # Also count assignment 'create' actions as potential submission signals
    asgn_creates = df_course[
        (df_course["action"] == "create") & (df_course["category"] == "assignments")
    ]
    daily_asgn = asgn_creates.groupby("date").size()

    # Combine submission signals
    daily_combined = daily_subs.add(daily_asgn, fill_value=0)

    threshold = daily_combined.mean() + SPIKE_STD * daily_combined.std()
    spike_dates = daily_combined[daily_combined > threshold].index.tolist()

    return spike_dates, daily_combined


def detect_quiz_assignment_spikes(df_course: pd.DataFrame) -> tuple[list, pd.Series]:
    """Detect assessment dates from quiz+assignment page view spikes."""
    assessment = df_course[
        df_course["category"].isin(["quizzes", "assignments"])
    ]
    if len(assessment) < 10:
        return [], pd.Series(dtype=float)

    daily = assessment.groupby("date").size()
    threshold = daily.mean() + SPIKE_STD * daily.std()
    spike_dates = daily[daily > threshold].index.tolist()

    return spike_dates, daily


def detect_activity_spikes(df_course: pd.DataFrame) -> tuple[list, pd.Series]:
    """Detect spikes from total activity."""
    daily = df_course.groupby("date").size()
    threshold = daily.mean() + SPIKE_STD * daily.std()
    spike_dates = daily[daily > threshold].index.tolist()

    return spike_dates, daily


def detect_trending_up(daily_series: pd.Series) -> list[tuple]:
    """Detect trending-up periods from rolling mean slope."""
    if len(daily_series) < ROLLING_WINDOW + 2:
        return []

    rolling = daily_series.rolling(ROLLING_WINDOW, min_periods=1).mean()
    slope = rolling.diff()

    threshold = slope.mean() + TREND_STD * slope.std()
    above = slope > threshold

    # Find consecutive runs of above-threshold slope
    periods = []
    start = None
    dates = daily_series.index.tolist()

    for i, date in enumerate(dates):
        if above.get(date, False):
            if start is None:
                start = date
        else:
            if start is not None:
                periods.append((start, dates[i - 1]))
                start = None
    if start is not None:
        periods.append((start, dates[-1]))

    return periods


def cluster_spikes(spike_dates: list, window_days: int = CLUSTER_DAYS) -> list[list]:
    """Group spike dates within window_days of each other into events."""
    if not spike_dates:
        return []

    sorted_dates = sorted(spike_dates)
    clusters = [[sorted_dates[0]]]

    for date in sorted_dates[1:]:
        if (pd.Timestamp(date) - pd.Timestamp(clusters[-1][-1])).days <= window_days:
            clusters[-1].append(date)
        else:
            clusters.append([date])

    return clusters


def classify_events(
    submission_spikes: list,
    assessment_view_spikes: list,
    activity_spikes: list,
) -> tuple[list, list]:
    """Classify detected events as 'certain' or 'inferred'.

    Certain: submission spike OR assessment-view spike
    Inferred: activity-only spike with no matching submission/assessment spike
    """
    # Convert to sets of dates for fast lookup
    sub_set = set(str(d) for d in submission_spikes)
    assess_set = set(str(d) for d in assessment_view_spikes)
    activity_set = set(str(d) for d in activity_spikes)

    certain_dates = []
    inferred_dates = []

    # Certain: any date with submission or assessment-view spike
    all_certain = sub_set | assess_set
    for d in sorted(all_certain):
        certain_dates.append(pd.Timestamp(d).date())

    # Inferred: activity spike with no matching certain event (within 1 day)
    for d in sorted(activity_set):
        d_ts = pd.Timestamp(d).date()
        # Check if any certain date is within 1 day
        is_explained = False
        for c in certain_dates:
            if abs((pd.Timestamp(c) - pd.Timestamp(d_ts)).days) <= 1:
                is_explained = True
                break
        if not is_explained:
            inferred_dates.append(d_ts)

    return certain_dates, inferred_dates


def get_course_info(df_course: pd.DataFrame, course_id: int) -> dict:
    """Summarize assessment structure of a course."""
    n_quiz_resources = df_course[df_course["category"] == "quizzes"]["resource_id"].nunique()
    n_asgn_resources = df_course[df_course["category"] == "assignments"]["resource_id"].nunique()
    n_submissions = len(df_course[df_course["action"] == "submission"])
    n_students = df_course["student_id"].nunique()

    return {
        "course_id": course_id,
        "n_students": n_students,
        "n_quiz_resources": n_quiz_resources,
        "n_asgn_resources": n_asgn_resources,
        "n_submissions": n_submissions,
        "has_canvas_assessments": n_quiz_resources > 0 or n_asgn_resources > 0,
    }


def plot_course(
    ax: plt.Axes,
    df_course: pd.DataFrame,
    course_id: int,
    info: dict,
):
    """Plot assessment detection for one course."""
    # Compute daily series
    daily_total = df_course.groupby("date").size()
    daily_quiz = df_course[df_course["category"] == "quizzes"].groupby("date").size()
    daily_asgn = df_course[df_course["category"] == "assignments"].groupby("date").size()

    # Full date range
    all_dates = pd.date_range(daily_total.index.min(), daily_total.index.max())
    daily_total = daily_total.reindex(all_dates, fill_value=0)
    daily_quiz = daily_quiz.reindex(all_dates, fill_value=0)
    daily_asgn = daily_asgn.reindex(all_dates, fill_value=0)

    # Detections
    submission_spikes, _ = detect_submission_spikes(df_course)
    assess_view_spikes, _ = detect_quiz_assignment_spikes(df_course)
    activity_spikes, _ = detect_activity_spikes(df_course)

    certain_dates, inferred_dates = classify_events(
        submission_spikes, assess_view_spikes, activity_spikes
    )

    trending_periods = detect_trending_up(daily_total)

    # Plot total activity (gray area)
    ax.fill_between(
        daily_total.index, daily_total.values,
        alpha=0.25, color="#bdc3c7", label="Actividad total",
    )
    ax.plot(daily_total.index, daily_total.values, color="#95a5a6", linewidth=0.5, alpha=0.7)

    # Plot quiz+assignment activity (stacked)
    ax.fill_between(
        daily_quiz.index, daily_quiz.values,
        alpha=0.6, color="#3498db", label="Quizzes",
    )
    ax.fill_between(
        daily_asgn.index, daily_asgn.values + daily_quiz.values,
        daily_quiz.values,
        alpha=0.5, color="#e67e22", label="Tareas",
    )

    # Trending-up periods (light yellow shading)
    for start, end in trending_periods:
        ax.axvspan(
            pd.Timestamp(start), pd.Timestamp(end),
            alpha=0.15, color="#f39c12", zorder=0,
        )

    # Certain assessment dates (green triangles at top)
    y_top = daily_total.max() * 0.95
    for d in certain_dates:
        ax.plot(
            pd.Timestamp(d), y_top, marker="v", color="#27ae60",
            markersize=8, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
        )

    # Inferred assessment dates (red diamonds at top)
    for d in inferred_dates:
        ax.plot(
            pd.Timestamp(d), y_top * 0.85, marker="D", color="#e74c3c",
            markersize=7, zorder=5, markeredgecolor="white", markeredgewidth=0.5,
        )

    # Styling
    ax.set_title(
        f"Curso {course_id}  "
        f"({info['n_students']} est. | "
        f"{info['n_quiz_resources']} quizzes, {info['n_asgn_resources']} tareas | "
        f"{info['n_submissions']} envíos)",
        fontsize=10, fontweight="bold",
    )
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel("Page views/día", fontsize=8)

    # Annotations: count events
    n_certain = len(certain_dates)
    n_inferred = len(inferred_dates)
    ax.text(
        0.98, 0.95,
        f"{n_certain} confirmadas\n{n_inferred} inferidas",
        transform=ax.transAxes, ha="right", va="top", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )


def main():
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = load_data()

    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    axes = axes.flatten()

    all_info = []

    for i, course_id in enumerate(COURSE_IDS):
        print(f"  Processing course {course_id}...")
        df_course = df[df["course_id"] == course_id]
        info = get_course_info(df_course, course_id)
        all_info.append(info)
        plot_course(axes[i], df_course, course_id, info)

    # Hide the unused 8th subplot — use for legend
    ax_legend = axes[7]
    ax_legend.axis("off")

    legend_elements = [
        Patch(facecolor="#bdc3c7", alpha=0.4, label="Actividad total"),
        Patch(facecolor="#3498db", alpha=0.6, label="Quizzes"),
        Patch(facecolor="#e67e22", alpha=0.5, label="Tareas (assignments)"),
        Patch(facecolor="#f39c12", alpha=0.25, label="Tendencia al alza (trending up)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#27ae60",
               markersize=12, label="Evaluación confirmada (submissions)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#e74c3c",
               markersize=10, label="Evaluación inferida (solo actividad)"),
    ]
    ax_legend.legend(
        handles=legend_elements, loc="center", fontsize=12,
        frameon=True, fancybox=True, shadow=True,
        title="Leyenda", title_fontsize=13,
    )
    ax_legend.text(
        0.5, 0.15,
        "Confirmada = spike en envíos de quizzes/tareas\n"
        "Inferida = spike en actividad total sin spike de evaluaciones\n"
        "(posible evaluación fuera de Canvas)",
        transform=ax_legend.transAxes, ha="center", va="center",
        fontsize=10, color="#555", style="italic",
    )

    fig.suptitle(
        "Detección de Evaluaciones por Curso: Confirmadas vs Inferidas",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / "assessment_detection_by_course.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")

    # Print summary table
    print(f"\n{'='*85}")
    print("Assessment Detection Summary")
    print(f"{'='*85}")
    print(f"{'Course':>8} {'Students':>9} {'Quizzes':>8} {'Assigns':>8} "
          f"{'Submissions':>12} {'Confirmed':>10} {'Inferred':>9}")
    print("-" * 85)

    for info in all_info:
        cid = info["course_id"]
        df_course = df[df["course_id"] == cid]
        sub_spikes, _ = detect_submission_spikes(df_course)
        assess_spikes, _ = detect_quiz_assignment_spikes(df_course)
        act_spikes, _ = detect_activity_spikes(df_course)
        certain, inferred = classify_events(sub_spikes, assess_spikes, act_spikes)

        print(
            f"{cid:>8} {info['n_students']:>9} {info['n_quiz_resources']:>8} "
            f"{info['n_asgn_resources']:>8} {info['n_submissions']:>12} "
            f"{len(certain):>10} {len(inferred):>9}"
        )

    print()


if __name__ == "__main__":
    main()
