#!/usr/bin/env python3
"""Calendar heatmap of daily LMS activity per course.

For each of the 7 benchmark courses, generates a weekly calendar heatmap:
- Rows = weeks (Mon-Sun), columns = days of week
- Color intensity = daily page view count (white → blue)
- Green down-triangle = quiz day (peak submission day per quiz resource_id)
- Blue down-triangle = assignment day (peak activity day per assignment resource_id)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path

COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
OUTPUT_DIR = Path("data/puc/sota_results/7courses_multiclass")

DAY_LABELS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet("data/puc/puc_fixed_data.parquet")
    df = df[df["course_id"].isin(COURSE_IDS)].copy()
    df["date"] = pd.to_datetime(df["created_at"]).dt.normalize()
    return df


def identify_quiz_shell_assignments(df_course: pd.DataFrame) -> set:
    """Returns set of assignment resource_ids that are quiz shells.

    Canvas creates an assignment shell for every quiz. When a student loads
    a quiz page, Canvas JS fires both the quiz request AND an assignment API
    call for the shell object within milliseconds. We detect this by checking
    temporal co-occurrence: for each assignment, what fraction of its
    (student, 10-second-bucket) pairs also appear in quiz page views?

    The distribution is perfectly bimodal:
      genuine assignments: 0-8% overlap
      quiz shells:         50-100% overlap
    Any threshold between 0.10 and 0.49 gives identical classification.
    """
    rid_numeric = pd.to_numeric(df_course["resource_id"], errors="coerce").fillna(0)
    valid_rid = rid_numeric > 0

    quizzes = df_course[(df_course["category"] == "quizzes") & valid_rid]
    assignments = df_course[(df_course["category"] == "assignments") & valid_rid]

    if quizzes.empty or assignments.empty:
        return set()

    # Build set of (student_id, 10s_bucket) for all quiz page views
    ts = pd.to_datetime(df_course["created_at"])
    quiz_ts = ts[quizzes.index].astype("int64") // 10**9 // 10
    quiz_signatures = set(zip(quizzes["student_id"], quiz_ts))

    # For each assignment resource_id, check overlap with quiz signatures
    shell_rids = set()
    asgn_ts = ts[assignments.index].astype("int64") // 10**9 // 10
    for rid, idx in assignments.groupby("resource_id").groups.items():
        if len(idx) < 5:
            continue
        asgn_sigs = set(zip(assignments.loc[idx, "student_id"], asgn_ts[idx]))
        overlap = len(asgn_sigs & quiz_signatures) / len(asgn_sigs)
        if overlap >= 0.15:
            shell_rids.add(rid)

    return shell_rids


MIN_STUDENT_COVERAGE = 0.25  # Assignment must be viewed by >=25% of students


def get_evaluation_days(df_course: pd.DataFrame) -> tuple[list, list, int, int]:
    """Get the peak day for each quiz and genuine assignment resource.

    Excludes quiz-shell assignments and low-engagement assignments.
    Returns (quiz_days, assignment_days, n_shells_excluded, n_low_coverage).
    """
    rid_numeric = pd.to_numeric(df_course["resource_id"], errors="coerce").fillna(0)
    valid_rid = rid_numeric > 0
    n_students = df_course["student_id"].nunique()

    quiz_days = []
    quizzes = df_course[(df_course["category"] == "quizzes") & valid_rid]
    for _, grp in quizzes.groupby("resource_id"):
        subs = grp[grp["action"] == "submission"]
        if len(subs) >= 3:
            peak_day = subs.groupby("date").size().idxmax()
        else:
            peak_day = grp.groupby("date").size().idxmax()
        quiz_days.append(peak_day)

    # Identify quiz-shell assignments
    shell_rids = identify_quiz_shell_assignments(df_course)
    n_shells_excluded = len(shell_rids)

    # Filter assignments: exclude shells and low-coverage
    assignment_days = []
    n_low_coverage = 0
    assignments = df_course[(df_course["category"] == "assignments") & valid_rid]
    for rid, grp in assignments.groupby("resource_id"):
        if rid in shell_rids:
            continue
        # Check student coverage
        students_viewing = grp["student_id"].nunique()
        if students_viewing < n_students * MIN_STUDENT_COVERAGE:
            n_low_coverage += 1
            continue
        shows = grp[grp["action"] == "show"]
        if len(shows) >= 3:
            peak_day = shows.groupby("date").size().idxmax()
        else:
            peak_day = grp.groupby("date").size().idxmax()
        assignment_days.append(peak_day)

    return quiz_days, assignment_days, n_shells_excluded, n_low_coverage


def build_calendar_matrix(
    daily_counts: pd.Series,
) -> tuple[np.ndarray, list[str], pd.Timestamp]:
    """Build a (n_weeks, 7) matrix of daily activity.

    Returns (matrix, week_labels, first_monday).
    """
    first_date = daily_counts.index.min()
    last_date = daily_counts.index.max()

    # Align to Monday
    first_monday = first_date - pd.Timedelta(days=first_date.dayofweek)
    last_sunday = last_date + pd.Timedelta(days=6 - last_date.dayofweek)

    all_dates = pd.date_range(first_monday, last_sunday)
    counts = daily_counts.reindex(all_dates, fill_value=0)

    n_weeks = len(all_dates) // 7
    matrix = counts.values[:n_weeks * 7].reshape(n_weeks, 7).astype(float)

    # Week labels (show month + day of the Monday)
    week_labels = []
    for w in range(n_weeks):
        monday = first_monday + pd.Timedelta(weeks=w)
        week_labels.append(monday.strftime("%d %b"))

    return matrix, week_labels, first_monday


def date_to_cell(date: pd.Timestamp, first_monday: pd.Timestamp) -> tuple[int, int]:
    """Convert a date to (row, col) in the calendar matrix."""
    delta = (date - first_monday).days
    if delta < 0:
        return -1, -1
    row = delta // 7
    col = delta % 7
    return row, col


def plot_course_heatmap(
    ax: plt.Axes,
    df_course: pd.DataFrame,
    course_id: int,
):
    """Plot calendar heatmap for one course."""
    daily_counts = df_course.groupby("date").size()
    matrix, week_labels, first_monday = build_calendar_matrix(daily_counts)
    n_weeks = matrix.shape[0]

    quiz_days, assignment_days, n_shells, n_low_cov = get_evaluation_days(df_course)
    n_students = df_course["student_id"].nunique()
    n_quizzes = len(quiz_days)
    n_assignments = len(assignment_days)
    n_excluded = n_shells + n_low_cov

    # Custom colormap: white → light blue → dark blue
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "activity", ["#f8f9fa", "#d4e6f1", "#5dade2", "#2471a3", "#1a5276"]
    )

    # Mask zero-activity cells as NaN so they show as light gray
    display = matrix.copy()
    display[display == 0] = np.nan

    im = ax.imshow(
        display, cmap=cmap, aspect="auto",
        vmin=0, vmax=np.nanpercentile(matrix[matrix > 0], 95) if np.any(matrix > 0) else 1,
    )

    # Draw grid
    for i in range(n_weeks + 1):
        ax.axhline(i - 0.5, color="white", linewidth=1.5)
    for j in range(8):
        ax.axvline(j - 0.5, color="white", linewidth=1.5)

    # Quiz markers (green triangles)
    for d in quiz_days:
        d_ts = pd.Timestamp(d)
        row, col = date_to_cell(d_ts, first_monday)
        if 0 <= row < n_weeks:
            ax.plot(
                col, row, marker="v", color="#27ae60",
                markersize=7, markeredgecolor="white", markeredgewidth=0.8,
                zorder=5,
            )

    # Assignment markers (blue triangles)
    for d in assignment_days:
        d_ts = pd.Timestamp(d)
        row, col = date_to_cell(d_ts, first_monday)
        if 0 <= row < n_weeks:
            ax.plot(
                col, row, marker="v", color="#2980b9",
                markersize=6, markeredgecolor="white", markeredgewidth=0.8,
                zorder=4,
            )

    # Axis labels
    ax.set_xticks(range(7))
    ax.set_xticklabels(DAY_LABELS, fontsize=8)
    ax.set_yticks(range(n_weeks))
    ax.set_yticklabels(week_labels, fontsize=7)
    ax.tick_params(axis="both", length=0)

    title = (
        f"Curso {course_id}  ({n_students} est. | "
        f"{n_quizzes} quizzes, {n_assignments} tareas"
    )
    if n_excluded > 0:
        title += f", {n_excluded} excl."
    title += ")"
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    return im


def main():
    print("Loading data...")
    df = load_data()

    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    axes = axes.flatten()

    ims = []
    print(f"\n{'Course':>8}  {'Quiz':>5}  {'Asgn':>5}  {'Shell':>5}  {'LowCov':>6}  {'Orig':>5}")
    print("-" * 48)
    for i, course_id in enumerate(COURSE_IDS):
        df_course = df[df["course_id"] == course_id]

        # Get original assignment count for comparison
        rid_num = pd.to_numeric(df_course["resource_id"], errors="coerce").fillna(0)
        orig_asgn = df_course.loc[
            (df_course["category"] == "assignments") & (rid_num > 0), "resource_id"
        ].nunique()

        quiz_days, asgn_days, n_sh, n_lc = get_evaluation_days(df_course)
        print(
            f"{course_id:>8}  {len(quiz_days):>5}  {len(asgn_days):>5}  "
            f"{n_sh:>5}  {n_lc:>6}  {orig_asgn:>5}"
        )

        im = plot_course_heatmap(axes[i], df_course, course_id)
        ims.append(im)
    print()

    # Use 8th subplot for legend + colorbar
    ax_legend = axes[7]
    ax_legend.axis("off")

    legend_elements = [
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#27ae60",
               markersize=12, markeredgecolor="white", markeredgewidth=0.8,
               label=f"Quiz (día pico de envíos)"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#2980b9",
               markersize=11, markeredgecolor="white", markeredgewidth=0.8,
               label=f"Tarea genuina (sin quiz-shell)"),
    ]
    ax_legend.legend(
        handles=legend_elements, loc="upper center", fontsize=11,
        frameon=True, fancybox=True, title="Evaluaciones", title_fontsize=12,
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.78, 0.12, 0.15, 0.02])
    cbar = fig.colorbar(ims[0], cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Page views / día", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    ax_legend.text(
        0.5, 0.30,
        "Intensidad: blanco = sin actividad\n"
        "azul oscuro = alta actividad\n\n"
        "Cada fila = 1 semana (lunes a domingo)\n"
        "Cada columna = 1 día de la semana",
        transform=ax_legend.transAxes, ha="center", va="center",
        fontsize=10, color="#555", style="italic",
    )

    fig.suptitle(
        "Calendario de Actividad LMS y Evaluaciones por Curso",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = OUTPUT_DIR / "calendar_heatmap_by_course.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
