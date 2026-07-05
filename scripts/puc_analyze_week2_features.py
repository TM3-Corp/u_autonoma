#!/usr/bin/env python3
"""Week 2 Feature Effect Size Analysis.

Computes and visualizes statistical differences between fail/pass students
at week 2 using the same features the benchmark models see.

Outputs:
  - Console table of top 20 features by |Cohen's D|
  - week2_feature_effects.png: 4x3 boxplot grid (top 12 features)
  - week2_effect_size_ranking.png: horizontal bar chart of Cohen's D
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Import feature functions from benchmark script
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "benchmark", str(SCRIPT_DIR / "puc_benchmark_sota.py")
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

calculate_all_features = benchmark.calculate_all_features
filter_by_cutoff = benchmark.filter_by_cutoff
get_course_starts = benchmark.get_course_starts
calculate_znorm = benchmark.calculate_znorm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COURSE_IDS = [54503, 54529, 55010, 55183, 55410, 54570, 54581]
CUTOFF_WEEK = 2
PERCENTILE = 0.20  # best pct from benchmark
FAIL_THRESHOLD = 4.0
OUTPUT_DIR = Path("data/puc/sota_results/7courses_multiclass")

FEATURE_LABELS = {
    "weekly_std": "Variabilidad Semanal",
    "modu_rank_pct_mean": "Proactividad en Módulos",
    "n_sessions": "Número de Sesiones",
    "n_sessions_znorm": "Sesiones (z-norm)",
    "quizzes_views": "Vistas de Quizzes",
    "quizzes_views_znorm": "Quizzes (z-norm)",
    "max_gap_hours": "Máx. Inactividad (hrs)",
    "unique_transitions": "Transiciones Únicas",
    "total_views": "Total Page Views",
    "total_views_znorm": "Page Views (z-norm)",
    "daily_consistency": "Consistencia Diaria",
    "mean_gap_hours": "Inactividad Media (hrs)",
    "resource_coverage_rate": "Cobertura de Recursos",
    "active_weeks": "Semanas Activas",
    "sessions_per_week": "Sesiones/Semana",
    "total_time_min": "Tiempo Total (min)",
    "hour_entropy": "Diversidad Horaria",
    "hour_entropy_znorm": "Diversidad Horaria (z-norm)",
    "early_late_ratio": "Ratio Inicio/Final",
    "file_rank_pct_mean": "Proactividad en Archivos",
    "file_rank_pct_mean_znorm": "Proactividad Archivos (z-norm)",
    "weekly_std_znorm": "Variabilidad Semanal (z-norm)",
    "external_tools_views": "Vistas Herram. Externas",
    "external_tools_views_znorm": "Herram. Externas (z-norm)",
    "modules_proact_std_pct": "Variab. Proactividad Módulos",
    "quizzes_proact_mean_pct": "Proact. Media Quizzes",
    "quizzes_unique": "Quizzes Únicos",
    "pages_views": "Vistas de Páginas",
    "assignments_views": "Vistas de Tareas",
    "modules_views": "Vistas de Módulos",
    "files_views": "Vistas de Archivos",
    "discussions_views": "Vistas de Foros",
    "announcements_views": "Vistas de Anuncios",
    "grades_views": "Vistas de Calificaciones",
    "avg_session_duration_min": "Duración Media Sesión (min)",
    "weekend_ratio": "Ratio Fin de Semana",
    "night_ratio": "Ratio Nocturno",
    "first_access_week": "Semana Primer Acceso",
}

COLORS = {
    "aprobados": "#6bcb77",
    "reprobados": "#ff6b6b",
}


# ---------------------------------------------------------------------------
# Statistical functions
# ---------------------------------------------------------------------------
def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Cohen's D with pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def cliffs_delta(group1: pd.Series, group2: pd.Series) -> float:
    """Cliff's Delta — non-parametric effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    # Count dominance pairs
    more = sum(1 for x in group1 for y in group2 if x > y)
    less = sum(1 for x in group1 for y in group2 if x < y)
    return (more - less) / (n1 * n2)


def cliffs_delta_fast(group1: np.ndarray, group2: np.ndarray) -> float:
    """Cliff's Delta using vectorized Mann-Whitney statistic."""
    n1, n2 = len(group1), len(group2)
    if n1 == 0 or n2 == 0:
        return 0.0
    # Mann-Whitney U gives count of (x > y) pairs
    u_stat, _ = stats.mannwhitneyu(group1, group2, alternative="two-sided")
    # U = number of (x > y) pairs; total pairs = n1*n2
    # Cliff's delta = (2*U / (n1*n2)) - 1
    return (2 * u_stat / (n1 * n2)) - 1


def benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.zeros(n)
    # BH correction
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            adjusted[i] = sorted_p[i]
        else:
            adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    # Map back to original order
    result = np.zeros(n)
    result[sorted_idx] = adjusted
    return result


def significance_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""


def label_for(feat: str) -> str:
    return FEATURE_LABELS.get(feat, feat)


# ---------------------------------------------------------------------------
# Data loading & feature computation
# ---------------------------------------------------------------------------
def load_and_compute_features() -> pd.DataFrame:
    """Load data, compute features at week 2, merge labels."""
    print("Loading data...")
    df_pv = pd.read_parquet("data/puc/puc_fixed_data.parquet")
    df_grades = pd.read_parquet("data/puc/puc_grades_clean.parquet")

    # Filter to 7 benchmark courses
    df_pv = df_pv[df_pv["course_id"].isin(COURSE_IDS)].copy()
    df_grades = df_grades[df_grades["course_id"].isin(COURSE_IDS)].copy()

    print(f"  Page views: {len(df_pv):,} rows, {df_pv['course_id'].nunique()} courses")
    print(f"  Grades: {len(df_grades):,} records")

    # Course starts at pct=0.20
    course_starts = get_course_starts(df_pv, PERCENTILE)

    # Temporal cutoff
    df_filtered = filter_by_cutoff(df_pv, course_starts, CUTOFF_WEEK)
    print(f"  After week {CUTOFF_WEEK} cutoff: {len(df_filtered):,} page views")

    # Compute features
    print("Computing features (this may take a minute)...")
    df_feat = calculate_all_features(
        df_filtered,
        course_starts,
        compute_pct=True,
        total_weeks=CUTOFF_WEEK,
        cutoff_weeks=CUTOFF_WEEK,
    )
    print(f"  Features computed: {df_feat.shape[0]} students × {df_feat.shape[1]} columns")

    # Merge grades
    df = df_feat.merge(
        df_grades[["student_id", "course_id", "grade"]],
        on=["student_id", "course_id"],
        how="inner",
    )
    df["fail"] = (df["grade"] < FAIL_THRESHOLD).astype(int)

    n_fail = df["fail"].sum()
    n_pass = len(df) - n_fail
    print(f"  After grade merge: {len(df)} student-course pairs")
    print(f"  Fail: {n_fail} ({100*n_fail/len(df):.1f}%), Pass: {n_pass} ({100*n_pass/len(df):.1f}%)")

    # Z-normalize
    feature_cols = [
        c for c in df.columns
        if c not in ("student_id", "course_id", "grade", "fail")
        and not c.endswith("_znorm")
    ]
    df = calculate_znorm(df, feature_cols)
    print(f"  After z-norm: {df.shape[1]} total columns")

    return df


# ---------------------------------------------------------------------------
# Effect size analysis
# ---------------------------------------------------------------------------
def compute_effect_sizes(df: pd.DataFrame) -> pd.DataFrame:
    """Compute effect sizes for all features."""
    feature_cols = [
        c for c in df.columns
        if c not in ("student_id", "course_id", "grade", "fail")
    ]

    pass_df = df[df["fail"] == 0]
    fail_df = df[df["fail"] == 1]

    rows = []
    for col in feature_cols:
        passed = pass_df[col].dropna()
        failed = fail_df[col].dropna()
        if len(passed) < 3 or len(failed) < 3:
            continue

        d = cohens_d(passed, failed)
        u_stat, p_val = stats.mannwhitneyu(passed, failed, alternative="two-sided")
        cliff_d = cliffs_delta_fast(passed.values, failed.values)

        rows.append({
            "feature": col,
            "cohens_d": d,
            "abs_cohens_d": abs(d),
            "p_value": p_val,
            "cliffs_delta": cliff_d,
            "median_pass": passed.median(),
            "median_fail": failed.median(),
            "median_diff": passed.median() - failed.median(),
            "mean_pass": passed.mean(),
            "mean_fail": failed.mean(),
            "n_pass": len(passed),
            "n_fail": len(failed),
        })

    results = pd.DataFrame(rows)

    # FDR correction
    results["p_adjusted"] = benjamini_hochberg(results["p_value"].values)
    results["sig"] = results["p_adjusted"].apply(significance_stars)

    # Effect size interpretation
    def interpret_d(d):
        d = abs(d)
        if d >= 0.8:
            return "grande"
        elif d >= 0.5:
            return "mediano"
        elif d >= 0.2:
            return "pequeño"
        return "negligible"

    results["effect_size"] = results["cohens_d"].apply(interpret_d)
    results = results.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)

    return results


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def plot_boxplots(df: pd.DataFrame, results: pd.DataFrame, output_path: Path):
    """4x3 grid of boxplots for top 12 features by |Cohen's D|."""
    # Exclude znorm features for boxplots (use raw scale for interpretability)
    raw_results = results[~results["feature"].str.endswith("_znorm")].head(12)

    if len(raw_results) < 12:
        # Fill with znorm if not enough raw features
        remaining = 12 - len(raw_results)
        znorm_extra = results[results["feature"].str.endswith("_znorm")].head(remaining)
        raw_results = pd.concat([raw_results, znorm_extra])

    features = raw_results["feature"].tolist()

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    pass_df = df[df["fail"] == 0]
    fail_df = df[df["fail"] == 1]

    for i, feat in enumerate(features):
        ax = axes[i]
        passed = pass_df[feat].dropna()
        failed = fail_df[feat].dropna()

        # Cap outliers at 99th percentile for visualization
        cap = np.percentile(pd.concat([passed, failed]), 99)
        floor = np.percentile(pd.concat([passed, failed]), 1)
        passed_capped = passed.clip(floor, cap)
        failed_capped = failed.clip(floor, cap)

        bp = ax.boxplot(
            [passed_capped, failed_capped],
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        bp["boxes"][0].set_facecolor(COLORS["aprobados"])
        bp["boxes"][0].set_alpha(0.85)
        bp["boxes"][1].set_facecolor(COLORS["reprobados"])
        bp["boxes"][1].set_alpha(0.85)
        for median in bp["medians"]:
            median.set_color("#d35400")
            median.set_linewidth(2)

        # Strip plot overlay (jitter) for small fail group
        n_fail = len(failed_capped)
        if n_fail <= 60:
            jitter_pass = 1 + np.random.normal(0, 0.04, len(passed_capped))
            jitter_fail = 2 + np.random.normal(0, 0.04, n_fail)
            ax.scatter(
                jitter_pass, passed_capped, alpha=0.25, s=8,
                color=COLORS["aprobados"], edgecolors="none", zorder=3,
            )
            ax.scatter(
                jitter_fail, failed_capped, alpha=0.4, s=12,
                color=COLORS["reprobados"], edgecolors="none", zorder=3,
            )

        # Annotation
        row = raw_results[raw_results["feature"] == feat].iloc[0]
        d_val = row["cohens_d"]
        sig = row["sig"]
        ax.text(
            0.5, 0.97, f"d={d_val:.2f}{sig}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.8),
        )

        ax.set_xticklabels(
            [f"Aprob.\n(n={len(passed)})", f"Reprob.\n(n={len(failed)})"],
            fontsize=8,
        )
        ax.set_title(label_for(feat), fontsize=10, fontweight="bold")
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused axes
    for j in range(len(features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Diferencias en Actividad LMS: Aprobados vs Reprobados (Semana 2)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.01,
        f"N={len(df)} estudiantes-curso | Umbral reprobación: nota < {FAIL_THRESHOLD} | "
        f"Cohen's d: pequeño≥0.2, mediano≥0.5, grande≥0.8",
        ha="center", fontsize=9, color="#666",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_effect_ranking(results: pd.DataFrame, output_path: Path):
    """Horizontal bar chart of Cohen's D for top 20 features."""
    # Use raw features only (no znorm duplicates)
    raw_results = results[~results["feature"].str.endswith("_znorm")].head(20)
    if len(raw_results) < 20:
        remaining = 20 - len(raw_results)
        znorm_extra = results[results["feature"].str.endswith("_znorm")]
        znorm_extra = znorm_extra[~znorm_extra["feature"].str.replace("_znorm", "").isin(
            raw_results["feature"]
        )].head(remaining)
        raw_results = pd.concat([raw_results, znorm_extra])

    top = raw_results.iloc[::-1]  # reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#ff6b6b" if d < 0 else "#6bcb77" for d in top["cohens_d"]]
    bars = ax.barh(range(len(top)), top["cohens_d"], color=colors, alpha=0.85, height=0.7)

    # Effect size threshold lines
    for threshold, lbl in [(0.2, "pequeño"), (0.5, "mediano"), (0.8, "grande")]:
        ax.axvline(threshold, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.axvline(-threshold, color="#888", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(threshold + 0.02, len(top) - 0.5, lbl, fontsize=7, color="#888", va="top")

    ax.axvline(0, color="black", linewidth=0.8)

    # Labels
    labels = [f"{label_for(f)} {s}" for f, s in zip(top["feature"], top["sig"])]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cohen's D (efecto)", fontsize=11)
    ax.set_title(
        "Tamaño del Efecto por Feature: Aprobados vs Reprobados (Semana 2)",
        fontsize=13, fontweight="bold",
    )

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#6bcb77", alpha=0.85, label="Aprobados > Reprobados"),
        Patch(facecolor="#ff6b6b", alpha=0.85, label="Reprobados > Aprobados"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load data and compute features
    df = load_and_compute_features()

    # 2. Compute effect sizes
    print("\nComputing effect sizes...")
    results = compute_effect_sizes(df)

    # 3. Print table
    print(f"\n{'='*90}")
    print(f"Top 20 Features by |Cohen's D| — Week {CUTOFF_WEEK}, binary_{FAIL_THRESHOLD}")
    print(f"{'='*90}")
    print(f"{'Feature':<35} {'Cohen D':>8} {'Cliff δ':>8} {'p(adj)':>10} {'Sig':>4} "
          f"{'Med Pass':>9} {'Med Fail':>9} {'Effect':>10}")
    print("-" * 90)

    for _, row in results.head(20).iterrows():
        feat = row["feature"]
        lbl = label_for(feat)
        if len(lbl) > 33:
            lbl = lbl[:30] + "..."
        print(
            f"{lbl:<35} {row['cohens_d']:>8.3f} {row['cliffs_delta']:>8.3f} "
            f"{row['p_adjusted']:>10.4f} {row['sig']:>4} "
            f"{row['median_pass']:>9.2f} {row['median_fail']:>9.2f} "
            f"{row['effect_size']:>10}"
        )

    # Count significant features
    n_sig = (results["p_adjusted"] < 0.05).sum()
    n_medium = (results["abs_cohens_d"] >= 0.5).sum()
    n_large = (results["abs_cohens_d"] >= 0.8).sum()
    print(f"\nSignificant features (p_adj<0.05): {n_sig}/{len(results)}")
    print(f"Medium+ effect (|d|≥0.5): {n_medium}")
    print(f"Large effect (|d|≥0.8): {n_large}")

    # Direction summary
    sig_results = results[results["p_adjusted"] < 0.05]
    n_pass_higher = (sig_results["cohens_d"] > 0).sum()
    n_fail_higher = (sig_results["cohens_d"] < 0).sum()
    print(f"\nDirection: {n_pass_higher} features higher in pass, "
          f"{n_fail_higher} features higher in fail")

    # 4. Visualizations
    print("\nGenerating visualizations...")
    plot_boxplots(df, results, OUTPUT_DIR / "week2_feature_effects.png")
    plot_effect_ranking(results, OUTPUT_DIR / "week2_effect_size_ranking.png")

    # 5. Save full results to CSV
    csv_path = OUTPUT_DIR / "week2_effect_sizes.csv"
    results.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
