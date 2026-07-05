#!/usr/bin/env python3
"""
Analyze Discussion Engagement Patterns.

The user is particularly interested in how discussion forum engagement
patterns (captured by PCA/t-SNE) relate to student outcomes.

Analysis includes:
1. PCA component correlation with final_score
2. Aprobados vs Reprobados comparison on discussion features
3. Per-course analysis of discussion-grade relationship
4. Statistical significance testing
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# Input files
PCA_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/pca_features.parquet')
PROACTIVITY = Path('/home/paul/projects/uautonoma/data/enriched_features/proactivity_features.parquet')
ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_DIR = Path('/home/paul/projects/uautonoma/data/report/visualizations/discussions')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = group1.var()
    var2 = group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def main():
    print('=' * 70)
    print('Discussion Engagement Pattern Analysis')
    print('=' * 70)
    print()

    # Load data
    print('Loading data...')
    pca_df = pd.read_parquet(PCA_FEATURES)
    proact_df = pd.read_parquet(PROACTIVITY)
    enrollments = pd.read_csv(ENROLLMENTS)
    enrollments['failed'] = enrollments['final_score'] < 57

    # Merge
    df = pca_df.merge(proact_df, on=['user_id', 'course_id'], how='left')
    df = df.merge(enrollments[['user_id', 'course_id', 'final_score', 'failed']],
                  on=['user_id', 'course_id'], how='left')
    df = df.dropna(subset=['failed'])

    print(f'Total students: {len(df)}')
    print(f'Aprobados: {(~df["failed"]).sum()}')
    print(f'Reprobados: {df["failed"].sum()}')
    print()

    # Discussion features
    disc_pca_cols = ['disc_pc1', 'disc_pc2', 'disc_pc3']
    disc_proact_cols = ['disc_mean_pct', 'disc_access_rate', 'disc_top25_rate', 'disc_median_pct']

    # 1. Correlation with final_score
    print('=' * 70)
    print('1. CORRELATION: Discussion Features vs Final Score')
    print('=' * 70)
    print()

    all_disc_cols = disc_pca_cols + disc_proact_cols
    correlations = []

    for col in all_disc_cols:
        if col in df.columns:
            valid = df[[col, 'final_score']].dropna()
            if len(valid) > 10:
                r, p = stats.pearsonr(valid[col], valid['final_score'])
                correlations.append({'feature': col, 'r': r, 'p': p})
                sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                print(f'{col:20s}: r = {r:+.3f} (p = {p:.4f}){sig}')

    print()

    # 2. Aprobados vs Reprobados comparison
    print('=' * 70)
    print('2. COMPARISON: Aprobados vs Reprobados on Discussion Features')
    print('=' * 70)
    print()

    print(f'{"Feature":25s} {"Aprobados":>12s} {"Reprobados":>12s} {"Diff":>10s} {"Cohen d":>10s} {"p-value":>10s}')
    print('-' * 85)

    comparison_results = []

    for col in all_disc_cols:
        if col not in df.columns:
            continue

        aprobados = df[~df['failed']][col].dropna()
        reprobados = df[df['failed']][col].dropna()

        if len(aprobados) < 5 or len(reprobados) < 5:
            continue

        mean_a = aprobados.mean()
        mean_r = reprobados.mean()
        diff = mean_a - mean_r
        d = cohens_d(aprobados, reprobados)

        # Mann-Whitney U test
        stat, p = stats.mannwhitneyu(aprobados, reprobados, alternative='two-sided')

        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        print(f'{col:25s} {mean_a:12.3f} {mean_r:12.3f} {diff:+10.3f} {d:+10.2f} {p:10.4f}{sig}')

        comparison_results.append({
            'feature': col,
            'mean_aprobados': mean_a,
            'mean_reprobados': mean_r,
            'cohens_d': d,
            'p_value': p
        })

    print()
    print('Significance: * p<0.05, ** p<0.01, *** p<0.001')
    print('Effect size: |d| > 0.2 small, |d| > 0.5 medium, |d| > 0.8 large')
    print()

    # 3. Per-course analysis
    print('=' * 70)
    print('3. PER-COURSE: Discussion-Grade Relationship')
    print('=' * 70)
    print()

    course_results = []
    courses = df['course_id'].unique()

    print(f'{"Course ID":>12s} {"N Students":>12s} {"N Resources":>12s} {"r(disc_pc1)":>12s} {"r(mean_pct)":>12s}')
    print('-' * 65)

    for course_id in sorted(courses):
        course_df = df[df['course_id'] == course_id]

        if len(course_df) < 10:
            continue

        n_students = len(course_df)
        n_resources = course_df['disc_n_resources'].iloc[0] if 'disc_n_resources' in course_df.columns else 0

        # Correlation with disc_pc1
        r_pc1 = 0
        if 'disc_pc1' in course_df.columns:
            valid = course_df[['disc_pc1', 'final_score']].dropna()
            if len(valid) > 5:
                r_pc1, _ = stats.pearsonr(valid['disc_pc1'], valid['final_score'])

        # Correlation with disc_mean_pct
        r_mean = 0
        if 'disc_mean_pct' in course_df.columns:
            valid = course_df[['disc_mean_pct', 'final_score']].dropna()
            if len(valid) > 5:
                r_mean, _ = stats.pearsonr(valid['disc_mean_pct'], valid['final_score'])

        print(f'{course_id:12.0f} {n_students:12d} {n_resources:12.0f} {r_pc1:+12.3f} {r_mean:+12.3f}')

        course_results.append({
            'course_id': course_id,
            'n_students': n_students,
            'n_resources': n_resources,
            'r_disc_pc1': r_pc1,
            'r_disc_mean_pct': r_mean
        })

    print()

    # 4. Create visualizations
    print('=' * 70)
    print('4. VISUALIZATIONS')
    print('=' * 70)
    print()

    # Plot 1: Discussion PCA vs Final Score (scatter)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, col in enumerate(disc_pca_cols):
        if col in df.columns:
            ax = axes[i]
            colors = ['green' if not failed else 'red' for failed in df['failed']]
            ax.scatter(df[col], df['final_score'], c=colors, alpha=0.5, s=30)
            ax.set_xlabel(col)
            ax.set_ylabel('Final Score (%)')
            ax.set_title(f'{col} vs Final Score')
            ax.axhline(y=57, color='gray', linestyle='--', alpha=0.5, label='Pass threshold')

            # Add correlation
            valid = df[[col, 'final_score']].dropna()
            r, _ = stats.pearsonr(valid[col], valid['final_score'])
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'discussion_pca_vs_score.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: discussion_pca_vs_score.png')

    # Plot 2: Aprobados vs Reprobados boxplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    plot_cols = ['disc_mean_pct', 'disc_access_rate', 'disc_top25_rate', 'disc_pc1']

    for i, col in enumerate(plot_cols):
        if col in df.columns:
            ax = axes[i]
            aprobados = df[~df['failed']][col].dropna()
            reprobados = df[df['failed']][col].dropna()

            bp = ax.boxplot([aprobados, reprobados], tick_labels=['Aprobados', 'Reprobados'],
                           patch_artist=True)
            bp['boxes'][0].set_facecolor('lightgreen')
            bp['boxes'][1].set_facecolor('lightcoral')

            ax.set_ylabel(col)
            ax.set_title(f'{col}\nAprobados vs Reprobados')

            # Add effect size
            d = cohens_d(aprobados, reprobados)
            ax.text(0.95, 0.95, f"Cohen's d = {d:.2f}", transform=ax.transAxes, fontsize=10,
                   ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'discussion_aprobados_vs_reprobados.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: discussion_aprobados_vs_reprobados.png')

    # Plot 3: Per-course correlation heatmap
    course_results_df = pd.DataFrame(course_results)
    if len(course_results_df) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Bar chart of correlations
        x = range(len(course_results_df))
        width = 0.35

        bars1 = ax.bar([i - width/2 for i in x], course_results_df['r_disc_pc1'],
                       width, label='disc_pc1', color='steelblue', alpha=0.7)
        bars2 = ax.bar([i + width/2 for i in x], course_results_df['r_disc_mean_pct'],
                       width, label='disc_mean_pct', color='darkorange', alpha=0.7)

        ax.set_xlabel('Course')
        ax.set_ylabel('Correlation with Final Score')
        ax.set_title('Discussion Engagement Correlation by Course')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{int(c)}' for c in course_results_df['course_id']], rotation=45)
        ax.legend()
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'discussion_correlation_by_course.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: discussion_correlation_by_course.png')

    print()

    # Summary
    print('=' * 70)
    print('KEY FINDINGS')
    print('=' * 70)
    print()

    # Best discussion features
    if comparison_results:
        sig_features = [r for r in comparison_results if r['p_value'] < 0.05]
        if sig_features:
            print('Statistically significant discussion features (p < 0.05):')
            for r in sorted(sig_features, key=lambda x: abs(x['cohens_d']), reverse=True):
                print(f"  - {r['feature']}: Cohen's d = {r['cohens_d']:+.2f}")
        else:
            print('No statistically significant differences found in discussion features.')
    print()

    # Best courses for discussion
    if course_results:
        best_courses = [r for r in course_results if abs(r['r_disc_mean_pct']) > 0.2]
        if best_courses:
            print('Courses with strong discussion-grade relationship (|r| > 0.2):')
            for r in sorted(best_courses, key=lambda x: abs(x['r_disc_mean_pct']), reverse=True):
                print(f"  - Course {int(r['course_id'])}: r = {r['r_disc_mean_pct']:+.3f} ({r['n_resources']:.0f} discussions)")
    print()

    print(f'Visualizations saved to: {OUTPUT_DIR}/')


if __name__ == '__main__':
    main()
