#!/usr/bin/env python3
"""
Regenerate all visualizations with consistent labeling.
All charts now use format: "Short Name (ID)" for cross-reference mapping.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

# Directories
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR = os.path.join(DATA_DIR, 'report', 'visualizations')

# Short names mapping (keep names short for readability)
SHORT_NAMES = {
    '86005': 'Comp.Dig.',
    '86676': 'Bus.Analytics',
    '84936': 'Microecon.',
    '84941': 'Microecon.',
    '84944': 'Macroecon.',
    '86020': 'Comp.Dig.',
    '79804': 'Tributarios',
    '79875': 'Comp.Dig.',
    '79913': 'Bus.Analytics',
    '88381': 'Mat.Negocios',
    '89099': 'Comp.Dig.',
    '89390': 'Gest.Talento',
    '89736': 'Macroecon.',
}


def get_label(course_id, include_name=True):
    """Generate consistent label: 'ShortName (ID)' or just 'ID'."""
    cid = str(course_id)
    if include_name:
        short_name = SHORT_NAMES.get(cid, 'Curso')
        return f"{short_name} ({cid})"
    return cid


def load_data():
    """Load all necessary data files."""
    data = {}

    # Course activity and design
    with open(os.path.join(DATA_DIR, 'course_activity_design.json')) as f:
        data['activity_design'] = json.load(f)

    # Course design detailed
    with open(os.path.join(DATA_DIR, 'course_design_detailed.json')) as f:
        data['design_detailed'] = json.load(f)

    # Student features for correlations (try multiple locations)
    features_paths = [
        os.path.join(DATA_DIR, 'engagement_dynamics', 'student_features.csv'),
        os.path.join(DATA_DIR, 'student_features.csv'),
        os.path.join(DATA_DIR, 'early_warning', 'student_features.csv'),
    ]
    for features_path in features_paths:
        if os.path.exists(features_path):
            data['student_features'] = pd.read_csv(features_path)
            print(f"  Loaded student features from: {features_path}")
            break

    # Hourly activity
    hourly_path = os.path.join(DATA_DIR, 'hourly_activity_by_course.json')
    if os.path.exists(hourly_path):
        with open(hourly_path) as f:
            data['hourly'] = json.load(f)

    return data


def create_course_design_stacked(data):
    """Create horizontal stacked bar chart for course design resources."""
    print("Creating course_design_stacked.png...")

    courses = data['activity_design']

    # Sort by total resources
    for c in courses:
        c['total'] = c['modules'] + c['assignments'] + c['quizzes'] + c['files'] + c['discussions'] + c['pages']
    courses = sorted(courses, key=lambda x: x['total'], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    labels = [get_label(c['course_id']) for c in courses]

    modules = [c['modules'] for c in courses]
    assignments = [c['assignments'] for c in courses]
    quizzes = [c['quizzes'] for c in courses]
    files = [c['files'] for c in courses]
    discussions = [c['discussions'] for c in courses]
    pages = [c['pages'] for c in courses]

    y = np.arange(len(labels))

    # Stacked bars
    ax.barh(y, modules, label='Módulos', color='#1f77b4')
    ax.barh(y, assignments, left=modules, label='Assignments', color='#ff7f0e')
    ax.barh(y, quizzes, left=np.array(modules)+np.array(assignments), label='Quizzes', color='#2ca02c')
    ax.barh(y, files, left=np.array(modules)+np.array(assignments)+np.array(quizzes), label='Files', color='#d62728')
    ax.barh(y, discussions, left=np.array(modules)+np.array(assignments)+np.array(quizzes)+np.array(files), label='Discussions', color='#9467bd')
    ax.barh(y, pages, left=np.array(modules)+np.array(assignments)+np.array(quizzes)+np.array(files)+np.array(discussions), label='Pages', color='#8c564b')

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Número de Recursos')
    ax.set_title('Composición del Diseño Instruccional por Curso')
    ax.legend(loc='lower right')

    # Add total count at end of each bar
    for i, total in enumerate([c['total'] for c in courses]):
        ax.text(total + 10, i, str(total), va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'course_design_stacked.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_resources_by_category(data):
    """Create bar chart for total resources by category."""
    print("Creating resources_by_category.png...")

    courses = data['activity_design']

    categories = ['Módulos', 'Assignments', 'Quizzes', 'Files', 'Discussions', 'Pages']
    totals = [
        sum(c['modules'] for c in courses),
        sum(c['assignments'] for c in courses),
        sum(c['quizzes'] for c in courses),
        sum(c['files'] for c in courses),
        sum(c['discussions'] for c in courses),
        sum(c['pages'] for c in courses),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = ax.bar(categories, totals, color=colors)

    ax.set_ylabel('Total de Recursos')
    ax.set_title('Distribución Total de Recursos por Categoría (13 cursos)')

    # Add value labels
    for bar, total in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(total), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'resources_by_category.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_course_activity_comparison(data):
    """Create activity comparison chart with engagement per student."""
    print("Creating course_activity_comparison.png...")

    courses = data['activity_design']

    # Sort by avg views per student
    courses = sorted(courses, key=lambda x: x['avg_views_per_student'], reverse=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    labels = [get_label(c['course_id']) for c in courses]
    y = np.arange(len(labels))

    # Left: Total views
    views = [c['total_views'] for c in courses]
    ax1.barh(y, views, color='#3498db')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Total de Visualizaciones')
    ax1.set_title('Actividad Total por Curso')
    for i, v in enumerate(views):
        ax1.text(v + 200, i, f'{v:,}', va='center', fontsize=8)

    # Right: Avg views per student
    avg_views = [c['avg_views_per_student'] for c in courses]
    ax2.barh(y, avg_views, color='#2ecc71')
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel('Visualizaciones Promedio por Estudiante')
    ax2.set_title('Engagement por Estudiante')
    for i, v in enumerate(avg_views):
        ax2.text(v + 10, i, f'{v:.0f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'course_activity_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_design_vs_engagement(data):
    """Create scatter plot of design complexity vs engagement."""
    print("Creating design_vs_engagement.png...")

    courses = data['activity_design']

    fig, ax = plt.subplots(figsize=(12, 8))

    for c in courses:
        total_resources = c['modules'] + c['assignments'] + c['quizzes'] + c['files'] + c['discussions'] + c['pages']
        avg_views = c['avg_views_per_student']

        ax.scatter(total_resources, avg_views, s=c['students']*5, alpha=0.6)
        ax.annotate(get_label(c['course_id']), (total_resources, avg_views),
                   fontsize=8, ha='left', va='bottom')

    ax.set_xlabel('Total de Recursos (Complejidad del Diseño)')
    ax.set_ylabel('Visualizaciones Promedio por Estudiante (Engagement)')
    ax.set_title('Relación entre Diseño Instruccional y Engagement Estudiantil\n(tamaño del punto = número de estudiantes)')

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'design_vs_engagement.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_hourly_heatmaps(data):
    """Recreate hourly heatmaps with ID-only labels."""
    print("Creating hourly heatmaps...")

    if 'hourly' not in data:
        print("  No hourly data found, skipping...")
        return

    hourly = data['hourly']

    # Custom colormap: white to dark blue
    colors = ['#ffffff', '#e6f2ff', '#cce5ff', '#99ccff', '#66b3ff',
              '#3399ff', '#0080ff', '#0066cc', '#004d99', '#003366']
    cmap = mcolors.LinearSegmentedColormap.from_list('white_blue', colors)

    days = ['L', 'M', 'X', 'J', 'V', 'S', 'D']

    # Create individual heatmaps (ID only in title)
    for course_id, matrix in hourly.items():
        matrix = np.array(matrix)
        data_T = matrix.T  # 24 hours x 7 days

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(data_T, cmap=cmap, aspect='auto')

        plt.colorbar(im, ax=ax, label='Interacciones')

        ax.set_xticks(range(7))
        ax.set_xticklabels(['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom'])
        ax.set_yticks(range(24))
        ax.set_yticklabels([f'{h:02d}:00' for h in range(24)])

        ax.set_xlabel('Día de la Semana')
        ax.set_ylabel('Hora del Día')
        ax.set_title(f'Patrón de Actividad - Curso {course_id}')

        # Add text annotations
        max_val = data_T.max()
        for i in range(24):
            for j in range(7):
                value = int(data_T[i, j])
                if value > 0:
                    text_color = 'white' if value > max_val * 0.5 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center',
                           color=text_color, fontsize=7)

        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, f'hourly_heatmap_{course_id}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Create combined heatmap with IDs only
    n_courses = len(hourly)
    cols = 3
    rows = (n_courses + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten()

    for idx, (course_id, matrix) in enumerate(hourly.items()):
        ax = axes[idx]
        matrix = np.array(matrix)
        data_T = matrix.T

        im = ax.imshow(data_T, cmap=cmap, aspect='auto')

        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_yticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'])

        ax.set_title(f'Curso {course_id}', fontsize=10)

        # Add numbers
        max_val = data_T.max()
        for i in range(24):
            for j in range(7):
                value = int(data_T[i, j])
                if value > 0:
                    text_color = 'white' if value > max_val * 0.5 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center',
                           color=text_color, fontsize=5)

    # Hide empty subplots
    for idx in range(len(hourly), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Patrones de Actividad Estudiantil por Curso (Hora del Día vs Día de la Semana)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'hourly_heatmaps_combined.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_correlation_heatmap(data):
    """Recreate correlation heatmap with course IDs."""
    print("Creating correlation_heatmap.png...")

    if 'student_features' not in data:
        print("  No student features found, skipping...")
        return

    df = data['student_features']

    # Get courses with valid grades
    good_courses = df.groupby('course_id').agg({
        'final_score': ['count', 'std']
    })
    good_courses.columns = ['count', 'std']
    good_courses = good_courses[(good_courses['count'] >= 10) & (good_courses['std'] > 5)]
    good_courses = good_courses.index.tolist()

    if len(good_courses) == 0:
        print("  No courses with valid grade variance, skipping...")
        return

    # Get all numeric features (excluding identifiers)
    exclude_cols = ['course_id', 'user_id', 'final_score', 'failed']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]

    # Calculate correlations for each course
    correlations = {}
    for course_id in good_courses:
        course_df = df[df['course_id'] == course_id]
        corr_values = {}
        for feat in feature_cols:
            if feat in course_df.columns:
                valid = course_df[[feat, 'final_score']].dropna()
                if len(valid) >= 5:
                    try:
                        r, p = stats.pearsonr(valid[feat], valid['final_score'])
                        if not np.isnan(r):
                            corr_values[feat] = r
                    except:
                        pass
        correlations[course_id] = corr_values

    # Get top 5 features per course
    top_features = set()
    for course_id, corrs in correlations.items():
        sorted_corrs = sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        for feat, _ in sorted_corrs:
            top_features.add(feat)

    features = sorted(list(top_features))

    if len(features) == 0:
        print("  No significant features found, skipping...")
        return

    # Build correlation matrix
    matrix = np.zeros((len(good_courses), len(features)))
    for i, course_id in enumerate(good_courses):
        for j, feat in enumerate(features):
            matrix[i, j] = correlations.get(course_id, {}).get(feat, 0)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label='Correlación de Pearson')

    # Labels with course IDs only
    ax.set_yticks(range(len(good_courses)))
    ax.set_yticklabels([f'Curso {cid}' for cid in good_courses], fontsize=9)

    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, rotation=45, ha='right', fontsize=8)

    ax.set_title('Correlación de Features con Calificación Final por Curso')
    ax.set_xlabel('Features')
    ax.set_ylabel('Curso')

    # Add correlation values
    for i in range(len(good_courses)):
        for j in range(len(features)):
            val = matrix[i, j]
            if abs(val) > 0.01:
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_grade_boxplot(data):
    """Recreate grade boxplot with course IDs."""
    print("Creating grade_boxplot.png...")

    if 'student_features' not in data:
        print("  No student features found, skipping...")
        return

    df = data['student_features']

    # Get courses with grades
    courses_with_grades = df[df['final_score'].notna()]['course_id'].unique()

    fig, ax = plt.subplots(figsize=(14, 6))

    box_data = []
    labels = []
    for cid in sorted(courses_with_grades):
        scores = df[df['course_id'] == cid]['final_score'].dropna()
        if len(scores) >= 5:
            box_data.append(scores)
            labels.append(get_label(cid))

    if not box_data:
        print("  No courses with enough grade data, skipping...")
        return

    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)

    # Color boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(box_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=57, color='r', linestyle='--', label='Umbral aprobación (57%)')
    ax.set_ylabel('Calificación Final (%)')
    ax.set_xlabel('Curso')
    ax.set_title('Distribución de Calificaciones por Curso')
    ax.legend()

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'grade_boxplot.png'), dpi=150, bbox_inches='tight')
    plt.close()


def create_pass_rate_bars(data):
    """Recreate pass rate bars with course IDs."""
    print("Creating pass_rate_bars.png...")

    if 'student_features' not in data:
        print("  No student features found, skipping...")
        return

    df = data['student_features']

    # Calculate pass rates
    pass_rates = []
    for cid in df['course_id'].unique():
        course_df = df[df['course_id'] == cid]
        if 'failed' in course_df.columns:
            total = len(course_df[course_df['failed'].notna()])
            if total >= 5:
                passed = len(course_df[course_df['failed'] == 0])
                pass_rates.append({
                    'course_id': cid,
                    'pass_rate': (passed / total) * 100,
                    'total': total
                })

    if not pass_rates:
        print("  No courses with pass/fail data, skipping...")
        return

    pass_rates = sorted(pass_rates, key=lambda x: x['pass_rate'], reverse=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    labels = [get_label(p['course_id']) for p in pass_rates]
    rates = [p['pass_rate'] for p in pass_rates]

    colors = ['#2ecc71' if r >= 60 else '#e74c3c' if r < 40 else '#f39c12' for r in rates]

    bars = ax.bar(range(len(labels)), rates, color=colors)

    ax.axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Objetivo 60%')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Tasa de Aprobación (%)')
    ax.set_title('Tasa de Aprobación por Curso')
    ax.set_ylim(0, 100)
    ax.legend()

    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.1f}%', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(VIZ_DIR, 'pass_rate_bars.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    print("=" * 80)
    print("REGENERATING ALL VISUALIZATIONS WITH CONSISTENT LABELS")
    print("=" * 80)

    os.makedirs(VIZ_DIR, exist_ok=True)

    data = load_data()

    # Design and activity charts
    create_course_design_stacked(data)
    create_resources_by_category(data)
    create_course_activity_comparison(data)
    create_design_vs_engagement(data)

    # Hourly heatmaps
    create_hourly_heatmaps(data)

    # Analytics charts
    create_correlation_heatmap(data)
    create_grade_boxplot(data)
    create_pass_rate_bars(data)

    print("\n" + "=" * 80)
    print("DONE! All visualizations regenerated with consistent labels.")
    print("Format: 'ShortName (CourseID)' for name+ID charts")
    print("Format: 'Curso ID' for ID-only charts (heatmaps)")
    print("=" * 80)


if __name__ == '__main__':
    main()
