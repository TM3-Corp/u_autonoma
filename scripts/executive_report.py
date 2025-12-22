#!/usr/bin/env python3
"""
Executive Report Generator - Diagnóstico de Factibilidad Técnica
Universidad Autónoma de Chile - Canvas LMS Analytics

This script generates a professional executive report with visualizations
showing course LMS design potential and prediction model performance.
"""

import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import requests
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Configuration
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import API_URL, API_TOKEN, DATA_DIR, HIGH_POTENTIAL_COURSES

# Style configuration for professional look
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'primary': '#1a5f7a',      # Dark teal
    'secondary': '#57c5b6',    # Light teal
    'accent': '#159895',       # Medium teal
    'success': '#2ecc71',      # Green
    'warning': '#f39c12',      # Orange
    'danger': '#e74c3c',       # Red
    'neutral': '#95a5a6',      # Gray
    'dark': '#2c3e50',         # Dark blue-gray
    'light': '#ecf0f1',        # Light gray
}

HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}


def fetch_course_details(course_ids):
    """Fetch detailed course information including resources count."""
    courses_data = []

    print(f"Fetching details for {len(course_ids)} courses...")

    for i, course_id in enumerate(course_ids):
        try:
            # Basic course info
            r = requests.get(
                f'{API_URL}/api/v1/courses/{course_id}',
                headers=HEADERS,
                params={'include[]': ['total_students', 'term']}
            )
            if r.status_code != 200:
                continue
            course = r.json()

            # Get modules count
            r_mod = requests.get(
                f'{API_URL}/api/v1/courses/{course_id}/modules',
                headers=HEADERS,
                params={'per_page': 100}
            )
            modules = len(r_mod.json()) if r_mod.status_code == 200 else 0

            # Get assignments count
            r_asgn = requests.get(
                f'{API_URL}/api/v1/courses/{course_id}/assignments',
                headers=HEADERS,
                params={'per_page': 100}
            )
            assignments = len(r_asgn.json()) if r_asgn.status_code == 200 else 0

            # Get pages count
            r_pages = requests.get(
                f'{API_URL}/api/v1/courses/{course_id}/pages',
                headers=HEADERS,
                params={'per_page': 100}
            )
            pages = len(r_pages.json()) if r_pages.status_code == 200 else 0

            # Get files count
            r_files = requests.get(
                f'{API_URL}/api/v1/courses/{course_id}/files',
                headers=HEADERS,
                params={'per_page': 100}
            )
            files = len(r_files.json()) if r_files.status_code == 200 else 0

            # Get quizzes count
            r_quiz = requests.get(
                f'{API_URL}/api/v1/courses/{course_id}/quizzes',
                headers=HEADERS,
                params={'per_page': 100}
            )
            quizzes = len(r_quiz.json()) if r_quiz.status_code == 200 else 0

            courses_data.append({
                'course_id': course_id,
                'name': course.get('name', 'N/A'),
                'code': course.get('course_code', 'N/A'),
                'students': course.get('total_students', 0) or 0,
                'modules': modules,
                'assignments': assignments,
                'pages': pages,
                'files': files,
                'quizzes': quizzes,
                'total_resources': modules + assignments + pages + files + quizzes,
            })

            print(f"  [{i+1}/{len(course_ids)}] {course.get('name', 'N/A')[:40]}: {modules + assignments + pages + files + quizzes} resources")

        except Exception as e:
            print(f"  Error fetching course {course_id}: {e}")
            continue

    return courses_data


def load_model_results():
    """Load prediction model results from JSON file."""
    results_path = os.path.join(DATA_DIR, 'prediction_models_results.json')
    with open(results_path, 'r') as f:
        return json.load(f)


def create_executive_summary_chart(model_results, courses_data, output_dir):
    """Create the main executive summary visualization."""

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Diagnóstico de Factibilidad Técnica\nUniversidad Autónoma de Chile - Canvas LMS Analytics',
                 fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Key Metrics Summary (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = [
        ('Cursos\nAnalizados', len(courses_data), COLORS['primary']),
        ('Estudiantes\nTotales', sum(c['students'] for c in courses_data), COLORS['secondary']),
        ('Cursos con\nModelos', model_results['courses_analyzed'], COLORS['accent']),
    ]

    bars = ax1.bar([m[0] for m in metrics], [m[1] for m in metrics],
                   color=[m[2] for m in metrics], edgecolor='white', linewidth=2)
    ax1.set_ylabel('Cantidad', fontsize=11)
    ax1.set_title('Alcance del Análisis', fontsize=13, fontweight='bold', pad=10)

    for bar, metric in zip(bars, metrics):
        height = bar.get_height()
        ax1.annotate(f'{int(height):,}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax1.set_ylim(0, max(m[1] for m in metrics) * 1.2)
    ax1.tick_params(axis='x', labelsize=10)

    # 2. Model Performance Comparison (top center)
    ax2 = fig.add_subplot(gs[0, 1])

    model_types = ['Todos los\nDatos', 'Solo\nActividad']
    r2_values = [model_results['summary']['all_data']['avg_r2'],
                 model_results['summary']['activity_only']['avg_r2']]
    f1_values = [model_results['summary']['all_data']['avg_f1'],
                 model_results['summary']['activity_only']['avg_f1']]

    x = np.arange(len(model_types))
    width = 0.35

    bars1 = ax2.bar(x - width/2, r2_values, width, label='R² (Regresión)',
                    color=COLORS['primary'], edgecolor='white', linewidth=2)
    bars2 = ax2.bar(x + width/2, f1_values, width, label='F1 (Clasificación)',
                    color=COLORS['success'], edgecolor='white', linewidth=2)

    ax2.set_ylabel('Score (0-1)', fontsize=11)
    ax2.set_title('Rendimiento de Modelos Predictivos', fontsize=13, fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_types, fontsize=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_ylim(0, 1.15)
    ax2.axhline(y=0.5, color=COLORS['warning'], linestyle='--', alpha=0.5, label='Umbral mínimo')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. Feasibility Gauge (top right)
    ax3 = fig.add_subplot(gs[0, 2])

    # Calculate overall feasibility score
    activity_r2 = model_results['summary']['activity_only']['avg_r2']
    activity_f1 = model_results['summary']['activity_only']['avg_f1']
    feasibility_score = (activity_r2 * 0.4 + activity_f1 * 0.6) * 100

    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1

    # Background arc (gray)
    ax3.plot(r * np.cos(theta), r * np.sin(theta), color=COLORS['light'], linewidth=20, solid_capstyle='round')

    # Score arc (colored based on score)
    score_color = COLORS['success'] if feasibility_score >= 70 else COLORS['warning'] if feasibility_score >= 50 else COLORS['danger']
    score_theta = np.linspace(0, np.pi * (feasibility_score / 100), 100)
    ax3.plot(r * np.cos(score_theta), r * np.sin(score_theta), color=score_color, linewidth=20, solid_capstyle='round')

    ax3.text(0, 0.3, f'{feasibility_score:.0f}%', ha='center', va='center',
             fontsize=32, fontweight='bold', color=score_color)
    ax3.text(0, -0.1, 'Score de\nFactibilidad', ha='center', va='center',
             fontsize=12, color=COLORS['dark'])

    ax3.set_xlim(-1.3, 1.3)
    ax3.set_ylim(-0.3, 1.3)
    ax3.axis('off')
    ax3.set_title('Factibilidad Técnica', fontsize=13, fontweight='bold', pad=10)

    # 4. Course Resources Distribution (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])

    if courses_data:
        resource_types = ['Módulos', 'Tareas', 'Páginas', 'Archivos', 'Quizzes']
        resource_totals = [
            sum(c['modules'] for c in courses_data),
            sum(c['assignments'] for c in courses_data),
            sum(c['pages'] for c in courses_data),
            sum(c['files'] for c in courses_data),
            sum(c['quizzes'] for c in courses_data),
        ]
        colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning'], COLORS['success']]

        wedges, texts, autotexts = ax4.pie(resource_totals, labels=resource_types, autopct='%1.0f%%',
                                            colors=colors, startangle=90,
                                            explode=[0.02]*5, shadow=False,
                                            textprops={'fontsize': 10})
        ax4.set_title('Distribución de Recursos LMS', fontsize=13, fontweight='bold', pad=10)

    # 5. Top Courses by LMS Design (bottom center + right)
    ax5 = fig.add_subplot(gs[1, 1:])

    if courses_data:
        # Sort by total resources and get top 10
        sorted_courses = sorted(courses_data, key=lambda x: x['total_resources'], reverse=True)[:10]

        course_names = [c['name'][:30] + '...' if len(c['name']) > 30 else c['name'] for c in sorted_courses]
        resources = [c['total_resources'] for c in sorted_courses]
        students = [c['students'] for c in sorted_courses]

        y_pos = np.arange(len(course_names))

        bars = ax5.barh(y_pos, resources, color=COLORS['primary'], edgecolor='white', linewidth=1, alpha=0.8)

        # Add student count as text
        for i, (bar, student_count) in enumerate(zip(bars, students)):
            width = bar.get_width()
            ax5.text(width + 5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)} recursos | {student_count} estudiantes',
                    ha='left', va='center', fontsize=9, color=COLORS['dark'])

        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(course_names, fontsize=9)
        ax5.invert_yaxis()
        ax5.set_xlabel('Total de Recursos', fontsize=11)
        ax5.set_title('Top 10 Cursos por Diseño Instruccional', fontsize=13, fontweight='bold', pad=10)
        ax5.set_xlim(0, max(resources) * 1.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_path = os.path.join(output_dir, 'executive_summary.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_model_performance_detail(model_results, output_dir):
    """Create detailed model performance visualization."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis Detallado del Rendimiento de Modelos Predictivos',
                 fontsize=16, fontweight='bold', color=COLORS['dark'], y=0.98)

    results = model_results['results']

    # 1. R² by Course - Both Models
    ax1 = axes[0, 0]
    course_names = [r['course_name'][:20] + '...' if len(r['course_name']) > 20 else r['course_name']
                   for r in results]

    # Get best R² for each model type (RF usually better)
    all_data_r2 = []
    activity_r2 = []
    for r in results:
        ad = r.get('all_data_model', {}).get('regression', {})
        ao = r.get('activity_only_model', {}).get('regression', {})
        all_data_r2.append(max(ad.get('linear', {}).get('r2', 0), ad.get('random_forest', {}).get('r2', 0)))
        activity_r2.append(max(ao.get('linear', {}).get('r2', 0), ao.get('random_forest', {}).get('r2', 0)))

    x = np.arange(len(course_names))
    width = 0.35

    bars1 = ax1.bar(x - width/2, all_data_r2, width, label='Todos los Datos', color=COLORS['primary'])
    bars2 = ax1.bar(x + width/2, activity_r2, width, label='Solo Actividad', color=COLORS['secondary'])

    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('R² por Curso (Predicción de Nota)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(course_names, rotation=45, ha='right', fontsize=8)
    ax1.legend(fontsize=9)
    ax1.axhline(y=0.5, color=COLORS['danger'], linestyle='--', alpha=0.7, linewidth=1)
    ax1.text(len(course_names)-0.5, 0.52, 'Umbral R²=0.5', fontsize=8, color=COLORS['danger'])

    # 2. Pass Rate vs Model Performance
    ax2 = axes[0, 1]
    pass_rates = [r['pass_rate'] * 100 for r in results]

    scatter1 = ax2.scatter(pass_rates, activity_r2, s=150, c=COLORS['secondary'],
                           label='Solo Actividad', edgecolors='white', linewidth=2, alpha=0.8)
    scatter2 = ax2.scatter(pass_rates, all_data_r2, s=150, c=COLORS['primary'],
                           label='Todos los Datos', edgecolors='white', linewidth=2, alpha=0.8, marker='s')

    ax2.set_xlabel('Tasa de Aprobación (%)', fontsize=11)
    ax2.set_ylabel('R² Score', fontsize=11)
    ax2.set_title('Relación: Tasa Aprobación vs R²', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.axhline(y=0.5, color=COLORS['danger'], linestyle='--', alpha=0.5)

    # Add annotations for each point
    for i, name in enumerate(course_names):
        ax2.annotate(name[:10], (pass_rates[i], activity_r2[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7, alpha=0.7)

    # 3. Feature Importance (Activity Only)
    ax3 = axes[1, 0]

    # Aggregate feature importance across all courses
    feature_importance = {}
    for r in results:
        ao_model = r.get('activity_only_model', {}).get('regression', {}).get('random_forest', {})
        for fi in ao_model.get('feature_importance', []):
            feat = fi['feature']
            imp = fi['importance']
            if feat in feature_importance:
                feature_importance[feat].append(imp)
            else:
                feature_importance[feat] = [imp]

    # Calculate average importance
    avg_importance = {k: np.mean(v) for k, v in feature_importance.items()}
    sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)

    feat_names = [f[0].replace('_', ' ').title() for f in sorted_features]
    feat_values = [f[1] for f in sorted_features]

    colors_bar = [COLORS['primary'] if v > 0.2 else COLORS['secondary'] if v > 0.1 else COLORS['neutral']
                  for v in feat_values]

    bars = ax3.barh(range(len(feat_names)), feat_values, color=colors_bar, edgecolor='white')
    ax3.set_yticks(range(len(feat_names)))
    ax3.set_yticklabels(feat_names, fontsize=10)
    ax3.invert_yaxis()
    ax3.set_xlabel('Importancia Promedio', fontsize=11)
    ax3.set_title('Importancia de Features (Solo Actividad)', fontsize=12, fontweight='bold')

    # 4. Students Distribution in Analyzed Courses
    ax4 = axes[1, 1]

    students = [r['n_students'] for r in results]
    course_short = [r['course_name'][:15] + '...' if len(r['course_name']) > 15 else r['course_name']
                   for r in results]

    colors_students = [COLORS['success'] if s > 40 else COLORS['warning'] if s > 25 else COLORS['neutral']
                      for s in students]

    bars = ax4.bar(range(len(course_short)), students, color=colors_students, edgecolor='white', linewidth=1)
    ax4.set_xticks(range(len(course_short)))
    ax4.set_xticklabels(course_short, rotation=45, ha='right', fontsize=8)
    ax4.set_ylabel('Número de Estudiantes', fontsize=11)
    ax4.set_title('Estudiantes por Curso Analizado', fontsize=12, fontweight='bold')
    ax4.axhline(y=30, color=COLORS['danger'], linestyle='--', alpha=0.5)
    ax4.text(len(course_short)-0.5, 32, 'Mínimo recomendado', fontsize=8, color=COLORS['danger'])

    for bar, student in zip(bars, students):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(student), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(output_dir, 'model_performance_detail.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_lms_design_analysis(courses_data, output_dir):
    """Create LMS design quality analysis visualization."""

    if not courses_data:
        print("No course data available for LMS design analysis")
        return None

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Análisis de Diseño Instruccional en Canvas LMS',
                 fontsize=16, fontweight='bold', color=COLORS['dark'], y=0.98)

    # 1. Resource Distribution Heatmap
    ax1 = axes[0, 0]

    # Create matrix for heatmap
    sorted_courses = sorted(courses_data, key=lambda x: x['total_resources'], reverse=True)[:15]
    course_names = [c['name'][:20] + '...' if len(c['name']) > 20 else c['name'] for c in sorted_courses]

    resource_matrix = np.array([
        [c['modules'], c['assignments'], c['pages'], c['files'], c['quizzes']]
        for c in sorted_courses
    ])

    # Normalize for better visualization (cap at 100)
    resource_matrix_norm = np.minimum(resource_matrix, 100)

    im = ax1.imshow(resource_matrix_norm, cmap='YlGnBu', aspect='auto')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(['Módulos', 'Tareas', 'Páginas', 'Archivos', 'Quizzes'], fontsize=9)
    ax1.set_yticks(range(len(course_names)))
    ax1.set_yticklabels(course_names, fontsize=8)
    ax1.set_title('Mapa de Recursos por Curso', fontsize=12, fontweight='bold')

    # Add text annotations
    for i in range(len(course_names)):
        for j in range(5):
            val = resource_matrix[i, j]
            color = 'white' if resource_matrix_norm[i, j] > 50 else 'black'
            ax1.text(j, i, str(val), ha='center', va='center', fontsize=8, color=color)

    plt.colorbar(im, ax=ax1, label='Cantidad')

    # 2. Course Quality Quadrant
    ax2 = axes[0, 1]

    resources = [c['total_resources'] for c in courses_data]
    students = [c['students'] for c in courses_data]

    ax2.scatter(resources, students, s=100, c=COLORS['primary'], alpha=0.7, edgecolors='white', linewidth=1)

    # Add quadrant lines
    median_resources = np.median(resources)
    median_students = np.median(students)
    ax2.axvline(x=median_resources, color=COLORS['neutral'], linestyle='--', alpha=0.5)
    ax2.axhline(y=median_students, color=COLORS['neutral'], linestyle='--', alpha=0.5)

    # Add quadrant labels
    ax2.text(max(resources)*0.9, max(students)*0.9, 'ALTO POTENCIAL', fontsize=10,
             ha='right', color=COLORS['success'], fontweight='bold')
    ax2.text(min(resources)*1.1, max(students)*0.9, 'Necesita\nContenido', fontsize=9,
             ha='left', color=COLORS['warning'])
    ax2.text(max(resources)*0.9, min(students)+2, 'Bajo\nAlcance', fontsize=9,
             ha='right', color=COLORS['warning'])
    ax2.text(min(resources)*1.1, min(students)+2, 'Bajo\nPotencial', fontsize=9,
             ha='left', color=COLORS['neutral'])

    ax2.set_xlabel('Total Recursos LMS', fontsize=11)
    ax2.set_ylabel('Estudiantes Inscritos', fontsize=11)
    ax2.set_title('Cuadrante de Calidad de Cursos', fontsize=12, fontweight='bold')

    # 3. Resource Type Comparison
    ax3 = axes[1, 0]

    resource_types = ['Módulos', 'Tareas', 'Páginas', 'Archivos', 'Quizzes']
    avg_resources = [
        np.mean([c['modules'] for c in courses_data]),
        np.mean([c['assignments'] for c in courses_data]),
        np.mean([c['pages'] for c in courses_data]),
        np.mean([c['files'] for c in courses_data]),
        np.mean([c['quizzes'] for c in courses_data]),
    ]

    colors = [COLORS['primary'], COLORS['secondary'], COLORS['accent'], COLORS['warning'], COLORS['success']]
    bars = ax3.bar(resource_types, avg_resources, color=colors, edgecolor='white', linewidth=2)

    for bar, val in zip(bars, avg_resources):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('Promedio por Curso', fontsize=11)
    ax3.set_title('Promedio de Recursos por Tipo', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', labelsize=10)

    # 4. Course Readiness Assessment
    ax4 = axes[1, 1]

    # Calculate readiness score for each course
    def calculate_readiness(course):
        score = 0
        if course['modules'] >= 5: score += 25
        elif course['modules'] >= 3: score += 15
        if course['assignments'] >= 10: score += 25
        elif course['assignments'] >= 5: score += 15
        if course['pages'] >= 20: score += 20
        elif course['pages'] >= 10: score += 10
        if course['files'] >= 10: score += 15
        elif course['files'] >= 5: score += 8
        if course['quizzes'] >= 3: score += 15
        elif course['quizzes'] >= 1: score += 8
        return min(score, 100)

    readiness_scores = [calculate_readiness(c) for c in courses_data]

    # Categorize
    high_ready = sum(1 for s in readiness_scores if s >= 70)
    medium_ready = sum(1 for s in readiness_scores if 40 <= s < 70)
    low_ready = sum(1 for s in readiness_scores if s < 40)

    categories = ['Alto\n(≥70)', 'Medio\n(40-69)', 'Bajo\n(<40)']
    counts = [high_ready, medium_ready, low_ready]
    colors_ready = [COLORS['success'], COLORS['warning'], COLORS['danger']]

    bars = ax4.bar(categories, counts, color=colors_ready, edgecolor='white', linewidth=2)

    for bar, count in zip(bars, counts):
        pct = (count / len(courses_data)) * 100
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{count}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax4.set_ylabel('Número de Cursos', fontsize=11)
    ax4.set_title('Nivel de Preparación para Analytics', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', labelsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = os.path.join(output_dir, 'lms_design_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def create_conclusion_chart(model_results, courses_data, output_dir):
    """Create conclusion and recommendations visualization."""

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle('Conclusiones y Recomendaciones',
                 fontsize=18, fontweight='bold', color=COLORS['dark'], y=0.98)

    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.3)

    # Calculate key metrics
    activity_r2 = model_results['summary']['activity_only']['avg_r2']
    activity_f1 = model_results['summary']['activity_only']['avg_f1']
    total_students = sum(c['students'] for c in courses_data) if courses_data else 0
    high_potential = sum(1 for c in courses_data if c['total_resources'] >= 50 and c['students'] > 0) if courses_data else 0

    # 1. Key Finding Icons (top row - span all columns)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.axis('off')

    findings = [
        ('FACTIBILIDAD\nCONFIRMADA', f'R² = {activity_r2:.2f}', COLORS['success'], '[OK]'),
        ('PREDICCIÓN\nTEMPRANA', f'F1 = {activity_f1:.2f}', COLORS['primary'], '[>>]'),
        ('ALCANCE\nPOTENCIAL', f'{total_students}+ estudiantes', COLORS['secondary'], '[++]'),
        ('CURSOS\nPREPARADOS', f'{high_potential} cursos', COLORS['accent'], '[##]'),
    ]

    for i, (title, value, color, icon) in enumerate(findings):
        x_pos = 0.125 + i * 0.25

        # Box background
        rect = mpatches.FancyBboxPatch((x_pos - 0.1, 0.2), 0.2, 0.7,
                                        boxstyle="round,pad=0.02",
                                        facecolor=color, alpha=0.1,
                                        edgecolor=color, linewidth=2,
                                        transform=ax1.transAxes)
        ax1.add_patch(rect)

        ax1.text(x_pos, 0.75, icon, ha='center', va='center', fontsize=28, transform=ax1.transAxes)
        ax1.text(x_pos, 0.5, title, ha='center', va='center', fontsize=11, fontweight='bold',
                color=color, transform=ax1.transAxes)
        ax1.text(x_pos, 0.32, value, ha='center', va='center', fontsize=13, fontweight='bold',
                color=COLORS['dark'], transform=ax1.transAxes)

    # 2. Next Steps (bottom left)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.axis('off')

    steps = [
        '1. Expandir análisis a POSTGRADO',
        '2. Implementar pipeline de\n   predicción en tiempo real',
        '3. Desarrollar dashboard\n   para profesores',
        '4. Integrar alertas tempranas\n   para estudiantes en riesgo',
    ]

    ax2.text(0.5, 0.95, 'Próximos Pasos', ha='center', va='top', fontsize=13,
             fontweight='bold', color=COLORS['dark'], transform=ax2.transAxes)

    for i, step in enumerate(steps):
        ax2.text(0.1, 0.78 - i*0.22, step, ha='left', va='top', fontsize=10,
                color=COLORS['dark'], transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor=COLORS['light'], edgecolor='none', alpha=0.5))

    # 3. Value Proposition (bottom center)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis('off')

    ax3.text(0.5, 0.95, 'Propuesta de Valor', ha='center', va='top', fontsize=13,
             fontweight='bold', color=COLORS['dark'], transform=ax3.transAxes)

    value_props = [
        ('Detección Temprana', 'Identificar estudiantes\nen riesgo ANTES del\nprimer examen'),
        ('Intervención Oportuna', 'Permitir acciones\npreventivas por parte\nde profesores y tutores'),
        ('Mejora Continua', 'Feedback sobre diseño\ninstruccional efectivo'),
    ]

    for i, (title, desc) in enumerate(value_props):
        y_pos = 0.75 - i * 0.28
        ax3.text(0.15, y_pos, '●', ha='center', va='center', fontsize=16,
                color=COLORS['primary'], transform=ax3.transAxes)
        ax3.text(0.25, y_pos, title, ha='left', va='center', fontsize=11,
                fontweight='bold', color=COLORS['primary'], transform=ax3.transAxes)
        ax3.text(0.25, y_pos - 0.12, desc, ha='left', va='center', fontsize=9,
                color=COLORS['dark'], transform=ax3.transAxes)

    # 4. Contact & Credits (bottom right)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis('off')

    ax4.text(0.5, 0.95, 'Información', ha='center', va='top', fontsize=13,
             fontweight='bold', color=COLORS['dark'], transform=ax4.transAxes)

    info_text = f"""
Fecha: {datetime.now().strftime('%d/%m/%Y')}
Ambiente: TEST

Cursos Analizados: {len(courses_data) if courses_data else 0}
Modelos Entrenados: {model_results['courses_analyzed']}

Metodología:
• Random Forest Regression
• Logistic Classification
• Train/Test Split 80/20
"""

    ax4.text(0.1, 0.8, info_text, ha='left', va='top', fontsize=9,
             color=COLORS['dark'], transform=ax4.transAxes, family='monospace')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    output_path = os.path.join(output_dir, 'conclusions.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved: {output_path}")
    return output_path


def generate_report():
    """Main function to generate the complete executive report."""

    print("=" * 70)
    print("GENERATING EXECUTIVE REPORT")
    print("Diagnóstico de Factibilidad Técnica - Universidad Autónoma de Chile")
    print("=" * 70)
    print()

    # Create output directory
    output_dir = os.path.join(DATA_DIR, 'report')
    os.makedirs(output_dir, exist_ok=True)

    # Load model results
    print("Loading model results...")
    model_results = load_model_results()
    print(f"  Loaded results for {model_results['courses_analyzed']} courses")

    # Fetch course details from API
    print("\nFetching course details from Canvas API...")
    courses_data = fetch_course_details(HIGH_POTENTIAL_COURSES)
    print(f"  Fetched details for {len(courses_data)} courses")

    # Generate visualizations
    print("\nGenerating visualizations...")

    charts = []

    # 1. Executive Summary
    print("\n1. Creating Executive Summary...")
    chart1 = create_executive_summary_chart(model_results, courses_data, output_dir)
    if chart1:
        charts.append(chart1)

    # 2. Model Performance Detail
    print("\n2. Creating Model Performance Analysis...")
    chart2 = create_model_performance_detail(model_results, output_dir)
    if chart2:
        charts.append(chart2)

    # 3. LMS Design Analysis
    print("\n3. Creating LMS Design Analysis...")
    chart3 = create_lms_design_analysis(courses_data, output_dir)
    if chart3:
        charts.append(chart3)

    # 4. Conclusions
    print("\n4. Creating Conclusions...")
    chart4 = create_conclusion_chart(model_results, courses_data, output_dir)
    if chart4:
        charts.append(chart4)

    # Summary
    print("\n" + "=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Charts generated: {len(charts)}")
    for chart in charts:
        print(f"  - {os.path.basename(chart)}")

    # Key metrics summary
    print("\n" + "-" * 70)
    print("KEY METRICS SUMMARY")
    print("-" * 70)
    print(f"Total courses with good LMS design: {len(courses_data)}")
    print(f"Total students reached: {sum(c['students'] for c in courses_data)}")
    print(f"Courses with prediction models: {model_results['courses_analyzed']}")
    print(f"Activity-Only R² (avg): {model_results['summary']['activity_only']['avg_r2']:.3f}")
    print(f"Activity-Only F1 (avg): {model_results['summary']['activity_only']['avg_f1']:.3f}")

    feasibility = (model_results['summary']['activity_only']['avg_r2'] * 0.4 +
                   model_results['summary']['activity_only']['avg_f1'] * 0.6) * 100
    print(f"\nFEASIBILITY SCORE: {feasibility:.0f}%")

    if feasibility >= 70:
        print("STATUS: ✅ HIGH FEASIBILITY - Ready for implementation")
    elif feasibility >= 50:
        print("STATUS: ⚠️ MODERATE FEASIBILITY - Needs improvement")
    else:
        print("STATUS: ❌ LOW FEASIBILITY - Not recommended")

    return output_dir, charts


if __name__ == '__main__':
    generate_report()
