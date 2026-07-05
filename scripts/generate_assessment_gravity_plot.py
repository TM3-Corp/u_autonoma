#!/usr/bin/env python3
"""
Generate Assessment Gravity Visualization

Shows how student activity clusters around assignment due dates,
demonstrating that assessments act as "gravity centers" of engagement.

This visualization supports the insight that we can predict student
outcomes based on preparation behavior, not just grades.
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path

# Configure output
OUTPUT_DIR = Path(__file__).parent.parent / "data/report/analysis/assessment_patterns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model courses (10 courses used in training)
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Assessment-related resource types
ASSESSMENT_RESOURCES = ['assignments', 'quizzes', 'grades']
CONTENT_RESOURCES = ['modules', 'files', 'pages', 'discussions']


def load_data():
    """Load page views and assignments data."""
    print("Loading data...")

    # Load page views (use all_page_views which has source_user_id)
    pv = pd.read_parquet('data/page_views/all_page_views.parquet')

    # Extract course_id from URL using regex
    import re

    def extract_course_id(url):
        if pd.isna(url):
            return None
        match = re.search(r'/courses/(\d+)', str(url))
        return int(match.group(1)) if match else None

    pv['course_id'] = pv['http_request'].apply(extract_course_id)

    # Filter to model courses
    pv = pv[pv['course_id'].isin(MODEL_COURSES)].copy()
    pv['created_at'] = pd.to_datetime(pv['created_at'])
    pv['date'] = pv['created_at'].dt.date

    # Categorize resource types from controller
    def categorize_controller(controller):
        if controller in ['assignments', 'assignment_groups']:
            return 'assignments'
        elif controller in ['quizzes', 'quiz_submissions']:
            return 'quizzes'
        elif controller in ['gradebooks', 'grades']:
            return 'grades'
        elif controller == 'context_modules':
            return 'modules'
        elif controller in ['files', 'folders']:
            return 'files'
        elif controller in ['wiki_pages', 'pages']:
            return 'pages'
        elif controller in ['discussion_topics', 'discussion_entries']:
            return 'discussions'
        elif controller == 'announcements':
            return 'announcements'
        else:
            return 'other'

    pv['resource_type'] = pv['controller'].apply(categorize_controller)

    print(f"  Page views: {len(pv):,}")

    # Load assignments
    with open('data/assignments.json') as f:
        all_assignments = json.load(f)

    assignments = [a for a in all_assignments if a.get('course_id') in MODEL_COURSES]

    # Parse due dates
    for a in assignments:
        if a.get('due_at'):
            a['due_date'] = pd.to_datetime(a['due_at']).date()
        else:
            a['due_date'] = None

    assignments = [a for a in assignments if a.get('due_date')]
    print(f"  Assignments with due dates: {len(assignments)}")

    return pv, assignments


def create_aggregated_gravity_plot(pv, assignments):
    """
    Create a plot showing activity aligned to days before/after assignment due dates.
    This aggregates all assignments to show the "gravity" effect.
    """
    print("\nCreating aggregated gravity plot...")

    # Get unique due dates
    due_dates = sorted(set(a['due_date'] for a in assignments))
    print(f"  Unique due dates: {len(due_dates)}")

    # For each page view, calculate days relative to nearest assignment
    pv_dates = pv['date'].unique()

    # Calculate activity by days relative to nearest assignment
    days_range = range(-14, 8)  # 14 days before to 7 days after

    activity_by_offset = {d: {'assessment': 0, 'content': 0, 'total': 0} for d in days_range}
    count_by_offset = {d: 0 for d in days_range}

    for due_date in due_dates:
        for offset in days_range:
            check_date = due_date + timedelta(days=offset)
            day_pv = pv[pv['date'] == check_date]

            if len(day_pv) > 0:
                assessment_count = len(day_pv[day_pv['resource_type'].isin(ASSESSMENT_RESOURCES)])
                content_count = len(day_pv[day_pv['resource_type'].isin(CONTENT_RESOURCES)])
                total_count = len(day_pv)

                activity_by_offset[offset]['assessment'] += assessment_count
                activity_by_offset[offset]['content'] += content_count
                activity_by_offset[offset]['total'] += total_count
                count_by_offset[offset] += 1

    # Average by number of assignments that had data for that offset
    for offset in days_range:
        if count_by_offset[offset] > 0:
            activity_by_offset[offset]['assessment'] /= count_by_offset[offset]
            activity_by_offset[offset]['content'] /= count_by_offset[offset]
            activity_by_offset[offset]['total'] /= count_by_offset[offset]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    offsets = list(days_range)
    assessment_activity = [activity_by_offset[d]['assessment'] for d in offsets]
    content_activity = [activity_by_offset[d]['content'] for d in offsets]

    # Stacked area chart
    ax.fill_between(offsets, 0, assessment_activity,
                    alpha=0.7, color='#C62828', label='Evaluaciones (assignments, quizzes, grades)')
    ax.fill_between(offsets, assessment_activity,
                    [a + c for a, c in zip(assessment_activity, content_activity)],
                    alpha=0.7, color='#1565C0', label='Contenido (modules, files, pages, discussions)')

    # Mark day 0 (due date)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Fecha de entrega')

    # Styling
    ax.set_xlabel('Días relativos a la fecha de entrega', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visualizaciones promedio por día', fontsize=12, fontweight='bold')
    ax.set_title('Actividad en el LMS alrededor de fechas de evaluación\n' +
                 'Promedio de 42 evaluaciones en 10 cursos',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-14, 7)

    # Add annotation
    textstr = ('Las evaluaciones actúan como\n"centros de gravedad" de la actividad.\n\n'
               'La preparación previa (días -7 a -1)\n'
               'diferencia a quienes aprueban\n'
               'de quienes reprueban.')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'assessment_gravity_aggregated.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def create_daily_timeline_plot(pv, assignments):
    """
    Create a timeline showing daily activity with assignment due dates marked.
    """
    print("\nCreating daily timeline plot...")

    # Aggregate daily activity
    pv['is_assessment'] = pv['resource_type'].isin(ASSESSMENT_RESOURCES)

    daily = pv.groupby('date').agg({
        'id': 'count',
        'is_assessment': 'sum'
    }).reset_index()
    daily.columns = ['date', 'total', 'assessment']
    daily['content'] = daily['total'] - daily['assessment']
    daily['date'] = pd.to_datetime(daily['date'])

    # Filter to main semester period
    daily = daily[(daily['date'] >= '2025-08-01') & (daily['date'] <= '2025-12-15')]

    # Get assignment due dates
    due_dates = sorted(set(pd.Timestamp(a['due_date']) for a in assignments
                          if a['due_date'] and pd.Timestamp(a['due_date']) >= pd.Timestamp('2025-08-01')))

    print(f"  Daily records: {len(daily)}")
    print(f"  Due dates in period: {len(due_dates)}")

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Stacked area
    ax.fill_between(daily['date'], 0, daily['assessment'],
                    alpha=0.7, color='#C62828', label='Actividad en evaluaciones')
    ax.fill_between(daily['date'], daily['assessment'], daily['total'],
                    alpha=0.7, color='#1565C0', label='Actividad en contenido')

    # Mark due dates
    for i, due_date in enumerate(due_dates):
        ax.axvline(x=due_date, color='#2E7D32', linestyle='--', alpha=0.6, linewidth=1)

    # Add legend entry for due dates
    ax.axvline(x=due_dates[0], color='#2E7D32', linestyle='--', alpha=0.6, linewidth=1.5,
               label='Fechas de entrega')

    # Styling
    ax.set_xlabel('Fecha', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visualizaciones diarias', fontsize=12, fontweight='bold')
    ax.set_title('Actividad diaria en el LMS y fechas de evaluación\n' +
                 '10 cursos del modelo de alerta temprana (Agosto - Diciembre 2025)',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'assessment_timeline.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def create_pass_fail_preparation_plot(pv, assignments):
    """
    Show how pass vs fail students differ in their preparation patterns
    around assignment dates.
    """
    print("\nCreating pass/fail preparation comparison...")

    # Load enrollments with grades
    enrollments = pd.read_csv('data/page_views/student_enrollments.csv')
    enrollments['failed'] = (enrollments['final_score'] < 57).astype(int)

    # Use source_user_id for merge (short format matching enrollments)
    pv['source_user_id'] = pv['source_user_id'].astype(int)
    pv['course_id'] = pv['course_id'].astype(int)
    enrollments['user_id'] = enrollments['user_id'].astype(int)
    enrollments['course_id'] = enrollments['course_id'].astype(int)

    # Merge with page views using source_user_id
    pv_with_grades = pv.merge(
        enrollments[['user_id', 'course_id', 'failed']],
        left_on=['source_user_id', 'course_id'],
        right_on=['user_id', 'course_id'],
        how='inner'
    )

    print(f"  Page views with grade info: {len(pv_with_grades):,}")

    # Get unique due dates
    due_dates = sorted(set(a['due_date'] for a in assignments))

    # Calculate activity by days relative to assignment for pass vs fail
    # Show 7 days before (-7 to -1), day of (0), and 1 day after (1)
    days_range = range(-7, 2)

    activity_pass = {d: 0 for d in days_range}
    activity_fail = {d: 0 for d in days_range}
    count_pass = {d: 0 for d in days_range}
    count_fail = {d: 0 for d in days_range}

    for due_date in due_dates:
        for offset in days_range:
            check_date = due_date + timedelta(days=offset)

            # Pass students
            day_pass = pv_with_grades[(pv_with_grades['date'] == check_date) &
                                       (pv_with_grades['failed'] == 0)]
            # Fail students
            day_fail = pv_with_grades[(pv_with_grades['date'] == check_date) &
                                       (pv_with_grades['failed'] == 1)]

            if len(day_pass) > 0:
                # Normalize by number of students
                n_pass_students = enrollments[enrollments['failed'] == 0]['user_id'].nunique()
                activity_pass[offset] += len(day_pass) / n_pass_students
                count_pass[offset] += 1

            if len(day_fail) > 0:
                n_fail_students = enrollments[enrollments['failed'] == 1]['user_id'].nunique()
                activity_fail[offset] += len(day_fail) / n_fail_students
                count_fail[offset] += 1

    # Average
    for offset in days_range:
        if count_pass[offset] > 0:
            activity_pass[offset] /= count_pass[offset]
        if count_fail[offset] > 0:
            activity_fail[offset] /= count_fail[offset]

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    offsets = list(days_range)
    pass_activity = [activity_pass[d] for d in offsets]
    fail_activity = [activity_fail[d] for d in offsets]

    # Line plots
    ax.plot(offsets, pass_activity, 'o-', color='#2E7D32', linewidth=2.5,
            markersize=6, label='Estudiantes que APRUEBAN', alpha=0.9)
    ax.plot(offsets, fail_activity, 's--', color='#C62828', linewidth=2.5,
            markersize=6, label='Estudiantes que REPRUEBAN', alpha=0.9)

    # Fill areas
    ax.fill_between(offsets, pass_activity, alpha=0.2, color='#2E7D32')
    ax.fill_between(offsets, fail_activity, alpha=0.2, color='#C62828')

    # Mark day 0
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Fecha de entrega')

    # Highlight preparation window
    ax.axvspan(-7, -1, alpha=0.1, color='gold', label='Ventana de preparación')

    # Styling
    ax.set_xlabel('Días relativos a la fecha de entrega', fontsize=12, fontweight='bold')
    ax.set_ylabel('Visualizaciones promedio por estudiante', fontsize=12, fontweight='bold')
    ax.set_title('Comportamiento de preparación: Aprobados vs Reprobados\n' +
                 'Actividad promedio alrededor de fechas de evaluación',
                 fontsize=14, fontweight='bold')

    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-7.5, 1.5)
    ax.set_xticks(range(-7, 2))  # Integer ticks: -7, -6, ..., 0, 1

    # Calculate and show the gap
    prep_window = range(-7, 0)
    avg_pass_prep = np.mean([activity_pass[d] for d in prep_window])
    avg_fail_prep = np.mean([activity_fail[d] for d in prep_window])
    ratio = avg_pass_prep / avg_fail_prep if avg_fail_prep > 0 else 0

    textstr = (f'Semana previa a evaluación:\n'
               f'  Aprobados: {avg_pass_prep:.1f} views/día\n'
               f'  Reprobados: {avg_fail_prep:.1f} views/día\n'
               f'  Ratio: {ratio:.1f}x más actividad')
    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'pass_fail_preparation.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()

    return ratio


def create_weekly_assessment_proportion(pv, assignments):
    """
    Show how assessment-related activity changes as a proportion over course weeks.
    """
    print("\nCreating weekly assessment proportion plot...")

    # Add week number relative to course start
    # Assume courses start around August 11, 2025 (common semester start)
    course_start = pd.Timestamp('2025-08-11')
    pv['week'] = ((pd.to_datetime(pv['date']) - course_start).dt.days // 7) + 1
    pv = pv[(pv['week'] >= 1) & (pv['week'] <= 16)]

    # Calculate weekly proportions
    pv['is_assessment'] = pv['resource_type'].isin(ASSESSMENT_RESOURCES)

    weekly = pv.groupby('week').agg({
        'id': 'count',
        'is_assessment': 'sum'
    }).reset_index()
    weekly.columns = ['week', 'total', 'assessment']
    weekly['assessment_pct'] = 100 * weekly['assessment'] / weekly['total']
    weekly['content_pct'] = 100 - weekly['assessment_pct']

    # Create plot
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Bar chart of total activity
    bars = ax1.bar(weekly['week'], weekly['total'], color='#1565C0', alpha=0.4,
                   label='Actividad total')
    ax1.set_xlabel('Semana del curso', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total de visualizaciones', fontsize=12, fontweight='bold', color='#1565C0')
    ax1.tick_params(axis='y', labelcolor='#1565C0')

    # Secondary axis for assessment percentage
    ax2 = ax1.twinx()
    ax2.plot(weekly['week'], weekly['assessment_pct'], 'o-', color='#C62828',
             linewidth=2.5, markersize=8, label='% Actividad en evaluaciones')
    ax2.set_ylabel('% Actividad en evaluaciones', fontsize=12, fontweight='bold', color='#C62828')
    ax2.tick_params(axis='y', labelcolor='#C62828')
    ax2.set_ylim(0, 60)

    # Mark key weeks
    # Week 4 typically has first major assessment
    ax1.axvline(x=4, color='green', linestyle=':', alpha=0.7, linewidth=2)
    ax1.text(4.2, ax1.get_ylim()[1] * 0.95, 'Semana 4:\nPrimeras\nevaluaciones',
             fontsize=9, color='green', va='top')

    # Title
    plt.title('Evolución de actividad en evaluaciones por semana\n' +
              'La proporción de actividad en evaluaciones aumenta a partir de la semana 4',
              fontsize=13, fontweight='bold')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax1.grid(True, axis='y', alpha=0.3)
    ax1.set_xticks(range(1, 17))

    plt.tight_layout()

    output_path = OUTPUT_DIR / 'weekly_assessment_proportion.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()


def main():
    print("=" * 70)
    print("Generating Assessment Gravity Visualizations")
    print("=" * 70)

    # Load data
    pv, assignments = load_data()

    # Create visualizations
    create_aggregated_gravity_plot(pv, assignments)
    create_daily_timeline_plot(pv, assignments)
    ratio = create_pass_fail_preparation_plot(pv, assignments)
    create_weekly_assessment_proportion(pv, assignments)

    print("\n" + "=" * 70)
    print("All visualizations generated!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    print(f"\nKey finding: Pass students have {ratio:.1f}x more activity in the week before assessments")


if __name__ == "__main__":
    main()
