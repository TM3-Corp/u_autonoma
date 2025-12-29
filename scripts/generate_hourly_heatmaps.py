#!/usr/bin/env python3
"""
Generate hourly activity heatmaps for each course.
Creates 24x7 heatmaps showing student activity patterns by hour and day of week.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
import requests

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Test courses (same as in other scripts)
TEST_COURSES = [
    {'id': 86005, 'name': 'TALL DE COMPETENCIAS DIGITALES-P01'},
    {'id': 86676, 'name': 'FUND DE BUSINESS ANALYTICS-P01'},
    {'id': 84936, 'name': 'FUNDAMENTOS DE MICROECONOMÍA-P03'},
    {'id': 84941, 'name': 'FUNDAMENTOS DE MICROECONOMÍA-P01'},
    {'id': 84944, 'name': 'FUNDAMENTOS DE MACROECONOMÍA-P03'},
    {'id': 86020, 'name': 'TALL DE COMPETENCIAS DIGITALES-P02'},
    {'id': 79804, 'name': 'FUNDAMENTOS TRIBUTARIOS-P01'},
    {'id': 79875, 'name': 'TALLER DE COMP DIGITALES-P01'},
    {'id': 79913, 'name': 'FUND. DE BUSINESS ANALYTICS-P01'},
    {'id': 88381, 'name': 'MATEMÁTICAS PARA LOS NEGOCIOS-P01'},
    {'id': 89099, 'name': 'TALLER DE COMP DIGITALES-P01'},
    {'id': 89390, 'name': 'GESTIÓN DEL TALENTO-P01'},
    {'id': 89736, 'name': 'FUNDAMENTOS DE MACROECONOMÍA-P01'},
]

# Short names for visualization
SHORT_NAMES = {
    86005: 'Comp. Digitales P01',
    86676: 'Business Analytics',
    84936: 'Microeconomía P03',
    84941: 'Microeconomía P01',
    84944: 'Macroeconomía P03',
    86020: 'Comp. Digitales P02',
    79804: 'Fund. Tributarios',
    79875: 'Comp. Digitales (79875)',
    79913: 'Business Analytics (79913)',
    88381: 'Matemáticas Negocios',
    89099: 'Comp. Digitales (89099)',
    89390: 'Gestión Talento',
    89736: 'Macroeconomía P01',
}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
VIZ_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'report', 'visualizations')


def paginate(url, params=None):
    """Paginate through Canvas API results."""
    if params is None:
        params = {}
    params['per_page'] = 100

    all_results = []
    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        return all_results

    data = r.json()
    if not data:
        return all_results

    all_results.extend(data if isinstance(data, list) else [data])

    while 'next' in r.links:
        r = requests.get(r.links['next']['url'], headers=headers)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        all_results.extend(data if isinstance(data, list) else [data])

    return all_results


def get_enrollments(course_id):
    """Get student enrollments for a course."""
    return paginate(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        {'type[]': 'StudentEnrollment'}
    )


def get_user_activity(course_id, user_id):
    """Get hourly activity for a student."""
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/users/{user_id}/activity',
        headers=headers
    )
    if r.status_code == 200:
        return r.json()
    return {}


def parse_hourly_activity(page_views_dict):
    """Parse page views dict into hour-of-day and day-of-week counts."""
    hourly_data = defaultdict(lambda: defaultdict(int))  # day_of_week -> hour -> count

    for ts_str, count in page_views_dict.items():
        try:
            # Parse ISO timestamp
            if 'T' in ts_str:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            else:
                dt = datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')

            day_of_week = dt.weekday()  # 0 = Monday, 6 = Sunday
            hour = dt.hour
            hourly_data[day_of_week][hour] += count
        except Exception:
            continue

    return hourly_data


def extract_course_hourly_data(course_id, course_name):
    """Extract hourly activity data for all students in a course."""
    print(f"  Extracting {course_name}...")

    enrollments = get_enrollments(course_id)
    print(f"    Found {len(enrollments)} students")

    # Aggregate hourly data across all students
    course_hourly = np.zeros((7, 24))  # 7 days x 24 hours

    students_processed = 0
    for i, enrollment in enumerate(enrollments):
        user_id = enrollment.get('user_id')
        if not user_id:
            continue

        activity = get_user_activity(course_id, user_id)
        page_views = activity.get('page_views', {})

        if page_views:
            hourly_data = parse_hourly_activity(page_views)
            for day, hours in hourly_data.items():
                for hour, count in hours.items():
                    course_hourly[day][hour] += count
            students_processed += 1

        if (i + 1) % 10 == 0:
            print(f"      Processed {i+1}/{len(enrollments)} students...")

    print(f"    Processed {students_processed} students with activity data")
    return course_hourly


def create_heatmap(data, course_name, course_id, output_path):
    """Create a single 24x7 heatmap for a course."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Transpose to have hours on Y-axis and days on X-axis
    data_T = data.T  # Now 24 hours x 7 days

    # Create custom colormap: white to dark blue
    colors = ['#ffffff', '#e6f2ff', '#cce5ff', '#99ccff', '#66b3ff',
              '#3399ff', '#0080ff', '#0066cc', '#004d99', '#003366']
    cmap = mcolors.LinearSegmentedColormap.from_list('white_blue', colors)

    # Plot heatmap
    im = ax.imshow(data_T, cmap=cmap, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Interacciones')

    # Set labels
    days = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
    hours = [f'{h:02d}:00' for h in range(24)]

    ax.set_xticks(range(7))
    ax.set_xticklabels(days)
    ax.set_yticks(range(24))
    ax.set_yticklabels(hours)

    ax.set_xlabel('Día de la Semana')
    ax.set_ylabel('Hora del Día')
    ax.set_title(f'Patrón de Actividad: {course_name}')

    # Add text annotations (interaction counts)
    for i in range(24):
        for j in range(7):
            value = int(data_T[i, j])
            if value > 0:
                # Choose text color based on background
                text_color = 'white' if value > data_T.max() * 0.5 else 'black'
                ax.text(j, i, str(value), ha='center', va='center',
                       color=text_color, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    Saved: {output_path}")


def create_combined_heatmap(all_course_data, output_path):
    """Create a combined visualization with all courses."""
    n_courses = len(all_course_data)
    cols = 3
    rows = (n_courses + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() if n_courses > 1 else [axes]

    # Custom colormap
    colors = ['#ffffff', '#e6f2ff', '#cce5ff', '#99ccff', '#66b3ff',
              '#3399ff', '#0080ff', '#0066cc', '#004d99', '#003366']
    cmap = mcolors.LinearSegmentedColormap.from_list('white_blue', colors)

    days = ['L', 'M', 'X', 'J', 'V', 'S', 'D']

    for idx, (course_id, data) in enumerate(all_course_data.items()):
        ax = axes[idx]
        data_T = data.T  # 24 hours x 7 days

        im = ax.imshow(data_T, cmap=cmap, aspect='auto')

        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.set_yticks([0, 6, 12, 18, 23])
        ax.set_yticklabels(['00:00', '06:00', '12:00', '18:00', '23:00'])

        short_name = SHORT_NAMES.get(course_id, f'Curso {course_id}')
        ax.set_title(short_name, fontsize=10)

        # Add numbers in cells (only if reasonable number of interactions)
        max_val = data_T.max()
        for i in range(24):
            for j in range(7):
                value = int(data_T[i, j])
                if value > 0:
                    text_color = 'white' if value > max_val * 0.5 else 'black'
                    ax.text(j, i, str(value), ha='center', va='center',
                           color=text_color, fontsize=5)

    # Hide empty subplots
    for idx in range(len(all_course_data), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Patrones de Actividad Estudiantil por Curso (Hora del Día vs Día de la Semana)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved combined heatmap: {output_path}")


def main():
    print("=" * 80)
    print("GENERATING HOURLY ACTIVITY HEATMAPS")
    print("=" * 80)

    os.makedirs(VIZ_DIR, exist_ok=True)

    all_course_data = {}

    for course in TEST_COURSES:
        course_id = course['id']
        course_name = course['name']

        hourly_data = extract_course_hourly_data(course_id, course_name)
        all_course_data[course_id] = hourly_data

        # Create individual heatmap
        output_file = os.path.join(VIZ_DIR, f'hourly_heatmap_{course_id}.png')
        short_name = SHORT_NAMES.get(course_id, course_name)
        create_heatmap(hourly_data, short_name, course_id, output_file)

    # Save raw data
    data_file = os.path.join(DATA_DIR, 'hourly_activity_by_course.json')
    with open(data_file, 'w') as f:
        json.dump({str(k): v.tolist() for k, v in all_course_data.items()}, f, indent=2)
    print(f"\nSaved raw data: {data_file}")

    # Create combined visualization
    combined_path = os.path.join(VIZ_DIR, 'hourly_heatmaps_combined.png')
    create_combined_heatmap(all_course_data, combined_path)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == '__main__':
    main()
