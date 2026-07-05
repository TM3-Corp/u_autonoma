#!/usr/bin/env python3
"""
Export Canvas LMS Data for Wave Visualizer Dashboard

Transforms Canvas data from Universidad Autónoma de Chile into the format
expected by the Wave Visualizer dashboard for early warning visualization.

Exports:
- students.json: EnhancedStudent[] with risk factors
- courses.json: EnhancedCourseAnalytics[] with sections
- heatmaps.json: Activity heatmaps per course (24x7)
- resources.json: Resource usage with grade impact
- recommendations.json: Peer-based resource recommendations
- risk_history.json: Weekly risk evolution per student
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional: XGBoost for predictions
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
PAGE_VIEWS_FILE = DATA_DIR / "page_views/categorized_page_views.parquet"
ENROLLMENTS_FILE = DATA_DIR / "page_views/student_enrollments.csv"
SUMMARIES_FILE = DATA_DIR / "student_summaries.json"
COURSES_FILE = DATA_DIR / "courses_raw.json"
HEATMAPS_FILE = DATA_DIR / "hourly_activity_by_course.json"
ASSIGNMENTS_FILE = DATA_DIR / "assignment_analytics.json"
OUTPUT_DIR = DATA_DIR / "dashboard_export"

# Model courses (those with sufficient data)
MODEL_COURSES = [79875, 79913, 84936, 84941, 84944, 86020, 86676, 88381, 89099, 89390]

# Risk thresholds (matching dashboard)
RISK_THRESHOLDS = {
    'Muy Alto': 80,
    'Alto': 60,
    'Medio': 40,
    'Bajo': 20,
    'Muy Bajo': 0
}

SESSION_GAP_MINUTES = 30


def normalize_user_id(user_id):
    """Normalize user IDs that may have timestamp prefixes."""
    if user_id > 10000000000:
        return user_id % 10000000000
    return user_id


def get_risk_level(risk_score):
    """Convert numeric risk score (0-100) to categorical level."""
    if risk_score >= 80:
        return 'Muy Alto'
    elif risk_score >= 60:
        return 'Alto'
    elif risk_score >= 40:
        return 'Medio'
    elif risk_score >= 20:
        return 'Bajo'
    else:
        return 'Muy Bajo'


def load_all_data():
    """Load all data sources."""
    print("=" * 60)
    print("LOADING CANVAS DATA")
    print("=" * 60)

    # Page views (categorized)
    print("\n1. Loading page views...")
    df_pv = pd.read_parquet(PAGE_VIEWS_FILE)
    df_pv['created_at'] = pd.to_datetime(df_pv['created_at']).dt.tz_localize(None)
    df_pv['user_id'] = df_pv['user_id'].apply(normalize_user_id)
    print(f"   {len(df_pv):,} page views loaded")

    # Enrollments with grades
    print("\n2. Loading enrollments...")
    df_enroll = pd.read_csv(ENROLLMENTS_FILE)
    df_enroll['failed'] = (df_enroll['final_score'] < 57).astype(int)
    print(f"   {len(df_enroll):,} enrollments loaded")

    # Student summaries
    print("\n3. Loading student summaries...")
    with open(SUMMARIES_FILE) as f:
        summaries = json.load(f)
    df_summaries = pd.DataFrame(summaries)
    print(f"   {len(df_summaries):,} student summaries loaded")

    # Courses
    print("\n4. Loading courses...")
    with open(COURSES_FILE) as f:
        courses = json.load(f)
    courses_dict = {c['id']: c for c in courses}
    print(f"   {len(courses):,} courses loaded")

    # Heatmaps (already computed)
    print("\n5. Loading activity heatmaps...")
    with open(HEATMAPS_FILE) as f:
        heatmaps = json.load(f)
    print(f"   {len(heatmaps)} course heatmaps loaded")

    # Assignments
    print("\n6. Loading assignments...")
    with open(ASSIGNMENTS_FILE) as f:
        assignments = json.load(f)
    df_assignments = pd.DataFrame(assignments)
    if len(df_assignments) > 0:
        df_assignments['due_at'] = pd.to_datetime(df_assignments['due_at'], errors='coerce')
    print(f"   {len(df_assignments):,} assignments loaded")

    return {
        'page_views': df_pv,
        'enrollments': df_enroll,
        'summaries': df_summaries,
        'courses': courses_dict,
        'heatmaps': heatmaps,
        'assignments': df_assignments
    }


def calculate_student_features(df_pv, df_enroll, df_summaries):
    """Calculate comprehensive features for each student."""
    print("\n" + "=" * 60)
    print("CALCULATING STUDENT FEATURES")
    print("=" * 60)

    features_list = []

    # Group page views by user-course
    pv_grouped = df_pv.groupby(['user_id', 'course_id'])

    for (user_id, course_id), group in pv_grouped:
        # Get enrollment data
        enroll = df_enroll[(df_enroll['user_id'] == user_id) & (df_enroll['course_id'] == course_id)]
        if len(enroll) == 0:
            continue
        enroll = enroll.iloc[0]

        # Get summary data
        summary = df_summaries[(df_summaries['id'] == user_id) & (df_summaries['course_id'] == course_id)]

        # Basic activity features
        group = group.sort_values('created_at')
        timestamps = pd.to_datetime(group['created_at'])

        # Session calculation
        gaps = timestamps.diff().dt.total_seconds() / 60
        session_starts = (gaps >= SESSION_GAP_MINUTES) | (gaps.isna())
        n_sessions = session_starts.sum()

        # Resource type counts
        resource_counts = group['resource_type'].value_counts().to_dict()

        # Calculate features
        features = {
            'user_id': int(user_id),
            'course_id': int(course_id),
            'final_score': float(enroll.get('final_score', 0) or 0),
            'current_score': float(enroll.get('current_score', 0) or 0),
            'failed': int(enroll.get('failed', 0)),

            # Activity features
            'total_page_views': len(group),
            'n_sessions': int(n_sessions),
            'total_activity_time': int(enroll.get('total_activity_time', 0) or 0),

            # Resource breakdown
            'files_views': resource_counts.get('files', 0),
            'discussions_views': resource_counts.get('discussions', 0),
            'pages_views': resource_counts.get('pages', 0),
            'modules_views': resource_counts.get('modules', 0),
            'quizzes_views': resource_counts.get('quizzes', 0),
            'assignments_views': resource_counts.get('assignments', 0),
            'grades_views': resource_counts.get('grades', 0),
            'announcements_views': resource_counts.get('announcements', 0),
        }

        # Add summary features if available
        if len(summary) > 0:
            summary = summary.iloc[0]
            features['page_views_level'] = int(summary.get('page_views_level', 0) or 0)
            features['participations'] = int(summary.get('participations', 0) or 0)
            features['participations_level'] = int(summary.get('participations_level', 0) or 0)

            # Tardiness breakdown
            tardiness = summary.get('tardiness_breakdown', {})
            if tardiness:
                features['on_time'] = int(tardiness.get('on_time', 0) or 0)
                features['late'] = int(tardiness.get('late', 0) or 0)
                features['missing'] = int(tardiness.get('missing', 0) or 0)
                total = features['on_time'] + features['late'] + features['missing']
                features['on_time_rate'] = features['on_time'] / total if total > 0 else 0

        features_list.append(features)

    df_features = pd.DataFrame(features_list)
    print(f"   Calculated features for {len(df_features)} student-course pairs")

    return df_features


def calculate_risk_predictions(df_features):
    """Calculate risk predictions for each student."""
    print("\n" + "=" * 60)
    print("CALCULATING RISK PREDICTIONS")
    print("=" * 60)

    # For now, use a heuristic based on available features
    # In production, load trained model and get actual predictions

    risk_scores = []

    for idx, row in df_features.iterrows():
        # Multi-factor risk calculation

        # 1. Grade factor (higher score = lower risk)
        grade_risk = max(0, 100 - row.get('final_score', 50))

        # 2. Engagement factor (normalized within course)
        course_df = df_features[df_features['course_id'] == row['course_id']]
        if len(course_df) > 1:
            pv_percentile = (course_df['total_page_views'] <= row['total_page_views']).mean() * 100
            engagement_risk = max(0, 100 - pv_percentile)
        else:
            engagement_risk = 50

        # 3. Participation factor
        participation_risk = max(0, 100 - (row.get('participations', 0) * 10))

        # 4. Tardiness factor
        on_time_rate = row.get('on_time_rate', 0.5)
        tardiness_risk = max(0, 100 - (on_time_rate * 100))

        # Weighted combination (matches dashboard weights)
        risk_score = (
            grade_risk * 0.30 +       # Academic performance
            engagement_risk * 0.25 +  # LMS engagement
            participation_risk * 0.25 + # Participations
            tardiness_risk * 0.20     # Assignment timeliness
        )

        risk_scores.append(min(100, max(0, risk_score)))

    df_features['risk_score'] = risk_scores
    df_features['risk_level'] = df_features['risk_score'].apply(get_risk_level)

    print(f"   Risk distribution:")
    for level in ['Muy Alto', 'Alto', 'Medio', 'Bajo', 'Muy Bajo']:
        count = (df_features['risk_level'] == level).sum()
        pct = count / len(df_features) * 100
        print(f"      {level}: {count} ({pct:.1f}%)")

    return df_features


def export_students(df_features, data, output_dir):
    """Export students.json in dashboard format."""
    print("\n" + "=" * 60)
    print("EXPORTING STUDENTS.JSON")
    print("=" * 60)

    students = []
    student_ids = df_features['user_id'].unique()

    for i, user_id in enumerate(student_ids):
        student_rows = df_features[df_features['user_id'] == user_id]

        # Aggregate across courses
        cursos = {}
        for _, row in student_rows.iterrows():
            course_id = row['course_id']
            course_info = data['courses'].get(course_id, {})
            course_name = course_info.get('name', f'Curso {course_id}')
            cursos[course_name] = float(row['risk_score'])

        avg_risk = student_rows['risk_score'].mean()

        # Risk factors from most recent/main course
        main_row = student_rows.iloc[0]

        # Pseudonymize student
        student = {
            'id': int(user_id),
            'nombre': f'Estudiante_{i+1:03d}',
            'riesgoPromedio': float(avg_risk),
            'nivelRiesgo': get_risk_level(avg_risk),
            'cursos': cursos,

            'riskFactors': {
                'involucramiento': float(min(100, main_row.get('total_page_views', 0) / 10)),
                'notas': float(main_row.get('final_score', 50) or 50),
                'asistencia': float((main_row.get('on_time_rate') or 0.5) * 100),
                'entregaTareas': float((main_row.get('on_time_rate') or 0.5) * 100),
                'participacionForos': float(min(100, (main_row.get('discussions_views') or 0) * 5)),
                'usoRecursos': float(min(100, ((main_row.get('files_views') or 0) + (main_row.get('pages_views') or 0)) / 5)),
                'puntajePSU': 550,  # Placeholder
                'tipoEscuela': 'subvencionada',
                'beca': False
            },

            'riskTrajectory': 'stable',  # Would need historical data
            'predictedRiskIn30Days': float(avg_risk),
            'successProbability': float(100 - avg_risk),
            'responsiveness': 50.0,

            # Demographic placeholders
            'carrera': 'Ingeniería en Control de Gestión',
            'semestre': 1,
            'edad': 20,
            'genero': 'M' if i % 2 == 0 else 'F',
            'region': 'Región Metropolitana'
        }

        students.append(student)

    # Replace NaN values with defaults before JSON export
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(item) for item in obj]
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return 0.0
        return obj

    students = clean_nan(students)

    output_file = output_dir / 'students.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(students, f, ensure_ascii=False, indent=2)

    print(f"   Exported {len(students)} students to {output_file}")
    return students


def export_courses(df_features, data, output_dir):
    """Export courses.json in dashboard format."""
    print("\n" + "=" * 60)
    print("EXPORTING COURSES.JSON")
    print("=" * 60)

    courses = []
    course_ids = df_features['course_id'].unique()

    for course_id in course_ids:
        course_info = data['courses'].get(course_id, {})
        course_df = df_features[df_features['course_id'] == course_id]

        # Risk distribution
        risk_dist = course_df['risk_level'].value_counts().to_dict()

        # Engagement metrics
        avg_participation = course_df['participations'].mean() if 'participations' in course_df else 0
        avg_resource_usage = (
            course_df['files_views'].mean() +
            course_df['pages_views'].mean() +
            course_df['modules_views'].mean()
        ) / 3 if 'files_views' in course_df else 0

        course = {
            'nombre': course_info.get('name', f'Curso {course_id}'),
            'codigo': course_info.get('course_code', f'CURSO-{course_id}'),
            'creditos': 6,
            'semestre': 1,
            'riesgoPromedio': float(course_df['risk_score'].mean()),

            'riskDistribution': {
                'Muy Alto': int(risk_dist.get('Muy Alto', 0)),
                'Alto': int(risk_dist.get('Alto', 0)),
                'Medio': int(risk_dist.get('Medio', 0)),
                'Bajo': int(risk_dist.get('Bajo', 0)),
                'Muy Bajo': int(risk_dist.get('Muy Bajo', 0))
            },

            'engagementMetrics': {
                'averageParticipation': float(avg_participation),
                'resourcesUtilization': float(avg_resource_usage),
                'forumActivity': float(course_df['discussions_views'].mean() if 'discussions_views' in course_df else 0),
                'assignmentCompletion': float(course_df['on_time_rate'].mean() * 100 if 'on_time_rate' in course_df else 50)
            },

            'courseRiskFactors': {
                'complejidadContenido': 60,
                'cargaTrabajo': 65,
                'prerequisitos': 40,
                'metodologiaEnsenanza': 70,
                'recursosDisponibles': 75,
                'apoyoDocente': 70
            },

            'enrolledStudents': len(course_df),
            'canvas_id': int(course_id)
        }

        courses.append(course)

    output_file = output_dir / 'courses.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)

    print(f"   Exported {len(courses)} courses to {output_file}")
    return courses


def export_heatmaps(data, output_dir):
    """Export heatmaps.json (transform existing format to dashboard format)."""
    print("\n" + "=" * 60)
    print("EXPORTING HEATMAPS.JSON")
    print("=" * 60)

    heatmaps = {}
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

    for course_id, hourly_data in data['heatmaps'].items():
        # hourly_data is a 7x24 matrix (days x hours)
        course_info = data['courses'].get(int(course_id), {})
        course_name = course_info.get('name', f'Curso {course_id}')

        # Normalize values to 0-100
        matrix = np.array(hourly_data)
        max_val = matrix.max()
        if max_val > 0:
            matrix = (matrix / max_val * 100).astype(int)

        # Convert to dashboard format
        heatmap_data = []
        for day_idx, day_name in enumerate(days):
            for hour in range(24):
                heatmap_data.append({
                    'dia': day_name,
                    'hora': hour,
                    'valor': int(matrix[day_idx][hour])
                })

        heatmaps[course_name] = {
            'data': heatmap_data,
            'canvas_id': int(course_id)
        }

    output_file = output_dir / 'heatmaps.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(heatmaps, f, ensure_ascii=False, indent=2)

    print(f"   Exported {len(heatmaps)} course heatmaps to {output_file}")
    return heatmaps


def export_resources(df_pv, df_features, data, output_dir):
    """Export resources.json with usage stats and grade impact."""
    print("\n" + "=" * 60)
    print("EXPORTING RESOURCES.JSON")
    print("=" * 60)

    resources_by_course = {}

    for course_id in df_pv['course_id'].dropna().unique():
        if pd.isna(course_id):
            continue
        course_id = int(course_id)
        course_pv = df_pv[df_pv['course_id'] == course_id]
        course_features = df_features[df_features['course_id'] == course_id]
        course_info = data['courses'].get(course_id, {})
        course_name = course_info.get('name', f'Curso {course_id}')

        total_students = len(course_features)

        # Group by resource type
        resource_types = ['files', 'discussions', 'pages', 'modules', 'quizzes', 'assignments', 'announcements']
        resources = []

        for rtype in resource_types:
            type_pv = course_pv[course_pv['resource_type'] == rtype]

            if len(type_pv) == 0:
                continue

            unique_viewers = type_pv['user_id'].nunique()
            total_views = len(type_pv)

            # Calculate grade impact (correlation)
            if f'{rtype}_views' in course_features.columns and 'final_score' in course_features.columns:
                views_col = course_features[f'{rtype}_views']
                grades_col = course_features['final_score']

                if views_col.std() > 0 and grades_col.std() > 0:
                    correlation = views_col.corr(grades_col)
                    impact = max(0, min(1, (correlation + 1) / 2))  # Normalize to 0-1
                else:
                    impact = 0.5
            else:
                impact = 0.5

            resources.append({
                'id': f'{course_id}_{rtype}',
                'nombre': rtype.replace('_', ' ').title(),
                'area': rtype,
                'vistas': int(total_views),
                'porcentajeEstudiantes': float(unique_viewers / total_students * 100) if total_students > 0 else 0,
                'impactoRendimiento': float(impact),
                'uniqueViewers': int(unique_viewers),
                'totalStudents': int(total_students)
            })

        resources_by_course[course_name] = {
            'resources': sorted(resources, key=lambda x: x['vistas'], reverse=True),
            'canvas_id': int(course_id)
        }

    output_file = output_dir / 'resources.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(resources_by_course, f, ensure_ascii=False, indent=2)

    print(f"   Exported resources for {len(resources_by_course)} courses to {output_file}")
    return resources_by_course


def export_recommendations(df_pv, df_features, data, output_dir):
    """Export recommendations.json for peer-based suggestions."""
    print("\n" + "=" * 60)
    print("EXPORTING RECOMMENDATIONS.JSON")
    print("=" * 60)

    recommendations_by_course = {}

    for course_id in df_pv['course_id'].dropna().unique():
        if pd.isna(course_id):
            continue
        course_id = int(course_id)
        course_pv = df_pv[df_pv['course_id'] == course_id]
        course_features = df_features[df_features['course_id'] == course_id]
        course_info = data['courses'].get(course_id, {})
        course_name = course_info.get('name', f'Curso {course_id}')

        total_students = len(course_features)

        # Get passing students (for success-based recommendations)
        passing_students = course_features[course_features['failed'] == 0]['user_id'].tolist()

        # Resources viewed by passing students
        passing_pv = course_pv[course_pv['user_id'].isin(passing_students)]

        # Top resources by passing students
        resource_popularity = course_pv.groupby('resource_type').agg({
            'user_id': 'nunique'
        }).reset_index()
        resource_popularity.columns = ['resource_type', 'viewers']
        resource_popularity['view_pct'] = resource_popularity['viewers'] / total_students * 100
        resource_popularity = resource_popularity.sort_values('view_pct', ascending=False)

        # Create recommendations
        recommendations = []
        for _, row in resource_popularity.iterrows():
            if row['view_pct'] >= 30:  # At least 30% of students viewed
                recommendations.append({
                    'resourceType': row['resource_type'],
                    'viewPercentage': float(row['view_pct']),
                    'waveStrength': 'steep' if row['view_pct'] >= 70 else ('moderate' if row['view_pct'] >= 50 else 'mild'),
                    'trending': row['view_pct'] >= 50,
                    'totalViewers': int(row['viewers']),
                    'totalStudents': int(total_students)
                })

        recommendations_by_course[course_name] = {
            'recommendations': recommendations,
            'canvas_id': int(course_id)
        }

    output_file = output_dir / 'recommendations.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(recommendations_by_course, f, ensure_ascii=False, indent=2)

    print(f"   Exported recommendations for {len(recommendations_by_course)} courses to {output_file}")
    return recommendations_by_course


def export_engagement_timeline(df_pv, data, output_dir):
    """Export daily engagement timeline for each course."""
    print("\n" + "=" * 60)
    print("EXPORTING ENGAGEMENT_TIMELINE.JSON")
    print("=" * 60)

    timelines = {}

    for course_id in df_pv['course_id'].dropna().unique():
        if pd.isna(course_id):
            continue
        course_id = int(course_id)
        course_pv = df_pv[df_pv['course_id'] == course_id].copy()
        course_info = data['courses'].get(course_id, {})
        course_name = course_info.get('name', f'Curso {course_id}')

        # Group by date
        course_pv['date'] = course_pv['created_at'].dt.date
        daily = course_pv.groupby('date').agg({
            'user_id': ['count', 'nunique']
        }).reset_index()
        daily.columns = ['fecha', 'totalInteracciones', 'estudiantesActivos']

        timeline = []
        for _, row in daily.iterrows():
            timeline.append({
                'fecha': row['fecha'].isoformat(),
                'totalInteracciones': int(row['totalInteracciones']),
                'estudiantesActivos': int(row['estudiantesActivos'])
            })

        timelines[course_name] = {
            'timeline': sorted(timeline, key=lambda x: x['fecha']),
            'canvas_id': int(course_id)
        }

    output_file = output_dir / 'engagement_timeline.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(timelines, f, ensure_ascii=False, indent=2)

    print(f"   Exported engagement timelines for {len(timelines)} courses to {output_file}")
    return timelines


def export_dashboard_summary(students, courses, output_dir):
    """Export a summary file for quick loading."""
    print("\n" + "=" * 60)
    print("EXPORTING DASHBOARD_SUMMARY.JSON")
    print("=" * 60)

    # Risk distribution
    risk_counts = defaultdict(int)
    for s in students:
        risk_counts[s['nivelRiesgo']] += 1

    summary = {
        'totalStudents': len(students),
        'totalCourses': len(courses),
        'lastUpdated': datetime.now().isoformat(),
        'semester': '2025-2',

        'riskDistribution': {
            'Muy Alto': risk_counts['Muy Alto'],
            'Alto': risk_counts['Alto'],
            'Medio': risk_counts['Medio'],
            'Bajo': risk_counts['Bajo'],
            'Muy Bajo': risk_counts['Muy Bajo']
        },

        'overallRisk': sum(s['riesgoPromedio'] for s in students) / len(students) if students else 0,

        'courseList': [{'nombre': c['nombre'], 'codigo': c['codigo'], 'riesgoPromedio': c['riesgoPromedio']} for c in courses],

        'dataSource': 'Canvas LMS - Universidad Autónoma de Chile',
        'exportVersion': '1.0'
    }

    output_file = output_dir / 'dashboard_summary.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"   Exported summary to {output_file}")
    return summary


def main():
    print("=" * 60)
    print("CANVAS LMS DATA EXPORT FOR WAVE VISUALIZER")
    print("=" * 60)
    print(f"Export started at: {datetime.now()}")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load all data
    data = load_all_data()

    # Calculate student features
    df_features = calculate_student_features(
        data['page_views'],
        data['enrollments'],
        data['summaries']
    )

    # Calculate risk predictions
    df_features = calculate_risk_predictions(df_features)

    # Export all files
    students = export_students(df_features, data, OUTPUT_DIR)
    courses = export_courses(df_features, data, OUTPUT_DIR)
    export_heatmaps(data, OUTPUT_DIR)
    export_resources(data['page_views'], df_features, data, OUTPUT_DIR)
    export_recommendations(data['page_views'], df_features, data, OUTPUT_DIR)
    export_engagement_timeline(data['page_views'], data, OUTPUT_DIR)
    export_dashboard_summary(students, courses, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("EXPORT COMPLETE!")
    print("=" * 60)
    print(f"\nFiles exported to: {OUTPUT_DIR}")
    print(f"  - students.json ({len(students)} students)")
    print(f"  - courses.json ({len(courses)} courses)")
    print(f"  - heatmaps.json")
    print(f"  - resources.json")
    print(f"  - recommendations.json")
    print(f"  - engagement_timeline.json")
    print(f"  - dashboard_summary.json")
    print(f"\nExport completed at: {datetime.now()}")


if __name__ == '__main__':
    main()
