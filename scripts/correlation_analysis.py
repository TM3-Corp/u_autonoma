"""
Comprehensive Correlation Analysis - Canvas LMS Student Performance
Universidad Autónoma de Chile

Analyzes correlations between all available activity metrics and student grades
(both partial and final) for each course with sufficient data.
"""

import json
import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from config import DATA_DIR, API_URL, API_TOKEN

# Minimum requirements for analysis
MIN_STUDENTS = 15
MIN_GRADE_VARIANCE = 5  # At least 5% variance in grades

def load_data():
    """Load all extracted data"""
    print("Loading extracted data...")

    data = {}
    files = ['enrollments.json', 'student_summaries.json', 'submissions.json',
             'assignments.json', 'courses_raw.json']

    for f in files:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path):
            with open(path, 'r') as file:
                data[f.replace('.json', '')] = json.load(file)
                print(f"  Loaded {f}: {len(data[f.replace('.json', '')])} records")
        else:
            print(f"  Warning: {f} not found")
            data[f.replace('.json', '')] = []

    return data


def build_course_dataframe(course_id, data):
    """Build a comprehensive dataframe for a single course combining all data sources"""

    # Filter data for this course
    enrollments = [e for e in data['enrollments'] if e.get('course_id') == course_id]
    summaries = [s for s in data['student_summaries'] if s.get('course_id') == course_id]
    submissions = [s for s in data['submissions'] if s.get('course_id') == course_id]
    assignments = [a for a in data['assignments'] if a.get('course_id') == course_id]

    if not enrollments:
        return None

    # Create base dataframe from enrollments
    students = {}
    for e in enrollments:
        user_id = e.get('user_id')
        if user_id:
            grades = e.get('grades', {}) or {}
            students[user_id] = {
                'user_id': user_id,
                'current_score': grades.get('current_score'),
                'final_score': grades.get('final_score'),
                'total_activity_time': e.get('total_activity_time', 0),
                'enrollment_state': e.get('enrollment_state'),
                'last_activity_at': e.get('last_activity_at'),
            }

    # Add student summary data (activity metrics)
    for s in summaries:
        user_id = s.get('id')
        if user_id in students:
            students[user_id].update({
                'page_views': s.get('page_views', 0),
                'page_views_level': s.get('page_views_level', 0),
                'participations': s.get('participations', 0),
                'participations_level': s.get('participations_level', 0),
            })

            # Tardiness breakdown
            tb = s.get('tardiness_breakdown', {}) or {}
            students[user_id].update({
                'on_time': tb.get('on_time', 0),
                'late': tb.get('late', 0),
                'missing': tb.get('missing', 0),
                'floating': tb.get('floating', 0),
            })

    # Process submissions to get per-student metrics
    student_submissions = {}
    for sub in submissions:
        user_id = sub.get('user_id')
        if user_id not in student_submissions:
            student_submissions[user_id] = []
        student_submissions[user_id].append(sub)

    # Calculate submission-based features
    for user_id, subs in student_submissions.items():
        if user_id in students:
            scores = [s.get('score') for s in subs if s.get('score') is not None]
            graded = [s for s in subs if s.get('workflow_state') == 'graded']
            submitted = [s for s in subs if s.get('submitted_at') is not None]

            students[user_id].update({
                'num_submissions': len(submitted),
                'num_graded': len(graded),
                'num_scores': len(scores),
                'avg_score': np.mean(scores) if scores else None,
                'min_score': np.min(scores) if scores else None,
                'max_score': np.max(scores) if scores else None,
                'score_std': np.std(scores) if len(scores) > 1 else 0,
            })

            # First assignment score (if available)
            if scores:
                # Sort by assignment_id to get first
                sorted_subs = sorted([s for s in subs if s.get('score') is not None],
                                     key=lambda x: x.get('assignment_id', 0))
                if sorted_subs:
                    students[user_id]['first_score'] = sorted_subs[0].get('score')

    # Build assignments lookup for scoring breakdown
    assignment_info = {}
    for a in assignments:
        assignment_info[a['id']] = {
            'name': a.get('name', ''),
            'points_possible': a.get('points_possible', 0),
            'due_at': a.get('due_at'),
            'assignment_group_id': a.get('assignment_group_id'),
        }

    # Calculate per-assignment scores for top assignments
    # Get sumativa (exam) assignments
    sumativa_ids = [aid for aid, info in assignment_info.items()
                    if 'sumativa' in info['name'].lower()]

    # Add individual exam scores if available
    for user_id in students:
        user_subs = student_submissions.get(user_id, [])
        for i, aid in enumerate(sorted(sumativa_ids)[:5]):  # Top 5 exams
            sub = next((s for s in user_subs if s.get('assignment_id') == aid), None)
            if sub and sub.get('score') is not None:
                students[user_id][f'exam_{i+1}_score'] = sub.get('score')

    # Convert to DataFrame
    df = pd.DataFrame(list(students.values()))

    # Calculate derived features (with safe column checks)
    if len(df) > 0:
        # Ensure submission columns exist
        if 'num_submissions' not in df.columns:
            df['num_submissions'] = 0
        if 'num_graded' not in df.columns:
            df['num_graded'] = 0
        if 'num_scores' not in df.columns:
            df['num_scores'] = 0

        # Submission rate
        total_assignments = len(assignments) if assignments else 1
        df['submission_rate'] = df['num_submissions'].fillna(0) / total_assignments

        # Ensure tardiness columns exist
        for col in ['on_time', 'late', 'missing', 'floating']:
            if col not in df.columns:
                df[col] = 0

        # On-time rate
        total_required = (df['on_time'].fillna(0) + df['late'].fillna(0) + df['missing'].fillna(0))
        df['on_time_rate'] = np.where(total_required > 0,
                                       df['on_time'].fillna(0) / total_required, 0)
        df['late_rate'] = np.where(total_required > 0,
                                    df['late'].fillna(0) / total_required, 0)
        df['missing_rate'] = np.where(total_required > 0,
                                       df['missing'].fillna(0) / total_required, 0)

        # Ensure activity columns exist
        if 'page_views' not in df.columns:
            df['page_views'] = 0
        if 'participations' not in df.columns:
            df['participations'] = 0

        # Activity engagement score
        pv_max = df['page_views'].max() if df['page_views'].max() > 0 else 1
        part_max = df['participations'].max() if df['participations'].max() > 0 else 1
        df['activity_engagement'] = (df['page_views'].fillna(0) / pv_max) * 0.5 + \
                                    (df['participations'].fillna(0) / part_max) * 0.5

    return df


def calculate_correlations(df, target_col, feature_cols):
    """Calculate Pearson and Spearman correlations with statistical significance"""

    results = []

    for col in feature_cols:
        if col not in df.columns or col == target_col:
            continue

        # Get valid pairs (non-null for both)
        valid = df[[col, target_col]].dropna()

        if len(valid) < 5:  # Need at least 5 data points
            continue

        # Check for variance
        if valid[col].std() == 0 or valid[target_col].std() == 0:
            continue

        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(valid[col], valid[target_col])

        # Spearman correlation (rank-based, more robust)
        spearman_r, spearman_p = stats.spearmanr(valid[col], valid[target_col])

        results.append({
            'feature': col,
            'n_samples': len(valid),
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'abs_pearson': abs(pearson_r),
            'significant': pearson_p < 0.05,
        })

    return pd.DataFrame(results)


def analyze_course(course_id, data, course_name=""):
    """Run complete correlation analysis for a single course"""

    print(f"\n{'='*80}")
    print(f"COURSE {course_id}: {course_name}")
    print('='*80)

    # Build dataframe
    df = build_course_dataframe(course_id, data)

    if df is None or len(df) < MIN_STUDENTS:
        print(f"  Skipped: Not enough students (need {MIN_STUDENTS})")
        return None

    # Check for grade data
    has_current = df['current_score'].notna().sum() > MIN_STUDENTS
    has_final = df['final_score'].notna().sum() > MIN_STUDENTS

    if not has_current and not has_final:
        print(f"  Skipped: No grade data available")
        return None

    # Check grade variance
    current_var = df['current_score'].std() if has_current else 0
    final_var = df['final_score'].std() if has_final else 0

    if current_var < MIN_GRADE_VARIANCE and final_var < MIN_GRADE_VARIANCE:
        print(f"  Skipped: Low grade variance (current={current_var:.1f}%, final={final_var:.1f}%)")
        return None

    print(f"  Students: {len(df)}")
    print(f"  With current_score: {df['current_score'].notna().sum()}")
    print(f"  With final_score: {df['final_score'].notna().sum()}")
    print(f"  Current score range: {df['current_score'].min():.1f}% - {df['current_score'].max():.1f}%")
    print(f"  Final score range: {df['final_score'].min():.1f}% - {df['final_score'].max():.1f}%")

    # Define feature columns for correlation
    feature_cols = [
        # Activity features
        'page_views', 'page_views_level', 'participations', 'participations_level',
        'total_activity_time', 'activity_engagement',

        # Submission behavior
        'on_time', 'late', 'missing', 'floating',
        'on_time_rate', 'late_rate', 'missing_rate',
        'num_submissions', 'num_graded', 'submission_rate',

        # Score features (for partial grades analysis)
        'num_scores', 'avg_score', 'min_score', 'max_score', 'score_std', 'first_score',

        # Individual exam scores
        'exam_1_score', 'exam_2_score', 'exam_3_score', 'exam_4_score', 'exam_5_score',
    ]

    results = {
        'course_id': course_id,
        'course_name': course_name,
        'n_students': len(df),
        'correlations': {}
    }

    # Correlations with CURRENT SCORE (partial/running grade)
    if has_current and current_var >= MIN_GRADE_VARIANCE:
        print(f"\n  === Correlations with CURRENT SCORE ===")
        corr_current = calculate_correlations(df, 'current_score', feature_cols)
        if len(corr_current) > 0:
            corr_current = corr_current.sort_values('abs_pearson', ascending=False)

            print(f"\n  Top correlations with current_score:")
            for _, row in corr_current.head(10).iterrows():
                sig = "*" if row['significant'] else ""
                print(f"    {row['feature']:25s}  r={row['pearson_r']:+.3f} (p={row['pearson_p']:.4f}){sig}  n={row['n_samples']}")

            results['correlations']['current_score'] = corr_current.to_dict('records')

    # Correlations with FINAL SCORE
    if has_final and final_var >= MIN_GRADE_VARIANCE:
        print(f"\n  === Correlations with FINAL SCORE ===")
        corr_final = calculate_correlations(df, 'final_score', feature_cols)
        if len(corr_final) > 0:
            corr_final = corr_final.sort_values('abs_pearson', ascending=False)

            print(f"\n  Top correlations with final_score:")
            for _, row in corr_final.head(10).iterrows():
                sig = "*" if row['significant'] else ""
                print(f"    {row['feature']:25s}  r={row['pearson_r']:+.3f} (p={row['pearson_p']:.4f}){sig}  n={row['n_samples']}")

            results['correlations']['final_score'] = corr_final.to_dict('records')

    # Correlations with FIRST EXAM (for early prediction)
    if 'exam_1_score' in df.columns and df['exam_1_score'].notna().sum() >= MIN_STUDENTS:
        print(f"\n  === Correlations with FIRST EXAM ===")
        # Use only pre-exam features
        pre_exam_features = [
            'page_views', 'page_views_level', 'participations', 'participations_level',
            'total_activity_time', 'activity_engagement',
            'on_time', 'late', 'missing', 'on_time_rate', 'late_rate', 'missing_rate',
            'num_submissions', 'submission_rate',
        ]
        corr_exam1 = calculate_correlations(df, 'exam_1_score', pre_exam_features)
        if len(corr_exam1) > 0:
            corr_exam1 = corr_exam1.sort_values('abs_pearson', ascending=False)

            print(f"\n  Top correlations with first exam (early warning features):")
            for _, row in corr_exam1.head(10).iterrows():
                sig = "*" if row['significant'] else ""
                print(f"    {row['feature']:25s}  r={row['pearson_r']:+.3f} (p={row['pearson_p']:.4f}){sig}  n={row['n_samples']}")

            results['correlations']['exam_1_score'] = corr_exam1.to_dict('records')

    results['dataframe_stats'] = {
        'columns': list(df.columns),
        'n_features': len([c for c in feature_cols if c in df.columns]),
        'current_score_stats': {
            'mean': df['current_score'].mean(),
            'std': df['current_score'].std(),
            'min': df['current_score'].min(),
            'max': df['current_score'].max(),
        } if has_current else None,
        'final_score_stats': {
            'mean': df['final_score'].mean(),
            'std': df['final_score'].std(),
            'min': df['final_score'].min(),
            'max': df['final_score'].max(),
        } if has_final else None,
    }

    return results


def find_global_strongest_correlations(all_results):
    """Aggregate correlations across all courses to find strongest predictors"""

    print("\n" + "="*80)
    print("GLOBAL ANALYSIS: STRONGEST PREDICTORS ACROSS ALL COURSES")
    print("="*80)

    # Aggregate correlations by feature
    feature_correlations = {'current_score': {}, 'final_score': {}, 'exam_1_score': {}}

    for result in all_results:
        if result is None:
            continue

        for target in ['current_score', 'final_score', 'exam_1_score']:
            if target not in result['correlations']:
                continue

            for corr in result['correlations'][target]:
                feature = corr['feature']
                if feature not in feature_correlations[target]:
                    feature_correlations[target][feature] = []

                if corr['significant']:  # Only include significant correlations
                    feature_correlations[target][feature].append({
                        'r': corr['pearson_r'],
                        'n': corr['n_samples'],
                        'course_id': result['course_id'],
                        'course_name': result['course_name'],
                    })

    # Calculate average correlations per feature
    global_summary = {}

    for target in ['current_score', 'final_score', 'exam_1_score']:
        print(f"\n=== Best Predictors for {target.upper().replace('_', ' ')} ===")

        feature_avgs = []
        for feature, correlations in feature_correlations[target].items():
            if len(correlations) >= 2:  # Need at least 2 courses
                avg_r = np.mean([c['r'] for c in correlations])
                total_n = sum(c['n'] for c in correlations)
                feature_avgs.append({
                    'feature': feature,
                    'avg_r': avg_r,
                    'n_courses': len(correlations),
                    'total_n': total_n,
                    'abs_avg_r': abs(avg_r),
                })

        if feature_avgs:
            feature_avgs = sorted(feature_avgs, key=lambda x: x['abs_avg_r'], reverse=True)

            print(f"\n  {'Feature':<25} {'Avg r':>8} {'# Courses':>10} {'Total N':>10}")
            print(f"  {'-'*55}")
            for f in feature_avgs[:15]:
                print(f"  {f['feature']:<25} {f['avg_r']:>+8.3f} {f['n_courses']:>10} {f['total_n']:>10}")

            global_summary[target] = feature_avgs

    return global_summary


def main():
    """Main analysis function"""

    print("="*80)
    print("CORRELATION ANALYSIS - STUDENT PERFORMANCE PREDICTORS")
    print(f"Universidad Autónoma de Chile - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)

    # Load data
    data = load_data()

    # Get unique courses
    course_ids = set()
    for e in data['enrollments']:
        cid = e.get('course_id')
        if cid:
            course_ids.add(cid)

    print(f"\nFound {len(course_ids)} courses with enrollment data")

    # Build course name lookup
    course_names = {}
    for c in data['courses_raw']:
        course_names[c['id']] = c.get('name', 'Unknown')

    # Analyze each course
    all_results = []
    analyzed_count = 0

    for course_id in sorted(course_ids):
        course_name = course_names.get(course_id, 'Unknown')
        result = analyze_course(course_id, data, course_name)
        if result:
            all_results.append(result)
            analyzed_count += 1

    print(f"\n{'='*80}")
    print(f"SUMMARY: Analyzed {analyzed_count} courses out of {len(course_ids)}")
    print('='*80)

    # Global analysis
    global_summary = find_global_strongest_correlations(all_results)

    # Save results
    output = {
        'analysis_date': datetime.now().isoformat(),
        'courses_analyzed': analyzed_count,
        'courses_total': len(course_ids),
        'per_course_results': all_results,
        'global_summary': global_summary,
    }

    output_path = os.path.join(DATA_DIR, 'correlation_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")

    # Print key insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    if 'final_score' in global_summary and global_summary['final_score']:
        top_predictor = global_summary['final_score'][0]
        print(f"\n1. STRONGEST PREDICTOR of final grade: {top_predictor['feature']}")
        print(f"   Average correlation: r = {top_predictor['avg_r']:+.3f}")
        print(f"   Found significant in {top_predictor['n_courses']} courses")

    # Activity-only predictors
    activity_features = ['page_views', 'participations', 'on_time_rate', 'missing_rate',
                         'late_rate', 'total_activity_time', 'activity_engagement']

    if 'final_score' in global_summary:
        activity_predictors = [f for f in global_summary['final_score']
                               if f['feature'] in activity_features]
        if activity_predictors:
            best_activity = activity_predictors[0]
            print(f"\n2. BEST ACTIVITY-ONLY PREDICTOR (for early warning):")
            print(f"   Feature: {best_activity['feature']}")
            print(f"   Average correlation: r = {best_activity['avg_r']:+.3f}")
            print(f"   This can predict grades BEFORE any exams are given!")

    return output


if __name__ == "__main__":
    main()
