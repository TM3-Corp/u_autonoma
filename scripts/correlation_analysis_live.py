"""
Live Correlation Analysis - Fetches fresh data from Canvas API
Analyzes ALL courses with sufficient grade data
"""

import requests
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from config import API_URL, API_TOKEN, HIGH_POTENTIAL_COURSES, DATA_DIR

headers = {'Authorization': f'Bearer {API_TOKEN}'}

MIN_STUDENTS = 10  # Lowered for more coverage
MIN_GRADE_VARIANCE = 3

def paginate(url, params=None):
    """Helper to paginate through Canvas API results"""
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
    all_results.extend(data)

    while 'next' in r.links:
        next_url = r.links['next']['url']
        r = requests.get(next_url, headers=headers)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        all_results.extend(data)

    return all_results


def fetch_course_data(course_id):
    """Fetch all relevant data for a course"""
    data = {'course_id': course_id}

    # 1. Enrollments with grades
    enrollments = paginate(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        {'type[]': 'StudentEnrollment', 'include[]': ['grades', 'total_scores']}
    )
    data['enrollments'] = enrollments

    # 2. Student summaries (activity)
    summaries = paginate(
        f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries'
    )
    data['summaries'] = summaries

    # 3. Submissions with scores
    submissions = paginate(
        f'{API_URL}/api/v1/courses/{course_id}/students/submissions',
        {'student_ids[]': 'all'}
    )
    data['submissions'] = submissions

    # 4. Assignments
    assignments = paginate(f'{API_URL}/api/v1/courses/{course_id}/assignments')
    data['assignments'] = assignments

    return data


def build_dataframe(course_data):
    """Build feature dataframe from course data"""
    enrollments = course_data['enrollments']
    summaries = course_data['summaries']
    submissions = course_data['submissions']
    assignments = course_data['assignments']

    if not enrollments:
        return None

    # Base: enrollments with grades
    students = {}
    for e in enrollments:
        user_id = e.get('user_id')
        if user_id and e.get('enrollment_state') == 'active':
            grades = e.get('grades', {}) or {}
            students[user_id] = {
                'user_id': user_id,
                'current_score': grades.get('current_score'),
                'final_score': grades.get('final_score'),
                'total_activity_time': e.get('total_activity_time', 0),
            }

    # Add summaries
    for s in summaries:
        user_id = s.get('id')
        if user_id in students:
            tb = s.get('tardiness_breakdown', {}) or {}
            students[user_id].update({
                'page_views': s.get('page_views', 0),
                'page_views_level': s.get('page_views_level', 0),
                'participations': s.get('participations', 0),
                'participations_level': s.get('participations_level', 0),
                'on_time': tb.get('on_time', 0),
                'late': tb.get('late', 0),
                'missing': tb.get('missing', 0),
                'floating': tb.get('floating', 0),
            })

    # Process submissions
    student_subs = {}
    for sub in submissions:
        uid = sub.get('user_id')
        if uid not in student_subs:
            student_subs[uid] = []
        student_subs[uid].append(sub)

    for uid, subs in student_subs.items():
        if uid in students:
            scores = [s.get('score') for s in subs if s.get('score') is not None]
            submitted = [s for s in subs if s.get('submitted_at')]
            graded = [s for s in subs if s.get('workflow_state') == 'graded']

            students[uid].update({
                'num_submissions': len(submitted),
                'num_graded': len(graded),
                'num_scores': len(scores),
                'avg_score': np.mean(scores) if scores else None,
                'min_score': np.min(scores) if scores else None,
                'max_score': np.max(scores) if scores else None,
                'score_std': np.std(scores) if len(scores) > 1 else 0,
            })

            # First score
            if scores:
                sorted_subs = sorted([s for s in subs if s.get('score') is not None],
                                     key=lambda x: x.get('assignment_id', 0))
                if sorted_subs:
                    students[uid]['first_score'] = sorted_subs[0].get('score')

    # Build DF
    df = pd.DataFrame(list(students.values()))

    if len(df) == 0:
        return None

    # Ensure columns exist
    for col in ['num_submissions', 'num_graded', 'num_scores', 'on_time', 'late', 'missing',
                'floating', 'page_views', 'participations', 'total_activity_time']:
        if col not in df.columns:
            df[col] = 0

    # Derived features
    total_assignments = len(assignments) if assignments else 1
    df['submission_rate'] = df['num_submissions'].fillna(0) / total_assignments

    total_req = df['on_time'].fillna(0) + df['late'].fillna(0) + df['missing'].fillna(0)
    df['on_time_rate'] = np.where(total_req > 0, df['on_time'].fillna(0) / total_req, 0)
    df['late_rate'] = np.where(total_req > 0, df['late'].fillna(0) / total_req, 0)
    df['missing_rate'] = np.where(total_req > 0, df['missing'].fillna(0) / total_req, 0)

    pv_max = df['page_views'].max() if df['page_views'].max() > 0 else 1
    part_max = df['participations'].max() if df['participations'].max() > 0 else 1
    df['activity_engagement'] = (df['page_views'].fillna(0) / pv_max) * 0.5 + \
                                (df['participations'].fillna(0) / part_max) * 0.5

    return df


def calculate_correlations(df, target_col, feature_cols):
    """Calculate correlations with significance testing"""
    results = []

    for col in feature_cols:
        if col not in df.columns or col == target_col:
            continue

        valid = df[[col, target_col]].dropna()
        if len(valid) < 5:
            continue
        if valid[col].std() == 0 or valid[target_col].std() == 0:
            continue

        pearson_r, pearson_p = stats.pearsonr(valid[col], valid[target_col])
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


def analyze_course_live(course_id, course_name=""):
    """Fetch fresh data and analyze correlations"""
    print(f"\n{'='*70}")
    print(f"COURSE {course_id}: {course_name[:50]}")
    print('='*70)

    # Fetch live data
    print("  Fetching data from API...", end=" ")
    data = fetch_course_data(course_id)

    if not data['enrollments']:
        print("No enrollments")
        return None

    print(f"{len(data['enrollments'])} students, {len(data['submissions'])} submissions")

    # Build dataframe
    df = build_dataframe(data)

    if df is None or len(df) < MIN_STUDENTS:
        print(f"  Skipped: Not enough students (need {MIN_STUDENTS}, have {len(df) if df is not None else 0})")
        return None

    # Check grades
    has_current = df['current_score'].notna().sum() >= MIN_STUDENTS
    has_final = df['final_score'].notna().sum() >= MIN_STUDENTS

    if not has_current and not has_final:
        print("  Skipped: No grade data")
        return None

    current_var = df['current_score'].std() if has_current else 0
    final_var = df['final_score'].std() if has_final else 0

    if current_var < MIN_GRADE_VARIANCE and final_var < MIN_GRADE_VARIANCE:
        print(f"  Skipped: Low variance (curr={current_var:.1f}%, final={final_var:.1f}%)")
        return None

    print(f"  Students: {len(df)} | Current scores: {df['current_score'].notna().sum()} | Final: {df['final_score'].notna().sum()}")
    print(f"  Grade range: {df['current_score'].min():.1f}%-{df['current_score'].max():.1f}% (current)")
    print(f"               {df['final_score'].min():.1f}%-{df['final_score'].max():.1f}% (final)")

    # Feature columns
    feature_cols = [
        'page_views', 'page_views_level', 'participations', 'participations_level',
        'total_activity_time', 'activity_engagement',
        'on_time', 'late', 'missing', 'floating',
        'on_time_rate', 'late_rate', 'missing_rate',
        'num_submissions', 'num_graded', 'submission_rate',
        'num_scores', 'avg_score', 'min_score', 'max_score', 'score_std', 'first_score',
    ]

    results = {
        'course_id': course_id,
        'course_name': course_name,
        'n_students': len(df),
        'correlations': {}
    }

    # Current score correlations
    if has_current and current_var >= MIN_GRADE_VARIANCE:
        print("\n  --- CURRENT SCORE (running grade) ---")
        corr = calculate_correlations(df, 'current_score', feature_cols)
        if len(corr) > 0:
            corr = corr.sort_values('abs_pearson', ascending=False)
            for _, row in corr.head(8).iterrows():
                sig = "*" if row['significant'] else ""
                print(f"    {row['feature']:<22} r={row['pearson_r']:+.3f}{sig}")
            results['correlations']['current_score'] = corr.to_dict('records')

    # Final score correlations
    if has_final and final_var >= MIN_GRADE_VARIANCE:
        print("\n  --- FINAL SCORE ---")
        corr = calculate_correlations(df, 'final_score', feature_cols)
        if len(corr) > 0:
            corr = corr.sort_values('abs_pearson', ascending=False)
            for _, row in corr.head(8).iterrows():
                sig = "*" if row['significant'] else ""
                print(f"    {row['feature']:<22} r={row['pearson_r']:+.3f}{sig}")
            results['correlations']['final_score'] = corr.to_dict('records')

    return results


def main():
    print("="*70)
    print("LIVE CORRELATION ANALYSIS - ALL COURSES WITH GRADE DATA")
    print(f"Universidad Autónoma de Chile - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    # Get course list from API
    print("\nFetching course list from API...")
    courses = paginate(
        f'{API_URL}/api/v1/accounts/719/courses',
        {'include[]': ['total_students', 'term']}
    )

    # Also add directly accessible courses
    direct_courses = [76755, 86005, 86020, 86676, 86689, 82725, 84939, 84947]
    course_ids_to_check = set([c['id'] for c in courses] + direct_courses + HIGH_POTENTIAL_COURSES)

    print(f"Checking {len(course_ids_to_check)} courses...")

    # Course name lookup
    course_names = {c['id']: c.get('name', 'Unknown') for c in courses}

    # Analyze each course
    all_results = []

    for course_id in sorted(course_ids_to_check):
        name = course_names.get(course_id, 'Unknown')
        result = analyze_course_live(course_id, name)
        if result:
            all_results.append(result)

    print(f"\n{'='*70}")
    print(f"ANALYSIS COMPLETE: {len(all_results)} courses with sufficient data")
    print('='*70)

    if not all_results:
        print("No courses had sufficient data for analysis.")
        return

    # Aggregate findings
    print("\n" + "="*70)
    print("GLOBAL FINDINGS: STRONGEST PREDICTORS")
    print("="*70)

    for target in ['current_score', 'final_score']:
        feature_data = {}

        for result in all_results:
            if target not in result['correlations']:
                continue
            for corr in result['correlations'][target]:
                if corr['significant']:
                    feat = corr['feature']
                    if feat not in feature_data:
                        feature_data[feat] = []
                    feature_data[feat].append({
                        'r': corr['pearson_r'],
                        'n': corr['n_samples'],
                        'course': result['course_name'][:30]
                    })

        print(f"\n=== BEST PREDICTORS FOR {target.upper().replace('_', ' ')} ===\n")
        print(f"{'Feature':<25} {'Avg r':>8} {'Courses':>8} {'Total N':>8}")
        print("-"*52)

        feature_avgs = []
        for feat, data in feature_data.items():
            if len(data) >= 2:
                avg_r = np.mean([d['r'] for d in data])
                feature_avgs.append({
                    'feature': feat,
                    'avg_r': avg_r,
                    'n_courses': len(data),
                    'total_n': sum(d['n'] for d in data),
                })

        for f in sorted(feature_avgs, key=lambda x: abs(x['avg_r']), reverse=True)[:12]:
            print(f"{f['feature']:<25} {f['avg_r']:>+8.3f} {f['n_courses']:>8} {f['total_n']:>8}")

    # Save results
    output_path = os.path.join(DATA_DIR, 'correlation_analysis_live.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'courses_analyzed': len(all_results),
            'results': all_results,
        }, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # Print key takeaways
    print("\n" + "="*70)
    print("KEY TAKEAWAYS FOR EARLY WARNING SYSTEM")
    print("="*70)

    # Activity-only features for early prediction
    activity_features = ['page_views', 'participations', 'on_time_rate', 'missing_rate',
                         'late_rate', 'total_activity_time', 'activity_engagement',
                         'page_views_level', 'participations_level', 'on_time', 'missing', 'late']

    print("\nActivity-only features (can predict BEFORE any grades exist):")
    for target in ['current_score', 'final_score']:
        feature_data = {}
        for result in all_results:
            if target not in result['correlations']:
                continue
            for corr in result['correlations'][target]:
                if corr['significant'] and corr['feature'] in activity_features:
                    feat = corr['feature']
                    if feat not in feature_data:
                        feature_data[feat] = []
                    feature_data[feat].append(corr['pearson_r'])

        if feature_data:
            print(f"\n  {target}:")
            for feat, rs in sorted(feature_data.items(), key=lambda x: abs(np.mean(x[1])), reverse=True)[:5]:
                avg = np.mean(rs)
                print(f"    {feat:<20} avg r = {avg:+.3f} ({len(rs)} courses)")


if __name__ == "__main__":
    main()
