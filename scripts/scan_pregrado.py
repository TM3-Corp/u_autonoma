#!/usr/bin/env python3
"""Scan Pregrado careers for high-potential courses."""

import requests
import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv('CANVAS_API_URL')
API_TOKEN = os.getenv('CANVAS_API_TOKEN')
headers = {'Authorization': f'Bearer {API_TOKEN}'}


def get_courses(account_id, term_id=336, min_students=15):
    """Get courses from account with minimum students."""
    courses = []
    url = f'{API_URL}/api/v1/accounts/{account_id}/courses'
    params = {
        'per_page': 100,
        'include[]': ['total_students', 'term'],
        'with_enrollments': True,
        'enrollment_term_id': term_id
    }

    while url:
        r = requests.get(url, headers=headers, params=params)
        if r.status_code != 200:
            break

        for c in r.json():
            if c.get('total_students', 0) >= min_students:
                courses.append({
                    'id': c['id'],
                    'name': c['name'],
                    'students': c.get('total_students', 0),
                    'account_id': c.get('account_id')
                })

        url = r.links.get('next', {}).get('url')
        params = {}

    return courses


def analyze_course(course_id):
    """Analyze course potential (grades, assignments, etc.)."""
    result = {
        'course_id': course_id,
        'has_grades': False,
        'n_students': 0,
        'grade_mean': None,
        'grade_std': None,
        'pass_rate': None,
        'n_assignments': 0,
        'recommendation': 'SKIP'
    }

    # Get enrollments with grades
    r = requests.get(
        f'{API_URL}/api/v1/courses/{course_id}/enrollments',
        headers=headers,
        params={'type[]': 'StudentEnrollment', 'per_page': 100, 'include[]': 'grades'}
    )
    if r.status_code != 200:
        return result

    enrollments = r.json()
    grades = [e['grades'].get('final_score') for e in enrollments
              if e.get('grades', {}).get('final_score') is not None and e['grades'].get('final_score') > 0]

    if len(grades) >= 10:
        result['has_grades'] = True
        result['n_students'] = len(grades)
        result['grade_mean'] = np.mean(grades)
        result['grade_std'] = np.std(grades)
        result['pass_rate'] = sum(1 for g in grades if g >= 57) / len(grades)

    # Count assignments
    r = requests.get(f'{API_URL}/api/v1/courses/{course_id}/assignments',
                     headers=headers, params={'per_page': 100})
    if r.status_code == 200:
        result['n_assignments'] = len(r.json())

    # Recommendation
    if result['has_grades'] and result['grade_std'] and result['grade_std'] > 10:
        if result['n_assignments'] >= 5 and result['pass_rate'] and 0.2 <= result['pass_rate'] <= 0.8:
            result['recommendation'] = 'HIGH'
        elif result['n_assignments'] >= 3:
            result['recommendation'] = 'MEDIUM'
        else:
            result['recommendation'] = 'LOW'
    elif result['has_grades']:
        result['recommendation'] = 'LOW-VAR'

    return result


def main():
    # Careers to scan (excluding Control de Gestión 719, 718)
    careers_to_scan = [
        (247, 'Psicología'),
        (248, 'Ing. Civil Informática'),
        (253, 'Derecho'),
        (255, 'Ing. Comercial'),
        (263, 'Ing. Civil Industrial'),
        (730, 'Ing. Civil Industrial (730)'),
        (249, 'Medicina'),
        (257, 'Enfermería'),
        (244, 'Odontología'),
        (254, 'Kinesiología'),
    ]

    print('SCANNING PREGRADO CAREERS (Term 336 - 2nd Sem 2025)')
    print('=' * 70)

    all_courses = []
    for acc_id, name in careers_to_scan:
        courses = get_courses(acc_id, term_id=336, min_students=15)
        print(f'{name} ({acc_id}): {len(courses)} courses')
        all_courses.extend(courses)

    print(f'\nTotal courses with 15+ students: {len(all_courses)}')

    if not all_courses:
        print('\nNo courses found. Trying term 322 (1st Sem 2025)...')
        for acc_id, name in careers_to_scan:
            courses = get_courses(acc_id, term_id=322, min_students=15)
            print(f'{name} ({acc_id}): {len(courses)} courses')
            all_courses.extend(courses)

    if not all_courses:
        print('\nNo courses found in either term.')
        return

    # Analyze top courses by enrollment
    print('\n' + '=' * 70)
    print('ANALYZING TOP COURSES FOR POTENTIAL')
    print('=' * 70)

    top_courses = sorted(all_courses, key=lambda x: x['students'], reverse=True)[:20]

    results = []
    for i, c in enumerate(top_courses):
        print(f'\n[{i+1}/{len(top_courses)}] Analyzing {c["id"]}: {c["name"][:40]}...')
        analysis = analyze_course(c['id'])
        analysis['course_name'] = c['name']
        analysis['enrolled'] = c['students']
        results.append(analysis)
        time.sleep(0.5)  # Rate limiting

    # Summary
    print('\n' + '=' * 70)
    print('RESULTS SUMMARY')
    print('=' * 70)

    high = [r for r in results if r['recommendation'] == 'HIGH']
    medium = [r for r in results if r['recommendation'] == 'MEDIUM']

    if high:
        print(f'\nHIGH POTENTIAL ({len(high)} courses):')
        print('-' * 70)
        for r in high:
            print(f"  {r['course_id']:6d} | {r['course_name'][:35]}")
            print(f"           Students: {r['n_students']}, Mean: {r['grade_mean']:.1f}%, StdDev: {r['grade_std']:.1f}")
            print(f"           Pass Rate: {r['pass_rate']:.0%}, Assignments: {r['n_assignments']}")
    else:
        print('\nNo HIGH potential courses found.')

    if medium:
        print(f'\nMEDIUM POTENTIAL ({len(medium)} courses):')
        print('-' * 70)
        for r in medium[:5]:
            print(f"  {r['course_id']:6d} | {r['course_name'][:35]}")
            print(f"           Students: {r['n_students']}, StdDev: {r['grade_std']:.1f}, Pass: {r['pass_rate']:.0%}")

    # Courses without grades
    no_grades = [r for r in results if not r['has_grades']]
    if no_grades:
        print(f'\nNO GRADES AVAILABLE ({len(no_grades)} courses):')
        for r in no_grades[:5]:
            print(f"  {r['course_id']:6d} | {r['course_name'][:45]}")


if __name__ == '__main__':
    main()
