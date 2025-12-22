"""
Student Performance Prediction Models
=====================================

Two models:
1. ALL-DATA model: Uses ALL features including grades/scores
2. ACTIVITY-ONLY model: Uses ONLY pure activity features (no grades, no quizzes)

Both models predict:
- Continuous: final_score (percentage grade)
- Categorical: pass/fail (>= 57% = pass)

Activity-Only Features (truly no grades/quiz data):
- page_views: Total page views in course
- participations: Total participations (forum posts, etc.)
- total_activity_time: Time spent in course (seconds)
- page_views_level: Canvas normalized level (0-3)
- participations_level: Canvas normalized level (0-3)

Note: The following are EXCLUDED from activity-only because they track assignment submissions:
- on_time_rate, late_rate, missing_rate (these track assignment deadlines)
- num_submissions, submission_rate (these track assignment submissions)
- floating (assignments without due dates - still submission-related)
"""

import requests
import json
import os
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

from config import API_URL, API_TOKEN, DATA_DIR

headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Feature definitions
ACTIVITY_ONLY_FEATURES = [
    'page_views',
    'participations',
    'total_activity_time',
    'page_views_level',
    'participations_level',
]

# These features are related to assignment submissions (not pure activity)
SUBMISSION_FEATURES = [
    'on_time', 'late', 'missing', 'floating',
    'on_time_rate', 'late_rate', 'missing_rate',
    'num_submissions', 'submission_rate',
]

# Grade-related features (definitely excluded from activity-only)
GRADE_FEATURES = [
    'current_score', 'final_score',
    'avg_score', 'min_score', 'max_score', 'score_std',
    'first_score', 'num_graded', 'num_scores',
]

# All possible features for all-data model
ALL_FEATURES = ACTIVITY_ONLY_FEATURES + SUBMISSION_FEATURES + [
    'avg_score', 'min_score', 'max_score', 'score_std',
    'first_score', 'num_graded', 'num_scores',
]

PASS_THRESHOLD = 57.0  # Percentage for pass/fail classification

MIN_STUDENTS = 10
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
                'total_activity_time': e.get('total_activity_time', 0) or 0,
            }

    # Add summaries (activity metrics)
    for s in summaries:
        user_id = s.get('id')
        if user_id in students:
            tb = s.get('tardiness_breakdown', {}) or {}
            students[user_id].update({
                'page_views': s.get('page_views', 0) or 0,
                'page_views_level': s.get('page_views_level', 0) or 0,
                'participations': s.get('participations', 0) or 0,
                'participations_level': s.get('participations_level', 0) or 0,
                'on_time': tb.get('on_time', 0) or 0,
                'late': tb.get('late', 0) or 0,
                'missing': tb.get('missing', 0) or 0,
                'floating': tb.get('floating', 0) or 0,
            })

    # Process submissions for grade features
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

    # Ensure columns exist with defaults
    for col in ALL_FEATURES + ['current_score', 'final_score']:
        if col not in df.columns:
            df[col] = 0

    # Derived features
    total_assignments = len(assignments) if assignments else 1
    df['submission_rate'] = df['num_submissions'].fillna(0) / total_assignments

    total_req = df['on_time'].fillna(0) + df['late'].fillna(0) + df['missing'].fillna(0)
    df['on_time_rate'] = np.where(total_req > 0, df['on_time'].fillna(0) / total_req, 0)
    df['late_rate'] = np.where(total_req > 0, df['late'].fillna(0) / total_req, 0)
    df['missing_rate'] = np.where(total_req > 0, df['missing'].fillna(0) / total_req, 0)

    return df


def prepare_features(df, feature_list, target_col='final_score'):
    """Prepare X and y for modeling"""
    # Filter to rows with valid target
    valid = df[df[target_col].notna()].copy()

    if len(valid) < MIN_STUDENTS:
        return None, None

    # Get features that exist in the dataframe
    available_features = [f for f in feature_list if f in valid.columns]

    if not available_features:
        return None, None

    X = valid[available_features].fillna(0)
    y = valid[target_col]

    return X, y


def evaluate_regression(y_true, y_pred, model_name=""):
    """Evaluate regression model"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def evaluate_classification(y_true, y_pred, y_prob=None, model_name=""):
    """Evaluate classification model"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    result = {
        'model': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
            result['auc'] = auc
        except:
            result['auc'] = None

    return result


def train_models(X, y, feature_type="all"):
    """Train both regression and classification models"""
    results = {
        'feature_type': feature_type,
        'n_samples': len(y),
        'n_features': X.shape[1],
        'features_used': list(X.columns),
        'regression': {},
        'classification': {}
    }

    # Create pass/fail target
    y_class = (y >= PASS_THRESHOLD).astype(int)

    # Check class balance
    pass_count = y_class.sum()
    fail_count = len(y_class) - pass_count
    results['class_balance'] = {
        'pass': int(pass_count),
        'fail': int(fail_count),
        'pass_rate': float(pass_count / len(y_class))
    }

    # Train/test split
    if len(X) < 20:
        # Too few samples for split
        return results

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_class_train = (y_train >= PASS_THRESHOLD).astype(int)
    y_class_test = (y_test >= PASS_THRESHOLD).astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # === REGRESSION MODELS ===

    # Linear Regression
    try:
        lr = LinearRegression()
        lr.fit(X_train_scaled, y_train)
        y_pred = lr.predict(X_test_scaled)
        results['regression']['linear'] = evaluate_regression(y_test, y_pred, "Linear Regression")

        # Feature importances (coefficients)
        coef_df = pd.DataFrame({
            'feature': X.columns,
            'coefficient': lr.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        results['regression']['linear']['feature_importance'] = coef_df.head(10).to_dict('records')
    except Exception as e:
        results['regression']['linear'] = {'error': str(e)}

    # Random Forest Regression
    try:
        rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results['regression']['random_forest'] = evaluate_regression(y_test, y_pred, "Random Forest")

        # Feature importances
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        results['regression']['random_forest']['feature_importance'] = imp_df.head(10).to_dict('records')
    except Exception as e:
        results['regression']['random_forest'] = {'error': str(e)}

    # === CLASSIFICATION MODELS ===

    # Check if we have both classes
    if len(np.unique(y_class_train)) < 2 or len(np.unique(y_class_test)) < 2:
        results['classification']['note'] = "Insufficient class diversity for classification"
        return results

    # Logistic Regression
    try:
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train_scaled, y_class_train)
        y_pred = log_reg.predict(X_test_scaled)
        y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
        results['classification']['logistic'] = evaluate_classification(
            y_class_test, y_pred, y_prob, "Logistic Regression"
        )

        # Confusion matrix
        cm = confusion_matrix(y_class_test, y_pred)
        results['classification']['logistic']['confusion_matrix'] = cm.tolist()
    except Exception as e:
        results['classification']['logistic'] = {'error': str(e)}

    # Random Forest Classification
    try:
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_clf.fit(X_train, y_class_train)
        y_pred = rf_clf.predict(X_test)
        y_prob = rf_clf.predict_proba(X_test)[:, 1]
        results['classification']['random_forest'] = evaluate_classification(
            y_class_test, y_pred, y_prob, "Random Forest"
        )

        # Confusion matrix
        cm = confusion_matrix(y_class_test, y_pred)
        results['classification']['random_forest']['confusion_matrix'] = cm.tolist()

        # Feature importances
        imp_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_clf.feature_importances_
        }).sort_values('importance', ascending=False)
        results['classification']['random_forest']['feature_importance'] = imp_df.head(10).to_dict('records')
    except Exception as e:
        results['classification']['random_forest'] = {'error': str(e)}

    return results


def analyze_course(course_id, course_name=""):
    """Full analysis for a single course"""
    print(f"\n{'='*70}")
    print(f"COURSE {course_id}: {course_name[:50]}")
    print('='*70)

    # Fetch data
    print("  Fetching data from API...", end=" ")
    data = fetch_course_data(course_id)

    if not data['enrollments']:
        print("No enrollments")
        return None

    print(f"{len(data['enrollments'])} students, {len(data['submissions'])} submissions")

    # Build dataframe
    df = build_dataframe(data)

    if df is None or len(df) < MIN_STUDENTS:
        print(f"  Skipped: Not enough students")
        return None

    # Check grades
    final_valid = df['final_score'].notna().sum()
    if final_valid < MIN_STUDENTS:
        print(f"  Skipped: Not enough grade data ({final_valid} students with grades)")
        return None

    grade_var = df['final_score'].std()
    if grade_var < MIN_GRADE_VARIANCE:
        print(f"  Skipped: Low grade variance ({grade_var:.1f}%)")
        return None

    print(f"  Students with grades: {final_valid}")
    print(f"  Grade range: {df['final_score'].min():.1f}% - {df['final_score'].max():.1f}%")
    print(f"  Pass rate (≥{PASS_THRESHOLD}%): {(df['final_score'] >= PASS_THRESHOLD).mean()*100:.1f}%")

    results = {
        'course_id': course_id,
        'course_name': course_name,
        'n_students': len(df),
        'n_with_grades': final_valid,
        'grade_mean': float(df['final_score'].mean()),
        'grade_std': float(grade_var),
        'pass_rate': float((df['final_score'] >= PASS_THRESHOLD).mean()),
    }

    # === MODEL 1: ALL DATA ===
    print("\n  --- MODEL 1: ALL DATA ---")
    X_all, y_all = prepare_features(df, ALL_FEATURES, 'final_score')

    if X_all is not None and len(X_all) >= 20:
        all_results = train_models(X_all, y_all, "all_data")
        results['all_data_model'] = all_results

        if 'linear' in all_results.get('regression', {}):
            r2 = all_results['regression']['linear'].get('r2', 'N/A')
            print(f"    Linear R²: {r2:.3f}" if isinstance(r2, float) else f"    Linear R²: {r2}")

        if 'random_forest' in all_results.get('classification', {}):
            f1 = all_results['classification']['random_forest'].get('f1', 'N/A')
            print(f"    RF F1-score: {f1:.3f}" if isinstance(f1, float) else f"    RF F1: {f1}")
    else:
        print("    Insufficient data for all-data model")

    # === MODEL 2: ACTIVITY ONLY ===
    print("\n  --- MODEL 2: ACTIVITY ONLY ---")
    X_act, y_act = prepare_features(df, ACTIVITY_ONLY_FEATURES, 'final_score')

    if X_act is not None and len(X_act) >= 20:
        act_results = train_models(X_act, y_act, "activity_only")
        results['activity_only_model'] = act_results

        if 'linear' in act_results.get('regression', {}):
            r2 = act_results['regression']['linear'].get('r2', 'N/A')
            print(f"    Linear R²: {r2:.3f}" if isinstance(r2, float) else f"    Linear R²: {r2}")

        if 'random_forest' in act_results.get('classification', {}):
            f1 = act_results['classification']['random_forest'].get('f1', 'N/A')
            print(f"    RF F1-score: {f1:.3f}" if isinstance(f1, float) else f"    RF F1: {f1}")
    else:
        print("    Insufficient data for activity-only model")

    return results


def main():
    print("="*70)
    print("PREDICTION MODELS - ALL DATA vs ACTIVITY ONLY")
    print(f"Universidad Autónoma de Chile - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*70)

    print(f"\nPass/Fail threshold: {PASS_THRESHOLD}%")
    print(f"\nActivity-only features: {ACTIVITY_ONLY_FEATURES}")
    print(f"\nFeatures excluded from activity-only:")
    print(f"  - Grade features: {GRADE_FEATURES}")
    print(f"  - Submission features: {SUBMISSION_FEATURES}")

    # Get courses to analyze
    # Start with known high-potential courses from Control de Gestión
    courses_to_analyze = [
        (76755, "PENSAMIENTO MATEMÁTICO-P03"),
        (86005, "TALL DE COMPETENCIAS DIGITALES-P01"),
        (86020, "FORMACIÓN INTEGRAL II-P01"),
        (86676, "TALLER PENSAMIENTO ANALÍTICO-P01"),
        (86689, "INTROD A LA ING EN CONTROL DE GEST-P01"),
    ]

    # Also fetch courses from Account 719 (Control de Gestión)
    print("\nFetching additional courses from Control de Gestión...")
    more_courses = paginate(
        f'{API_URL}/api/v1/accounts/719/courses',
        {'include[]': ['total_students', 'term']}
    )

    for c in more_courses:
        if c.get('total_students', 0) >= MIN_STUDENTS:
            cid = c['id']
            if cid not in [x[0] for x in courses_to_analyze]:
                courses_to_analyze.append((cid, c.get('name', 'Unknown')))

    print(f"Total courses to analyze: {len(courses_to_analyze)}")

    # Analyze each course
    all_results = []

    for course_id, course_name in courses_to_analyze:
        result = analyze_course(course_id, course_name)
        if result:
            all_results.append(result)

    # === SUMMARY ===
    print("\n" + "="*70)
    print("SUMMARY: MODEL COMPARISON")
    print("="*70)

    if not all_results:
        print("\nNo courses had sufficient data for modeling.")
        return

    # Aggregate results
    all_data_r2 = []
    all_data_f1 = []
    activity_r2 = []
    activity_f1 = []

    for r in all_results:
        # All-data model
        if 'all_data_model' in r:
            reg = r['all_data_model'].get('regression', {})
            clf = r['all_data_model'].get('classification', {})

            if 'linear' in reg and 'r2' in reg['linear']:
                all_data_r2.append(reg['linear']['r2'])
            if 'random_forest' in clf and 'f1' in clf['random_forest']:
                all_data_f1.append(clf['random_forest']['f1'])

        # Activity-only model
        if 'activity_only_model' in r:
            reg = r['activity_only_model'].get('regression', {})
            clf = r['activity_only_model'].get('classification', {})

            if 'linear' in reg and 'r2' in reg['linear']:
                activity_r2.append(reg['linear']['r2'])
            if 'random_forest' in clf and 'f1' in clf['random_forest']:
                activity_f1.append(clf['random_forest']['f1'])

    print(f"\nCourses analyzed: {len(all_results)}")

    print("\n=== REGRESSION (Predicting Final Grade %) ===")
    print(f"{'Model':<20} {'Avg R²':>10} {'Courses':>10}")
    print("-"*42)
    if all_data_r2:
        print(f"{'All-Data':<20} {np.mean(all_data_r2):>10.3f} {len(all_data_r2):>10}")
    if activity_r2:
        print(f"{'Activity-Only':<20} {np.mean(activity_r2):>10.3f} {len(activity_r2):>10}")

    print("\n=== CLASSIFICATION (Predicting Pass/Fail) ===")
    print(f"{'Model':<20} {'Avg F1':>10} {'Courses':>10}")
    print("-"*42)
    if all_data_f1:
        print(f"{'All-Data':<20} {np.mean(all_data_f1):>10.3f} {len(all_data_f1):>10}")
    if activity_f1:
        print(f"{'Activity-Only':<20} {np.mean(activity_f1):>10.3f} {len(activity_f1):>10}")

    # Feature importance summary
    print("\n=== TOP FEATURES (Activity-Only Model) ===")
    feature_importance = {}
    for r in all_results:
        if 'activity_only_model' in r:
            clf = r['activity_only_model'].get('classification', {})
            if 'random_forest' in clf and 'feature_importance' in clf['random_forest']:
                for fi in clf['random_forest']['feature_importance']:
                    feat = fi['feature']
                    imp = fi['importance']
                    if feat not in feature_importance:
                        feature_importance[feat] = []
                    feature_importance[feat].append(imp)

    if feature_importance:
        print(f"{'Feature':<25} {'Avg Importance':>15}")
        print("-"*42)
        sorted_feats = sorted(feature_importance.items(),
                             key=lambda x: np.mean(x[1]), reverse=True)
        for feat, imps in sorted_feats[:10]:
            print(f"{feat:<25} {np.mean(imps):>15.4f}")

    # Save results
    output_path = os.path.join(DATA_DIR, 'prediction_models_results.json')
    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'pass_threshold': PASS_THRESHOLD,
            'activity_only_features': ACTIVITY_ONLY_FEATURES,
            'all_features': ALL_FEATURES,
            'courses_analyzed': len(all_results),
            'summary': {
                'all_data': {
                    'avg_r2': float(np.mean(all_data_r2)) if all_data_r2 else None,
                    'avg_f1': float(np.mean(all_data_f1)) if all_data_f1 else None,
                },
                'activity_only': {
                    'avg_r2': float(np.mean(activity_r2)) if activity_r2 else None,
                    'avg_f1': float(np.mean(activity_f1)) if activity_f1 else None,
                }
            },
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\n\nResults saved to: {output_path}")

    # Key insights
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)

    if activity_r2 and all_data_r2:
        r2_diff = np.mean(all_data_r2) - np.mean(activity_r2)
        print(f"\n1. Adding grade data improves R² by {r2_diff:.3f} on average")
        print(f"   - All-data R²: {np.mean(all_data_r2):.3f}")
        print(f"   - Activity-only R²: {np.mean(activity_r2):.3f}")

    if activity_f1 and all_data_f1:
        f1_diff = np.mean(all_data_f1) - np.mean(activity_f1)
        print(f"\n2. For pass/fail classification:")
        print(f"   - All-data F1: {np.mean(all_data_f1):.3f}")
        print(f"   - Activity-only F1: {np.mean(activity_f1):.3f}")
        print(f"   - Improvement from grades: {f1_diff:.3f}")

    if activity_r2:
        print(f"\n3. Activity-only model CAN predict grades (R²={np.mean(activity_r2):.3f})")
        print("   This enables EARLY WARNING before any grades exist!")


if __name__ == "__main__":
    main()
