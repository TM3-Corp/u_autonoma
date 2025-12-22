"""
Phase 1: Baseline Model Training for Control de Gestión

This script trains prediction models on 7 courses with reliable grade data:
- 2 completed courses (use final_score): FUNDAMENTOS DE MICROECONOMÍA
- 5 ongoing courses (use current_score): Various courses with Canvas grades

The goal is to establish baseline model performance before expanding to
the full 21-course dataset with external grades.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import API_URL, API_TOKEN, DATA_DIR
from utils.pagination import (
    get_enrollments, get_student_summaries, get_submissions, get_assignments
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API headers
HEADERS = {'Authorization': f'Bearer {API_TOKEN}'}

# Constants
PASS_THRESHOLD = 57.0  # Chilean grading: 4.0/7.0 = ~57%
MIN_STUDENTS = 10  # Minimum students for model training

# Baseline courses configuration
BASELINE_COURSES = {
    # Completed courses (use final_score)
    84936: {'name': 'FUNDAMENTOS DE MICROECONOMÍA-P03', 'target': 'final_score', 'type': 'completed'},
    84941: {'name': 'FUNDAMENTOS DE MICROECONOMÍA-P01', 'target': 'final_score', 'type': 'completed'},
    # Ongoing courses (use current_score)
    86005: {'name': 'TALL DE COMPETENCIAS DIGITALES-P01', 'target': 'current_score', 'type': 'ongoing'},
    86020: {'name': 'TALL DE COMPETENCIAS DIGITALES-P02', 'target': 'current_score', 'type': 'ongoing'},
    86676: {'name': 'FUND DE BUSINESS ANALYTICS-P01', 'target': 'current_score', 'type': 'ongoing'},
    86689: {'name': 'GESTIÓN DEL TALENTO-P01', 'target': 'current_score', 'type': 'ongoing'},
    76755: {'name': 'PENSAMIENTO MATEMÁTICO-P03', 'target': 'current_score', 'type': 'ongoing'},
}

# Feature definitions
ACTIVITY_ONLY_FEATURES = [
    'page_views',
    'participations',
    'total_activity_time',
    'page_views_level',
    'participations_level',
]

ALL_DATA_FEATURES = ACTIVITY_ONLY_FEATURES + [
    'on_time', 'late', 'missing', 'floating',
    'on_time_rate', 'late_rate', 'missing_rate',
    'num_submissions', 'submission_rate',
    'avg_score', 'min_score', 'max_score', 'score_std',
    'first_score', 'num_graded', 'num_scores',
]


def extract_course_data(course_id: int) -> Dict[str, Any]:
    """Extract all data for a single course using pagination utility."""
    logger.info(f"Extracting data for course {course_id}")

    data = {
        'course_id': course_id,
        'extraction_time': datetime.now().isoformat(),
    }

    # Get enrollments with grades
    logger.info("  Fetching enrollments...")
    data['enrollments'] = get_enrollments(API_URL, HEADERS, course_id, include_grades=True)
    logger.info(f"  -> {len(data['enrollments'])} enrollments")

    # Get student summaries (activity)
    logger.info("  Fetching student summaries...")
    data['student_summaries'] = get_student_summaries(API_URL, HEADERS, course_id)
    logger.info(f"  -> {len(data['student_summaries'])} summaries")

    # Get submissions
    logger.info("  Fetching submissions...")
    data['submissions'] = get_submissions(API_URL, HEADERS, course_id)
    logger.info(f"  -> {len(data['submissions'])} submissions")

    # Get assignments
    logger.info("  Fetching assignments...")
    data['assignments'] = get_assignments(API_URL, HEADERS, course_id)
    logger.info(f"  -> {len(data['assignments'])} assignments")

    return data


def build_feature_dataframe(course_data: Dict, target_column: str) -> pd.DataFrame:
    """Build feature dataframe from extracted course data."""
    enrollments = course_data['enrollments']
    summaries = course_data['student_summaries']
    submissions = course_data['submissions']
    assignments = course_data['assignments']

    # Start with enrollments
    df = pd.DataFrame([
        {
            'user_id': e['user_id'],
            'current_score': e.get('grades', {}).get('current_score'),
            'final_score': e.get('grades', {}).get('final_score'),
            'total_activity_time': e.get('total_activity_time', 0) or 0,
        }
        for e in enrollments
        if e.get('type') == 'StudentEnrollment'
    ])

    if df.empty:
        return df

    # Add activity data from summaries
    summary_df = pd.DataFrame([
        {
            'user_id': s['id'],
            'page_views': s.get('page_views', 0) or 0,
            'page_views_level': s.get('page_views_level', 0) or 0,
            'participations': s.get('participations', 0) or 0,
            'participations_level': s.get('participations_level', 0) or 0,
            'on_time': s.get('tardiness_breakdown', {}).get('on_time', 0) or 0,
            'late': s.get('tardiness_breakdown', {}).get('late', 0) or 0,
            'missing': s.get('tardiness_breakdown', {}).get('missing', 0) or 0,
            'floating': s.get('tardiness_breakdown', {}).get('floating', 0) or 0,
        }
        for s in summaries
    ])

    if not summary_df.empty:
        df = df.merge(summary_df, on='user_id', how='left')

    # Process submissions per user
    sub_df = pd.DataFrame(submissions)
    if not sub_df.empty and 'user_id' in sub_df.columns:
        user_submissions = sub_df.groupby('user_id').agg({
            'score': ['count', 'mean', 'min', 'max', 'std', 'first'],
            'workflow_state': lambda x: (x == 'graded').sum()
        }).reset_index()

        user_submissions.columns = [
            'user_id', 'num_submissions', 'avg_score', 'min_score', 'max_score',
            'score_std', 'first_score', 'num_graded'
        ]

        # Count submissions with scores
        scores_df = sub_df[sub_df['score'].notna()].groupby('user_id').size().reset_index(name='num_scores')
        user_submissions = user_submissions.merge(scores_df, on='user_id', how='left')
        user_submissions['num_scores'] = user_submissions['num_scores'].fillna(0)

        df = df.merge(user_submissions, on='user_id', how='left')

    # Calculate derived features
    total_assignments = len(assignments) if assignments else 1
    df['submission_rate'] = df.get('num_submissions', 0) / max(total_assignments, 1)

    # Tardiness rates
    total_required = df['on_time'].fillna(0) + df['late'].fillna(0) + df['missing'].fillna(0)
    df['on_time_rate'] = np.where(total_required > 0, df['on_time'] / total_required, 0)
    df['late_rate'] = np.where(total_required > 0, df['late'] / total_required, 0)
    df['missing_rate'] = np.where(total_required > 0, df['missing'] / total_required, 0)

    # Fill NaN values
    for col in ALL_DATA_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Set target variable
    df['target'] = df[target_column]

    return df


def train_models(
    df: pd.DataFrame,
    features: List[str],
    feature_type: str
) -> Dict[str, Any]:
    """Train regression and classification models."""
    # Filter to students with target values
    df_valid = df[df['target'].notna()].copy()

    if len(df_valid) < MIN_STUDENTS:
        logger.warning(f"Insufficient samples ({len(df_valid)}), skipping {feature_type}")
        return None

    # Prepare features
    available_features = [f for f in features if f in df_valid.columns]
    X = df_valid[available_features].fillna(0)
    y = df_valid['target']

    # Create classification target
    y_class = (y >= PASS_THRESHOLD).astype(int)

    # Split data
    if len(df_valid) < 20:
        # Too few samples for reliable split
        X_train, X_test, y_train, y_test = X, X, y, y
        y_train_class, y_test_class = y_class, y_class
        logger.warning(f"Small dataset ({len(df_valid)}), using full data for training")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        y_train_class = (y_train >= PASS_THRESHOLD).astype(int)
        y_test_class = (y_test >= PASS_THRESHOLD).astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {
        'feature_type': feature_type,
        'n_samples': len(df_valid),
        'n_features': len(available_features),
        'features_used': available_features,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

    # Regression models
    results['regression'] = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    results['regression']['linear'] = {
        'r2': r2_score(y_test, y_pred_lr),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'mae': mean_absolute_error(y_test, y_pred_lr),
        'feature_importance': [
            {'feature': f, 'coefficient': c}
            for f, c in sorted(zip(available_features, lr.coef_), key=lambda x: abs(x[1]), reverse=True)[:10]
        ]
    }

    # Random Forest Regression
    rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    rf_reg.fit(X_train, y_train)
    y_pred_rf = rf_reg.predict(X_test)

    results['regression']['random_forest'] = {
        'r2': r2_score(y_test, y_pred_rf),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'mae': mean_absolute_error(y_test, y_pred_rf),
        'feature_importance': [
            {'feature': f, 'importance': i}
            for f, i in sorted(zip(available_features, rf_reg.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
        ]
    }

    # Classification models (if class diversity exists)
    results['classification'] = {}

    train_classes = set(y_train_class.unique())
    test_classes = set(y_test_class.unique())

    if len(train_classes) < 2:
        logger.warning(f"No class diversity in training set: {train_classes}")
        results['classification']['status'] = 'skipped_no_diversity'
        results['class_balance'] = {
            'pass': int((y_class == 1).sum()),
            'fail': int((y_class == 0).sum()),
            'pass_rate': float((y_class == 1).mean())
        }
        return results

    if len(test_classes) < 2:
        logger.warning(f"No class diversity in test set: {test_classes}")
        results['classification']['status'] = 'test_no_diversity'
        results['class_balance'] = {
            'pass': int((y_class == 1).sum()),
            'fail': int((y_class == 0).sum()),
            'pass_rate': float((y_class == 1).mean())
        }
        return results

    # Logistic Regression
    try:
        log_reg = LogisticRegression(max_iter=1000, random_state=42)
        log_reg.fit(X_train_scaled, y_train_class)
        y_pred_log = log_reg.predict(X_test_scaled)
        y_proba_log = log_reg.predict_proba(X_test_scaled)[:, 1]

        results['classification']['logistic'] = {
            'accuracy': accuracy_score(y_test_class, y_pred_log),
            'precision': precision_score(y_test_class, y_pred_log, zero_division=0),
            'recall': recall_score(y_test_class, y_pred_log, zero_division=0),
            'f1': f1_score(y_test_class, y_pred_log, zero_division=0),
            'auc': roc_auc_score(y_test_class, y_proba_log) if len(set(y_test_class)) > 1 else None,
            'confusion_matrix': confusion_matrix(y_test_class, y_pred_log).tolist()
        }
    except Exception as e:
        logger.error(f"Logistic regression failed: {e}")
        results['classification']['logistic'] = {'error': str(e)}

    # Random Forest Classification
    try:
        rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf_clf.fit(X_train, y_train_class)
        y_pred_rf_clf = rf_clf.predict(X_test)
        y_proba_rf_clf = rf_clf.predict_proba(X_test)[:, 1]

        results['classification']['random_forest'] = {
            'accuracy': accuracy_score(y_test_class, y_pred_rf_clf),
            'precision': precision_score(y_test_class, y_pred_rf_clf, zero_division=0),
            'recall': recall_score(y_test_class, y_pred_rf_clf, zero_division=0),
            'f1': f1_score(y_test_class, y_pred_rf_clf, zero_division=0),
            'auc': roc_auc_score(y_test_class, y_proba_rf_clf) if len(set(y_test_class)) > 1 else None,
            'confusion_matrix': confusion_matrix(y_test_class, y_pred_rf_clf).tolist(),
            'feature_importance': [
                {'feature': f, 'importance': i}
                for f, i in sorted(zip(available_features, rf_clf.feature_importances_), key=lambda x: x[1], reverse=True)[:10]
            ]
        }
    except Exception as e:
        logger.error(f"Random Forest classification failed: {e}")
        results['classification']['random_forest'] = {'error': str(e)}

    # Class balance
    results['class_balance'] = {
        'pass': int((y_class == 1).sum()),
        'fail': int((y_class == 0).sum()),
        'pass_rate': float((y_class == 1).mean())
    }

    return results


def process_course(course_id: int, course_config: Dict) -> Dict[str, Any]:
    """Process a single course: extract data, build features, train models."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {course_config['name']} (ID: {course_id})")
    logger.info(f"Type: {course_config['type']}, Target: {course_config['target']}")
    logger.info(f"{'='*60}")

    # Extract data
    course_data = extract_course_data(course_id)

    # Save raw data
    raw_data_path = os.path.join(DATA_DIR, 'baseline', f'course_{course_id}_raw.json')
    with open(raw_data_path, 'w') as f:
        json.dump(course_data, f, indent=2, default=str)
    logger.info(f"Saved raw data to {raw_data_path}")

    # Build feature dataframe
    df = build_feature_dataframe(course_data, course_config['target'])

    if df.empty:
        logger.warning(f"No data for course {course_id}")
        return None

    # Calculate statistics
    valid_targets = df['target'].dropna()

    result = {
        'course_id': course_id,
        'course_name': course_config['name'],
        'course_type': course_config['type'],
        'target_variable': course_config['target'],
        'n_students': len(df),
        'n_with_target': len(valid_targets),
        'target_mean': float(valid_targets.mean()) if len(valid_targets) > 0 else None,
        'target_std': float(valid_targets.std()) if len(valid_targets) > 0 else None,
        'pass_rate': float((valid_targets >= PASS_THRESHOLD).mean()) if len(valid_targets) > 0 else None,
        'extraction_time': course_data['extraction_time'],
    }

    # Train ALL-DATA model
    logger.info("\nTraining ALL-DATA model...")
    all_data_results = train_models(df, ALL_DATA_FEATURES, 'all_data')
    if all_data_results:
        result['all_data_model'] = all_data_results

    # Train ACTIVITY-ONLY model
    logger.info("\nTraining ACTIVITY-ONLY model...")
    activity_results = train_models(df, ACTIVITY_ONLY_FEATURES, 'activity_only')
    if activity_results:
        result['activity_only_model'] = activity_results

    return result


def aggregate_results(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate results across all courses."""
    valid_results = [r for r in results if r is not None]

    if not valid_results:
        return {}

    # Calculate averages for each model type
    summary = {
        'total_courses': len(valid_results),
        'total_students': sum(r['n_students'] for r in valid_results),
        'total_with_targets': sum(r['n_with_target'] for r in valid_results),
    }

    # ALL-DATA model averages
    all_data_r2 = [r['all_data_model']['regression']['random_forest']['r2']
                   for r in valid_results if r.get('all_data_model')]
    all_data_f1 = [r['all_data_model']['classification']['random_forest']['f1']
                   for r in valid_results
                   if r.get('all_data_model') and 'random_forest' in r['all_data_model'].get('classification', {})
                   and 'f1' in r['all_data_model']['classification']['random_forest']]

    summary['all_data'] = {
        'avg_r2': np.mean(all_data_r2) if all_data_r2 else None,
        'avg_f1': np.mean(all_data_f1) if all_data_f1 else None,
        'n_courses': len(all_data_r2),
    }

    # ACTIVITY-ONLY model averages
    activity_r2 = [r['activity_only_model']['regression']['random_forest']['r2']
                   for r in valid_results if r.get('activity_only_model')]
    activity_f1 = [r['activity_only_model']['classification']['random_forest']['f1']
                   for r in valid_results
                   if r.get('activity_only_model') and 'random_forest' in r['activity_only_model'].get('classification', {})
                   and 'f1' in r['activity_only_model']['classification']['random_forest']]

    summary['activity_only'] = {
        'avg_r2': np.mean(activity_r2) if activity_r2 else None,
        'avg_f1': np.mean(activity_f1) if activity_f1 else None,
        'n_courses': len(activity_r2),
    }

    return summary


def main():
    """Main execution function."""
    logger.info("="*70)
    logger.info("PHASE 1: BASELINE MODEL TRAINING")
    logger.info(f"Processing {len(BASELINE_COURSES)} courses")
    logger.info("="*70)

    results = []

    for course_id, config in BASELINE_COURSES.items():
        try:
            result = process_course(course_id, config)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Failed to process course {course_id}: {e}")
            import traceback
            traceback.print_exc()

    # Aggregate results
    summary = aggregate_results(results)

    # Prepare final output
    output = {
        'phase': 'Phase 1 - Baseline',
        'execution_time': datetime.now().isoformat(),
        'summary': summary,
        'courses': results,
    }

    # Save results
    output_path = os.path.join(DATA_DIR, 'baseline', 'baseline_models_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\n{'='*70}")
    logger.info("BASELINE TRAINING COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"\nSUMMARY:")
    logger.info(f"  Courses processed: {summary.get('total_courses', 0)}")
    logger.info(f"  Total students: {summary.get('total_students', 0)}")

    if summary.get('all_data'):
        logger.info(f"\n  ALL-DATA Model:")
        logger.info(f"    Avg R²: {summary['all_data'].get('avg_r2', 'N/A'):.3f}" if summary['all_data'].get('avg_r2') else "    Avg R²: N/A")
        logger.info(f"    Avg F1: {summary['all_data'].get('avg_f1', 'N/A'):.3f}" if summary['all_data'].get('avg_f1') else "    Avg F1: N/A")

    if summary.get('activity_only'):
        logger.info(f"\n  ACTIVITY-ONLY Model:")
        logger.info(f"    Avg R²: {summary['activity_only'].get('avg_r2', 'N/A'):.3f}" if summary['activity_only'].get('avg_r2') else "    Avg R²: N/A")
        logger.info(f"    Avg F1: {summary['activity_only'].get('avg_f1', 'N/A'):.3f}" if summary['activity_only'].get('avg_f1') else "    Avg F1: N/A")

    return output


if __name__ == '__main__':
    main()
