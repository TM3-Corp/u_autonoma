#!/usr/bin/env python3
"""
EARLY WARNING SYSTEM FOR STUDENT FAILURE PREDICTION

Universidad Autónoma de Chile - Canvas LMS Analysis

This system aims to identify at-risk students EARLY - before the first exam,
when intervention can still make a difference. Every student flagged correctly
is a potential life changed.

Key Principles:
1. Use only data available EARLY in the semester
2. Prioritize recall (catching at-risk students) over precision
3. Generate actionable insights for intervention
4. Respect student privacy while enabling support

Based on research:
- Oviedo et al.: 80.1% accuracy at 10% course completion
- Our findings: Morning studiers avg 97.8%, Evening studiers avg 31.5%
- Early module access correlates with success (r=0.229, p=0.051)
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

# ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, roc_auc_score,
    f1_score, recall_score, precision_score
)
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/paul/projects/uautonoma/scripts')
from config import API_URL, API_TOKEN, DATA_DIR
from utils.pagination import paginate_canvas

headers = {'Authorization': f'Bearer {API_TOKEN}'}

# Courses with complete data
ANALYSIS_COURSES = [84936, 84941]  # FUNDAMENTOS DE MICROECONOMÍA P03, P01

# Failure threshold (Chilean grading: 4.0/7.0 = 57%)
FAIL_THRESHOLD = 57.0


class EarlyWarningSystem:
    """
    Early Warning System for predicting student failure.

    Designed to identify at-risk students early enough for intervention.
    """

    def __init__(self, courses: List[int]):
        self.courses = courses
        self.students_data = []
        self.features_df = None
        self.models = {}

    def extract_comprehensive_features(self) -> pd.DataFrame:
        """
        Extract all available features from Canvas API.

        Features are categorized by when they become available:
        - WEEK 1: First login, initial page views
        - WEEK 2-3: Early module progress, assignment engagement
        - WEEK 4+: Patterns, trends, early grades
        """
        print("=" * 70)
        print("EXTRACTING COMPREHENSIVE FEATURES FOR EARLY WARNING")
        print("=" * 70)

        all_students = []

        for course_id in self.courses:
            print(f"\n{'='*50}")
            print(f"Processing Course: {course_id}")
            print("=" * 50)

            # 1. Get course info
            course_info = self._get_course_info(course_id)
            course_start = course_info.get('start_at')
            print(f"Course: {course_info.get('name', 'Unknown')}")
            print(f"Start date: {course_start}")

            # 2. Get enrollments with grades
            print("\nFetching enrollments...")
            enrollments = paginate_canvas(
                f'{API_URL}/api/v1/courses/{course_id}/enrollments',
                headers,
                params={'type[]': 'StudentEnrollment', 'include[]': ['grades', 'total_scores']}
            )
            print(f"  Found {len(enrollments)} students")

            # 3. Get modules
            print("\nFetching modules...")
            modules = self._get_modules(course_id)
            print(f"  Found {len(modules)} modules")

            # 4. Get assignments
            print("\nFetching assignments...")
            assignments = self._get_assignments(course_id)
            print(f"  Found {len(assignments)} assignments")

            # 5. Get student summaries (aggregate activity)
            print("\nFetching student summaries...")
            summaries = self._get_student_summaries(course_id)
            summaries_dict = {s['id']: s for s in summaries}
            print(f"  Found {len(summaries)} summaries")

            # 6. Process each student
            print("\nExtracting per-student features...")
            for i, enrollment in enumerate(enrollments):
                if (i + 1) % 10 == 0:
                    print(f"  Progress: {i+1}/{len(enrollments)}")

                user_id = enrollment['user_id']

                # Get student-specific data
                student_features = self._extract_student_features(
                    course_id=course_id,
                    user_id=user_id,
                    enrollment=enrollment,
                    summary=summaries_dict.get(user_id, {}),
                    modules=modules,
                    assignments=assignments,
                    course_start=course_start
                )

                all_students.append(student_features)

        self.features_df = pd.DataFrame(all_students)
        print(f"\n{'='*70}")
        print(f"EXTRACTION COMPLETE: {len(self.features_df)} students")
        print("=" * 70)

        return self.features_df

    def _get_course_info(self, course_id: int) -> dict:
        """Get course metadata."""
        response = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}',
            headers=headers,
            timeout=30
        )
        return response.json() if response.status_code == 200 else {}

    def _get_modules(self, course_id: int) -> list:
        """Get all modules for a course."""
        response = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/modules',
            headers=headers,
            params={'include[]': 'items', 'per_page': 100},
            timeout=30
        )
        return response.json() if response.status_code == 200 else []

    def _get_assignments(self, course_id: int) -> list:
        """Get all assignments for a course."""
        response = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/assignments',
            headers=headers,
            params={'per_page': 100, 'order_by': 'due_at'},
            timeout=30
        )
        return response.json() if response.status_code == 200 else []

    def _get_student_summaries(self, course_id: int) -> list:
        """Get activity summaries for all students."""
        response = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/analytics/student_summaries',
            headers=headers,
            params={'per_page': 100},
            timeout=30
        )
        return response.json() if response.status_code == 200 else []

    def _get_student_modules(self, course_id: int, user_id: int) -> list:
        """Get module progress for specific student."""
        response = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/modules',
            headers=headers,
            params={'student_id': user_id},
            timeout=30
        )
        return response.json() if response.status_code == 200 else []

    def _get_student_activity(self, course_id: int, user_id: int) -> dict:
        """Get hourly activity for specific student."""
        response = requests.get(
            f'{API_URL}/api/v1/courses/{course_id}/analytics/users/{user_id}/activity',
            headers=headers,
            timeout=30
        )
        return response.json() if response.status_code == 200 else {}

    def _extract_student_features(
        self,
        course_id: int,
        user_id: int,
        enrollment: dict,
        summary: dict,
        modules: list,
        assignments: list,
        course_start: Optional[str]
    ) -> dict:
        """
        Extract comprehensive features for a single student.

        Features are organized by:
        1. ENROLLMENT: Basic enrollment data
        2. ACTIVITY: Page views, participations
        3. TIMING: When they access, early vs late
        4. MODULES: Progress through course structure
        5. ASSIGNMENTS: Submission patterns
        6. OUTCOME: Final grade (target variable)
        """
        features = {
            'course_id': course_id,
            'user_id': user_id,
        }

        # === ENROLLMENT FEATURES ===
        features['total_activity_time'] = enrollment.get('total_activity_time', 0)
        features['last_activity_at'] = enrollment.get('last_activity_at')

        # === GRADE (TARGET) ===
        grades = enrollment.get('grades', {})
        features['current_score'] = grades.get('current_score')
        features['final_score'] = grades.get('final_score')
        features['failed'] = 1 if (features['final_score'] or 0) < FAIL_THRESHOLD else 0

        # === ACTIVITY FEATURES (from summaries) ===
        features['page_views'] = summary.get('page_views', 0)
        features['page_views_level'] = summary.get('page_views_level', 0)
        features['participations'] = summary.get('participations', 0)
        features['participations_level'] = summary.get('participations_level', 0)

        # Tardiness breakdown
        tardiness = summary.get('tardiness_breakdown', {})
        features['on_time'] = tardiness.get('on_time', 0)
        features['late'] = tardiness.get('late', 0)
        features['missing'] = tardiness.get('missing', 0)
        features['floating'] = tardiness.get('floating', 0)

        # Derived tardiness features
        total_assignments = features['on_time'] + features['late'] + features['missing']
        if total_assignments > 0:
            features['on_time_rate'] = features['on_time'] / total_assignments
            features['late_rate'] = features['late'] / total_assignments
            features['missing_rate'] = features['missing'] / total_assignments
        else:
            features['on_time_rate'] = 0
            features['late_rate'] = 0
            features['missing_rate'] = 0

        # === MODULE FEATURES ===
        student_modules = self._get_student_modules(course_id, user_id)

        completed_modules = [m for m in student_modules if m.get('state') == 'completed']
        features['modules_completed'] = len(completed_modules)
        features['modules_total'] = len(student_modules)
        features['module_completion_rate'] = (
            len(completed_modules) / len(student_modules)
            if student_modules else 0
        )

        # Module completion timing
        completion_times = []
        for m in completed_modules:
            if m.get('completed_at'):
                try:
                    ct = datetime.fromisoformat(m['completed_at'].replace('Z', '+00:00'))
                    completion_times.append(ct)
                except:
                    pass

        if completion_times:
            features['first_module_completed_at'] = min(completion_times).isoformat()
            features['last_module_completed_at'] = max(completion_times).isoformat()
            features['module_completion_span_days'] = (max(completion_times) - min(completion_times)).days
        else:
            features['first_module_completed_at'] = None
            features['last_module_completed_at'] = None
            features['module_completion_span_days'] = 0

        # === ACTIVITY TIMING FEATURES ===
        activity = self._get_student_activity(course_id, user_id)
        page_views_hourly = activity.get('page_views', {})

        # Parse timestamps and calculate time-of-day distribution
        hour_counts = {'morning': 0, 'afternoon': 0, 'evening': 0, 'night': 0}
        all_timestamps = []
        total_views = 0

        for ts_str, count in page_views_hourly.items():
            try:
                ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
                all_timestamps.append(ts)
                total_views += count
                hour = ts.hour

                if 6 <= hour < 12:
                    hour_counts['morning'] += count
                elif 12 <= hour < 18:
                    hour_counts['afternoon'] += count
                elif 18 <= hour < 22:
                    hour_counts['evening'] += count
                else:
                    hour_counts['night'] += count
            except:
                pass

        features['morning_activity'] = hour_counts['morning']
        features['afternoon_activity'] = hour_counts['afternoon']
        features['evening_activity'] = hour_counts['evening']
        features['night_activity'] = hour_counts['night']

        # Dominant time of day
        if total_views > 0:
            features['dominant_time'] = max(hour_counts, key=hour_counts.get)
            features['is_morning_studier'] = 1 if hour_counts['morning'] == max(hour_counts.values()) else 0
            features['is_evening_studier'] = 1 if hour_counts['evening'] == max(hour_counts.values()) else 0
        else:
            features['dominant_time'] = 'none'
            features['is_morning_studier'] = 0
            features['is_evening_studier'] = 0

        # Activity timing
        if all_timestamps:
            features['first_activity_at'] = min(all_timestamps).isoformat()
            features['last_detailed_activity_at'] = max(all_timestamps).isoformat()
            features['activity_span_days'] = (max(all_timestamps) - min(all_timestamps)).days
            features['unique_active_hours'] = len(page_views_hourly)
        else:
            features['first_activity_at'] = None
            features['last_detailed_activity_at'] = None
            features['activity_span_days'] = 0
            features['unique_active_hours'] = 0

        # Days until first activity (from course start)
        if course_start and features['first_activity_at']:
            try:
                start = datetime.fromisoformat(course_start.replace('Z', '+00:00'))
                first = datetime.fromisoformat(features['first_activity_at'])
                features['days_to_first_activity'] = (first - start).days
            except:
                features['days_to_first_activity'] = None
        else:
            features['days_to_first_activity'] = None

        # Participations (submissions, quiz attempts)
        participations = activity.get('participations', [])
        features['participation_count'] = len(participations)

        return features

    def calculate_early_access_scores(self) -> None:
        """
        Calculate early access scores (who accessed modules first).

        This is a key Oviedo-style metric: students who engage early
        tend to succeed more often.
        """
        if self.features_df is None:
            raise ValueError("Must extract features first")

        print("\nCalculating early access scores...")

        # For each course, rank students by first module completion
        for course_id in self.courses:
            course_mask = self.features_df['course_id'] == course_id
            course_df = self.features_df[course_mask].copy()

            # Parse first module completion times
            times = []
            for idx, row in course_df.iterrows():
                if row['first_module_completed_at']:
                    try:
                        t = datetime.fromisoformat(row['first_module_completed_at'])
                        times.append((idx, t))
                    except:
                        times.append((idx, None))
                else:
                    times.append((idx, None))

            # Rank by time (earlier = lower rank)
            valid_times = [(idx, t) for idx, t in times if t is not None]
            valid_times.sort(key=lambda x: x[1])

            # Assign ranks
            for rank, (idx, _) in enumerate(valid_times):
                normalized_rank = rank / (len(valid_times) - 1) if len(valid_times) > 1 else 0.5
                self.features_df.loc[idx, 'early_access_rank'] = rank + 1
                self.features_df.loc[idx, 'early_access_score'] = 1 - normalized_rank  # Higher = earlier

            # Students without completion get score 0
            for idx, t in times:
                if t is None:
                    self.features_df.loc[idx, 'early_access_rank'] = len(valid_times) + 1
                    self.features_df.loc[idx, 'early_access_score'] = 0

        print(f"  Early access scores calculated for {len(self.features_df)} students")

    def analyze_feature_importance(self) -> pd.DataFrame:
        """
        Analyze which features are most predictive of failure.

        This helps identify what behaviors to look for in early warning.
        """
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)

        # Features to analyze (numeric only)
        feature_cols = [
            'page_views', 'page_views_level', 'participations', 'participations_level',
            'total_activity_time', 'on_time', 'late', 'missing',
            'on_time_rate', 'late_rate', 'missing_rate',
            'modules_completed', 'module_completion_rate',
            'morning_activity', 'afternoon_activity', 'evening_activity', 'night_activity',
            'is_morning_studier', 'is_evening_studier',
            'activity_span_days', 'unique_active_hours',
            'early_access_score'
        ]

        # Filter to available features
        available_features = [f for f in feature_cols if f in self.features_df.columns]

        # Calculate correlations with failure
        correlations = []
        for feat in available_features:
            valid = self.features_df[[feat, 'failed']].dropna()
            if len(valid) > 5:
                corr = valid[feat].corr(valid['failed'])
                correlations.append({
                    'feature': feat,
                    'correlation_with_failure': corr,
                    'abs_correlation': abs(corr),
                    'direction': 'increases failure' if corr > 0 else 'decreases failure'
                })

        corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)

        print("\nFeature Correlations with Failure (sorted by importance):")
        print("-" * 60)
        for _, row in corr_df.iterrows():
            direction = "↑ RISK" if row['correlation_with_failure'] > 0 else "↓ RISK"
            print(f"  {row['feature']:30s} r={row['correlation_with_failure']:+.3f} {direction}")

        return corr_df

    def train_early_warning_models(self) -> dict:
        """
        Train models optimized for early warning.

        Key considerations:
        - Prioritize RECALL (catch at-risk students)
        - Use only features available early
        - Balance sensitivity vs false alarms
        """
        print("\n" + "=" * 70)
        print("TRAINING EARLY WARNING MODELS")
        print("=" * 70)

        # Early warning features (available in first 2-3 weeks)
        early_features = [
            'page_views', 'page_views_level',
            'total_activity_time',
            'morning_activity', 'afternoon_activity', 'evening_activity', 'night_activity',
            'is_morning_studier', 'is_evening_studier',
            'activity_span_days', 'unique_active_hours',
            'early_access_score'
        ]

        # Filter to available features
        available = [f for f in early_features if f in self.features_df.columns]

        # Prepare data
        df = self.features_df.dropna(subset=['failed'])
        X = df[available].fillna(0)
        y = df['failed']

        print(f"\nTraining on {len(X)} students")
        print(f"Features: {available}")
        print(f"Failure rate: {y.mean():.1%} ({y.sum()}/{len(y)})")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Models to test
        models = {
            'Logistic Regression': LogisticRegression(
                random_state=42, max_iter=1000, class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42, class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=42
            )
        }

        results = {}

        print("\n" + "-" * 60)
        print("MODEL COMPARISON (5-fold CV)")
        print("-" * 60)

        for name, model in models.items():
            # Multiple metrics
            f1_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1')
            recall_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='recall')
            precision_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='precision')

            try:
                auc_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
                auc_mean = auc_scores.mean()
            except:
                auc_mean = 0

            results[name] = {
                'f1': f1_scores.mean(),
                'f1_std': f1_scores.std(),
                'recall': recall_scores.mean(),
                'recall_std': recall_scores.std(),
                'precision': precision_scores.mean(),
                'precision_std': precision_scores.std(),
                'auc': auc_mean
            }

            print(f"\n{name}:")
            print(f"  F1 Score:  {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
            print(f"  Recall:    {recall_scores.mean():.3f} ± {recall_scores.std():.3f} (catching at-risk)")
            print(f"  Precision: {precision_scores.mean():.3f} ± {precision_scores.std():.3f} (false alarm rate)")
            print(f"  AUC:       {auc_mean:.3f}")

        # Train best model on full data
        best_model_name = max(results, key=lambda k: results[k]['recall'])
        best_model = models[best_model_name]
        best_model.fit(X_scaled, y)

        print(f"\n{'='*60}")
        print(f"BEST MODEL FOR EARLY WARNING: {best_model_name}")
        print(f"  (Selected for highest RECALL - catching at-risk students)")
        print("=" * 60)

        # Feature importances
        if hasattr(best_model, 'feature_importances_'):
            importances = list(zip(available, best_model.feature_importances_))
            importances.sort(key=lambda x: -x[1])
            print("\nFeature Importances (what predicts failure):")
            for feat, imp in importances[:10]:
                print(f"  {feat:30s}: {imp:.3f}")
        elif hasattr(best_model, 'coef_'):
            coefs = list(zip(available, best_model.coef_[0]))
            coefs.sort(key=lambda x: -abs(x[1]))
            print("\nFeature Coefficients (what predicts failure):")
            for feat, coef in coefs[:10]:
                direction = "↑ RISK" if coef > 0 else "↓ RISK"
                print(f"  {feat:30s}: {coef:+.3f} {direction}")

        self.models = {
            'scaler': scaler,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'features': available,
            'results': results
        }

        return results

    def generate_risk_report(self) -> pd.DataFrame:
        """
        Generate a risk report for all students.

        This is what would be used for intervention.
        """
        if not self.models:
            raise ValueError("Must train models first")

        print("\n" + "=" * 70)
        print("GENERATING STUDENT RISK REPORT")
        print("=" * 70)

        scaler = self.models['scaler']
        model = self.models['best_model']
        features = self.models['features']

        # Prepare data
        df = self.features_df.copy()
        X = df[features].fillna(0)
        X_scaled = scaler.transform(X)

        # Predict probabilities
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_scaled)[:, 1]
        else:
            probs = model.predict(X_scaled)

        df['failure_risk_score'] = probs
        df['risk_level'] = pd.cut(
            probs,
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        # Summary
        print("\nRisk Distribution:")
        print(df['risk_level'].value_counts().sort_index())

        # High risk students
        high_risk = df[df['risk_level'].isin(['High', 'Critical'])]
        print(f"\nStudents requiring intervention: {len(high_risk)}")

        # Validate against actual outcomes
        if 'failed' in df.columns:
            actual_failures = df[df['failed'] == 1]
            flagged_correctly = len(high_risk[high_risk['failed'] == 1])

            print(f"\nValidation against actual outcomes:")
            print(f"  Actual failures: {len(actual_failures)}")
            print(f"  Correctly flagged as high risk: {flagged_correctly}")
            print(f"  Catch rate: {flagged_correctly/len(actual_failures)*100:.1f}%")

        return df[['course_id', 'user_id', 'final_score', 'failed',
                   'failure_risk_score', 'risk_level', 'page_views',
                   'early_access_score', 'is_morning_studier']].sort_values(
                       'failure_risk_score', ascending=False)

    def save_results(self, output_dir: str) -> None:
        """Save all results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save features
        if self.features_df is not None:
            self.features_df.to_csv(
                os.path.join(output_dir, 'student_features.csv'),
                index=False
            )

        # Save model results
        if self.models:
            results = {
                'best_model': self.models['best_model_name'],
                'features_used': self.models['features'],
                'model_results': self.models['results']
            }
            with open(os.path.join(output_dir, 'model_results.json'), 'w') as f:
                json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {output_dir}")


def main():
    """Run the Early Warning System."""
    print("=" * 70)
    print("EARLY WARNING SYSTEM FOR STUDENT FAILURE PREDICTION")
    print("Universidad Autónoma de Chile - Canvas LMS Analysis")
    print("=" * 70)
    print("\nEvery student identified early is a potential intervention.")
    print("Every intervention is a chance to change a life.")
    print("-" * 70)

    # Initialize system
    ews = EarlyWarningSystem(ANALYSIS_COURSES)

    # Extract features
    features_df = ews.extract_comprehensive_features()

    # Calculate early access scores
    ews.calculate_early_access_scores()

    # Analyze feature importance
    feature_importance = ews.analyze_feature_importance()

    # Train models
    model_results = ews.train_early_warning_models()

    # Generate risk report
    risk_report = ews.generate_risk_report()

    # Save results
    output_dir = os.path.join(DATA_DIR, 'early_warning')
    ews.save_results(output_dir)

    # Print final summary
    print("\n" + "=" * 70)
    print("EARLY WARNING SYSTEM - FINAL SUMMARY")
    print("=" * 70)

    print(f"\nStudents analyzed: {len(features_df)}")
    print(f"Courses analyzed: {len(ANALYSIS_COURSES)}")

    best_result = ews.models['results'][ews.models['best_model_name']]
    print(f"\nBest Model: {ews.models['best_model_name']}")
    print(f"  Recall: {best_result['recall']:.1%} (catches {best_result['recall']*100:.0f}% of at-risk students)")
    print(f"  Precision: {best_result['precision']:.1%}")
    print(f"  F1 Score: {best_result['f1']:.3f}")

    print("\n" + "-" * 70)
    print("KEY EARLY WARNING INDICATORS:")
    print("-" * 70)
    print("  1. Low early_access_score → Student accesses modules late")
    print("  2. Low page_views → Student not engaging with content")
    print("  3. is_evening_studier → Evening-dominant study pattern")
    print("  4. Low activity_span_days → Concentrated/cramming behavior")

    print("\n" + "=" * 70)
    print("Remember: Behind every data point is a student's future.")
    print("=" * 70)


if __name__ == '__main__':
    main()
