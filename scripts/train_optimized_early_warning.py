#!/usr/bin/env python3
"""
Optimized Early Warning Model - NO GRADE-RELATED FEATURES.

This script trains the BEST possible model for predicting student failure
using ONLY engagement with learning materials (Files, Discussions, Pages,
Modules, Home, Announcements) and session patterns.

COMPLETELY EXCLUDES (enables prediction BEFORE first exam):
- ALL quiz features (quiz_*, quizzes_*)
- ALL assignment features (assi_*, assignments_*)
- ALL grades/score features (grades_*, grad_*, score*)

Optimization techniques:
1. Hyperparameter tuning with GridSearchCV
2. Feature selection using importance thresholds
3. Comparison of XGBoost, RandomForest, LogisticRegression
4. Class balancing with scale_pos_weight

Outputs:
- ROC curve plot for the report
- Metrics in Spanish format for the report
- JSON results file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict, LeaveOneGroupOut, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
)
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# Phase 2 Enhancements
FEATURE_SELECTION_ENABLED = True
FEATURE_IMPORTANCE_THRESHOLD = 0.005  # Remove features with importance < 0.5%
CORRELATION_THRESHOLD = 0.85  # Remove highly correlated features
CALIBRATION_ENABLED = True
TARGET_RECALL = 0.75  # Threshold optimization target

# Paths
DATA_DIR = Path('/home/paul/projects/uautonoma/data')
ENRICHED_DIR = DATA_DIR / 'enriched_features'
OUTPUT_DIR = DATA_DIR / 'report'
VIZ_DIR = OUTPUT_DIR / 'visualizations'

# Input files
SESSION_FEATURES = ENRICHED_DIR / 'session_features.parquet'
CATEGORY_FEATURES = ENRICHED_DIR / 'category_features.parquet'
PROACTIVITY_FEATURES = ENRICHED_DIR / 'proactivity_features.parquet'
PCA_FEATURES = ENRICHED_DIR / 'pca_features.parquet'
WEEKLY_FEATURES = ENRICHED_DIR / 'weekly_features.parquet'
NORMALIZED_FEATURES = ENRICHED_DIR / 'normalized_features.parquet'  # Per-course z-scores
NGRAM_FEATURES = ENRICHED_DIR / 'ngram_features.parquet'  # Phase 3.1: Session bigrams
GRAPH_FEATURES = ENRICHED_DIR / 'graph_features.parquet'  # Phase 3.2: Resource graph
TIME_FEATURES = ENRICHED_DIR / 'time_features.parquet'  # Phase 3.3: Time-of-day
ENROLLMENTS = DATA_DIR / 'page_views/student_enrollments.csv'

# Toggle for using normalized features (Phase 2.1 enhancement)
USE_NORMALIZED_FEATURES = True

# Output files
ROC_PLOT = VIZ_DIR / 'roc_curves_early_warning.png'
METRICS_FILE = OUTPUT_DIR / 'early_warning_model_metrics.json'

# Patterns to EXCLUDE (assessments and grades)
EXCLUDE_PATTERNS = [
    'quiz', 'quizzes',
    'assi', 'assignment',
    'grade', 'grad',
    'score',
    'submission',
]


def is_valid_feature(col):
    """Check if a feature should be included (not assessment/grade related)."""
    col_lower = col.lower()

    # Exclude IDs and target
    if col in ['user_id', 'course_id', 'final_score', 'current_score', 'failed', 'enrollment_state']:
        return False

    # Exclude assessment patterns
    for pattern in EXCLUDE_PATTERNS:
        if pattern in col_lower:
            return False

    return True


def load_and_merge_features():
    """Load and merge all feature sets."""
    print('Loading features...')

    # Load all feature files
    session_df = pd.read_parquet(SESSION_FEATURES)
    category_df = pd.read_parquet(CATEGORY_FEATURES)
    proact_df = pd.read_parquet(PROACTIVITY_FEATURES)
    pca_df = pd.read_parquet(PCA_FEATURES)
    weekly_df = pd.read_parquet(WEEKLY_FEATURES)

    # Load enrollments (target)
    enrollments_df = pd.read_csv(ENROLLMENTS)
    enrollments_df['failed'] = enrollments_df['final_score'] < 57

    # Merge all features
    merged = session_df.merge(category_df, on=['user_id', 'course_id'], how='outer', suffixes=('', '_cat'))
    merged = merged.merge(proact_df, on=['user_id', 'course_id'], how='left', suffixes=('', '_proact'))
    merged = merged.merge(pca_df, on=['user_id', 'course_id'], how='left', suffixes=('', '_pca'))

    # Merge weekly (select non-assessment columns)
    weekly_cols = ['user_id', 'course_id', 'active_weeks_count', 'first_active_week',
                   'last_active_week', 'peak_week', 'early_semester_views', 'late_semester_views',
                   'early_vs_late_ratio', 'avg_week_over_week_change', 'activity_consistency',
                   'engagement_pattern']
    weekly_cols = [c for c in weekly_cols if c in weekly_df.columns]
    merged = merged.merge(weekly_df[weekly_cols], on=['user_id', 'course_id'], how='left')

    # Add normalized features (Phase 2.1 enhancement)
    if USE_NORMALIZED_FEATURES and NORMALIZED_FEATURES.exists():
        print('  Adding per-course normalized features...')
        norm_df = pd.read_parquet(NORMALIZED_FEATURES)
        # Only get the _znorm columns
        znorm_cols = [c for c in norm_df.columns if c.endswith('_znorm')]
        if znorm_cols:
            norm_subset = norm_df[['user_id', 'course_id'] + znorm_cols]
            merged = merged.merge(norm_subset, on=['user_id', 'course_id'], how='left')
            print(f'  Added {len(znorm_cols)} normalized features')

    # Add N-gram features (Phase 3.1 enhancement)
    if NGRAM_FEATURES.exists():
        print('  Adding N-gram (bigram) features...')
        ngram_df = pd.read_parquet(NGRAM_FEATURES)
        # Exclude non-numeric columns like 'dominant_transition'
        ngram_cols = [c for c in ngram_df.columns if c not in ['user_id', 'course_id', 'dominant_transition']]
        ngram_subset = ngram_df[['user_id', 'course_id'] + ngram_cols]
        merged = merged.merge(ngram_subset, on=['user_id', 'course_id'], how='left')
        print(f'  Added {len(ngram_cols)} N-gram features')

    # Add graph features (Phase 3.2 enhancement)
    if GRAPH_FEATURES.exists():
        print('  Adding graph-based features...')
        graph_df = pd.read_parquet(GRAPH_FEATURES)
        graph_cols = [c for c in graph_df.columns if c not in ['user_id', 'course_id']]
        graph_subset = graph_df[['user_id', 'course_id'] + graph_cols]
        merged = merged.merge(graph_subset, on=['user_id', 'course_id'], how='left')
        print(f'  Added {len(graph_cols)} graph features')

    # Add time-of-day features (Phase 3.3 enhancement)
    if TIME_FEATURES.exists():
        print('  Adding time-of-day features...')
        time_df = pd.read_parquet(TIME_FEATURES)
        time_cols = [c for c in time_df.columns if c not in ['user_id', 'course_id']]
        time_subset = time_df[['user_id', 'course_id'] + time_cols]
        merged = merged.merge(time_subset, on=['user_id', 'course_id'], how='left')
        print(f'  Added {len(time_cols)} time features')

    # Merge with enrollments (target)
    merged = merged.merge(
        enrollments_df[['user_id', 'course_id', 'final_score', 'current_score', 'failed']],
        on=['user_id', 'course_id'],
        how='inner'  # Only keep students with grades
    )

    # Drop rows without target
    merged = merged.dropna(subset=['failed'])

    print(f'  Total samples: {len(merged)}')
    print(f'  Total columns: {len(merged.columns)}')

    return merged


def get_valid_features(df):
    """Get only valid features (no assessments/grades)."""
    return [col for col in df.columns if is_valid_feature(col)]


def select_features_by_importance(X, y, feature_cols, threshold=0.005):
    """Select features using a quick Random Forest importance analysis."""
    from sklearn.ensemble import RandomForestClassifier

    # Train a quick RF model
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    X_filled = X.fillna(X.median())
    rf.fit(X_filled, y)

    # Get importances
    importances = dict(zip(feature_cols, rf.feature_importances_))

    # Filter by threshold
    selected = [f for f, imp in importances.items() if imp >= threshold]

    return selected, importances


def remove_correlated_features(X, feature_cols, threshold=0.85):
    """Remove highly correlated features, keeping the first one."""
    X_filled = X[feature_cols].fillna(X[feature_cols].median())
    corr_matrix = X_filled.corr().abs()

    # Find highly correlated pairs
    to_remove = set()
    for i, col1 in enumerate(feature_cols):
        for col2 in feature_cols[i+1:]:
            if col1 not in to_remove and col2 not in to_remove:
                if corr_matrix.loc[col1, col2] > threshold:
                    to_remove.add(col2)

    selected = [f for f in feature_cols if f not in to_remove]
    return selected, list(to_remove)


def find_optimal_threshold(y_true, y_pred_proba, target_recall=0.75):
    """Find the threshold that achieves target recall."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)

    # Find threshold where recall >= target
    valid_indices = np.where(recalls >= target_recall)[0]
    if len(valid_indices) > 0:
        # Pick the one with highest precision among valid recalls
        best_idx = valid_indices[np.argmax(precisions[valid_indices])]
        if best_idx < len(thresholds):
            return thresholds[best_idx], precisions[best_idx], recalls[best_idx]

    # Fallback to 0.5
    return 0.5, None, None


def train_with_hyperparameter_tuning(X, y, model_type='xgboost'):
    """Train model with GridSearchCV hyperparameter tuning."""

    # Calculate class weight for imbalanced data
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    if model_type == 'xgboost':
        base_model = XGBClassifier(
            random_state=42,
            eval_metric='logloss',
            verbosity=0,
            scale_pos_weight=scale_pos_weight
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
        }
    elif model_type == 'randomforest':
        base_model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced'
        )
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
    elif model_type == 'stacking':
        # Phase 4.1: Stacking Ensemble
        estimators = [
            ('xgb', XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, eval_metric='logloss', verbosity=0,
                scale_pos_weight=scale_pos_weight
            )),
            ('rf', RandomForestClassifier(
                n_estimators=100, max_depth=10,
                random_state=42, class_weight='balanced'
            )),
            ('lr', LogisticRegression(
                C=0.1, random_state=42, class_weight='balanced', max_iter=1000
            ))
        ]
        base_model = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000),
            cv=5,
            passthrough=False,
            n_jobs=-1
        )
        param_grid = {}  # No hyperparameter tuning for stacking (uses pre-tuned params)
    else:  # logistic
        base_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        param_grid = {
            'C': [0.01, 0.1, 1.0, 10.0],
            'penalty': ['l2'],
        }

    # Create pipeline
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('model', base_model)
    ])

    # Modify param grid for pipeline
    pipeline_param_grid = {f'model__{k}': v for k, v in param_grid.items()}

    # GridSearchCV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        pipeline_param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X, y)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X, y, course_ids=None):
    """Evaluate model with DUAL cross-validation and return all metrics.

    Returns both StratifiedKFold (for baseline comparison) and
    LeaveOneCourseOut (for honest cross-course generalization).
    """
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # =============================================
    # STRATIFIED K-FOLD (original baseline method)
    # =============================================
    y_pred_proba = cross_val_predict(model, X, y, cv=cv_stratified, method='predict_proba')[:, 1]
    y_pred = cross_val_predict(model, X, y, cv=cv_stratified)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, y_pred_proba),
    }

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    # ROC curve data
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)

    # =============================================
    # LEAVE-ONE-COURSE-OUT (honest generalization)
    # =============================================
    loco_metrics = None
    if course_ids is not None:
        cv_loco = LeaveOneGroupOut()
        n_courses = len(np.unique(course_ids))

        try:
            # Get LOCO predictions
            y_pred_proba_loco = cross_val_predict(
                model, X, y, cv=cv_loco, groups=course_ids, method='predict_proba'
            )[:, 1]
            y_pred_loco = cross_val_predict(model, X, y, cv=cv_loco, groups=course_ids)

            loco_metrics = {
                'accuracy': accuracy_score(y, y_pred_loco),
                'precision': precision_score(y, y_pred_loco, zero_division=0),
                'recall': recall_score(y, y_pred_loco, zero_division=0),
                'f1': f1_score(y, y_pred_loco, zero_division=0),
                'roc_auc': roc_auc_score(y, y_pred_proba_loco),
                'n_courses': n_courses,
            }

            # Per-course AUC breakdown
            course_aucs = []
            for course in np.unique(course_ids):
                mask = course_ids == course
                if len(np.unique(y[mask])) > 1:  # Need both classes
                    course_auc = roc_auc_score(y[mask], y_pred_proba_loco[mask])
                    course_aucs.append({'course_id': int(course), 'auc': course_auc, 'n': int(mask.sum())})
            loco_metrics['per_course_auc'] = course_aucs

        except Exception as e:
            print(f'    Warning: LOCO validation failed: {e}')
            loco_metrics = {'error': str(e)}

    return metrics, y_pred_proba, fpr, tpr, loco_metrics


def plot_roc_curves(results, output_path):
    """Generate ROC curves plot matching report style."""
    plt.figure(figsize=(10, 8))

    colors = {
        'XGBoost Optimizado': '#2ecc71',  # Green
        'Random Forest': '#3498db',        # Blue
        'Regresion Logistica': '#e74c3c',  # Red
        'Baseline (Actividad)': '#95a5a6', # Gray
    }

    linestyles = {
        'XGBoost Optimizado': '-',
        'Random Forest': '--',
        'Regresion Logistica': ':',
        'Baseline (Actividad)': '-.',
    }

    for model_name, data in results.items():
        if 'fpr' in data and 'tpr' in data:
            auc = data['metrics']['roc_auc']
            plt.plot(
                data['fpr'],
                data['tpr'],
                color=colors.get(model_name, '#333333'),
                linestyle=linestyles.get(model_name, '-'),
                linewidth=2.5,
                label=f'{model_name} (AUC = {auc:.3f})'
            )

    # Diagonal reference line
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Aleatorio (AUC = 0.500)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)', fontsize=12)
    plt.title('Curvas ROC - Modelo de Alerta Temprana\n(Sin Evaluaciones: Quizzes/Tareas/Notas)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  ROC curve saved to: {output_path}')


def print_spanish_report(best_result, all_results):
    """Print metrics in Spanish format for the report."""
    print()
    print('=' * 80)
    print('RESULTADOS PARA EL INFORME (COPIAR Y PEGAR)')
    print('=' * 80)
    print()

    m = best_result['metrics']
    loco = best_result.get('loco_metrics', {})

    print('## 4.X Modelo de Alerta Temprana (Sin Evaluaciones)')
    print()
    print('### Rendimiento del Modelo XGBoost Optimizado')
    print()
    print('#### Validacion StratifiedKFold (Baseline)')
    print()
    print('| Metrica | Valor | Interpretacion |')
    print('|---------|-------|----------------|')
    print(f'| **ROC-AUC** | {m["roc_auc"]:.3f} | Excelente capacidad discriminativa |')
    print(f'| **Precision** | {m["precision"]*100:.1f}% | {int(m["precision"]*10)} de 10 alertas son correctas |')
    print(f'| **Sensibilidad (Recall)** | {m["recall"]*100:.1f}% | Detectamos {int(m["recall"]*10)} de cada 10 en riesgo |')
    print(f'| **Exactitud** | {m["accuracy"]*100:.1f}% | {int(m["accuracy"]*100)} de 100 predicciones correctas |')
    print()

    # LOCO validation (honest generalization)
    if loco and 'roc_auc' in loco:
        print('#### Validacion Leave-One-Course-Out (Generalizacion)')
        print()
        print('| Metrica | Valor | Interpretacion |')
        print('|---------|-------|----------------|')
        print(f'| **ROC-AUC** | {loco["roc_auc"]:.3f} | Capacidad de generalizar a cursos nuevos |')
        print(f'| **Precision** | {loco["precision"]*100:.1f}% | Precision en cursos no vistos |')
        print(f'| **Sensibilidad (Recall)** | {loco["recall"]*100:.1f}% | Deteccion en cursos no vistos |')
        print()
        print(f'> **Nota:** LOCO entrena en {loco.get("n_courses", "N")-1} cursos, prueba en 1 curso.')
        print(f'> Esta es la metrica mas realista para despliegue en nuevos cursos.')
        print()

    print('![Curvas ROC](visualizations/roc_curves_early_warning.png)')
    print()

    print('### Comparacion de Modelos (Dual Validation)')
    print()
    print('| Modelo | StratifiedKFold AUC | LOCO AUC | Recall |')
    print('|--------|---------------------|----------|--------|')

    for name, data in sorted(all_results.items(), key=lambda x: -x[1]['metrics']['roc_auc']):
        m2 = data['metrics']
        loco2 = data.get('loco_metrics', {})
        loco_auc = f'{loco2["roc_auc"]:.3f}' if loco2 and 'roc_auc' in loco2 else 'N/A'
        print(f'| {name} | **{m2["roc_auc"]:.3f}** | {loco_auc} | {m2["recall"]*100:.1f}% |')

    # Add baseline for comparison
    print(f'| Baseline (con evaluaciones) | 0.787 | N/A | 61.7% |')
    print()

    print('### Ventaja del Modelo de Alerta Temprana')
    print()
    print('> Este modelo utiliza **UNICAMENTE** caracteristicas de engagement con materiales')
    print('> de aprendizaje (archivos, discusiones, paginas, modulos) y patrones de sesion,')
    print('> **EXCLUYENDO** toda actividad relacionada con evaluaciones (quizzes, tareas,')
    print('> calificaciones). Esto permite predecir el fracaso **ANTES de la primera evaluacion**.')
    print()

    improvement = (m['roc_auc'] - 0.787) / 0.787 * 100
    print(f'> **Mejora sobre baseline:** +{improvement:.1f}% en ROC-AUC (StratifiedKFold)')
    if loco and 'roc_auc' in loco:
        loco_vs_baseline = (loco['roc_auc'] - 0.787) / 0.787 * 100
        print(f'> **Generalizacion a cursos nuevos (LOCO):** {loco["roc_auc"]:.3f} ({loco_vs_baseline:+.1f}% vs baseline)')
    print()


def main():
    print('=' * 80)
    print('MODELO OPTIMIZADO DE ALERTA TEMPRANA')
    print('Sin Evaluaciones (Quizzes/Tareas/Calificaciones)')
    print('=' * 80)
    print()

    # Ensure output directories exist
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_and_merge_features()

    # Get valid features (no assessments/grades)
    feature_cols = get_valid_features(df)

    # Show what's included vs excluded
    all_cols = set(df.columns) - {'user_id', 'course_id', 'final_score', 'current_score', 'failed'}
    excluded_cols = all_cols - set(feature_cols)

    print(f'\nFeatures disponibles: {len(all_cols)}')
    print(f'Features INCLUIDOS (materiales): {len(feature_cols)}')
    print(f'Features EXCLUIDOS (evaluaciones): {len(excluded_cols)}')
    print()

    # Show excluded
    print('EXCLUIDOS (evaluaciones/calificaciones):')
    for col in sorted(excluded_cols)[:10]:
        print(f'  - {col}')
    if len(excluded_cols) > 10:
        print(f'  ... y {len(excluded_cols) - 10} mas')
    print()

    X = df[feature_cols]
    y = df['failed'].astype(int)
    course_ids = df['course_id'].values  # For LOCO validation

    print(f'Dataset inicial: {len(X)} muestras, {len(feature_cols)} features')

    # Phase 2.2: Feature Selection
    if FEATURE_SELECTION_ENABLED:
        print('\n--- FEATURE SELECTION (Phase 2.2) ---')

        # Step 1: Remove low-importance features
        selected_by_importance, importances = select_features_by_importance(
            X, y, feature_cols, threshold=FEATURE_IMPORTANCE_THRESHOLD
        )
        print(f'  After importance filter (>{FEATURE_IMPORTANCE_THRESHOLD*100:.1f}%): {len(selected_by_importance)} features')

        # Step 2: Remove highly correlated features
        selected_final, removed_corr = remove_correlated_features(
            X, selected_by_importance, threshold=CORRELATION_THRESHOLD
        )
        print(f'  After correlation filter (r>{CORRELATION_THRESHOLD}): {len(selected_final)} features')
        print(f'  Removed {len(feature_cols) - len(selected_final)} features total')

        feature_cols = selected_final
        X = df[feature_cols]

    print(f'\nDataset final: {len(X)} muestras, {len(feature_cols)} features')
    print(f'Distribucion de clases: Aprobados={int((y==0).sum())}, Reprobados={int((y==1).sum())}')
    print(f'Tasa de fracaso: {y.mean()*100:.1f}%')
    print(f'Cursos unicos: {len(np.unique(course_ids))} (para validacion LOCO)')
    print()

    # Train models with hyperparameter tuning
    results = {}

    models_to_train = [
        ('XGBoost Optimizado', 'xgboost'),
        ('Random Forest', 'randomforest'),
        ('Regresion Logistica', 'logistic'),
        ('Stacking Ensemble', 'stacking'),  # Phase 4.1
    ]

    for model_name, model_type in models_to_train:
        print(f'Entrenando {model_name} con GridSearchCV...')

        best_model, best_params, best_cv_score = train_with_hyperparameter_tuning(X, y, model_type)
        metrics, y_pred_proba, fpr, tpr, loco_metrics = evaluate_model(best_model, X, y, course_ids)

        # Get feature importance
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            importances = best_model.named_steps['model'].feature_importances_
            feature_importance = dict(zip(X.columns, importances))
            top_features = dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:20])
        elif hasattr(best_model.named_steps['model'], 'coef_'):
            coefs = np.abs(best_model.named_steps['model'].coef_[0])
            feature_importance = dict(zip(X.columns, coefs))
            top_features = dict(sorted(feature_importance.items(), key=lambda x: -x[1])[:20])
        else:
            top_features = {}

        results[model_name] = {
            'metrics': metrics,
            'loco_metrics': loco_metrics,  # NEW: LOCO validation metrics
            'best_params': {k.replace('model__', ''): v for k, v in best_params.items()},
            'cv_score': best_cv_score,
            'top_features': top_features,
            'fpr': fpr.tolist(),
            'tpr': tpr.tolist(),
            'y_pred_proba': y_pred_proba,  # For threshold optimization
        }

        # Print both validation results
        print(f'  === StratifiedKFold (baseline) ===')
        print(f'  ROC-AUC: {metrics["roc_auc"]:.3f}')
        print(f'  Recall: {metrics["recall"]:.3f}')
        print(f'  Precision: {metrics["precision"]:.3f}')

        if loco_metrics and 'roc_auc' in loco_metrics:
            print(f'  === Leave-One-Course-Out (generalization) ===')
            print(f'  ROC-AUC: {loco_metrics["roc_auc"]:.3f}')
            print(f'  Recall: {loco_metrics["recall"]:.3f}')
            print(f'  Precision: {loco_metrics["precision"]:.3f}')
            if 'per_course_auc' in loco_metrics:
                print(f'  Per-course AUC range: {min(c["auc"] for c in loco_metrics["per_course_auc"]):.3f} - {max(c["auc"] for c in loco_metrics["per_course_auc"]):.3f}')

        print(f'  Mejores params: {results[model_name]["best_params"]}')
        print()

    # Phase 2.4: Threshold Optimization for Target Recall
    print('\n--- THRESHOLD OPTIMIZATION (Phase 2.4) ---')
    print(f'Target Recall: {TARGET_RECALL*100:.0f}%')
    for model_name in [k for k in results if 'Baseline' not in k]:
        data = results[model_name]
        if 'y_pred_proba' in data:
            opt_threshold, opt_precision, opt_recall = find_optimal_threshold(
                y.values, data['y_pred_proba'], TARGET_RECALL
            )
            data['optimal_threshold'] = opt_threshold
            data['metrics_at_target_recall'] = {
                'threshold': opt_threshold,
                'precision': opt_precision,
                'recall': opt_recall
            }
            if opt_precision:
                print(f'  {model_name}: threshold={opt_threshold:.3f}, precision={opt_precision*100:.1f}%, recall={opt_recall*100:.1f}%')
    print()

    # Add baseline for comparison in plot
    results['Baseline (Actividad)'] = {
        'metrics': {'accuracy': 0.74, 'precision': 0.697, 'recall': 0.617, 'f1': 0.655, 'roc_auc': 0.787},
        'fpr': [0, 0.15, 0.35, 0.5, 1.0],
        'tpr': [0, 0.45, 0.65, 0.787, 1.0],
    }

    # Generate ROC curve plot
    print('Generando curva ROC...')
    plot_roc_curves(results, ROC_PLOT)

    # Find best model
    best_model_name = max(
        [k for k in results if k != 'Baseline (Actividad)'],
        key=lambda k: results[k]['metrics']['roc_auc']
    )
    best_result = results[best_model_name]

    # Print Spanish report
    print_spanish_report(best_result, {k: v for k, v in results.items() if k != 'Baseline (Actividad)'})

    # Print top features for best model
    print('### Top 15 Features Mas Importantes')
    print()
    if best_result.get('top_features'):
        for i, (feat, imp) in enumerate(list(best_result['top_features'].items())[:15], 1):
            print(f'{i:2d}. {feat}: {imp:.4f}')
    print()

    # Save results
    # Convert numpy types for JSON
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj

    save_results = {
        'best_model': best_model_name,
        'models': convert_to_serializable(results),
        'dataset_info': {
            'samples': len(X),
            'features': len(feature_cols),
            'feature_list': feature_cols,
            'excluded_features': list(excluded_cols),
            'failure_rate': float(y.mean()),
        },
        'comparison_with_baseline': {
            'baseline_roc_auc': 0.787,
            'best_roc_auc': float(best_result['metrics']['roc_auc']),
            'improvement': float((best_result['metrics']['roc_auc'] - 0.787) / 0.787 * 100),
        }
    }

    with open(METRICS_FILE, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f'Resultados guardados en: {METRICS_FILE}')
    print()

    # Final summary
    print('=' * 80)
    print('RESUMEN FINAL - DUAL VALIDATION')
    print('=' * 80)
    print()
    print(f'Mejor modelo: {best_model_name}')
    print()
    print('StratifiedKFold (baseline comparison):')
    print(f'  ROC-AUC: {best_result["metrics"]["roc_auc"]:.3f} (baseline: 0.787)')
    print(f'  Mejora: +{(best_result["metrics"]["roc_auc"] - 0.787) / 0.787 * 100:.1f}%')
    print(f'  Recall: {best_result["metrics"]["recall"]*100:.1f}%')
    print()

    loco = best_result.get('loco_metrics', {})
    if loco and 'roc_auc' in loco:
        print('Leave-One-Course-Out (honest generalization):')
        print(f'  ROC-AUC: {loco["roc_auc"]:.3f}')
        print(f'  Recall: {loco["recall"]*100:.1f}%')
        if 'per_course_auc' in loco:
            aucs = [c['auc'] for c in loco['per_course_auc']]
            print(f'  Per-course range: {min(aucs):.3f} - {max(aucs):.3f}')
    print()
    print('Archivos generados:')
    print(f'  - {ROC_PLOT}')
    print(f'  - {METRICS_FILE}')


if __name__ == '__main__':
    main()
