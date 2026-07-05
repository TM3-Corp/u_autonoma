#!/usr/bin/env python3
"""
SHAP Explanations for Early Warning Model

Generates interpretable explanations for model predictions using SHAP values.
This helps educators understand WHY a student is flagged as at-risk.

Outputs:
- Summary plot of feature importance
- Force plots for top N at-risk predictions
- Text explanations for individual students
"""

import pandas as pd
import numpy as np
from pathlib import Path
import shap
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path('/home/paul/projects/uautonoma/data')
ENRICHED_DIR = DATA_DIR / 'enriched_features'
OUTPUT_DIR = DATA_DIR / 'report/visualizations'

# Feature name translations (English -> Spanish)
FEATURE_TRANSLATIONS = {
    'total_time_min': 'Tiempo total (min)',
    'total_time_min_znorm': 'Tiempo total (normalizado)',
    'total_views': 'Vistas totales',
    'total_views_x_znorm': 'Vistas totales (normalizado)',
    'session_count': 'Num. sesiones',
    'sessions_per_week': 'Sesiones/semana',
    'sessions_per_week_znorm': 'Sesiones/semana (norm)',
    'modules_views': 'Vistas modulos',
    'files_views': 'Vistas archivos',
    'discussions_views': 'Vistas discusiones',
    'pages_views': 'Vistas paginas',
    'pages_views_pct': '% Vistas paginas',
    'content_vs_assessment_ratio': 'Ratio contenido/evaluacion',
    'total_transitions': 'Transiciones navegacion',
    'transition_entropy': 'Diversidad navegacion',
    'resource_coverage': 'Cobertura recursos',
    'jaccard_to_passing': 'Similitud con aprobados',
    'pct_night': '% Actividad nocturna',
    'pct_weekend': '% Fin de semana',
    'late_night_ratio': '% Madrugada',
    'discussions_unique_resources': 'Recursos discusion unicos',
    'files_pc1': 'Patron archivos (PC1)',
    'pages_pc1': 'Patron paginas (PC1)',
    'last_active_week': 'Ultima semana activa',
    'peak_hour': 'Hora pico actividad',
}


def translate_feature(name):
    """Translate feature name to Spanish."""
    return FEATURE_TRANSLATIONS.get(name, name)


def load_data():
    """Load and merge all feature files."""
    # Load features
    session_df = pd.read_parquet(ENRICHED_DIR / 'session_features.parquet')
    category_df = pd.read_parquet(ENRICHED_DIR / 'category_features.parquet')
    proact_df = pd.read_parquet(ENRICHED_DIR / 'proactivity_features.parquet')
    pca_df = pd.read_parquet(ENRICHED_DIR / 'pca_features.parquet')

    # Optional features
    ngram_path = ENRICHED_DIR / 'ngram_features.parquet'
    graph_path = ENRICHED_DIR / 'graph_features.parquet'
    time_path = ENRICHED_DIR / 'time_features.parquet'

    # Merge
    merged = session_df.merge(category_df, on=['user_id', 'course_id'], how='outer')
    merged = merged.merge(proact_df, on=['user_id', 'course_id'], how='left')
    merged = merged.merge(pca_df, on=['user_id', 'course_id'], how='left')

    if ngram_path.exists():
        ngram_df = pd.read_parquet(ngram_path)
        ngram_cols = [c for c in ngram_df.columns if c not in ['user_id', 'course_id', 'dominant_transition']]
        merged = merged.merge(ngram_df[['user_id', 'course_id'] + ngram_cols], on=['user_id', 'course_id'], how='left')

    if graph_path.exists():
        graph_df = pd.read_parquet(graph_path)
        graph_cols = [c for c in graph_df.columns if c not in ['user_id', 'course_id']]
        merged = merged.merge(graph_df[['user_id', 'course_id'] + graph_cols], on=['user_id', 'course_id'], how='left')

    if time_path.exists():
        time_df = pd.read_parquet(time_path)
        time_cols = [c for c in time_df.columns if c not in ['user_id', 'course_id']]
        merged = merged.merge(time_df[['user_id', 'course_id'] + time_cols], on=['user_id', 'course_id'], how='left')

    # Enrollments
    enrollments = pd.read_csv(DATA_DIR / 'page_views/student_enrollments.csv')
    enrollments['failed'] = enrollments['final_score'] < 57
    merged = merged.merge(
        enrollments[['user_id', 'course_id', 'failed']],
        on=['user_id', 'course_id'],
        how='inner'
    )

    return merged


def get_valid_features(df, exclude_patterns=None):
    """Get feature columns, excluding specified patterns."""
    if exclude_patterns is None:
        exclude_patterns = ['quiz', 'quizzes', 'assi', 'assignment', 'grade', 'grad', 'score', 'submission']

    valid = []
    for col in df.columns:
        if col in ['user_id', 'course_id', 'failed']:
            continue
        col_lower = col.lower()
        if any(p in col_lower for p in exclude_patterns):
            continue
        valid.append(col)
    return valid


def main():
    print('=' * 60)
    print('SHAP EXPLANATIONS FOR EARLY WARNING MODEL')
    print('=' * 60)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print('Loading data...')
    df = load_data()
    print(f'  Loaded {len(df)} samples')

    # Get features
    feature_cols = get_valid_features(df)
    X = df[feature_cols].copy()
    y = df['failed'].astype(int)

    print(f'  Features: {len(feature_cols)}')

    # Impute and scale
    print('Preprocessing...')
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()

    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_imputed),
        columns=X.columns,
        index=X.index
    )

    # Train XGBoost model
    print('Training XGBoost model...')
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss',
        verbosity=0,
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_scaled, y)

    # Get predictions
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]
    df['risk_probability'] = y_pred_proba

    print()

    # SHAP Analysis
    print('Computing SHAP values...')
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    print()

    # 1. Summary Plot
    print('Generating summary plot...')
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_scaled, show=False, max_display=20)
    plt.title('Importancia de Caracteristicas (SHAP)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUTPUT_DIR}/shap_summary.png')

    # 2. Bar Plot (mean absolute SHAP)
    print('Generating bar plot...')
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_scaled, plot_type='bar', show=False, max_display=15)
    plt.title('Impacto Promedio en Prediccion (|SHAP|)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'shap_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {OUTPUT_DIR}/shap_bar.png')

    # 3. Top at-risk students explanations
    print('\nGenerating individual explanations for top at-risk students...')

    # Get top 10 at-risk students
    df_sorted = df.sort_values('risk_probability', ascending=False)
    top_at_risk = df_sorted.head(10)

    explanations = []

    for idx, (_, row) in enumerate(top_at_risk.iterrows()):
        student_idx = df.index.get_loc(row.name)
        student_shap = shap_values[student_idx]

        # Get top factors (positive = increases risk, negative = decreases)
        shap_df = pd.DataFrame({
            'feature': X_scaled.columns,
            'shap_value': student_shap,
            'feature_value': X_imputed.iloc[student_idx].values
        })
        shap_df = shap_df.sort_values('shap_value', ascending=False)

        # Top 5 risk factors
        risk_factors = shap_df.head(5)
        protective_factors = shap_df.tail(3)

        explanation = {
            'user_id': row['user_id'],
            'course_id': row['course_id'],
            'risk_probability': row['risk_probability'],
            'actual_failed': row['failed'],
            'top_risk_factors': [],
            'protective_factors': []
        }

        for _, f in risk_factors.iterrows():
            explanation['top_risk_factors'].append({
                'feature': translate_feature(f['feature']),
                'impact': f['shap_value'],
                'value': f['feature_value']
            })

        for _, f in protective_factors.iterrows():
            explanation['protective_factors'].append({
                'feature': translate_feature(f['feature']),
                'impact': f['shap_value'],
                'value': f['feature_value']
            })

        explanations.append(explanation)

    # Save explanations
    import json
    with open(OUTPUT_DIR / 'shap_explanations.json', 'w') as f:
        json.dump(explanations, f, indent=2, default=str)
    print(f'  Saved: {OUTPUT_DIR}/shap_explanations.json')

    # Print sample explanations
    print('\n' + '=' * 60)
    print('SAMPLE EXPLANATIONS (Top 3 At-Risk Students)')
    print('=' * 60)

    for i, exp in enumerate(explanations[:3], 1):
        print(f'\n--- Estudiante {i} ---')
        print(f'Usuario: {exp["user_id"]}, Curso: {exp["course_id"]}')
        print(f'Probabilidad de Riesgo: {exp["risk_probability"]*100:.1f}%')
        print(f'Resultado Real: {"REPROBO" if exp["actual_failed"] else "APROBO"}')
        print('\nFactores de Riesgo:')
        for j, factor in enumerate(exp['top_risk_factors'][:3], 1):
            direction = "aumenta" if factor['impact'] > 0 else "reduce"
            print(f'  {j}. {factor["feature"]} ({direction} riesgo en {abs(factor["impact"]):.3f})')
        print('\nFactores Protectores:')
        for j, factor in enumerate(exp['protective_factors'][:2], 1):
            direction = "reduce" if factor['impact'] < 0 else "aumenta"
            print(f'  {j}. {factor["feature"]} ({direction} riesgo en {abs(factor["impact"]):.3f})')

    print('\n' + '=' * 60)
    print('SHAP Analysis Complete')
    print('=' * 60)


if __name__ == '__main__':
    main()
