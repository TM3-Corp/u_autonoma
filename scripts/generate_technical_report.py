#!/usr/bin/env python3
"""
Generador de Reporte Técnico de Análisis Predictivo LMS.

Este script genera un documento técnico completo en español que detalla
todo el proceso de análisis predictivo de fracaso estudiantil usando
datos de Canvas LMS.

Usa la API de Claude para generar texto técnico de alta calidad.
"""

import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import anthropic
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

# Configuración
OUTPUT_DIR = Path("data/report")
VIZ_DIR = OUTPUT_DIR / "visualizations"
REPORT_FILE = OUTPUT_DIR / "REPORTE_TECNICO_ANALISIS_PREDICTIVO.md"

# API de Claude
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)


# =============================================================================
# CARGA DE DATOS
# =============================================================================

def load_all_data():
    """Cargar todos los archivos de datos necesarios."""
    data = {}

    # Análisis por curso (pure activity)
    with open("data/engagement_dynamics/pure_activity_analysis.json") as f:
        data["per_course"] = json.load(f)

    # Resultados del modelo pooled
    with open("data/pooled_analysis/model_results.json") as f:
        data["model_results"] = json.load(f)

    # Insights accionables
    with open("data/pooled_analysis/actionable_insights.json") as f:
        data["insights"] = json.load(f)

    # Features de estudiantes
    data["student_features"] = pd.read_csv("data/engagement_dynamics/student_features.csv")

    # Documentación de features (primeras 200 líneas)
    try:
        with open("docs/ENGAGEMENT_FEATURES_DOCUMENTATION.md") as f:
            data["feature_docs"] = f.read()
    except FileNotFoundError:
        data["feature_docs"] = ""

    # Análisis de diseño LMS
    try:
        with open("data/lms_design_analysis.json") as f:
            data["lms_design"] = json.load(f)
    except FileNotFoundError:
        data["lms_design"] = {}

    return data


# =============================================================================
# GENERACIÓN DE VISUALIZACIONES
# =============================================================================

def create_visualizations(data):
    """Generar todas las visualizaciones necesarias."""
    os.makedirs(VIZ_DIR, exist_ok=True)

    # Configuración de estilo
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.family"] = "sans-serif"

    # 1. Boxplot de notas por curso
    create_grade_boxplot(data)

    # 2. Barras de tasa de aprobación por curso
    create_pass_rate_bars(data)

    # 3. Heatmap de correlaciones top features
    create_correlation_heatmap(data)

    # 4. Comparación aprobados vs reprobados
    create_pass_fail_comparison(data)

    # 5. Distribución de diversidad de clase
    create_diversity_distribution(data)

    # Copiar visualizaciones existentes
    existing_viz = [
        ("data/pooled_analysis/visualizations/roc_curves.png", "roc_curves.png"),
        ("data/pooled_analysis/visualizations/feature_importance.png", "feature_importance.png"),
        ("data/pooled_analysis/visualizations/risk_factors.png", "risk_factors.png"),
    ]
    for src, dst in existing_viz:
        if os.path.exists(src):
            shutil.copy(src, VIZ_DIR / dst)


def create_grade_boxplot(data):
    """Boxplot de distribución de notas por curso."""
    df = data["student_features"]

    # Solo cursos con diversidad GOOD
    good_courses = [c["course_id"] for c in data["per_course"]
                    if c.get("class_diversity") == "GOOD"]
    df_good = df[df["course_id"].astype(str).isin(good_courses)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Crear boxplot
    courses = df_good.groupby("course_id")["final_score"].apply(list).to_dict()
    course_ids = list(courses.keys())
    course_data = [courses[cid] for cid in course_ids]

    bp = ax.boxplot(course_data, patch_artist=True)

    # Colorear por tasa de aprobación
    per_course_dict = {c["course_id"]: c for c in data["per_course"]}
    colors = []
    for cid in course_ids:
        cdata = per_course_dict.get(str(cid), {})
        pass_rate = cdata.get("pass_rate", 50)
        if pass_rate < 40:
            colors.append("#ff6b6b")  # Rojo
        elif pass_rate < 60:
            colors.append("#ffd93d")  # Amarillo
        else:
            colors.append("#6bcb77")  # Verde

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=57, color="red", linestyle="--", linewidth=2, label="Umbral aprobación (57%)")
    ax.set_xticklabels([str(cid)[-5:] for cid in course_ids], rotation=45)
    ax.set_xlabel("ID de Curso (últimos 5 dígitos)")
    ax.set_ylabel("Nota Final (%)")
    ax.set_title("Distribución de Notas por Curso")
    ax.legend()

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "grade_boxplot.png")
    plt.close()


def create_pass_rate_bars(data):
    """Barras de tasa de aprobación por curso."""
    fig, ax = plt.subplots(figsize=(12, 6))

    courses = []
    pass_rates = []
    n_students = []

    for c in data["per_course"]:
        if c.get("class_diversity") == "GOOD":
            courses.append(str(c["course_id"])[-5:])
            pass_rates.append(c["pass_rate"])
            n_students.append(c["n_students"])

    # Ordenar por tasa de aprobación
    sorted_idx = np.argsort(pass_rates)
    courses = [courses[i] for i in sorted_idx]
    pass_rates = [pass_rates[i] for i in sorted_idx]
    n_students = [n_students[i] for i in sorted_idx]

    # Colores
    colors = ["#ff6b6b" if pr < 40 else "#ffd93d" if pr < 60 else "#6bcb77"
              for pr in pass_rates]

    bars = ax.bar(courses, pass_rates, color=colors, alpha=0.8, edgecolor="black")

    # Añadir número de estudiantes encima de cada barra
    for bar, n in zip(bars, n_students):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"n={n}", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ID de Curso")
    ax.set_ylabel("Tasa de Aprobación (%)")
    ax.set_title("Tasa de Aprobación por Curso (Solo Diversidad GOOD)")
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "pass_rate_bars.png")
    plt.close()


def create_correlation_heatmap(data):
    """Heatmap de correlaciones de los top features por curso.

    Calcula TODAS las correlaciones para los features que aparecen en el
    top 5 de cualquier curso, mostrando el panorama completo.
    """
    from scipy import stats

    fig, ax = plt.subplots(figsize=(16, 9))

    # Extraer top 5 features por curso (unión de todos)
    good_courses = []
    all_features = set()

    for c in data["per_course"]:
        if c.get("class_diversity") == "GOOD":
            good_courses.append(c["course_id"])
            for feat, _ in c.get("top_correlations", [])[:5]:
                all_features.add(feat)

    # Calcular TODAS las correlaciones desde datos crudos
    df = data["student_features"]
    features = sorted(list(all_features))
    matrix = np.zeros((len(good_courses), len(features)))

    for i, course_id in enumerate(good_courses):
        course_df = df[df["course_id"].astype(str) == str(course_id)]
        for j, feat in enumerate(features):
            if feat in course_df.columns and "final_score" in course_df.columns:
                valid = course_df[[feat, "final_score"]].dropna()
                if len(valid) >= 5:
                    with np.errstate(invalid='ignore'):
                        r, _ = stats.pearsonr(valid[feat], valid["final_score"])
                    matrix[i, j] = r if not np.isnan(r) else 0
                else:
                    matrix[i, j] = np.nan

    # Crear heatmap con máscara para NaN
    mask = np.isnan(matrix)
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdYlGn",
                xticklabels=features,
                yticklabels=[str(c) for c in good_courses],
                center=0, vmin=-0.8, vmax=0.8, ax=ax,
                mask=mask, cbar_kws={"label": "Correlación (r)"})

    ax.set_xlabel("Feature (de Top 5 de cualquier curso)")
    ax.set_ylabel("Curso")
    ax.set_title("Correlaciones Feature-Nota: Todas las correlaciones para Features en Top 5")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def create_pass_fail_comparison(data):
    """Comparación de features entre aprobados y reprobados."""
    df = data["student_features"]

    # Features clave
    key_features = [
        "session_count", "sessions_per_week", "total_page_views",
        "unique_active_hours", "session_gap_mean"
    ]
    feature_labels = [
        "Sesiones", "Sesiones/Semana", "Page Views",
        "Horas Activas", "Gap entre Sesiones"
    ]

    # Filtrar y calcular medias
    df["passed"] = df["final_score"] >= 57
    df_valid = df[df["final_score"].notna()]

    fig, axes = plt.subplots(1, len(key_features), figsize=(15, 5))

    for i, (feat, label) in enumerate(zip(key_features, feature_labels)):
        if feat not in df_valid.columns:
            continue

        ax = axes[i]
        passed = df_valid[df_valid["passed"]][feat].dropna()
        failed = df_valid[~df_valid["passed"]][feat].dropna()

        bp = ax.boxplot([passed, failed], patch_artist=True)
        bp["boxes"][0].set_facecolor("#6bcb77")  # Verde para aprobados
        bp["boxes"][1].set_facecolor("#ff6b6b")  # Rojo para reprobados

        ax.set_xticklabels(["Aprobados", "Reprobados"])
        ax.set_title(label)
        ax.set_ylabel("Valor")

    plt.suptitle("Comparación de Features: Aprobados vs Reprobados", fontsize=14)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / "pass_fail_comparison.png")
    plt.close()


def create_diversity_distribution(data):
    """Distribución de cursos por diversidad de clase."""
    fig, ax = plt.subplots(figsize=(8, 6))

    diversity_counts = {"GOOD": 0, "MODERATE": 0, "LOW DIVERSITY": 0}
    for c in data["per_course"]:
        div = c.get("class_diversity", "UNKNOWN")
        if div in diversity_counts:
            diversity_counts[div] += 1

    labels = list(diversity_counts.keys())
    values = list(diversity_counts.values())
    colors = ["#6bcb77", "#ffd93d", "#ff6b6b"]

    bars = ax.bar(labels, values, color=colors, edgecolor="black", alpha=0.8)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")

    ax.set_xlabel("Diversidad de Clase")
    ax.set_ylabel("Número de Cursos")
    ax.set_title("Distribución de Cursos por Diversidad de Clase")

    plt.tight_layout()
    plt.savefig(VIZ_DIR / "diversity_distribution.png")
    plt.close()


# =============================================================================
# GENERACIÓN DE TEXTO CON CLAUDE API
# =============================================================================

def generate_section(section_name, context, max_tokens=2000):
    """Generar una sección del reporte usando Claude API."""
    print(f"  Generando sección: {section_name}...")

    prompts = {
        "resumen_ejecutivo": """
Genera un RESUMEN EJECUTIVO profesional para un reporte técnico de análisis predictivo de fracaso estudiantil usando datos de Canvas LMS.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Máximo 300 palabras
- Incluye: objetivo, metodología resumida, hallazgos principales, métricas clave
- NO uses emojis
- Formato Markdown
""",
        "metodologia_seleccion": """
Genera la sección de METODOLOGÍA DE SELECCIÓN DE CURSOS para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Explica los criterios de selección:
  1. Diversidad de clase (20% < tasa aprobación < 80%)
  2. Varianza de notas (σ > 10%)
  3. Mínimo de estudiantes (≥20)
  4. Diseño instruccional adecuado
- Incluye tabla de cursos seleccionados vs descartados
- NO uses emojis
- ~500 palabras
- Formato Markdown con tablas
""",
        "features": """
Genera la sección de INGENIERÍA DE FEATURES para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Explica las 7 categorías de features:
  1. Regularidad de Sesiones (7 features)
  2. Bloques de Tiempo (11 features)
  3. Coeficientes DCT (12 features)
  4. Trayectoria de Engagement (6 features)
  5. Dinámica de Carga (10 features)
  6. Tiempo de Acceso (4 features)
  7. Agregados Crudos (4 features)
- Para cada categoría, menciona los features más importantes y qué representan
- Menciona los endpoints de Canvas API usados
- NO uses emojis
- ~800 palabras
- Formato Markdown con tablas
""",
        "metodologia_modelos": """
Genera la sección de METODOLOGÍA DE MODELOS PREDICTIVOS para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Explica los modelos usados:
  1. Regresión Logística (interpretable)
  2. Random Forest (importancia de features)
  3. XGBoost (mejor rendimiento)
- Explica la validación cruzada estratificada 5-fold
- Explica las métricas: ROC-AUC, Recall, Precisión, F1
- Menciona el manejo de desbalance de clases (class_weight='balanced')
- NO uses emojis
- ~400 palabras
- Formato Markdown
""",
        "resultados": """
Genera la sección de RESULTADOS GENERALES para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Presenta los resultados de los modelos en tabla comparativa
- Destaca el mejor modelo (XGBoost con ROC-AUC 0.787)
- Interpreta los resultados en términos prácticos
- Menciona los top 5 features más importantes
- NO uses emojis
- ~600 palabras
- Formato Markdown con tablas
""",
        "insights": """
Genera la sección de INSIGHTS ACCIONABLES para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Presenta cada insight en formato de tabla con:
  * Factor de Riesgo
  * Riesgo Relativo (RR)
  * Comparación de tasas de fracaso
  * Significancia estadística
- Interpreta cada insight en términos prácticos para la universidad
- Agrupa por categoría: patrones de sesión, tiempo de estudio, engagement
- NO uses emojis
- ~500 palabras
- Formato Markdown con tablas
""",
        "analisis_por_curso": """
Genera la sección de ANÁLISIS POR CURSO para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Para cada uno de los 10 cursos con diversidad GOOD:
  * ID del curso
  * Número de estudiantes
  * Tasa de aprobación
  * Top 3-5 predictores (con correlación)
- Agrupa cursos por patrones similares si es posible
- Identifica cursos con patrones únicos
- NO uses emojis
- ~1000 palabras
- Formato Markdown con tablas
""",
        "conclusiones": """
Genera la sección de CONCLUSIONES Y RECOMENDACIONES para un reporte técnico.

CONTEXTO:
{context}

INSTRUCCIONES:
- Escribe en español técnico profesional
- Resume los hallazgos principales
- Proporciona recomendaciones concretas:
  * Corto plazo (inmediatas)
  * Mediano plazo (próximo semestre)
  * Largo plazo (institucional)
- Menciona limitaciones del estudio
- Sugiere próximos pasos para investigación
- NO uses emojis
- ~400 palabras
- Formato Markdown
"""
    }

    prompt = prompts.get(section_name, "").format(context=context)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        print(f"  Error generando {section_name}: {e}")
        return f"[Error generando sección {section_name}]"


def prepare_context(data, section):
    """Preparar el contexto específico para cada sección."""
    contexts = {
        "resumen_ejecutivo": f"""
Datos del análisis:
- 373 estudiantes analizados (de 10 cursos con buena diversidad)
- 54 features de engagement extraídos
- Modelo XGBoost alcanzó ROC-AUC de 0.787
- 8 insights estadísticamente significativos (p < 0.05)
- Tasa de aprobación global: 60.1%
- Recall del modelo: 61.7% (detecta 2/3 de estudiantes en riesgo)

Insights principales:
{json.dumps([{"factor": i["description"], "rr": i["relative_risk"]} for i in data["insights"][:5]], indent=2, ensure_ascii=False)}
""",
        "metodologia_seleccion": f"""
Cursos analizados:
{json.dumps([{
    "id": c["course_id"],
    "n": c["n_students"],
    "pass_rate": c["pass_rate"],
    "grade_std": c["grade_std"],
    "diversity": c["class_diversity"]
} for c in data["per_course"]], indent=2)}

Criterios de exclusión:
- Cursos con tasa de aprobación >80% (efecto techo)
- Cursos con tasa de aprobación <20% (todos reprueban)
- Cursos con <20 estudiantes (muestra insuficiente)
- Cursos con σ < 10% (poca varianza)
""",
        "features": f"""
Categorías de features (54 total):

1. REGULARIDAD DE SESIONES (7): session_count, session_gap_min/max/mean/std, session_regularity, sessions_per_week
   - Calculados desde: timestamps de page_views
   - Interpretación: Frecuencia y consistencia del estudio

2. BLOQUES DE TIEMPO (11): weekday_morning/afternoon/evening/night_pct, weekend_*, varianzas
   - Calculados desde: horas de actividad
   - Interpretación: CUÁNDO estudian los alumnos

3. COEFICIENTES DCT (12): dct_coef_0 a dct_coef_11
   - Calculados desde: transformada discreta del coseno sobre vector de 168 horas semanales
   - Interpretación: Patrones periódicos de actividad

4. TRAYECTORIA ENGAGEMENT (6): engagement_velocity/acceleration, weekly_cv, trend_reversals, early_engagement_ratio, late_surge
   - Calculados desde: conteos semanales de actividad
   - Interpretación: Cómo EVOLUCIONA el engagement en el tiempo

5. DINÁMICA DE CARGA (10): peak_count_type1/2/3, peak_ratio, slopes, weekly_range
   - Calculados desde: variaciones semana a semana
   - Interpretación: Intensidad y variabilidad del esfuerzo

6. TIEMPO DE ACCESO (4): first_access_day, first_module_day, first_assignment_day, access_time_pct
   - Calculados desde: fechas de primeros accesos
   - Interpretación: Procrastinación vs engagement temprano

7. AGREGADOS CRUDOS (4): total_page_views, activity_span_days, unique_active_hours
   - Calculados desde: Canvas Analytics API (student_summaries)
   - Interpretación: Métricas básicas de actividad

Endpoints de Canvas API usados:
- GET /api/v1/courses/:id/analytics/student_summaries (page_views, participations)
- GET /api/v1/courses/:id/analytics/users/:id/activity (actividad por hora)
- GET /api/v1/courses/:id/enrollments (grades.final_score)
""",
        "metodologia_modelos": f"""
Modelos entrenados:
1. Regresión Logística: max_iter=1000, class_weight='balanced'
2. Random Forest: n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight='balanced'
3. XGBoost: n_estimators=200, max_depth=5, learning_rate=0.1, scale_pos_weight ajustado

Validación:
- Stratified 5-fold cross-validation
- Stratificación mantiene proporción de clases en cada fold
- 373 estudiantes total (224 aprobados, 149 reprobados)

Métricas evaluadas:
- ROC-AUC: Área bajo curva ROC (discriminación)
- Recall: Sensibilidad (captura de casos positivos)
- Precision: Proporción de predicciones correctas
- F1: Media armónica de precision y recall
""",
        "resultados": f"""
Resultados de modelos:
{json.dumps(data["model_results"], indent=2)}

Dataset:
- Total estudiantes: 373
- Aprobados (≥57%): 224 (60.1%)
- Reprobados (<57%): 149 (39.9%)

Interpretación:
- XGBoost tiene mejor ROC-AUC (0.787) y mejor balance precision/recall
- Random Forest tiene mejor precision pero menor recall
- Regresión Logística es más interpretable pero menos precisa
""",
        "insights": f"""
Insights accionables con significancia estadística:
{json.dumps(data["insights"], indent=2, ensure_ascii=False)}

Nota: Solo se reportan insights con p < 0.05 (estadísticamente significativos)
""",
        "analisis_por_curso": f"""
Análisis por curso (10 cursos con diversidad GOOD):
{json.dumps([{
    "id": c["course_id"],
    "n": c["n_students"],
    "mean": c["grade_mean"],
    "std": c["grade_std"],
    "pass_rate": c["pass_rate"],
    "top_5_corr": c["top_correlations"][:5],
    "top_3_import": c["top_importances"][:3]
} for c in data["per_course"] if c.get("class_diversity") == "GOOD"], indent=2)}
""",
        "conclusiones": f"""
Resumen de hallazgos:
- ROC-AUC 0.787 indica buena capacidad predictiva
- 8 factores de riesgo identificados con significancia estadística
- Features de sesiones (frequencia, regularidad) son los más predictivos
- El modelo puede detectar ~62% de estudiantes en riesgo

Limitaciones:
- Solo 373 estudiantes de 10 cursos
- Datos de un único semestre
- Solo cursos con buena diversidad de clase
- Features basados en actividad, no contenido

Métricas de rendimiento:
- Mejor modelo: XGBoost
- ROC-AUC: 0.787
- Recall: 61.7%
- Precision: 69.7%
"""
    }

    return contexts.get(section, "")


# =============================================================================
# ENSAMBLAJE DEL REPORTE
# =============================================================================

def assemble_report(sections):
    """Ensamblar el reporte final en Markdown."""
    header = f"""# Reporte Técnico: Análisis Predictivo de Fracaso Estudiantil
## Universidad Autónoma de Chile - Canvas LMS

**Fecha de generación:** {datetime.now().strftime("%d de %B de %Y")}
**Programa analizado:** Ingeniería en Control de Gestión y otros
**Ambiente:** TEST (uautonoma.test.instructure.com)

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Metodología de Selección de Cursos](#metodología-de-selección-de-cursos)
3. [Fuentes de Datos](#fuentes-de-datos)
4. [Ingeniería de Features](#ingeniería-de-features)
5. [Metodología de Modelos Predictivos](#metodología-de-modelos-predictivos)
6. [Resultados Generales](#resultados-generales)
7. [Insights Accionables](#insights-accionables)
8. [Análisis por Curso](#análisis-por-curso)
9. [Conclusiones y Recomendaciones](#conclusiones-y-recomendaciones)

---

"""

    body = ""

    # Sección 1: Resumen Ejecutivo
    body += "## 1. Resumen Ejecutivo\n\n"
    body += sections.get("resumen_ejecutivo", "[Pendiente]") + "\n\n---\n\n"

    # Sección 2: Metodología de Selección
    body += "## 2. Metodología de Selección de Cursos\n\n"
    body += sections.get("metodologia_seleccion", "[Pendiente]") + "\n\n"
    body += "![Distribución de Diversidad](visualizations/diversity_distribution.png)\n\n"
    body += "![Tasa de Aprobación por Curso](visualizations/pass_rate_bars.png)\n\n---\n\n"

    # Sección 3: Fuentes de Datos
    body += """## 3. Fuentes de Datos (Canvas API)

### Endpoints Utilizados

| Endpoint | Propósito | Campos Extraídos |
|----------|-----------|------------------|
| `GET /courses/:id/enrollments` | Notas finales | `grades.final_score` |
| `GET /courses/:id/analytics/student_summaries` | Métricas agregadas | `page_views`, `participations`, `tardiness_breakdown` |
| `GET /courses/:id/analytics/users/:id/activity` | Actividad por hora | `page_views` (dict con timestamps) |
| `GET /courses/:id/modules` | Estructura del curso | `state`, `completed_at` |

### Flujo de Extracción

```
Canvas API → Extracción por curso →
Procesamiento de timestamps → 54 Features por estudiante →
Normalización within-course (z-scores) → Modelo predictivo
```

---

"""

    # Sección 4: Features
    body += "## 4. Ingeniería de Features\n\n"
    body += sections.get("features", "[Pendiente]") + "\n\n---\n\n"

    # Sección 5: Metodología de Modelos
    body += "## 5. Metodología de Modelos Predictivos\n\n"
    body += sections.get("metodologia_modelos", "[Pendiente]") + "\n\n---\n\n"

    # Sección 6: Resultados
    body += "## 6. Resultados Generales\n\n"
    body += sections.get("resultados", "[Pendiente]") + "\n\n"
    body += "![Curvas ROC](visualizations/roc_curves.png)\n\n"
    body += "![Importancia de Features](visualizations/feature_importance.png)\n\n---\n\n"

    # Sección 7: Insights
    body += "## 7. Insights Accionables\n\n"
    body += sections.get("insights", "[Pendiente]") + "\n\n"
    body += "![Factores de Riesgo](visualizations/risk_factors.png)\n\n"
    body += "![Comparación Aprobados vs Reprobados](visualizations/pass_fail_comparison.png)\n\n---\n\n"

    # Sección 8: Análisis por Curso
    body += "## 8. Análisis por Curso\n\n"
    body += sections.get("analisis_por_curso", "[Pendiente]") + "\n\n"
    body += "![Boxplot de Notas](visualizations/grade_boxplot.png)\n\n"
    body += "![Heatmap de Correlaciones](visualizations/correlation_heatmap.png)\n\n---\n\n"

    # Sección 9: Conclusiones
    body += "## 9. Conclusiones y Recomendaciones\n\n"
    body += sections.get("conclusiones", "[Pendiente]") + "\n\n---\n\n"

    # Footer
    footer = f"""
---

*Reporte generado automáticamente por `scripts/generate_technical_report.py`*
*Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}*
"""

    return header + body + footer


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Función principal."""
    print("=" * 70)
    print("GENERADOR DE REPORTE TÉCNICO - ANÁLISIS PREDICTIVO LMS")
    print("=" * 70)

    # Crear directorios
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIZ_DIR, exist_ok=True)

    # 1. Cargar datos
    print("\n1. Cargando datos...")
    data = load_all_data()
    print(f"   - {len(data['per_course'])} cursos cargados")
    print(f"   - {len(data['student_features'])} estudiantes")
    print(f"   - {len(data['insights'])} insights")

    # 2. Generar visualizaciones
    print("\n2. Generando visualizaciones...")
    create_visualizations(data)
    print(f"   - Visualizaciones guardadas en {VIZ_DIR}")

    # 3. Generar secciones con Claude API
    print("\n3. Generando secciones con Claude API...")
    sections = {}

    section_order = [
        "resumen_ejecutivo",
        "metodologia_seleccion",
        "features",
        "metodologia_modelos",
        "resultados",
        "insights",
        "analisis_por_curso",
        "conclusiones"
    ]

    for section in section_order:
        context = prepare_context(data, section)
        sections[section] = generate_section(section, context)

    # 4. Ensamblar reporte
    print("\n4. Ensamblando reporte final...")
    report = assemble_report(sections)

    # 5. Guardar reporte
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'=' * 70}")
    print("REPORTE GENERADO EXITOSAMENTE")
    print(f"{'=' * 70}")
    print(f"\nArchivo: {REPORT_FILE}")
    print(f"Visualizaciones: {VIZ_DIR}/")

    # Estadísticas
    word_count = len(report.split())
    print(f"\nEstadísticas:")
    print(f"  - Palabras: ~{word_count}")
    print(f"  - Secciones: {len(sections)}")


if __name__ == "__main__":
    main()
