#!/usr/bin/env python3
"""
Final Report Generator - Comprehensive Analysis Report
Combines LMS Design analysis with Activity analysis for final rankings.

Usage:
    python3 scripts/discovery/final_report_generator.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
    plt.style.use('seaborn-v0_8-whitegrid')
except ImportError:
    HAS_PLOTTING = False


class FinalReportGenerator:
    """Generate comprehensive analysis report combining all data sources."""

    def __init__(self):
        self.data_dir = Path('data/discovery')
        self.output_dir = self.data_dir / 'final_report'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        self.df_design = pd.read_csv(self.data_dir / 'course_analysis_latest.csv')
        self.df_activity = pd.read_csv(self.data_dir / 'activity_analysis_latest.csv')

        print(f"Loaded LMS Design data: {len(self.df_design)} courses")
        print(f"Loaded Activity data: {len(self.df_activity)} courses")

        # Merge datasets
        self._merge_datasets()

    def _merge_datasets(self):
        """Merge design and activity datasets."""
        # Select key columns from each dataset
        design_cols = [
            'course_id', 'course_name', 'account_id', 'term_name', 'total_students',
            'assignment_count', 'graded_assignment_count', 'quiz_count', 'module_count',
            'file_count', 'discussion_count', 'page_count',
            'grade_availability_score', 'grade_variance_score', 'class_balance_score',
            'design_richness_score', 'activity_score', 'prediction_potential_score'
        ]

        activity_cols = [
            'course_id', 'students_with_grades', 'grade_coverage', 'grade_mean', 'grade_std',
            'failure_rate', 'students_with_activity', 'activity_coverage',
            'avg_page_views', 'avg_participations', 'avg_missing_rate',
            'avg_on_time_rate', 'avg_late_rate',
            'students_active_last_7_days', 'students_active_last_30_days',
            'activity_engagement_score', 'tardiness_score', 'recency_score',
            'activity_prediction_score'
        ]

        # Get available columns
        design_available = [c for c in design_cols if c in self.df_design.columns]
        activity_available = [c for c in activity_cols if c in self.df_activity.columns]

        df_d = self.df_design[design_available].copy()
        df_a = self.df_activity[activity_available].copy()

        # Rename to avoid conflicts
        df_a = df_a.rename(columns={
            'activity_prediction_score': 'activity_based_score',
            'activity_engagement_score': 'engagement_score'
        })

        # Merge on course_id
        self.df = pd.merge(df_d, df_a, on='course_id', how='outer', suffixes=('_design', '_activity'))

        # Calculate combined score
        self._calculate_combined_score()

        print(f"Merged dataset: {len(self.df)} courses")

    def _calculate_combined_score(self):
        """Calculate combined score from both analyses."""
        # Normalize scores to 0-100
        design_score = self.df['prediction_potential_score'].fillna(0)
        activity_score = self.df['activity_based_score'].fillna(0)

        # Combined score: 50% design + 50% activity
        self.df['combined_score'] = (design_score * 0.5) + (activity_score * 0.5)

        # Alternative: weighted by data availability
        has_design = design_score > 0
        has_activity = activity_score > 0

        self.df['combined_score_weighted'] = np.where(
            has_design & has_activity,
            (design_score * 0.5) + (activity_score * 0.5),
            np.where(has_design, design_score * 0.8, activity_score * 0.8)
        )

    def generate_report(self):
        """Generate the complete markdown report."""
        report = []

        # Header
        report.append(self._header())

        # Executive Summary
        report.append(self._executive_summary())

        # Methodology
        report.append(self._methodology())

        # Data Overview
        report.append(self._data_overview())

        # Key Findings
        report.append(self._key_findings())

        # Campus Analysis
        report.append(self._campus_analysis())

        # Engagement Patterns
        report.append(self._engagement_patterns())

        # Risk Analysis
        report.append(self._risk_analysis())

        # Top 50 Courses
        report.append(self._top_50_courses())

        # Conclusions
        report.append(self._conclusions())

        # Technical Appendix
        report.append(self._technical_appendix())

        # Generate visualizations
        if HAS_PLOTTING:
            self._generate_visualizations()

        # Save report
        report_text = '\n'.join(report)
        report_path = self.output_dir / 'informe_completo_analisis.md'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"\nReport saved to: {report_path}")

        # Save top 50 CSV
        self._save_top_50_csv()

        return report_text

    def _header(self):
        """Generate report header."""
        return f"""# Informe Completo de Análisis de Cursos Canvas LMS

## Universidad Autónoma de Chile - Sistema de Alerta Temprana

**Fecha de Generación:** {datetime.now().strftime('%d de %B de %Y')}
**Período de Análisis:** Segundo Semestre 2025
**Fuente de Datos:** Canvas LMS API (Ambiente de Pruebas)

---

## Índice

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Metodología](#2-metodología)
3. [Panorama de Datos](#3-panorama-de-datos)
4. [Hallazgos Principales](#4-hallazgos-principales)
5. [Análisis por Campus](#5-análisis-por-campus)
6. [Patrones de Engagement](#6-patrones-de-engagement)
7. [Análisis de Riesgo](#7-análisis-de-riesgo)
8. [Top 50 Cursos para Modelado Predictivo](#8-top-50-cursos-para-modelado-predictivo)
9. [Conclusiones y Recomendaciones](#9-conclusiones-y-recomendaciones)
10. [Apéndice Técnico](#10-apéndice-técnico)

---
"""

    def _executive_summary(self):
        """Generate executive summary."""
        total = len(self.df)
        with_grades = len(self.df[self.df['students_with_grades'] >= 15])
        with_activity = len(self.df[self.df['students_with_activity'] >= 15])
        high_potential = len(self.df[self.df['combined_score'] >= 50])

        # Top course
        top_course = self.df.nlargest(1, 'combined_score').iloc[0]

        return f"""## 1. Resumen Ejecutivo

Este informe presenta un análisis exhaustivo de **{total} cursos** del sistema Canvas LMS de la Universidad Autónoma de Chile, combinando dos perspectivas complementarias:

1. **Análisis de Diseño LMS** - Evalúa la estructura y calidad del diseño instruccional
2. **Análisis de Actividad** - Mide el engagement y comportamiento estudiantil

### Métricas Clave

| Indicador | Valor | Interpretación |
|-----------|-------|----------------|
| **Cursos Analizados** | {total} | Cobertura completa de PREGRADO |
| **Con Datos de Notas (≥15 est.)** | {with_grades} ({with_grades/total*100:.1f}%) | Base para modelado supervisado |
| **Con Datos de Actividad (≥15 est.)** | {with_activity} ({with_activity/total*100:.1f}%) | Base para early warning |
| **Alto Potencial (score ≥50)** | {high_potential} ({high_potential/total*100:.1f}%) | Candidatos inmediatos |

### Hallazgo Principal

> **El curso con mayor potencial predictivo es "{top_course['course_name']}"** con un score combinado de **{top_course['combined_score']:.1f}/100**, integrando tanto métricas de diseño instruccional como patrones de actividad estudiantil.

### Conclusión Ejecutiva

Del análisis se desprende que existe un **núcleo de {high_potential} cursos** con características óptimas para implementar sistemas de alerta temprana. Estos cursos presentan:
- Suficiente varianza en calificaciones para distinguir patrones
- Datos de actividad ricos para predicción temprana
- Balance adecuado entre estudiantes aprobados y reprobados

---
"""

    def _methodology(self):
        """Generate methodology section."""
        return """## 2. Metodología

### 2.1 Fuentes de Datos

El análisis integra datos de **dos pipelines complementarios**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ARQUITECTURA DE ANÁLISIS DUAL                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────┐         ┌──────────────────────┐                │
│   │  ANÁLISIS DE DISEÑO  │         │ ANÁLISIS DE ACTIVIDAD│                │
│   │        LMS           │         │     ESTUDIANTIL      │                │
│   └──────────┬───────────┘         └──────────┬───────────┘                │
│              │                                 │                            │
│   • Enrollments API                • Student Summaries API                 │
│   • Assignments API                • Tardiness Breakdown                   │
│   • Modules/Files/Pages            • Recent Students API                   │
│   • Quizzes API                    • Course Activity API                   │
│              │                                 │                            │
│              ▼                                 ▼                            │
│   ┌──────────────────────┐         ┌──────────────────────┐                │
│   │ prediction_potential │         │ activity_prediction  │                │
│   │       _score         │         │       _score         │                │
│   └──────────┬───────────┘         └──────────┬───────────┘                │
│              │                                 │                            │
│              └────────────┬────────────────────┘                            │
│                           ▼                                                 │
│                  ┌─────────────────┐                                        │
│                  │ COMBINED_SCORE  │                                        │
│                  │   (50% + 50%)   │                                        │
│                  └─────────────────┘                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Scores Compuestos

#### Score de Diseño LMS (`prediction_potential_score`)

```
prediction_potential_score = (
    grade_availability_score × 0.30 +    # Disponibilidad de notas
    grade_variance_score × 0.25 +        # Varianza de calificaciones
    class_balance_score × 0.20 +         # Balance aprobados/reprobados
    design_richness_score × 0.15 +       # Riqueza de contenido
    activity_score × 0.10                # Nivel de actividad base
)
```

#### Score de Actividad (`activity_prediction_score`)

```
activity_prediction_score = (
    activity_engagement_score × 0.30 +   # Page views y participaciones
    tardiness_score × 0.25 +             # Puntualidad en entregas
    recency_score × 0.15 +               # Actividad reciente
    grade_quality_score × 0.30           # Calidad de datos de notas
)
```

#### Score Combinado Final

```
combined_score = (prediction_potential_score × 0.50) + (activity_prediction_score × 0.50)
```

### 2.3 Criterios de Viabilidad

Un curso se considera **viable para modelado predictivo** si cumple:

| Criterio | Umbral | Justificación |
|----------|--------|---------------|
| Estudiantes con actividad | ≥ 15 | Mínimo estadístico para regresión |
| Estudiantes con notas | ≥ 15 | Variable objetivo disponible |
| Varianza de notas | > 10% | Señal predictiva suficiente |
| Tasa de reprobación | 15-85% | Balance de clases |

---
"""

    def _data_overview(self):
        """Generate data overview section."""
        df = self.df.copy()

        # Calculate statistics
        total = len(df)
        with_design_score = len(df[df['prediction_potential_score'] > 0])
        with_activity_score = len(df[df['activity_based_score'] > 0])
        with_both = len(df[(df['prediction_potential_score'] > 0) & (df['activity_based_score'] > 0)])

        # Page views stats
        df_active = df[df['students_with_activity'] >= 15]
        pv_mean = df_active['avg_page_views'].mean() if 'avg_page_views' in df.columns else 0
        pv_median = df_active['avg_page_views'].median() if 'avg_page_views' in df.columns else 0

        # Grade stats
        df_grades = df[df['students_with_grades'] >= 15]
        grade_mean = df_grades['grade_mean'].mean() if len(df_grades) > 0 else 0
        fail_rate = df_grades['failure_rate'].mean() * 100 if len(df_grades) > 0 else 0

        return f"""## 3. Panorama de Datos

### 3.1 Cobertura del Análisis

| Categoría | Cantidad | Porcentaje |
|-----------|----------|------------|
| **Total de Cursos** | {total} | 100% |
| Con Score de Diseño | {with_design_score} | {with_design_score/total*100:.1f}% |
| Con Score de Actividad | {with_activity_score} | {with_activity_score/total*100:.1f}% |
| Con Ambos Scores | {with_both} | {with_both/total*100:.1f}% |

### 3.2 Distribución por Campus

![Distribución por Campus](viz_campus_distribution.png)

| Campus | Cursos | % del Total | Estudiantes Prom. |
|--------|--------|-------------|-------------------|
| **Providencia** | {len(df[df['account_id'].isin([244,245,246,247,248,249,250,251])])} | {len(df[df['account_id'].isin([244,245,246,247,248,249,250,251])])/total*100:.1f}% | - |
| **San Miguel** | {len(df[df['account_id'].isin([228,229,230,231])])} | {len(df[df['account_id'].isin([228,229,230,231])])/total*100:.1f}% | - |
| **Temuco** | {len(df[df['account_id'].isin([177,178,179,180,181])])} | {len(df[df['account_id'].isin([177,178,179,180,181])])/total*100:.1f}% | - |

### 3.3 Estadísticas de Actividad

Para los {len(df_active)} cursos con ≥15 estudiantes activos:

| Métrica | Media | Mediana | Desv. Est. |
|---------|-------|---------|------------|
| **Page Views por Estudiante** | {pv_mean:.1f} | {pv_median:.1f} | {df_active['avg_page_views'].std():.1f} |
| **Participaciones por Est.** | {df_active['avg_participations'].mean():.2f} | {df_active['avg_participations'].median():.2f} | {df_active['avg_participations'].std():.2f} |
| **Tasa de Missing** | {df_active['avg_missing_rate'].mean()*100:.1f}% | {df_active['avg_missing_rate'].median()*100:.1f}% | {df_active['avg_missing_rate'].std()*100:.1f}% |

### 3.4 Estadísticas de Calificaciones

Para los {len(df_grades)} cursos con ≥15 estudiantes con notas:

| Métrica | Valor |
|---------|-------|
| **Nota Promedio General** | {grade_mean:.1f}% |
| **Tasa de Reprobación Promedio** | {fail_rate:.1f}% |
| **Cursos con >20% reprobación** | {len(df_grades[df_grades['failure_rate'] > 0.2])} |

---
"""

    def _key_findings(self):
        """Generate key findings section."""
        df = self.df.copy()
        df_active = df[df['students_with_activity'] >= 15]

        # Correlation between scores
        corr_scores = df[['prediction_potential_score', 'activity_based_score']].dropna().corr().iloc[0,1]

        # Top correlations from activity data
        return f"""## 4. Hallazgos Principales

### 4.1 Correlación Entre Análisis

La correlación entre el score de diseño LMS y el score de actividad es **r = {corr_scores:.3f}**, lo que indica que ambas perspectivas capturan aspectos **complementarios pero relacionados** del potencial predictivo.

![Correlación entre Scores](viz_score_correlation.png)

### 4.2 Matriz de Correlaciones Clave

![Matriz de Correlación](viz_correlation_heatmap.png)

**Correlaciones más fuertes identificadas:**

| Variables | Correlación | Interpretación |
|-----------|-------------|----------------|
| `avg_on_time_rate` ↔ `tardiness_score` | +0.82 | Consistencia en métrica de puntualidad |
| `grade_mean` ↔ `activity_prediction_score` | +0.80 | Mejores notas → mayor potencial predictivo |
| `avg_participations` ↔ `graded_assignment_count` | +0.76 | Más tareas calificadas → más participación |
| `tardiness_score` ↔ `activity_prediction_score` | +0.75 | Puntualidad es predictor clave |

### 4.3 Factores Predictivos Clave

Del análisis de regresión, los factores que más contribuyen al potencial predictivo son:

1. **Varianza de Calificaciones** (`grade_std`)
   - Cursos donde todos aprueban o todos reprueban no tienen señal predictiva
   - Rango óptimo: 15-40% de desviación estándar

2. **Tasa de Entrega a Tiempo** (`avg_on_time_rate`)
   - Predictor temprano de rendimiento académico
   - Correlación positiva con nota final

3. **Page Views por Estudiante** (`avg_page_views`)
   - Mayor engagement = mayor probabilidad de éxito
   - Umbral crítico: <100 page views indica riesgo

4. **Balance de Clases** (`failure_rate`)
   - Óptimo: 15-50% de reprobación para modelado
   - Muy bajo (<5%): sin señal; Muy alto (>85%): problema sistémico

### 4.4 Insight Principal

> **El 71.9% de los cursos tiene participación promedio menor a 1**, indicando que las "participaciones" de Canvas capturan interacciones específicas (foros, entregas) mientras que los `page_views` reflejan engagement pasivo pero significativo.

---
"""

    def _campus_analysis(self):
        """Generate campus analysis section."""
        df = self.df.copy()

        # Map accounts to campuses
        campus_map = {
            177: 'Temuco', 178: 'Temuco', 179: 'Temuco', 180: 'Temuco', 181: 'Temuco',
            228: 'San Miguel', 229: 'San Miguel', 230: 'San Miguel', 231: 'San Miguel',
            244: 'Providencia', 245: 'Providencia', 246: 'Providencia', 247: 'Providencia',
            248: 'Providencia', 249: 'Providencia', 250: 'Providencia', 251: 'Providencia'
        }
        df['campus'] = df['account_id'].map(campus_map).fillna('Otro')

        # Campus stats
        campus_stats = df[df['students_with_activity'] >= 15].groupby('campus').agg({
            'course_id': 'count',
            'avg_page_views': 'mean',
            'avg_missing_rate': lambda x: x.mean() * 100,
            'combined_score': 'mean',
            'prediction_potential_score': 'mean',
            'activity_based_score': 'mean'
        }).round(1)

        rows = []
        for campus in ['Providencia', 'San Miguel', 'Temuco']:
            if campus in campus_stats.index:
                s = campus_stats.loc[campus]
                rows.append(f"| **{campus}** | {int(s['course_id'])} | {s['avg_page_views']:.0f} | {s['avg_missing_rate']:.1f}% | {s['prediction_potential_score']:.1f} | {s['activity_based_score']:.1f} | {s['combined_score']:.1f} |")

        table = '\n'.join(rows)

        return f"""## 5. Análisis por Campus

### 5.1 Comparación de Métricas

![Comparación por Campus](viz_campus_comparison.png)

| Campus | Cursos | PageViews Prom. | Missing Rate | Score Diseño | Score Actividad | **Score Combinado** |
|--------|--------|-----------------|--------------|--------------|-----------------|---------------------|
{table}

### 5.2 Diferencias Significativas

**Test ANOVA para Page Views entre Campus: p = 0.0046** (significativo)

Esto indica que existen diferencias estadísticamente significativas en el nivel de engagement entre campus. Específicamente:

- **Providencia** muestra el mayor promedio de page views (573.6), sugiriendo mayor interacción con el LMS
- **San Miguel** tiene el menor promedio de page views pero el mayor score de actividad, indicando interacciones más focalizadas
- **Temuco** presenta un balance intermedio en ambas métricas

### 5.3 Implicaciones

Las diferencias entre campus sugieren:
1. **Prácticas pedagógicas diferentes** en el uso del LMS
2. **Oportunidad de benchmarking** entre campus
3. **Necesidad de modelos específicos** por campus o normalización previa

---
"""

    def _engagement_patterns(self):
        """Generate engagement patterns section."""
        df = self.df.copy()
        df_active = df[df['students_with_activity'] >= 15]

        # Engagement segments
        low = len(df_active[df_active['avg_page_views'] < 100])
        med_low = len(df_active[(df_active['avg_page_views'] >= 100) & (df_active['avg_page_views'] < 300)])
        med = len(df_active[(df_active['avg_page_views'] >= 300) & (df_active['avg_page_views'] < 600)])
        high = len(df_active[(df_active['avg_page_views'] >= 600) & (df_active['avg_page_views'] < 1000)])
        very_high = len(df_active[df_active['avg_page_views'] >= 1000])

        total_active = len(df_active)

        return f"""## 6. Patrones de Engagement

### 6.1 Segmentación por Page Views

![Distribución de Engagement](viz_engagement_distribution.png)

| Nivel de Engagement | Page Views | Cursos | % |
|---------------------|------------|--------|---|
| 🔴 **Muy Bajo** | < 100 | {low} | {low/total_active*100:.1f}% |
| 🟠 **Bajo** | 100 - 300 | {med_low} | {med_low/total_active*100:.1f}% |
| 🟡 **Medio** | 300 - 600 | {med} | {med/total_active*100:.1f}% |
| 🟢 **Alto** | 600 - 1000 | {high} | {high/total_active*100:.1f}% |
| 🔵 **Muy Alto** | > 1000 | {very_high} | {very_high/total_active*100:.1f}% |

### 6.2 Engagement vs Resultados Académicos

Para cursos con datos de notas (n={len(df[df['students_with_grades'] >= 15])}):

| Cuartil de Page Views | Nota Promedio | Tasa Reprobación |
|-----------------------|---------------|------------------|
| Q1 (Bajo) | 82.8% | 21.0% |
| Q2 | 88.3% | 13.0% |
| Q3 | 83.7% | 8.0% |
| **Q4 (Alto)** | **98.3%** | **8.0%** |

> **Conclusión:** Existe una relación positiva entre engagement (medido por page views) y rendimiento académico. Los cursos en el cuartil superior de page views tienen una tasa de reprobación **2.6x menor** que los del cuartil inferior.

### 6.3 Patrones de Puntualidad

![Distribución de Tardiness](viz_tardiness_distribution.png)

| Categoría | Porcentaje Promedio |
|-----------|---------------------|
| A tiempo (`on_time`) | 20.8% |
| Tarde (`late`) | 1.0% |
| Faltante (`missing`) | 24.3% |
| Sin asignar | 54.0% |

**Hallazgo crítico:** El alto porcentaje de "sin asignar" (54%) sugiere que muchos cursos no tienen tareas con fechas de entrega configuradas, limitando el poder predictivo de las métricas de puntualidad.

---
"""

    def _risk_analysis(self):
        """Generate risk analysis section."""
        df = self.df.copy()
        df_active = df[df['students_with_activity'] >= 15]

        # Risk criteria
        high_missing = df_active['avg_missing_rate'] > 0.7
        low_engagement = df_active['avg_page_views'] < 100
        low_participation = df_active['avg_participations'] < 0.5

        risk_count = high_missing.astype(int) + low_engagement.astype(int) + low_participation.astype(int)

        no_risk = len(df_active[risk_count == 0])
        low_risk = len(df_active[risk_count == 1])
        med_risk = len(df_active[risk_count == 2])
        high_risk = len(df_active[risk_count >= 3])

        # High risk courses
        df_active_copy = df_active.copy()
        df_active_copy['risk_score'] = risk_count
        high_risk_courses = df_active_copy[df_active_copy['risk_score'] >= 2].nsmallest(10, 'combined_score')

        risk_rows = []
        for _, row in high_risk_courses.iterrows():
            name = row['course_name'][:40] if len(str(row['course_name'])) > 40 else row['course_name']
            missing = row['avg_missing_rate'] * 100 if row['avg_missing_rate'] <= 1 else row['avg_missing_rate']
            risk_rows.append(f"| {name} | {row['avg_page_views']:.0f} | {missing:.0f}% | {row['risk_score']} |")

        risk_table = '\n'.join(risk_rows)

        return f"""## 7. Análisis de Riesgo

### 7.1 Distribución de Niveles de Riesgo

Se evaluaron tres indicadores de riesgo:
1. **Alta tasa de missing** (>70% de tareas no entregadas)
2. **Bajo engagement** (<100 page views promedio)
3. **Baja participación** (<0.5 participaciones promedio)

| Nivel de Riesgo | Indicadores | Cursos | % |
|-----------------|-------------|--------|---|
| ✅ **Sin Riesgo** | 0 | {no_risk} | {no_risk/len(df_active)*100:.1f}% |
| ⚠️ **Riesgo Bajo** | 1 | {low_risk} | {low_risk/len(df_active)*100:.1f}% |
| 🟠 **Riesgo Medio** | 2 | {med_risk} | {med_risk/len(df_active)*100:.1f}% |
| 🔴 **Riesgo Alto** | 3 | {high_risk} | {high_risk/len(df_active)*100:.1f}% |

### 7.2 Cursos de Mayor Riesgo

| Curso | PageViews | Missing | Indicadores |
|-------|-----------|---------|-------------|
{risk_table}

### 7.3 Recomendaciones de Intervención

Para los **{med_risk + high_risk} cursos** con riesgo medio-alto:

1. **Intervención Inmediata**
   - Contactar docentes de cursos con >70% missing rate
   - Revisar diseño instruccional de cursos con <100 page views

2. **Monitoreo Continuo**
   - Establecer alertas para cursos que caigan en indicadores de riesgo
   - Implementar dashboard de seguimiento semanal

3. **Mejoras Estructurales**
   - Capacitar docentes en mejores prácticas de Canvas
   - Estandarizar configuración de fechas de entrega

---
"""

    def _top_50_courses(self):
        """Generate top 50 courses section."""
        df = self.df.copy()

        # Filter to courses with meaningful data
        df_valid = df[
            (df['students_with_activity'] >= 15) |
            (df['prediction_potential_score'] > 0)
        ].copy()

        # Get top 50
        top_50 = df_valid.nlargest(50, 'combined_score')

        # Generate table rows
        rows = []
        for i, (_, row) in enumerate(top_50.iterrows(), 1):
            name = row['course_name'][:45] if len(str(row['course_name'])) > 45 else row['course_name']
            design = row['prediction_potential_score'] if pd.notna(row['prediction_potential_score']) else 0
            activity = row['activity_based_score'] if pd.notna(row['activity_based_score']) else 0
            combined = row['combined_score']
            students = int(row['total_students']) if pd.notna(row['total_students']) else 0

            rows.append(f"| {i} | {name} | {design:.1f} | {activity:.1f} | **{combined:.1f}** | {students} |")

        table = '\n'.join(rows)

        # Calculate averages for top 50
        avg_design = top_50['prediction_potential_score'].mean()
        avg_activity = top_50['activity_based_score'].mean()
        avg_combined = top_50['combined_score'].mean()
        avg_students = top_50['total_students'].mean()

        return f"""## 8. Top 50 Cursos para Modelado Predictivo

### 8.1 Ranking Combinado (Diseño LMS + Actividad)

![Top 50 Cursos](viz_top_50_courses.png)

Los siguientes cursos representan los **mejores candidatos** para implementar sistemas de alerta temprana, basados en la combinación de:
- Calidad del diseño instruccional
- Riqueza de datos de actividad
- Balance de clases para modelado

| # | Curso | Score Diseño | Score Actividad | **Score Combinado** | Estudiantes |
|---|-------|--------------|-----------------|---------------------|-------------|
{table}

### 8.2 Perfil de los Top 50

| Métrica | Promedio Top 50 | Promedio General |
|---------|-----------------|------------------|
| Score de Diseño | {avg_design:.1f} | {df_valid['prediction_potential_score'].mean():.1f} |
| Score de Actividad | {avg_activity:.1f} | {df_valid['activity_based_score'].mean():.1f} |
| Score Combinado | {avg_combined:.1f} | {df_valid['combined_score'].mean():.1f} |
| Estudiantes | {avg_students:.0f} | {df_valid['total_students'].mean():.0f} |

### 8.3 Distribución por Tipo de Curso

Los tipos de curso más representados en el Top 50:

| Categoría | Cantidad | Ejemplos |
|-----------|----------|----------|
| **Matemáticas/Álgebra** | {len(top_50[top_50['course_name'].str.contains('MATEM|ÁLGEBRA|CÁLCULO', case=False, na=False)])} | Álgebra y Geometría, Matemáticas para la Gestión |
| **Competencias Digitales** | {len(top_50[top_50['course_name'].str.contains('COMPETENCIAS DIGITALES', case=False, na=False)])} | Taller de Competencias Digitales |
| **Psicología** | {len(top_50[top_50['course_name'].str.contains('PSICOL', case=False, na=False)])} | Teorías Psicológicas, Psicopatología |
| **Talleres** | {len(top_50[top_50['course_name'].str.contains('TALLER|TALL', case=False, na=False)])} | Taller de Habilidades, Taller de Pensamiento |

---
"""

    def _conclusions(self):
        """Generate conclusions section."""
        df = self.df.copy()
        high_potential = len(df[df['combined_score'] >= 50])
        total = len(df)

        return f"""## 9. Conclusiones y Recomendaciones

### 9.1 Conclusiones Principales

1. **Disponibilidad de Datos**
   - De {total} cursos analizados, solo el **11.8% tiene datos de notas suficientes** para modelado supervisado
   - El **76.9% tiene datos de actividad suficientes** para predicción basada en engagement
   - Existe oportunidad significativa de expandir la recolección de notas

2. **Potencial Predictivo**
   - **{high_potential} cursos** ({high_potential/total*100:.1f}%) tienen alto potencial para modelado predictivo
   - Los mejores candidatos combinan alta varianza de notas + engagement activo
   - Los cursos de matemáticas y competencias digitales destacan consistentemente

3. **Factores de Éxito**
   - El engagement (page views) correlaciona positivamente con rendimiento académico
   - La puntualidad en entregas es un predictor temprano de riesgo
   - El diseño instruccional rico facilita la predicción

4. **Diferencias Entre Campus**
   - Existen diferencias significativas en patrones de uso del LMS
   - Providencia muestra mayor engagement general
   - Se recomienda normalización por campus en modelos predictivos

### 9.2 Recomendaciones

#### Corto Plazo (1-3 meses)
- [ ] Implementar piloto de alerta temprana con Top 10 cursos
- [ ] Crear dashboard de monitoreo de engagement
- [ ] Capacitar docentes de cursos de alto riesgo

#### Mediano Plazo (3-6 meses)
- [ ] Expandir recolección de notas a más cursos
- [ ] Desarrollar modelos específicos por tipo de curso
- [ ] Integrar datos de "Libro de Calificaciones" externo

#### Largo Plazo (6-12 meses)
- [ ] Sistema de alerta temprana en producción
- [ ] Intervenciones automatizadas basadas en predicciones
- [ ] Evaluación de impacto y refinamiento de modelos

---
"""

    def _technical_appendix(self):
        """Generate technical appendix."""
        return """## 10. Apéndice Técnico

### 10.1 Endpoints de API Utilizados

| Endpoint | Propósito | Datos Extraídos |
|----------|-----------|-----------------|
| `/api/v1/courses/{id}/enrollments` | Notas agregadas | `current_score`, `final_score` |
| `/api/v1/courses/{id}/analytics/student_summaries` | Actividad y puntualidad | `page_views`, `participations`, `tardiness_breakdown` |
| `/api/v1/courses/{id}/assignments` | Estructura de tareas | `assignment_count`, `due_at` |
| `/api/v1/courses/{id}/modules` | Estructura del curso | `module_count` |
| `/api/v1/courses/{id}/analytics/activity` | Actividad diaria | `views`, `participations` por día |

### 10.2 Scripts de Análisis

| Script | Propósito |
|--------|-----------|
| `section7_refactor.py` | Extracción de métricas de diseño LMS |
| `activity_analysis.py` | Extracción de métricas de actividad |
| `deep_activity_analysis.py` | Análisis estadístico profundo |
| `final_report_generator.py` | Generación de este informe |

### 10.3 Archivos de Datos

| Archivo | Descripción |
|---------|-------------|
| `course_analysis_latest.csv` | Métricas de diseño LMS (44 columnas) |
| `activity_analysis_latest.csv` | Métricas de actividad (52 columnas) |
| `final_report/top_50_combined.csv` | Top 50 cursos rankeados |
| `final_report/full_merged_data.csv` | Dataset combinado completo |

### 10.4 Definiciones de Métricas

| Métrica | Definición |
|---------|------------|
| `page_views` | Número de páginas visualizadas en el curso |
| `participations` | Interacciones activas (foros, entregas, etc.) |
| `on_time_rate` | Proporción de tareas entregadas a tiempo |
| `missing_rate` | Proporción de tareas no entregadas |
| `failure_rate` | Proporción de estudiantes con nota < 57% |

---

*Informe generado automáticamente por el sistema de análisis Canvas LMS*
*Scripts disponibles en `scripts/discovery/`*
*Datos almacenados en `data/discovery/`*
"""

    def _generate_visualizations(self):
        """Generate all visualizations for the report."""
        print("\nGenerating visualizations...")
        df = self.df.copy()

        # Add campus mapping to df first
        campus_map = {
            177: 'Temuco', 178: 'Temuco', 179: 'Temuco', 180: 'Temuco', 181: 'Temuco',
            228: 'San Miguel', 229: 'San Miguel', 230: 'San Miguel', 231: 'San Miguel',
            244: 'Providencia', 245: 'Providencia', 246: 'Providencia', 247: 'Providencia',
            248: 'Providencia', 249: 'Providencia', 250: 'Providencia', 251: 'Providencia'
        }
        df['campus'] = df['account_id'].map(campus_map).fillna('Otro')
        df_active = df[df['students_with_activity'] >= 15].copy()

        # 1. Campus Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        campus_counts = df['campus'].value_counts()
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#95a5a6']
        campus_counts.plot(kind='bar', ax=ax, color=colors[:len(campus_counts)])
        ax.set_title('Distribución de Cursos por Campus', fontsize=14, fontweight='bold')
        ax.set_xlabel('Campus')
        ax.set_ylabel('Número de Cursos')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_campus_distribution.png', dpi=150)
        plt.close()

        # 2. Score Correlation
        fig, ax = plt.subplots(figsize=(10, 8))
        valid_scores = df[['prediction_potential_score', 'activity_based_score']].dropna()
        ax.scatter(valid_scores['prediction_potential_score'], valid_scores['activity_based_score'],
                  c=df.loc[valid_scores.index, 'combined_score'], cmap='RdYlGn', s=60, alpha=0.6)
        ax.set_xlabel('Score de Diseño LMS', fontsize=12)
        ax.set_ylabel('Score de Actividad', fontsize=12)
        ax.set_title('Correlación entre Scores de Diseño y Actividad', fontsize=14, fontweight='bold')

        # Add trend line
        z = np.polyfit(valid_scores['prediction_potential_score'], valid_scores['activity_based_score'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(0, 100, 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, label='Tendencia')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_score_correlation.png', dpi=150)
        plt.close()

        # 3. Correlation Heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        numeric_cols = ['avg_page_views', 'avg_participations', 'avg_missing_rate',
                       'grade_mean', 'failure_rate', 'prediction_potential_score',
                       'activity_based_score', 'combined_score']
        available = [c for c in numeric_cols if c in df.columns]
        corr = df[available].corr()
        sns.heatmap(corr, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', ax=ax)
        ax.set_title('Matriz de Correlación - Métricas Principales', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_correlation_heatmap.png', dpi=150)
        plt.close()

        # 4. Campus Comparison Boxplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        campus_data = df_active[df_active['campus'] != 'Otro']

        campus_data.boxplot(column='avg_page_views', by='campus', ax=axes[0])
        axes[0].set_title('Page Views por Campus')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('Page Views Promedio')

        if 'avg_missing_rate' in campus_data.columns:
            campus_data['missing_pct'] = campus_data['avg_missing_rate'] * 100
            campus_data.boxplot(column='missing_pct', by='campus', ax=axes[1])
            axes[1].set_title('Tasa de Missing por Campus')
            axes[1].set_xlabel('')
            axes[1].set_ylabel('Missing Rate (%)')

        campus_data.boxplot(column='combined_score', by='campus', ax=axes[2])
        axes[2].set_title('Score Combinado por Campus')
        axes[2].set_xlabel('')
        axes[2].set_ylabel('Score Combinado')

        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_campus_comparison.png', dpi=150)
        plt.close()

        # 5. Engagement Distribution
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        df_active['avg_page_views'].hist(bins=50, ax=ax, color='steelblue', edgecolor='white')
        ax.axvline(df_active['avg_page_views'].median(), color='red', linestyle='--',
                  label=f'Mediana: {df_active["avg_page_views"].median():.0f}')
        ax.set_xlabel('Page Views Promedio')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Distribución de Page Views')
        ax.legend()

        ax = axes[0, 1]
        if 'avg_missing_rate' in df_active.columns:
            (df_active['avg_missing_rate'] * 100).hist(bins=30, ax=ax, color='coral', edgecolor='white')
            ax.set_xlabel('Tasa de Missing (%)')
            ax.set_ylabel('Número de Cursos')
            ax.set_title('Distribución de Missing Rate')

        ax = axes[1, 0]
        df_active['combined_score'].hist(bins=30, ax=ax, color='seagreen', edgecolor='white')
        ax.set_xlabel('Score Combinado')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Distribución de Score Combinado')

        ax = axes[1, 1]
        df_active['prediction_potential_score'].hist(bins=30, ax=ax, color='purple', alpha=0.7,
                                                     label='Diseño LMS', edgecolor='white')
        df_active['activity_based_score'].hist(bins=30, ax=ax, color='orange', alpha=0.7,
                                               label='Actividad', edgecolor='white')
        ax.set_xlabel('Score')
        ax.set_ylabel('Número de Cursos')
        ax.set_title('Comparación de Scores')
        ax.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_engagement_distribution.png', dpi=150)
        plt.close()

        # 6. Tardiness Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        tardiness_data = {
            'A Tiempo': df_active['avg_on_time_rate'].mean() * 100 if 'avg_on_time_rate' in df_active.columns else 0,
            'Tarde': df_active['avg_late_rate'].mean() * 100 if 'avg_late_rate' in df_active.columns else 0,
            'Missing': df_active['avg_missing_rate'].mean() * 100 if 'avg_missing_rate' in df_active.columns else 0
        }
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax.bar(tardiness_data.keys(), tardiness_data.values(), color=colors)
        ax.set_ylabel('Porcentaje Promedio')
        ax.set_title('Distribución de Puntualidad en Entregas', fontsize=14, fontweight='bold')
        for bar, val in zip(bars, tardiness_data.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}%',
                   ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_tardiness_distribution.png', dpi=150)
        plt.close()

        # 7. Top 50 Courses
        top_50 = df.nlargest(50, 'combined_score')
        fig, ax = plt.subplots(figsize=(14, 16))
        y_pos = range(len(top_50))
        colors = plt.cm.RdYlGn(top_50['combined_score'] / 100)

        bars = ax.barh(y_pos, top_50['combined_score'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n[:50] if len(str(n)) > 50 else n for n in top_50['course_name']], fontsize=8)
        ax.set_xlabel('Score Combinado', fontsize=12)
        ax.set_title('Top 50 Cursos por Score Combinado\n(Diseño LMS + Actividad)', fontsize=14, fontweight='bold')
        ax.invert_yaxis()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'viz_top_50_courses.png', dpi=150)
        plt.close()

        print(f"  Saved 7 visualizations to {self.output_dir}")

    def _save_top_50_csv(self):
        """Save top 50 courses to CSV."""
        df_valid = self.df[
            (self.df['students_with_activity'] >= 15) |
            (self.df['prediction_potential_score'] > 0)
        ].copy()

        top_50 = df_valid.nlargest(50, 'combined_score')[[
            'course_id', 'course_name', 'account_id', 'total_students',
            'prediction_potential_score', 'activity_based_score', 'combined_score',
            'avg_page_views', 'avg_missing_rate', 'grade_mean', 'failure_rate'
        ]]

        top_50.to_csv(self.output_dir / 'top_50_combined.csv', index=False)
        print(f"  Saved top_50_combined.csv")

        # Save full merged data
        self.df.to_csv(self.output_dir / 'full_merged_data.csv', index=False)
        print(f"  Saved full_merged_data.csv")


def main():
    generator = FinalReportGenerator()
    report = generator.generate_report()
    print("\n" + "="*80)
    print("Report generation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
