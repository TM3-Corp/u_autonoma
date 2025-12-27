# Informe Completo de Análisis de Cursos Canvas LMS

## Universidad Autónoma de Chile - Sistema de Alerta Temprana

**Fecha de Generación:** 26 de December de 2025
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

## 1. Resumen Ejecutivo

Este informe presenta un análisis exhaustivo de **623 cursos** del sistema Canvas LMS de la Universidad Autónoma de Chile, combinando dos perspectivas complementarias:

1. **Análisis de Diseño LMS** - Evalúa la estructura y calidad del diseño instruccional
2. **Análisis de Actividad** - Mide el engagement y comportamiento estudiantil

### Métricas Clave

| Indicador | Valor | Interpretación |
|-----------|-------|----------------|
| **Cursos Analizados** | 623 | Cobertura completa de PREGRADO |
| **Con Datos de Notas (≥15 est.)** | 71 (11.4%) | Base para modelado supervisado |
| **Con Datos de Actividad (≥15 est.)** | 462 (74.2%) | Base para early warning |
| **Alto Potencial (score ≥50)** | 41 (6.6%) | Candidatos inmediatos |

### Hallazgo Principal

> **El curso con mayor potencial predictivo es "TALL DE COMPETENCIAS DIGITALES-S04"** con un score combinado de **97.7/100**, integrando tanto métricas de diseño instruccional como patrones de actividad estudiantil.

### Conclusión Ejecutiva

Del análisis se desprende que existe un **núcleo de 41 cursos** con características óptimas para implementar sistemas de alerta temprana. Estos cursos presentan:
- Suficiente varianza en calificaciones para distinguir patrones
- Datos de actividad ricos para predicción temprana
- Balance adecuado entre estudiantes aprobados y reprobados

---

## 2. Metodología

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

## 3. Panorama de Datos

### 3.1 Cobertura del Análisis

| Categoría | Cantidad | Porcentaje |
|-----------|----------|------------|
| **Total de Cursos** | 623 | 100% |
| Con Score de Diseño | 68 | 10.9% |
| Con Score de Actividad | 600 | 96.3% |
| Con Ambos Scores | 64 | 10.3% |

### 3.2 Distribución por Campus

![Distribución por Campus](viz_campus_distribution.png)

| Campus | Cursos | % del Total | Estudiantes Prom. |
|--------|--------|-------------|-------------------|
| **Providencia** | 200 | 32.1% | - |
| **San Miguel** | 200 | 32.1% | - |
| **Temuco** | 200 | 32.1% | - |

### 3.3 Estadísticas de Actividad

Para los 462 cursos con ≥15 estudiantes activos:

| Métrica | Media | Mediana | Desv. Est. |
|---------|-------|---------|------------|
| **Page Views por Estudiante** | 481.3 | 306.8 | 637.4 |
| **Participaciones por Est.** | 0.98 | 0.07 | 1.86 |
| **Tasa de Missing** | 24.3% | 0.0% | 32.9% |

### 3.4 Estadísticas de Calificaciones

Para los 71 cursos con ≥15 estudiantes con notas:

| Métrica | Valor |
|---------|-------|
| **Nota Promedio General** | 88.4% |
| **Tasa de Reprobación Promedio** | 12.7% |
| **Cursos con >20% reprobación** | 15 |

---

## 4. Hallazgos Principales

### 4.1 Correlación Entre Análisis

La correlación entre el score de diseño LMS y el score de actividad es **r = 0.865**, lo que indica que ambas perspectivas capturan aspectos **complementarios pero relacionados** del potencial predictivo.

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

## 5. Análisis por Campus

### 5.1 Comparación de Métricas

![Comparación por Campus](viz_campus_comparison.png)

| Campus | Cursos | PageViews Prom. | Missing Rate | Score Diseño | Score Actividad | **Score Combinado** |
|--------|--------|-----------------|--------------|--------------|-----------------|---------------------|
| **Providencia** | 160 | 551 | 18.2% | 7.1 | 26.7 | 16.9 |
| **San Miguel** | 141 | 341 | 26.6% | 9.2 | 31.4 | 20.3 |
| **Temuco** | 140 | 500 | 28.8% | 7.7 | 30.7 | 19.2 |

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

## 6. Patrones de Engagement

### 6.1 Segmentación por Page Views

![Distribución de Engagement](viz_engagement_distribution.png)

| Nivel de Engagement | Page Views | Cursos | % |
|---------------------|------------|--------|---|
| 🔴 **Muy Bajo** | < 100 | 85 | 18.4% |
| 🟠 **Bajo** | 100 - 300 | 141 | 30.5% |
| 🟡 **Medio** | 300 - 600 | 135 | 29.2% |
| 🟢 **Alto** | 600 - 1000 | 57 | 12.3% |
| 🔵 **Muy Alto** | > 1000 | 44 | 9.5% |

### 6.2 Engagement vs Resultados Académicos

Para cursos con datos de notas (n=71):

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

## 7. Análisis de Riesgo

### 7.1 Distribución de Niveles de Riesgo

Se evaluaron tres indicadores de riesgo:
1. **Alta tasa de missing** (>70% de tareas no entregadas)
2. **Bajo engagement** (<100 page views promedio)
3. **Baja participación** (<0.5 participaciones promedio)

| Nivel de Riesgo | Indicadores | Cursos | % |
|-----------------|-------------|--------|---|
| ✅ **Sin Riesgo** | 0 | 118 | 25.5% |
| ⚠️ **Riesgo Bajo** | 1 | 234 | 50.6% |
| 🟠 **Riesgo Medio** | 2 | 108 | 23.4% |
| 🔴 **Riesgo Alto** | 3 | 2 | 0.4% |

### 7.2 Cursos de Mayor Riesgo

| Curso | PageViews | Missing | Indicadores |
|-------|-----------|---------|-------------|
| OPTATIVO DE ESPECIALIDAD III-P03 | 13 | 0% | 2 |
| OPTATIVO DE ESPECIALIDAD IV-P03 | 4 | 0% | 2 |
| SEMANA SMART-T02 | 8 | 0% | 2 |
| ESTRUCTURA DE MADERA-T02 | 11 | 0% | 2 |
| PSICOLOGÍA JURÍDICA-P07 | 17 | 0% | 2 |
| OPTATIVO DE ESPECIALIDAD IV-S03 | 14 | 0% | 2 |
| OPTATIVO DE ESPECIALIDAD IV-S01 | 10 | 0% | 2 |
| INTERV PSICOED CONTEX ESC A S-S01 | 16 | 0% | 2 |
| INTERVENCIÓN EN ORGANIZACIONES-S11 | 15 | 0% | 2 |
| INTERV PSICOED CONTEX ESC A S-P11 | 15 | 0% | 2 |

### 7.3 Recomendaciones de Intervención

Para los **110 cursos** con riesgo medio-alto:

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

## 8. Top 50 Cursos para Modelado Predictivo

### 8.1 Ranking Combinado (Diseño LMS + Actividad)

![Top 50 Cursos](viz_top_50_courses.png)

Los siguientes cursos representan los **mejores candidatos** para implementar sistemas de alerta temprana, basados en la combinación de:
- Calidad del diseño instruccional
- Riqueza de datos de actividad
- Balance de clases para modelado

| # | Curso | Score Diseño | Score Actividad | **Score Combinado** | Estudiantes |
|---|-------|--------------|-----------------|---------------------|-------------|
| 1 | TALL DE COMPETENCIAS DIGITALES-S04 | 100.0 | 95.3 | **97.7** | 26 |
| 2 | TALL DE COMPETENCIAS DIGITALES-T06 | 100.0 | 94.9 | **97.5** | 25 |
| 3 | MATEMÁTICAS PARA LA GESTIÓN II-P03 | 98.4 | 95.8 | **97.1** | 26 |
| 4 | TALL DE COMPETENCIAS DIGITALES-T07 | 98.5 | 95.1 | **96.8** | 29 |
| 5 | ÁLGEBRA Y GEOMETRÍA-S04 | 93.8 | 95.7 | **94.8** | 18 |
| 6 | ÁLGEBRA Y GEOMETRÍA-S01 | 94.2 | 94.2 | **94.2** | 29 |
| 7 | DIBUJO Y MODELADO DIGITAL-S03 | 95.6 | 92.4 | **94.0** | 24 |
| 8 | TEORÍAS PSICOLÓGICAS IV-S02 | 95.7 | 89.8 | **92.8** | 39 |
| 9 | TALLER DE HABILID. PROF. II-P02 | 90.0 | 92.9 | **91.5** | 32 |
| 10 | TRAB. SOCIAL CON COLECTIVOS II-T04 | 87.8 | 87.8 | **87.8** | 21 |
| 11 | FUNDAMENTOS DE MATEMÁTICAS-T04 | 85.4 | 90.1 | **87.8** | 35 |
| 12 | DIRECCIÓN DE ARTE-P01 | 83.5 | 88.2 | **85.8** | 20 |
| 13 | DIBUJO Y MODELADO DIGITAL-S01 | 79.3 | 89.0 | **84.2** | 28 |
| 14 | CLÍN INTEG DEL NIÑO Y ADOLES I-P01 | 86.1 | 81.6 | **83.8** | 51 |
| 15 | TEORÍAS PSICOLÓGICAS IV-S01 | 81.7 | 85.7 | **83.7** | 40 |
| 16 | AMBIENTE CONSTRUIDO-T02 | 87.9 | 78.2 | **83.1** | 30 |
| 17 | FUND. ANTROPOLÓGICOS INTERV.-S02 | 77.6 | 87.6 | **82.6** | 30 |
| 18 | PRECLÍNICO BIOMAT DENTALES I-P06 | 84.7 | 78.3 | **81.5** | 36 |
| 19 | PSICOPAT Y PSICOF INF JUVENIL-T01 | 71.4 | 89.9 | **80.7** | 35 |
| 20 | MATEMÁTICAS PARA LA GESTIÓN II-S01 | 75.1 | 83.4 | **79.2** | 31 |
| 21 | EPIDEMIOLOGÍA Y SALUD-T01 | 72.3 | 83.5 | **77.9** | 46 |
| 22 | MATEMÁTICAS PARA LA GESTIÓN II-P01 | 71.8 | 81.9 | **76.8** | 31 |
| 23 | FUND. ANTROPOLÓGICOS INTERV.-S01 | 69.3 | 83.8 | **76.5** | 35 |
| 24 | ÁLGEBRA Y GEOMETRÍA-T01 | 70.3 | 82.1 | **76.2** | 28 |
| 25 | ANATOMÍA DE CUELLO Y CABEZA-P01 | 69.3 | 81.8 | **75.5** | 35 |
| 26 | OPTATIVO DE ESPECIALIDAD II-T03 | 70.3 | 80.4 | **75.3** | 15 |
| 27 | CARIOLOG Y ODONTO PREVEN (A S)-T01 | 67.9 | 82.3 | **75.1** | 24 |
| 28 | INTERVENCIÓN CLÍNICA ADULTO-P06 | 63.7 | 83.3 | **73.5** | 54 |
| 29 | DIRECCIÓN DE ARTE-P03 | 66.1 | 80.2 | **73.2** | 17 |
| 30 | PSIC DEL DES II: ADOLES Y ADUL-S07 | 72.6 | 73.3 | **72.9** | 45 |
| 31 | DIRECCIÓN DE ARTE-P02 | 68.6 | 77.0 | **72.8** | 24 |
| 32 | INTERV PSICOED CONTEX ESC A S-S06 | 70.9 | 73.6 | **72.2** | 48 |
| 33 | PRECLÍNICO BIOMAT DENTALES I-P01 | 70.0 | 73.6 | **71.8** | 32 |
| 34 | PSIC DEL DES II: ADOLES Y ADUL-P01 | 66.6 | 72.1 | **69.3** | 50 |
| 35 | FUNDAMENTOS DE MATEMÁTICAS-T01 | 65.5 | 70.0 | **67.8** | 22 |
| 36 | MÓD DIAG II: CIR ORAL Y PERIOD-P01 | 57.5 | 76.4 | **67.0** | 48 |
| 37 | CLÍN INTEG DEL NIÑO Y ADOLES I-T01 | 53.1 | 78.5 | **65.8** | 57 |
| 38 | TRAB. SOCIAL E INTERV. SOCIAL-S01 | 51.8 | 74.9 | **63.4** | 32 |
| 39 | ADMINISTRACIÓN PÚBLICA CHILENA-P02 | 56.9 | 61.6 | **59.2** | 33 |
| 40 | MET. Y ANÁLISIS DE DATOS CUAL.-S01 | 46.7 | 58.5 | **52.6** | 39 |
| 41 | BIOFÍSICA-P01 | 20.0 | 81.4 | **50.7** | 36 |
| 42 | HISTOLOGÍA Y EMBRIOLOGÍA GRAL.-T01 | 19.7 | 79.7 | **49.7** | 45 |
| 43 | DESEMPEÑO ÉTICO DEL PSICÓLOGO-S07 | 20.0 | 78.0 | **49.0** | 48 |
| 44 | BIOFÍSICA-T04 | 18.6 | 77.9 | **48.2** | 29 |
| 45 | METOD. DE LA INVEST.-P01 | 20.0 | 75.6 | **47.8** | 29 |
| 46 | OPTATIVO DE ESPECIALIDAD I-S02 | 20.0 | 75.3 | **47.6** | 15 |
| 47 | PSICOLOGÍA JURÍDICA-S04 | 20.0 | 73.8 | **46.9** | 48 |
| 48 | FUNDAMENTOS DE MATEMÁTICAS-T07 | 18.7 | 74.8 | **46.8** | 22 |
| 49 | BIOFÍSICA-P03 | 20.0 | 73.3 | **46.6** | 15 |
| 50 | BIOFÍSICA-T01 | 18.4 | 74.6 | **46.5** | 35 |

### 8.2 Perfil de los Top 50

| Métrica | Promedio Top 50 | Promedio General |
|---------|-----------------|------------------|
| Score de Diseño | 65.3 | 8.1 |
| Score de Actividad | 81.6 | 30.1 |
| Score Combinado | 73.4 | 18.8 |
| Estudiantes | 33 | 34 |

### 8.3 Distribución por Tipo de Curso

Los tipos de curso más representados en el Top 50:

| Categoría | Cantidad | Ejemplos |
|-----------|----------|----------|
| **Matemáticas/Álgebra** | 9 | Álgebra y Geometría, Matemáticas para la Gestión |
| **Competencias Digitales** | 3 | Taller de Competencias Digitales |
| **Psicología** | 3 | Teorías Psicológicas, Psicopatología |
| **Talleres** | 4 | Taller de Habilidades, Taller de Pensamiento |

---

## 9. Conclusiones y Recomendaciones

### 9.1 Conclusiones Principales

1. **Disponibilidad de Datos**
   - De 623 cursos analizados, solo el **11.8% tiene datos de notas suficientes** para modelado supervisado
   - El **76.9% tiene datos de actividad suficientes** para predicción basada en engagement
   - Existe oportunidad significativa de expandir la recolección de notas

2. **Potencial Predictivo**
   - **41 cursos** (6.6%) tienen alto potencial para modelado predictivo
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


## 10. Apéndice Técnico

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
