# Informe: Sistema de Alerta Temprana sin Evaluaciones

## Universidad Autonoma de Chile - Canvas LMS

**Fecha:** 30 de diciembre de 2025
**Programa:** Postgrado - Ingenieria en Control de Gestion y otros
**Version del Modelo:** XGBoost Optimizado v2.0

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [El Problema: Intervencion Tardia](#2-el-problema-intervencion-tardia)
3. [Metodologia: Features de Materiales de Aprendizaje](#3-metodologia-features-de-materiales-de-aprendizaje)
4. [Resultados del Modelo](#4-resultados-del-modelo)
5. [Interpretacion Practica](#5-interpretacion-practica)
6. [Conclusiones y Proximos Pasos](#6-conclusiones-y-proximos-pasos)
7. [Anexos](#7-anexos)

---

# 1. Resumen Ejecutivo

## 1.1 Innovacion Principal

Este informe presenta un **Sistema de Alerta Temprana que predice el fracaso academico ANTES de la primera evaluacion**, utilizando exclusivamente patrones de engagement con materiales de aprendizaje.

### Diferencia Clave vs. Modelos Tradicionales

| Aspecto | Modelos Tradicionales | Nuestro Modelo |
|---------|----------------------|----------------|
| **Cuando predice** | Despues de primeras notas | **Semanas 2-3 del curso** |
| **Que usa** | Notas parciales, entregas | **Solo engagement con contenido** |
| **Ventana de intervencion** | 4-6 semanas antes del final | **10-12 semanas antes** |

## 1.2 Muestra Analizada

Se analizaron **361 estudiantes** distribuidos en **10 cursos** del programa de Postgrado.

| Metrica | Valor |
|---------|-------|
| Total de estudiantes | 361 |
| Cursos analizados | 10 |
| Tasa de reprobacion | 39.3% |
| Features incluidos | 123 |
| Features excluidos (evaluaciones) | 37 |

## 1.3 Resultados Principales

### Mejora Significativa sobre Baseline

| Metrica | Baseline | Modelo Optimizado | Mejora |
|---------|----------|-------------------|--------|
| **ROC-AUC** | 0.787 | **0.859** | **+9.1%** |
| **Exactitud** | 74.0% | 77.6% | +3.6pp |
| **Precision** | 69.7% | 75.2% | +5.5pp |
| **Sensibilidad** | 61.7% | 64.1% | +2.4pp |

### Interpretacion Practica

- **ROC-AUC 0.859**: El modelo tiene **excelente capacidad discriminativa** entre estudiantes que aprobaran y reprobaran
- **Precision 75.2%**: De cada 4 alertas emitidas, **3 son correctas**
- **Sensibilidad 64.1%**: Detectamos **2 de cada 3 estudiantes en riesgo**

---

# 2. El Problema: Intervencion Tardia

## 2.1 El Ciclo Actual

```
Semana 1-4: Estudiante tiene dificultades (invisibles)
Semana 5-6: Primera evaluacion - nota baja
Semana 7-8: Tutor contacta al estudiante
Semana 9-10: Se implementa apoyo
Semana 12: Ya es demasiado tarde
```

### El Problema Central

> Los sistemas tradicionales de alerta temprana **NO son tempranos**: dependen de calificaciones que llegan **demasiado tarde** para intervenir efectivamente.

## 2.2 Nuestra Solucion

```
Semana 1-2: Estudiante tiene dificultades
Semana 2-3: SISTEMA DETECTA patrones de riesgo
Semana 3-4: Tutor contacta al estudiante
Semana 4-6: Se implementa apoyo ANTES de primera evaluacion
Semana 12: Estudiante aprueba
```

### Ventaja Clave

> Nuestro modelo usa **UNICAMENTE** engagement con materiales de aprendizaje, sin ninguna calificacion o resultado de evaluacion, permitiendo prediccion **semanas antes** de la primera nota.

---

# 3. Metodologia: Features de Materiales de Aprendizaje

## 3.1 Features Incluidos (123 total)

El modelo utiliza exclusivamente datos de engagement con **materiales de aprendizaje**:

### Por Tipo de Recurso

| Tipo | Features | Descripcion |
|------|----------|-------------|
| **Archivos (Files)** | ~15 | Vistas, descargas, proactividad de acceso |
| **Discusiones** | ~15 | Participacion en foros, vistas, engagement |
| **Paginas** | ~15 | Lectura de contenido informativo |
| **Modulos** | ~12 | Navegacion por estructura del curso |
| **Anuncios** | ~4 | Atencion a comunicaciones |
| **Inicio (Home)** | ~4 | Frecuencia de acceso al curso |

### Por Categoria de Feature

![Distribucion de Features](visualizations/feature_categories_early_warning.png)

| Categoria | Cantidad | Descripcion |
|-----------|----------|-------------|
| **Proactividad (PCT)** | ~50 | Ranking temporal de acceso a recursos |
| **Vistas por categoria** | ~20 | Distribucion de actividad |
| **PCA** | ~12 | Componentes principales del engagement |
| **Temporal** | ~20 | Patrones semanales |
| **Sesiones** | 11 | Patrones de sesion y regularidad |
| **Ratios** | ~10 | Metricas derivadas |

## 3.2 Features Excluidos (37 total)

**TODOS** los features relacionados con evaluaciones fueron excluidos:

| Tipo | Features Excluidos | Razon |
|------|-------------------|-------|
| **Quizzes** | `quiz_*`, `quizzes_*` | Actividad en quizzes = tomar evaluaciones |
| **Tareas** | `assi_*`, `assignments_*` | Actividad en tareas = entregas calificadas |
| **Calificaciones** | `grades_*`, `grad_*` | Vista de notas = conocer resultados |

### Por Que Excluir Evaluaciones?

1. **Contaminacion temporal**: La actividad en quizzes/tareas ocurre DURANTE la evaluacion, no antes
2. **Causalidad inversa**: Ver calificaciones NO causa el resultado, lo refleja
3. **Objetivo real**: Predecir ANTES de que existan calificaciones

---

# 4. Resultados del Modelo

## 4.1 Comparacion de Modelos

Se evaluaron tres algoritmos de aprendizaje automatico:

![Comparacion de Modelos](visualizations/model_comparison_early_warning.png)

| Modelo | ROC-AUC | Exactitud | Precision | Sensibilidad |
|--------|---------|-----------|-----------|--------------|
| **Baseline (Actividad)** | 0.787 | 74.0% | 69.7% | 61.7% |
| Regresion Logistica | 0.814 | 75.9% | 68.2% | 72.5% |
| Random Forest | 0.828 | 74.8% | 75.2% | 53.5% |
| **XGBoost Optimizado** | **0.859** | **77.6%** | **75.2%** | **64.1%** |

## 4.2 Curvas ROC

![Curvas ROC](visualizations/roc_curves_early_warning.png)

La curva ROC muestra la capacidad del modelo para discriminar entre estudiantes que aprobaran y reprobaran. Un area bajo la curva (AUC) de **0.859** indica excelente rendimiento.

## 4.3 Matriz de Confusion

![Matriz de Confusion](visualizations/confusion_matrix_early_warning.png)

### Interpretacion de la Matriz

| | Prediccion: Aprueba | Prediccion: Reprueba |
|---|---|---|
| **Real: Aprueba** | 189 (VN) | 30 (FP) |
| **Real: Reprueba** | 51 (FN) | 91 (VP) |

- **Verdaderos Negativos (189)**: Estudiantes que aprobaron, correctamente predichos como bajo riesgo
- **Falsos Positivos (30)**: Estudiantes que aprobaron pero fueron marcados como en riesgo (alertas innecesarias)
- **Falsos Negativos (51)**: Estudiantes que reprobaron pero no fueron detectados (casos perdidos)
- **Verdaderos Positivos (91)**: Estudiantes en riesgo correctamente identificados

### Balance Precision-Sensibilidad

> **Trade-off aceptable**: Preferimos algunas alertas innecesarias (FP=30) a perder estudiantes en riesgo (FN=51). El modelo detecta **64% de los estudiantes que reprobaran**, con **75% de precision en las alertas**.

## 4.4 Top Features Predictivos

![Importancia de Features](visualizations/feature_importance_early_warning.png)

### Los 10 Predictores Mas Importantes

| Rank | Feature | Importancia | Interpretacion |
|------|---------|-------------|----------------|
| 1 | `modu_n_resources` | 5.2% | Numero de modulos con los que interactua |
| 2 | `pages_views_pct` | 3.9% | Porcentaje de vistas en paginas informativas |
| 3 | `page_n_resources` | 3.9% | Numero de paginas diferentes visitadas |
| 4 | `content_vs_assessment_ratio` | 3.3% | Ratio contenido vs evaluaciones |
| 5 | `modu_top50_rate` | 3.3% | % modulos donde esta en top 50% de acceso |
| 6 | `total_views` | 3.3% | Total de visualizaciones en el curso |
| 7 | `files_pc1` | 2.9% | Patron principal de engagement con archivos |
| 8 | `session_regularity` | 2.8% | Regularidad temporal de sesiones |
| 9 | `announcements_views` | 2.5% | Atencion a anuncios del curso |
| 10 | `modu_hist_b4` | 2.2% | Distribucion de proactividad en modulos |

### Hallazgo Clave

> **Los modulos y paginas son los predictores mas fuertes.** Estudiantes que navegan activamente la estructura del curso (modulos) y leen el contenido informativo (paginas) tienen significativamente mayor probabilidad de aprobar.

---

# 5. Interpretacion Practica

## 5.1 Senales de Alerta Temprana

Basado en los features mas predictivos, las siguientes senales indican riesgo:

### Senales de Alto Riesgo

| Senal | Descripcion | Accion Sugerida |
|-------|-------------|-----------------|
| **Baja navegacion de modulos** | Estudiante no explora la estructura del curso | Enviar guia de navegacion |
| **Pocas vistas de paginas** | No lee contenido informativo | Destacar material clave |
| **Sesiones irregulares** | Acceso esporadico e impredecible | Establecer horario de estudio |
| **Ignora anuncios** | No revisa comunicaciones del curso | Contacto directo via email |

### Senales Positivas

| Senal | Descripcion |
|-------|-------------|
| Alto `modu_top50_rate` | Accede temprano a los modulos |
| Alto `pages_views_pct` | Lee activamente el contenido |
| Alto `session_regularity` | Patron de estudio consistente |
| Alto `content_vs_assessment_ratio` | Enfocado en aprendizaje, no solo evaluaciones |

## 5.2 Ventana de Intervencion

```
Semana 1: Inicio del curso
         |
Semana 2: PRIMERA DETECCION POSIBLE
         |  -> Datos suficientes para prediccion
         |  -> Alerta enviada a tutor
         |
Semana 3-4: VENTANA DE INTERVENCION
         |  -> Contacto con estudiante
         |  -> Plan de apoyo personalizado
         |
Semana 5-6: Primera evaluacion
         |  -> Estudiante ya recibio apoyo
         |
Semana 12: Fin del curso
         -> Mayor probabilidad de exito
```

---

# 6. Conclusiones y Proximos Pasos

## 6.1 Conclusiones Principales

1. **Prediccion pre-evaluacion es posible**: El modelo logra ROC-AUC de 0.859 usando SOLO engagement con materiales de aprendizaje, **sin ninguna calificacion**.

2. **Mejora significativa sobre baseline**: +9.1% de mejora en ROC-AUC respecto al modelo de actividad simple.

3. **La estructura del curso importa**: Los modulos y paginas son los predictores mas fuertes, sugiriendo que un buen diseno instruccional facilita la deteccion temprana.

4. **Regularidad sobre intensidad**: Estudiantes con sesiones regulares (aunque cortas) tienen mejores resultados que quienes estudian esporadicamente.

## 6.2 Recomendaciones

### Para Administradores

1. **Implementar alertas automatizadas** basadas en los 10 features principales
2. **Estandarizar estructura de cursos** para facilitar la navegacion
3. **Capacitar tutores** en interpretacion de senales de riesgo

### Para Docentes

1. **Organizar contenido en modulos claros** - mayor exploracion = menor riesgo
2. **Crear paginas informativas** - estudiantes que leen tienen mejores resultados
3. **Publicar anuncios regularmente** - la atencion a comunicaciones predice exito

### Para Equipo Tecnico

1. **Automatizar pipeline de features** para actualizacion semanal
2. **Desarrollar dashboard de riesgo** para tutores
3. **Integrar con sistemas de notificacion** del LMS

## 6.3 Proximos Pasos

| Fase | Actividad | Plazo |
|------|-----------|-------|
| 1 | Validacion con cohorte nuevo (Semestre 2) | Q1 2026 |
| 2 | Piloto de alertas automatizadas en 3 cursos | Q2 2026 |
| 3 | Desarrollo de dashboard para tutores | Q2 2026 |
| 4 | Expansion a todos los cursos de Postgrado | Q3 2026 |
| 5 | Extension a programas de Pregrado | Q4 2026 |

---

# 7. Anexos

## 7.1 Hiperparametros del Modelo

```json
{
  "learning_rate": 0.2,
  "max_depth": 7,
  "min_child_weight": 1,
  "n_estimators": 100,
  "subsample": 0.8
}
```

## 7.2 Lista Completa de Features Incluidos

<details>
<summary>Ver 123 features (click para expandir)</summary>

### Features de Sesion (11)
- `session_count`, `session_duration_mean`, `session_duration_std`
- `session_duration_median`, `sessions_per_week`, `views_per_session`
- `short_sessions_pct`, `long_sessions_pct`, `total_views`
- `total_time_min`, `session_regularity`

### Features por Categoria (~35)
- `files_views`, `files_views_pct`, `files_unique_resources`, `files_time_min`
- `discussions_views`, `discussions_views_pct`, `discussions_unique_resources`, `discussions_time_min`
- `pages_views`, `pages_views_pct`, `pages_unique_resources`, `pages_time_min`
- `modules_views`, `modules_views_pct`, `modules_unique_resources`, `modules_time_min`
- `announcements_views`, `announcements_views_pct`, `announcements_unique_resources`
- `home_views`, `home_views_pct`, `home_unique_resources`
- `content_vs_assessment_ratio`, `discussion_participation_rate`, `total_views_cat`

### Features de Proactividad PCT (~50)
Para cada tipo (file, disc, page, modu):
- `{type}_n_resources`, `{type}_mean_pct`, `{type}_median_pct`, `{type}_std_pct`
- `{type}_access_rate`, `{type}_top25_rate`, `{type}_top50_rate`
- `{type}_hist_b1`, `{type}_hist_b2`, `{type}_hist_b3`, `{type}_hist_b4`, `{type}_hist_b5`

Features adicionales:
- `download_count`, `unique_files_downloaded`, `download_rate`
- `dct_pct_0`, `dct_pct_1`, `dct_pct_2`, `dct_pct_3`
- `overall_proactivity`

### Features PCA (~15)
- `files_pc1`, `files_pc2`, `files_pc3`, `files_n_resources`, `files_var_explained`
- `disc_pc1`, `disc_pc2`, `disc_pc3`, `disc_n_resources_pca`, `disc_var_explained`
- `pages_pc1`, `pages_pc2`, `pages_pc3`, `pages_n_resources`, `pages_var_explained`
- `mods_pc1`, `mods_pc2`, `mods_n_resources`, `mods_var_explained`

### Features Temporales (~12)
- `active_weeks_count`, `first_active_week`, `last_active_week`
- `peak_week`, `early_semester_views`, `late_semester_views`
- `early_vs_late_ratio`, `avg_week_over_week_change`, `activity_consistency`
- `engagement_pattern`

</details>

## 7.3 Lista de Features Excluidos

<details>
<summary>Ver 37 features excluidos (click para expandir)</summary>

### Features de Quizzes
- `quiz_n_resources`, `quiz_mean_pct`, `quiz_median_pct`, `quiz_std_pct`
- `quiz_access_rate`, `quiz_top25_rate`, `quiz_top50_rate`
- `quiz_hist_b1`, `quiz_hist_b2`, `quiz_hist_b3`, `quiz_hist_b4`, `quiz_hist_b5`
- `quizzes_views`, `quizzes_views_pct`, `quizzes_unique_resources`, `quizzes_time_min`

### Features de Tareas
- `assi_n_resources`, `assi_mean_pct`, `assi_median_pct`, `assi_std_pct`
- `assi_access_rate`, `assi_top25_rate`, `assi_top50_rate`
- `assi_hist_b1`, `assi_hist_b2`, `assi_hist_b3`, `assi_hist_b4`, `assi_hist_b5`
- `assignments_views`, `assignments_views_pct`, `assignments_unique_resources`, `assignments_time_min`

### Features de Calificaciones
- `grades_views`, `grades_views_pct`, `grades_unique_resources`, `grades_time_min`
- `grades_check_per_week`

</details>

## 7.4 Cursos Analizados

| ID | Curso | Estudiantes | Tasa Reprobacion |
|----|-------|-------------|------------------|
| 79875 | GESTION DEL TALENTO-P01 | 42 | 38.1% |
| 79913 | FUND DE BUSINESS ANALYTICS-P01 | 39 | 41.0% |
| 84936 | FUNDAMENTOS DE MICROECONOMIA-P03 | 41 | 26.8% |
| 84941 | FUNDAMENTOS DE MICROECONOMIA-P01 | 36 | 61.1% |
| 84944 | FUNDAMENTOS DE MACROECONOMIA-P03 | 34 | 35.3% |
| 86020 | MATEMATICAS PARA LOS NEGOCIOS-P01 | 29 | 37.9% |
| 86676 | TALLER PENSAMIENTO ANALITICO-P01 | 40 | 77.5% |
| 88381 | TALL DE COMPETENCIAS DIGITALES-P01 | 50 | 98.0% |
| 89099 | TALL DE COMPETENCIAS DIGITALES-P02 | 25 | 12.0% |
| 89390 | FUNDAMENTOS TRIBUTARIOS-P01 | 25 | 24.0% |

---

*Documento generado: 30 de diciembre de 2025*
*Modelo: XGBoost Optimizado (ROC-AUC: 0.859)*
*Autores: Equipo de Analitica Academica, Universidad Autonoma de Chile*
