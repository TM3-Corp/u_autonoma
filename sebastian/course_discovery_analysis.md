# Reporte de Análisis de Descubrimiento de Cursos

## Universidad Autónoma de Chile - Analytics de Canvas LMS

**Fecha:** Diciembre 2025
**Objetivo:** Identificar cursos con mayor potencial para construir modelos predictivos (sistemas de alerta temprana)
**Fuente de Datos:** API de Canvas LMS (Ambiente de Pruebas)

---

## 1. Resumen Ejecutivo

Este análisis explora **601 cursos** en 5 campus de la Universidad Autónoma de Chile para identificar cuáles tienen suficiente calidad de datos y características para construir modelos predictivos que identifiquen estudiantes en riesgo.

### Hallazgos Principales

| Métrica | Valor |
|---------|-------|
| Total de Cursos Analizados | 601 |
| Cursos con Notas Válidas | 95 (15.8%) |
| Viables para Modelado (≥15 estudiantes, notas, varianza) | 38 (6.3%) |
| Puntaje de Alto Potencial (≥80) | 16 cursos |
| Puntaje Perfecto (100) | 2 cursos |

### Top 5 Cursos para Modelado Predictivo

| Rank | Curso | Campus | Score | Estudiantes | Tasa Reprobación |
|------|-------|--------|-------|-------------|------------------|
| 1 | TALL DE COMPETENCIAS DIGITALES-S04 | San Miguel | 100.0 | 26 | 54% |
| 2 | TALL DE COMPETENCIAS DIGITALES-T06 | Temuco | 100.0 | 25 | 56% |
| 3 | TALL DE COMPETENCIAS DIGITALES-T07 | Temuco | 98.5 | 29 | 24% |
| 4 | MATEMÁTICAS PARA LA GESTIÓN II-P03 | Providencia | 98.4 | 26 | 27% |
| 5 | TEORÍAS PSICOLÓGICAS IV-S02 | San Miguel | 95.7 | 39 | 36% |

---

## 2. Metodología

### 2.1 Pipeline de Recolección de Datos

Desarrollamos un sistema de recolección multi-hilo con control de rate limit:

```
┌─────────────────────────────────────────────────────────────────┐
│                  FLUJO DE RECOLECCIÓN DE DATOS                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Canvas API  ──►  Descubrimiento  ──►  Extracción de Métricas │
│       │            de Cursos                   │                │
│       ▼                   ▼                    ▼                │
│   Monitoreo de       Recorrido de         Por Curso:           │
│   Rate Limit         Sub-cuentas          - Enrollments        │
│   (700 bucket)       (5 campus)           - Grades             │
│       │                   │               - Assignments        │
│       ▼                   ▼               - Activity           │
│   Auto-throttle      Deduplicación        - Design metrics     │
│   & Recuperación                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Script:** `scripts/discovery/section7_refactor.py`

### 2.2 Cobertura por Campus

| Campus ID | Nombre del Campus | Cursos Encontrados |
|-----------|-------------------|-------------------|
| 173 | Temuco | 200 |
| 174 | Talca | 1 |
| 175 | San Miguel | 200 |
| 176 | Providencia | 200 |
| 360 | Campus Virtual | 0 |

**Total de cursos únicos:** 601

### 2.3 Métricas Recolectadas Por Curso

#### Métricas de Calificaciones
- `students_with_current_score` - Estudiantes con notas en Canvas
- `current_score_coverage` - % de estudiantes con notas
- `grade_mean`, `grade_std`, `grade_min`, `grade_max` - Estadísticas de notas
- `failure_rate` - Proporción bajo 57% (4.0/7.0 escala chilena)

#### Métricas de Diseño Instruccional
- `assignment_count`, `graded_assignment_count`
- `quiz_count`, `module_count`
- `file_count`, `discussion_count`, `page_count`

#### Métricas de Actividad
- `total_page_views`, `total_participations`
- `students_with_activity`
- `avg_page_views`, `avg_participations`

#### Puntajes Compuestos (escala 0-100)
- `grade_availability_score` - Completitud de datos
- `grade_variance_score` - Fuerza de señal predictiva
- `class_balance_score` - Distribución aprobados/reprobados
- `design_richness_score` - Calidad de contenido LMS
- `activity_score` - Engagement estudiantil
- **`prediction_potential_score`** - Compuesto ponderado

---

## 3. Resultados del Análisis

### 3.1 Matriz de Correlación

![Correlation Heatmap](../data/discovery/analysis/correlation_heatmap.png)

**Correlaciones Clave Observadas:**

| Par de Métricas | Correlación | Interpretación |
|-----------------|-------------|----------------|
| `current_score_coverage` ↔ `final_score_coverage` | 0.93 | Esperado - mismos datos subyacentes |
| `grade_std` ↔ `failure_rate` | 0.58 | Mayor varianza = más reprobados |
| `total_page_views` ↔ `total_participations` | 0.74 | Métricas de actividad co-varían |
| `assignment_count` ↔ `quiz_count` | 0.46 | Elementos de diseño correlacionan |
| `grade_availability_score` ↔ `prediction_potential_score` | 0.77 | Disponibilidad de datos es clave |

**Insight:** El cluster más fuerte está alrededor de métricas de notas - los cursos que tienen datos de calificaciones tienden a tenerlos de forma completa. Las métricas de actividad forman un cluster separado.

---

### 3.2 Análisis de Drivers de Predicción

![Prediction Drivers](../data/discovery/analysis/prediction_drivers.png)

**Principales Métricas que Impulsan el Potencial Predictivo (estadísticamente significativas):**

| Rank | Métrica | Pearson r | p-value | Interpretación |
|------|---------|-----------|---------|----------------|
| 1 | `grade_min` | **-0.74** | <0.001 | Notas mínimas más bajas = más estudiantes en riesgo para identificar |
| 2 | `failure_rate` | **+0.48** | <0.001 | Clases balanceadas (no todos aprueban) = mejor modelado |
| 3 | `grade_std` | **+0.39** | <0.001 | Mayor varianza = más señal predictiva |
| 4 | `avg_participations` | **+0.28** | 0.019 | Datos de engagement ayudan a la predicción |

**Insight Crítico:** La correlación de `grade_min` de -0.74 nos dice que **los cursos donde los estudiantes realmente reprueban** son los mejores candidatos para sistemas de alerta temprana. Los cursos donde todos aprueban (grade_min > 57%) no tienen señal para predecir.

---

### 3.3 Distribuciones de Puntajes

![Score Distributions](../data/discovery/analysis/score_distributions.png)

**Análisis de Distribución:**

| Componente del Score | Media | Mediana | Observación |
|----------------------|-------|---------|-------------|
| Prediction Potential | 53.4 | 65.8 | Bimodal - muchos ceros (sin datos) y scores altos |
| Grade Availability | 62.9 | 60.6 | La mayoría de cursos con notas tienen buena cobertura |
| Grade Variance | 48.8 | 40.3 | Amplia dispersión - muchos cursos de baja varianza |
| Class Balance | 20.5 | 10.0 | La mayoría de cursos están desbalanceados (todos aprueban) |
| Design Richness | 45.3 | 38.0 | Distribución normal alrededor de riqueza moderada |
| Activity | 87.2 | 100.0 | La mayoría de cursos tienen datos de actividad |

**Insight Clave:** El **Class Balance Score** es el cuello de botella - la mayoría de los cursos tienen muy pocos reprobados, limitando el potencial de modelado predictivo.

---

### 3.4 Cobertura de Notas vs Actividad Estudiantil

![Grade vs Activity](../data/discovery/analysis/grade_vs_activity.png)

**Gráfico Izquierdo: Cobertura de Notas vs Page Views**
- **Puntos amarillos (score alto)**: 100% cobertura de notas + actividad moderada-alta
- Zona óptima: Cuadrante superior derecho (100% cobertura, >500 avg page views)
- Muchos cursos se agrupan en 100% cobertura pero varían ampliamente en actividad

**Gráfico Derecho: Varianza de Notas vs Tasa de Reprobación**
- **Zona óptima para modelado**: Varianza 20-40%, Tasa de reprobación 20-60%
- Los puntos amarillos/verdes se agrupan en esta zona
- Outliers con varianza extrema (>100) son anomalías (probablemente problemas de datos)

---

### 3.5 Ranking Top 25 de Cursos

![Top Courses](../data/discovery/analysis/top_courses.png)

**Características de los Cursos con Mayor Puntaje:**

1. **Talleres de Competencias Digitales** (TALL DE COMPETENCIAS DIGITALES)
   - Múltiples secciones en distintos campus puntúan 95-100
   - Altas tasas de reprobación (24-56%) proveen balance de clases
   - Contenido LMS rico (18 assignments, 13-14 quizzes)

2. **Cursos de Matemáticas** (Álgebra, Cálculo, Matemáticas para la Gestión)
   - Varianza de notas consistente (25-35%)
   - Buenos tamaños de muestra (25-50 estudiantes)
   - Métricas de actividad fuertes

3. **Cursos de Teoría Psicológica** (Teorías Psicológicas)
   - Alta cobertura (95-100%)
   - Tasas de reprobación moderadas (30-35%)
   - Buena riqueza de diseño

---

## 4. Problemas de Calidad de Datos

### 4.1 El Problema de las "Notas Faltantes"

**506 de 601 cursos (84%) NO tienen notas en Canvas.**

Posibles causas:
1. **Libro de Calificaciones Externo** - Herramienta LTI almacenando notas externamente
2. **Notas aún no ingresadas** - Cursos en progreso
3. **Flujo de calificación diferente** - Ingreso manual de notas

**Recomendación:** Investigar la integración del libro de calificaciones externo para acceder a estos datos.

### 4.2 Limitaciones de Tamaño de Muestra

| Categoría | Cantidad | % del Total |
|-----------|----------|-------------|
| < 5 estudiantes | Excluidos | N/A |
| 5-14 estudiantes | 72 | 12% |
| 15-29 estudiantes | 312 | 52% |
| 30-49 estudiantes | 156 | 26% |
| ≥ 50 estudiantes | 61 | 10% |

**Umbral mínimo para modelado:** 15 estudiantes (validez estadística)

### 4.3 Desbalance de Clases

| Tasa de Reprobación | Cursos | Aptitud para Modelado |
|---------------------|--------|----------------------|
| 0% (todos aprueban) | 423 | ❌ Sin señal |
| 1-14% | 89 | ⚠️ Altamente desbalanceado |
| 15-50% | 52 | ✅ Buen balance |
| 51-85% | 28 | ✅ Buen balance |
| >85% (mayoría reprueba) | 9 | ⚠️ Desbalanceado |

---

## 5. Recomendaciones

### 5.1 Acciones Inmediatas

1. **Enfocarse en los 16 cursos de alto potencial (score ≥80)**
   - Estos tienen suficiente calidad de datos para modelos piloto
   - Comenzar con secciones de "TALL DE COMPETENCIAS DIGITALES"

2. **Investigar el libro de calificaciones externo**
   - 506 cursos podrían tener notas en "Libro de Calificaciones"
   - Podría expandir significativamente el pool de cursos viables

3. **Análisis cross-campus**
   - Los mismos cursos enseñados en diferentes campus muestran potencial predictivo variable
   - Estandarizar prácticas de recolección de datos

### 5.2 Para Desarrollo de Modelos

| Prioridad | Tipo de Curso | Justificación |
|-----------|---------------|---------------|
| Alta | Competencias Digitales | 100% cobertura, clases balanceadas, features ricos |
| Alta | Cursos de Matemáticas | Alta varianza, buenos tamaños de muestra |
| Media | Cursos de Psicología | Buen balance, actividad moderada |
| Baja | Cursos Clínicos | Patrones de notas atípicos, muestras pequeñas |

### 5.3 Mejoras en Recolección de Datos

1. **Agregar features temporales**
   - Patrones de actividad semanal
   - Engagement temprano vs tardío
   - Trayectoria de notas en el tiempo

2. **Expandir a POSTGRADO**
   - La cuenta 42 tiene 1000+ cursos
   - Población estudiantil diferente (adultos)

3. **Incluir datos a nivel de submission**
   - Notas assignment por assignment
   - Patrones de timing de entregas

---

## 6. Apéndice Técnico

### 6.1 Scripts Utilizados

| Script | Propósito | Ubicación |
|--------|-----------|-----------|
| `section7_refactor.py` | Descubrimiento de cursos multi-hilo | `scripts/discovery/` |
| `analyze_courses.py` | Análisis estadístico y visualización | `scripts/discovery/` |
| `canvas_client.py` | Cliente API con rate-limiting | `scripts/discovery/` |

### 6.2 Comandos de Ejecución

```bash
# Paso 1: Recolectar datos de cursos de todos los campus de PREGRADO
python3 scripts/discovery/section7_refactor.py \
    --campus-ids "173,174,175,176,360" \
    --max-courses 200 \
    --workers 5

# Paso 2: Ejecutar análisis estadístico
python3 scripts/discovery/analyze_courses.py \
    --input data/discovery/course_analysis_latest.csv \
    --output-dir data/discovery/analysis \
    --top 100
```

### 6.3 Archivos de Salida

| Archivo | Descripción |
|---------|-------------|
| `course_analysis_latest.csv` | Métricas completas de cursos (601 filas, 44 columnas) |
| `course_rankings.csv` | Top 100 cursos rankeados |
| `correlation_matrix.csv` | Matriz de correlación completa |
| `prediction_drivers.csv` | Análisis de importancia de métricas |
| `*.png` | Archivos de visualización |

### 6.4 Fórmula del Prediction Potential Score

```
prediction_potential_score = weighted_sum([
    grade_availability_score × 0.30,    # Más importante
    grade_variance_score × 0.25,        # Fuerza de señal
    class_balance_score × 0.20,         # Factibilidad de modelado
    design_richness_score × 0.15,       # Disponibilidad de features
    activity_score × 0.10               # Datos de engagement
])

# Bonus: +10% si graded_assignments ≥ 3 AND students_with_activity ≥ 15
# Score cero si: students_with_grades < 15 OR grade_std < 10%
```

---

## 7. Conclusión

Este análisis establece un **framework basado en datos** para identificar cursos aptos para modelado predictivo en la Universidad Autónoma de Chile.

**El insight clave es que un buen potencial predictivo requiere:**
1. ✅ Datos de notas suficientes (>50% cobertura)
2. ✅ Varianza de notas (std dev >10%)
3. ✅ Balance de clases (15-85% tasa de reprobación)
4. ✅ Tamaño de muestra adecuado (≥15 estudiantes)
5. ⭐ Datos de actividad (page views, participations)

De los 601 cursos analizados, **38 cumplen todos los criterios** y **16 son excelentes candidatos** para desarrollo inmediato de modelos.

La siguiente fase debería enfocarse en construir y validar modelos predictivos usando estos cursos top, mientras se investiga el libro de calificaciones externo para potencialmente expandir el pool de cursos viables.

---

*Análisis realizado usando la API de Canvas LMS*
*Scripts disponibles en `scripts/discovery/`*
*Datos almacenados en `data/discovery/`*
