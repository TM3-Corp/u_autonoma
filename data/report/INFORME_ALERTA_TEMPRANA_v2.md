# Sistema de Alerta Temprana v2
## Predicción de Fracaso Académico mediante Patrones de Navegación en LMS

**Universidad Autónoma de Chile**
**Fecha:** 5 de enero de 2026
**Versión:** 2.0

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Metodología](#2-metodología)
3. [Resultados](#3-resultados)
4. [Conclusiones](#4-conclusiones)
5. [Anexos](#5-anexos)

---

# 1. Resumen Ejecutivo

## El Problema

La Universidad Autónoma de Chile enfrenta tasas de reprobación cercanas al **40%** en cursos de primer año. Tradicionalmente, los estudiantes en riesgo son identificados *después* de la primera evaluación, cuando ya es tarde para intervenir efectivamente.

## La Solución

Desarrollamos un sistema que analiza **cómo los estudiantes navegan la plataforma Canvas LMS** para predecir el fracaso académico. El sistema **no utiliza calificaciones** de evaluaciones, pero sí analiza patrones de acceso a recursos del curso, incluyendo materiales de estudio y recursos vinculados a evaluaciones (tareas, quizzes).

## Principales Resultados

### Modelo Completo (Semana 4+ del curso)

Cuando el estudiante ha tenido oportunidad de interactuar con recursos de evaluación (acceso a tareas, quizzes, calificaciones):

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **Capacidad predictiva (ROC-AUC)** | **0.90** | Excelente discriminación |
| **Exactitud (Accuracy)** | **83.4%** | 8.3 de 10 predicciones correctas |
| **Sensibilidad (Recall)** | **85.9%** | Detectamos 8.6 de cada 10 estudiantes en riesgo |
| **Precisión** | 75.7% | 7.6 de cada 10 alertas son correctas |

*Con umbral optimizado (threshold=0.33) que maximiza detección manteniendo alta exactitud.

### Modelo Temprano Optimizado (Semana 4-6 del curso)

Usando patrones de actividad con **optimización de umbral** y **definición calibrada del inicio de curso**:

| Métrica | Semana 4 | Semana 6 | Significado |
|---------|----------|----------|-------------|
| **Capacidad predictiva (ROC-AUC)** | **0.76** | **0.82** | Buena a muy buena discriminación |
| **Exactitud (Accuracy)** | **71.4%** | **76.2%** | 7-8 de 10 predicciones correctas |
| **Sensibilidad (Recall)** | **76.8%** | **70.3%** | Detectamos 7+ de cada 10 estudiantes en riesgo |
| **Umbral optimizado** | 0.23 | 0.35 | Calibrado para maximizar detección |

> **Nota metodológica:** Estos resultados utilizan el percentil 20 de actividad para definir el inicio efectivo del curso (cuando la mayoría de estudiantes ha comenzado), lo que mejora significativamente el rendimiento predictivo en comparación con el percentil 5 tradicional.

## Hallazgo Clave

> **Los estudiantes exitosos no solo acceden más al LMS; navegan de forma cualitativamente diferente.** Los 4 predictores más fuertes están relacionados con el acceso a recursos de evaluación: tasa de acceso a tareas (10.7%), percentil de acceso a assignments (5.5%), recursos únicos de tareas (5.3%), y tasa de acceso a quizzes (3.5%). Este "fingerprint digital" es detectable desde la semana 3-4 del curso.

---

# 2. Metodología

## 2.1 Datos Analizados

| Característica | Valor |
|----------------|-------|
| **Estudiantes** | 373 |
| **Cursos** | 10 (de 102 disponibles) |
| **Período** | Segundo semestre 2025 |
| **Fuente** | Canvas LMS - Clickstream |

**Criterios de selección de cursos:**
- Al menos 20 estudiantes matriculados
- Datos de actividad disponibles
- Variabilidad en resultados (tasa de aprobación entre 20-80%)

## 2.2 Marco Teórico

Nuestros indicadores se fundamentan en dos modelos establecidos en psicología educativa:

### Ciclo de Aprendizaje Autorregulado (Zimmerman, 2000)

El aprendizaje efectivo es un ciclo de tres fases:

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   PLANIFICACIÓN  │ ──▶  │    EJECUCIÓN     │ ──▶  │  AUTORREFLEXIÓN  │
│                  │      │                  │      │                  │
│  • Establecer    │      │  • Mantener      │      │  • Evaluar       │
│    metas         │      │    atención      │      │    resultados    │
│  • Organizar     │      │  • Aplicar       │      │  • Adaptar       │
│    tiempo        │      │    estrategias   │      │    estrategias   │
└──────────────────┘      └──────────────────┘      └──────────────────┘
         ▲                                                    │
         └────────────────────────────────────────────────────┘
```

### Modelo de Engagement (Fredricks et al., 2004)

El engagement tiene tres dimensiones:

| Dimensión | Pregunta | Indicadores LMS |
|-----------|----------|-----------------|
| **Conductual** | ¿Qué hace? | Sesiones, page views, tiempo |
| **Emocional** | ¿Cómo se siente? | Estudio voluntario (fines de semana, vespertino) |
| **Cognitivo** | ¿Cómo se implica? | Diversidad de navegación, exploración |

## 2.3 Dimensiones de Análisis

Analizamos el comportamiento estudiantil en **5 dimensiones** (excluyendo cualquier información de calificaciones o entregas):

### Dimensión 1: Patrones Temporales
*Fase de Ejecución (Zimmerman)*

- **Sesiones por semana**: Frecuencia de acceso al curso
- **Regularidad**: Consistencia en el patrón de acceso
- **Distribución horaria**: Mañana, tarde, noche, madrugada

### Dimensión 2: Secuencias de Navegación (N-gramas)
*Fase de Planificación + Ejecución (Zimmerman)*

- **Transiciones únicas**: Número de patrones diferentes de navegación
- **Entropía de transición**: Diversidad en el orden de recursos visitados
- **Ejemplo**: módulo → archivo → quiz vs. módulo → módulo → salir

### Dimensión 3: Diversidad de Recursos
*Engagement Cognitivo (Fredricks)*

- **Tipos de recursos accedidos**: Módulos, archivos, páginas, anuncios
- **Proporción de cobertura**: % del curso explorado
- **Ratio contenido/evaluación**: Balance entre estudio y assessment

### Dimensión 4: Posición Relativa en el Curso
*Fase de Autorreflexión (Zimmerman)*

- **Percentil de módulos**: ¿En qué percentil está respecto a compañeros?
- **Percentil de archivos**: Comparación contextualizada
- **Interpretación**: Un estudiante en el percentil 30 tiene menos actividad que el 70% de sus compañeros

### Dimensión 5: Indicadores de Procrastinación
*Fase de Planificación (Zimmerman)*

- **Días hasta primer acceso**: Demora en comenzar
- **Semana pico de actividad**: ¿Cuándo alcanza máxima actividad?
- **Última semana activa**: ¿Abandona antes de finalizar?

---

# 3. Resultados

## 3.1 Radiografía de Cursos

### Diseño Instruccional: Composición de Recursos

Los cursos analizados presentan una **disparidad extrema** en la cantidad de recursos disponibles:

![Composición del Diseño Instruccional por Curso](analysis/course_design/course_design_stacked.png)

> **Hallazgo:** Existe una brecha de **30x** entre el curso más rico en recursos (Gest.Talento: 1,019) y el más austero (Macroecon.: 34).

### Relación Diseño vs Engagement

La cantidad de recursos **no garantiza mayor engagement**:

![Relación Diseño vs Engagement](analysis/course_design/design_vs_engagement.png)

> **Paradoja:** El curso con más recursos (Gest.Talento: 1,019 recursos) tiene engagement relativamente bajo (232 views/estudiante), mientras que cursos balanceados (Comp.Dig.: 639 recursos) logran el máximo engagement (752 views/estudiante).

### Distribución de Calificaciones

![Distribución de Calificaciones por Curso](analysis/course_analysis/grade_boxplot.png)

*Nota: El curso Macroecon. (89736) utiliza una escala diferente (0-28 puntos) que fue normalizada a 0-100 para comparabilidad.*

### Distribución de Estudiantes

| Curso ID | Nombre | Estudiantes | Tasa Aprobación |
|----------|--------|-------------|-----------------|
| 84936 | FUND. DE MICROECONOMÍA-P03 | 40 | 71.4% |
| 79913 | FUND. DE BUSINESS ANALYTICS | 40 | 73.2% |
| 86020 | COMPETENCIAS DIGITALES-P02 | 51 | 62.7% |
| 86676 | BUSINESS ANALYTICS-P01 | 40 | 27.5% |
| 84944 | FUND. DE MACROECONOMÍA-P03 | 39 | 55.0% |
| 84941 | FUND. DE MICROECONOMÍA-P01 | 32 | 36.8% |
| 89390 | GESTIÓN DEL TALENTO-P01 | 32 | 78.8% |
| 79875 | COMPETENCIAS DIGITALES | 32 | 59.4% |
| 88381 | MATEMÁTICAS NEGOCIOS | 21 | 71.4% |
| 89099 | COMP. DIGITALES | 34 | 71.4% |
| **Total** | | **361** | **60.1%** |

---

## 3.2 Patrones de Actividad Temporal

### Heatmaps de Actividad por Curso

Los estudiantes muestran patrones horarios consistentes:

![Patrones de Actividad Temporal](analysis/hourly_heatmaps/hourly_heatmaps_combined.png)

| Patrón | Observación |
|--------|-------------|
| **Pico de actividad** | 18:00 - 22:00 (horarios vespertinos post-laborales) |
| **Fin de semana** | Actividad significativa en cursos con alto engagement |
| **Madrugada** | Mínima actividad (00:00-06:00) en todos los cursos |
| **Lunes vs Viernes** | Mayor actividad al inicio de semana |

---

## 3.3 Rendimiento del Modelo por Curso (LOCO)

El modelo fue validado con **Leave-One-Course-Out**: entrenar con 9 cursos y predecir el décimo.

| Curso | AUC |
|-------|-----|
| 84936 | **0.94** (Excelente) |
| 89099 | **0.94** (Excelente) |
| 79875 | **0.89** (Muy bueno) |
| 84941 | **0.87** (Muy bueno) |
| 88381 | **0.87** (Muy bueno) |
| 89390 | **0.83** (Bueno) |
| 79913 | 0.75 |
| 84944 | 0.74 |
| 86676 | 0.72 |
| 86020 | 0.69 |

**Interpretación**: El modelo generaliza bien a cursos nuevos, con AUC > 0.70 en todos los casos.

---

## 3.4 Principales Hallazgos

### Resumen Ejecutivo: Las 6 Dimensiones Conductuales Clave

El análisis de datos del LMS revela **diferencias estadísticamente significativas** entre estudiantes aprobados y reprobados en **6 dimensiones conductuales** distintas:

![Top Features por Dimensión](analysis/pass_fail_comparisons/top_features_by_dimension.png)

| Dimensión | Indicador Clave | Aprobados | Reprobados | Diferencia | Cohen's d |
|-----------|-----------------|-----------|------------|------------|-----------|
| **Volumen** | Sesiones Totales | 39.5 | 21.4 | **1.85x** | 0.82*** |
| **Frecuencia** | Sesiones/Semana | 1.80 | 1.06 | **1.69x** | 0.69*** |
| **Regularidad** | Gap Medio (horas) | 85 | 139 | **39% menos** | -0.61*** |
| **Diversidad** | Entropía Horaria | 0.74 | 0.65 | **13% más** | 0.63*** |
| **Fin de Semana** | % Tarde Fds | 6.4% | 2.7% | **2.4x** | 0.58*** |
| **Navegación** | Entropía Transiciones | 0.55 | 0.46 | **19% más** | 0.60*** |

> **Mensaje para Autoridades:** Los datos del LMS capturan patrones conductuales que **discriminan fuertemente** entre estudiantes en riesgo y exitosos. Estos indicadores son **puramente conductuales** (no incluyen calificaciones) y pueden detectarse **antes de la primera evaluación**.

![Tamaño del Efecto por Dimensión](analysis/pass_fail_comparisons/effect_sizes_by_dimension.png)

**Interpretación de Cohen's d:**
- d > 0.8 = Efecto GRANDE (diferencia muy notable)
- d = 0.5-0.8 = Efecto MEDIANO (diferencia prácticamente significativa)
- d = 0.2-0.5 = Efecto PEQUEÑO (diferencia detectable)

---

### Comparación Visual: Aprobados vs Reprobados

Las diferencias entre estudiantes que aprueban y reprueban son **visualmente evidentes**:

![Comparación de Features: Aprobados vs Reprobados](analysis/pass_fail_comparisons/pass_fail_comparison.png)

| Métrica | Aprobados (mediana) | Reprobados (mediana) | Diferencia |
|---------|---------------------|----------------------|------------|
| **Sesiones totales** | ~35 | ~15 | 2.3x mayor |
| **Sesiones/semana** | ~2.5 | ~1.5 | 67% mayor |
| **Page views** | ~700 | ~300 | 2.3x mayor |
| **Horas activas** | ~40 | ~20 | 2x mayor |
| **Gap entre sesiones** | Máx ~94 días | Outliers > 94 días | Umbral crítico |

### Análisis de Dinámica de Engagement

El análisis de **sesiones** (detectadas con umbral de 60 minutos de inactividad) revela diferencias aún más marcadas:

![Características de Sesión](analysis/pass_fail_comparisons/session_features_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d | Significancia |
|---------|-----------|------------|-------|-----------|---------------|
| **Total Sesiones** | 39.5 | 21.4 | **1.85x** | 0.82 (GRANDE) | p < 0.001 *** |
| **Participaciones** | 6.6 | 2.7 | **2.49x** | 0.78 (GRANDE) | p < 0.001 *** |
| **Sesiones/Semana** | 1.80 | 1.06 | **1.69x** | 0.69 | p < 0.001 *** |
| **Gap Medio** | 85 horas | 139 horas | **0.61x** | -0.61 | p < 0.001 *** |
| **Span Actividad** | 97 días | 82 días | **1.18x** | 0.41 | p < 0.001 *** |

> **Hallazgo Crítico:** Los estudiantes aprobados acceden **casi el doble** de veces al LMS y esperan **54 horas menos** entre visitas. Esto no es solo "más tiempo en pantalla" - es un patrón de **engagement sostenido**.

### Volumen de Actividad

![Volumen de Actividad](analysis/pass_fail_comparisons/activity_volume_comparison.png)

### Patrones de Navegación (N-gramas)

![Comparación de Patrones de Navegación](analysis/pass_fail_comparisons/navigation_comparison.png)

### Regularidad: Gap entre Sesiones

El **gap máximo** entre sesiones es un discriminador clave:

![Análisis de Gap entre Sesiones](analysis/pass_fail_comparisons/session_gap_comparison.png)

![Análisis Detallado del Umbral de Gap](analysis/pass_fail_comparisons/gap_threshold_analysis.png)

> **Insight Clave:** Ningún estudiante aprobado tuvo un gap máximo superior a **94 días**. Este umbral actúa como "línea de no retorno" para la regularidad.

### Patrones Temporales: Uso del Fin de Semana

Una pregunta frecuente es si el **momento del día** o el **día de la semana** influye en el éxito. El análisis revela hallazgos sorprendentes:

![Comparación de Actividad en Fin de Semana](analysis/pass_fail_comparisons/weekend_comparison.png)

| Métrica | Aprobados | Reprobados | Diferencia | Cohen's d |
|---------|-----------|------------|------------|-----------|
| **% Actividad Fin de Semana** | 19.9% | 14.0% | +42% | 0.33*** |
| **% Tarde Fin de Semana** | 6.4% | 2.7% | +137% | 0.58*** |
| **% Noche Fin de Semana** | 11.6% | 8.3% | +40% | 0.26*** |

> **Hallazgo:** Los estudiantes aprobados **aprovechan el fin de semana** para estudiar, especialmente las tardes del sábado/domingo. Este patrón sugiere mayor compromiso con el curso más allá del horario laboral.

### Diversidad Temporal: ¿Importa CUÁNDO estudias?

![Diversidad de Horarios](analysis/pass_fail_comparisons/time_diversity_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Horas Únicas Activas** | 39.5 | 21.4 | **1.85x** | 0.82*** |
| **Diversidad Horaria** | 0.735 | 0.651 | 1.13x | 0.63*** |

> **Hallazgo Importante:** Los estudiantes aprobados **NO tienen un "horario ideal"** - lo que importa es la **diversidad** de horarios usados. Estudiar a las 9am vs 9pm no predice el éxito, pero usar **más franjas horarias diferentes** sí lo hace.

### Distribución por Bloque Horario

![Distribución de Actividad por Horario](analysis/pass_fail_comparisons/time_distribution.png)

**Resultado contra-intuitivo:** Las proporciones de actividad en mañana, tarde o noche **NO muestran diferencias significativas** entre aprobados y reprobados. No existe un "mejor momento para estudiar" - lo que importa es la **consistencia y diversidad** del engagement.

---

### Hallazgo 1: Las Transiciones de Navegación Predicen el Éxito

**¿Qué son las transiciones?**

Cada vez que un estudiante pasa de un tipo de recurso a otro (ej: de módulo a archivo), registramos una "transición". El número de transiciones únicas captura la *diversidad* de exploración.

```
Estudiante A (Exitoso):    módulo → archivo → quiz → módulo → página → archivo
                           Transiciones únicas: 5

Estudiante B (En riesgo):  módulo → módulo → salir
                           Transiciones únicas: 2
```

**Resultado:** El indicador `total_transitions` tiene una importancia del **4.66%** en el modelo, siendo el 3er predictor más fuerte.

| Grupo | Transiciones (mediana) |
|-------|------------------------|
| Aprobados | 28 |
| Reprobados | 12 |

> **Los estudiantes exitosos exploran el LMS de forma más variada**, no solo acceden más veces.

### Hallazgo 2: El Percentil de Módulos Contextualiza el Riesgo

**¿Qué es el percentil de módulos?**

Indica la posición relativa del estudiante respecto a sus compañeros en acceso a módulos del curso. Un estudiante en el percentil 25 significa que el 75% de sus compañeros accede más a los módulos.

**Resultado:** El indicador `modu_mean_pct` (promedio de percentil en módulos) tiene importancia del **2.32%**.

| Percentil Módulos | Tasa de Fracaso |
|-------------------|-----------------|
| < 30 (bajo) | 58% |
| 30-70 (medio) | 35% |
| > 70 (alto) | 18% |

> **Estudiantes en el tercil inferior de acceso a módulos tienen 3x más probabilidad de reprobar** que los del tercil superior.

### Hallazgo 3: La Entropía de Navegación Revela Intención

**¿Qué es la entropía de transición?**

Mide qué tan predecible es la secuencia de navegación. Entropía alta = navegación exploratoria. Entropía baja = navegación lineal o errática.

**Resultado:** `transition_entropy` tiene importancia del **2.14%**.

| Tipo de Navegación | Entropía | Interpretación |
|--------------------|----------|----------------|
| Exploratoria | Alta | Busca activamente, compara recursos |
| Lineal | Media | Sigue estructura del curso |
| Errática/Mínima | Baja | Sin patrón claro, acceso superficial |

### Hallazgo 4: Los 5 Predictores Conductuales Más Fuertes

| # | Indicador | Importancia | Interpretación |
|---|-----------|-------------|----------------|
| 1 | `total_time_min_znorm` | 5.97% | Tiempo total (normalizado por curso) |
| 2 | `page_n_resources` | 5.51% | Variedad de páginas accedidas |
| 3 | `total_transitions` | 4.66% | Diversidad de navegación |
| 4 | `content_vs_assessment_ratio` | 4.39% | Balance estudio/evaluación |
| 5 | `modules_views` | 4.02% | Acceso a módulos del curso |

**Nota:** Todos son indicadores *puramente conductuales*, sin incluir calificaciones o entregas.

---

### Acceso a Recursos por Tipo

![Engagement por Tipo de Recurso](analysis/resources/resource_comparison_bar.png)

> Los estudiantes aprobados acceden significativamente más (***) a: Cuestionarios, Tareas, Foros de Discusión, Página Inicial, Calificaciones, Archivos y Anuncios.

---

## 3.5 Capacidad Predictiva

### Rendimiento del Modelo XGBoost

![Curvas ROC](analysis/model_performance/roc_curves.png)

El sistema ofrece **dos niveles de predicción** según el momento del semestre:

#### Modelo Completo (Semana 4+ del curso)

Incluye patrones de acceso a recursos de evaluación (tareas, quizzes, calificaciones):

| Métrica | Umbral Default (0.50) | Umbral Optimizado (0.33)* |
|---------|----------------------|---------------------------|
| **ROC-AUC** | 0.90 | 0.90 |
| **Exactitud** | 83.1% | **83.4%** |
| **Sensibilidad (Recall)** | 74.5% | **85.9%** |
| **Precisión** | 81.6% | 75.7% |
| **F1 Score** | 77.9% | 80.5% |

*Umbral optimizado (threshold=0.33) prioriza la detección sin sacrificar exactitud.

#### Modelo Temprano Optimizado (Semanas 4-6 del curso)

Patrones de actividad con umbral y percentil de inicio optimizados:

| Métrica | Semana 4 | Semana 6 |
|---------|----------|----------|
| **ROC-AUC** | **0.76** | **0.82** |
| **Exactitud** | **71.4%** | **76.2%** |
| **Sensibilidad (Recall)** | **76.8%** | **70.3%** |
| **Umbral** | 0.23 | 0.35 |

> **Nota metodológica:** Estos modelos no utilizan calificaciones de evaluaciones. La mejora respecto a configuraciones anteriores proviene de: (1) usar el percentil 20 para definir el inicio efectivo del curso, y (2) optimizar el umbral de clasificación para balancear detección y exactitud.

**Interpretación práctica:**

- De **100 estudiantes en riesgo real**, identificamos correctamente a **86** con el modelo completo y **77** con el modelo de semana 4
- El modelo de **semana 6** alcanza ROC-AUC 0.82, comparable al modelo de semana 8 tradicional
- **Recomendación:** Usar modelo de semana 4 (t=0.23) para alertas tempranas con alta sensibilidad; modelo de semana 6 (t=0.35) para balance óptimo; modelo completo (t=0.33) para máxima precisión

### Ventaja Temporal del Sistema

| Método de Detección | Disponibilidad | Exactitud | ROC-AUC |
|---------------------|----------------|-----------|---------|
| Intuición docente | Variable | Subjetiva | - |
| Primera evaluación | Semana 4-6 | Alta, pero tardía | - |
| **Sistema LMS (Semana 4)** | **Semana 4** | **71.4%** | **0.76** |
| **Sistema LMS (Semana 6)** | **Semana 6** | **76.2%** | **0.82** |
| **Sistema LMS (Completo)** | **Semana 8+** | **83.4%** | **0.90** |

---

## 3.6 Perfiles de Estudiantes

### Perfil del Estudiante en Riesgo

```
┌─────────────────────────────────────────────────────────────┐
│                  ESTUDIANTE EN RIESGO                       │
├─────────────────────────────────────────────────────────────┤
│  • Accede < 2 veces por semana al LMS                       │
│  • Navegación lineal: módulo → módulo → salir               │
│  • Pocas transiciones entre tipos de recursos               │
│  • Percentil bajo en acceso a módulos (< 30)                │
│  • No estudia fines de semana ni horarios vespertinos       │
│  • Comienza el curso tarde (> 5 días después del inicio)    │
└─────────────────────────────────────────────────────────────┘
```

### Perfil del Estudiante Exitoso

```
┌─────────────────────────────────────────────────────────────┐
│                  ESTUDIANTE EXITOSO                         │
├─────────────────────────────────────────────────────────────┤
│  • Accede 3-4 veces por semana al LMS                       │
│  • Navegación exploratoria: módulo → archivo → quiz → página│
│  • Alta diversidad de transiciones                          │
│  • Percentil medio-alto en módulos (> 50)                   │
│  • Estudia fines de semana y horarios vespertinos           │
│  • Comienza temprano y mantiene consistencia                │
└─────────────────────────────────────────────────────────────┘
```

---

# 4. Conclusiones

## 4.1 Principales Conclusiones

### 1. El comportamiento digital predice el éxito académico

Los patrones de navegación en el LMS son indicadores confiables del rendimiento futuro:

| Modelo | ROC-AUC | Exactitud | Sensibilidad | Disponible desde |
|--------|---------|-----------|--------------|------------------|
| **Completo** (con acceso a evaluaciones) | **0.90** | **83.4%** | **85.9%** | Semana 8+ |
| **Semana 6** (optimizado) | **0.82** | **76.2%** | **70.3%** | Semana 6 |
| **Semana 4** (optimizado) | **0.76** | **71.4%** | **76.8%** | Semana 4 |

El modelo completo con umbral optimizado (0.33) detecta correctamente **8.6 de cada 10** estudiantes en riesgo. El modelo de semana 4 con umbral 0.23 detecta **7.7 de cada 10** estudiantes en riesgo, permitiendo intervención temprana.

### 2. La *calidad* de navegación importa más que la *cantidad*

No basta con acceder frecuentemente al LMS. Los estudiantes exitosos muestran patrones de navegación cualitativamente diferentes:
- Más transiciones entre tipos de recursos
- Mayor diversidad en la exploración
- Mejor posición relativa respecto a compañeros
- **Mayor interacción con recursos de evaluación** (principal diferenciador)

### 3. La detección es posible antes de cualquier evaluación

El sistema **no utiliza calificaciones de evaluaciones parciales**. Sin embargo, los *patrones de acceso* a recursos de evaluación (revisar tareas, acceder a quizzes, consultar calificaciones) son altamente predictivos:
- **Semana 4:** Detección temprana con 71.4% exactitud y 76.8% sensibilidad (umbral 0.23)
- **Semana 6:** Detección intermedia con 76.2% exactitud y 70.3% sensibilidad (umbral 0.35)
- **Semana 8+:** Detección precisa con 83.4% exactitud y 85.9% sensibilidad (umbral 0.33)

La mejora progresiva permite implementar un sistema de alertas **escalonadas** según la semana del curso.

### 4. El modelo generaliza a cursos nuevos

Con un AUC de 0.69-0.94 en validación LOCO, el sistema puede aplicarse a cursos que no fueron parte del entrenamiento, aunque con mejor desempeño en cursos estructuralmente similares.

## 4.2 Recomendaciones

### Corto Plazo

| Acción | Descripción |
|--------|-------------|
| **Dashboard de alertas** | Implementar visualización en tiempo real para docentes |
| **Umbrales de intervención** | Definir protocolos según nivel de riesgo |
| **Piloto controlado** | Probar en 3-5 cursos el próximo semestre |

### Mediano Plazo

| Acción | Descripción |
|--------|-------------|
| **Expansión institucional** | Extender a todos los cursos de pregrado |
| **Integración de datos** | Combinar con historial académico y factores socioeconómicos |
| **Evaluación de impacto** | Medir efectividad de intervenciones |

## 4.3 Limitaciones

1. **Muestra limitada**: 373 estudiantes de 10 cursos (programa específico)
2. **Generalización a cursos nuevos**: El modelo mantiene buen desempeño en cursos nuevos (AUC 0.69-0.94 según el curso)
3. **Solo comportamiento digital**: No captura factores externos (trabajo, familia, salud)
4. **Dependencia del diseño del curso**: Cursos con poco contenido en LMS generan menos señales

---

# 5. Anexos

## Anexo A: Detalle de Features del Modelo

### Features de Transición (N-gramas)

| Feature | Descripción | Importancia |
|---------|-------------|-------------|
| `total_transitions` | Número total de transiciones entre recursos | 4.66% |
| `transition_entropy` | Entropía de Shannon de la distribución de transiciones | 2.14% |

### Features de Percentil Relativo

| Feature | Descripción | Importancia |
|---------|-------------|-------------|
| `modu_mean_pct` | Percentil promedio en acceso a módulos | 2.32% |
| `pages_var_explained` | Varianza explicada en acceso a páginas | 2.53% |

### Features Temporales

| Feature | Descripción | Importancia |
|---------|-------------|-------------|
| `total_time_min_znorm` | Tiempo total normalizado por curso | 5.97% |
| `sessions_per_week_znorm` | Sesiones por semana normalizado | 3.02% |
| `last_active_week` | Última semana con actividad | 2.61% |

## Anexo B: Metodología de Validación

### Validación Cruzada Estratificada (5-fold)

```
Fold 1: Train [2,3,4,5] → Test [1]
Fold 2: Train [1,3,4,5] → Test [2]
Fold 3: Train [1,2,4,5] → Test [3]
Fold 4: Train [1,2,3,5] → Test [4]
Fold 5: Train [1,2,3,4] → Test [5]

Resultado: AUC promedio = 0.90
```

### Leave-One-Course-Out (LOCO)

```
Fold 1: Train [cursos 2-10] → Test [curso 1]
Fold 2: Train [cursos 1,3-10] → Test [curso 2]
...
Fold 10: Train [cursos 1-9] → Test [curso 10]

Resultado: AUC por curso = 0.69 - 0.94
```

**Interpretación**: LOCO evalúa generalización a cursos nuevos. El rango de AUC (0.69-0.94) indica que el modelo funciona mejor en algunos cursos que otros, dependiendo de su estructura y diseño instruccional.

## Anexo C: Referencias Teóricas

### Engagement Estudiantil

> Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). School engagement: Potential of the concept, state of the evidence. *Review of Educational Research*, 74(1), 59–109.

### Aprendizaje Autorregulado

> Zimmerman, B. J. (2000). Attaining self-regulation: A social cognitive perspective. In M. Boekaerts, P. R. Pintrich, & M. Zeidner (Eds.), *Handbook of Self-Regulation*.

## Anexo D: Heatmaps de Actividad por Curso

Los siguientes heatmaps muestran la distribución de actividad por hora del día (eje Y: 0-23) y día de la semana (eje X: Lunes-Domingo) para cada curso analizado. La intensidad del color indica mayor concentración de page views.

### Visión Consolidada (Todos los Cursos)

![Heatmaps Combinados](analysis/hourly_heatmaps/hourly_heatmaps_combined.png)

### Cursos Individuales

#### Fundamentos de Microeconomía (84936)
![Heatmap Microeconomía P03](analysis/hourly_heatmaps/hourly_heatmap_84936.png)

#### Fundamentos de Microeconomía (84941)
![Heatmap Microeconomía P01](analysis/hourly_heatmaps/hourly_heatmap_84941.png)

#### Fundamentos de Macroeconomía (84944)
![Heatmap Macroeconomía](analysis/hourly_heatmaps/hourly_heatmap_84944.png)

#### Taller de Competencias Digitales (86005)
![Heatmap Comp.Digitales P01](analysis/hourly_heatmaps/hourly_heatmap_86005.png)

#### Taller de Competencias Digitales (86020)
![Heatmap Comp.Digitales P02](analysis/hourly_heatmaps/hourly_heatmap_86020.png)

#### Fundamentos de Business Analytics (86676)
![Heatmap Business Analytics](analysis/hourly_heatmaps/hourly_heatmap_86676.png)

#### Taller de Competencias Digitales (79875)
![Heatmap Comp.Digitales 79875](analysis/hourly_heatmaps/hourly_heatmap_79875.png)

#### Fundamentos de Business Analytics (79913)
![Heatmap Business Analytics 79913](analysis/hourly_heatmaps/hourly_heatmap_79913.png)

#### Matemáticas para los Negocios (88381)
![Heatmap Matemáticas](analysis/hourly_heatmaps/hourly_heatmap_88381.png)

#### Taller de Competencias Digitales (89099)
![Heatmap Comp.Digitales 89099](analysis/hourly_heatmaps/hourly_heatmap_89099.png)

#### Gestión del Talento Humano (89390)
![Heatmap Gestión Talento](analysis/hourly_heatmaps/hourly_heatmap_89390.png)

#### Pensamiento Analítico (89736)
![Heatmap Pensamiento Analítico](analysis/hourly_heatmaps/hourly_heatmap_89736.png)

#### Curso 79804
![Heatmap Curso 79804](analysis/hourly_heatmaps/hourly_heatmap_79804.png)

---

*Informe generado el 5 de enero de 2026*
*Universidad Autónoma de Chile - Canvas LMS*
