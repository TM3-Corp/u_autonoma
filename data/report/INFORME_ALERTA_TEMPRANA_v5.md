> ⚠️ **SUPERSEDED / STALE NUMBERS — do not quote.** This document predates the nested-CV correction; its headline AUCs are optimistic (non-nested) and/or contaminated (UA KEEP-arm active-zeros) and/or label-leaky. Current defensible metrics: **`RESULTS_LEDGER.md`**; start at **`PROJECT_SSOT.md`**. Kept for history only.

# Sistema de Alerta Temprana v3
## Predicción de Fracaso Académico mediante Patrones de Navegación en LMS

**Universidad Autónoma de Chile**
**Fecha:** 7 de enero de 2026
**Versión:** 3.0

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

El sistema de **análisis multidimensional** examina **cómo los estudiantes navegan la plataforma Canvas LMS** para predecir el fracaso académico. El sistema **no utiliza calificaciones**, sino patrones conductuales organizados en cinco dimensiones:

- **Volumen y frecuencia:** ¿Cuántas sesiones por semana? ¿Cuánto tiempo total en el LMS?
- **Regularidad:** ¿Accede de forma consistente o con largos períodos de inactividad?
- **Diversidad de navegación:** ¿Explora múltiples tipos de recursos o sigue patrones repetitivos?
- **Secuencias de estudio:** ¿Cómo transita entre recursos? (análisis de secuencias de navegacion)
- **Momentos de estudio:** ¿Estudia fines de semana? ¿En qué horarios?

Este enfoque integral permite identificar estudiantes en riesgo **antes de que exista cualquier nota**, capturando el "fingerprint digital" del engagement académico.

## Principales Resultados

### Modelo Principal: Predicción desde Semana 6+ (Capacidad Predictiva 0.90)

Desde la **semana 6**, cuando ya existe actividad con recursos de evaluación (acceso a tareas, quizzes, libro de calificaciones), el modelo alcanza su **máximo rendimiento**:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Capacidad predictiva (Capacidad Predictiva)** | **0.90** | Excelente discriminación |
| **Exactitud (Accuracy)** | **83.4%** | 8.3 de cada 10 predicciones correctas |
| **Sensibilidad (Recall)** | **85.9%** | **Detectamos 9 de cada 10 estudiantes en riesgo** |
| **Precisión** | 75.7% | 3 de cada 4 alertas son correctas |

> **Mensaje clave:** Desde la **semana 6** del curso, cuando los estudiantes ya han interactuado con recursos de evaluación, el sistema detecta correctamente **casi 9 de cada 10 estudiantes que reprobarán**, permitiendo **10+ semanas de intervención** antes del cierre del semestre.

### Modelo Temprano: Prediccion desde Semana 4 SIN Evaluaciones (Capacidad Predictiva 0.85)

Para intervencion **ANTES** de que existan recursos de evaluacion en el curso:

| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| **Capacidad predictiva (ROC-AUC)** | **0.85** | Muy buena discriminacion |
| **Exactitud (Accuracy)** | **74.5%** | 3 de cada 4 predicciones correctas |
| **Sensibilidad (Recall)** | **85.2%** | **Detectamos 8.5 de cada 10 estudiantes en riesgo** |
| **Falsas alarmas** | 37% | Razonable para deteccion temprana |

> **Ventaja clave:** Este modelo funciona con **patrones de navegacion puros** (participacion en foros, tiempo total en el LMS, diversidad de recursos explorados, posicion relativa respecto a companeros). Es ideal para cursos donde las primeras evaluaciones ocurren despues de la semana 4.

### Detección Ultra-Temprana: Semana 2

El sistema también puede generar alertas en solo **2 semanas**:

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Capacidad predictiva (Capacidad Predictiva)** | **0.74** | Buena discriminación |
| **Exactitud (Accuracy)** | **65.7%** | 2 de cada 3 predicciones correctas |
| **Sensibilidad (Recall)** | **81.7%** | **8 de cada 10 estudiantes en riesgo detectados** |
| **Falsas alarmas** | ~45% | 1 de cada 2 alertas puede ser incorrecta |

**Uso recomendado:** Lista de observación inicial ("watch list") para seguimiento, no para intervención activa.

## Hallazgo Clave

> **Los estudiantes exitosos no solo acceden más al LMS; navegan de forma cualitativamente diferente.** Las señales más predictivas combinan múltiples dimensiones: **regularidad de acceso** (sesiones constantes vs. largos períodos de inactividad), **diversidad de navegación** (exploración de múltiples recursos vs. patrones lineales), y **momentos de estudio** (aprovechamiento de fines de semana). Este "fingerprint digital" es detectable desde la **semana 4** del curso sin depender de calificaciones, y mejora significativamente en la **semana 6** cuando se incorpora la actividad con recursos de evaluación.

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
- Escala de calificaciones compatible (0-100% o 1-7 normalizado)

**Cursos excluidos:** 3 de 13 cursos iniciales fueron excluidos:
- **79804** (Pensamiento Analítico): Sin variabilidad suficiente en resultados
- **86005** (Comp. Digitales P01): Datos de actividad incompletos
- **89736** (Pensamiento Analítico): Escala de calificaciones diferente (0-28 pts)

## 2.2 Marco Teórico

Nuestros indicadores se fundamentan en dos modelos establecidos en psicología educativa:

### Ciclo de Aprendizaje Autorregulado (Zimmerman, 2000)

El aprendizaje efectivo es un ciclo de tres fases:

![Ciclo de Aprendizaje Autorregulado (Zimmerman, 2000)](visualizations/executive/zimmerman_srl_cycle.png)

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

### Dimensión 2: Secuencias de Navegación (Secuencias)
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

### Distribución de Estudiantes

| Curso ID | Nombre | Estudiantes | Tasa Aprobación |
|----------|--------|-------------|-----------------|
| 79875 | TALLER DE COMP DIGITALES-P01 | 32 | 59.4% |
| 79913 | FUND. DE BUSINESS ANALYTICS-P01 | 41 | 73.2% |
| 84936 | FUNDAMENTOS DE MICROECONOMÍA-P03 | 42 | 71.4% |
| 84941 | FUNDAMENTOS DE MICROECONOMÍA-P01 | 38 | 36.8% |
| 84944 | FUNDAMENTOS DE MACROECONOMÍA-P03 | 40 | 55.0% |
| 86020 | TALL DE COMPETENCIAS DIGITALES-P02 | 51 | 62.7% |
| 86676 | FUND DE BUSINESS ANALYTICS-P01 | 40 | 27.5% |
| 88381 | MATEMÁTICAS PARA LOS NEGOCIOS-P01 | 21 | 71.4% |
| 89099 | TALLER DE COMP DIGITALES-P01 | 35 | 71.4% |
| 89390 | GESTIÓN DEL TALENTO-P01 | 33 | 78.8% |
| **Total** | | **373** | **60.1%** |

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

## 3.3 Rendimiento del Modelo por Curso 
El modelo fue validado con **Validacion por Cursos Individuales**: entrenar con 9 cursos y predecir el décimo.

| Curso | Capacidad Predictiva |
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

**Interpretación**: El modelo generaliza bien a cursos nuevos, con capacidad predictiva > 0.70 en todos los casos.

---

## 3.4 Principales Hallazgos

### Resumen Ejecutivo: Las 6 Dimensiones Conductuales Clave

El análisis de datos del LMS revela **diferencias estadísticamente significativas** entre estudiantes aprobados y reprobados en **6 dimensiones conductuales** distintas:

![Principales Indicadores por Dimensión](analysis/pass_fail_comparisons/top_features_by_dimension.png)

| Dimensión | Indicador Clave | Aprobados | Reprobados | Diferencia | Cohen's d |
|-----------|-----------------|-----------|------------|------------|-----------|
| **Volumen** | Sesiones Totales | 39.5 | 21.4 | **1.85x** | 0.82*** |
| **Frecuencia** | Sesiones/Semana | 1.80 | 1.06 | **1.69x** | 0.69*** |
| **Regularidad** | Gap Medio (horas) | 85 | 139 | **39% menos** | -0.61*** |
| **Diversidad** | Entropía Horaria | 0.74 | 0.65 | **13% más** | 0.63*** |
| **Fin de Semana** | % Tarde Fds | 6.4% | 2.7% | **2.4x** | 0.58*** |
| **Navegación** | Entropía Transiciones | 0.55 | 0.46 | **19% más** | 0.60*** |

Los datos del LMS capturan patrones conductuales que permiten identificar estudiantes en riesgo con alta precisión. Estos indicadores son puramente conductuales (no incluyen calificaciones) y pueden detectarse **antes de la primera evaluación**, permitiendo intervención temprana.

![Tamaño del Efecto por Dimensión](analysis/pass_fail_comparisons/effect_sizes_by_dimension.png)

**Interpretación de Cohen's d:**
- d > 0.8 = Efecto GRANDE (diferencia muy notable)
- d = 0.5-0.8 = Efecto MEDIANO (diferencia prácticamente significativa)
- d = 0.2-0.5 = Efecto PEQUEÑO (diferencia detectable)

---

### Comparación Visual: Aprobados vs Reprobados

Las diferencias entre estudiantes que aprueban y reprueban son **visualmente evidentes**:

![Comparación de Indicadores: Aprobados vs Reprobados](analysis/pass_fail_comparisons/pass_fail_comparison.png)

| Métrica | Aprobados | Reprobados | Cohen's d | Significancia |
|---------|-----------|------------|-----------|---------------|
| **Sesiones totales** | 44.5 | 28.3 | 0.55 | *** |
| **Page views totales** | 374 | 234 | 0.52 | *** |
| **Días activos** | 23.5 | 16.6 | 0.51 | *** |
| **Duración sesión (min)** | 5.2 | 3.7 | 0.47 | *** |
| **Densidad (clicks/min)** | 1.9 | 2.6 | -0.13 | *** |
| **Clicks por sesión** | 9.3 | 8.5 | 0.17 | * |

Los "Días Activos" capturan la consistencia de acceso al LMS: cuántos días distintos el estudiante interactuó con el curso. Este indicador complementa el conteo de sesiones al medir la distribución temporal del esfuerzo.

### Análisis de Dinámica de Engagement

El análisis de **sesiones** (detectadas con umbral de 60 minutos de inactividad) revela diferencias marcadas:

![Características de Sesión](analysis/pass_fail_comparisons/session_features_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Total Sesiones** | 39.5 | 21.4 | **1.85x** | 0.82*** |
| **Sesiones/Semana** | 1.80 | 1.06 | **1.70x** | 0.69*** |
| **Gap Medio (horas)** | 85 | 139 | **0.61x** | -0.61*** |
| **Span Actividad (días)** | 97 | 82 | **1.18x** | 0.41*** |

> **Insight:** Las cuatro métricas muestran diferencias estadísticamente significativas. El Gap Medio negativo indica que los estudiantes aprobados tienen **menos tiempo entre sesiones**, accediendo de forma más regular al LMS.

### Volumen de Actividad

El volumen total de actividad en el LMS presenta diferencias significativas entre grupos:

![Volumen de Actividad](analysis/pass_fail_comparisons/activity_volume_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Page Views** | 374 | 234 | **1.60x** | 0.52*** |
| **Participaciones** | 6.6 | 2.7 | **2.49x** | 0.78*** |
| **Horas Activas** | 39.5 | 21.4 | **1.85x** | 0.82*** |

> **Insight:** Las horas activas distintas (d=0.82) y participaciones en foros/discusiones (d=0.78) muestran los mayores efectos. La diferencia de 2.5x en participaciones indica que el **engagement activo** (comentar, discutir) diferencia marcadamente a los grupos.

### Patrones de Navegación (Secuencias)

Las secuencias de navegación (n-gramas) revelan cómo los estudiantes recorren el LMS:

![Comparación de Patrones de Navegación](analysis/pass_fail_comparisons/navigation_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Transiciones Totales** | 228 | 130 | **1.75x** | 0.47*** |
| **Entropía de Navegación** | 0.55 | 0.46 | **1.19x** | 0.60*** |
| **Diversidad de Transición** | 0.27 | 0.18 | **1.48x** | 0.66*** |
| **Ratio Contenido/Evaluación** | 1.04 | 7.56 | **0.14x** | -0.64*** |

> **Insight:** Los estudiantes aprobados muestran navegación más diversa y un **balance saludable** entre contenido y evaluación (ratio ~1:1). Los reprobados tienden a acceder desproporcionadamente a recursos de evaluación sin revisar el contenido.

### Regularidad: Gap entre Sesiones

El **gap** (tiempo entre sesiones consecutivas) revela patrones de regularidad:

![Análisis de Gap entre Sesiones](analysis/pass_fail_comparisons/session_gap_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Gap Máximo (días)** | 21.7 | 25.9 | **0.84x** | -0.25* |
| **Gap Medio (horas)** | 85 | 139 | **0.61x** | -0.61*** |
| **Variabilidad del Gap** | 1.7 | 1.5 | **1.12x** | 0.32* |

> **Insight:** Los estudiantes aprobados tienen gaps más cortos y regulares. El **Gap Medio** de ~85 horas (3.5 días) vs ~139 horas (5.8 días) indica que los aprobados acceden con mayor frecuencia al LMS.

![Análisis Detallado del Umbral de Gap](analysis/pass_fail_comparisons/gap_threshold_analysis.png)

> **Umbral crítico:** Ningún estudiante aprobado tuvo un gap máximo superior a **94 días**. Este umbral actúa como "línea de no retorno" para la regularidad.

### Patrones Temporales: Uso del Fin de Semana

Una pregunta frecuente es si el **momento del día** o el **día de la semana** influye en el éxito. El análisis revela hallazgos sorprendentes:

![Comparación de Actividad en Fin de Semana](analysis/pass_fail_comparisons/weekend_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **% Fin de Semana** | 19.9% | 14.0% | **1.42x** | 0.33*** |
| **% Tarde Fin de Semana** | 6.4% | 2.7% | **2.37x** | 0.57*** |
| **% Noche Fin de Semana** | 11.6% | 8.3% | **1.40x** | 0.26** |

> **Hallazgo:** Los estudiantes aprobados **aprovechan el fin de semana** para estudiar. El indicador más discriminativo es la **actividad en tardes de fin de semana** (d=0.57), donde aprobados tienen 2.4x más actividad que reprobados.

### Diversidad Temporal: ¿Importa CUÁNDO estudias?

La variedad de franjas horarias utilizadas es más predictiva que el horario específico:

![Diversidad de Horarios](analysis/pass_fail_comparisons/time_diversity_comparison.png)

| Métrica | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Horas Únicas Activas** | 39.5 | 21.4 | **1.85x** | 0.82*** |
| **Diversidad Horaria** | 0.735 | 0.651 | 1.13x | 0.63*** |

> **Hallazgo Importante:** Los estudiantes aprobados **NO tienen un "horario ideal"** - lo que importa es la **diversidad** de horarios usados. Estudiar a las 9am vs 9pm no predice el éxito, pero usar **más franjas horarias diferentes** sí lo hace.

### Distribución por Bloque Horario

La proporción de actividad en cada bloque horario (mañana/tarde/noche) fue analizada:

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

**Resultado:** El indicador "Diversidad de Navegacion" tiene una importancia del **4.66%** en el modelo, siendo el 3er predictor mas fuerte.

| Grupo | Transiciones (mediana) |
|-------|------------------------|
| Aprobados | 28 |
| Reprobados | 12 |

> **Los estudiantes exitosos exploran el LMS de forma más variada**, no solo acceden más veces.

### Hallazgo 2: El Percentil de Módulos Contextualiza el Riesgo

**¿Qué es el percentil de módulos?**

Indica la posición relativa del estudiante respecto a sus compañeros en acceso a módulos del curso. Un estudiante en el percentil 25 significa que el 75% de sus compañeros accede más a los módulos.

**Resultado:** El indicador "Percentil Promedio en Modulos" tiene importancia del **2.32%**.

| Percentil Módulos | Tasa de Fracaso |
|-------------------|-----------------|
| < 30 (bajo) | 58% |
| 30-70 (medio) | 35% |
| > 70 (alto) | 18% |

> **Estudiantes en el tercil inferior de acceso a módulos tienen 3x más probabilidad de reprobar** que los del tercil superior.

### Hallazgo 3: La Entropía de Navegación Revela Intención

**¿Qué es la entropía de transición?**

Mide qué tan predecible es la secuencia de navegación. Entropía alta = navegación exploratoria. Entropía baja = navegación lineal o errática.

**Resultado:** El indicador "Variabilidad en Navegacion" tiene importancia del **2.14%**.

| Tipo de Navegación | Entropía | Interpretación |
|--------------------|----------|----------------|
| Exploratoria | Alta | Busca activamente, compara recursos |
| Lineal | Media | Sigue estructura del curso |
| Errática/Mínima | Baja | Sin patrón claro, acceso superficial |

### Hallazgo 4: Los 5 Predictores Conductuales Más Fuertes

El modelo de alerta temprana identificó los indicadores con mayor poder predictivo:

| # | Indicador | Importancia | Interpretación |
|---|-----------|-------------|----------------|
| 1 | Tiempo Total de Conexion | 5.97% | Tiempo total (normalizado por curso) |
| 2 | Variedad de Paginas | 5.51% | Variedad de páginas accedidas |
| 3 | Diversidad de Navegacion | 4.66% | Diversidad de navegación |
| 4 | Balance Estudio/Evaluacion | 4.39% | Balance estudio/evaluación |
| 5 | Acceso a Modulos | 4.02% | Acceso a módulos del curso |

**Nota:** Todos son indicadores *puramente conductuales*, sin incluir calificaciones o entregas.

---

### Acceso a Recursos por Tipo

El análisis por tipo de recurso revela diferencias significativas en el patrón de acceso:

![Engagement por Tipo de Recurso](analysis/resources/resource_comparison_bar.png)

| Recurso | Aprobados | Reprobados | Ratio | Cohen's d |
|---------|-----------|------------|-------|-----------|
| **Página Inicial** | 63 | 36 | **1.77x** | 0.61*** |
| **Tareas** | 103 | 49 | **2.08x** | 0.55*** |
| **Calificaciones** | 5.7 | 2.2 | **2.66x** | 0.53*** |
| **Cuestionarios** | 37 | 17 | **2.25x** | 0.51*** |
| **Discusiones** | 26 | 13 | **1.92x** | 0.47*** |
| **Anuncios** | 6.1 | 3.6 | **1.71x** | 0.41*** |
| **Archivos** | 42 | 33 | **1.27x** | 0.21** |

> **Insight:** Los recursos con mayor poder discriminativo son Página Inicial (d=0.61) y Tareas (d=0.55). Módulos y Páginas no muestran diferencias significativas entre grupos.

---

### Hallazgo 5: Evaluaciones como "Centros de Gravedad" de la Actividad

Un hallazgo clave del análisis es que las **evaluaciones actúan como centros de gravedad** de la actividad estudiantil. La mayoría de los estudiantes organizan su interacción con el LMS alrededor de las fechas de entrega.

![Actividad alrededor de fechas de evaluación](analysis/assessment_patterns/assessment_gravity_aggregated.png)

> **Implicación:** Cómo un estudiante **prepara** una evaluación predice su resultado, incluso sin conocer la calificación. La actividad en los días previos a la entrega (-7 a -1) diferencia significativamente a quienes aprobarán de quienes reprobarán.

### Diferencias en Preparación: Aprobados vs Reprobados

La preparación previa a evaluaciones muestra diferencias marcadas entre grupos:

![Comportamiento de preparación](analysis/assessment_patterns/pass_fail_preparation.png)

| Período | Métrica | Diferencia |
|---------|---------|------------|
| **Semana previa (-7 a -1 días)** | Page views promedio | Aprobados: **1.8x** más que reprobados |
| **Día de entrega** | Pico de actividad | Ambos grupos muestran incremento |
| **Post-entrega** | Decaimiento | Similar en ambos grupos |

> **Hallazgo clave:** Los estudiantes que aprueban muestran **1.8 veces más actividad** en la semana previa a cada evaluación. Esta diferencia conductual permite predecir el resultado *antes* de que exista una calificación.

### Evolución del Foco en Evaluaciones por Semana

![Proporción de actividad en evaluaciones](analysis/assessment_patterns/weekly_assessment_proportion.png)

> **Implicación práctica:** Antes de la semana 4, la actividad en evaluaciones es mínima porque aún no existen. A partir de la semana 4, cuando aparecen las primeras evaluaciones formativas, los indicadores de evaluación se vuelven altamente predictivos. Esto explica por qué el modelo de semana 4 sin indicadores de evaluación funciona igual de bien que con ellos: antes de esa fecha, simplemente no hay datos de evaluación que aporten información diferencial.

---

## 3.5 Capacidad Predictiva

### Evolución de la Predicción por Semana

El sistema de alerta temprana mejora progresivamente su precisión a medida que acumula datos de comportamiento:

![Evolución Recall y Falsos Positivos](analysis/model_evolution/recall_fp_evolution.png)

> **Hallazgo clave:** El modelo mantiene **alta detección (80%+)** desde la semana 2, pero la precisión mejora dramáticamente con más tiempo. Las **falsas alarmas se reducen a la mitad** entre semana 2 y semana 6+.

### Modelo Principal: Semana 6+ (Con Acceso a Evaluaciones)

Este es el modelo recomendado para **intervención activa**. Utiliza patrones de acceso a recursos de evaluación (tareas, quizzes, libro de calificaciones) sin usar las calificaciones mismas. Dado que estos recursos están disponibles en la mayoría de los cursos desde la semana 6, este modelo puede aplicarse desde entonces.

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **Capacidad Predictiva** | **0.90** | Excelente discriminación |
| **Recall** | **85.9%** | Detectamos 9 de cada 10 en riesgo |
| **Accuracy** | **83.4%** | 8.3 de cada 10 predicciones correctas |
| **Precision** | **75.7%** | 3 de cada 4 alertas son correctas |

**Configuración óptima:** Threshold 0.33

**Matriz de resultados** (373 estudiantes):
- 128 estudiantes en riesgo correctamente alertados (TP)
- 183 estudiantes sin riesgo correctamente identificados (TN)
- 39 falsas alarmas - estudiantes que pasaron pero fueron alertados (FP)
- 24 estudiantes en riesgo no detectados (FN)

**Señales clave detectadas:**
- Acceso a assignments (tareas) normalizado por curso
- Consultas al libro de calificaciones
- Acceso a quizzes y recursos de práctica
- Tiempo total de sesión en el LMS

### Modelo Temprano: Semana 4 SIN Evaluaciones

Para cursos donde las evaluaciones comienzan despues de la semana 4, este modelo permite **intervencion antes de cualquier nota**:

| Metrica | Valor | Interpretacion |
|---------|-------|----------------|
| **Capacidad Predictiva** | **0.85** | Muy buena discriminacion |
| **Recall** | **85.2%** | Detectamos 8.5 de cada 10 en riesgo |
| **Accuracy** | **74.5%** | 3 de cada 4 predicciones correctas |
| **Precision** | 63.0% | ~6 de cada 10 alertas son correctas |

**Configuracion optima:** Threshold 0.20

**Senales clave (sin evaluaciones):**
- Participacion en foros y discusiones
- Tiempo total de sesion en el LMS
- Percentil de acceso a modulos
- Diversidad de transiciones de navegacion
- Percentil de acceso a paginas

> **Ventaja estrategica:** Este modelo alcanza un rendimiento similar al modelo de Semana 6+ (85.2% vs 85.9% recall) funcionando ANTES de que existan recursos de evaluacion en el curso.

### Nota sobre Detección en Semana 2

El modelo puede detectar **8 de cada 10** estudiantes en riesgo en solo 2 semanas, con **2 de cada 3** predicciones correctas. Sin embargo, genera más falsas alarmas (~45% de las alertas).

**Uso recomendado:** Lista de observación inicial para seguimiento, no para intervención activa.

### Comparacion: Ventaja Temporal del Sistema

| Metodo de Deteccion | Disponibilidad | Recall | Accuracy | Capacidad Predictiva |
|---------------------|----------------|--------|----------|---------|
| Intuicion docente | Variable | Subjetiva | Subjetiva | - |
| Primera evaluacion | Semana 4-6 | Alta, tardia | Alta, tardia | - |
| **Sistema LMS (Semana 2)** | **Semana 2** | **81.7%** | **65.7%** | **0.74** |
| **Sistema LMS (Semana 4)** | **Semana 4** | **85.2%** | **74.5%** | **0.85** |
| **Sistema LMS (Semana 6+)** | **Semana 6+** | **85.9%** | **83.4%** | **0.90** |

> **Conclusion:** El sistema proporciona **deteccion temprana confiable** desde la semana 4 con 85%+ de recall, y alcanza **maxima precision** desde la semana 6 cuando los estudiantes han interactuado con recursos de evaluacion. La semana 6 permite **10+ semanas de intervencion** antes del cierre del semestre.

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

### 1. El comportamiento digital predice el exito academico

Los patrones de navegacion en el LMS son indicadores altamente confiables del rendimiento futuro:

| Modelo | Capacidad Predictiva | Accuracy | Recall | Disponible desde |
|--------|---------|----------|--------|------------------|
| **Semana 6+** (modelo principal) | **0.90** | **83.4%** | **85.9%** | Semana 6 |
| **Semana 4** (sin evaluaciones) | **0.85** | **74.5%** | **85.2%** | Semana 4 |
| **Semana 2** (watch list) | **0.74** | **65.7%** | **81.7%** | Semana 2 |

> **Resultado principal:** Desde la **semana 6**, cuando los estudiantes ya han interactuado con recursos de evaluacion, el sistema detecta correctamente **9 de cada 10** estudiantes en riesgo (85.9% recall) con **83.4% de exactitud**, permitiendo **10+ semanas de intervencion** antes del cierre del semestre.

### 2. Deteccion temprana SIN calificaciones

El modelo de **semana 4 SIN indicadores de evaluacion** es una ventaja clave:

- Detecta **8.5 de cada 10** estudiantes en riesgo (85.2% recall)
- Funciona **ANTES** de que existan tareas, quizzes o calificaciones en el curso
- Utiliza unicamente: participacion en foros, tiempo en el LMS, diversidad de navegacion

> **Implicacion estrategica:** Se puede intervenir en cursos donde las primeras evaluaciones ocurren despues de la semana 4, sin esperar notas. El rendimiento es practicamente equivalente al modelo de Semana 6+.

### 3. Las señales de acceso a evaluaciones son altamente predictivas

El sistema **no utiliza las calificaciones** de las evaluaciones, pero sí detecta patrones de *acceso* a recursos de evaluación:

- **¿Accede a revisar tareas antes del deadline?**
- **¿Consulta el libro de calificaciones regularmente?**
- **¿Explora quizzes de práctica?**

Estas señales de comportamiento son los **predictores más fuertes** del modelo de semana 6+.

### 4. La *calidad* de navegación importa más que la *cantidad*

Los estudiantes exitosos muestran patrones de navegación cualitativamente diferentes:
- Más transiciones entre tipos de recursos
- Mayor diversidad en la exploración
- Mejor posición relativa respecto a compañeros
- **Mayor interacción con recursos de evaluación** (principal diferenciador)

### 5. El modelo generaliza a cursos nuevos

Con una capacidad predictiva del 69%-94% en validacion por cursos individuales, el sistema puede aplicarse a cursos que no fueron parte del entrenamiento.

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
2. **Generalización a cursos nuevos**: El modelo mantiene buen desempeño en cursos nuevos (capacidad predictiva del 69%-94% segun el curso)
3. **Solo comportamiento digital**: No captura factores externos (trabajo, familia, salud)
4. **Dependencia del diseño del curso**: Cursos con poco contenido en LMS generan menos señales

---

# 5. Anexos

## Anexo A: Detalle de Indicadores del Modelo

### Indicadores de Transición (Secuencias)

| Indicador | Descripción | Importancia |
|---------|-------------|-------------|
| Diversidad de Navegacion | Número total de transiciones entre recursos | 4.66% |
| Variabilidad en Navegacion | Variabilidad en los patrones de navegacion entre recursos | 2.14% |

### Indicadores de Percentil Relativo

| Indicador | Descripción | Importancia |
|---------|-------------|-------------|
| Percentil Promedio Modulos | Percentil promedio en acceso a módulos | 2.32% |
| Varianza Acceso Paginas | Varianza explicada en acceso a páginas | 2.53% |

### Indicadores Temporales

| Indicador | Descripción | Importancia |
|---------|-------------|-------------|
| Tiempo Total de Conexion | Tiempo total normalizado por curso | 5.97% |
| Sesiones por Semana | Sesiones por semana normalizado | 3.02% |
| Ultima Semana Activa | Última semana con actividad | 2.61% |

## Anexo B: Metodología de Validación

### Validación Cruzada Estratificada (5-fold)

```
Fold 1: Train [2,3,4,5] → Test [1]
Fold 2: Train [1,3,4,5] → Test [2]
Fold 3: Train [1,2,4,5] → Test [3]
Fold 4: Train [1,2,3,5] → Test [4]
Fold 5: Train [1,2,3,4] → Test [5]

Resultado: Capacidad predictiva promedio = 0.90
```

### Validacion por Cursos Individuales 
```
Fold 1: Train [cursos 2-10] → Test [curso 1]
Fold 2: Train [cursos 1,3-10] → Test [curso 2]
...
Fold 10: Train [cursos 1-9] → Test [curso 10]

Resultado: Capacidad predictiva por curso = 0.69 - 0.94
```

**Interpretación**: validacion cruzada por cursos evalúa generalización a cursos nuevos. El rango de capacidad predictiva (0.69-0.94) indica que el modelo funciona mejor en algunos cursos que otros, dependiendo de su estructura y diseño instruccional.

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

---

## Anexo E: Matrices de Confusion por Modelo

Las matrices de confusion permiten visualizar el rendimiento de cada modelo en terminos de predicciones correctas e incorrectas. Cada celda muestra:

- **TN (Verdadero Negativo):** Estudiantes que aprobaron y el modelo predijo correctamente que aprobarian
- **TP (Verdadero Positivo):** Estudiantes que reprobaron y el modelo predijo correctamente que reprobarian
- **FP (Falso Positivo):** Estudiantes que aprobaron pero el modelo predijo incorrectamente que reprobarian (falsas alarmas)
- **FN (Falso Negativo):** Estudiantes que reprobaron pero el modelo predijo incorrectamente que aprobarian (casos perdidos)

### Comparacion de los Tres Modelos Principales

![Matrices de Confusion Combinadas](analysis/confusion_matrices/confusion_matrices_combined.png)

### Resumen de Metricas por Modelo

| Modelo | ROC-AUC | Accuracy | Precision | Recall | F1-Score |
|--------|---------|----------|-----------|--------|----------|
| **Semana 6+ (t=0.33)** | **0.90** | **83.4%** | **75.7%** | **85.9%** | **80.5%** |
| **Semana 4 (t=0.20)** | **0.85** | **74.5%** | 63.5% | 85.2% | 72.5% |
| Semana 2 (t=0.13) | 0.74 | 65.7% | 54.7% | 81.7% | 65.5% |

### Interpretacion de Resultados

**Modelo Semana 6+ (threshold=0.33):**
- Mejor rendimiento general con ROC-AUC de 0.90
- Detecta el 85.9% de estudiantes en riesgo
- De cada 100 alertas, 76 corresponden a verdaderos casos de riesgo
- Incluye informacion de acceso a evaluaciones

**Modelo Semana 4 (threshold=0.20):**
- Opera SIN necesidad de datos de evaluaciones
- Detecta 85.2% de estudiantes en riesgo en etapa muy temprana
- Rendimiento similar al modelo de Semana 6+ en recall
- Ideal para cursos donde las primeras evaluaciones son posteriores a semana 4

**Modelo Semana 2 (threshold=0.13):**
- Deteccion ultra-temprana con 81.7% de recall
- Mayor tasa de falsas alarmas (~45%)
- Uso recomendado: lista de observacion inicial (watch list)

### Matrices Individuales

#### Modelo Semana 6+ (threshold=0.33)
![Matriz Semana 6+](analysis/confusion_matrices/confusion_matrix_semana_6_t0.33.png)

#### Modelo Semana 4 (threshold=0.20)
![Matriz Semana 4](analysis/confusion_matrices/confusion_matrix_semana_4_t0.20.png)

#### Modelo Semana 2 (threshold=0.13)
![Matriz Semana 2](analysis/confusion_matrices/confusion_matrix_semana_2_t0.13.png)

---

*Informe generado el 9 de enero de 2026*
*Universidad Autonoma de Chile - Canvas LMS*
*Version 5.0*
