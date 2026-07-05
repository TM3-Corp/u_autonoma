# Informe: Impacto de Recursos LMS en Rendimiento Académico

## Universidad Autónoma de Chile - Canvas LMS

**Fecha:** 30 de diciembre de 2025
**Ambiente:** TEST (uautonoma.test.instructure.com)

---

## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Metodología](#2-metodología)
3. [Análisis por Tipo de Recurso](#3-análisis-por-tipo-de-recurso)
4. [Comparación: Aprobados vs Reprobados](#4-comparación-aprobados-vs-reprobados)
5. [Factores de Riesgo por Tipo de Recurso](#5-factores-de-riesgo-por-tipo-de-recurso)
6. [Conclusiones y Recomendaciones](#6-conclusiones-y-recomendaciones)
7. [Anexos](#7-anexos)

---

# 1. Resumen Ejecutivo

## El Hallazgo Principal

> **Los estudiantes con bajo engagement en cuestionarios (quizzes) tienen 2.57 veces más probabilidad de reprobar.**

Este análisis examina cómo la interacción con diferentes tipos de recursos del LMS se relaciona con el rendimiento académico, comparando estudiantes que aprobaron versus aquellos que reprobaron.

---

### Números Clave del Estudio

| Métrica | Valor |
|---------|-------|
| **Estudiantes analizados** | 361 |
| **Cursos evaluados** | 10 |
| **Tasa de aprobación** | 60.7% |
| **Tipos de recurso analizados** | 9 |
| **Factores de riesgo significativos** | 7 |

---

### Top 3 Factores de Riesgo


| **1. Cuestionarios** |
|---|
| Bajo engagement: **56.3%** tasa de fracaso |
| Alto engagement: **21.9%** tasa de fracaso |
| **Ratio de riesgo: 2.57x** |


| **2. Calificaciones** |
|---|
| Bajo engagement: **52.9%** tasa de fracaso |
| Alto engagement: **21.3%** tasa de fracaso |
| **Ratio de riesgo: 2.49x** |


| **3. Página Inicial** |
|---|
| Bajo engagement: **50.5%** tasa de fracaso |
| Alto engagement: **27.9%** tasa de fracaso |
| **Ratio de riesgo: 1.81x** |


---

# 2. Metodología

## 2.1 Fuentes de Datos

Los datos se extrajeron de Canvas LMS utilizando la API de Page Views, que registra cada interacción del estudiante con la plataforma.

### Tipos de Recursos Analizados

| Tipo | Descripción | Patrón URL |
|------|-------------|------------|
| **Cuestionarios** | Evaluaciones formativas y sumativas | `/quizzes/{id}` |
| **Tareas** | Entregas y actividades calificadas | `/assignments/{id}` |
| **Foros** | Discusiones y participación | `/discussion_topics/{id}` |
| **Módulos** | Navegación por unidades | `/modules/{id}` |
| **Archivos** | Descarga de materiales | `/files/{id}` |
| **Páginas** | Contenido informativo | `/pages/{slug}` |
| **Calificaciones** | Consulta de notas | `/grades` |
| **Anuncios** | Comunicaciones del curso | `/announcements` |
| **Página Inicial** | Home del curso | `/courses/{id}` |

## 2.2 Definición de Aprobado/Reprobado

- **Aprobado:** Nota final ≥ 57%
- **Reprobado:** Nota final < 57%

## 2.3 Tests Estadísticos

- **Mann-Whitney U:** Comparación no paramétrica entre grupos
- **Tamaño del efecto (Cohen's d):** Magnitud práctica de las diferencias
- **Correlación de Pearson:** Relación lineal con la nota final

---

# 3. Análisis por Tipo de Recurso

## 3.1 Comparación Visual

![Comparación por Recurso](visualizations/resources/resource_comparison_bar.png)

## 3.2 Distribuciones

![Box Plots](visualizations/resources/resource_boxplots.png)

## 3.3 Resumen por Recurso

| Recurso | Media Aprobados | Media Reprobados | p-value | Cohen's d | Significativo |
|---------|-----------------|------------------|---------|-----------|---------------|
| Cuestionarios | 37.2 | 16.5 | 0.0000 | 0.51 | **Sí** |
| Tareas | 102.8 | 49.5 | 0.0000 | 0.55 | **Sí** |
| Foros de Discusión | 25.9 | 13.5 | 0.0000 | 0.47 | **Sí** |
| Página Inicial | 63.0 | 35.6 | 0.0000 | 0.61 | **Sí** |
| Calificaciones | 5.7 | 2.2 | 0.0000 | 0.53 | **Sí** |
| Archivos | 42.4 | 33.3 | 0.0088 | 0.21 | **Sí** |
| Módulos | 58.0 | 59.3 | 0.5825 | -0.02 | No |
| Anuncios | 6.1 | 3.6 | 0.0000 | 0.41 | **Sí** |
| Páginas | 15.3 | 12.8 | 0.0607 | 0.08 | No |


### Interpretación de Cohen's d

| Valor | Interpretación |
|-------|----------------|
| < 0.2 | Negligible |
| 0.2 - 0.5 | Pequeño |
| 0.5 - 0.8 | Medio |
| > 0.8 | Grande |

---

# 4. Comparación: Aprobados vs Reprobados

## 4.1 Tamaño del Efecto

![Tamaños de Efecto](visualizations/resources/effect_sizes.png)

## 4.2 Hallazgos Clave


### Página Inicial

> Los estudiantes **aprobados** acceden en promedio **27.4 veces más** a página inicial que los reprobados.
>
> - Media Aprobados: **63.0**
> - Media Reprobados: **35.6**
> - Tamaño del efecto: **d = 0.61** (medium)


### Tareas

> Los estudiantes **aprobados** acceden en promedio **53.3 veces más** a tareas que los reprobados.
>
> - Media Aprobados: **102.8**
> - Media Reprobados: **49.5**
> - Tamaño del efecto: **d = 0.55** (medium)


### Calificaciones

> Los estudiantes **aprobados** acceden en promedio **3.6 veces más** a calificaciones que los reprobados.
>
> - Media Aprobados: **5.7**
> - Media Reprobados: **2.2**
> - Tamaño del efecto: **d = 0.53** (medium)


### Cuestionarios

> Los estudiantes **aprobados** acceden en promedio **20.6 veces más** a cuestionarios que los reprobados.
>
> - Media Aprobados: **37.2**
> - Media Reprobados: **16.5**
> - Tamaño del efecto: **d = 0.51** (medium)


### Foros de Discusión

> Los estudiantes **aprobados** acceden en promedio **12.4 veces más** a foros de discusión que los reprobados.
>
> - Media Aprobados: **25.9**
> - Media Reprobados: **13.5**
> - Tamaño del efecto: **d = 0.47** (small)


## 4.3 Correlaciones con Nota Final

![Matriz de Correlación](visualizations/resources/resource_correlation_heatmap.png)

---

# 5. Factores de Riesgo por Tipo de Recurso

## 5.1 Visualización de Riesgos

![Factores de Riesgo](visualizations/resources/risk_factors_bar.png)

## 5.2 Tabla de Factores de Riesgo

| Recurso | Falla (Bajo Eng.) | Falla (Alto Eng.) | Ratio Riesgo | Significativo |
|---------|-------------------|-------------------|--------------|---------------|
| Cuestionarios | 56.3% | 21.9% | **2.57x** | **Sí** |
| Calificaciones | 52.9% | 21.3% | **2.49x** | **Sí** |
| Página Inicial | 50.5% | 27.9% | **1.81x** | **Sí** |
| Anuncios | 47.7% | 29.0% | **1.65x** | **Sí** |
| Tareas | 48.6% | 29.8% | **1.63x** | **Sí** |
| Foros de Discusión | 47.2% | 30.4% | **1.55x** | **Sí** |
| Páginas | 45.6% | 32.1% | **1.42x** | **Sí** |


## 5.3 Umbrales de Alerta

```
┌─────────────────────────────────────────────────────────────┐
│             UMBRALES DE ALERTA POR RECURSO                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ⚠️  Cuestionarios < mediana  → 2.57x más riesgo de reprobar│
│                                                             │
│  ⚠️  Calificaciones < mediana → 2.49x más riesgo            │
│                                                             │
│  ⚠️  Página Inicial < mediana → 1.81x más riesgo            │
│                                                             │
│  ⚠️  Anuncios < mediana       → 1.65x más riesgo            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

# 6. Conclusiones y Recomendaciones

## 6.1 Conclusiones Principales

### 1. Los cuestionarios son el recurso más predictivo
El engagement con cuestionarios muestra la mayor diferencia entre aprobados y reprobados. Los estudiantes que no interactúan con quizzes tienen **2.57 veces más probabilidad de reprobar**.

### 2. La consulta de calificaciones es un indicador de compromiso
Los estudiantes aprobados revisan sus notas **2.6 veces más frecuentemente**. Esto sugiere un mayor automonitoreo del progreso académico.

### 3. La página inicial del curso predice engagement general
Acceder frecuentemente al home del curso indica compromiso general. Los aprobados visitan la página inicial **1.77 veces más** que los reprobados.

### 4. Los foros de discusión muestran engagement activo
La participación en foros tiene un efecto medio (d=0.47). Los cursos con foros activos podrían beneficiarse de monitorear este indicador.

## 6.2 Recomendaciones

### Para el Sistema de Alerta Temprana

| Recurso | Umbral de Alerta | Acción Sugerida |
|---------|------------------|-----------------|
| **Cuestionarios** | < 15 accesos/semestre | Alerta crítica inmediata |
| **Calificaciones** | < 2 consultas/semana | Notificación de seguimiento |
| **Página Inicial** | < 30 visitas/semestre | Revisar engagement general |
| **Foros** | 0 participaciones | Incentivo de participación |

### Para Diseño Instruccional

1. **Incrementar uso de cuestionarios formativos** - Son el mejor predictor
2. **Facilitar acceso a calificaciones** - Promueve automonitoreo
3. **Activar foros de discusión** - En cursos sin foros, considerar añadirlos
4. **Estructurar página inicial** - Como hub central del curso

---

# 7. Anexos

## Anexo A: Metodología Estadística

### Test Mann-Whitney U
Prueba no paramétrica que compara las distribuciones de dos grupos independientes. Se utilizó porque los datos de engagement típicamente no siguen una distribución normal.

### Cohen's d
Medida estandarizada del tamaño del efecto:
```
d = (Media_1 - Media_2) / Desviación_Estándar_Agrupada
```

### Ratio de Riesgo
```
Ratio = Tasa_Fracaso_Bajo_Engagement / Tasa_Fracaso_Alto_Engagement
```

## Anexo B: Limitaciones

1. **Datos de un solo semestre** - Resultados pueden variar entre períodos
2. **Cursos heterogéneos** - Diferente uso de recursos según diseño instruccional
3. **Correlación ≠ Causalidad** - El engagement puede ser efecto, no causa del rendimiento

---

*Informe generado automáticamente el {datetime.now().strftime('%d de %B de %Y').replace('January', 'enero').replace('February', 'febrero').replace('March', 'marzo').replace('April', 'abril').replace('May', 'mayo').replace('June', 'junio').replace('July', 'julio').replace('August', 'agosto').replace('September', 'septiembre').replace('October', 'octubre').replace('November', 'noviembre').replace('December', 'diciembre')}*
*Universidad Autónoma de Chile - Canvas LMS*
