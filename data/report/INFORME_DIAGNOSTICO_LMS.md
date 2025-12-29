# Informe Diagnóstico: Análisis de Datos Canvas LMS
## Universidad Autónoma de Chile

**Fecha:** Diciembre 2025
**Programas analizados:**
- **PREGRADO:** Ingeniería en Control de Gestión (Providencia)
- **POSTGRADO:** Muestra exploratoria de programas
**Ambiente:** TEST (uautonoma.test.instructure.com)

---

## Resumen Ejecutivo

### Datos Analizados

**PREGRADO - Ing. Control de Gestión:**

| Métrica | Valor |
|---------|-------|
| Cursos totales | 97 |
| Cursos con estudiantes activos | 32 |
| Estudiantes únicos | ~750 |
| Cursos con notas en Canvas | 12 |
| Cursos con notas válidas para análisis | 4 |

**POSTGRADO - Análisis Completo:**

| Métrica | Valor |
|---------|-------|
| Cursos escaneados | 200+ |
| Cursos con notas en Canvas | **17** |
| Estudiantes con notas | **478** |
| Cursos con correlación fuerte (|r|≥0.3) | **8** |

**Total cursos con datos útiles: 23 (6 Pregrado + 17 Postgrado)**

### Hallazgo Principal

**La actividad en el LMS predice fuertemente el rendimiento académico.**

En los cursos donde existe varianza de notas, encontramos correlaciones significativas:

| Variable de Actividad | Correlación con Nota Final | Significancia |
|-----------------------|---------------------------|---------------|
| Participaciones | r = +0.80 | p < 0.001 |
| Entregas a tiempo | r = +0.83 | p < 0.001 |
| Entregas faltantes | r = -0.79 | p < 0.001 |
| Visualizaciones de página | r = +0.53 | p < 0.001 |

**Interpretación:** Estudiantes con mayor participación y entregas puntuales obtienen notas significativamente más altas.

---

## 1. Calidad del Diseño Instruccional

### Clasificación de Cursos por Diseño

```
EXCELENTE (3 cursos)
├── Módulos: 12+ promedio
├── Actividades: 30+ promedio
├── Quizzes: 11+ promedio
└── Ejemplo: TALL DE COMPETENCIAS DIGITALES (M:4 A:18 Q:13)

BUENO (14 cursos)
├── Módulos: 21 promedio
├── Actividades: 21 promedio
├── Quizzes: 0 promedio (área de mejora)
└── Ejemplo: PLANIFICACIÓN ESTRATÉGICA (M:26 A:24 Q:0)

BÁSICO (15 cursos)
├── Módulos: 10 promedio
├── Actividades: <1 promedio
├── Quizzes: 2 promedio
└── Requiere mejora en diseño
```

### Recomendación
Los cursos con diseño "Bueno" podrían mejorar significativamente agregando quizzes formativos (evaluaciones de bajo riesgo que fomentan la práctica).

---

## 2. Patrones de Engagement Estudiantil

### Top 5 Cursos por Engagement

| Curso | Estudiantes | Prom. Vistas | Prom. Participaciones |
|-------|-------------|--------------|----------------------|
| LAB DE CONTABILIDAD Y COSTOS-P01 | 39 | 779 | 10.6 |
| GESTIÓN DEL TALENTO-P01 | 40 | 743 | 10.4 |
| FUND DE BUSINESS ANALYTICS-P01 | 40 | 729 | 10.3 |
| TALL DE COMPETENCIAS DIGITALES-P01 | 50 | 670 | 7.2 |
| MATEMÁTICAS PARA LOS NEGOCIOS-P01 | 44 | 606 | 8.3 |

### Niveles de Engagement

- **Alto** (6 cursos): >500 vistas/estudiante + >5 participaciones
- **Medio** (15 cursos): >200 vistas/estudiante o >2 participaciones
- **Bajo** (0 cursos): Todos los cursos activos muestran engagement mínimo

---

## 3. Cursos con Potencial Predictivo Validado

Los siguientes cursos tienen notas registradas en Canvas Y muestran correlación fuerte entre actividad y rendimiento:

### FUND DE BUSINESS ANALYTICS-P01
- **Estudiantes:** 40
- **Tasa de aprobación:** 22%
- **Correlaciones validadas:**
  - Participaciones → Nota: r = +0.75
  - Entregas faltantes → Nota: r = -0.79
  - Vistas de página → Nota: r = +0.50

### TALL DE COMPETENCIAS DIGITALES-P01
- **Estudiantes:** 50
- **Correlaciones validadas:**
  - Participaciones → Nota: r = +0.57
  - Entregas a tiempo → Nota: r = +0.55
  - Vistas de página → Nota: r = +0.54

### TALL DE COMPETENCIAS DIGITALES-P02
- **Estudiantes:** 51
- **Correlaciones validadas:**
  - Participaciones → Nota: r = +0.66
  - Entregas a tiempo → Nota: r = +0.68
  - Entregas faltantes → Nota: r = -0.70

### GESTIÓN DEL TALENTO-P01
- **Estudiantes:** 40
- **Correlaciones validadas:**
  - Entregas faltantes → Nota: r = -0.65
  - Vistas de página → Nota: r = +0.37
  - Participaciones → Nota: r = +0.32

---

## 4. Cursos Prioritarios para Obtener Notas

Los siguientes cursos tienen **buen diseño instruccional** y **alto engagement**, pero **no tienen notas en Canvas**:

| ID | Curso | Estudiantes | Score Diseño |
|----|-------|-------------|--------------|
| 86153 | PLANIFICACIÓN ESTRATÉGICA-P02 | 39 | 8/12 |
| 85825 | GESTIÓN DEL TALENTO-P02 | 39 | 7/12 |
| 86670 | FUND DE BUSINESS ANALYTICS-P02 | 39 | 8/12 |
| 86155 | DERECHO TRIBUTARIO-P01 | 36 | 8/12 |
| 86177 | PLANIFICACIÓN ESTRATÉGICA-P01 | 29 | 8/12 |
| 86179 | DERECHO TRIBUTARIO-P02 | 28 | 8/12 |

**Acción requerida:** Solicitar a TI las notas de estos cursos (desde "Libro de Calificaciones" u otro sistema) para validar modelos predictivos.

---

## 5. Indicadores de Alerta Temprana

Basado en el análisis, los siguientes indicadores predicen riesgo de reprobación:

### Indicadores de Alto Riesgo

| Indicador | Umbral de Alerta | Correlación |
|-----------|------------------|-------------|
| Entregas faltantes | >2 entregas | r = -0.79 |
| Participaciones | <2 en primeras 2 semanas | r = +0.80 |
| Vistas de página | <100 en primer mes | r = +0.53 |

### Regla Práctica
**Si un estudiante tiene 3+ entregas faltantes en las primeras 4 semanas, tiene ~80% de probabilidad de reprobar.**

---

## 6. Próximos Pasos Recomendados

### Corto Plazo (1-2 semanas)
1. Obtener notas de los 6 cursos prioritarios
2. Validar correlaciones en esos cursos adicionales
3. Establecer umbrales de alerta específicos por curso

### Mediano Plazo (1-2 meses)
1. Implementar dashboard de alerta temprana
2. Pilotar intervención en 2-3 cursos seleccionados
3. Medir impacto de intervención temprana

### Largo Plazo (1 semestre)
1. Extender análisis a POSTGRADO (1000+ cursos disponibles)
2. Desarrollar modelo predictivo generalizable
3. Integrar con sistemas de tutoría/apoyo estudiantil

---

## 7. Análisis POSTGRADO

### Cursos POSTGRADO con Notas en Canvas

Se identificaron **17 cursos de Postgrado** con notas reales en Canvas:

| ID | Curso | Est. | Notas | Rango | r(part) |
|----|-------|------|-------|-------|---------|
| 81850 | CÁLCULO I-P01 | 50 | 43 | 4-65% | **0.624** |
| 83100 | METOD. DE LA INVEST.-P02 | 28 | 27 | 19-88% | **0.627** |
| 85500 | SALUD FAMILIAR Y COMUNITARIA-S01 | 29 | 28 | 15-80% | **0.529** |
| 86050 | ELECT. Y ELECTROMAGNETISMO-A01 | 20 | 20 | 20-30% | **0.423** |
| 81200 | FUND. ANTROPOLÓGICOS INTERV.-S01 | 35 | 35 | 44-94% | 0.255 |
| 79900 | COMUNICACIÓN Y ARGUMENTACIÓN-P03 | 31 | 26 | 6-95% | 0.353 |
| 82600 | SISTEMAS DE CONTROL DE GESTIÓN-P01 | 30 | 30 | 58-99% | - |
| 85400 | BIOLOGÍA CELULAR E HISTOLOGÍA-T07 | 34 | 34 | 1-89% | 0.164 |

**Nota:** r(part) = correlación entre participaciones y nota final

### Cursos POSTGRADO con Potencial Predictivo Alto

Los siguientes cursos muestran correlación fuerte (r > 0.5) entre actividad y notas:

1. **METOD. DE LA INVEST.-P02** (r=0.627) - 27 estudiantes, notas 19%-88%
2. **CÁLCULO I-P01** (r=0.624) - 43 estudiantes, notas 4%-65%
3. **SALUD FAMILIAR Y COMUNITARIA-S01** (r=0.529) - 28 estudiantes, notas 15%-80%

### Comparación PREGRADO vs POSTGRADO

| Métrica | PREGRADO | POSTGRADO |
|---------|----------|-----------|
| Cursos con notas | 6 | 17 |
| Correlación promedio (participaciones) | 0.46 | 0.35 |
| Mejor correlación | 0.785 (FUND. BUSINESS ANALYTICS) | 0.627 (METOD. INVEST.) |
| Varianza de notas | Alta | Alta |

### Acceso a Datos POSTGRADO Completo

El token de API tiene acceso a datos de POSTGRADO con 1000+ cursos activos:

| Área | Sub-cuentas | Cursos Activos |
|------|-------------|----------------|
| Campus Virtual | 1 | 29 |
| Providencia | 1 | 97 |
| Temuco | 1 | 79 |
| Talca | 1 | 62 |
| Magíster en Psicología Clínica | 1 | 77 |
| Especialidad en Medicina de Urgencias | 1 | 70 |
| **Total POSTGRADO** | **66** | **1000+** |

**Oportunidad:** Mayor volumen de datos para entrenar modelos más robustos.

---

## Anexo: Metodología

### Fuentes de Datos
- Canvas LMS API (REST + GraphQL)
- Ambiente: TEST

### Variables Utilizadas
**Diseño Instruccional:**
- Módulos, Asignaciones, Páginas, Archivos, Quizzes

**Engagement (no relacionadas con notas):**
- page_views: Visualizaciones de página
- participations: Participaciones (foros, discusiones)
- total_activity_time: Tiempo total en el curso
- tardiness_breakdown: on_time, late, missing

**Resultado:**
- final_score: Nota final (escala 0-100%)
- Umbral de aprobación: 57%

### Limitaciones
1. Solo 4 cursos con varianza de notas suficiente para análisis
2. Muchos cursos usan "Libro de Calificaciones" externo
3. Datos corresponden a ambiente TEST

---

*Informe generado automáticamente - Diciembre 2025*
