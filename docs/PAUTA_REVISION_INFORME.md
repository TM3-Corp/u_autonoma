# Pauta de Revision de Informes para Autoridades

## Audiencia Target

- Directores de carrera
- Decanos
- Autoridades universitarias sin formacion tecnica en data science

---

## Criterios de Redaccion

### 1. No referenciar el proceso interno de desarrollo

El lector externo NO conoce la cronologia del proyecto. No sabe que indicadores son "nuevos" o "antiguos" porque desconoce el proceso de data science. Solo ve el resultado final.

- "Nueva feature", "Indicador recientemente agregado", "Version actualizada"
- "Despues de varias iteraciones", "En la version anterior"
- Presentar todo como el estado actual, sin referencias temporales internas

### 2. Evitar jerga tecnica de data science

- "feature", "pipeline", "modelo XGBoost", "hyperparametros"
- "indicador", "proceso de analisis", "modelo predictivo", "configuracion del modelo"

### 3. No dirigirse explicitamente al receptor

- "Mensaje para Autoridades:", "Nota para el lector:"
- Escribir directamente en tono apropiado para el receptor

### 4. Evitar notas metadiscursivas

- "Las siguientes secciones se organizan por..."
- Usar encabezados jerarquicos claros que hablen por si solos

### 5. Metricas estadisticas

- Cohen's d: Incluir interpretacion (pequeno/mediano/grande)
- p-value: Usar asteriscos (*p<0.05, **p<0.01, ***p<0.001) sin explicar cada vez
- Preferir porcentajes y ratios sobre valores absolutos

### 6. Visualizaciones

- Evitar superposicion de texto en graficos
- Usar siglas con leyenda cuando hay muchos elementos
- Tamano de fuente minimo 8pt para legibilidad

### 7. Eliminacion de redundancias

- Cada metrica debe aparecer UNA SOLA VEZ en tablas resumen
- Si hay dos metricas similares, mantener la de mayor poder discriminativo (Cohen's d)
- Ejemplo: Gap Medio (d=-0.61) > Gap Mediana (d=-0.46)

### 8. Concision y estructura de Anexos

El contenido principal debe ser conciso: profesional, consistente y explicativo pero NO sobre-explicativo.

- El texto principal ofrece explicaciones conceptuales de alto nivel
- Los detalles tecnicos (formulas, metodologia detallada, validaciones) van en Anexos
- El lector decide si profundizar consultando los Anexos
- Evitar parrafos largos con explicaciones tecnicas en el cuerpo principal
- Usar viñetas y tablas para presentar informacion de forma compacta

**Estructura recomendada:**
1. Cuerpo principal: Que encontramos, por que importa, que hacer
2. Anexos: Como lo calculamos, detalles metodologicos, validaciones estadisticas

### 9. Consistencia entre graficos, tablas y texto (CRITICO)

**Este es el criterio mas importante.** Cada sub-seccion debe mantener coherencia interna entre:
- El grafico que muestra
- La tabla que lo acompaña
- El texto explicativo

**Reglas de consistencia:**

1. **Las variables de la tabla deben corresponder al grafico**
   - Si un grafico muestra 4 barras (ej: Sesiones, Frecuencia, Gap, Dias Activos), la tabla debe mostrar esas mismas 4 variables
   - NO incluir variables que no aparecen en el grafico de la sub-seccion

2. **El texto debe referirse a lo que muestra el grafico**
   - No mencionar metricas que no estan visualizadas
   - El texto introduce, el grafico muestra, la tabla cuantifica

3. **Hilo conductor entre secciones**
   - Cada seccion debe fluir logicamente hacia la siguiente
   - Evitar saltos tematicos abruptos
   - Usar transiciones que conecten los hallazgos

**Estructura recomendada para cada sub-seccion de hallazgos:**
```
### Titulo del Hallazgo

[1-2 oraciones introductorias]

![Grafico](ruta/al/grafico.png)

| Variable | Aprobados | Reprobados | Diferencia | Cohen's d |
|----------|-----------|------------|------------|-----------|
| Var1     | X         | Y          | Z          | d=N       |
| Var2     | ...       | ...        | ...        | ...       |

> **Insight:** [Interpretacion del hallazgo]
```

---

## Checklist Pre-Publicacion

- [ ] Sin referencias al proceso interno de desarrollo (nada es "nuevo" o "viejo")
- [ ] Sin jerga tecnica de data science
- [ ] Sin direcciones explicitas al lector
- [ ] Encabezados jerarquicos claros (sin notas metadiscursivas)
- [ ] Graficos sin texto superpuesto
- [ ] Metricas no duplicadas
- [ ] Cohen's d interpretados
- [ ] Hallazgos accionables para intervencion
- [ ] Contenido principal conciso (detalles tecnicos en Anexos)
- [ ] **CRITICO: Consistencia grafico-tabla-texto en cada sub-seccion**

---

## Ejemplos de Errores Comunes

### Error: Referencia al timeline interno

**Incorrecto:**
> **Nueva feature:** Los "Dias Activos" miden la consistencia temporal...

**Correcto:**
> Los "Dias Activos" capturan la consistencia de acceso al LMS: cuantos dias distintos el estudiante interactuo con el curso.

### Error: Direccion explicita al receptor

**Incorrecto:**
> **Mensaje para Autoridades:** Los datos del LMS capturan patrones...

**Correcto:**
> Los datos del LMS capturan patrones conductuales que permiten identificar estudiantes en riesgo con alta precision.

### Error: Nota metadiscursiva

**Incorrecto:**
> **Organizacion de las siguientes secciones:** Los analisis se agrupan por...

**Correcto:**
(Simplemente eliminar la nota y dejar que los encabezados hablen por si solos)

### Error: Sobre-explicacion en texto principal

**Incorrecto:**
> El modelo utiliza validacion cruzada Leave-One-Course-Out (LOCO), donde se entrena con 9 cursos y se evalua en el curso restante. Este proceso se repite 10 veces, una por cada curso, y se promedian las metricas. La ventaja de LOCO sobre k-fold tradicional es que simula el escenario real donde el modelo debe predecir en un curso completamente nuevo...

**Correcto:**
> El modelo fue validado simulando su uso en cursos nuevos (ver Anexo B para metodologia de validacion).

### Error: Inconsistencia grafico-tabla (CRITICO)

**Incorrecto:**
```
### Analisis de Dinamica de Engagement

![Caracteristicas de Sesion](session_features_comparison.png)
[Grafico muestra: Sesiones Totales, Frecuencia, Gap Medio, Dias Activos]

| Metrica | Aprobados | Reprobados | Cohen's d |
|---------|-----------|------------|-----------|
| Participaciones | 6.6 | 2.7 | 0.78 |      <-- NO esta en el grafico!
| Span Actividad | 97 dias | 82 dias | 0.41 |  <-- NO esta en el grafico!
```

**Correcto:**
```
### Analisis de Dinamica de Engagement

![Caracteristicas de Sesion](session_features_comparison.png)
[Grafico muestra: Sesiones Totales, Frecuencia, Gap Medio, Dias Activos]

| Metrica | Aprobados | Reprobados | Cohen's d |
|---------|-----------|------------|-----------|
| Sesiones Totales | 39.5 | 21.4 | 0.82 |     <-- Coincide con grafico
| Frecuencia (ses/sem) | 1.80 | 1.06 | 0.69 | <-- Coincide con grafico
| Gap Medio (horas) | 85 | 139 | -0.61 |       <-- Coincide con grafico
| Dias Activos | 23.5 | 16.6 | 0.51 |         <-- Coincide con grafico
```

---

*Esta pauta debe consultarse antes de entregar cualquier informe a autoridades universitarias.*
