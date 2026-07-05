# Predicción temprana de reprobación — Resultados por semana (PUC)

## Contexto

El modelo estima, a partir de la actividad de cada estudiante en la plataforma (Canvas), la probabilidad de que repruebe el curso (Nota Final < 4,0 en escala 1–7). El análisis se realiza sobre 7 cursos de la Pontificia Universidad Católica (semestre 2023-1), con 560 pares estudiante-curso y su nota oficial de acta.

La pregunta central es **cuán temprano y con qué precisión el modelo distingue a los estudiantes en riesgo**, en función de cuántas semanas de actividad se han observado.

## Desempeño por semana de corte

| Semana de corte | Poder de discriminación (ROC-AUC) |
|---|---|
| Semana 2  | 0,78 |
| Semana 4  | 0,82 |
| Semana 6  | 0,83 |
| Semana 8  | 0,84 |
| Curso completo | 0,83 |

El ROC-AUC corresponde a la probabilidad de que el modelo asigne mayor riesgo a un estudiante que efectivamente reprueba que a uno que aprueba (0,50 equivale a azar; 1,00 a una separación perfecta).

## Lectura de los resultados

- **Desde la semana 2**, con muy poca información acumulada, el modelo ya discrimina de forma clara (0,78).
- Hacia la **semana 4** alcanza aproximadamente 0,82 y se estabiliza en torno a **0,83–0,84** durante la segunda mitad del curso.
- El desempeño se mantiene consistente entre modelos de *gradient boosting* (XGBoost y CatBoost), lo que indica que el resultado no depende de una elección puntual de algoritmo.

En términos prácticos, el modelo permite identificar a la mayoría de los estudiantes en riesgo con anticipación suficiente para intervenir dentro del semestre.

## Validación

- **Validación cruzada por curso** (leave-one-course-out): el modelo se evalúa en cursos que no formaron parte de su entrenamiento, de modo que las cifras estiman su desempeño esperado en cursos nuevos.
- **Probabilidades calibradas**: los puntajes reflejan probabilidades de reprobación interpretables, no únicamente un orden relativo.
- Todas las configuraciones se comparan sobre las mismas particiones y con semilla fija, para asegurar comparaciones equivalentes.

## Explicabilidad

Cada predicción se acompaña de los factores que más contribuyen al riesgo de cada estudiante, expresados en lenguaje comprensible (por ejemplo: accesos a evaluaciones, regularidad de las sesiones y tendencia semanal de actividad). Esto permite acompañar cada puntaje de riesgo con una explicación accionable a nivel individual.
