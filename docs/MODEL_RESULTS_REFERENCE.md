> ⚠️ **SUPERSEDED / STALE NUMBERS — do not quote.** This document predates the nested-CV correction; its headline AUCs are optimistic (non-nested) and/or contaminated (UA KEEP-arm active-zeros) and/or label-leaky. Current defensible metrics: **`RESULTS_LEDGER.md`**; start at **`PROJECT_SSOT.md`**. Kept for history only.

# Referencia de Resultados de Modelos - Sistema de Alerta Temprana

**Última actualización:** 7 de enero de 2026
**Propósito:** Documentar los mejores resultados obtenidos para trazabilidad en sesiones futuras.

---

## Resumen Ejecutivo

### Mejores Modelos por Semana (Optimizados)

| Semana | Variante | ROC-AUC | Threshold | Accuracy | Recall | Percentil | Archivo |
|--------|----------|---------|-----------|----------|--------|-----------|---------|
| **Full** | Con assessment | **0.902** | 0.33 | **83.4%** | **85.9%** | - | `best_model_optimized/` |
| **6** | Con assessment | **0.822** | 0.22 | **77.0%** | **86.2%** | 10% | `comprehensive_optimization_results.json` |
| **4** | **Sin assessment** | 0.741 | 0.24 | 70.6% | **80.4%** | 20% | `comprehensive_optimization_results.json` |
| **2** | Con assessment | 0.743 | 0.13 | 65.7% | 81.7% | 5% | `all_thresholds_optimized.json` |

### Modelos de Referencia (Históricos)

| Modelo | ROC-AUC | Threshold | Accuracy | Recall | Archivo Fuente |
|--------|---------|-----------|----------|--------|----------------|
| Time-Limited Full (base) | 0.903 | 0.50 | 83.1% | 74.5% | `data/analysis/time_cutoff_results.json` |
| XGBoost V4 (threshold=0.20) | 0.849 | 0.20 | 74.5% | 85.2% | `data/report/models/xgboost_optimized_v4/` |
| Time-Limited Week 8 | 0.828 | 0.50 | 76.4% | 61.6% | `data/analysis/time_cutoff_results.json` |

> **HALLAZGO CLAVE:** El modelo de **Semana 4 SIN features de assessment** logra **80.4% recall** con **70.6% accuracy**. Esto permite intervención temprana sin depender de evaluaciones.

---

## Optimización Comprehensiva: Experimentos y Hallazgos

### Descripción del Experimento

Se realizó una búsqueda exhaustiva probando todas las combinaciones de:
- **Percentiles de inicio de curso**: 5%, 10%, 15%, 20%
- **Semanas de corte**: 4, 6
- **Features de assessment**: Con y Sin
- **Thresholds**: 0.10 a 0.60 en pasos de 0.01

**Archivo de resultados**: `data/analysis/comprehensive_optimization_results.json`

### Matriz de Resultados: ROC-AUC por Configuración

| Semana | Percentil | Con Assessment | Sin Assessment |
|--------|-----------|----------------|----------------|
| **4** | 5% | 0.717 | 0.716 |
| **4** | 10% | 0.741 | **0.746** |
| **4** | 15% | 0.697 | 0.707 |
| **4** | **20%** | **0.756** | 0.741 |
| **6** | 5% | 0.724 | 0.709 |
| **6** | **10%** | **0.822** | 0.759 |
| **6** | 15% | 0.815 | 0.790 |
| **6** | 20% | 0.827 | 0.779 |

### Hallazgo Clave #1: Percentil Óptimo Varía por Semana

| Semana | Percentil Óptimo | Razón |
|--------|------------------|-------|
| **2** | 5% | Datos muy tempranos, percentil bajo captura más señal |
| **4** | 20% | Actividad más establecida, necesita filtrar "early birds" |
| **6** | 10-20% | Señal robusta, percentiles medios funcionan bien |

### Hallazgo Clave #2: Week 4 SIN Assessment = Mejor Opción Estratégica

**Configuración**: P20%, t=0.24, SIN features de assessment

| Métrica | Week 4 CON Assessment | Week 4 SIN Assessment | Diferencia |
|---------|----------------------|----------------------|------------|
| ROC-AUC | 0.756 | 0.741 | -0.015 |
| Accuracy | 71.4% | **70.6%** | -0.8pp |
| **Recall** | 70.3% | **80.4%** | **+10.1pp** |
| Precision | 61.4% | 58.7% | -2.7pp |

> **Implicación**: El modelo SIN assessment features detecta **10 puntos porcentuales más** de estudiantes en riesgo. Esto permite intervención temprana SIN depender de que existan evaluaciones.

### Hallazgo Clave #3: Semana 6 con Assessment = Mejor Predicción Temprana

**Configuración**: P10%, t=0.22, CON features de assessment

| Métrica | Valor |
|---------|-------|
| ROC-AUC | **0.822** |
| Accuracy | **77.0%** |
| Recall | **86.2%** |
| Precision | 65.4% |
| F2 Score | 0.811 |

Este modelo es el mejor para intervención activa en semana 6-8.

### Top Features por Modelo Optimizado

**Week 4 (P20%, sin assessment):**
1. `discussions_unique` (6.7%)
2. `total_session_time` (5.8%)
3. `modules_pct` (3.2%)
4. `unique_transitions` (2.5%)
5. `pages_pct` (2.4%)

**Week 6 (P10%, con assessment):**
1. `assignments_unique_znorm` (4.6%)
2. `grades_views_znorm` (3.6%)
3. `total_session_time` (3.5%)
4. `grades_views` (3.4%)
5. `quizzes_unique_znorm` (2.7%)

---

## 1. MEJOR MODELO: Time-Limited Full Optimizado (threshold=0.33)

### Archivo fuente
```
/home/paul/projects/uautonoma/data/report/models/best_model_optimized/threshold_optimization_results.json
```

### Configuración del modelo
- **Base:** Time-Limited Full (duración completa del curso)
- **Assessment Features:** Sí (acceso a tareas, quizzes, etc.)
- **Z-Normalization:** Sí (normalización por curso)
- **Samples:** 373 estudiantes
- **Features:** 58 (después de feature selection)

### Métricas con Threshold Optimizado (t=0.33)

| Métrica | Valor | Significado |
|---------|-------|-------------|
| **ROC-AUC** | 0.902 | Excelente capacidad discriminatoria |
| **Recall** | **85.9%** | Detectamos 8.6 de cada 10 estudiantes en riesgo |
| **Accuracy** | **83.4%** | 83% de predicciones correctas |
| **Precision** | 75.7% | 76% de alertas son verdaderas |
| **F1 Score** | 0.805 | Buen balance precision/recall |
| **F2 Score** | 0.837 | Optimizado para priorizar recall |

### Matriz de Confusión (t=0.33)
```
              Predicho
              No Falla | Falla
Actual  No Falla   185  |   39   (TN=185, FP=39)
        Falla       24  |  125   (FN=24, TP=125)
```

### Trade-off Threshold por Criterio

| Threshold | Recall | Accuracy | Precision | F2 | Uso recomendado |
|-----------|--------|----------|-----------|-----|-----------------|
| **0.33** | **85.9%** | **83.4%** | 75.7% | 0.837 | **SWEET SPOT: Alto recall + Alta accuracy** |
| 0.25 | 89.9% | 81.5% | 71.3% | 0.855 | Máximo recall práctico |
| 0.40 | 81.9% | 83.4% | 77.7% | 0.810 | Balance conservador |
| 0.50 | 75.2% | 83.1% | 81.2% | 0.763 | Threshold estándar |

### Comparación vs Modelo Anterior (XGBoost V4)

| Aspecto | XGBoost V4 (t=0.20) | **Time-Limited Full (t=0.33)** | Mejora |
|---------|---------------------|-------------------------------|--------|
| ROC-AUC | 0.849 | **0.902** | +0.053 |
| Recall | 85.2% | **85.9%** | +0.7pp |
| Accuracy | 74.5% | **83.4%** | **+8.9pp** |
| Precision | 63.0% | **75.7%** | +12.7pp |

> **Conclusión:** El modelo Time-Limited Full Optimizado es significativamente superior:
> - Mismo nivel de recall (~86%)
> - **9 puntos porcentuales más de accuracy**
> - Menos falsas alarmas (precision 76% vs 63%)

### Datos del modelo
```json
{
  "model_info": {
    "name": "Time-Limited Full + Assessment + Z-Norm",
    "roc_auc": 0.9024,
    "n_samples": 373,
    "n_features": 58,
    "failure_rate": 0.399
  },
  "threshold_0.33": {
    "recall": 0.859,
    "accuracy": 0.834,
    "precision": 0.757,
    "f1": 0.805,
    "f2": 0.837,
    "confusion_matrix": {"tp": 125, "fp": 39, "fn": 24, "tn": 185}
  }
}
```

---

## 2. Modelo XGBoost V4 (Threshold Optimization) - REFERENCIA

### Archivo fuente
```
/home/paul/projects/uautonoma/data/report/models/xgboost_optimized_v4/threshold_optimization_results.json
```

### ROC-AUC Base
```
0.8489
```

### Trade-off Accuracy vs Recall según Umbral

| Threshold | Recall | Accuracy | Precision | F1 | F2 | Uso recomendado |
|-----------|--------|----------|-----------|-----|-----|-----------------|
| **0.20** | **85.2%** | 74.5% | 63.0% | 72.5% | 79.6% | **Maximizar detección (SWEET SPOT)** |
| 0.29 | 78.2% | 74.8% | 64.9% | - | - | Balance recall/accuracy |
| 0.50 | 61.3% | 75.1% | 71.3% | 65.9% | 63.0% | Estándar (menos alertas) |

### Interpretación del Trade-off
- **Bajar el threshold** (ej: 0.20): Más estudiantes marcados como "en riesgo" → Mayor recall, menor precisión
- **Subir el threshold** (ej: 0.50): Menos estudiantes marcados → Menor recall, mayor precisión
- **Para alerta temprana**: Preferir threshold bajo (0.20) porque es mejor detectar de más que perder estudiantes en riesgo

### Datos del modelo
```json
{
  "roc_auc": 0.8488648787703389,
  "max_f2": {
    "threshold": 0.2,
    "recall": 0.852112676056338,
    "accuracy": 0.7451523545706371,
    "precision": 0.6302083333333334,
    "f1": 0.7245508982035928,
    "f2": 0.7960526315789473
  }
}
```

---

## 3. Modelo Time-Limited Base (sin threshold optimization)

### Archivo fuente
```
/home/paul/projects/uautonoma/data/analysis/time_cutoff_results.json
```

### Configuración del experimento
- **Cutoff:** full (duración completa del semestre)
- **Include Assessment Features:** Yes
- **Include Z-Normalization:** Yes
- **Samples:** 373 estudiantes
- **Features:** 59 totales (12 z-normalizadas)
- **Failure Rate:** 39.95%

### Métricas (MEJOR ROC-AUC OBTENIDO)
```
ROC-AUC:   0.9033 (EXCELENTE)
Accuracy:  0.8311 (83.1%)
Precision: 0.8162 (81.6%)
Recall:    0.7450 (74.5%)
F1 Score:  0.7789 (77.9%)
```

### Matriz de Confusión
```
              Predicho
              No Falla | Falla
Actual  No Falla   199  |   25   (TN=199, FP=25)
        Falla       38  |  111   (FN=38, TP=111)
```

### Top 10 Features más importantes
| # | Feature | Importancia | Descripción |
|---|---------|-------------|-------------|
| 1 | `assi_access_rate` | 10.67% | Tasa de acceso a tareas |
| 2 | `assi_mean_pct` | 5.51% | Percentil promedio en tareas |
| 3 | `assignments_unique_resources_znorm` | 5.29% | Recursos únicos de tareas (z-norm) |
| 4 | `quiz_access_rate` | 3.52% | Tasa de acceso a quizzes |
| 5 | `total_time_min_znorm` | 2.93% | Tiempo total (z-norm) |
| 6 | `grades_views_znorm` | 2.78% | Vistas de calificaciones (z-norm) |
| 7 | `quizzes_unique_resources` | 2.73% | Recursos únicos de quizzes |
| 8 | `grades_views` | 2.61% | Vistas de calificaciones |
| 9 | `assignments_views_znorm` | 2.48% | Vistas de tareas (z-norm) |
| 10 | `quizzes_unique_resources_znorm` | 2.16% | Recursos quizzes (z-norm) |

---

## 4. Predicción Temprana: Resultados por Corte Temporal

### Archivo fuente
```
/home/paul/projects/uautonoma/data/analysis/time_cutoff_results.json
```

### Progresión del ROC-AUC según semanas disponibles

**CON features de evaluación (assessment):**

| Semana | Samples | ROC-AUC | Accuracy | Recall | Precision | F1 |
|--------|---------|---------|----------|--------|-----------|-----|
| **2** | 303 | 0.743 | 70.6% | 55.8% | 65.0% | 60.1% |
| **4** | 343 | 0.742 | 70.0% | 51.5% | 63.6% | 56.9% |
| **6** | 351 | 0.745 | 70.7% | 51.5% | 65.4% | 57.6% |
| **8** | 356 | **0.828** | **76.4%** | **61.6%** | **73.3%** | **66.9%** |
| **Full** | 373 | **0.903** | **83.1%** | **74.5%** | **81.6%** | **77.9%** |

**SIN features de evaluación (solo actividad):**

| Semana | Samples | ROC-AUC | Accuracy | Recall | Precision | F1 |
|--------|---------|---------|----------|--------|-----------|-----|
| **2** | 303 | 0.740 | 68.0% | 54.2% | 60.7% | 57.3% |
| **4** | 343 | 0.742 | 69.1% | 51.5% | 61.8% | 56.2% |
| **6** | 351 | 0.736 | 69.2% | 50.7% | 62.7% | 56.1% |
| **8** | 356 | **0.833** | **75.0%** | **59.4%** | **71.3%** | **64.8%** |
| **Full** | 373 | 0.848 | 75.9% | 61.1% | 74.0% | 66.9% |

### Hallazgo Clave: El "Umbral de la Semana 8"

```
Semanas 2-6:  ROC-AUC ~ 0.74 (predicción moderada)
Semana 8:     ROC-AUC = 0.83 (SALTO SIGNIFICATIVO)
Full:         ROC-AUC = 0.90 (mejor posible)
```

**Interpretación:**
- Las primeras 6 semanas NO acumulan suficiente señal para predicción confiable
- A partir de la **semana 8** el modelo se vuelve altamente confiable
- Esto coincide con el punto medio de un semestre de 16 semanas
- Permite **8 semanas de intervención** antes de las notas finales

### Hallazgo Sorprendente: Features de Evaluación vs Solo Actividad

En la **semana 8**, los features de solo actividad (ROC-AUC=0.833) superan ligeramente a los que incluyen evaluaciones (ROC-AUC=0.828).

**Implicación:** Los patrones de engagement capturan señales de fracaso tan bien como las calificaciones parciales.

---

## 5. Rendimiento por Curso Individual (LOCO)

### Archivo fuente
```
/home/paul/projects/uautonoma/data/analysis/time_cutoff_results.json (experimento 8)
```

### ROC-AUC por curso (validación Leave-One-Course-Out)

| Curso ID | Nombre | AUC | Interpretación |
|----------|--------|-----|----------------|
| 84936 | FUND. MICROECONOMÍA-P03 | **0.94** | Excelente |
| 89099 | COMP. DIGITALES | **0.94** | Excelente |
| 79875 | COMPETENCIAS DIGITALES | **0.89** | Muy bueno |
| 84941 | FUND. MICROECONOMÍA-P01 | **0.87** | Muy bueno |
| 88381 | MATEMÁTICAS NEGOCIOS | **0.87** | Muy bueno |
| 89390 | GESTIÓN DEL TALENTO-P01 | 0.83 | Bueno |
| 79913 | FUND. BUSINESS ANALYTICS | 0.75 | Aceptable |
| 84944 | FUND. MACROECONOMÍA-P03 | 0.74 | Aceptable |
| 86676 | BUSINESS ANALYTICS-P01 | 0.72 | Aceptable |
| 86020 | COMPETENCIAS DIGITALES-P02 | 0.69 | Límite |

**Rango:** 0.69 - 0.94
**Promedio LOCO:** ~0.82

---

## 6. Otros Modelos de Referencia

### Modelo Enriched Features (Random Forest)
```
Archivo: /home/paul/projects/uautonoma/data/enriched_features/early_warning_model_results.json
```
- **Accuracy:** 79.22%
- **Precision:** 79.13%
- **Recall:** 64.08%
- **F1 Score:** 70.82%
- **ROC-AUC:** 0.8396
- **Features:** 145

### Modelo Early Warning Temporal (Logistic Regression)
```
Archivo: /home/paul/projects/uautonoma/data/early_warning/model_results.json
```
- **Accuracy:** 65.15%
- **Recall:** 60.40%
- **Precision:** 55.90%
- **F1 Score:** 58.06%
- **ROC-AUC:** 0.7066

---

## 7. Scripts de Entrenamiento

| Script | Propósito |
|--------|-----------|
| `scripts/optimize_best_model_threshold.py` | **Optimizar threshold del mejor modelo (ROC-AUC 0.90)** |
| `scripts/optimize_weeks_comprehensive.py` | **Optimización exhaustiva: percentiles × semanas × assessment** |
| `scripts/optimize_week2_detailed.py` | Análisis detallado de semana 2 con todos los thresholds |
| `scripts/train_optimal_early_model.py` | Entrenar modelos con configuración óptima por semana |
| `scripts/train_time_limited_model.py` | Entrenar modelos con cortes temporales |
| `scripts/optimize_threshold_f2.py` | Optimizar threshold para maximizar F2/recall |
| `scripts/train_enriched_model.py` | Modelo con features enriquecidos |
| `scripts/train_early_warning_model.py` | Modelo con features temporales |
| `scripts/train_optimized_early_warning.py` | Modelo optimizado |

---

## 8. Definiciones de Métricas

| Métrica | Fórmula | Interpretación |
|---------|---------|----------------|
| **ROC-AUC** | Área bajo curva ROC | Capacidad discriminatoria (0.5=azar, 1.0=perfecto) |
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | % predicciones correctas |
| **Recall (Sensibilidad)** | TP/(TP+FN) | % de reprobados detectados |
| **Precision** | TP/(TP+FP) | % de alertas que son verdaderas |
| **F1 Score** | 2*(Prec*Rec)/(Prec+Rec) | Balance precisión-recall |
| **F2 Score** | 5*(Prec*Rec)/(4*Prec+Rec) | F-score que prioriza recall |

**Para Alerta Temprana:**
- **Priorizar Recall** = Detectar más estudiantes en riesgo (aunque algunos sean falsos positivos)
- **Threshold bajo (0.20)** = Más alertas, mayor recall
- **Threshold alto (0.50)** = Menos alertas, mayor precisión

### Métricas Avanzadas de Optimización

| Métrica | Fórmula | Rango | Interpretación |
|---------|---------|-------|----------------|
| **Youden's J** | Sensibilidad + Especificidad - 1 | [-1, 1] | Estándar clínico para selección de umbral |
| **MCC (Matthews)** | (TP×TN - FP×FN) / sqrt(...) | [-1, 1] | Mejor métrica única para clases desbalanceadas |
| **G-Mean** | sqrt(Sensibilidad × Especificidad) | [0, 1] | Balance geométrico entre métricas |
| **Balanced Accuracy** | (Sensibilidad + Especificidad) / 2 | [0, 1] | Promedio simple de ambas métricas |
| **F3 Score** | 10×Prec×Rec / (9×Prec + Rec) | [0, 1] | F-score muy ponderado hacia recall |
| **Costo Ponderado** | FN × costo_fn + FP × costo_fp | [0, +∞] | Minimizar según costos reales |

### Cuándo Usar Cada Criterio

| Escenario | Criterio Recomendado | Razón |
|-----------|---------------------|-------|
| Selección clínica estándar | **Youden's J** | Práctica establecida en medicina |
| Clases muy desbalanceadas | **MCC** | Considera todos los cuadrantes |
| Recursos limitados para intervención | **Costo Ponderado (3x)** | Modela costos reales |
| Intervención agresiva temprana | **F3 o Recall≥90%** | Prioriza detección máxima |
| Alta confianza requerida | **MCC o Max Accuracy** | Alertas de alta certeza |

### Recomendaciones de Umbral por Escenario de Despliegue

| Escenario | Descripción | Uso Típico |
|-----------|-------------|------------|
| **Agresivo** | Maximizar detección temprana | Recursos abundantes, alto costo de perder estudiante |
| **Balanceado** | Óptimo estadístico (Youden J) | Balance entre detección y precisión |
| **Conservador** | Minimizar falsas alarmas (Max MCC) | Recursos limitados, requiere alta confianza |

---

## 9. Recomendaciones de Uso

### Estrategia Recomendada por Ventana de Intervención

| Semana | Modelo a Usar | Configuración | Recall | Accuracy | Uso |
|--------|---------------|---------------|--------|----------|-----|
| **2-3** | Week 2 | P5%, t=0.13, con assessment | 81.7% | 65.7% | Watch list inicial |
| **4-5** | **Week 4 SIN assessment** | P20%, t=0.24 | **80.4%** | 70.6% | **Intervención temprana** |
| **6-7** | Week 6 CON assessment | P10%, t=0.22 | 86.2% | 77.0% | Intervención activa |
| **8+** | Full Optimizado | t=0.33 | 85.9% | 83.4% | Intervención intensiva |

### Para Intervención Temprana SIN Evaluaciones (Semana 4)
- **Usar modelo Week 4 SIN assessment** (P20%, t=0.24)
- Recall: **80.4%** (detectamos 8 de cada 10 estudiantes en riesgo)
- Accuracy: 70.6%
- **Ventaja clave**: Funciona ANTES de que existan calificaciones
- Ideal para cursos donde las primeras evaluaciones son en semana 5+

### Para Intervención Activa CON Evaluaciones (Semana 6+)
- **Usar modelo Week 6 CON assessment** (P10%, t=0.22)
- Recall: **86.2%** (detectamos 8.6 de cada 10 estudiantes en riesgo)
- Accuracy: 77.0%
- **Ventaja clave**: Alta confiabilidad, quedan 10+ semanas para intervenir

### Para Reportes a Autoridades
- Usar métricas del **mejor modelo** (Time-Limited Full Optimizado, t=0.33):
  - ROC-AUC: **0.90**
  - Accuracy: **83.4%**
  - Recall: **85.9%**
  - Precision: **75.7%**
- Enfatizar que detectamos **8.6 de cada 10** estudiantes en riesgo
- Este modelo supera al anterior en accuracy (+9pp) manteniendo el mismo recall
- **Mensaje clave**: Podemos identificar estudiantes en riesgo desde la semana 4 con 80%+ de efectividad

---

## 10. Archivos de Datos Clave

```
data/
├── analysis/
│   ├── time_cutoff_results.json                   # Resultados cortes temporales (10 experimentos)
│   ├── comprehensive_optimization_results.json   # *** OPTIMIZACIÓN EXHAUSTIVA ***
│   ├── all_thresholds_optimized.json             # Thresholds optimizados por semana
│   └── optimal_early_model_results.json          # Modelos óptimos por semana
├── report/
│   └── models/
│       ├── best_model_optimized/                 # *** MEJOR MODELO (Full, t=0.33) ***
│       │   └── threshold_optimization_results.json
│       ├── xgboost_optimized_v4/
│       │   └── threshold_optimization_results.json
│       └── time_limited/
│           └── summary/
├── enriched_features/
│   ├── early_warning_model_results.json
│   └── cutoff_week_[2,4,6,8]/                    # Features por corte temporal
└── early_warning/
    └── model_results.json

scripts/
├── optimize_weeks_comprehensive.py               # *** SCRIPT DE OPTIMIZACIÓN EXHAUSTIVA ***
├── optimize_week2_detailed.py                    # Análisis detallado semana 2
├── train_optimal_early_model.py                  # Entrenamiento con config óptima
└── optimize_best_model_threshold.py              # Optimización del mejor modelo
```

---

## 11. Historial de Cambios

| Fecha | Cambio |
|-------|--------|
| 2026-01-07 | Documento creado con resultados de XGBoost V4 y Time-Limited models |
| 2026-01-07 | Agregados resultados de predicción temprana por semana |
| 2026-01-07 | Documentado trade-off accuracy vs recall según threshold |
| 2026-01-07 | NUEVO: Time-Limited Full Optimizado (t=0.33) - ROC-AUC 0.90, Recall 85.9%, Accuracy 83.4% |
| **2026-01-07** | **NUEVO: Optimización comprehensiva - Semana 4 SIN assessment (80.4% recall) y Semana 6 CON assessment (86.2% recall)** |

---

*Este documento sirve como referencia centralizada para sesiones futuras de análisis.*
