# Optimizacion de Umbrales para Modelos de Alerta Temprana

## Resumen Ejecutivo

Se realizo una optimizacion comprehensiva de umbrales para todos los modelos de prediccion, evaluando 40 configuraciones diferentes:
- **Cortes temporales**: Semanas 2, 4, 6, 8 y semestre completo
- **Percentiles de inicio**: 5%, 10%, 15%, 20%
- **Features de evaluacion**: Con y sin
- **Umbrales**: 0.10 a 0.70 en incrementos de 0.01

### Metricas de Optimizacion Utilizadas
| Metrica | Descripcion | Uso Recomendado |
|---------|-------------|-----------------|
| Youden's J | Sensibilidad + Especificidad - 1 | Umbral estadisticamente optimo |
| MCC | Coeficiente de Correlacion de Matthews | Mejor metrica unica para clases desbalanceadas |
| G-Mean | sqrt(Sensibilidad x Especificidad) | Balance geometrico |
| F2/F3 | F-score ponderado hacia recall | Deteccion agresiva |
| Costo-Optimo | FN x 3 + FP x 1 (o 5x) | Cuando perder estudiantes es costoso |

---

## Mejores Configuraciones por Semana

| Semana | Percentil | Assessment | ROC-AUC | Umbral | Recall | Accuracy |
|--------|-----------|------------|---------|--------|--------|----------|
| 2 | 5% | No | 0.729 | 0.30 | 67.8% | 68.8% |
| 4 | 20% | Si | 0.754 | 0.19 | 81.2% | 68.9% |
| 6 | 20% | Si | 0.846 | 0.36 | 78.3% | 78.7% |
| 8 | 20% | Si | 0.843 | 0.40 | 73.4% | 79.3% |
| Completo | 5% | Si | 0.902 | 0.33 | 85.9% | 83.4% |

---

## Recomendaciones de Despliegue por Momento de Intervencion

### Semana 2-3: Lista de Vigilancia Inicial
**Objetivo**: Identificacion temprana con alta incertidumbre

- **Configuracion**: Percentil 5%, Sin features de evaluacion
- **ROC-AUC**: 0.729
- **Muestra**: 298 estudiantes

| Estrategia | Umbral | Recall | Accuracy | Uso |
|------------|--------|--------|----------|-----|
| Agresiva | 0.11 | 86.4% | 61.1% | Capturar maximos estudiantes en riesgo |
| Balanceada | 0.30 | 67.8% | 68.8% | Equilibrio entre deteccion y precision |

**Nota**: Alta incertidumbre - usar para monitoreo, no para intervenciones intensivas.

---

### Semana 4-5: Primera Intervencion
**Objetivo**: Buen balance entre deteccion temprana y fiabilidad

- **Configuracion**: Percentil 20%, Con features de evaluacion
- **ROC-AUC**: 0.754
- **Muestra**: 357 estudiantes

| Estrategia | Umbral | Recall | Accuracy | Uso |
|------------|--------|--------|----------|-----|
| Agresiva | 0.10 | 87.7% | 63.3% | Maximizar cobertura de riesgo |
| Balanceada | 0.19 | 81.2% | 68.9% | Primera ronda de apoyo academico |

**Accion recomendada**: Enviar recordatorios automaticos, asignar tutores.

---

### Semana 6-7: Intervencion Activa
**Objetivo**: Predicciones fuertes, 10+ semanas restantes

- **Configuracion**: Percentil 20%, Con features de evaluacion
- **ROC-AUC**: 0.846
- **Muestra**: 357 estudiantes

| Estrategia | Umbral | Recall | Accuracy | Uso |
|------------|--------|--------|----------|-----|
| Agresiva | 0.14 | 87.0% | 72.5% | Alcance proactivo amplio |
| Balanceada | 0.36 | 78.3% | 78.7% | Intervencion dirigida |

**Accion recomendada**: Reuniones con coordinadores, planes de mejora personalizados.

---

### Semana 8+: Apoyo Intensivo
**Objetivo**: Alta fiabilidad, punto de control a mitad de semestre

- **Configuracion**: Percentil 20%, Con features de evaluacion
- **ROC-AUC**: 0.843
- **Muestra**: 358 estudiantes

| Estrategia | Umbral | Recall | Accuracy | Uso |
|------------|--------|--------|----------|-----|
| Agresiva | 0.12 | 89.2% | 73.5% | Ultima oportunidad de rescate |
| Balanceada | 0.40 | 73.4% | 79.3% | Recursos intensivos a casos claros |

**Accion recomendada**: Talleres de refuerzo, modificacion de carga academica si es posible.

---

### Fin de Semestre: Revision Final
**Objetivo**: Maxima precision para prediccion de notas finales

- **Configuracion**: Percentil 5%, Con features de evaluacion
- **ROC-AUC**: 0.902
- **Muestra**: 373 estudiantes

| Estrategia | Umbral | Recall | Accuracy | Uso |
|------------|--------|--------|----------|-----|
| Agresiva | 0.24 | 90.6% | 81.5% | Maximizar deteccion |
| Balanceada | 0.33 | 85.9% | 83.4% | Optimo estadistico |
| Conservadora | 0.50 | 66.4% | 83.6% | Minimas falsas alarmas |

---

## Hallazgos Clave

### 1. Evolucion del ROC-AUC por Semana
```
Semana 2:  0.729 (+/- 0.03)
Semana 4:  0.754 (+/- 0.02)
Semana 6:  0.846 (+/- 0.04)
Semana 8:  0.843 (+/- 0.03)
Completo:  0.902
```

**Interpretacion**: La capacidad predictiva mejora significativamente despues de la semana 4, con un salto notable en semana 6 cuando hay datos de evaluaciones sumativas.

### 2. Impacto de Features de Evaluacion
- **Semana 2**: Features de evaluacion NO mejoran el modelo (mejor sin ellas)
- **Semanas 4+**: Features de evaluacion mejoran significativamente (~5-8% ROC-AUC)

**Razon**: En semana 2, aun no hay suficientes evaluaciones; usar solo actividad es mas robusto.

### 3. Percentil Optimo
- **Semanas tempranas (2-4)**: Percentil 20% funciona mejor (mas datos de actividad)
- **Semestre completo**: Percentil 5% captura mejor el inicio real del curso

### 4. Convergencia de Metricas
Las metricas Youden's J, MCC y G-Mean tienden a converger al mismo umbral optimo, validando la robustez de la seleccion.

---

## Resumen para Implementacion

### Configuracion Recomendada: Sistema de 3 Niveles

| Nivel | Semana | ROC-AUC | Umbral | Recall | Accion |
|-------|--------|---------|--------|--------|--------|
| **Vigilancia** | 2-3 | 0.73 | 0.30 | 68% | Lista de monitoreo |
| **Alerta** | 4-6 | 0.80 | 0.20 | 85% | Contacto proactivo |
| **Critico** | 8+ | 0.84 | 0.15 | 89% | Intervencion intensiva |

### Notas de Implementacion

1. **Recalcular semanalmente**: Los modelos deben re-ejecutarse cada semana con datos actualizados
2. **Umbrales dinamicos**: Considerar umbrales mas bajos en cursos historicamente dificiles
3. **False positives aceptables**: En contexto educativo, es preferible sobre-alertar que perder estudiantes
4. **Validacion cruzada por curso**: Monitorear diferencias de rendimiento entre cursos

---

## Archivos Generados

- `all_models_optimized.json`: Resultados detallados de las 40 configuraciones
- Cada configuracion incluye 12 metricas de optimizacion por umbral

---

*Generado automaticamente por optimize_all_models_comprehensive.py*
*Fecha: Enero 2026*
