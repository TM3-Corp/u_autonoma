# Contexto: Análisis de Page Views - Investigación Completada

## Resumen

Este documento documenta la investigación sobre el API de page_views de Canvas para mejorar el modelo predictivo de fracaso estudiantil.

## Conclusión Principal

**El endpoint async `POST /api/v1/users/{id}/page_views/query` permite acceso programático a 1 año de datos históricos de page_views.** Requisito crítico: las fechas deben ser primer día del mes (YYYY-MM-01).

## Estado Final

### Investigación Completada (30 Diciembre 2025)
- **Hallazgo API estándar**: Canvas API `/users/{id}/page_views` solo retorna 30 días
- **Hallazgo API async**: `POST /api/v1/users/{id}/page_views/query` ✅ FUNCIONA con 1 año de datos
- **Requisito crítico**: Las fechas start_date y end_date DEBEN ser primer día del mes (YYYY-MM-01)
- **Validación**: User 33017 - API async = 9,215 rows = CSV manual (coincidencia exacta)

## Comparación de Métodos de Acceso

| Método | Retención | Granularidad | Programático | Notas |
|--------|-----------|--------------|--------------|-------|
| **API `/users/{id}/page_views`** | 30 días | URLs + timestamps | ✅ Sí | Paginación por bookmark |
| **API `/page_views/query` (async)** | 1 año | CSV/JSONL | ✅ Sí | Fechas deben ser 1er día del mes |
| **Admin UI CSV Export** | 1 año | URLs + timestamps + más | ❌ Manual | Requiere acceso admin |
| **Analytics API** | Histórico | Hourly bins | ✅ Sí | Datos agregados solamente |
| **Student Summaries** | Histórico | Totales | ✅ Sí | Solo conteos finales |

## Evidencia de Retención 30 días

### Test realizado: 30 Diciembre 2025

| User ID | Curso Origen | Page Views | Fecha Más Antigua |
|---------|--------------|------------|-------------------|
| 117656 | 86005 | 537 | **2025-11-30** |
| 89587 | 86005 | 443 | **2025-11-30** |
| 86579 | 84936 | 2091 | **2025-11-30** |
| 107496 | 84936 | 261 | **2025-11-30** |
| 113050 | 79913 | 507 | 2025-12-01 |
| 107667 | 79913 | 12 | 2025-12-11 |

**6 de 10 usuarios** tienen exactamente 2025-11-30 como fecha más antigua (30 días desde hoy).

### Caso de Estudio: User 86579

- **Page Views API**: 2091 views (Nov 30 - Dec 30)
- **Course 84936 en Page Views**: 0 views (no aparece)
- **Course 84936 en Analytics**: 256 views (Aug 18 - Oct 7)

El curso 84936 tuvo actividad en Ago-Oct pero está fuera de la ventana de retención.

## Descubrimiento: CSV Export via Admin UI

### Evidencia Comparativa (User 33017)

| Fuente | Page Views | Período |
|--------|------------|---------|
| API `/users/33017/page_views` | 2,771 | 2025-11-30 a 2025-12-30 (30 días) |
| CSV Export Admin UI | 9,215 | 2025-09-08 a 2025-12-30 (112 días) |

**El CSV tiene 6,444 registros adicionales que el API no puede acceder.**

### Características del CSV Export

El archivo CSV contiene columnas ricas:
- `created_at`: Timestamp preciso
- `http_request`: URL completa de la página
- `controller`, `action`: Tipo de recurso accedido
- `participated`: Booleano de participación activa
- `canvas_context_type`, `canvas_context_id`: Contexto (Course, Group, etc.)
- `interaction_seconds`: Tiempo de interacción estimado
- `vhost`: Host de Canvas (canvas.uautonoma.cl)

### Endpoint Async ✅ FUNCIONA

El endpoint para exportación programática de 1 año de datos:

```python
# 1. Iniciar query (fechas DEBEN ser primer día del mes)
POST /api/v1/users/{user_id}/page_views/query
Headers: Content-Type: application/json
Body: {
    "start_date": "2025-09-01",  # Primer día del mes
    "end_date": "2025-12-01",    # Primer día del mes siguiente
    "results_format": "csv"
}
# Response: {"poll_url": "/api/v1/users/33017/page_views/query/{query_id}"}

# 2. Poll hasta status = "finished"
GET /api/v1/users/{user_id}/page_views/query/{query_id}
# Response: {"status": "finished", "results_url": "..."}

# 3. Descargar resultados
GET /api/v1/users/{user_id}/page_views/query/{query_id}/results
# Response: CSV con todos los page views
```

**Validación**: User 33017 - API async retorna 9,215 rows = CSV manual 9,215 rows ✓

## Decisión Final

✅ **PROCEDER con extracción de page_views granulares**

El endpoint async `POST /api/v1/users/{id}/page_views/query` funciona correctamente y permite:
1. Acceso programático a 1 año de datos históricos
2. Extracción de todos los 373 estudiantes
3. Generación de features mejorados para el modelo predictivo

**Próximos pasos:**
1. Crear script de extracción masiva usando el endpoint async
2. Procesar page_views para generar features de sesión (30 min threshold)
3. Calcular features por categoría de recurso
4. Re-entrenar modelo con features enriquecidos

## Scripts Creados (No utilizados)

Los siguientes scripts fueron creados pero no se ejecutaron debido a la limitación de datos:

- `scripts/extract_page_views.py` - Extracción con bookmark pagination
- `scripts/categorize_page_views.py` - Categorización por tipo de recurso
- `scripts/calculate_session_features.py` - Features de sesión (30 min)
- `scripts/calculate_category_features.py` - Features por categoría
- `scripts/train_enriched_model.py` - Entrenamiento comparativo
- `scripts/check_page_view_availability.py` - Diagnóstico de disponibilidad

## Modelo Actual (Sin cambios)

| Métrica | Valor |
|---------|-------|
| Accuracy | 74.0% |
| ROC-AUC | 0.787 |
| Precision | 69.7% |
| Recall | 61.7% |
| F1 Score | 65.5% |

## Recomendaciones Futuras

### Opción 1: Extracción Manual por Lotes
Si se desea extraer datos históricos:
1. Descargar CSV manualmente desde `/accounts/46/users/{user_id}` para cada estudiante
2. Automatizar con Selenium/Playwright si son muchos usuarios
3. Procesar los CSVs descargados con scripts existentes

### Opción 2: Contactar Instructure
Solicitar a Instructure:
1. Habilitar el endpoint async `page_views/query`
2. Aumentar retención del API de page_views
3. Acceso a exportación bulk de page_views

### Opción 3: Captura Prospectiva
Para futuros semestres:
1. Implementar extracción diaria vía API mientras cursos están activos
2. Comenzar desde día 1 del próximo semestre
3. Almacenar en base de datos local para retención indefinida

### Cursos Activos
Los cursos con datos recientes disponibles vía API (últimos 30 días):
- Course 86005: Datos de Nov 30 - Dic 30
- Course 86676: Datos de Nov 30 - Dic 30
- Otros cursos del Term 336 con actividad reciente

---
*Última actualización: 30 Diciembre 2025*
*Estado: Investigación completada - Mejora abandonada por limitación de retención*
