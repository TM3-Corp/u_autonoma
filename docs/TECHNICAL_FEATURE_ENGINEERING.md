# Documentacion Tecnica: Pipeline de Ingenieria de Features
## Sistema de Alerta Temprana para Prediccion de Fracaso Academico
### Universidad Autonoma de Chile - Canvas LMS

---

## 1. Resumen del Pipeline

Este documento describe el pipeline completo de ingenieria de features desarrollado para predecir el fracaso academico de estudiantes utilizando **UNICAMENTE** datos de engagement con materiales de aprendizaje, **EXCLUYENDO** toda actividad relacionada con evaluaciones (quizzes, tareas, calificaciones).

### Resultados Clave

| Metrica | Baseline | Modelo Optimizado | Mejora |
|---------|----------|-------------------|--------|
| **ROC-AUC** | 0.787 | **0.860** | +9.3% |
| **Exactitud** | 74.0% | 78.4% | +4.4pp |
| **Precision** | 69.7% | 72.9% | +3.2pp |
| **Sensibilidad** | 61.7% | 71.8% | +10.1pp |

### Innovacion Principal

El modelo utiliza **203 features** derivados exclusivamente de:
- Archivos (Files)
- Discusiones (Discussions)
- Paginas (Pages)
- Modulos (Modules)
- Anuncios (Announcements)
- Pagina de inicio (Home)
- Patrones de sesion
- Patrones de navegacion (N-gramas)
- Patrones temporales (Horarios)
- Analisis de grafo de recursos

**54 features excluidos** (evaluaciones):
- Quizzes: `quiz_*`, `quizzes_*`
- Tareas: `assi_*`, `assignments_*`
- Calificaciones: `grades_*`, `grad_*`

---

## 2. Fuentes de Datos (Canvas API)

### 2.1 Page Views API

**Endpoint:** `GET /api/v1/users/{user_id}/page_views`

**Parametros:**
- `start_time`: Inicio del periodo
- `end_time`: Fin del periodo
- `per_page`: 100 (maximo)

**Datos extraidos:**
```json
{
  "created_at": "2025-10-15T14:30:00Z",
  "url": "/courses/86005/files/123",
  "interaction_seconds": 45.2,
  "http_method": "GET",
  "controller": "files",
  "action": "show",
  "participated": true
}
```

**Limitacion importante:** No existe filtro por `course_id`. Se debe filtrar post-extraccion parseando la URL.

### 2.2 Enrollments API

**Endpoint:** `GET /api/v1/courses/{course_id}/enrollments`

**Parametros:**
- `type[]`: `StudentEnrollment`
- `include[]`: `grades`, `total_scores`

**Datos extraidos:**
```json
{
  "user_id": 117656,
  "course_id": 86005,
  "grades": {
    "current_score": 79.07,
    "final_score": 46.65
  }
}
```

### 2.3 Student Summaries API

**Endpoint:** `GET /api/v1/courses/{course_id}/analytics/student_summaries`

**Datos extraidos:**
- `page_views`: Total de vistas
- `participations`: Total de participaciones
- `tardiness_breakdown`: Entregas a tiempo, tarde, faltantes

---

## 3. Extraccion de Datos Crudos

### 3.1 Script de Extraccion

**Archivo:** `scripts/extract_page_views_async.py`

**Proceso:**
1. Obtener lista de estudiantes inscritos por curso
2. Para cada estudiante, extraer page views via API
3. Filtrar por periodo del curso
4. Guardar en formato Parquet

**Output:** `data/page_views/page_views_YYYY_MM_DD.parquet`

### 3.2 Categorizacion de Page Views

**Archivo:** `scripts/categorize_page_views.py`

**Proceso de categorizacion:**

```python
def categorize_page_view(url, controller, action):
    """Categoriza un page view por tipo de recurso."""

    # Extraer course_id y resource_id de la URL
    course_match = re.search(r'/courses/(\d+)', url)

    # Mapeo de controladores a tipos
    CONTROLLER_MAP = {
        'files': 'files',
        'discussion_topics': 'discussions',
        'quizzes': 'quizzes',
        'assignments': 'assignments',
        'wiki_pages': 'pages',
        'context_modules': 'modules',
        'announcements': 'announcements',
        'gradebooks': 'grades'
    }

    return CONTROLLER_MAP.get(controller, 'other')
```

**Output:** `data/page_views/categorized_page_views.parquet`

---

## 4. Ingenieria de Features

El pipeline genera **8 conjuntos de features**, totalizando **203 features** para el modelo de alerta temprana.

### 4.1 Session Features (11 features)

**Archivo:** `scripts/calculate_session_features.py`

**Definicion de sesion:** Secuencia de page views con gap < 30 minutos entre ellos.

```python
SESSION_GAP_MINUTES = 30

def calculate_sessions(df_user):
    """Identifica sesiones usando gap de 30 minutos."""
    timestamps = pd.to_datetime(df_user['created_at']).sort_values()
    gaps = timestamps.diff().dt.total_seconds() / 60
    session_starts = gaps >= SESSION_GAP_MINUTES
    session_ids = session_starts.cumsum()
    return session_ids
```

**Features generados:**

| Feature | Descripcion | Formula |
|---------|-------------|---------|
| `session_count` | Numero total de sesiones | Count de session_ids unicos |
| `session_duration_mean` | Duracion promedio (min) | mean(duration) |
| `session_duration_std` | Variabilidad duracion | std(duration) |
| `session_duration_median` | Mediana duracion | median(duration) |
| `sessions_per_week` | Sesiones por semana | session_count / weeks |
| `views_per_session` | Vistas promedio por sesion | total_views / session_count |
| `short_sessions_pct` | % sesiones < 5 min | count(duration < 5) / total |
| `long_sessions_pct` | % sesiones > 30 min | count(duration > 30) / total |
| `total_views` | Total page views | count(page_views) |
| `total_time_min` | Tiempo total (min) | sum(duration) |
| `session_regularity` | Regularidad temporal | 1 - (std_gap / mean_gap) |

**Output:** `data/enriched_features/session_features.parquet`

---

### 4.2 Category Features (~50 features)

**Archivo:** `scripts/calculate_category_features.py`

**Categorias analizadas:**
- files, discussions, quizzes, assignments, pages, modules, grades, announcements, home

**Features por categoria:**

| Feature | Descripcion |
|---------|-------------|
| `{cat}_views` | Total vistas de la categoria |
| `{cat}_views_pct` | % del total de vistas |
| `{cat}_unique_resources` | Recursos unicos visitados |
| `{cat}_time_min` | Tiempo estimado (min) |

**Features derivados:**

```python
# Ratio contenido vs evaluacion
content_views = files_views + pages_views + discussions_views
assessment_views = quizzes_views + assignments_views
content_vs_assessment_ratio = content_views / assessment_views

# Frecuencia de chequeo de notas
grades_check_per_week = grades_views / span_weeks

# Tasa de participacion en discusiones
discussion_participation_rate = participated_count / discussion_views
```

**Output:** `data/enriched_features/category_features.parquet`

---

### 4.3 Proactivity Features (PCT Ranking) (~60 features)

**Archivo:** `scripts/calculate_proactivity_features.py`

**Concepto clave: PCT (Percentile) Ranking**

Para cada recurso en un curso, los estudiantes son rankeados por CUANDO accedieron por primera vez:

```python
def calculate_pct_rankings(df_course, enrolled_users):
    """
    Calcula ranking PCT para todos los recursos.

    PCT = (N - rank + 1) / N

    Donde:
    - N = numero de estudiantes que accedieron
    - rank = posicion temporal (1 = primero)

    Resultado:
    - Primer estudiante: PCT = 1.0
    - Ultimo estudiante: PCT ≈ 0
    - Nunca accedio: PCT = 0
    """
```

**Features por tipo de recurso:**

| Feature | Descripcion |
|---------|-------------|
| `{type}_n_resources` | Numero de recursos del tipo |
| `{type}_mean_pct` | Promedio PCT |
| `{type}_median_pct` | Mediana PCT |
| `{type}_std_pct` | Desviacion estandar PCT |
| `{type}_access_rate` | % recursos accedidos |
| `{type}_top25_rate` | % en top 25% |
| `{type}_top50_rate` | % en top 50% |

**Reduccion de dimensionalidad con histogramas:**

Dado que cada curso tiene diferente numero de recursos, usamos histogramas de 5 bins para representar la distribucion de PCT:

```python
# Bins del histograma
# b1: PCT = 0 (nunca accedio)
# b2: 0 < PCT <= 0.25 (cuartil inferior)
# b3: 0.25 < PCT <= 0.50 (segundo cuartil)
# b4: 0.50 < PCT <= 0.75 (tercer cuartil)
# b5: 0.75 < PCT <= 1.0 (cuartil superior)

features[f'{prefix}_hist_b1'] = np.sum(pct_array == 0) / n_resources
features[f'{prefix}_hist_b2'] = np.sum((pct_array > 0) & (pct_array <= 0.25)) / n_resources
features[f'{prefix}_hist_b3'] = np.sum((pct_array > 0.25) & (pct_array <= 0.50)) / n_resources
features[f'{prefix}_hist_b4'] = np.sum((pct_array > 0.50) & (pct_array <= 0.75)) / n_resources
features[f'{prefix}_hist_b5'] = np.sum(pct_array > 0.75) / n_resources
```

**DCT (Discrete Cosine Transform):**

Para capturar patrones en la secuencia de PCT:

```python
from scipy.fftpack import dct

def calculate_dct_features(pct_values, n_coeffs=4):
    """Aplica DCT a la secuencia de PCT."""
    pct_array = np.array(pct_values)
    pct_array = pct_array / pct_array.sum()  # Normalizar
    coeffs = dct(pct_array, norm='ortho')
    return {f'dct_pct_{i}': coeffs[i] for i in range(n_coeffs)}
```

**Output:** `data/enriched_features/proactivity_features.parquet`

---

### 4.4 PCA Features (~12 features)

**Archivo:** `scripts/calculate_pca_features.py`

**Solo materiales de aprendizaje (EXCLUYE quizzes y assignments):**

```python
RESOURCE_TYPES = ['files', 'discussions', 'pages', 'modules']

COMPONENTS = {
    'files': 3,       # 3 componentes principales
    'discussions': 3,
    'pages': 3,
    'modules': 2
}
```

**Proceso PCA por curso:**

```python
def extract_pca_features(df_type, enrolled_users, n_components, prefix):
    """
    1. Crear matriz PCT (estudiantes x recursos)
    2. Aplicar StandardScaler
    3. Aplicar PCA
    4. Extraer componentes fijos
    """
    # Crear pivot table
    pivot = df_type.pivot_table(
        index='user_id',
        columns='resource_id',
        values='pct',
        aggfunc='first'
    )

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(pivot.values)

    # PCA
    pca = PCA(n_components=actual_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, pca.explained_variance_ratio_
```

**Features generados:**

| Feature | Descripcion |
|---------|-------------|
| `files_pc1`, `files_pc2`, `files_pc3` | Componentes PCA archivos |
| `disc_pc1`, `disc_pc2`, `disc_pc3` | Componentes PCA discusiones |
| `pages_pc1`, `pages_pc2`, `pages_pc3` | Componentes PCA paginas |
| `mods_pc1`, `mods_pc2` | Componentes PCA modulos |
| `{type}_var_explained` | Varianza explicada por tipo |

**Output:** `data/enriched_features/pca_features.parquet`

---

### 4.5 Weekly Features (~20 features)

**Archivo:** `scripts/calculate_weekly_features.py`

**Calculo de semana del curso:**

```python
def get_course_week(timestamp, course_start):
    """Semana relativa al inicio del curso."""
    delta = (timestamp - course_start).days
    return max(1, (delta // 7) + 1)
```

**Features temporales:**

| Feature | Descripcion |
|---------|-------------|
| `active_weeks_count` | Semanas con actividad |
| `first_active_week` | Primera semana activa |
| `last_active_week` | Ultima semana activa |
| `peak_week` | Semana con mas actividad |
| `early_semester_views` | Vistas primera mitad |
| `late_semester_views` | Vistas segunda mitad |
| `early_vs_late_ratio` | Ratio primera/segunda mitad |
| `avg_week_over_week_change` | Cambio promedio semanal |
| `activity_consistency` | Coef. variacion semanal |
| `engagement_pattern` | Patron (1=temprano, 2=variable, 3=tarde, 4=consistente) |

**Output:** `data/enriched_features/weekly_features.parquet`

---

### 4.6 Engagement Ratios (~9 features)

**Archivo:** `scripts/calculate_engagement_ratios.py`

**Ratios calculados:**

| Feature | Formula |
|---------|---------|
| `content_vs_assessment_ratio` | (files + pages + disc) / (quiz + assi) |
| `discussion_participation_rate` | participated / discussion_views |
| `download_rate` | downloads / total_file_views |

---

### 4.7 N-gram Features (20 features) **NUEVO**

**Archivo:** `scripts/calculate_ngram_features.py`

**Concepto:** Captura secuencias de navegacion (transiciones entre tipos de recursos).

**Features de transicion (bigramas):**

```python
# Ejemplo: bigram_files_to_home
# Frecuencia de transiciones de Files -> Home
transition_matrix[from_type][to_type] / total_transitions
```

**Features generados:**

| Feature | Descripcion |
|---------|-------------|
| `bigram_X_to_Y` | Frecuencia de transicion de tipo X a tipo Y |
| `total_transitions` | Total de transiciones registradas |
| `transition_entropy` | Entropia de Shannon de transiciones |
| `transition_diversity` | Cantidad de tipos de transicion unicos |
| `self_loop_ratio` | % de transiciones al mismo tipo |
| `dominant_transition` | Transicion mas frecuente (codificada) |

**Output:** `data/enriched_features/ngram_features.parquet`

---

### 4.8 Graph Features (6 features) **NUEVO**

**Archivo:** `scripts/calculate_graph_features.py`

**Concepto:** Analiza el grafo de recursos accedidos por estudiante.

**Features generados:**

| Feature | Descripcion |
|---------|-------------|
| `unique_resources` | Recursos unicos accedidos |
| `resource_coverage` | % de recursos del curso accedidos |
| `resource_diversity` | Diversidad de tipos de recursos (Shannon) |
| `resources_vs_avg` | Recursos vs promedio del curso |
| `jaccard_to_passing` | Similitud Jaccard con estudiantes aprobados |
| `access_cluster` | Cluster de patron de acceso (K-means) |

**Output:** `data/enriched_features/graph_features.parquet`

---

### 4.9 Time Features (11 features) **NUEVO**

**Archivo:** `scripts/calculate_time_features.py`

**Concepto:** Patrones horarios de actividad (franjas del dia, fin de semana).

**Definicion de franjas:**

```python
HOUR_BINS = {
    'morning': (6, 12),      # 06:00 - 11:59
    'afternoon': (12, 18),   # 12:00 - 17:59
    'evening': (18, 24),     # 18:00 - 23:59
    'night': (0, 6)          # 00:00 - 05:59
}
```

**Features generados:**

| Feature | Descripcion |
|---------|-------------|
| `pct_morning/afternoon/evening/night` | Distribucion por franja |
| `pct_weekend` | % actividad en fin de semana |
| `work_hours_ratio` | % en horario laboral (9-18h L-V) |
| `late_night_ratio` | % actividad de madrugada (00-06h) |
| `peak_hour` | Hora de maxima actividad |
| `peak_day` | Dia de maxima actividad |
| `hour_diversity` | Diversidad de horarios (Shannon) |
| `time_consistency` | Consistencia horaria (1 - CV) |

**Output:** `data/enriched_features/time_features.parquet`

---

## 5. Seleccion de Features para Modelo

### 5.1 Features Incluidos (203 disponibles, ~40 despues de seleccion)

```python
LEARNING_MATERIAL_PREFIXES = [
    'file', 'disc', 'page', 'modu', 'mods', 'pages', 'files',
    'home', 'announcements', 'bigram', 'transition', 'resource',
    'pct_', 'hour_', 'time_', 'peak_'
]
```

**Categorias incluidas:**

| Categoria | Features | Descripcion |
|-----------|----------|-------------|
| Sesiones | 11 | Patrones de sesion |
| Vistas por categoria | 40 | Distribucion de actividad |
| Proactividad (PCT) | 80 | Rankings temporales |
| PCA | 19 | Componentes principales |
| Temporal (semanal) | 16 | Patrones semanales |
| N-gramas | 20 | Secuencias de navegacion |
| Grafo | 6 | Analisis de red |
| Tiempo (horario) | 11 | Patrones horarios |

**Seleccion de features:**
- Se excluyen features con importancia <0.5%
- Se eliminan features altamente correlacionados (>0.85)
- Resultado: ~40 features finales

### 5.2 Features Excluidos (54)

```python
EXCLUDE_PATTERNS = [
    'quiz', 'quizzes',      # Actividad en quizzes = evaluaciones
    'assi', 'assignment',   # Actividad en tareas = evaluaciones
    'grade', 'grad',        # Vistas de calificaciones = resultado
    'score',                # Cualquier puntaje
    'submission',           # Entregas
]
```

**Razon de exclusion:** Estos features revelan informacion sobre evaluaciones, lo que contamina la prediccion temprana. El objetivo es predecir fracaso ANTES de la primera evaluacion.

---

## 6. Entrenamiento del Modelo

### 6.1 Configuracion

**Archivo:** `scripts/train_optimized_early_warning.py`

```python
# Modelo base
XGBClassifier(
    eval_metric='logloss',
    verbosity=0,
    random_state=42
)

# Grid de hiperparametros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0],
    'min_child_weight': [1, 3]
}

# Validacion cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

### 6.2 Mejores Hiperparametros

```json
{
  "learning_rate": 0.2,
  "max_depth": 7,
  "min_child_weight": 1,
  "n_estimators": 100,
  "subsample": 0.8
}
```

### 6.3 Top 20 Features Predictivos

| Rank | Feature | Importancia | Categoria |
|------|---------|-------------|-----------|
| 1 | `total_time_min_znorm` | 5.97% | Sesiones |
| 2 | `page_n_resources` | 5.51% | Proactividad |
| 3 | `total_transitions` | 4.66% | N-gramas |
| 4 | `content_vs_assessment_ratio` | 4.39% | Categorias |
| 5 | `total_views_x_znorm` | 4.36% | Sesiones |
| 6 | `modules_views` | 4.02% | Categorias |
| 7 | `mods_n_resources` | 3.71% | PCA |
| 8 | `discussions_unique_resources` | 3.55% | Categorias |
| 9 | `modules_views_znorm` | 3.55% | Normalizado |
| 10 | `sessions_per_week_znorm` | 3.02% | Normalizado |
| 11 | `pages_pc1` | 2.88% | PCA |
| 12 | `total_time_min` | 2.80% | Sesiones |
| 13 | `files_pc1` | 2.74% | PCA |
| 14 | `last_active_week` | 2.61% | Semanal |
| 15 | `pages_var_explained` | 2.53% | PCA |
| 16 | `discussions_time_min_znorm` | 2.32% | Normalizado |
| 17 | `modu_mean_pct` | 2.32% | Proactividad |
| 18 | `hour_diversity` | 2.17% | Tiempo |
| 19 | `transition_entropy` | 2.14% | N-gramas |
| 20 | `files_views_pct` | 2.12% | Categorias |

---

## 7. Resultados del Modelo

### 7.1 Metricas de Rendimiento

| Modelo | ROC-AUC | Exactitud | Precision | Sensibilidad |
|--------|---------|-----------|-----------|--------------|
| Baseline (Actividad) | 0.787 | 74.0% | 69.7% | 61.7% |
| Regresion Logistica | 0.834 | 76.5% | 68.9% | 73.2% |
| Random Forest | 0.837 | 75.6% | 73.8% | 59.2% |
| **XGBoost Optimizado** | **0.860** | **78.4%** | **72.9%** | **71.8%** |

### 7.2 Matriz de Confusion

```
                    Prediccion
                    Aprueba  Reprueba
Real    Aprueba       181       38
        Reprueba       40      102
```

- **Verdaderos Negativos (VN):** 181 estudiantes correctamente predichos como aprobados
- **Falsos Positivos (FP):** 38 estudiantes aprobados incorrectamente marcados como en riesgo
- **Falsos Negativos (FN):** 40 estudiantes reprobados no detectados
- **Verdaderos Positivos (VP):** 102 estudiantes en riesgo correctamente identificados

### 7.3 Interpretacion

- **Precision 72.9%:** De cada 10 alertas, ~7 son correctas
- **Sensibilidad 71.8%:** Detectamos 7 de cada 10 estudiantes en riesgo
- **ROC-AUC 0.860:** Excelente capacidad discriminativa (+9.3% vs baseline)

---

## 8. Archivos del Pipeline

### 8.1 Scripts de Feature Engineering

| Script | Descripcion | Output |
|--------|-------------|--------|
| `extract_page_views_async.py` | Extraccion de datos via API | `page_views/*.parquet` |
| `categorize_page_views.py` | Categorizacion de URLs | `categorized_page_views.parquet` |
| `calculate_session_features.py` | Features de sesion | `session_features.parquet` |
| `calculate_category_features.py` | Features por categoria | `category_features.parquet` |
| `calculate_proactivity_features.py` | Features PCT | `proactivity_features.parquet` |
| `calculate_pca_features.py` | Componentes PCA | `pca_features.parquet` |
| `calculate_weekly_features.py` | Features temporales | `weekly_features.parquet` |
| `calculate_ngram_features.py` | Secuencias navegacion | `ngram_features.parquet` |
| `calculate_graph_features.py` | Analisis de grafo | `graph_features.parquet` |
| `calculate_time_features.py` | Patrones horarios | `time_features.parquet` |

### 8.2 Scripts de Modelado

| Script | Descripcion | Output |
|--------|-------------|--------|
| `train_learning_material_model.py` | Modelo basico | Metricas en consola |
| `train_optimized_early_warning.py` | Modelo optimizado | `early_warning_model_metrics.json` |
| `generate_early_warning_visualizations.py` | Visualizaciones | `visualizations/*.png` |

### 8.3 Archivos de Datos

```
data/
├── page_views/
│   ├── page_views_2024_12_*.parquet    # Datos crudos
│   ├── categorized_page_views.parquet   # Datos categorizados
│   └── student_enrollments.csv          # Inscripciones + notas
├── enriched_features/
│   ├── session_features.parquet         # 11 features
│   ├── category_features.parquet        # 40 features
│   ├── proactivity_features.parquet     # 80 features
│   ├── pca_features.parquet             # 19 features
│   ├── weekly_features.parquet          # 16 features
│   ├── ngram_features.parquet           # 20 features (NUEVO)
│   ├── graph_features.parquet           # 6 features (NUEVO)
│   └── time_features.parquet            # 11 features (NUEVO)
├── models/
│   └── v4_optimized/                     # Carpeta organizada (NUEVO)
│       ├── README.md
│       ├── features/                     # Copia de features
│       ├── results/                      # Metricas del modelo
│       ├── visualizations/               # Graficos
│       └── scripts/                      # Scripts de entrenamiento
└── report/
    ├── early_warning_model_metrics.json
    └── visualizations/
        ├── roc_curves_early_warning.png
        ├── feature_importance_early_warning.png
        ├── confusion_matrix_early_warning.png
        └── model_comparison_early_warning.png
```

---

## 9. Reproduccion del Pipeline

### 9.1 Requisitos

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy
```

### 9.2 Ejecucion Completa

```bash
# 1. Extraer page views (requiere credenciales Canvas)
python scripts/extract_page_views_async.py

# 2. Categorizar page views
python scripts/categorize_page_views.py

# 3. Calcular features (en orden)
python scripts/calculate_session_features.py
python scripts/calculate_category_features.py
python scripts/calculate_proactivity_features.py
python scripts/calculate_pca_features.py
python scripts/calculate_weekly_features.py
python scripts/calculate_ngram_features.py     # NUEVO
python scripts/calculate_graph_features.py     # NUEVO
python scripts/calculate_time_features.py      # NUEVO

# 4. Entrenar modelo optimizado
python scripts/train_optimized_early_warning.py

# 5. Generar visualizaciones
python scripts/generate_early_warning_visualizations.py
```

---

## 10. Limitaciones y Trabajo Futuro

### 10.1 Limitaciones Actuales

1. **Datos de un solo semestre:** Modelo entrenado con 361 estudiantes de 10 cursos
2. **Cursos de postgrado:** Mayoria de cursos son de postgrado, puede no generalizar a pregrado
3. **Sin validacion temporal:** Falta prueba con cohortes futuras

### 10.2 Trabajo Futuro

1. **Validacion temporal:** Entrenar con semestre 1, validar con semestre 2
2. **Expansion a pregrado:** Incluir cursos de pregrado para mayor diversidad
3. **Modelo por tipo de curso:** Modelos especializados por area academica
4. **Integracion en produccion:** Pipeline automatizado para alertas en tiempo real

---

*Documento actualizado: Enero 2026*
*Version: v4 (203 features)*
*Autores: Equipo de Analitica Academica, Universidad Autonoma de Chile*
