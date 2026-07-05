#!/usr/bin/env python3
"""
Generate Spanish executive report on resource type impact.

Creates visualizations and markdown report comparing Aprobados vs Reprobados
engagement patterns by resource type.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from datetime import datetime

# Input/Output paths
ANALYSIS_FILE = Path('/home/paul/projects/uautonoma/data/analysis/resource_impact_analysis.json')
CATEGORY_FEATURES = Path('/home/paul/projects/uautonoma/data/enriched_features/category_features.parquet')
ENGAGEMENT_RATIOS = Path('/home/paul/projects/uautonoma/data/enriched_features/engagement_ratios.parquet')
ENROLLMENTS = Path('/home/paul/projects/uautonoma/data/page_views/student_enrollments.csv')
OUTPUT_DIR = Path('/home/paul/projects/uautonoma/data/report')
VIZ_DIR = OUTPUT_DIR / 'visualizations' / 'resources'
REPORT_FILE = OUTPUT_DIR / 'INFORME_IMPACTO_RECURSOS_LMS.md'

# Ensure directories exist
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'aprobados': '#2ecc71', 'reprobados': '#e74c3c'}
PASS_THRESHOLD = 57.0

# Resource type names in Spanish
RESOURCE_NAMES_ES = {
    'quizzes': 'Cuestionarios',
    'modules': 'Módulos',
    'discussions': 'Foros de Discusión',
    'files': 'Archivos',
    'assignments': 'Tareas',
    'pages': 'Páginas',
    'grades': 'Calificaciones',
    'announcements': 'Anuncios',
    'home': 'Página Inicial'
}

# Course names
COURSE_NAMES = {
    79875: 'TALLER DE COMP DIGITALES-P01',
    79913: 'FUND. DE BUSINESS ANALYTICS-P01',
    84936: 'FUNDAMENTOS DE MICROECONOMÍA-P03',
    84941: 'FUNDAMENTOS DE MICROECONOMÍA-P01',
    84944: 'FUNDAMENTOS DE MACROECONOMÍA-P03',
    86020: 'TALL DE COMPETENCIAS DIGITALES-P02',
    86676: 'FUND DE BUSINESS ANALYTICS-P01',
    88381: 'MATEMÁTICAS PARA LOS NEGOCIOS-P01',
    89099: 'TALLER DE COMP DIGITALES-P01',
    89390: 'GESTIÓN DEL TALENTO-P01',
}


def load_data():
    """Load all required data."""
    with open(ANALYSIS_FILE, 'r') as f:
        analysis = json.load(f)

    category_df = pd.read_parquet(CATEGORY_FEATURES)
    enrollments_df = pd.read_csv(ENROLLMENTS)

    # Merge
    df = category_df.merge(
        enrollments_df[['user_id', 'course_id', 'final_score']],
        on=['user_id', 'course_id'],
        how='left'
    )
    df['passed'] = df['final_score'] >= PASS_THRESHOLD
    df = df.dropna(subset=['final_score'])

    # Load engagement ratios if available
    try:
        ratios_df = pd.read_parquet(ENGAGEMENT_RATIOS)
    except:
        ratios_df = None

    return analysis, df, ratios_df


def create_bar_chart_comparison(df, analysis):
    """Create bar chart comparing mean engagement by resource type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    resources = ['quizzes', 'assignments', 'discussions', 'home', 'grades', 'files', 'modules', 'announcements', 'pages']
    x = np.arange(len(resources))
    width = 0.35

    means_apr = []
    means_rep = []
    significant = []

    for r in resources:
        key = f'{r}_views'
        if key in analysis['resource_analysis']:
            data = analysis['resource_analysis'][key]
            means_apr.append(data['mean_aprobados'])
            means_rep.append(data['mean_reprobados'])
            significant.append(data['significant'])
        else:
            means_apr.append(0)
            means_rep.append(0)
            significant.append(False)

    bars1 = ax.bar(x - width/2, means_apr, width, label='Aprobados', color=COLORS['aprobados'])
    bars2 = ax.bar(x + width/2, means_rep, width, label='Reprobados', color=COLORS['reprobados'])

    # Add significance markers
    for i, sig in enumerate(significant):
        if sig:
            max_val = max(means_apr[i], means_rep[i])
            ax.text(i, max_val + 2, '***', ha='center', fontsize=12, fontweight='bold')

    ax.set_ylabel('Promedio de Visualizaciones', fontsize=12)
    ax.set_xlabel('Tipo de Recurso', fontsize=12)
    ax.set_title('Engagement por Tipo de Recurso: Aprobados vs Reprobados', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([RESOURCE_NAMES_ES.get(r, r) for r in resources], rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'resource_comparison_bar.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_risk_factors_chart(analysis):
    """Create horizontal bar chart of risk factors."""
    risk_factors = analysis.get('risk_factors', [])
    if not risk_factors:
        return

    # Filter to significant only
    significant_rf = [rf for rf in risk_factors if rf.get('significant', False)]
    significant_rf = sorted(significant_rf, key=lambda x: x['risk_ratio'], reverse=True)

    if not significant_rf:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    resources = [RESOURCE_NAMES_ES.get(rf['resource_type'], rf['resource_type']) for rf in significant_rf]
    risk_ratios = [rf['risk_ratio'] for rf in significant_rf]

    colors = ['#e74c3c' if rr > 2 else '#f39c12' if rr > 1.5 else '#3498db' for rr in risk_ratios]

    bars = ax.barh(resources, risk_ratios, color=colors)

    # Add value labels
    for bar, val in zip(bars, risk_ratios):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
               f'{val:.2f}x', va='center', fontsize=10, fontweight='bold')

    ax.axvline(x=1, color='gray', linestyle='--', linewidth=1, label='Sin diferencia (1.0x)')
    ax.axvline(x=2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Riesgo doble (2.0x)')

    ax.set_xlabel('Ratio de Riesgo (Bajo Engagement / Alto Engagement)', fontsize=12)
    ax.set_title('Factores de Riesgo por Tipo de Recurso', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'risk_factors_bar.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_boxplots(df):
    """Create box plots comparing distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 10))
    axes = axes.flatten()

    resources = ['quizzes_views', 'assignments_views', 'discussions_views',
                'home_views', 'grades_views', 'files_views']

    for i, col in enumerate(resources):
        if col not in df.columns:
            continue

        ax = axes[i]

        # Prepare data
        data_apr = df[df['passed'] == True][col].dropna()
        data_rep = df[df['passed'] == False][col].dropna()

        bp = ax.boxplot([data_apr, data_rep],
                       labels=['Aprobados', 'Reprobados'],
                       patch_artist=True)

        bp['boxes'][0].set_facecolor(COLORS['aprobados'])
        bp['boxes'][1].set_facecolor(COLORS['reprobados'])

        resource_name = col.replace('_views', '')
        ax.set_title(RESOURCE_NAMES_ES.get(resource_name, resource_name), fontsize=12, fontweight='bold')
        ax.set_ylabel('Visualizaciones')

    plt.suptitle('Distribución de Engagement por Tipo de Recurso', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'resource_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_correlation_heatmap(df):
    """Create correlation heatmap between resources and grade."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Select relevant columns
    cols = ['final_score', 'quizzes_views', 'assignments_views', 'discussions_views',
           'home_views', 'grades_views', 'files_views', 'modules_views', 'pages_views']
    cols = [c for c in cols if c in df.columns]

    corr_matrix = df[cols].corr()

    # Rename for display
    rename_map = {
        'final_score': 'Nota Final',
        'quizzes_views': 'Cuestionarios',
        'assignments_views': 'Tareas',
        'discussions_views': 'Foros',
        'home_views': 'Pág. Inicial',
        'grades_views': 'Calificaciones',
        'files_views': 'Archivos',
        'modules_views': 'Módulos',
        'pages_views': 'Páginas'
    }
    corr_matrix = corr_matrix.rename(index=rename_map, columns=rename_map)

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
               cmap='RdYlGn', center=0, ax=ax,
               annot_kws={'fontsize': 10})

    ax.set_title('Correlación: Engagement por Recurso vs Nota Final', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'resource_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def create_effect_size_chart(analysis):
    """Create chart showing effect sizes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Get effect sizes
    effects = []
    for key, data in analysis.get('resource_analysis', {}).items():
        if '_views' in key and data.get('significant', False):
            resource = key.replace('_views', '')
            effects.append({
                'resource': RESOURCE_NAMES_ES.get(resource, resource),
                'cohens_d': abs(data.get('cohens_d', 0)),
                'direction': '+' if data.get('mean_aprobados', 0) > data.get('mean_reprobados', 0) else '-'
            })

    if not effects:
        return

    effects = sorted(effects, key=lambda x: x['cohens_d'], reverse=True)

    resources = [e['resource'] for e in effects]
    d_values = [e['cohens_d'] for e in effects]
    colors = ['#2ecc71' if e['direction'] == '+' else '#e74c3c' for e in effects]

    bars = ax.barh(resources, d_values, color=colors)

    # Reference lines
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Pequeño (0.2)')
    ax.axvline(x=0.5, color='orange', linestyle='--', alpha=0.5, label='Medio (0.5)')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.5, label='Grande (0.8)')

    ax.set_xlabel("Tamaño del Efecto (Cohen's d)", fontsize=12)
    ax.set_title('Magnitud del Efecto: Aprobados vs Reprobados', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'effect_sizes.png', dpi=150, bbox_inches='tight')
    plt.close()


def generate_markdown_report(analysis, df):
    """Generate the Spanish markdown report."""
    info = analysis.get('dataset_info', {})
    resource_analysis = analysis.get('resource_analysis', {})
    risk_factors = analysis.get('risk_factors', [])

    # Sort risk factors
    risk_factors_sorted = sorted([rf for rf in risk_factors if rf.get('significant', False)],
                                  key=lambda x: x['risk_ratio'], reverse=True)

    # Get top significant findings
    significant = [(k, v) for k, v in resource_analysis.items()
                  if v.get('significant', False) and '_views' in k]
    significant = sorted(significant, key=lambda x: abs(x[1].get('cohens_d', 0)), reverse=True)

    report = f"""# Informe: Impacto de Recursos LMS en Rendimiento Académico

## Universidad Autónoma de Chile - Canvas LMS

**Fecha:** {datetime.now().strftime('%d de %B de %Y').replace('January', 'enero').replace('February', 'febrero').replace('March', 'marzo').replace('April', 'abril').replace('May', 'mayo').replace('June', 'junio').replace('July', 'julio').replace('August', 'agosto').replace('September', 'septiembre').replace('October', 'octubre').replace('November', 'noviembre').replace('December', 'diciembre')}
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
| **Estudiantes analizados** | {info.get('total_students', 0)} |
| **Cursos evaluados** | {info.get('n_courses', 0)} |
| **Tasa de aprobación** | {info.get('pass_rate', 0)*100:.1f}% |
| **Tipos de recurso analizados** | 9 |
| **Factores de riesgo significativos** | {len(risk_factors_sorted)} |

---

### Top 3 Factores de Riesgo

"""

    for i, rf in enumerate(risk_factors_sorted[:3], 1):
        resource_es = RESOURCE_NAMES_ES.get(rf['resource_type'], rf['resource_type'])
        report += f"""
| **{i}. {resource_es}** |
|---|
| Bajo engagement: **{rf['failure_rate_low_engagement']*100:.1f}%** tasa de fracaso |
| Alto engagement: **{rf['failure_rate_high_engagement']*100:.1f}%** tasa de fracaso |
| **Ratio de riesgo: {rf['risk_ratio']:.2f}x** |

"""

    report += """
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
"""

    for key in ['quizzes_views', 'assignments_views', 'discussions_views', 'home_views',
               'grades_views', 'files_views', 'modules_views', 'announcements_views', 'pages_views']:
        if key in resource_analysis:
            data = resource_analysis[key]
            resource_es = RESOURCE_NAMES_ES.get(key.replace('_views', ''), key)
            sig = '**Sí**' if data.get('significant', False) else 'No'
            report += f"| {resource_es} | {data.get('mean_aprobados', 0):.1f} | {data.get('mean_reprobados', 0):.1f} | {data.get('p_value', 1):.4f} | {data.get('cohens_d', 0):.2f} | {sig} |\n"

    report += """

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

"""

    for key, data in significant[:5]:
        resource_es = RESOURCE_NAMES_ES.get(key.replace('_views', ''), key)
        direction = "más" if data.get('mean_aprobados', 0) > data.get('mean_reprobados', 0) else "menos"
        diff = abs(data.get('mean_aprobados', 0) - data.get('mean_reprobados', 0))
        report += f"""
### {resource_es}

> Los estudiantes **aprobados** acceden en promedio **{diff:.1f} veces {direction}** a {resource_es.lower()} que los reprobados.
>
> - Media Aprobados: **{data.get('mean_aprobados', 0):.1f}**
> - Media Reprobados: **{data.get('mean_reprobados', 0):.1f}**
> - Tamaño del efecto: **d = {data.get('cohens_d', 0):.2f}** ({data.get('effect_size_interpretation', 'N/A')})

"""

    report += """
## 4.3 Correlaciones con Nota Final

![Matriz de Correlación](visualizations/resources/resource_correlation_heatmap.png)

---

# 5. Factores de Riesgo por Tipo de Recurso

## 5.1 Visualización de Riesgos

![Factores de Riesgo](visualizations/resources/risk_factors_bar.png)

## 5.2 Tabla de Factores de Riesgo

| Recurso | Falla (Bajo Eng.) | Falla (Alto Eng.) | Ratio Riesgo | Significativo |
|---------|-------------------|-------------------|--------------|---------------|
"""

    for rf in risk_factors_sorted:
        resource_es = RESOURCE_NAMES_ES.get(rf['resource_type'], rf['resource_type'])
        report += f"| {resource_es} | {rf['failure_rate_low_engagement']*100:.1f}% | {rf['failure_rate_high_engagement']*100:.1f}% | **{rf['risk_ratio']:.2f}x** | {'**Sí**' if rf.get('significant', False) else 'No'} |\n"

    report += """

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
"""

    return report


def main():
    print('=' * 60)
    print('Generating Resource Impact Report')
    print('=' * 60)
    print()

    # Load data
    print('Loading data...')
    analysis, df, ratios_df = load_data()
    print(f'  Loaded {len(df)} student records')
    print()

    # Create visualizations
    print('Creating visualizations...')

    print('  - Bar chart comparison...')
    create_bar_chart_comparison(df, analysis)

    print('  - Risk factors chart...')
    create_risk_factors_chart(analysis)

    print('  - Box plots...')
    create_boxplots(df)

    print('  - Correlation heatmap...')
    create_correlation_heatmap(df)

    print('  - Effect sizes chart...')
    create_effect_size_chart(analysis)

    print()

    # Generate report
    print('Generating markdown report...')
    report = generate_markdown_report(analysis, df)

    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f'  Saved to {REPORT_FILE}')
    print()
    print('Done!')
    print()
    print(f'Visualizations saved to: {VIZ_DIR}')
    print(f'Report saved to: {REPORT_FILE}')


if __name__ == '__main__':
    main()
