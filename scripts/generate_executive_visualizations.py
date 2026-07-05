#!/usr/bin/env python3
"""
Generate executive report visualizations:
1. Executive summary infographic
2. Fredricks' 3 dimensions diagram
3. Zimmerman's SRL cycle diagram
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch, Arc
import numpy as np
import os

# Output directory
OUTPUT_DIR = '/home/paul/projects/uautonoma/data/report/visualizations/executive'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color palette
COLORS = {
    'primary': '#1E3A5F',      # Dark blue
    'secondary': '#2E86AB',    # Medium blue
    'accent': '#F18F01',       # Orange
    'success': '#28A745',      # Green
    'danger': '#DC3545',       # Red
    'light': '#F8F9FA',        # Light gray
    'dark': '#343A40',         # Dark gray
    'behavioral': '#2E86AB',   # Blue
    'emotional': '#E74C3C',    # Red
    'cognitive': '#27AE60',    # Green
}


def create_executive_infographic():
    """Create executive summary infographic with key numbers."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(7, 9.5, 'RESUMEN EJECUTIVO', fontsize=24, fontweight='bold',
            ha='center', color=COLORS['primary'])
    ax.text(7, 9.0, 'Predicción de Fracaso Estudiantil mediante Engagement Digital',
            fontsize=14, ha='center', color=COLORS['dark'])

    # Key numbers boxes
    boxes = [
        {'x': 1.5, 'y': 7, 'value': '373', 'label': 'Estudiantes\nAnalizados', 'color': COLORS['secondary']},
        {'x': 4.5, 'y': 7, 'value': '10', 'label': 'Cursos\nEvaluados', 'color': COLORS['secondary']},
        {'x': 7.5, 'y': 7, 'value': '54', 'label': 'Indicadores de\nEngagement', 'color': COLORS['secondary']},
        {'x': 10.5, 'y': 7, 'value': '8', 'label': 'Factores de\nRiesgo (p<0.05)', 'color': COLORS['accent']},
    ]

    for box in boxes:
        # Box background
        rect = FancyBboxPatch((box['x']-1.2, box['y']-0.8), 2.4, 1.6,
                              boxstyle="round,pad=0.05",
                              facecolor=box['color'], edgecolor='none', alpha=0.9)
        ax.add_patch(rect)
        # Value
        ax.text(box['x'], box['y']+0.2, box['value'], fontsize=28, fontweight='bold',
                ha='center', va='center', color='white')
        # Label
        ax.text(box['x'], box['y']-0.5, box['label'], fontsize=10,
                ha='center', va='center', color='white', linespacing=1.2)

    # Model performance section
    ax.text(3.5, 5.2, 'CAPACIDAD PREDICTIVA', fontsize=16, fontweight='bold',
            ha='center', color=COLORS['primary'])

    perf_data = [
        ('ROC-AUC', '0.787', 'Buena discriminación'),
        ('Recall', '61.7%', '2 de 3 en riesgo detectados'),
        ('Precisión', '69.7%', '7 de 10 alertas correctas'),
    ]

    for i, (metric, value, desc) in enumerate(perf_data):
        y = 4.5 - i * 0.8
        ax.text(1.5, y, metric + ':', fontsize=12, fontweight='bold', ha='left', color=COLORS['dark'])
        ax.text(3.2, y, value, fontsize=14, fontweight='bold', ha='left', color=COLORS['accent'])
        ax.text(4.2, y, desc, fontsize=10, ha='left', color=COLORS['dark'])

    # Main finding box
    rect = FancyBboxPatch((7.5, 3.0), 5.5, 2.5,
                          boxstyle="round,pad=0.1",
                          facecolor=COLORS['danger'], edgecolor='none', alpha=0.1)
    ax.add_patch(rect)

    ax.text(10.25, 5.2, 'HALLAZGO PRINCIPAL', fontsize=14, fontweight='bold',
            ha='center', color=COLORS['danger'])
    ax.text(10.25, 4.5, '2x', fontsize=48, fontweight='bold',
            ha='center', color=COLORS['danger'])
    ax.text(10.25, 3.8, 'Mayor riesgo de fracaso', fontsize=12,
            ha='center', color=COLORS['dark'])
    ax.text(10.25, 3.4, 'en estudiantes con baja', fontsize=10,
            ha='center', color=COLORS['dark'])
    ax.text(10.25, 3.1, 'frecuencia de sesiones semanales', fontsize=10,
            ha='center', color=COLORS['dark'])

    # Risk factors section
    ax.text(7, 2.2, 'TOP 5 FACTORES DE RIESGO', fontsize=14, fontweight='bold',
            ha='center', color=COLORS['primary'])

    factors = [
        ('Baja frecuencia sesiones/semana', '2.01x'),
        ('Bajo total de visualizaciones', '1.93x'),
        ('Pocas sesiones de estudio', '1.82x'),
        ('Sin estudio fines de semana', '1.81x'),
        ('Sin estudio vespertino', '1.76x'),
    ]

    for i, (factor, rr) in enumerate(factors):
        y = 1.7 - i * 0.35
        # Bar
        bar_width = float(rr[:-1]) / 2.5 * 5  # Scale to fit
        rect = FancyBboxPatch((1, y-0.12), bar_width, 0.24,
                              boxstyle="round,pad=0.02",
                              facecolor=COLORS['danger'], edgecolor='none',
                              alpha=0.3 + 0.1 * (5-i))
        ax.add_patch(rect)
        ax.text(1.1, y, f'{i+1}. {factor}', fontsize=9, ha='left', va='center', color=COLORS['dark'])
        ax.text(bar_width + 1.2, y, rr, fontsize=10, fontweight='bold', ha='left', va='center', color=COLORS['danger'])

    # Footer
    ax.text(7, 0.1, 'Universidad Autónoma de Chile - Canvas LMS - Diciembre 2025',
            fontsize=9, ha='center', color=COLORS['dark'], alpha=0.6)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/executive_summary_infographic.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Created: {OUTPUT_DIR}/executive_summary_infographic.png")


def create_fredricks_diagram():
    """Create Fredricks' 3 dimensions of engagement diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'TRES DIMENSIONES DEL STUDENT ENGAGEMENT', fontsize=18, fontweight='bold',
            ha='center', color=COLORS['primary'])
    ax.text(6, 9.0, 'Modelo de Fredricks, Blumenfeld & Paris (2004)', fontsize=12,
            ha='center', color=COLORS['dark'], style='italic')

    # Central circle - Student Engagement
    center_circle = Circle((6, 6), 1.2, facecolor=COLORS['primary'], edgecolor='none', alpha=0.9)
    ax.add_patch(center_circle)
    ax.text(6, 6.2, 'STUDENT', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(6, 5.8, 'ENGAGEMENT', fontsize=12, fontweight='bold', ha='center', color='white')

    # Three dimension circles
    dimensions = [
        {'x': 2.5, 'y': 6.5, 'name': 'CONDUCTUAL', 'question': '¿Qué HACE?',
         'color': COLORS['behavioral'], 'indicators': ['Sesiones', 'Page views', 'Participaciones']},
        {'x': 9.5, 'y': 6.5, 'name': 'EMOCIONAL', 'question': '¿Cómo se SIENTE?',
         'color': COLORS['emotional'], 'indicators': ['Estudio fin de semana', 'Horarios vespertinos', 'Compromiso voluntario']},
        {'x': 6, 'y': 2.5, 'name': 'COGNITIVO', 'question': '¿Cómo se IMPLICA?',
         'color': COLORS['cognitive'], 'indicators': ['Trayectoria engagement', 'Respuesta a demandas', 'Autorregulación']},
    ]

    for dim in dimensions:
        # Main circle
        circle = Circle((dim['x'], dim['y']), 1.5, facecolor=dim['color'], edgecolor='none', alpha=0.85)
        ax.add_patch(circle)

        # Dimension name
        ax.text(dim['x'], dim['y']+0.5, dim['name'], fontsize=13, fontweight='bold',
                ha='center', color='white')
        ax.text(dim['x'], dim['y']-0.1, dim['question'], fontsize=10,
                ha='center', color='white')

        # Indicators box
        if dim['x'] < 6:
            box_x = dim['x'] - 2.3
            align = 'left'
        elif dim['x'] > 6:
            box_x = dim['x'] + 2.3
            align = 'right'
        else:
            box_x = dim['x']
            align = 'center'

        if dim['y'] > 5:
            box_y = dim['y'] - 2.5
        else:
            box_y = dim['y'] - 2.0

        # Draw indicators
        indicators_text = '\n'.join([f'• {ind}' for ind in dim['indicators']])

        if dim['x'] < 6:  # Left
            ax.text(0.3, dim['y']-0.3, 'Nuestros indicadores:', fontsize=9,
                    fontweight='bold', ha='left', color=dim['color'])
            ax.text(0.3, dim['y']-0.7, indicators_text, fontsize=9,
                    ha='left', va='top', color=COLORS['dark'], linespacing=1.5)
        elif dim['x'] > 6:  # Right
            ax.text(11.7, dim['y']-0.3, 'Nuestros indicadores:', fontsize=9,
                    fontweight='bold', ha='right', color=dim['color'])
            ax.text(11.7, dim['y']-0.7, indicators_text, fontsize=9,
                    ha='right', va='top', color=COLORS['dark'], linespacing=1.5)
        else:  # Bottom
            ax.text(6, 0.5, 'Nuestros indicadores:', fontsize=9,
                    fontweight='bold', ha='center', color=dim['color'])
            ax.text(6, 0.2, '  •  '.join(dim['indicators']), fontsize=9,
                    ha='center', color=COLORS['dark'])

    # Connecting lines
    ax.annotate('', xy=(3.8, 6.3), xytext=(4.9, 6.1),
                arrowprops=dict(arrowstyle='-', color=COLORS['dark'], alpha=0.5, lw=2))
    ax.annotate('', xy=(8.2, 6.3), xytext=(7.1, 6.1),
                arrowprops=dict(arrowstyle='-', color=COLORS['dark'], alpha=0.5, lw=2))
    ax.annotate('', xy=(6, 3.8), xytext=(6, 4.9),
                arrowprops=dict(arrowstyle='-', color=COLORS['dark'], alpha=0.5, lw=2))

    # Reference
    ax.text(6, 0.1, 'Fredricks, J. A., Blumenfeld, P. C., & Paris, A. H. (2004). Review of Educational Research.',
            fontsize=8, ha='center', color=COLORS['dark'], alpha=0.6, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/fredricks_3_dimensions.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Created: {OUTPUT_DIR}/fredricks_3_dimensions.png")


def create_zimmerman_srl_diagram():
    """Create Zimmerman's Self-Regulated Learning cycle diagram."""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(6, 9.5, 'CICLO DE APRENDIZAJE AUTORREGULADO', fontsize=18, fontweight='bold',
            ha='center', color=COLORS['primary'])
    ax.text(6, 9.0, 'Modelo de Zimmerman (2000)', fontsize=12,
            ha='center', color=COLORS['dark'], style='italic')

    # Phase boxes
    phases = [
        {'x': 6, 'y': 7.5, 'name': 'PLANIFICACIÓN', 'subtitle': 'Antes de actuar',
         'color': '#3498DB', 'components': ['Establecer metas', 'Planificar estrategia', 'Autoeficacia'],
         'indicator': 'first_access_day\n(inicio temprano)'},
        {'x': 10, 'y': 4.5, 'name': 'EJECUCIÓN', 'subtitle': 'Durante la acción',
         'color': '#E67E22', 'components': ['Atención focalizada', 'Uso de estrategias', 'Monitoreo'],
         'indicator': 'session_regularity\nsessions_per_week'},
        {'x': 2, 'y': 4.5, 'name': 'AUTORREFLEXIÓN', 'subtitle': 'Después de actuar',
         'color': '#27AE60', 'components': ['Autoevaluación', 'Atribuciones', 'Adaptación'],
         'indicator': 'engagement_velocity\n(¿mejora?)'},
    ]

    for phase in phases:
        # Main box
        rect = FancyBboxPatch((phase['x']-1.8, phase['y']-1.2), 3.6, 2.4,
                              boxstyle="round,pad=0.1",
                              facecolor=phase['color'], edgecolor='none', alpha=0.9)
        ax.add_patch(rect)

        # Phase name
        ax.text(phase['x'], phase['y']+0.7, phase['name'], fontsize=14, fontweight='bold',
                ha='center', color='white')
        ax.text(phase['x'], phase['y']+0.2, phase['subtitle'], fontsize=10,
                ha='center', color='white', style='italic')

        # Components
        components_text = '\n'.join([f'• {c}' for c in phase['components']])
        ax.text(phase['x'], phase['y']-0.7, components_text, fontsize=9,
                ha='center', va='top', color='white', linespacing=1.3)

    # Arrows forming cycle
    # Planificación -> Ejecución
    ax.annotate('', xy=(8.3, 6.0), xytext=(7.5, 6.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    # Ejecución -> Autorreflexión
    ax.annotate('', xy=(3.7, 4.0), xytext=(8.3, 4.0),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2,
                               connectionstyle='arc3,rad=-0.3'))

    # Autorreflexión -> Planificación
    ax.annotate('', xy=(4.5, 6.8), xytext=(3.7, 5.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['dark'], lw=2,
                               connectionstyle='arc3,rad=-0.2'))

    # Indicator boxes
    indicator_boxes = [
        {'x': 6, 'y': 5.0, 'phase': 'PLANIFICACIÓN', 'indicator': 'first_access_day\nfirst_module_day', 'color': '#3498DB'},
        {'x': 9, 'y': 2.2, 'phase': 'EJECUCIÓN', 'indicator': 'session_regularity\nsessions_per_week', 'color': '#E67E22'},
        {'x': 3, 'y': 2.2, 'phase': 'AUTORREFLEXIÓN', 'indicator': 'engagement_velocity\ntrend_reversals', 'color': '#27AE60'},
    ]

    for box in indicator_boxes:
        # Box
        rect = FancyBboxPatch((box['x']-1.3, box['y']-0.6), 2.6, 1.2,
                              boxstyle="round,pad=0.05",
                              facecolor=box['color'], edgecolor='none', alpha=0.2)
        ax.add_patch(rect)

        ax.text(box['x'], box['y']+0.3, 'Nuestro indicador:', fontsize=8,
                ha='center', color=box['color'], fontweight='bold')
        ax.text(box['x'], box['y']-0.2, box['indicator'], fontsize=9,
                ha='center', va='top', color=COLORS['dark'], family='monospace')

    # Central text
    ax.text(6, 4.0, 'CICLO\nCONTINUO', fontsize=11, fontweight='bold',
            ha='center', va='center', color=COLORS['dark'], alpha=0.5)

    # Key insight box
    rect = FancyBboxPatch((1, 0.3), 10, 1.2,
                          boxstyle="round,pad=0.1",
                          facecolor=COLORS['light'], edgecolor=COLORS['primary'], alpha=0.5)
    ax.add_patch(rect)

    ax.text(6, 1.1, 'INSIGHT: Los estudiantes con baja autorregulación muestran:', fontsize=10,
            ha='center', fontweight='bold', color=COLORS['primary'])
    ax.text(6, 0.6, 'Inicio tardío (mala planificación) → Sesiones irregulares (ejecución pobre) → Engagement decreciente (sin adaptación)',
            fontsize=9, ha='center', color=COLORS['dark'])

    # Reference
    ax.text(6, 0.05, 'Zimmerman, B. J. (2000). Handbook of Self-Regulation.',
            fontsize=8, ha='center', color=COLORS['dark'], alpha=0.6, style='italic')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/zimmerman_srl_cycle.png', dpi=150,
                bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Created: {OUTPUT_DIR}/zimmerman_srl_cycle.png")


if __name__ == '__main__':
    print("Generating executive report visualizations...")
    create_executive_infographic()
    create_fredricks_diagram()
    create_zimmerman_srl_diagram()
    print(f"\nAll visualizations saved to: {OUTPUT_DIR}/")
