# Guía: Conversión de Markdown a DOCX Profesional con Claude Code

Esta guía documenta el proceso para crear documentos Word profesionales a partir de archivos Markdown utilizando Claude Code y el script `convert_md_to_docx.py`.

---

## Resumen del Proceso

```
1. Escribir contenido en Markdown (.md)
2. Preparar imágenes en la carpeta del reporte
3. Ejecutar script de conversión
4. Revisar y ajustar si es necesario
```

---

## 1. Estructura del Archivo Markdown

### 1.1 Encabezado del Documento

El script extrae automáticamente metadatos del documento. Usa este formato al inicio:

```markdown
# Título Principal del Documento
## Subtítulo o Descripción

**Universidad Autónoma de Chile**
**Fecha:** 5 de enero de 2026
**Versión:** 2.0

---
```

El script detecta:
- `# Título` → Título de portada
- `## Subtítulo` → Subtítulo de portada
- `**Fecha:**` → Fecha en portada
- `**Versión:**` → Versión en portada

### 1.2 Tabla de Contenidos (Opcional en MD)

Si incluyes una tabla de contenidos manual en el Markdown, el script la **filtra automáticamente** porque genera una TOC propia en el DOCX:

```markdown
## Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Metodología](#2-metodología)
...
```

**Recomendación:** Incluirla en el MD para referencia durante la edición, pero saber que no aparecerá duplicada en el DOCX.

### 1.3 Secciones Principales

Usa `# Título` (H1) para secciones principales. Cada H1 genera un **salto de página** automático:

```markdown
# 1. Resumen Ejecutivo

Contenido de la sección...

# 2. Metodología

Contenido de la siguiente sección...
```

### 1.4 Subsecciones

Usa `## Título` (H2) y `### Título` (H3) para subsecciones:

```markdown
## 2.1 Datos Analizados

Descripción de los datos...

### Criterios de Selección

Lista de criterios...
```

---

## 2. Elementos de Formato Soportados

### 2.1 Texto con Formato

| Markdown | Resultado |
|----------|-----------|
| `**texto**` | **Negrita** |
| `*texto*` | *Cursiva* |
| Texto normal | Texto justificado |

### 2.2 Listas

**Viñetas:**
```markdown
- Primer elemento
- Segundo elemento
- Tercer elemento
```

**Numeradas:**
```markdown
1. Primer paso
2. Segundo paso
3. Tercer paso
```

### 2.3 Tablas

Formato estándar de Markdown:

```markdown
| Columna 1 | Columna 2 | Columna 3 |
|-----------|-----------|-----------|
| Dato 1    | Dato 2    | Dato 3    |
| Dato 4    | Dato 5    | Dato 6    |
```

El script genera tablas con:
- Encabezado azul oscuro con texto blanco
- Filas alternadas (gris claro / blanco)
- Bordes azules

### 2.4 Blockquotes (Citas Destacadas)

Ideal para hallazgos importantes o notas:

```markdown
> **Hallazgo importante:** Los estudiantes exitosos navegan de forma
> cualitativamente diferente en el LMS.
```

Se renderiza con:
- Borde izquierdo naranja
- Fondo crema claro
- Indentación

### 2.5 Bloques de Código / Diagramas ASCII

Para diagramas o código:

~~~markdown
```
┌──────────────────┐      ┌──────────────────┐
│   PLANIFICACIÓN  │ ──▶  │    EJECUCIÓN     │
│                  │      │                  │
│  • Establecer    │      │  • Mantener      │
│    metas         │      │    atención      │
└──────────────────┘      └──────────────────┘
```
~~~

Se renderiza en:
- Fuente monoespaciada (Consolas)
- Fondo gris claro
- Caja con bordes

**Opción:** Usar `--ascii-as-image` para convertir bloques de código a imágenes PNG (mejor para diagramas complejos).

### 2.6 Imágenes

```markdown
![Descripción de la imagen](visualizations/nombre_imagen.png)
```

- La ruta es **relativa** al directorio del reporte (`data/report/`)
- La descripción se convierte en caption centrado bajo la imagen
- Ancho automático: 15cm (12cm para heatmaps individuales)

### 2.7 Líneas Horizontales

```markdown
---
```

Se renderiza como línea decorativa gris centrada.

---

## 3. Preparación de Imágenes

### 3.1 Ubicación

Todas las imágenes deben estar en `data/report/visualizations/`:

```
data/report/
├── INFORME_*.md
├── INFORME_*.docx
└── visualizations/
    ├── correlation_heatmap.png
    ├── pass_rate_bars.png
    ├── grade_boxplot.png
    └── ...
```

### 3.2 Formato Recomendado

- **Formato:** PNG (preferido) o JPG
- **Resolución:** Mínimo 150 DPI para impresión
- **Tamaño:** El script escala a ~15cm de ancho
- **Fondo:** Blanco o transparente

### 3.3 Referencia en Markdown

```markdown
![Distribución de calificaciones por curso](visualizations/grade_boxplot.png)
```

---

## 4. Ejecución del Script

### 4.1 Comando Básico

```bash
python scripts/convert_md_to_docx.py
```

Usa los valores por defecto:
- Input: `data/report/INFORME_ALERTA_TEMPRANA_v2.md`
- Output: `data/report/INFORME_ALERTA_TEMPRANA_v2.docx`

### 4.2 Especificar Archivos

```bash
python scripts/convert_md_to_docx.py \
    --input data/report/MI_INFORME.md \
    --output data/report/MI_INFORME.docx
```

### 4.3 Diagramas ASCII como Imágenes

```bash
python scripts/convert_md_to_docx.py --ascii-as-image
```

Requiere `Pillow` instalado:
```bash
pip install Pillow
```

### 4.4 Salida Esperada

```
Convirtiendo: data/report/INFORME_ALERTA_TEMPRANA_v2.md
Salida: data/report/INFORME_ALERTA_TEMPRANA_v2.docx
ASCII como imagen: False

Título: Sistema de Alerta Temprana v2
Subtítulo: Predicción de Fracaso Académico mediante Patrones de Navegación en LMS

Parseando Markdown...
  Elementos parseados: 156
  Elementos después de filtrar TOC: 148
    - blockquote: 5
    - bullet_list: 12
    - code_block: 3
    - heading: 28
    - horizontal_rule: 4
    - image: 8
    - numbered_list: 6
    - paragraph: 72
    - table: 10

Creando documento...
  - Agregando portada...
  - Generando tabla de contenidos...
  - Configurando headers/footers...
  - Renderizando contenido...

Guardando documento...

¡Documento generado exitosamente!
Archivo: data/report/INFORME_ALERTA_TEMPRANA_v2.docx
Tamaño: 1.45 MB
```

---

## 5. Personalización del Estilo

### 5.1 Colores Corporativos

Los colores están definidos al inicio del script:

```python
# Colores corporativos (Universidad Autónoma de Chile)
PRIMARY_COLOR = RGBColor(0x1E, 0x3A, 0x5F)      # Azul oscuro - títulos H1
SECONDARY_COLOR = RGBColor(0x2E, 0x86, 0xAB)    # Azul medio - títulos H2, bordes tabla
ACCENT_COLOR = RGBColor(0xF1, 0x8F, 0x01)       # Naranja - blockquotes
TEXT_COLOR = RGBColor(0x30, 0x30, 0x30)         # Gris oscuro - texto normal
```

Para cambiar colores, edita estas constantes.

### 5.2 Fuentes

```python
FONT_NAME = 'Arial'           # Texto principal
FONT_NAME_MONO = 'Consolas'   # Bloques de código
```

### 5.3 Logo

El logo debe estar en la raíz del proyecto:

```python
LOGO_FILE = BASE_DIR / "LOGO-UA-color.jpg"
```

Si no existe, la portada se genera sin logo.

### 5.4 Márgenes y Tamaño de Página

```python
# A4 con márgenes estándar
section.page_width = Cm(21)
section.page_height = Cm(29.7)
section.left_margin = Cm(2.5)
section.right_margin = Cm(2.5)
section.top_margin = Cm(2.5)
section.bottom_margin = Cm(2)
```

---

## 6. Estructura del Documento Generado

El DOCX generado incluye:

1. **Portada** (página 1)
   - Logo centrado
   - Línea decorativa
   - Título en azul oscuro (26pt)
   - Subtítulo en azul medio (14pt)
   - Fecha y versión

2. **Tabla de Contenidos** (página 2)
   - Generada automáticamente desde H1 y H2
   - H1 en negrita azul
   - H2 con indentación

3. **Contenido** (página 3+)
   - Cada H1 inicia en página nueva
   - Encabezado: título del documento (derecha)
   - Pie: "Universidad Autónoma de Chile | Página X"

---

## 7. Flujo de Trabajo con Claude Code

### 7.1 Crear Nuevo Informe

1. **Crear el Markdown:**
   ```
   Pídele a Claude: "Crea un informe sobre [tema] en formato Markdown
   compatible con nuestro convertidor DOCX"
   ```

2. **Generar visualizaciones:**
   ```
   Pídele a Claude: "Genera los gráficos necesarios y guárdalos en
   data/report/visualizations/"
   ```

3. **Convertir a DOCX:**
   ```bash
   python scripts/convert_md_to_docx.py --input data/report/NUEVO_INFORME.md --output data/report/NUEVO_INFORME.docx
   ```

### 7.2 Modificar Informe Existente

1. Edita el archivo `.md` directamente
2. Re-ejecuta el script de conversión
3. El DOCX se sobrescribe con los cambios

### 7.3 Ejemplo de Prompt para Claude

```
Necesito crear un informe técnico sobre [tema]. El informe debe:

1. Seguir el formato Markdown compatible con convert_md_to_docx.py
2. Incluir:
   - Título y subtítulo descriptivos
   - Fecha y versión
   - Resumen ejecutivo
   - Metodología
   - Resultados con tablas y gráficos
   - Conclusiones
3. Usar blockquotes para hallazgos importantes
4. Incluir tablas para datos comparativos
5. Referenciar imágenes desde visualizations/

Guarda el archivo en data/report/NOMBRE_INFORME.md
```

---

## 8. Solución de Problemas

### 8.1 Imagen No Encontrada

**Síntoma:** `[Imagen no encontrada: visualizations/nombre.png]`

**Solución:**
- Verificar que la imagen existe en `data/report/visualizations/`
- Verificar que el nombre en el Markdown coincide exactamente (case-sensitive)

### 8.2 Tabla Mal Formateada

**Síntoma:** Tabla no se renderiza correctamente

**Solución:** Asegurar formato correcto:
```markdown
| Col1 | Col2 |
|------|------|
| A    | B    |
```
- Debe tener línea separadora con `|------|`
- Cada fila debe empezar y terminar con `|`

### 8.3 Caracteres Especiales

**Síntoma:** Caracteres como `│`, `─`, `▶` no se ven bien

**Solución:** Usar `--ascii-as-image` para convertir bloques de código a imágenes PNG.

### 8.4 Documento Muy Grande

**Síntoma:** DOCX supera 5MB

**Causa:** Imágenes muy grandes

**Solución:**
- Comprimir imágenes antes de incluirlas
- Reducir resolución a 150 DPI
- Usar formato JPG en lugar de PNG para fotografías

---

## 9. Dependencias

### Instalación

```bash
pip install python-docx Pillow
```

| Paquete | Versión | Propósito |
|---------|---------|-----------|
| `python-docx` | >=0.8.11 | Crear documentos Word |
| `Pillow` | >=9.0 | Convertir ASCII a imagen (opcional) |

### Verificar Instalación

```python
from docx import Document
print("python-docx instalado correctamente")

try:
    from PIL import Image
    print("Pillow instalado correctamente")
except ImportError:
    print("Pillow no instalado (opcional)")
```

---

## 10. Archivos de Referencia

| Archivo | Descripción |
|---------|-------------|
| `scripts/convert_md_to_docx.py` | Script principal de conversión |
| `LOGO-UA-color.jpg` | Logo para portada |
| `data/report/*.md` | Informes fuente en Markdown |
| `data/report/*.docx` | Informes generados |
| `data/report/visualizations/` | Imágenes para los informes |

---

## 11. Ejemplos de Documentos Generados

| Documento | Descripción |
|-----------|-------------|
| `INFORME_ALERTA_TEMPRANA_v2.docx` | Informe técnico completo con gráficos |
| `INFORME_EJECUTIVO_ENGAGEMENT_ESTUDIANTIL.docx` | Resumen ejecutivo para directivos |

---

## Resumen de Sintaxis Markdown Soportada

```markdown
# H1 - Sección principal (salto de página)
## H2 - Subsección
### H3 - Sub-subsección

**negrita** y *cursiva*

- Viñeta
- Otra viñeta

1. Numerado
2. Otro numerado

| Col1 | Col2 |
|------|------|
| A    | B    |

> Blockquote destacado

![Caption](visualizations/imagen.png)

---

```código o diagrama ASCII```
```

---

**Última actualización:** Enero 2026
