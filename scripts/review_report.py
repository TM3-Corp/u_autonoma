#!/usr/bin/env python3
"""
Pipeline de Revision de Informes para Autoridades.

Lee el informe seccion por seccion y verifica cumplimiento de la pauta.
Usa Claude API para identificar problemas y sugerir correcciones.
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import anthropic

# Configuracion
REPORT_PATH = Path("data/report/INFORME_ALERTA_TEMPRANA_v3.md")
PAUTA_PATH = Path("docs/PAUTA_REVISION_INFORME.md")
OUTPUT_PATH = Path("data/report/analysis/revision_results.json")

client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


def load_pauta():
    """Cargar la pauta de revision."""
    with open(PAUTA_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def split_into_sections(content):
    """Dividir el documento en secciones por encabezados H2/H3."""
    # Split by ## or ### headers
    sections = []
    current_section = {"title": "Introduccion", "content": "", "line_start": 1}

    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if line.startswith('## ') or line.startswith('### '):
            if current_section["content"].strip():
                sections.append(current_section)
            current_section = {
                "title": line.lstrip('#').strip(),
                "content": line + '\n',
                "line_start": i
            }
        else:
            current_section["content"] += line + '\n'

    if current_section["content"].strip():
        sections.append(current_section)

    return sections


def review_section(section, pauta, section_num, total_sections):
    """Revisar una seccion contra la pauta usando Claude API."""

    prompt = f"""Eres un revisor de documentos tecnicos para autoridades universitarias.

PAUTA DE REVISION:
{pauta}

SECCION A REVISAR ({section_num}/{total_sections}):
Titulo: {section['title']}
Linea inicial: {section['line_start']}

Contenido:
{section['content'][:3000]}

INSTRUCCIONES:
1. Revisa el contenido contra CADA criterio de la pauta
2. Identifica problemas especificos con numeros de linea aproximados
3. Sugiere correcciones concretas

Responde en JSON con este formato exacto:
{{
    "section_title": "{section['title']}",
    "passes_review": true/false,
    "issues": [
        {{
            "criterion": "nombre del criterio violado",
            "problem": "descripcion del problema",
            "original_text": "texto problematico exacto",
            "suggested_fix": "texto corregido sugerido",
            "severity": "alta/media/baja"
        }}
    ],
    "summary": "resumen de 1 linea del estado de la seccion"
}}

Si no hay problemas, devuelve issues como lista vacia y passes_review como true.
SOLO devuelve el JSON, sin texto adicional."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        result_text = response.content[0].text.strip()

        # Extract JSON from response
        if result_text.startswith('```'):
            result_text = re.sub(r'^```json?\n?', '', result_text)
            result_text = re.sub(r'\n?```$', '', result_text)

        return json.loads(result_text)

    except json.JSONDecodeError as e:
        return {
            "section_title": section['title'],
            "passes_review": None,
            "issues": [],
            "summary": f"Error parseando respuesta: {str(e)}",
            "raw_response": result_text if 'result_text' in dir() else "No response"
        }
    except Exception as e:
        return {
            "section_title": section['title'],
            "passes_review": None,
            "issues": [],
            "summary": f"Error en API: {str(e)}"
        }


def main():
    print("=" * 70)
    print("PIPELINE DE REVISION DE INFORME")
    print("=" * 70)

    # Cargar documentos
    print("\nCargando documentos...")
    pauta = load_pauta()

    with open(REPORT_PATH, 'r', encoding='utf-8') as f:
        report_content = f.read()

    # Dividir en secciones
    sections = split_into_sections(report_content)
    print(f"  Secciones encontradas: {len(sections)}")

    # Revisar cada seccion
    results = {
        "report_file": str(REPORT_PATH),
        "pauta_file": str(PAUTA_PATH),
        "total_sections": len(sections),
        "sections_with_issues": 0,
        "total_issues": 0,
        "high_severity_issues": 0,
        "section_reviews": []
    }

    print("\nRevisando secciones...")
    print("-" * 70)

    for i, section in enumerate(sections, 1):
        print(f"  [{i}/{len(sections)}] {section['title'][:50]}...", end=" ", flush=True)

        review = review_section(section, pauta, i, len(sections))
        results["section_reviews"].append(review)

        if review.get("issues"):
            results["sections_with_issues"] += 1
            results["total_issues"] += len(review["issues"])
            high_sev = sum(1 for iss in review["issues"] if iss.get("severity") == "alta")
            results["high_severity_issues"] += high_sev

            status = f"PROBLEMAS: {len(review['issues'])} ({high_sev} alta)"
            print(status)
        else:
            print("OK")

    # Guardar resultados
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE REVISION")
    print("=" * 70)
    print(f"  Secciones revisadas:     {results['total_sections']}")
    print(f"  Secciones con problemas: {results['sections_with_issues']}")
    print(f"  Total de issues:         {results['total_issues']}")
    print(f"  Issues de alta severidad: {results['high_severity_issues']}")
    print(f"\nResultados guardados en: {OUTPUT_PATH}")

    # Mostrar issues de alta severidad
    if results["high_severity_issues"] > 0:
        print("\n" + "-" * 70)
        print("ISSUES DE ALTA SEVERIDAD:")
        print("-" * 70)
        for review in results["section_reviews"]:
            for issue in review.get("issues", []):
                if issue.get("severity") == "alta":
                    print(f"\n[{review['section_title']}]")
                    print(f"  Criterio: {issue.get('criterion')}")
                    print(f"  Problema: {issue.get('problem')}")
                    print(f"  Original: \"{issue.get('original_text', '')[:100]}...\"")
                    print(f"  Sugerido: \"{issue.get('suggested_fix', '')[:100]}...\"")


if __name__ == "__main__":
    main()
