# Propuesta de actualización de cifras — `metricas-tecnicas-udla.html`
**Preparado por la sesión Tier-2 · 2026-07-03 · GATED: solo propuesta, NO se editó el HTML.**
La decisión de qué cifras publicar es de Paul. Este archivo compara la tabla ACTUAL del HTML con la tabla NUEVA honesta (confirmatoria, PUC), en tono descriptivo neutral.

Fuente de las cifras NUEVAS: `tier2_push/confirmatory_results.json` (PUC, CatBoost calibrado, validación cruzada anidada LOCO, datos limpios, 5-seed bagging, seed 42). Ver `TIER2_RESULTS.md` §1/§6.

---

## A. Tabla "lo mejor por semana" — ROC-AUC (PUC)

| Semana | HTML ACTUAL | **NUEVA (honesta)** | IC 95% (nueva) |
|--------|-------------|---------------------|----------------|
| 2  | 0.83 | **0.78** | [0.69, 0.85] |
| 4  | 0.87 | **0.81** | [0.73, 0.87] |
| 6  | 0.86 | **0.82** | [0.75, 0.89] |
| 8  | 0.86 | **0.84** | [0.77, 0.89] |
| Fin del curso | 0.90 | **0.79** | [0.70, 0.88] |

*La curva actual (0.83→0.90) es no-anidada y optimista. La nueva es anidada (cursos completos retenidos, LOCO), sobre datos limpios y modelo calibrado — la cifra defendible bajo escrutinio técnico.*

## B. Recall a capacidad de revisión (PUC, nuevo — reemplaza los pares "recall/precisión" del punto de operación)

Tono descriptivo: "marcando al X% de mayor riesgo se detecta Y% de los reprobados".

| Semana | Recall @20% de capacidad | Recall @25% |
|--------|--------------------------|-------------|
| 2  | 61% | 68% |
| 4  | 66% | 68% |
| 6  | 68% | 76% |
| 8  | 66% | 71% |
| Fin | 66% | 73% |

*(La tabla actual cita "82%/86%/88% de recall" en un punto de operación de máximo-F1 no reportado junto a su precisión ni capacidad; el marco de recall-a-capacidad es autocontenido y honesto.)*

## C. Calibración (se mantiene — sigue siendo fuerte)

ECE 0.013–0.017, Brier 0.053–0.060 → "un 70% de riesgo significa ~70% observado". Esta afirmación del HTML se sostiene con las cifras nuevas.

## D. Fila "ROC-AUC en cursos nunca vistos: 0.83 – 0.87"

Reemplazar por el rango honesto anidado: **0.78 – 0.84** (semanas 2–8, LOCO, IC 95% que en las semanas 6–8 no toca el azar; en semanas tempranas el IC inferior baja a ~0.69–0.73).

---

## E. Cambios estructurales sugeridos (dirección permanente de Paul)

1. **Eliminar la sección "Cada afirmación / Se dijo / Verdad"** (el concepto contrastivo). Sustituir por una presentación descriptiva directa de la curva honesta y su procedencia.
2. **Nota de procedencia única** (una frase, para reemplazar los múltiples asteriscos correctivos):
   > *Cifras bajo validación cruzada anidada con cursos completos retenidos (LOCO), sobre datos limpios desde el clickstream (deduplicación de 3 niveles + zona horaria America/Santiago), con modelo CatBoost calibrado (Platt) y explicaciones Shapley por estudiante.*
3. **PUC-only**: no incluir cifras UA en esta página (las cifras UA honestas — rango 0.81 DROP-A a 0.87 KEEP con salvedad de etiqueta, 0.61–0.69 LOCO — dependen de labels con contaminación conocida y de la llegada de las actas oficiales; ver `TIER2_RESULTS.md` §5).

---

## F. Nota sobre el número "0.903 / 0.89" del encabezado

El HTML destaca "ROC-AUC ~0.89" y "semestre completo 0.903". Bajo el protocolo honesto (anidado, LOCO, calibrado) el fin-de-curso es **0.79** (raw-bagged) / 0.83 (calibrado). Sugerencia: retirar el "0.89–0.903" del encabezado y liderar con la **semana 8 = 0.84**, que es el corte honesto más fuerte y el más relevante para alerta temprana.

---

**STOP.** El HTML `~/projects/tm3-roi-diagnostico/metricas-tecnicas-udla.html` NO fue modificado. Paul decide qué cifras se publican.
