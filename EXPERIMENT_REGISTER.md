# Registro de Experimentos y Matriz de Cobertura SOTA
**Pipeline de predicción de reprobación (PUC + U. Autónoma) · 2026-07-02**
Construido desde: 2.640 corridas del benchmark, todos los scripts/notebooks del repo, y checklist de literatura (C&E, JLA, LAK/EDM, NeurIPS 2019-2026). Estados: ✅ HECHO · 🔬 PROBADO-DESCARTADO (con evidencia) · 🔴 GAP (vale la pena) · ⏭️ SKIP JUSTIFICADO.

## Veredicto ejecutivo

**Modelado y evaluación: en o sobre SOTA.** GBDT + class weights + calibración + LOCO CV es el consenso 2024-26 y lo cumplimos; nuestro feature engineering excede la taxonomía canónica (Marras EDM'21). **La capa upstream (pre-features) tiene 3 gaps de higiene** que la literatura considera must-have y nunca auditamos: deduplicación de clicks, timezone local, y sensibilidad del timeout de sesión. **La frase "los mejores modelos posibles" requiere cerrar Tier 1** (abajo); hoy la frase defendible es: *"cubrimos exhaustivamente el espacio de técnicas con ganancia esperada positiva a esta escala; lo no probado tiene justificación documentada"*.

## 1. Procesamiento de datos (upstream) — NUNCA AUDITADO HASTA HOY

| Ítem | Estado | Detalle / evidencia |
|---|---|---|
| Sessionización 30 min | ✅ consistente / 🔴 sin sensibilidad | `SESSION_GAP_MINUTES=30` idéntico en 3 scripts. Lit: 30 min es herencia de web analytics, no validado para LMS — exige justificación empírica + sensitivity check (barato) |
| **Dedup de clicks rápidos** | 🔴 **GAP #1** | CERO deduplicación (verificado: ningún drop_duplicates sobre eventos). Reloads/redirects/doble-clicks inflan todos los conteos, densidades y bigramas. Must-have de higiene en la lit |
| **Timezone** | 🔴 **GAP #2** | hour/weekend_pct/day_entropy calculados en UTC; Chile = UTC-3/4 → 23:00 local cuenta como madrugada. Cero `America/Santiago` en el repo. Corrige toda la familia temporal |
| Inicio de curso | 🔴 inconsistente | 3 definiciones conviven (pctl-5 de eventos / pctl por-alumno / sweep 5-20%); clicks de setup del instructor pueden correr la semana 1 (staff no se filtra ANTES del quantile) |
| Fin de curso | 🔴 magia | Horizonte 16 semanas hardcodeado; incorrecto para cursos bimestrales PUC (sesga weeks_since_last, early_late_ratio, DCT) |
| Filtro staff/bots/test | 🟡 parcial | El inner-join a alumnos con nota filtra staff de los FEATURES, pero no del cálculo de course_start. Sin filtro de bots/test-students. Piso de actividad = 2 views (bajo), no documentado |
| Procedencia raíz | 🔴 doc | `filtered_page_views.parquet` (semilla de todo PUC) se construyó fuera de ambos repos; qué filtró es indocumentado |
| Estandarización sin fuga | ✅ | Scaling y selección de features por-fold, solo con train |
| Homónimos peligrosos | 🔴 doc | `total_time_min` significa cosas distintas en 2 scripts; resource_id real vs hasheado entre paths PUC |

## 2. Feature engineering — FUERTE (≥ taxonomía canónica)

Taxonomía Marras/Vignoud/Käser (EDM 2021): regularity, effort, consistency, proactivity, control, assessment.

| Familia lit | Nuestro estado |
|---|---|
| Effort (volumen) | ✅ views/tiempo/por-categoría (contaminado por GAP #1) |
| Regularity | ✅ session_regularity, day_entropy, DCT espectral (≥ FWH de Boroujeni EC-TEL'16) · 🟡 falta PDH/PWD entropy-de-histograma nombrado (equivalentes parciales existen) |
| Proactivity / lead-time | ✅ pre-deadline access, proactivity indices · 🟡 falta composite procrastinación×regularidad (Li/Baker/Warschauer: "amplifica la señal") |
| Consistency / temporal | ✅ weekly trend, momentum, decay, inactivity episodes (corregir tras GAP #2) |
| Assessment | ✅ + toggle with/without (dimensión del benchmark) |
| Course-relative norm | ✅✅ znorm por curso — must-have multi-curso, lo tenemos pervasivo |
| Agregación semanal | ✅ (efecto más grande según Arizmendi'22: p<.001 — cumplido) |
| Extras no exigidos por lit | ✅ n-gramas navegación, features de grafo, PCA/agglomeration, timing histograms |

## 3. Selección de features

✅ Composite 6-métodos per-fold sin fuga (MI+ANOVA+RF+LASSO+LGBM+Boruta) · ✅ RFECV+ElasticNet+estabilidad bootstrap/LOCO (track research) · 🔬 reducción agresiva DESCARTADA con evidencia propia (sweep N∈{2..34}: full>few; consistente con Grinsztajn NeurIPS'22) · 🟡 SHAP-selection y permutation importance: importados, nunca invocados · ⏭️ mRMR/genéticos: sin evidencia de ganancia a este N.

## 4. Modelos

✅ 15 variantes: XGB/LGBM/RF/GB/LogReg/SVM(+balanced) + MLP/MLP_deep (única NN) + Voting + Stacking · ✅ Optuna TPE 50 trials (5 familias, top-3 por config, F2 objective) · 🔴 **CatBoost no probado** (must-have candidato según lit; HistGB gratis de paso) · ⏭️ TabNet/FT-Transformer/secuenciales: skip justificado — cero papers creíbles de NN>GBDT-tuned a N≈500-1000 (verificado en la búsqueda; Grinsztajn; nuestro propio N) · 🟡 SVM/MLP/ensembles sin HPO (no son ganadores; costo>beneficio).

## 5. Desbalance

✅ class weights + scale_pos_weight (fijo y tuned 1-20) · 🔬 SMOTE/Borderline PROBADO Y RETIRADO (ablation propia: AUC 0.745 vs 0.797, ECE 2× peor — consenso van den Goorbergh'22/Carriero'25 confirmado en nuestros datos) · ✅ threshold-moving 12 criterios · ✅ calibración Platt (ECE 0.048→0.015) · ⏭️ focal loss / cost-matrix-in-loss: expectativa nula documentada (review adversarial previa) · ⏭️ ADASYN/undersampling: dominados por el hallazgo anti-resampling propio.

## 6. Targets

✅ binary_4.0 (principal) · 🔬 binary_4.5 DESCARTADO con comparación justa + CIs (drop AUC intrínseco; marginales se comportan como aprobados) · ✅ 4class, 3class_marginal (benchmark) · ✅ ordinal Frank-Hall y regresión-then-threshold (demos validadas) · 🟡 binary_5.0/3class/oviedo: definidos, nunca corridos (baja prioridad — 4.5 ya respondió la pregunta del corte).

## 7. Validación y HPO honesto

✅ StratifiedGroupKFold por curso (LOCO) · ✅ cortes semanales 2/4/6/8/full · ✅ bootstrap CIs (B=2000, pareados) · 🔴 **nested CV NO hecho** (Optuna afina en 3-fold interno y se evalúa en el mismo 5-fold externo → optimismo leve en los números tuned; must-have de la lit para reportar "honesto") · 🟡 semilla única (sin repeated-CV) · 🟡 sin hold-out de cohorte/año (no hay segunda cohorte disponible aún — se convierte en test natural cuando llegue UDLA).

## 8. Explicabilidad

✅ **SHAP YA EXISTE**: `generate_shap_explanations.py` (TreeExplainer, summary/force plots, texto por estudiante) + `pooled_binary_classifier.py` · 🔴 no integrado al benchmark ni al material de venta — activarlo es victoria barata y habilita el claim "explicable con Shapley values" · ⏭️ LIME/counterfactuals: SHAP es el estándar LA; suficiente.

## 9. Otros (lit stage-8)

✅ calibración (hecha) · 🟡 fairness audit: sin atributos demográficos en los datos actuales → no factible; documentado (re-evaluar con datos UDLA) · 🟡 snapshot-ensembling temporal: ganancia de estabilidad, no de AUC; opcional · ⏭️ semi-supervised con retirados: los grade-NaN son outcome-ambiguos (el retiro ES señal de dropout, no unlabeled limpio).

---

# Shortlist priorizada (todo CPU local, cero quota de LLM)

**Tier 1 — mueve números o es requisito del claim:**
1. **Dedup de clicks** (de-bounce same-URL <10-30s) + rebuild features + A/B vs actual. El multiplicador no-auditado más grande. *Nota: si el AUC actual ya es 0.83-0.90 CON ruido, limpiar solo puede revelar señal — upside asimétrico.*
2. **Fix timezone** → `America/Santiago` para hour/weekend/day_entropy + A/B familia temporal.
3. **Nested CV** para las configs headline → los números que se publican quedan sin optimismo.
4. **CatBoost + HistGB** al zoo con espacio Optuna (cierra el gap de modelos citado por la lit).
5. **Activar SHAP** en producción (per-student profiles) — 0 ganancia AUC, alto valor de deployment/venta.

**Tier 2 — defensibilidad barata:** sensitivity del timeout de sesión (15/30/60/120) · PDH/PWD entropy + composite procrastinación×regularidad · unificar definición de course-start (y excluir staff del quantile) · documentar piso de actividad · renombrar homónimos (`total_time_min`).

**Tier 3 — skips documentados (no tocar):** deep tabular/secuencial · focal loss · resampling sintético · reducción agresiva de features · semi-supervised con retirados · fairness (hasta tener demografía).

---

## La frase defendible

**Hoy:** "Cubrimos 15+ modelos (incl. redes neuronales y stacking) con HPO bayesiano, 6 métodos de selección de features, todos los tratamientos de desbalance recomendados —dos descartados con ablations propios—, validado en cursos nunca vistos con intervalos de confianza; lo no probado tiene justificación documentada en literatura o en nuestra propia evidencia."

**Tras Tier 1:** "…y el pipeline completo, desde el click crudo hasta la predicción, está alineado con las prácticas de Computers & Education y el Journal of Learning Analytics, con números reportados bajo nested cross-validation y explicaciones Shapley por estudiante." — Esa es la versión "mejores modelos posibles con estos datos".

*Fuentes clave del checklist: Grinsztajn et al. NeurIPS'22; Marras/Vignoud/Käser EDM'21; Boroujeni et al. EC-TEL'16; Arizmendi et al. 2022 (Behav. Res. Methods); LAK'24 study-regularity (OULAD). Detalle de auditorías: transcripts de agentes 2026-07-02.*

---

# Adenda 2026-07-03 — Evidencia del notebook de tesis (`Plataforma-Educativa/DATA/API test (1).ipynb`, celdas 53-287)

Revisión de 3 agentes sobre el proceso de análisis de la tesis (PUC). Cambia el estado de varios ítems de la matriz:

## Lo que la tesis hacía BIEN y producción PERDIÓ (recetas listas para portar)

| Ítem | Celda | Receta |
|---|---|---|
| **Dedup nivel 1: filas exactas** | 109-110 | `drop_duplicates()` — eliminó **2.906.063 filas (~18% de 16M)** |
| **Dedup nivel 2: par HTML+API** | 134 (patrón) | un view genera 2 filas: `/courses/X/files/Y` + `/api/v1/courses/X/files/Y` — colapsar por (user, resource, timestamp) |
| **Dedup nivel 3: repeticiones consecutivas misma URL** | 114 | dentro de (User, Session); OJO: el comentario dice keep-first pero el código hace keep-last; calcular duración DESPUÉS del dedup |
| **Timezone América/Santiago** | 112 | `pd.to_datetime(col).dt.tz_convert('America/Santiago')` — DST verificado en outputs (-03:00 y -04:00) |
| **Duración por dwell** | 113, 115 | View Duration = tiempo-al-próximo-click; último view → `interaction_seconds` de Canvas capped 1800s; Session Duration = SUM (no max−min → elimina sesiones de 1 click con duración 0) |
| ~~`interaction_seconds`~~ | 111, 271 | **DESCARTADO por análisis de Paul**: outliers y valores poco fiables (consistente con el heartbeat impreciso de Canvas). NO usar como canal de tiempo-activo; en el modelo de duración, reemplazar el fallback del último view por mediana-de-sesión o exclusión. Pendiente: cuantificar la no-fiabilidad (15 min) durante el rebuild para documentar la exclusión |
| **Anclaje por calendario académico** | 67 | `define_semester` (ago-nov→S2, dic-jul→S1) — más defensible que percentil-5 con clicks de staff |
| **Semántica de participación por categoría** | 156-157, 164 | quiz participation = POST a `quizzes/quizzes` (GET a submission-API NO cuenta); assignment = `submissions/create` |

## Lo que la tesis dejó PENDIENTE (ahora follow-ups de alto valor)

1. **Validación de sesiones contra ground truth**: `get_user_logins()` quedó `# PENDING` (celda 267) — el 30-min nació como afirmación (celda 113) y producción lo heredó verbatim. Ejecutar la comparación auth-sessions vs page-view-sessions = justificación empírica definitiva del timeout.
2. **Match-rate URL→recurso**: celdas 139-141 andamiadas, NUNCA ejecutadas (0 outputs). `get_course_files()` da el ground truth de Canvas. Correrlo cuantifica la fidelidad de la extracción regex que producción asume.
3. Histograma de gaps inter-click (`Time Diff`, celda 113) → análisis de sensibilidad del timeout que ni tesis ni producción hicieron.

## Hallazgo empírico citable (celda 257, correlaciones vs Nota Final)

**Conteo de sesiones r=0.41-0.44 > tiempo 0.31 > views 0.29 > Engagement Score composite de 14 partes 0.24.** El composite rinde peor que su ingrediente crudo — argumento contra sobre-ingeniería de scores.

## Features de la tesis ausentes en producción (candidatas Tier 2)

Familia inactividad/lealtad (`Sessions Between Course`, ratios de inactividad cross-curso) · **intensity** (desviación semanal vs promedio propio del alumno) · workload-slope (descomposición pendientes +/− con sum/count/ratio) · taxonomía de peaks (local-max + umbrales 25/50/100%) · 3 denominadores anidados (curso / todos-los-cursos / todo-el-LMS: `Course_to_Weekly` = concentración de engagement) · `Week_Interactions_Ratio` por categoría.

## Advertencias (higiene del notebook — NO contamina producción)

ML de la tesis (celdas 251-266): leakage temporal (features de semestre completo), ranks pre-split, `User` ID como feature espurio (r=0.33), sin outputs guardados → NO citar métricas de ahí (producción reconstruyó el ML correctamente con folds/cortes/LOCO). Estado guardado no-reproducible (celda 224 crasheó; muestreo con 3 definiciones conviviendo, viva la de 784 sin rationale). Ventana tesis = 17 semanas (11-27); el "16" de producción no traza a nada validado.

## Shortlist Tier 1 REVISADA (con recetas de la tesis)

1. **Dedup 3-niveles** (recetas exactas arriba) + rebuild + A/B — upside asimétrico confirmado: el AUC actual se midió con ~18% de filas duplicadas.
2. **Timezone Santiago** (receta celda 112) + A/B familia temporal.
3. **Nested CV** para números headline.
4. **CatBoost + HistGB** al zoo.
5. **Activar SHAP** (ya existe en repo).

Tier 2 nuevo/actualizado: validación auth-sessions (PENDING de tesis) · match-rate URL→recurso (andamiado en tesis) · sensitivity timeout con histograma de gaps · anclaje calendario + horizonte validado · features tesis (inactividad, intensity, slopes, denominadores anidados).

---

# Adenda 2026-07-03 (2) — Definición del universo y pareo de notas ("qué datos, contra qué notas")

## PUC — pareo SÓLIDO ✅

**Cadena de procedencia** (trace completo): acta oficial PUC (Nota Final, escala 1-7, elegida sobre Controles/Tareas/Décimas) → match identidad por **email @uc.cl → Canvas user_id** (tasas medidas 90.5–100% por curso) → `students_grades_processed_with_sigla.csv` (mapeo Sigla→course_id embebido, 1:1 verificado, multi-secciones OK; fuente humana = `.numbers` en Plataforma-Educativa/DATA) → `puc_fix_data.py` (drop 41 retiros NaN) → `puc_grades_clean.parquet`. Semestre 2023-1 limpio, sin mezcla.
**Universo definitivo**: master histórico **89.337.099 views** intacto en `C:\TM3\Canvas_Files\all_page_views_dask.parquet`; operativo = 17.2M (2023-1) / **784 estudiantes** / 841 pares con clickstream. Modelado: **7 cursos / 560 pares / 41 fails** = cursos con cobertura 100% Y ≥2 reprobados (criterio de Paul, ahora cuantificado). Opcional producción: +3 cursos cobertura-100% con 0 fails (53493/54947/56867, +167 pares). Excluir: 114 pares parciales (81% matrícula cruzada, sesgados) y 766 pares sin clickstream (irrecuperables sin token).
Riesgos menores: dropouts del match por email (0–10%/curso, medidos); homónimos `grades.csv`≡`_with_sigla.csv` (gemelos de formato).

## UA — pareo de identidad trivial, pero 🔴 INTEGRIDAD DEL TARGET COMPROMETIDA

**Cadena**: Canvas Enrollments API → `grades.final_score` (% de puntos, snapshot 2025-12-30) → `failed = final_score < 57` (≈4.0/7.0) → merge por mismo Canvas user_id (con normalización de shard global `1551·10¹³+id` → id local, `normalize_user_id`). Identidad: riesgo bajo. PERO:
- **51 de 149 "reprobados" (34%) son estudiantes ACTIVOS en el LMS (≥20 views, medianas 98-124) con final_score = 0.0** — consistente con notas en el "Libro de Calificaciones" LTI (inaccesible por API, advertido en docs del repo), NO con reprobación real. Test decisivo ejecutado 2026-07-03.
- Cursos 🔴 (>15% matrícula activa-con-0): **84941 (20/38), 84936 (10/42), 79875 (6/32)**. El resto 🟡 con 2-4 casos c/u.
- Scores >100 en 88381/89390 (esquemas de puntos) → "57%≡4.0" no uniforme por curso. Snapshot inmediatamente post-término S2-2025 (última actividad hasta 12-28) — notas posiblemente no todas posteadas.
- **Implicación**: los AUC UA (0.74–0.90) se midieron con ~1/3 de los positivos potencialmente mal etiquetados. Dirección del sesgo: activos-con-0 lucen como aprobados conductualmente → el modelo es CASTIGADO por clasificarlos bien → **el desempeño real probablemente está subestimado**. Limpiar puede subir los números (upside asimétrico de nuevo). Contrapartida: parte del recall actual viene de abandonos triviales (inactivo+0 = fail fácil).

**Opciones de remediación UA** (decisión pendiente):
A. (rápida) Excluir los 51 activos-con-0 como label-desconocido → re-etiquetar → re-run semanal → números honestos.
B. Excluir los 3 cursos 🔴 completos (universo 7 cursos / ~261 pares).
C. (oro) Pedir a U. Autónoma las notas de acta oficiales de las 373 matrículas → target real como PUC. Natural de pedir junto al piloto UDLA.

**Definición operativa para Tier 1**: PUC = 7 cursos/560 pares vs Nota Final oficial <4.0 (los A/B de dedup/timezone se corren aquí — pareo confiable). UA = NO re-correr hasta aplicar mínimo la opción A.

### Refinamientos del trace completo (2026-07-03, segunda pasada)

**PUC**: cadena completa = `DetalleCursosPaul.csv` (export registro, 1.714 filas, solo email — sin IDs LMS) → [**PASO FALTANTE**: join email→user_lms_id; código no ubicado en ningún repo — hipótesis: SQL contra BD plataforma TM3, sujeto a verificación] → `students_grades_processed.csv` → research.ipynb cell 6 (**drop 66 filas user_lms_id=-1, 3.9%, NO aleatorio**: ING1004-1 19/80=24% del roster; ICS2813-2 —curso del benchmark— 11; IIC2213-1 14; IIQ1003-1 9) → `_with_sigla.csv` → puc_fix_data.py (drop 41 NaN retiros) → parquet. **Fidelidad verificada: grade==NotaFinal en 100% de las 1.607 filas** (cero transformación). Mapeo curso: fuente autoritativa `processed_courses.csv` (BD TM3). El path "Manual Grades Upload" del notebook (celdas 42-52) NO es la fuente del target (exploración con componentes Controles/Tareas/Décimas). `grades.csv` verificado `equals()==True` con `_with_sigla` (gemelo de dtype). Riesgo residual del benchmark: survivorship de los 66 no-matcheados (ICS2813-2 ~9% del roster).
**UA**: riesgo de snapshot RETIRADO (2025-12-30 = post-cierre del año; cursos terminados). Riesgo confirmado = concepto del label (final_score con LTI-gradebook externo; 61/373 filas con score 0, de las cuales 51 ACTIVAS). Nuevo: **inconsistencia 57 vs 60** — label `<57` (train_time_limited_model.py:153) pero "passing users" `>=60` en calculate_features_with_cutoff.py:553 y analyze_course_time_ranges.py:184 → la banda 57-60 recibe trato contradictorio; unificar al remediar labels.

### Cierre UA (2026-07-03, verificación exhaustiva a pedido de Paul)

Paul recordaba haber descartado cursos con muchas notas 0 — **su memoria es correcta para la era anterior** (`prediction_models.py`, "6 cursos/258 estudiantes", con checks de pass_rate y descartes por datos insuficientes). Pero la extracción actual usó una lista NUEVA hardcodeada de 10 cursos (`MODEL_COURSES`, fetch_enrollment_grades.py:24) **sin re-aplicar ese vetting**. Verificado con certeza:
1. **Los 51 activos-con-0 SÍ entraron a los datasets semanales** (features guardadas: wk2 42/51, wk4 48/51, wk8 **51/51**). Los AUC UA citados (0.74-0.90) se midieron con esos labels.
2. `enrollment_state` no discrimina: los 61 ceros son todos 'active' — el test de actividad es el único discriminador.
3. **Remediación A** (excluir 51 activos-con-0): n 373→322, fails 149→98, prevalencia 40.0%→**30.4%**; tasas por curso pasan a rangos plausibles (6-42%)… excepto:
4. **86676 (Taller Pensamiento Analítico): gradebook PARCIAL confirmado** — scores no-cero en 4 bandas apretadas ([15-20]×13, [35-44]×11, [53-64]×3, [76-82]×9) con techo 82.2 = solo parte de la ponderación en Canvas. TODOS sus labels sospechosos, no solo los ceros. (Era "test course, good variance" en CLAUDE.md — la varianza era en parte artefacto.)
5. **Remediación recomendada A+**: excluir 51 activos-con-0 Y curso 86676 completo → **n=286, fails=73, prevalencia 25.5%** — el dataset UA honesto disponible hoy. La solución definitiva sigue siendo la opción C (actas oficiales UA; en UA solo hubo llave de Canvas, nunca notas externas — a diferencia de PUC).
⚠️ IMPLICACIÓN COMERCIAL: la prevalencia UA citada en materiales (39-40%) y los AUC cambiarán tras remediar — NO tocar materiales de venta hasta re-correr con A+.
