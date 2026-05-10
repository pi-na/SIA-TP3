# Diseño — Ej3 MVP-B: re-run sobre cross_v1 ganador + plots para defensa oral

| | |
|---|---|
| **Fecha** | 2026-05-10 |
| **Autor** | Nicolás (con asistencia de Claude Opus 4.7) |
| **Estado** | Pendiente de aprobación del usuario |
| **Materia** | ITBA SIA — TP3 (1Q 2026) |
| **Sesión origen** | Brainstorming 2026-05-10 post-presentación fallida del 2026-05-05 |
| **Próximo paso** | `superpowers:writing-plans` para producir el plan de implementación detallado |

---

## 1. Contexto

### 1.1 Punto de partida

- La presentación del TP3 del **2026-05-05 salió mal**. La cátedra dio feedback escrito (ver `1.2`).
- El Ejercicio 3 **ya tiene infraestructura armada** en `ejercicio3/`: configs, `final_eval.py`, 4 plots de presentación, README detallado. Test_acc actual del ganador (`pack_c/l2_aug.json` sobre base `[784,100,50,10]`): **0.9688 (96.88%)**. Falta 1.12pp para el target del 98%.
- En paralelo, en el branch `main` se hizo el **cross_v1** (sweep cruzado LR × Opt × Arch + estrella batch + tiebreaker, 2026-05-09/10) que definió el ganador real del Ej2: **`arch_shallow` [784,128,10] + Adam@1e-3 + batch=64 + ES patience=20**.
- Ese ganador del cross_v1 **no se usó como base del Ej3** (el Ej3 fue construido antes, con la ganadora de la Fase 1 vieja del worktree `tp3-mlp`). Hay un disconnect que conviene resolver antes del recu.

### 1.2 Feedback específico de la cátedra para Ej3

1. ✅ **Excelente**: plantean hipótesis (gap del Ej2) y plan secuencial.
2. ❌ **Falta justificación**: introducen regularización sin explicar por qué eligen regularización y no cambios de arquitectura o más HP tuning.
3. ❌ **Reflexiones sin respaldo**: las reflexiones sobre por qué falta 1.12pp para el 98% están bien conceptualmente, pero **no hay resultados que las respalden**.

Feedback general aplicable: "no hubo justificación desde la parte teórica ni desde resultados", "deben entender los términos y experimentos incluidos", "ante preguntas que conteste quien tiene menos asignado".

### 1.3 Lo que cambió en estos días en main

Tres insights del cross_v1 que el Ej3 debe internalizar:

- **El bottleneck del Ej2 es overfit, no capacidad.** Train_loss ≈ 0 con val_loss ≈ 0.17 → train-val gap = 0.35 en Adam@1e-3. Esa firma justifica la elección "regularización" en lugar de "más capacidad / más HP".
- **Cross_v1 ya barrió arch × LR × opt × batch (249 jobs).** No quedan HPs por descubrir. La palanca disponible que no se exploró es la regularización.
- **Adam tiene el train-val gap MÁS GRANDE de los 3 optimizadores.** El cross_v1 mostró 0.35 (Adam) vs 0.28 (Momentum) vs 0.22 (SGD). El optimizer que más se beneficia de regularización es exactamente el que elegimos. Eso es defendible.

---

## 2. Objetivo

Producir un Ej3 **MVP-B** que:

1. Esté construido sobre el ganador del cross_v1 (coherencia narrativa con Ej2).
2. Tenga **resultados que respalden** las reflexiones del README v1 sobre por qué falta 1.12pp.
3. Tenga **plots claros** y un **guion conciso** donde los plots hablan más que las palabras.
4. Permita al grupo defender oralmente cada decisión sin tener Claude al lado.

### 2.1 In scope

- Re-correr 6 configs (1 base + 5 Pack C) sobre el ganador del cross_v1.
- Final eval del nuevo ganador contra `digits_test.csv` (una sola vez).
- 7 plots para la presentación (3 nuevos + 4 regenerados).
- Re-escribir `ejercicio3/README.md` y producir `ejercicio3/guion_presentacion.md`.

### 2.2 Out of scope

- Augmentación geométrica (rotaciones, desplazamientos) — fuera del MVP, mencionada como future work.
- Análisis de calibración — opcional teórico, no requerido.
- Robustez a ruido sobre `digits_test.csv` — opcional práctico, no requerido.
- Re-correr el Ej2 cross_v1 — ya está hecho, sólo se citan sus datos.
- Modificar `mlp/` — todo se hace con la infraestructura actual.

---

## 3. Arquitectura experimental

### 3.1 Configs nuevos

```
ejercicio3/configs/
├── base_extra_data_v2.json       # NUEVO base: shallow + Adam@1e-3 + batch=64 + patience=20 + max_epochs=40
└── pack_c_v2/
    ├── l2.json                   # l2 = 1e-4 solo
    ├── l2_aug.json               # l2 = 1e-4 + aug gaussiano σ = 0.05 (presumed ganador)
    ├── dropout.json              # dropout = 0.2 solo (negative control: aporta poco)
    ├── wider_l2_aug.json         # arch [784,256,128,10] + l2 + aug (negative control: más capacidad)
    └── aug_s010.json             # l2 + aug σ = 0.10 (negative control: ruido alto)
```

**Total: 6 configs.** Cada Pack C se elige para respaldar una conclusión específica del README v1 (ver tabla en sección 5.1).

### 3.2 Hiperparámetros base v2 (heredados del cross_v1 ganador)

| Parámetro | Valor | Justificación |
|---|---|---|
| `architecture.layer_sizes` | `[784, 128, 10]` | arch_shallow ganador del cross_v1 + tiebreaker (vs arch_wider) |
| `architecture.activations` | `["relu", "softmax"]` | Estándar para multiclase |
| `architecture.initializer` | `"auto"` | He para ReLU |
| `training.optimizer.name` | `"adam"` | Ganador cross_v1 stage 2 (val_acc 0.957) |
| `training.optimizer.lr` | `0.001` | Ganador cross_v1 stage 2 |
| `training.batch_size` | `64` | Ganador cross_v1 stage 2b (estrella batch) |
| `training.epochs` | `40` | max_epochs auditado para Adam@1e-3 |
| `training.early_stopping_patience` | `20` | Auditado en cross_v1 |
| `preprocessing.normalization` | `"zscore"` | Igual que cross_v1 |
| `split.k_folds` | `5` | Estratificado |
| `split.random_seed` | `42` | Seed canónico |
| `dataset.extra_csv_paths` | `["data and documentation/more_digits.csv"]` | Datos extra del enunciado |

### 3.3 Pack C — variantes

| Config | Cambio vs base | Para respaldar |
|---|---|---|
| `l2.json` | `regularization.l2 = 1e-4` | "L2 aporta marginal positivo" |
| `l2_aug.json` | `l2 = 1e-4` + `augmentation = {type:"gaussian_noise", sigma:0.05}` | "L2 + aug es el sweet spot" |
| `dropout.json` | `regularization.dropout = 0.2` | "Dropout no transfiere a test (reduce varianza fold-specific, no shift)" |
| `wider_l2_aug.json` | `layer_sizes = [784,256,128,10]` + l2 + aug | **"Más capacidad EMPEORA → bottleneck no es capacidad"** |
| `aug_s010.json` | l2 + aug `sigma = 0.10` | "Más ruido no compensa shift geométrico" |

### 3.4 Runner

`ejercicio3/scripts/run_pack_c_v2.py` — basado en patrón de `ejercicio2_experimentacion/scripts/cross_v1/runner.py`:

- `ProcessPoolExecutor` con 8 workers.
- `OMP_NUM_THREADS=1` en env para evitar contención.
- Cada cell persiste su `run_summary.csv` + `epoch_history.csv` + `confusion_matrix.csv` ANTES de devolver al master.
- `errors.log` para crashes sin matar el pipeline.
- `STATUS.txt` actualizado en cada cell completada (timestamp + progreso).
- **Modelo de paralelismo (igual que cross_v1):** una "cell" = un **config completo**. Paralelismo OUTER sobre cells; K=5 folds **secuenciales adentro de cada worker** (con OMP=1 para evitar contención). Pool de 8 workers; con 6 configs todas las cells corren concurrentes.
- **Compute estimado:** 6 cells × ~50 min/cell (K=5 secuencial × ~10 min/fold con `more_digits.csv`) en pool de 8 workers concurrentes → **~50 min wall-clock total** (limitado por la cell más lenta, no por la suma).

### 3.5 Final eval

Re-usa `ejercicio3/final_eval.py` (no se modifica), apuntando al ganador de v2:

1. Entrena con `digits.csv ∪ more_digits.csv` completo (sin K-fold).
2. Evalúa **una sola vez** contra `digits_test.csv`.
3. Reporta: accuracy global, matriz de confusión 10×10, P/R/F1 por clase.

### 3.6 Layout de output

```
ejercicio3/output/v2/
├── STATUS.txt
├── errors.log
├── raw_summary.csv                # consolidación de los 6 run_summary
├── raw_epoch_history.csv          # consolidación de los 6 epoch_history
├── base_extra_data_v2_<ts>/
├── pack_c_v2_l2_<ts>/
├── pack_c_v2_l2_aug_<ts>/
├── pack_c_v2_dropout_<ts>/
├── pack_c_v2_wider_l2_aug_<ts>/
├── pack_c_v2_aug_s010_<ts>/
└── final_eval_ganador_v2/         # test_accuracy + cm + per_class
```

---

## 4. Plots — taxonomía

| # | Archivo | Tipo | Fuente | Conclusión que debe transmitir | Estado |
|---|---|---|---|---|---|
| **01** | `01_comparacion_ej2_vs_ej3.png` | Bar chart progresivo | Ej2 final + Ej3 v2 final | "Pasamos de val_acc X (Ej2) a Y (Ej3 ganador), falta 1.12pp para 0.98" | Regenerate |
| **02** | `02_confusion_matrix_ej3.png` | Heatmap 10×10 | `final_eval` test | "Errores concentrados en clases A/B/C" | Regenerate |
| **03** | `03_per_class_metrics_ej3.png` | Bar chart agrupado | `final_eval` test | "Clase minoritaria 5 + dígitos confundibles tiran abajo macro_f1" | Regenerate |
| **04** | `04_curvas_aprendizaje_ej3.png` | Curvas con band | epoch_history ganador v2 | "Convergencia limpia, train-val gap reducido vs Ej2 baseline" | Regenerate |
| **05** | `05_why_regularization.png` | Bar chart agrupado | cross_v1 stage 2 epoch_history | "Adam tiene el gap más grande → bottleneck es overfit → regularización es la palanca correcta" | **NUEVO** |
| **06** | `06_marginal_contribution.png` | Bar chart horizontal | Ej2 final + Ej3 v2 raw_summary | "Más datos = +10pp (dominante). L2/aug marginales positivos. Dropout/wider/σ_alto NEGATIVOS." | **NUEVO** |
| **07** | `07_val_vs_test_gap.png` | Scatter con y=x | Ej3 v2 + final_eval | "Todas las configs caen ~1pp por debajo de y=x → distribution shift residual confirmado" | **NUEVO** |

**Convenciones para todos los plots:**
- Título descriptivo + subtítulo con métrica explícita (regla 3 CLAUDE.md: declarar el eje de promediación, ej. "media sobre 5 folds × 1 seed").
- Caption en la presentación (ver `guion_presentacion.md`).
- Sin emojis. Paleta consistente (colores fijos por optimizer / técnica para mantener identidad visual entre plots).
- Output PNG @ 130 DPI con `bbox_inches='tight'`.

**Scripts:**
```
ejercicio3/analisis/
├── plot_01_comparacion_ej2_vs_ej3.py     # actualizado
├── plot_02_confusion_matrix.py           # actualizado
├── plot_03_per_class_metrics.py          # actualizado
├── plot_04_curvas_aprendizaje.py         # actualizado
├── plot_05_why_regularization.py         # NUEVO
├── plot_06_marginal_contribution.py      # NUEVO
└── plot_07_val_vs_test_gap.py            # NUEVO
```

---

## 5. Guión de presentación

**Deliverable:** `ejercicio3/guion_presentacion.md`. **Conciso** — los plots hablan, el texto justifica. Cada slide ≤ 10 líneas.

### 5.1 Mapeo slide → feedback profe

| Slide | Plot | Feedback atacado |
|---|---|---|
| 1. Hipótesis (de Ej2) | — | Refuerza el ✅ ya reconocido |
| 2. **Por qué regularización (no arch/HP)** | **05** | ❌ "falta justificación" |
| 3. Plan + resultado global | 01 | Story progression |
| 4. **Qué ayudó y qué no** | **06** | ❌ "reflexiones sin respaldo" (parte 1) |
| 5. Dónde erra el modelo | 02 + 03 | Diagnóstico (pre-requisito para slide 6) |
| 6. **Por qué no llegamos a 0.98** | **07** | ❌ "reflexiones sin respaldo" (parte 2) |
| 7. (opcional) Convergencia | 04 | Si quedó tiempo |

### 5.2 Estructura de cada slide en el guion

```
## Slide N — <título>
- **Plot:** `0N_archivo.png`
- **Qué representa:** <1 línea literal>
- **Por qué lo mostramos:** <1 línea con el feedback que ataca>
- **Conclusión:** <2-3 bullets máx>
- **Pregunta anticipada:** <pregunta + respuesta breve>
```

---

## 6. Validación y sanity checks

### 6.1 Por cada config

- K=5 folds completados (no crash).
- val_acc > 0.95 en al menos 4 de 5 folds.
- ES dispara antes de max_epochs en al menos 3 de 5 folds.

### 6.2 Cross-cell

- Plot 06: delta de "+more_digits vs Ej2 baseline" debe ser positivo y grande (~+10pp esperado). Si no, hay bug en el pipeline de datos.
- Plot 07: los 6 puntos deben caer **consistentemente** por debajo de y=x. Si están dispersos, no hay shift y hay que pivotar la conclusión del slide 6.

### 6.3 Final eval

- Comparar test_acc del nuevo ganador vs Ej3 v1 (0.9688). Si nuevo ≥ v1 → narrativa clean. Si nuevo < v1 → documentar ambos como honest reporting, sin rollback.

### 6.4 Code-level

- Cada cell persiste antes de devolver al master.
- `errors.log` y `STATUS.txt` actualizados en tiempo real.

---

## 7. Riesgos y plan B

| # | Riesgo | Prob | Mitigación |
|---|---|---|---|
| 1 | Nuevo ganador test_acc < 0.9688 | media | Documentar ambos. La coherencia con cross_v1 sigue ganando. |
| 2 | Plot 07 no muestra shift | baja | Pivotar slide 6 a "variance entre seeds" + discutir SEM. |
| 3 | Una cell crashea | media | Saltar + log + continuar. Re-run con menos workers si era el ganador esperado. |
| 4 | Compute > 2h | baja | Dropear `aug_s010`. Quedan 5 configs. |
| 5 | Formato distinto de `more_digits.csv` | muy baja | Smoke test del primer fold del primer config → si crashea, parar y revisar `parse_features`. |

**Plan B independiente del resultado numérico:** la narrativa funciona aunque no lleguemos a 0.98. Plot 07 ES el respaldo a la reflexión "por qué falta 1.12pp" — ese es el respaldo de datos que pidió la profe.

---

## 8. Deliverables y milestones

| Milestone | Deliverable | Criterio de aceptación |
|---|---|---|
| **M0** | 6 configs creados | JSONs válidos, parsean con `load_config` |
| **M1** | 6 configs corridos | `STATUS.txt` muestra 6/6, `raw_summary.csv` consolidado |
| **M2** | Final eval del ganador v2 | `final_eval_ganador_v2/` con accuracy + cm + per_class |
| **M3** | 7 plots generados | PNGs en `ejercicio3/presentacion/` |
| **M4** | README v2 + guión | `ejercicio3/README.md` reescrito + `guion_presentacion.md` nuevo |
| **M5** | Commit + push | Todo en main |

---

## 9. Decisiones explícitas

1. **No tocar `mlp/`** — todo se hace con la infraestructura actual del engine.
2. **No tocar Ej1 ni Ej2** — sólo se citan sus datos.
3. **No rollback** si el nuevo ganador es peor que el v1 — honest reporting siempre.
4. **6 configs, no más, no menos** — los 5 Pack C cubren las reflexiones del README v1 una a una.
5. **`final_eval.py` se reusa**, no se modifica.
6. **Plots concisos** — los plots cuentan la historia, el texto del guion sólo justifica y anticipa preguntas.
7. **Spec y guión en castellano** — consistente con el resto del repo y con la materia.

---

## 10. Open questions / future work (fuera de scope)

- **Augmentación geométrica** (rotaciones, desplazamientos pequeños). Hipótesis: cerraría parte del 1.12pp residual atacando el shift geométrico del test. Mencionada en slide 6 como contra-prueba teórica.
- **Análisis de calibración** del ganador (reliability diagrams). Opcional teórico Ej1, aplicable a Ej3.
- **Robustez a ruido** sobre `digits_test.csv` a distintos σ. Opcional práctico del enunciado.
- **Re-correr el cross_v1 con `more_digits.csv` incluido** — podría cambiar el ranking de arquitecturas. Costo: alto, beneficio: incierto.

---

## 11. Aprobación

- **Diseñado por:** Claude (Opus 4.7) en sesión de brainstorming 2026-05-10 con Nicolás.
- **Aprobado por:** pendiente — Nicolás revisa este documento.
- **Próximo paso:** invocar `superpowers:writing-plans` para producir el plan de implementación detallado milestone por milestone.
