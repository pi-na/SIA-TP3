# Todos los experimentos del Ej2 — sesión del 2026-05-09 / 2026-05-10

Resumen de **TODO** lo que se hizo en esta sesión, con links a los análisis correspondientes y la lista completa de archivos creados o editados.

---

## ✅ Resultado final del Ej2

**Configuración óptima:** `arch_shallow` (784→128→10) + Adam (β1=0.9, β2=0.999, ε=1e-8) + **LR = 1e-3** + **batch_size = 64**.

- val_acc ≈ 0.957 ± 0.005 (sobre 75 corridas en el [tiebreaker](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Arch_tiebreaker/analisis.md))
- macro F1 ≈ 0.852
- best_epoch ≈ 5 (ES patience=20)

**Cómo se llegó a esa decisión** (cadena de evidencias):

1. [Arch sweep](Arquitectura.md) → `shallow` (con caveat: Adam@1e-3 fijo).
2. [Cross-experiment LR×Opt×Arch](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md) → confirmó Adam y reveló que `wider` aparenta ganar levemente a `shallow` en LR=1e-3.
3. [Pre-experimento LR×Batch×Opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md) → batch óptimo para Adam@1e-3 = **64** (no 32).
4. [Tiebreaker arch (15 seeds × k=5)](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Arch_tiebreaker/analisis.md) → **z=0.65, wider vs shallow indistinguibles** al 95%. Por Occam, ganador = `shallow`.

**Correlaciones HP×HP descubiertas:** ver [`IMPORTANTE_CORRELACIONES.md`](IMPORTANTE_CORRELACIONES.md) (LR×Opt, LR×Batch, Arch×LR, Arch×Opt + correlaciones ocultas en código).

---

## Línea de tiempo de experimentos

### 1. Análisis del Arch sweep previo (one-at-a-time)

**Qué se hizo:** lectura del `summary.csv` del [Arch sweep](Arquitectura.md) corrido previamente (4 archs × 5 seeds × 5 folds, Adam@1e-3, 50 ep, ES patience=10, batch=32). Se escribió el análisis con conclusión.

**Conclusión inicial:** `arch_shallow` (784→128→10) gana en val_acc, F1, precision, recall y val_loss simultáneamente. Por Occam + métricas → arquitectura óptima.

**Caveat detectado más tarde:** se hizo con Adam@1e-3 fijo. La conclusión está condicionada a ese LR.

### 2. LR_segundo_intento — sweep de LR extendido a 500 épocas

**Qué se hizo:** segundo sweep de LR para SGD con `arch_shallow` (la "óptima" de paso 1), 5 LRs × 5 seeds × 5 folds × 500 épocas, sin ES, sin regularización. Objetivo: ver si los LRs bajos converjan dado tiempo suficiente (en el sweep original quedaban a media-aprender en 50 ep).

**Lo que pasó:**
- Primer intento: crash a mitad, sin persistencia incremental → se perdieron 17/25 cells. Se rescataron 8 cells del tmpdir (LR=1e-4 5 seeds, LR=5e-4 3 seeds).
- Plan revisado: relanzar sólo lo faltante (LR=1e-3 250ep, LR=5e-3 150ep, LR=1e-2 150ep, 3 seeds).
- Plan abortado: el usuario notó que un LR sweep sólo con SGD es de valor limitado dado que ya tenemos optimizer sweep.

**Aprendizaje metodológico:** el runner debe persistir cada cell **antes** de devolver al master, no acumular en memoria.

### 3. Audit del sistema de early stopping y `max_epochs` (parallel agents)

**Qué se hizo:** dos agentes en paralelo auditaron, **sobre datos reales** de los sweeps anteriores:
- ¿Es `patience=20` el valor correcto para ES?
- ¿Son los `max_epochs` propuestos suficientes para no cortar antes de convergencia?

#### Datos fuente del audit (todos disponibles en el repo)

| Archivo                                                                  | Contenido                                                                                                         | Cómo se usó                                                                                                        |           |                  |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | --------- | ---------------- |
| `ejercicio2_experimentacion/analisis/optimizer/epoch_history.csv`        | Momentum + Adam × 5 LR × 5 seeds × 5 folds × 70 ep, arch_base                                                     | Curvas para detectar `best_epoch` por (opt, LR), longitud de subidas transitorias, y                               | Δval_loss | cerca del mínimo |
| `ejercicio2_experimentacion/analisis/lr/epoch_history.csv`               | SGD × 5 LR × 4 archs × 5 seeds × 5 folds × 50 ep                                                                  | Confirmación de que SGD@LR-bajo seguía bajando al final del presupuesto (best_epoch=49 en 4 de los 5 LRs)          |           |                  |
| `ejercicio2_experimentacion/output/lr_segundo_intento/epoch_history.csv` | Datos rescatados del crash anterior: SGD@1e-4 con 5 seeds y SGD@5e-4 con 3 seeds, arch_shallow, **500 ep** sin ES | Única fuente directa para juzgar si SGD@1e-4 converge dado tiempo suficiente. Es lo que justifica capear esa celda |           |                  |

#### Evidencias específicas detrás de cada conclusión

**`patience=20` está bien como valor único:**
- En el optimizer sweep, la subida transitoria máxima observada en val_loss (épocas que sube antes de bajar a un mínimo mejor) fue de **6 épocas** (Momentum@5e-4, p99 ≤ 5.2 en todos los regímenes que llegan a un mínimo interior).
- `patience=20` deja margen ~3× sobre el peor caso real → no corta espuriamente.
- Para Adam@LR alto (curvas que sólo suben tras el mínimo en ep 0-3), `patience=20` "gasta" 20 épocas inútiles tras el mínimo. Más eficiente sería `patience=10` (max run recuperada=5, p99≤4.9), pero la simplicidad del valor único pesa más.

**`SGD@1e-4` no converge en presupuesto razonable:**
- En el LR sweep extendido a 500 épocas (`output/lr_segundo_intento/epoch_history.csv`), SGD@1e-4 con arch_shallow:
  - `best_epoch` = 499 (la última época) → la curva sigue bajando al final del presupuesto.
  - `slope` ≈ −1.4e-4 / época (mejora real, pero del orden del ruido `std ≈ 2.3e-4`).
  - `ep99%` (época en la que se alcanza el 99% del descenso final) ≈ 368.
- Extrapolación: para que esa celda toque un mínimo claro habría que correr ≥1000 ep. Costo no se justifica para un LR que la teoría ya identifica como "demasiado bajo para SGD vanilla".
- **Decisión:** capear a 200 ep y reportar explícitamente como "no convergida en presupuesto" en cada análisis donde aparezca. Es coherente con la [clase de optimizadores](../../docs/clase_optimizadores/clase%20optimizadores.pdf) (SGD vanilla con LR muy bajo necesita muchísimas épocas).

**`Adam@1e-2` diverge desde init:**
- En el optimizer sweep, Adam@1e-2 tiene `best_epoch = 0` en la curva promedio → val_loss mínimo está en la **inicialización**, antes del primer paso. Cualquier paso de Adam con ese LR empeora val_loss.
- Esto es coherente con el aprendizaje del optimizer sweep: Adam adapta internamente el tamaño de paso, así que un LR nominal alto rompe la trayectoria inmediatamente.
- **Decisión:** capear a 30 ep. ES con patience=20 mata la celda rápido y los `best_weights` restaurados son los iniciales (sin entrenamiento). Reportar como "diverge desde init".

**`max_epochs` celda a celda:** la tabla por (opt, LR) que terminó usándose en el cross-experiment está en [`PLAN_cross_v1.md`](PLAN%20de%20todos%20los%20experimentos%20cruzados%20cross_v1.md). Cada valor está justificado por el `best_epoch` observado en los datos fuente arriba + margen para que ES (patience=20) tenga lugar de actuar antes del techo.

### 4. Fix #4 — métricas finales consistentes entre celdas

**Bug detectado en el audit:** `mlp/network.py` sólo guardaba `best_weights` cuando ES estaba activo. Si ES no disparaba, las métricas finales se computaban sobre los pesos del **último** epoch, no del **mejor**. Comparar celdas con/sin ES dispara producía mediciones de objetos distintos.

**Fix aplicado:**
- `mlp/network.py:fit()` → `best_weights` se trackea **siempre**, se restaura al cerrar fit en cualquier caso.
- `mlp/train.py` → métricas finales se evalúan con los pesos restaurados; `train_loss_final` y `val_loss_final` se toman de `history[best_epoch]` (no `[-1]`). Se agregaron `train_loss_last` y `val_loss_last` por si se necesitan.

**Verificado** con un test sintético: con ES off, los pesos post-fit son los del best_epoch, no del último.

### 5. Plan brainstormed del cross-experiment

**Qué se hizo:** discusión sobre objetivos de un experimento cruzado. Se descartó el factorial 2^k (por rigidez de niveles) y el centro+slices puro (por la trampa de las "slices 2D mienten"). Se llegó al diseño final de 2 etapas:
- Etapa 1 (Pre LR×Batch×Opt): decidir batch óptimo por (opt, LR).
- Etapa 2 (Cross LR×Opt×Arch): grid 3D principal con batch heredado.
- Etapa 2b (Estrella batch): perturbar el centro en batch_size con resolución fina.

**Plan documentado:** [`PLAN_cross_v1.md`](PLAN%20de%20todos%20los%20experimentos%20cruzados%20cross_v1.md).

### 6. Pipeline cross_v1 — ejecución unattended

**Qué se hizo:** pipeline en un solo proceso bajo `caffeinate -dimsu` (la Mac no entra en sleep) que encadena las 3 etapas + plots + notas + commit + push de forma autónoma. Robustez: cada cell persiste al disco antes de devolver al master, errores no matan el pipeline, status file actualizado en cada cell.

**Resultado:**
- Stage 1: 54/54 OK en 45 min → `best_batch.json` decidido.
- Stage 2 main: 180/180 OK en ~4 h.
- Stage 2b: 15/15 OK en 5 min.
- **Total: 249 jobs, 0 failures, 4h 44min wall clock.**
- Commit `e896346` pusheado a `main` automáticamente.

**Análisis:** [`Pre_LR_Batch_Opt/analisis.md`](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md) + [`Cross_LR_Opt_Arch/analisis.md`](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md).

### 7. Descubrimiento de correlaciones cruzadas

**Qué se hizo:** análisis transversal de los datos de cross_v1 para detectar interacciones HP×HP que los sweeps one-at-a-time no podían ver.

**Hallazgos:**
- LR × Optimizer (10× diferencia entre Adam y SGD).
- LR × Batch (regla lineal confirmada empíricamente para Adam).
- Arch × LR (el ranking de archs **se reordena** según el LR — `arch_wider` gana en LR=1e-3 y colapsa en LR=1e-2).
- Arch × Optimizer (shallow es robusta; wider es especialista de Adam).
- Correlaciones ocultas en código (seed=split+init, batch order independiente de seed, patience×max_epochs, fix #4 ya aplicado).

**Análisis:** [`IMPORTANTE_CORRELACIONES.md`](IMPORTANTE_CORRELACIONES.md).

### 8. Arch tiebreaker (en curso al cierre)

**Qué se está haciendo:** el cross_v1 dejó un empate estadístico entre `arch_wider` y `arch_shallow` (diff 0.0011, SEM ~0.001). Tiebreaker: 4 archs × Adam × {5e-4, 1e-3} × 12 seeds NUEVOS × k=5 = 96 jobs. Se combina con los 3 seeds previos del cross_v1 → 15 seeds totales = 75 corridas/cell, SEM ≈ 0.0006, suficiente para resolver.

**Ejecución:** corriendo bajo `caffeinate`, ~60 min wall. Al terminar el script auto-escribe `Arch_tiebreaker/analisis.md` con tabla, plot, ganador y test z-score wider vs shallow.

---

## Archivos creados o editados en esta sesión

### Markdown (`.md`)

**Notas (deliverables del Ej2):**
- `Notas/ejercicio 2/Experimentos y analisis/Arch/Arquitectura.md` *(editado: análisis del Arch sweep + conclusión "shallow óptima")*
- `Notas/ejercicio 2/Experimentos y analisis/IMPORTANTE_CORRELACIONES.md` *(nuevo: correlaciones cruzadas y técnica de descubrimiento)*
- `Notas/ejercicio 2/Experimentos y analisis/Todos los experimentos.md` *(este archivo)*
- `Notas/ejercicio 2/Experimentos/PLAN_cross_v1.md` *(nuevo: plan completo del cross-experiment)*
- `Notas/ejercicio 2/Experimentos/Pre_LR_Batch_Opt/analisis.md` *(nuevo: stage 1 análisis)*
- `Notas/ejercicio 2/Experimentos/Cross_LR_Opt_Arch/analisis.md` *(nuevo: stage 2 + 2b análisis)*
- `Notas/ejercicio 2/Experimentos/Arch_tiebreaker/analisis.md` *(pendiente: lo escribe el script al terminar)*

### No-Markdown (código, datos, configs, plots)

**Código del MLP (fixes):**
- `mlp/network.py` *(editado: fix #4 — `best_weights` siempre)*
- `mlp/train.py` *(editado: fix #4 — métricas en best_epoch, agrega `*_last` y renombra `total_epochs_run`)*

**Scripts del runner cross_v1 (nuevos):**
- `ejercicio2_experimentacion/scripts/cross_v1/runner.py` *(runner universal cell-list, basado en plantilla multiprocess)*
- `ejercicio2_experimentacion/scripts/cross_v1/pipeline.py` *(orquestador de las 3 etapas)*
- `ejercicio2_experimentacion/scripts/cross_v1/plot_and_notes.py` *(plots + notas auto-generadas)*
- `ejercicio2_experimentacion/scripts/cross_v1/arch_tiebreaker.py` *(tiebreaker de arquitecturas)*

**Scripts del LR_segundo_intento (descartados parcialmente):**
- `ejercicio2_experimentacion/scripts/run_lr_segundo_intento.py` *(primer runner, abandonado tras crash)*
- `ejercicio2_experimentacion/scripts/run_lr_segundo_intento_phase2.py` *(segundo intento, también abortado por replanteo)*
- `ejercicio2_experimentacion/scripts/rebuild_lr_segundo_intento.py` *(rescate de tmpdir tras crash)*
- `ejercicio2_experimentacion/scripts/combine_and_plot_lr_segundo_intento.py` *(combine + plots)*
- `ejercicio2_experimentacion/scripts/plot_lr_segundo_intento.py` *(plots stand-alone)*

**Configs (nuevos):**
- `ejercicio2_experimentacion/configs/sweeps/lr_segundo_intento/arch_shallow.json`
- `ejercicio2_experimentacion/configs/sweeps/lr_segundo_intento/sweep_config.json`
- `ejercicio2_experimentacion/configs/sweeps/lr_segundo_intento/sweep_config_phase2.json`

**Outputs raw del cross_v1 (auto-generados):**
- `ejercicio2_experimentacion/output/cross_v1/best_batch.json`
- `ejercicio2_experimentacion/output/cross_v1/{stage1,stage2,stage2b}/raw.csv`
- `ejercicio2_experimentacion/output/cross_v1/{stage1,stage2,stage2b}/epoch_history.csv`
- `ejercicio2_experimentacion/output/cross_v1/{stage1,stage2,stage2b}/<cell_id>/{summary.csv,history.csv}` (249 cells)
- `ejercicio2_experimentacion/output/cross_v1/{STATUS.txt,pipeline.log,pipeline_stdout.log}`

**Plots y CSVs agregados (cross_v1 análisis):**
- `ejercicio2_experimentacion/analisis/cross_v1/stage1/{stage1_summary.csv, stage1_heatmap_val_acc.png, stage1_val_acc_vs_batch.png}`
- `ejercicio2_experimentacion/analisis/cross_v1/stage2/{stage2_summary.csv, stage2_val_acc_vs_lr_per_opt.png, stage2_heatmap_arch_lr.png, stage2_convergence_shallow.png}`
- `ejercicio2_experimentacion/analisis/cross_v1/stage2b/{stage2b_summary.csv, stage2b_val_acc_vs_batch.png}`

**Plots/CSVs copiados a las notas:**
- `Notas/ejercicio 2/Experimentos/Pre_LR_Batch_Opt/{stage1_agg.csv, stage1_heatmap_val_acc.png, stage1_val_acc_vs_batch.png}`
- `Notas/ejercicio 2/Experimentos/Cross_LR_Opt_Arch/{stage2_agg.csv, stage2b_agg.csv, *.png}`

**Tiebreaker (en curso, se generará al terminar):**
- `ejercicio2_experimentacion/output/arch_tiebreaker/{raw.csv, summary_combined.csv, raw_combined.csv, run.log, STATUS.txt}`
- `ejercicio2_experimentacion/output/arch_tiebreaker/<cell_id>/` (96 cells)
- `ejercicio2_experimentacion/analisis/arch_tiebreaker/tiebreaker_val_acc.png`
- `Notas/ejercicio 2/Experimentos/Arch_tiebreaker/{analisis.md, summary_combined.csv, tiebreaker_val_acc.png}`

**Cosmético:**
- `.obsidian/workspace.json` *(state local de Obsidian, ignorable)*
