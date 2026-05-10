# Plan del experimento cruzado `cross_v1`

**Fecha:** 2026-05-09 (lanzado de noche, resultados en commit del 2026-05-10).

## Motivación

Las decisiones del Ej2 hasta ahora se tomaron one-at-a-time:
- **Arch sweep** se hizo con `Adam@1e-3` fijo → no sabemos si `arch_shallow` también gana con SGD/Momentum.
- **Optimizer sweep** se hizo con `arch_base` fijo → la decisión "Adam@1e-3" se tomó sobre la arquitectura que después descartamos.
- **LR sweep** se hizo con SGD only y arch_base, con épocas insuficientes para que LR bajos converjan.
- **Batch size** nunca se exploró: todo se hizo con `batch=32`.

Además, durante la auditoría de `cross_v1` identificamos correlaciones potencialmente importantes que las pruebas anteriores trataron como independientes:
- LR × Optimizer (segura, ya observada).
- LR × Batch_size (regla teórica del curso: doblar el batch ≈ doblar el LR).
- Arch × Optimizer (no medida, posible).

Este experimento cruzado **valida** el conjunto de decisiones anteriores bajo perturbación, **mide** las interacciones que la teoría predice, y **reporta** la mejor configuración global con caveats explícitos.

---

## Estructura: dos etapas + estrella

### Etapa 1 — Pre-experimento `LR × Batch × Optimizer`

**Objetivo:** decidir `best_batch_size` por (optimizer, LR), para usarlo como hiperparámetro heredado en la etapa 2.

**Por qué se hace primero:** si dejamos batch fijo en 32 en la etapa 2, repetimos el sesgo de los sweeps anteriores. Si lo metemos como factor adicional en la etapa 2, explota el compute. La solución es decidirlo "por debajo" en una etapa pequeña.

| Factor | Niveles | Cantidad |
|---|---|---|
| Optimizer | `sgd`, `momentum`, `adam` | 3 |
| LR | `5e-4`, `1e-3`, `5e-3` | 3 |
| Batch size | `16`, `64`, `256` | 3 |
| Seeds | `42`, `7` | 2 |
| Arquitectura (fijo) | `arch_shallow` `[784, 128, 10]` | 1 |
| k-folds | 5 | — |

**Total:** 3·3·3 = 27 cells × 2 seeds = **54 jobs × 5 folds = 270 corridas internas**.

**Decisión producida:** un `best_batch.json` con la entrada ganadora por (opt, LR), elegida por máximo `val_acc_final` medio sobre 10 corridas (2 seeds × 5 folds).

**Selección de niveles:**
- LR = {5e-4, 1e-3, 5e-3}: cubren la zona "óptima" de los 3 optimizadores según el optimizer sweep anterior. No incluimos 1e-4 (sub-entrenado en SGD) ni 1e-2 (diverge en Adam) porque no querríamos heredar sus batches a celdas centrales.
- Batch = {16, 64, 256}: span 16×, suficiente para detectar la regla de escalado lineal LR↔batch del curso.
- 2 seeds: SEM ≈ 0.0019, suficiente para decidir batch (no para reportar diferencias finas — eso lo hace la etapa 2).

---

### Etapa 2 — Grid principal `LR × Optimizer × Arquitectura`

**Objetivo:** medir interacciones entre los 3 factores principales sobre el `batch_size` óptimo de etapa 1.

| Factor | Niveles | Cantidad |
|---|---|---|
| LR | `1e-4`, `5e-4`, `1e-3`, `5e-3`, `1e-2` | 5 |
| Optimizer | `sgd`, `momentum`, `adam` | 3 |
| Arquitectura | `arch_shallow`, `arch_base`, `arch_wider`, `arch_deeper` | 4 |
| Batch size | (heredado de etapa 1 por celda) | — |
| Seeds | `42`, `7`, `13` | 3 |
| k-folds | 5 | — |

**Total:** 5·3·4 = 60 cells × 3 seeds = **180 jobs × 5 folds = 900 corridas internas**.

**Herencia de batch para LRs fuera de etapa 1:**
- `LR=1e-4` → batch del LR `5e-4` (más cercano dentro del mismo opt).
- `LR=1e-2` → batch del LR `5e-3` (más cercano dentro del mismo opt).

Asunción explícita: el batch óptimo no varía bruscamente entre LRs adyacentes para el mismo optimizer.

---

### Etapa 2b — Estrella batch alrededor del centro

**Objetivo:** caracterizar el efecto del batch_size en el centro `shallow + Adam@1e-3` con resolución fina, ya que la etapa 1 sólo midió 3 batches.

| Factor | Niveles |
|---|---|
| Batch size | `16`, `32`, `64`, `128`, `256` |
| Seeds | `42`, `7`, `13` |
| Arch (fijo) | `arch_shallow` |
| Opt (fijo) | `adam` |
| LR (fijo) | `1e-3` |

**Total:** 5 batches × 3 seeds = **15 jobs × 5 folds = 75 corridas internas**.

---

## Hiperparámetros fijos en TODAS las celdas (las 3 etapas)

| Parámetro | Valor |
|---|---|
| Loss | `cross_entropy` (combinada con `softmax` en última capa) |
| Preprocessing | `zscore` (sobre features), `one_hot_targets=true` |
| Split | k-folds = **5**, estratificado |
| Inicialización | `auto` → He para ReLU |
| Activaciones (todas las archs) | ocultas: `relu`; salida: `softmax` |
| Regularización | l2=0, dropout=0, sin lr_schedule, sin augmentation |
| Early stopping | **`patience=20`** sobre `val_loss` (cross-entropy), restaura `best_weights` al cortar |
| Métricas reportadas en final | val_acc, macro_precision/recall/F1, val_loss CE, train_loss CE — todas evaluadas en `best_weights` (fix #4 aplicado al MLP) |
| `momentum` β | 0.9 |
| `adam` β1, β2, ε | 0.9, 0.999, 1e-8 |

### `max_epochs` por (opt, LR)

Auditado previamente con datos del optimizer sweep + LR sweep extendido. El corte real lo decide la curva (vía ES); `max_epochs` actúa sólo como cota dura.

| optimizer | 1e-4 | 5e-4 | 1e-3 | 5e-3 | 1e-2 |
|---|---|---|---|---|---|
| sgd | **200*** | 300 | 200 | 100 | 80 |
| momentum | 250 | 150 | 80 | 40 | 40 |
| adam | 60 | 40 | 40 | 30 | 30 |

\*`SGD@1e-4` capeado a 200: la auditoría mostró que no converge en 600 ep tampoco; se reporta como referencia "LR demasiado bajo para SGD" — **la celda no estará convergida y se documenta así explícitamente** en el análisis.

---

## Matriz completa: 249 jobs

| Etapa | Cells | Seeds | Jobs | Corridas internas (k=5) |
|---|---|---|---|---|
| 1 — Pre LR×Batch×Opt | 27 | 2 | 54 | 270 |
| 2 — Main LR×Opt×Arch | 60 | 3 | 180 | 900 |
| 2b — Estrella batch | 5 | 3 | 15 | 75 |
| **Total** | **92** | — | **249** | **1245** |

### Mapeo completo de cells, etapa por etapa

**Etapa 1** (27 cells, 2 seeds c/u → 54 jobs):

```
opt × lr × batch
{sgd, momentum, adam} × {5e-4, 1e-3, 5e-3} × {16, 64, 256}
```

**Etapa 2 main** (60 cells, 3 seeds c/u → 180 jobs):

```
arch × opt × lr
{shallow, base, wider, deeper} × {sgd, momentum, adam} × {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}
```

con `batch_size` derivado del `best_batch.json` de etapa 1.

**Etapa 2b** (5 cells, 3 seeds c/u → 15 jobs):

```
batch ∈ {16, 32, 64, 128, 256}
```

con `arch=shallow`, `opt=adam`, `lr=1e-3` fijos.

---

## Compute estimado

Basado en mediciones previas (~2s por fold-época en `arch_shallow`, escala ~1.2× en `arch_wider`).

| Etapa | Wall-clock (8 workers) |
|---|---|
| 1 | ~1.0 h |
| 2 main | ~5.0 h |
| 2b | ~0.3 h |
| Plots + notas + commit | ~0.3 h |
| **Total** | **~6.5 h** |

Margen sobre el budget de 8 h → ~1.5 h de holgura.

---

## Pipeline automatizado

Un solo proceso (`pipeline.py`) corre todo bajo `caffeinate -dimsu` (la Mac no entra en sleep):

```
1. Stage 1 (54 jobs)            → output/cross_v1/stage1/
2. decide_best_batch.py         → output/cross_v1/best_batch.json
3. Stage 2 main (180 jobs)      → output/cross_v1/stage2/
4. Stage 2b (15 jobs)           → output/cross_v1/stage2b/
5. Plots                        → ejercicio2_experimentacion/analisis/cross_v1/
6. Notas                        → Notas/ejercicio 2/Experimentos/{Pre_LR_Batch_Opt, Cross_LR_Opt_Arch}/
7. git add -A && git commit && git push
```

### Garantías de robustez

- **Persistencia inmediata por cell:** cada job escribe sus CSVs a `output/cross_v1/<stage>/<cell_id>/{summary.csv, history.csv}` ANTES de devolver al master. Si se cae la Mac/proceso, no se pierde lo que ya terminó.
- **Errores no matan al pipeline:** si una cell crashea, se loguea en `errors.log` y el pipeline sigue. Las cells faltantes se reportan explícitamente en el análisis final.
- **Status file:** `output/cross_v1/STATUS.txt` se actualiza cada cell completada con timestamp y progreso.
- **Commit defensivo:** si el push falla (auth, red), el commit queda local — se pushea a la mañana sin pérdida.

---

## Deliverables

Al despertar:

| Archivo | Contenido |
|---|---|
| `Notas/ejercicio 2/Experimentos/PLAN_cross_v1.md` | (este archivo) |
| `Notas/ejercicio 2/Experimentos/Pre_LR_Batch_Opt/analisis.md` | Tablas + plots de etapa 1, decisión `best_batch` |
| `Notas/ejercicio 2/Experimentos/Cross_LR_Opt_Arch/analisis.md` | Tablas + plots etapas 2 + 2b, mejor config global, caveats |
| `output/cross_v1/{stage1,stage2,stage2b}/raw.csv` | Datos crudos consolidados |
| `output/cross_v1/{stage1,stage2,stage2b}/epoch_history.csv` | Curvas por época |
| `output/cross_v1/best_batch.json` | Decisión de batch por (opt, LR) |
| `ejercicio2_experimentacion/analisis/cross_v1/{stage1,stage2,stage2b}/*.png` | Plots |

---

## Limitaciones conocidas (declaradas a priori)

- **`SGD@1e-4` no converge en 200 ep**: se incluye sólo como referencia visual. Reportada como "no convergida" en las tablas.
- **3 seeds en etapa 2 main**: SEM ≈ 0.0016 sobre val_acc. Distinguir diferencias ≥0.005 con confianza, no menores.
- **Batch size en etapa 2b**: medido sólo en el centro `shallow + Adam@1e-3`. Si el efecto del batch interactúa con arch o opt, no lo veríamos.
- **No se varía L2/dropout/augmentation/init/activación**: este experimento es sobre **optimización**, no regularización. La regularización es el experimento siguiente sobre el centro encontrado acá.
- **Herencia de batch a LRs extremos** (1e-4 y 1e-2): asunción no medida.
- **Patience=20** auditada como compromiso único; podría ser sub-óptima para Adam (más eficiente con 10) o SGD plano (más cómoda con 40), pero la comparabilidad entre celdas pesa más.

---

## Hipótesis a contrastar

1. **`arch_shallow` sigue ganando bajo SGD y Momentum** (no sólo Adam) → si NO, hay interacción Arch×Opt y la decisión "shallow es la mejor" se desinfla.
2. **El LR óptimo de cada opt no cambia mucho entre arquitecturas** → si NO, la auditoría tenía razón en que no se puede separar LR de Arch.
3. **El batch óptimo escala con el LR siguiendo la regla lineal del curso** → si SÍ, tenemos la "interacción esperada" reportable.
4. **`Adam@1e-3 + arch_shallow` sigue siendo la mejor config global** → si NO, hay que actualizar el centro del Ej2.
