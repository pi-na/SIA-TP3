# Análisis del sweep de arquitecturas — Ejercicio 2

**Experimento:** 4 arquitecturas × 5 seeds × 5 folds = 25 corridas por arquitectura (100 totales).
**Fijo en todas:** Adam (lr=1e-3), batch=32, 50 épocas máx, early stopping (patience=10), z-score, init `auto` (He para ReLU), softmax + cross-entropy.
**Datos crudos:** `ejercicio2_experimentacion/analisis/arch/raw.csv` · `summary.csv`

> Nota sobre promedios (regla 3 del CLAUDE.md): todas las medias y desvíos en este análisis se calculan sobre **25 corridas = 5 seeds × 5 folds**. La varianza reportada mezcla las dos fuentes (no se separa seed de fold).

---

## Arquitecturas comparadas

| Nombre | `layer_sizes` | Activaciones | Capas ocultas | Parámetros aprox. |
|---|---|---|---|---|
| `arch_shallow` | `[784, 128, 10]`         | relu, softmax           | 1 | ~101 k |
| `arch_base`    | `[784, 128, 64, 10]`     | relu, relu, softmax     | 2 | ~109 k |
| `arch_wider`   | `[784, 256, 128, 10]`    | relu, relu, softmax     | 2 | ~235 k |
| `arch_deeper`  | `[784, 128, 64, 32, 10]` | relu, relu, relu, softmax | 3 | ~111 k |

Hipótesis previa (anclada en clase de regularización + métricas/sobreajuste):

- **`shallow` vs `base/wider`**: si el dataset (8000 imágenes 28×28) es chico relativo a los parámetros, agregar capas/anchos no debería ayudar y podría empeorar la varianza de validación (más capacidad → más overfitting).
- **`deeper`**: 3 capas ocultas con ReLU tiende a tener problemas de optimización en redes desde cero sin batch-norm; puede converger más lento o a peor mínimo.
- **`wider`**: más parámetros pero misma profundidad que `base`; debería ser equivalente o levemente mejor en train, pero igual o peor en val si hay overfitting.

---

## Resultados — media ± std sobre 25 corridas (5 seeds × 5 folds)

| Arquitectura | val_acc | F1 macro | macro_precision | macro_recall | val_loss (CE) | train_acc | best_epoch | total_epochs |
|---|---|---|---|---|---|---|---|---|
| **`arch_shallow`** | **0.9576 ± 0.0061** | **0.8522 ± 0.0089** | 0.8554 ± 0.0092 | 0.8499 ± 0.0096 | 0.2030 ± 0.041 | 0.9975 ± 0.002 | 4.2 ± 1.3 | 15.2 ± 1.3 |
| `arch_wider`   | 0.9557 ± 0.0062 | 0.8505 ± 0.0075 | 0.8535 ± 0.0074 | 0.8484 ± 0.0090 | 0.2511 ± 0.052 | 0.9929 ± 0.005 | 2.2 ± 1.0 | 13.2 ± 1.0 |
| `arch_base`    | 0.9554 ± 0.0059 | 0.8500 ± 0.0078 | 0.8534 ± 0.0071 | 0.8476 ± 0.0093 | 0.2272 ± 0.040 | 0.9937 ± 0.005 | 2.6 ± 1.1 | 13.6 ± 1.1 |
| `arch_deeper`  | 0.9522 ± 0.0059 | 0.8471 ± 0.0079 | 0.8504 ± 0.0088 | 0.8446 ± 0.0084 | 0.2498 ± 0.034 | 0.9897 ± 0.005 | 2.1 ± 0.8 | 13.1 ± 0.8 |

(Todas las métricas P/R/F1 son **macro-average sobre las 10 clases**, regla 4.)

---

## Observaciones

### 1. La arquitectura no es el factor decisivo
Las 4 configuraciones quedan en una franja angosta: val_acc ∈ [0.952, 0.958] (rango ≈ 0.006, comparable al desvío intra-arquitectura de ~0.006). En F1 macro pasa lo mismo: rango ≈ 0.005 vs std ≈ 0.008. **Las diferencias entre arquitecturas son del mismo orden que el ruido seed×fold** → ninguna gana de forma robusta sobre las demás en términos de capacidad de clasificación.

### 2. `shallow` gana en todas las métricas, con el modelo más chico
Pese al empate estadístico, `shallow` queda primero en **val_acc, F1, precision y recall** simultáneamente, con la **menor val_loss** (0.203 vs 0.227–0.251). Que la val_loss sea ~10–20% menor mientras las accuracies son casi iguales sugiere que `shallow` produce **probabilidades mejor calibradas**: la CE penaliza confianza alta en la clase equivocada, y los modelos más grandes (especialmente `wider` y `deeper`) muestran señales leves de overconfidence en errores.

### 3. Más capacidad → leve overfitting
`shallow` tiene la mayor train_acc (0.9975) pero también la mejor val_acc. Sin embargo, mirando **gap = train − val**:
- shallow: 0.9975 − 0.9576 = **0.0399**
- base:    0.9937 − 0.9554 = 0.0383
- wider:   0.9929 − 0.9557 = 0.0372
- deeper:  0.9897 − 0.9522 = 0.0375

Los gaps son comparables, así que no es overfitting "clásico", pero la val_loss más alta de los modelos profundos/anchos sí indica que las predicciones erradas son más confiadas — consistente con la clase de sobreajuste (más capacidad ⇒ más probabilidad de memorizar patrones espurios).

### 4. Convergencia: todos paran rápido por early stopping
`best_epoch` ≈ 2–4 en todas las configs (con patience=10 ⇒ `total_epochs` ≈ 13–15). Adam con lr=1e-3 sobre un dataset de 8000 ejemplos converge en pocos pasos para todas las arquitecturas. Esto significa que **el early stopping se está disparando antes de que la profundidad/anchura tenga tiempo de mostrar diferencias estructurales**. Sería distinto sin early stopping y con LR más chico — pero ese es justamente el siguiente experimento (sweep LR con SGD básico).

### 5. `deeper` es la más débil
`deeper` queda último en val_acc, F1 y val_loss. Coincide con la hipótesis: 3 capas ReLU sin técnicas modernas (batch-norm, residuals) optimizan peor. No está colapsando — sigue por encima de 95% — pero no aporta nada y tiene la peor val_loss.

---

## Conclusión: arquitectura óptima

**`arch_shallow` (784 → 128 → 10)** es la arquitectura óptima de este sweep:

- **Mejor en todas las métricas** (val_acc, F1, precision, recall, val_loss).
- **Modelo más chico** (1 capa oculta vs 2–3) → más rápido de entrenar, menos parámetros, menos riesgo de overfitting en el dataset reducido del Ej2.
- Las ganancias de `wider`/`base`/`deeper` no aparecen, y la val_loss más alta de los modelos profundos sugiere calibración peor.

Por **navaja de Occam + métricas** → seguimos los siguientes experimentos (LR, optimizador, regularización) sobre `arch_shallow`.

> **Caveat:** la diferencia con `wider`/`base` es de la magnitud del ruido seed×fold. La elección se justifica más por **simplicidad + val_loss** que por superioridad estadística clara en accuracy.
