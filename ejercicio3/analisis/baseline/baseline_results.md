### Resultados paso 1 — Baseline Ej3 con `more_digits.csv`

**A) Convergencia** (sobre 15 corridas = 3 seeds × 5 folds del CV interno).

![[baseline_optimal_convergence.png]]

- `best_epoch` promedio = **5.7 ± 1.1** (range `[4, 8]`).
- `stop_epoch` (corte ES) promedio = **25.7 ± 1.1** (range `[24, 28]`).
- `max_epochs=50`: 0/15 corridas llegaron al límite duro.

**B) Generalización interna (CV)**.

| Métrica | Train (CV, 15 corridas) | Val (CV, 15 corridas) |
| --- | --- | --- |
| accuracy        | 0.9984 ± 0.0012 | **0.9699 ± 0.0029** |
| macro_precision | (no almacenada) | 0.9609 ± 0.0052 |
| macro_recall    | (no almacenada) | 0.9541 ± 0.0057 |
| macro_F1        | (no almacenada) | **0.9572 ± 0.0047** |
| CE loss         | 0.0119 ± 0.0050 | 0.1238 ± 0.0105 |
| best_epoch      | — | 5.7 ± 1.1 |

**C) Generalización externa (test sobre `digits_test.csv`)**.

![[baseline_test_confusion_matrix.png]]

| Métrica | Val CV (interno) | **Test** (digits_test.csv) | Δ (val CV − test) |
| --- | --- | --- | --- |
| accuracy        | 0.9699 ± 0.0029 | **0.9616 ± 0.0025** | +0.0084 |
| macro_precision | 0.9609 | 0.9618 ± 0.0026 | -0.0009 |
| macro_recall    | 0.9541 | 0.9606 ± 0.0026 | -0.0066 |
| macro_F1        | 0.9572 | **0.9609 ± 0.0026** | -0.0037 |

**Métricas por clase en test** (mean sobre 3 seeds):

| clase | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| 0 | 0.964 | 0.990 | 0.977 | 245 |
| 1 | 0.979 | 0.989 | 0.984 | 283 |
| 2 | 0.956 | 0.960 | 0.958 | 258 |
| 3 | 0.934 | 0.983 | 0.958 | 252 |
| 4 | 0.982 | 0.973 | 0.977 | 245 |
| 5 | 0.967 | 0.912 | 0.938 | 223 |
| 6 | 0.960 | 0.978 | 0.969 | 239 |
| 7 | 0.953 | 0.966 | 0.959 | 257 |
| **8** | **0.968** | **0.909** | **0.938** | 243 |
| 9 | 0.955 | 0.946 | 0.950 | 252 |


**Test acc excluyendo clase 8** = 0.9672 ± 0.0017


**Comparación con Ej2 (sin `more_digits.csv`)**:

| Configuración | Test acc | Test macro_F1 |
| --- | --- | --- |
| Ej2 (sin more_digits, sin reg) | 0.8529 ± 0.0034 | 0.8062 ± 0.0034 |
| **Ej3 baseline (+more_digits, sin reg)** | **0.9616 ± 0.0025** | **0.9609 ± 0.0026** |
| Δ (Ej3 − Ej2) | **+0.1087** | **+0.1547** |
