# Sweep LR multi-seed — perceptrón nonlinear

Experimento: 5 seeds × 3 LRs × 5 folds = 75 entrenamientos. `epochs=500` (suficiente para convergencia segun single-seed). Seeds: [np.int64(7), np.int64(13), np.int64(21), np.int64(42), np.int64(99)].

![Dispersion](dispersion.png)

## Resumen agregado por LR

**Total (todos los seeds × folds):**

| lr | MSE test (mean ± std) | ‖w‖ (mean ± std) | F1 (mean ± std) |
|---|---|---|---|
| 0.0001 | 0.01128 ± 0.00052 | 1.7510 ± 0.0096 | 0.5097 ± 0.0112 |
| 0.001 | 0.01099 ± 0.00058 | 1.9829 ± 0.0196 | 0.4996 ± 0.0108 |
| 0.01 | 0.01099 ± 0.00058 | 1.9792 ± 0.0195 | 0.5011 ± 0.0115 |

**Dispersion entre seeds** (cada celda usa el promedio sobre folds de cada seed):

| lr | MSE test seed-std | ‖w‖ seed-std | F1 seed-std |
|---|---|---|---|
| 0.0001 | 0.00000 | 0.0013 | 0.0008 |
| 0.001 | 0.00000 | 0.0003 | 0.0001 |
| 0.01 | 0.00000 | 0.0018 | 0.0012 |

## Per-seed (mean sobre folds)

| lr | seed | MSE test | ‖w‖ | F1 |
|---|---|---|---|---|
| 0.0001 | 7 | 0.01128 | 1.7528 | 0.5084 |
| 0.0001 | 13 | 0.01128 | 1.7517 | 0.5105 |
| 0.0001 | 21 | 0.01128 | 1.7498 | 0.5094 |
| 0.0001 | 42 | 0.01129 | 1.7496 | 0.5102 |
| 0.0001 | 99 | 0.01128 | 1.7512 | 0.5098 |
| 0.001 | 7 | 0.01098 | 1.9825 | 0.4995 |
| 0.001 | 13 | 0.01099 | 1.9830 | 0.4996 |
| 0.001 | 21 | 0.01098 | 1.9832 | 0.4998 |
| 0.001 | 42 | 0.01099 | 1.9831 | 0.4994 |
| 0.001 | 99 | 0.01098 | 1.9829 | 0.4996 |
| 0.01 | 7 | 0.01099 | 1.9768 | 0.5002 |
| 0.01 | 13 | 0.01100 | 1.9783 | 0.5014 |
| 0.01 | 21 | 0.01099 | 1.9793 | 0.5026 |
| 0.01 | 42 | 0.01099 | 1.9814 | 0.5016 |
| 0.01 | 99 | 0.01099 | 1.9802 | 0.4995 |

## Datos crudos

- `raw.csv` — una fila por (lr, seed, fold) con todas las metricas + ‖w‖.
- `per_seed.csv` — agregado por (lr, seed).
- `summary.csv` — agregado por lr.