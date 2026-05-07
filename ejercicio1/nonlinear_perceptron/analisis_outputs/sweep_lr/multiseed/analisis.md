# Sweep LR multi-seed — perceptrón nonlinear

Experimento: 5 seeds × 3 LRs × 5 folds = 75 entrenamientos. `epochs=500` (suficiente para convergencia segun single-seed). Seeds: [7, 13, 21, 42, 99].

**Métricas reportadas** (regla del repo: error apropiado a la loss + Acc/Prec/Rec/F1):

- **MSE test**: error de la loss usada para entrenar (objetivo de la knowledge distillation contra `big_model_fraud_probability`).
- **Accuracy / Precision / Recall / F1**: clasificación binaria contra `flagged_fraud`, con threshold de decisión = 0.5 sobre la salida del perceptrón.
- **‖w‖**: norma L2 del vector de pesos final (slide *L2 Penalty Norm / Weight Decay* de la clase de regularización), reportada como diagnóstico de capacidad efectiva, no como término de loss.

**Convención de promedios** (regla del repo): cada celda aclara qué se promedia y sobre qué eje. `mean ± std` total = sobre las 25 corridas (5 seeds × 5 folds). `seed-std` = std de los promedios por seed (cada uno ya promediado sobre 5 folds), aislando la dispersión inter-seed.

![Dispersion](dispersion.png)

## Resumen agregado por LR — todas las métricas

**Total (mean ± std sobre 5 seeds × 5 folds = 25 corridas):**

| lr | MSE test | Accuracy | Precision | Recall | F1 | ‖w‖ |
|---|---|---|---|---|---|---|
| 0.0001 | 0.01128 ± 0.00052 | 0.7769 ± 0.0100 | 0.3421 ± 0.0101 | 1.0000 ± 0.0000 | 0.5097 ± 0.0112 | 1.7510 ± 0.0096 |
| 0.001 | 0.01099 ± 0.00058 | 0.7677 ± 0.0101 | 0.3330 ± 0.0096 | 1.0000 ± 0.0000 | 0.4996 ± 0.0108 | 1.9829 ± 0.0196 |
| 0.01 | 0.01099 ± 0.00058 | 0.7690 ± 0.0107 | 0.3344 ± 0.0102 | 1.0000 ± 0.0000 | 0.5011 ± 0.0115 | 1.9792 ± 0.0195 |

**Dispersión entre seeds** (std de los promedios-por-seed; cada promedio-por-seed es media sobre los 5 folds):

| lr | MSE test seed-std | Acc seed-std | Prec seed-std | Rec seed-std | F1 seed-std | ‖w‖ seed-std |
|---|---|---|---|---|---|---|
| 0.0001 | 0.00000 | 0.0007 | 0.0008 | 0.0000 | 0.0008 | 0.0013 |
| 0.001 | 0.00000 | 0.0001 | 0.0002 | 0.0000 | 0.0001 | 0.0003 |
| 0.01 | 0.00000 | 0.0011 | 0.0011 | 0.0000 | 0.0012 | 0.0018 |

## Per-seed (cada celda = media sobre los 5 folds del CV)

| lr | seed | MSE test | Accuracy | Precision | Recall | F1 | ‖w‖ |
|---|---|---|---|---|---|---|---|
| 0.0001 | 7 | 0.01128 | 0.7759 | 0.3409 | 1.0000 | 0.5084 | 1.7528 |
| 0.0001 | 13 | 0.01128 | 0.7775 | 0.3429 | 1.0000 | 0.5105 | 1.7517 |
| 0.0001 | 21 | 0.01128 | 0.7765 | 0.3419 | 1.0000 | 0.5094 | 1.7498 |
| 0.0001 | 42 | 0.01129 | 0.7775 | 0.3425 | 1.0000 | 0.5102 | 1.7496 |
| 0.0001 | 99 | 0.01128 | 0.7769 | 0.3422 | 1.0000 | 0.5098 | 1.7512 |
| 0.001 | 7 | 0.01098 | 0.7677 | 0.3329 | 1.0000 | 0.4995 | 1.9825 |
| 0.001 | 13 | 0.01099 | 0.7677 | 0.3331 | 1.0000 | 0.4996 | 1.9830 |
| 0.001 | 21 | 0.01098 | 0.7677 | 0.3333 | 1.0000 | 0.4998 | 1.9832 |
| 0.001 | 42 | 0.01099 | 0.7676 | 0.3329 | 1.0000 | 0.4994 | 1.9831 |
| 0.001 | 99 | 0.01098 | 0.7676 | 0.3331 | 1.0000 | 0.4996 | 1.9829 |
| 0.01 | 7 | 0.01099 | 0.7684 | 0.3335 | 1.0000 | 0.5002 | 1.9768 |
| 0.01 | 13 | 0.01100 | 0.7693 | 0.3347 | 1.0000 | 0.5014 | 1.9783 |
| 0.01 | 21 | 0.01099 | 0.7703 | 0.3358 | 1.0000 | 0.5026 | 1.9793 |
| 0.01 | 42 | 0.01099 | 0.7696 | 0.3348 | 1.0000 | 0.5016 | 1.9814 |
| 0.01 | 99 | 0.01099 | 0.7675 | 0.3330 | 1.0000 | 0.4995 | 1.9802 |

## Datos crudos

- `raw.csv` — una fila por (lr, seed, fold) con todas las métricas + ‖w‖.
- `per_seed.csv` — agregado por (lr, seed) (mean sobre los 5 folds).
- `summary.csv` — agregado por lr (mean/std sobre los 25 (seed, fold), y seed-std sobre los 5 promedios-por-seed).