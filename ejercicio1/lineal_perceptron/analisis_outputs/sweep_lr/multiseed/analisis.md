# Sweep LR multi-seed — perceptrón linear

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
| 1e-05 | 0.02622 ± 0.00119 | 0.8390 ± 0.0062 | 0.4175 ± 0.0099 | 0.9839 ± 0.0091 | 0.5862 ± 0.0105 | 0.1945 ± 0.0008 |
| 0.0001 | 0.02651 ± 0.00139 | 0.8427 ± 0.0062 | 0.4237 ± 0.0102 | 0.9883 ± 0.0065 | 0.5930 ± 0.0104 | 0.2027 ± 0.0010 |
| 0.001 | 0.04599 ± 0.00616 | 0.8198 ± 0.0097 | 0.3918 ± 0.0129 | 1.0000 ± 0.0000 | 0.5629 ± 0.0133 | 0.2810 ± 0.0027 |

**Dispersión entre seeds** (std de los promedios-por-seed; cada promedio-por-seed es media sobre los 5 folds):

| lr | MSE test seed-std | Acc seed-std | Prec seed-std | Rec seed-std | F1 seed-std | ‖w‖ seed-std |
|---|---|---|---|---|---|---|
| 1e-05 | 0.00002 | 0.0006 | 0.0010 | 0.0012 | 0.0010 | 0.0001 |
| 0.0001 | 0.00006 | 0.0009 | 0.0016 | 0.0015 | 0.0017 | 0.0006 |
| 0.001 | 0.00085 | 0.0031 | 0.0041 | 0.0000 | 0.0042 | 0.0025 |

## Per-seed (cada celda = media sobre los 5 folds del CV)

| lr | seed | MSE test | Accuracy | Precision | Recall | F1 | ‖w‖ |
|---|---|---|---|---|---|---|---|
| 1e-05 | 7 | 0.02621 | 0.8383 | 0.4164 | 0.9851 | 0.5853 | 0.1945 |
| 1e-05 | 13 | 0.02622 | 0.8396 | 0.4189 | 0.9850 | 0.5877 | 0.1946 |
| 1e-05 | 21 | 0.02619 | 0.8396 | 0.4183 | 0.9827 | 0.5868 | 0.1945 |
| 1e-05 | 42 | 0.02625 | 0.8385 | 0.4168 | 0.9839 | 0.5855 | 0.1946 |
| 1e-05 | 99 | 0.02621 | 0.8389 | 0.4173 | 0.9827 | 0.5858 | 0.1946 |
| 0.0001 | 7 | 0.02647 | 0.8425 | 0.4231 | 0.9873 | 0.5923 | 0.2021 |
| 0.0001 | 13 | 0.02654 | 0.8440 | 0.4261 | 0.9908 | 0.5958 | 0.2029 |
| 0.0001 | 21 | 0.02644 | 0.8431 | 0.4240 | 0.9873 | 0.5932 | 0.2021 |
| 0.0001 | 42 | 0.02658 | 0.8427 | 0.4235 | 0.9873 | 0.5927 | 0.2032 |
| 0.0001 | 99 | 0.02651 | 0.8415 | 0.4216 | 0.9885 | 0.5911 | 0.2033 |
| 0.001 | 7 | 0.04502 | 0.8199 | 0.3916 | 1.0000 | 0.5628 | 0.2782 |
| 0.001 | 13 | 0.04608 | 0.8247 | 0.3984 | 1.0000 | 0.5697 | 0.2805 |
| 0.001 | 21 | 0.04528 | 0.8203 | 0.3924 | 1.0000 | 0.5635 | 0.2793 |
| 0.001 | 42 | 0.04713 | 0.8172 | 0.3882 | 1.0000 | 0.5592 | 0.2831 |
| 0.001 | 99 | 0.04642 | 0.8172 | 0.3884 | 1.0000 | 0.5594 | 0.2841 |

## Datos crudos

- `raw.csv` — una fila por (lr, seed, fold) con todas las métricas + ‖w‖.
- `per_seed.csv` — agregado por (lr, seed) (mean sobre los 5 folds).
- `summary.csv` — agregado por lr (mean/std sobre los 25 (seed, fold), y seed-std sobre los 5 promedios-por-seed).