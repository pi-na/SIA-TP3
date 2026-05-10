# Cross-experimento: LR × Optimizer × Arquitectura

## Motivación

Las decisiones del Ej2 hasta este punto se tomaron one-at-a-time, cada una **condicionada al valor fijo de otro factor**:

- El [Arch sweep](Arquitectura.md) se hizo con `Adam@1e-3` fijo → no sabíamos si `arch_shallow` ganaba también con SGD/Momentum.
- El [Optimizer sweep](analisis_optimizer.md) se hizo con `arch_base` fijo → la decisión "Adam@1e-3" se tomó sobre la arquitectura que después descartamos.
- El [LR sweep](analisis_lr.md) se hizo con SGD only y arch_base, con 50 ep insuficientes para LRs bajos.
- `batch_size=32` nunca se exploró: todo se hizo con ese default.

**El problema:** si dos hiperparámetros interactúan (regla teórica del curso para LR×Batch, sospechado entre Arch×LR y Arch×Opt), la conclusión del sweep one-at-a-time **depende del valor que se eligió como fijo**. No es un resultado robusto.

**Objetivo:** validar la elección de hiperparámetros **testeando el supuesto de independencia** entre LR, optimizer y arquitectura simultáneamente, y reportar la mejor configuración global junto con las interacciones explícitas.

**Diseño elegido:** grid 3D **LR×Opt×Arch** con `batch_size` heredado de un pre-experimento dedicado ([Pre_LR_Batch_Opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md)) + estrella batch alrededor del centro. Plan completo en [`PLAN_cross_v1.md`](PLAN%20de%20todos%20los%20experimentos%20cruzados%20cross_v1.md).

**Por qué un grid 3D y no 2D:** las "slices 2D" (ej. Arch×Opt con LR fijo) **mienten cuando los factores no medidos están correlacionados** con los medidos — al fijar LR=1e-3 (óptimo de Adam), SGD entraría con LR sub-óptimo y la slice concluiría falsamente "Adam le gana a SGD en todas las archs". Discusión metodológica completa en [`IMPORTANTE_CORRELACIONES.md`](IMPORTANTE_CORRELACIONES.md).

## Configuración completa

**Hiperparams fijos en TODAS las celdas:**

| Parámetro             | Valor                                                                   |
| --------------------- | ----------------------------------------------------------------------- |
| Loss                  | `cross_entropy`                                                         |
| Preprocessing         | `zscore`, `one_hot_targets=true`                                        |
| Split                 | k-folds=5 estratificado                                                 |
| Regularización        | l2=0, dropout=0, sin lr_schedule, sin augmentation                      |
| Early stopping        | patience=20 sobre val_loss (CE), restaura best_weights al cortar        |
| Inicialización        | `auto` (He para ReLU; Xavier para tanh/sigmoid)                         |
| Output                | softmax + cross_entropy combinados (regla de la cátedra)                |
| Seeds (stage 2 main)  | [42, 7, 13]                                                             |
| Seeds (stage 2b)      | [42, 7, 13]                                                             |
| Optimizer hyperparams | sgd: solo lr · momentum: lr, β=0.9 · adam: lr, β1=0.9, β2=0.999, ε=1e-8 |

**Factores variados (stage 2 main):**

- LR: ['1e-4', '5e-4', '1e-3', '5e-3', '1e-2']
- Optimizer: ['sgd', 'momentum', 'adam']
- Arquitectura: ['arch_shallow', 'arch_base', 'arch_wider', 'arch_deeper']
- Batch size: heredado del pre-experimento (Pre_LR_Batch_Opt) por (opt, LR)

**`batch_size` por celda (resultado del pre-experimento):**

| optimizer | LR=1e-4 (heredado) | 5e-4 | 1e-3 | 5e-3 | 1e-2 (heredado) |
|---|---|---|---|---|---|
| sgd | 16 | 16 | 16 | 16 | 16 |
| momentum | 16 | 16 | 16 | 16 | 16 |
| adam | 16 | 16 | 64 | 256 | 256 |

**`max_epochs` por (opt, LR) (con ES patience=20, el corte real lo decide la curva):**

| optimizer | 1e-4 | 5e-4 | 1e-3 | 5e-3 | 1e-2 |
|---|---|---|---|---|---|
| sgd | 200 | 300 | 200 | 100 | 80 |
| momentum | 250 | 150 | 80 | 40 | 40 |
| adam | 60 | 40 | 40 | 30 | 30 |

*Nota: `SGD@1e-4` capeado a 200 ep — auditoría previa estableció que no converge dentro de presupuesto razonable; se reporta como referencia de 'LR demasiado bajo' para SGD.*

**Arquitecturas comparadas:**

| arch | layer_sizes | hidden layers | params aprox. |
|---|---|---|---|
| arch_shallow | [784, 128, 10] | 1 | ~101k |
| arch_base    | [784, 128, 64, 10] | 2 | ~109k |
| arch_wider   | [784, 256, 128, 10] | 2 | ~235k |
| arch_deeper  | [784, 128, 64, 32, 10] | 3 | ~111k |

Total stage 2 main: 5 × 3 × 4 = 60 cells × 3 seeds = **180 jobs × 5 folds = 900 corridas**.

## Resultados — val_acc media ± std (sobre 3 seeds × 5 folds = 15 corridas)

### Qué significa cada columna

Antes de leer las tablas, qué representa exactamente cada columna y cómo se calcula:

| Columna           | Qué es                                                                                                                                              | Cómo se calcula                                                                                                                                                                                                                                                                                                                                                                                                                             | Por qué la reportamos                                                                                                                                                                                       |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **arch**          | Identificador de la arquitectura. Una de `arch_shallow`, `arch_base`, `arch_wider`, `arch_deeper`.                                                  | (no se calcula, es un nombre)                                                                                                                                                                                                                                                                                                                                                                                                               | Factor variado del experimento. Ver [diagramas](../../visualizacion%20arquitecturas/Arquitecturas.md).                                                                                                      |
| **opt**           | Optimizer usado. Uno de `sgd`, `momentum`, `adam`.                                                                                                  | (factor)                                                                                                                                                                                                                                                                                                                                                                                                                                    | Factor variado. Ver [Optimizer sweep previo](../../Primera%20tanda%20de%20experimentos/Optimizer/analisis_optimizer.md).                                                                                    |
| **LR**            | Learning rate. Uno de {1e-4, 5e-4, 1e-3, 5e-3, 1e-2}.                                                                                               | (factor)                                                                                                                                                                                                                                                                                                                                                                                                                                    | Factor variado.                                                                                                                                                                                             |
| **val_acc**       | Validation accuracy: fracción de imágenes del fold de validación que el modelo clasifica bien. Reportado como `media ± std`.                        | Para cada corrida (1 seed, 1 fold): `val_acc = mean(argmax(softmax(forward(X_val))) == y_val)`. Después se agrega: `media` y `std` sobre las **15 corridas** de la celda (3 seeds × 5 folds). Importante: gracias al [fix #4](../../Experimentos%20y%20analisis/Todos%20los%20experimentos.md), siempre se evalúa con los `best_weights` (los del epoch que minimizó val_loss durante el entrenamiento), no con los pesos del último epoch. | Métrica principal de ranking. Ver [Justificación de métricas](../Justificacion_metricas.md).                                                                                                                |
| **macro_f1**      | F1 macro: media aritmética del F1 por clase, sin pesar por frecuencia. Reportado como `media ± std`.                                                | Para cada corrida: por clase c se calcula `f1_c = 2·P_c·R_c/(P_c+R_c)` con `P_c = TP_c/(TP_c+FP_c)`, `R_c = TP_c/(TP_c+FN_c)`. Macro promedia los 10 `f1_c` con peso igual (no por frecuencia). Después media y std sobre 15 corridas.                                                                                                                                                                                                      | Métrica de control: detecta si el modelo es injusto con la clase 5 (la minoritaria, 271 ejemplos). Ver [Justificación](../Justificacion_metricas.md).                                                       |
| **val_loss CE**   | Cross-entropy promedio sobre el fold de validación, evaluada con `best_weights`. Es la **misma loss que el modelo minimiza** durante entrenamiento. | `val_loss = -mean(sum(y_true_onehot * log(softmax(forward(X_val))), axis=1))`. Reportada como media sobre las 15 corridas (sin std en la tabla porque ya está implícita en val_acc/F1).                                                                                                                                                                                                                                                     | Mide qué tan **calibradas** están las probabilidades, no sólo si el argmax acierta. Dos modelos pueden tener igual val_acc pero distinta val_loss → el de menor val_loss tiene predicciones más confiables. |
| **train_loss CE** | Cross-entropy sobre el fold de **entrenamiento**, también con `best_weights`.                                                                       | Igual que val_loss CE pero sobre `X_train`, `y_train`.                                                                                                                                                                                                                                                                                                                                                                                      | Sirve para detectar **sobreajuste**: gap = train_loss CE − val_loss CE. Si `train_loss → 0` mientras `val_loss` sube, el modelo memoriza.                                                                   |
| **best_epoch**    | Época en la que se alcanzó el mínimo de `val_loss` durante el entrenamiento.                                                                        | `best_epoch = argmin(val_loss_per_epoch)` sobre la curva de épocas de cada corrida. Reportado como media sobre 15 corridas.                                                                                                                                                                                                                                                                                                                 | Indica qué tan rápido converge la combinación. Para Adam@1e-3 ≈ 3-5 ép; para SGD@1e-4 alcanza el techo de `max_epochs` (no convergió).                                                                      |

> **Sobre `± std`:** todos los `±` reportados son **desvíos estándar sobre las 15 corridas** (3 seeds × 5 folds). El **error estándar de la media (SEM)** = `std / √15 ≈ std / 3.87`. Para distinguir dos celdas estadísticamente al 95% se necesita que su diferencia sea > `1.96 · √(SEM₁² + SEM₂²)`. Si dos cells tienen `val_acc` con SEM ≈ 0.001, sólo distinguen diffs ≥ 0.003.

> **Por qué `val_loss` y `train_loss` no llevan `±`:** sólo se reporta la media para no inflar la tabla. El comportamiento de la varianza es similar al de val_acc/F1.

### Tabla de resultados completa

| arch         | opt      | LR   | val_acc         | macro_f1        | val_loss CE | train_loss CE | best_epoch |
| ------------ | -------- | ---- | --------------- | --------------- | ----------- | ------------- | ---------- |
| arch_shallow | sgd      | 1e-4 | 0.9244 ± 0.0052 | 0.8131 ± 0.0073 | 0.2798      | 0.2084        | 199.0      |
| arch_shallow | sgd      | 5e-4 | 0.9487 ± 0.0046 | 0.8423 ± 0.0073 | 0.1997      | 0.0431        | 296.9      |
| arch_shallow | sgd      | 1e-3 | 0.9501 ± 0.0049 | 0.8439 ± 0.0077 | 0.1981      | 0.0317        | 187.4      |
| arch_shallow | sgd      | 5e-3 | 0.9505 ± 0.0051 | 0.8442 ± 0.0078 | 0.1957      | 0.0311        | 37.8       |
| arch_shallow | sgd      | 1e-2 | 0.9509 ± 0.0053 | 0.8444 ± 0.0081 | 0.1942      | 0.0311        | 18.6       |
| arch_shallow | momentum | 1e-4 | 0.9500 ± 0.0050 | 0.8437 ± 0.0076 | 0.1979      | 0.0302        | 197.8      |
| arch_shallow | momentum | 5e-4 | 0.9505 ± 0.0050 | 0.8443 ± 0.0076 | 0.1959      | 0.0302        | 39.1       |
| arch_shallow | momentum | 1e-3 | 0.9508 ± 0.0051 | 0.8445 ± 0.0078 | 0.1941      | 0.0301        | 18.9       |
| arch_shallow | momentum | 5e-3 | 0.9526 ± 0.0053 | 0.8466 ± 0.0078 | 0.2000      | 0.0268        | 4.2        |
| arch_shallow | momentum | 1e-2 | 0.9543 ± 0.0062 | 0.8499 ± 0.0081 | 0.2341      | 0.0196        | 4.7        |
| arch_shallow | adam     | 1e-4 | 0.9546 ± 0.0044 | 0.8491 ± 0.0073 | 0.1746      | 0.0252        | 15.9       |
| arch_shallow | adam     | 5e-4 | 0.9567 ± 0.0049 | 0.8518 ± 0.0075 | 0.1695      | 0.0203        | 4.8        |
| arch_shallow | adam     | 1e-3 | 0.9572 ± 0.0041 | 0.8521 ± 0.0067 | 0.1701      | 0.0180        | 5.7        |
| arch_shallow | adam     | 5e-3 | 0.9546 ± 0.0046 | 0.8493 ± 0.0069 | 0.1926      | 0.0242        | 3.2        |
| arch_shallow | adam     | 1e-2 | 0.9472 ± 0.0063 | 0.8421 ± 0.0087 | 0.2493      | 0.0531        | 2.1        |
| arch_base    | sgd      | 1e-4 | 0.9263 ± 0.0034 | 0.8144 ± 0.0054 | 0.2684      | 0.1756        | 199.0      |
| arch_base    | sgd      | 5e-4 | 0.9461 ± 0.0038 | 0.8393 ± 0.0051 | 0.2136      | 0.0399        | 185.9      |
| arch_base    | sgd      | 1e-3 | 0.9463 ± 0.0036 | 0.8395 ± 0.0050 | 0.2130      | 0.0395        | 92.7       |
| arch_base    | sgd      | 5e-3 | 0.9466 ± 0.0037 | 0.8398 ± 0.0055 | 0.2090      | 0.0397        | 18.0       |
| arch_base    | sgd      | 1e-2 | 0.9476 ± 0.0047 | 0.8407 ± 0.0062 | 0.2061      | 0.0379        | 9.1        |
| arch_base    | momentum | 1e-4 | 0.9464 ± 0.0037 | 0.8396 ± 0.0052 | 0.2130      | 0.0398        | 92.3       |
| arch_base    | momentum | 5e-4 | 0.9465 ± 0.0038 | 0.8394 ± 0.0055 | 0.2084      | 0.0405        | 17.6       |
| arch_base    | momentum | 1e-3 | 0.9478 ± 0.0042 | 0.8416 ± 0.0059 | 0.2054      | 0.0353        | 9.6        |
| arch_base    | momentum | 5e-3 | 0.9506 ± 0.0039 | 0.8437 ± 0.0063 | 0.2121      | 0.0292        | 3.0        |
| arch_base    | momentum | 1e-2 | 0.9467 ± 0.0077 | 0.8408 ± 0.0100 | 0.2278      | 0.0574        | 2.4        |
| arch_base    | adam     | 1e-4 | 0.9528 ± 0.0029 | 0.8469 ± 0.0044 | 0.1848      | 0.0280        | 10.5       |
| arch_base    | adam     | 5e-4 | 0.9541 ± 0.0050 | 0.8483 ± 0.0072 | 0.1741      | 0.0266        | 2.9        |
| arch_base    | adam     | 1e-3 | 0.9548 ± 0.0044 | 0.8493 ± 0.0050 | 0.1751      | 0.0229        | 3.5        |
| arch_base    | adam     | 5e-3 | 0.9533 ± 0.0058 | 0.8472 ± 0.0077 | 0.1889      | 0.0303        | 2.5        |
| arch_base    | adam     | 1e-2 | 0.9465 ± 0.0078 | 0.8406 ± 0.0074 | 0.2153      | 0.0643        | 1.6        |
| arch_wider   | sgd      | 1e-4 | 0.9293 ± 0.0048 | 0.8195 ± 0.0066 | 0.2610      | 0.1594        | 199.0      |
| arch_wider   | sgd      | 5e-4 | 0.9483 ± 0.0056 | 0.8426 ± 0.0074 | 0.2097      | 0.0321        | 183.1      |
| arch_wider   | sgd      | 1e-3 | 0.9482 ± 0.0056 | 0.8426 ± 0.0073 | 0.2091      | 0.0327        | 90.3       |
| arch_wider   | sgd      | 5e-3 | 0.9488 ± 0.0053 | 0.8435 ± 0.0070 | 0.2056      | 0.0313        | 17.7       |
| arch_wider   | sgd      | 1e-2 | 0.9498 ± 0.0053 | 0.8445 ± 0.0072 | 0.2019      | 0.0299        | 8.9        |
| arch_wider   | momentum | 1e-4 | 0.9480 ± 0.0054 | 0.8424 ± 0.0072 | 0.2092      | 0.0331        | 89.6       |
| arch_wider   | momentum | 5e-4 | 0.9491 ± 0.0050 | 0.8436 ± 0.0066 | 0.2057      | 0.0320        | 17.7       |
| arch_wider   | momentum | 1e-3 | 0.9500 ± 0.0051 | 0.8448 ± 0.0066 | 0.2018      | 0.0299        | 8.9        |
| arch_wider   | momentum | 5e-3 | 0.9531 ± 0.0054 | 0.8481 ± 0.0083 | 0.2026      | 0.0216        | 2.9        |
| arch_wider   | momentum | 1e-2 | 0.9540 ± 0.0061 | 0.8478 ± 0.0079 | 0.2191      | 0.0276        | 3.2        |
| arch_wider   | adam     | 1e-4 | 0.9547 ± 0.0047 | 0.8500 ± 0.0060 | 0.1761      | 0.0187        | 7.9        |
| arch_wider   | adam     | 5e-4 | 0.9553 ± 0.0050 | 0.8503 ± 0.0071 | 0.1775      | 0.0318        | 2.2        |
| arch_wider   | adam     | 1e-3 | 0.9583 ± 0.0036 | 0.8537 ± 0.0050 | 0.1701      | 0.0148        | 3.4        |
| arch_wider   | adam     | 5e-3 | 0.9531 ± 0.0048 | 0.8478 ± 0.0060 | 0.1917      | 0.0384        | 2.0        |
| arch_wider   | adam     | 1e-2 | 0.9450 ± 0.0054 | 0.8383 ± 0.0075 | 0.2283      | 0.0840        | 1.1        |
| arch_deeper  | sgd      | 1e-4 | 0.9286 ± 0.0042 | 0.8157 ± 0.0062 | 0.2656      | 0.1516        | 199.0      |
| arch_deeper  | sgd      | 5e-4 | 0.9440 ± 0.0048 | 0.8361 ± 0.0066 | 0.2273      | 0.0434        | 123.1      |
| arch_deeper  | sgd      | 1e-3 | 0.9435 ± 0.0051 | 0.8356 ± 0.0069 | 0.2262      | 0.0473        | 58.7       |
| arch_deeper  | sgd      | 5e-3 | 0.9455 ± 0.0046 | 0.8379 ± 0.0056 | 0.2206      | 0.0423        | 12.0       |
| arch_deeper  | sgd      | 1e-2 | 0.9463 ± 0.0036 | 0.8394 ± 0.0053 | 0.2153      | 0.0419        | 5.9        |
| arch_deeper  | momentum | 1e-4 | 0.9434 ± 0.0048 | 0.8356 ± 0.0067 | 0.2261      | 0.0470        | 59.1       |
| arch_deeper  | momentum | 5e-4 | 0.9457 ± 0.0047 | 0.8385 ± 0.0061 | 0.2203      | 0.0418        | 12.3       |
| arch_deeper  | momentum | 1e-3 | 0.9464 ± 0.0041 | 0.8398 ± 0.0058 | 0.2164      | 0.0397        | 6.2        |
| arch_deeper  | momentum | 5e-3 | 0.9500 ± 0.0072 | 0.8429 ± 0.0101 | 0.2154      | 0.0376        | 2.8        |
| arch_deeper  | momentum | 1e-2 | 0.9425 ± 0.0055 | 0.8339 ± 0.0082 | 0.2425      | 0.0781        | 2.5        |
| arch_deeper  | adam     | 1e-4 | 0.9511 ± 0.0049 | 0.8449 ± 0.0066 | 0.1919      | 0.0261        | 9.5        |
| arch_deeper  | adam     | 5e-4 | 0.9521 ± 0.0061 | 0.8455 ± 0.0078 | 0.1847      | 0.0347        | 2.6        |
| arch_deeper  | adam     | 1e-3 | 0.9535 ± 0.0078 | 0.8476 ± 0.0097 | 0.1821      | 0.0274        | 3.3        |
| arch_deeper  | adam     | 5e-3 | 0.9513 ± 0.0050 | 0.8463 ± 0.0076 | 0.1948      | 0.0448        | 2.1        |
| arch_deeper  | adam     | 1e-2 | 0.9455 ± 0.0045 | 0.8376 ± 0.0063 | 0.2209      | 0.0711        | 1.9        |

## Top configs (ordenadas por val_acc)

| #   | arch         | opt      | LR   | val_acc         | macro_f1 | best_epoch |
| --- | ------------ | -------- | ---- | --------------- | -------- | ---------- |
| 1   | arch_wider   | adam     | 1e-3 | 0.9583 ± 0.0036 | 0.8537   | 3.4        |
| 2   | arch_shallow | adam     | 1e-3 | 0.9572 ± 0.0041 | 0.8521   | 5.7        |
| 3   | arch_shallow | adam     | 5e-4 | 0.9567 ± 0.0049 | 0.8518   | 4.8        |
| 4   | arch_wider   | adam     | 5e-4 | 0.9553 ± 0.0050 | 0.8503   | 2.2        |
| 5   | arch_base    | adam     | 1e-3 | 0.9548 ± 0.0044 | 0.8493   | 3.5        |
| 6   | arch_wider   | adam     | 1e-4 | 0.9547 ± 0.0047 | 0.8500   | 7.9        |
| 7   | arch_shallow | adam     | 1e-4 | 0.9546 ± 0.0044 | 0.8491   | 15.9       |
| 8   | arch_shallow | adam     | 5e-3 | 0.9546 ± 0.0046 | 0.8493   | 3.2        |
| 9   | arch_shallow | momentum | 1e-2 | 0.9543 ± 0.0062 | 0.8499   | 4.7        |
| 10  | arch_base    | adam     | 5e-4 | 0.9541 ± 0.0050 | 0.8483   | 2.9        |

## Stage 2b — Estrella batch alrededor del centro

Centro: `arch_shallow` + `adam` + LR=`1e-3`. Batches probados: [16, 32, 64, 128, 256]. Seeds: [42, 7, 13]. k=5.

| batch | val_acc | macro_f1 | val_loss CE | best_epoch |
|---|---|---|---|---|
| 16 | 0.9540 ± 0.0036 | 0.8477 ± 0.0061 | 0.1785 | 2.3 |
| 32 | 0.9568 ± 0.0054 | 0.8510 ± 0.0079 | 0.1700 | 3.9 |
| 64 | 0.9572 ± 0.0041 | 0.8521 ± 0.0067 | 0.1701 | 5.7 |
| 128 | 0.9556 ± 0.0045 | 0.8502 ± 0.0069 | 0.1742 | 7.8 |
| 256 | 0.9548 ± 0.0045 | 0.8493 ± 0.0073 | 0.1771 | 11.2 |

## Plots

![val_acc vs LR por (arch, opt)](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/stage2_val_acc_vs_lr_per_opt.png)

![Heatmap val_acc por arch × LR para cada opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/stage2_heatmap_arch_lr.png)

![Convergencia val_loss por opt y LR (arch_shallow)](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/stage2_convergence_shallow.png)

![Estrella batch (centro: shallow + Adam@1e-3)](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/stage2b_val_acc_vs_batch.png)

## Configuración óptima encontrada

**`arch_wider` + `adam` + LR=`1e-3`**

- val_acc: 0.9583 ± 0.0036
- macro_f1: 0.8537 ± 0.0050
- val_loss: 0.1701
- best_epoch promedio: 3.4

## Limitaciones / caveats

- **`SGD@1e-4` capeado a 200 ep**: ya sabemos por la auditoría previa que no converge en presupuesto razonable; se incluye para mostrar la curva 'LR demasiado bajo' pero la celda **no está convergida**.
- **`batch_size` heredado del pre-experimento sobre `arch_shallow`**: asumimos que el batch óptimo no depende fuertemente de la arquitectura. Es una suposición razonable pero no medida — si el grid muestra que el ranking de archs cambia mucho entre opts, vale la pena rever esto.
- **3 seeds × 5 folds = 15 corridas/celda**: SEM ≈ 0.0016 sobre val_acc. Distingue diferencias ≥0.005 con confianza pero no menores.
- **Patience=20**: auditoría previa mostró que cubre con ~3× la subida transitoria máxima observada en los sweeps anteriores. Para `SGD@1e-4` (descenso lento, monótono) el patience no dispara espuriamente porque cada epoch mejora estrictamente.
- **No se varió L2/dropout/data augmentation**: este experimento no testea regularización, sólo optimización. La regularización se atacaría en un experimento siguiente sobre el centro encontrado acá.
- **No se varió la inicialización**: `auto` selecciona He para ReLU, sin variantes.
- **No se varió la activación**: ReLU + softmax fijo. La activación es factor del Pack B, fuera del scope.
- **batch_size en stage 2b**: medido SÓLO en el centro `shallow + Adam@1e-3`. Si quisiéramos certificar que el efecto del batch generaliza a otras celdas, haría falta un mini-grid 2D adicional batch×opt o batch×arch.
