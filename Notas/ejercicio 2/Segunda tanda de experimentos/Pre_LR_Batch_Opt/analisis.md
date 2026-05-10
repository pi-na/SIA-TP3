# Pre-experimento: LR × Batch × Optimizer

## Motivación

Todos los sweeps anteriores del Ej2 ([Arch](Arquitectura.md), [LR](analisis_lr.md), [Optimizer](analisis_optimizer.md)) usaron `batch_size=32` por default, sin justificarlo y sin medir si era óptimo. La regla teórica de la cátedra (clase de optimizadores) predice que **el LR óptimo escala con el batch_size** ("doblar el batch ≈ doblar el LR"), así que dejar batch fijo en 32 mientras barremos LR significa que para algunas combinaciones (opt, LR) estábamos midiendo configuraciones lejos del óptimo real.

**El problema concreto:** si en el [cross-experiment principal](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md) corremos `Adam@5e-3` con batch=32, podríamos concluir falsamente que "Adam@5e-3 funciona peor que Adam@1e-3" cuando en realidad el problema es batch demasiado chico para ese LR.

**Dos opciones para resolverlo:**
1. **Meter `batch_size` como cuarto factor en el cross-experiment** → grid 4D (LR×Opt×Arch×Batch). Multiplica el costo por ~5 (5 valores de batch). Inviable en el budget.
2. **Decidir `batch_size` óptimo por (opt, LR) en un pre-experimento chico** y heredarlo al grid principal. Costo marginal: ~1h vs los ~5h del grid principal.

**Decisión:** opción 2 — este pre-experimento. **Objetivo:** decidir el `batch_size` óptimo por (optimizer, learning rate) para usar como hiperparámetro heredado en el grid principal del [Cross_LR_Opt_Arch](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md).

**Diseño:** grid 3D pequeño 3 LR × 3 batch × 3 opt sobre `arch_shallow` (la "óptima" del Arch sweep, asumida estable como punto de anclaje), con 2 seeds × 5 folds = 10 corridas/cell. SEM ≈ 0.0019 — suficiente para decidir batch (no para reportar diferencias finas, eso lo hace el grid principal).

**Selección de niveles** (justificación en el [PLAN](PLAN_cross_v1.md)):
- LRs = {5e-4, 1e-3, 5e-3}: cubren la zona "óptima" de los 3 optimizadores según el optimizer sweep anterior. Excluimos 1e-4 (sub-entrenado en SGD) y 1e-2 (diverge en Adam) — no querríamos heredar sus batches a celdas centrales.
- Batches = {16, 64, 256}: span 16×, suficiente para detectar la regla de escalado lineal LR↔batch del curso.

**Caveat asumido:** el batch óptimo no varía bruscamente entre arquitecturas. Si lo hiciera, esta herencia sería inválida. Se decidió aceptar el riesgo a cambio del ahorro de cómputo, y declararlo como limitación.

## Configuración completa
Parámetros **explícitos** de la corrida (todos los que no se varían son fijos):
| Parámetro | Valor |
|---|---|
| Arquitectura | `arch_shallow` = `[784, 128, 10]`, activations `[relu, softmax]`, init `auto` (He) |
| Loss | `cross_entropy` |
| Preprocessing | `zscore`, `one_hot_targets=true` |
| Split | k-folds=5 estratificado, val_fraction_if_k1=0.2 |
| Regularización | l2=0, dropout=0, sin lr_schedule, sin augmentation |
| Early stopping | patience=20 sobre val_loss (CE), restaura best_weights al cortar |
| Seeds | [42, 7] |
| Optimizers | sgd, momentum (β=0.9), adam (β1=0.9, β2=0.999, ε=1e-8) — defaults del módulo |
Factores variados:
- LR: ['5e-4', '1e-3', '5e-3']
- Batch size: [16, 64, 256]
- Optimizer: sgd, momentum, adam

`max_epochs` por (opt, LR) (auditado previamente):

| optimizer | LR=5e-4 | 1e-3 | 5e-3 |
|---|---|---|---|
| sgd | 300 | 200 | 100 |
| momentum | 150 | 80 | 40 |
| adam | 40 | 40 | 30 |

Total cells: 3 LR × 3 batch × 3 opt = 27. Con 2 seeds = **54 jobs × 5 folds = 270 corridas**.

## Resultados crudos — val_acc media ± std (sobre 2 seeds × 5 folds = 10 corridas)

| opt      | LR   | batch | val_acc         | macro_f1        | val_loss CE |
| -------- | ---- | ----- | --------------- | --------------- | ----------- |
| sgd      | 5e-4 | 16    | 0.9481 ± 0.0052 | 0.8416 ± 0.0089 | 0.1962      |
| sgd      | 5e-4 | 64    | 0.9355 ± 0.0039 | 0.8283 ± 0.0070 | 0.2371      |
| sgd      | 5e-4 | 256   | 0.9025 ± 0.0032 | 0.7800 ± 0.0076 | 0.3533      |
| sgd      | 1e-3 | 16    | 0.9495 ± 0.0057 | 0.8433 ± 0.0095 | 0.1946      |
| sgd      | 1e-3 | 64    | 0.9394 ± 0.0044 | 0.8327 ± 0.0084 | 0.2237      |
| sgd      | 1e-3 | 256   | 0.9122 ± 0.0040 | 0.7961 ± 0.0071 | 0.3189      |
| sgd      | 5e-3 | 16    | 0.9498 ± 0.0060 | 0.8435 ± 0.0096 | 0.1920      |
| sgd      | 5e-3 | 64    | 0.9473 ± 0.0056 | 0.8410 ± 0.0093 | 0.1980      |
| sgd      | 5e-3 | 256   | 0.9322 ± 0.0045 | 0.8238 ± 0.0078 | 0.2469      |
| momentum | 5e-4 | 16    | 0.9498 ± 0.0058 | 0.8435 ± 0.0093 | 0.1923      |
| momentum | 5e-4 | 64    | 0.9491 ± 0.0057 | 0.8430 ± 0.0094 | 0.1947      |
| momentum | 5e-4 | 256   | 0.9387 ± 0.0046 | 0.8319 ± 0.0083 | 0.2267      |
| momentum | 1e-3 | 16    | 0.9503 ± 0.0060 | 0.8441 ± 0.0096 | 0.1906      |
| momentum | 1e-3 | 64    | 0.9490 ± 0.0056 | 0.8428 ± 0.0093 | 0.1939      |
| momentum | 1e-3 | 256   | 0.9393 ± 0.0044 | 0.8325 ± 0.0085 | 0.2240      |
| momentum | 5e-3 | 16    | 0.9533 ± 0.0059 | 0.8476 ± 0.0091 | 0.1971      |
| momentum | 5e-3 | 64    | 0.9507 ± 0.0053 | 0.8447 ± 0.0085 | 0.1901      |
| momentum | 5e-3 | 256   | 0.9473 ± 0.0055 | 0.8410 ± 0.0088 | 0.1971      |
| adam     | 5e-4 | 16    | 0.9566 ± 0.0050 | 0.8517 ± 0.0083 | 0.1693      |
| adam     | 5e-4 | 64    | 0.9551 ± 0.0052 | 0.8496 ± 0.0079 | 0.1704      |
| adam     | 5e-4 | 256   | 0.9536 ± 0.0055 | 0.8480 ± 0.0089 | 0.1761      |
| adam     | 1e-3 | 16    | 0.9541 ± 0.0043 | 0.8474 ± 0.0074 | 0.1792      |
| adam     | 1e-3 | 64    | 0.9563 ± 0.0042 | 0.8506 ± 0.0074 | 0.1697      |
| adam     | 1e-3 | 256   | 0.9538 ± 0.0049 | 0.8478 ± 0.0084 | 0.1746      |
| adam     | 5e-3 | 16    | 0.9344 ± 0.0090 | 0.8310 ± 0.0108 | 0.5502      |
| adam     | 5e-3 | 64    | 0.9431 ± 0.0099 | 0.8369 ± 0.0110 | 0.2578      |
| adam     | 5e-3 | 256   | 0.9541 ± 0.0053 | 0.8490 ± 0.0082 | 0.1909      |

## Decisión: best `batch_size` por (opt, LR)

Criterio: máxima val_acc media sobre 10 corridas.

| optimizer | LR=5e-4 | 1e-3 | 5e-3 |
|---|---|---|---|
| sgd | 16 | 16 | 16 |
| momentum | 16 | 16 | 16 |
| adam | 16 | 64 | 256 |

*Para LR fuera del set de etapa 1 (1e-4, 1e-2), el grid principal hereda el batch del LR más cercano dentro del mismo optimizer.*

## Plots

![Heatmap val_acc por (lr, batch) y opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/stage1_heatmap_val_acc.png)

![Curvas val_acc vs batch por LR y opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/stage1_val_acc_vs_batch.png)

## Limitaciones

- 2 seeds × 5 folds = 10 corridas: SEM ≈ 0.0019 (suficiente para decidir batch, no para reportar diferencias finas).
- Hecho **sólo sobre `arch_shallow`**. Asumimos que el batch óptimo no depende fuertemente de la arquitectura para hereparlo en el grid principal.
- Sólo 3 LR; los LRs extremos (1e-4, 1e-2) no se midieron acá y heredan del LR más cercano.
- No exploramos batches < 16 ni > 256.
