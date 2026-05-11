# Ejercicio 3 — Plan y resultados

> **Estado:** corriendo (lanzado durante la noche). Las secciones de resultados se van completando a medida que terminan los pasos. El plan está congelado.

## Contexto del Ej3

CompanyX pidió **accuracy ≥ 98%** sobre `digits_test.csv`. En el Ej2 alcanzamos `test_acc = 0.853 ± 0.003` con la mejor configuración encontrada (`shallow + Adam@1e-3 + batch=64`, sin regularización). La caída desde el CV interno (0.957) se explicó por **la clase 8 ausente en `digits.csv`** (ver sección "Convergencia y generalización" del Ej2). El Ej3 hace dos cosas en orden:

1. **Sumar `more_digits.csv`** al training (que SÍ tiene 585 ejemplos de clase 8). Es la palanca de mayor impacto esperado.
2. **Explorar regularización** (L2 + gaussian noise) sobre el nuevo baseline, sin dropout ni LR schedule.

Las técnicas elegidas (L2 + augmentation gaussiana) son **las dos profundizadas explícitamente en la clase de regularización** (Marina Fuster, parte 1). Dropout sólo se menciona en la slide "existen otros más"; LR schedule no aparece en la clase. Por eso quedan fuera del grid.

## Estado del módulo `mlp/` (verificado antes de arrancar)

| Técnica | Implementada en `mlp/` | Cómo se activa en el config |
|---|---|---|
| Sumar `more_digits.csv` | ✅ `mlp/train.py:282` (concatena `extra_csv_paths`) | `dataset.extra_csv_paths = ["data and documentation/more_digits.csv"]` |
| L2 / weight decay | ✅ `mlp/network.py:168-174`, no penaliza bias | `regularization.l2 = <float>` |
| Augmentation gaussiana | ✅ `mlp/network.py:160-165` (ruido N(0, σ) por minibatch) | `regularization.augmentation = {"type":"gaussian_noise","sigma":<float>}` |
| Dropout | ✅ inverted (forward + backward correctos) | **no usado** (decisión del equipo) |
| LR schedule (step) | ✅ | **no usado** (no aparece en la clase) |

---

## Paso 1 — Baseline Ej3 (sumar `more_digits.csv`)

### Intención

Aislar el efecto de **más datos sin tocar regularización**. Comparar 1-a-1 con el "shallow + Adam@1e-3 + bs=64" del Ej2: misma arquitectura, mismos hiperparámetros, sólo cambia el dataset.

### Configuración (todo lo que queda fijo)

| Hiperparam | Valor | Heredado de |
|---|---|---|
| `arch.layer_sizes` | `[784, 128, 10]` (shallow) | Ej2 decisión Occam |
| `arch.activations` | `[relu, softmax]` | módulo |
| `arch.initializer` | `auto` (He) | Ej2 |
| `optimizer` | Adam, β₁=0.9, β₂=0.999, ε=1e-8 | Ej2 |
| `lr` | `1e-3` | Ej2 (LR×ARCH: óptimo de Adam en las 4 archs) |
| `batch_size` | `64` | Ej2 (stage 2b confirmó pico unimodal) |
| `loss` | `cross_entropy` (combinado con softmax en última capa) | clase cátedra |
| `preprocessing` | `zscore` + `one_hot_targets=true` | Ej2 |
| `split` (CV interno) | `k_folds=5`, estratificado | Ej2 |
| `early_stopping_patience` | 20 sobre val_loss CE | Ej2 |
| `max_epochs` | **50** (subido desde 40) | con más datos, best_epoch puede subir; margen extra |
| `regularization` | todo cero / null | es el baseline |
| **dataset principal** | `digits.csv` | — |
| **dataset extra** | `more_digits.csv` | **CAMBIO Ej3** |
| **Seeds** | `[42, 7, 13]` | Ej2 |

### Tamaños

| Set | N samples |
|---|---|
| `digits.csv` (Ej2 train) | 12 449 |
| + `more_digits.csv` (suma Ej3) | 12 449 + 15 741 = **28 190** |
| `digits_test.csv` (intocado) | 2 497 |

### Experimentos

- **A) Convergencia (CV interno)**: 3 seeds × 5 folds = **15 corridas** → `epoch_history.csv` agregado → plot train_loss/val_loss/train_acc/val_acc vs epoch con bandas ± std. Reportar `best_epoch` (mean ± std) y `stop_epoch`.
- **B) Generalización interna (CV)**: las 4 métricas (accuracy, macro_precision, macro_recall, macro_F1) + CE en train y val, promedio sobre las 15 corridas.
- **C) Generalización externa**: 3 corridas de `ejercicio3/final_eval.py` (full train con split 90/10 para ES + eval sobre `digits_test.csv`). Reportar mean ± std de las 4 métricas en test + matriz de confusión promedio + métricas per-clase.

### Outputs paso 1 (rutas donde quedan los archivos)

- **Configs**: `ejercicio3/configs/baseline_ej3.json` (CV), `ejercicio3/configs/final_config_ej3_baseline.json` (final_eval).
- **Raw output CV**: `ejercicio3/output/baseline/baseline_ej3_seed{42,7,13}/`.
- **Raw output test**: `ejercicio3/output/final_eval/baseline/final_eval_*_seed{42,7,13}_*/`.
- **Análisis agregado**: `ejercicio3/analisis/baseline/{cv_internal_summary.csv, test_summary.csv, test_per_class.csv, optimal_convergence.png, test_confusion_matrix.png, optimal_convergence_table.csv}`.
- **Scripts**: `ejercicio3/scripts/step1_baseline.py`, `ejercicio3/scripts/run_cv_paralelo.py`, `ejercicio3/scripts/analyze_baseline.py`.

### Resultados paso 1

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
 — Grid de regularización (L2 × σ)

### Intención

Sobre el baseline del paso 1 (que ya incluye `more_digits.csv`), medir cuánto suma cada técnica de regularización **explicada en la clase**. Excluimos dropout (decisión del equipo) y LR schedule (no aparece en la clase).

### Grid

| Factor | Niveles | Justificación |
|---|---|---|
| **L2** (weight decay) | `{0, 1e-5, 1e-4, 1e-3}` | Cubre desde "casi sin penalización" hasta "moderada" (1e-2 ya es demasiado para Adam@1e-3 según práctica común; lo evitamos para no estropear el optimizer) |
| **σ** (gaussian noise) | `{0, 0.03, 0.1, 0.2}` | Píxeles zscore-normalizados → σ se interpreta como "fracción del desvío estándar del pixel". 0.03 es ruido sutil; 0.2 es ruido fuerte. Más alto rompería el patrón visual. |

Grid 4 × 4 = **16 combinaciones**. La celda `(L2=0, σ=0)` coincide con el baseline del paso 1 → la re-corro acá igual para tener una tabla homogénea (3 seeds, mismo runner). Las otras 15 son nuevas.

**Total CV interno**: 16 combos × 3 seeds × 5 folds = **240 corridas**.

### Hiperparams fijos

Idénticos al paso 1. Lo único que varía entre celdas del grid es `regularization.l2` y `regularization.augmentation.sigma`.

### Análisis previsto

- **Tabla 16 filas** (l2, σ, val_acc±std, macro_F1±std, val_loss CE, train_loss CE, gap, best_epoch).
- **Heatmap 4×4 de val_acc** (filas=L2, columnas=σ) con std en cada celda.
- **Heatmap 4×4 de gap val−train** para visualizar dónde la regularización reduce el sobreajuste.
- **Identificar best combo** por val_acc media. Si hay empate estadístico (Δ < 0.003), Occam → menor regularización.
- **Best combo** → `final_eval.py` con 3 seeds sobre `digits_test.csv` → matriz de confusión + métricas test.
- **Comparación**: baseline paso 1 vs best combo paso 2 vs Ej2 (sin more_digits, sin reg).

### Outputs paso 2

- **Raw CV**: `ejercicio3/output/grid_reg/l2_<X>_sigma_<Y>_seed<Z>/`.
- **Final eval best**: `ejercicio3/output/final_eval/best_reg/`.
- **Análisis agregado**: `ejercicio3/analisis/grid_reg/{grid_summary.csv, val_acc_heatmap.png, gap_heatmap.png, best_combo_*.png}`.
- **Scripts**: `ejercicio3/scripts/step2_grid_reg.py`, `ejercicio3/scripts/analyze_grid_reg.py`.

### Resultados paso 2

### Resultados paso 2 — Grid de regularización (L2 × σ)

Grid 4×4 = 16 combinaciones × 3 seeds × 5 folds = **240 corridas CV**.

**Tabla agregada (15 corridas/combo, ordenada por val_acc):**

| L2 | σ | val_acc (±std) | macro_F1 (±std) | val_loss CE | train_loss | gap | best_epoch |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1e-3 | 0 | 0.9750 ± 0.0018 | 0.9644 ± 0.0035 | 0.0917 | 0.0168 | 0.0750 | 39.3 |
| 1e-3 | 0.1 | 0.9746 ± 0.0016 | 0.9632 ± 0.0034 | 0.0939 | 0.0180 | 0.0759 | 38.3 |
| 1e-4 | 0 | 0.9743 ± 0.0032 | 0.9630 ± 0.0042 | 0.1058 | 0.0054 | 0.1004 | 22.7 |
| 1e-3 | 0.2 | 0.9740 ± 0.0017 | 0.9631 ± 0.0034 | 0.0949 | 0.0189 | 0.0761 | 37.3 |
| 1e-3 | 0.03 | 0.9740 ± 0.0023 | 0.9622 ± 0.0039 | 0.0944 | 0.0183 | 0.0761 | 34.3 |
| 1e-4 | 0.03 | 0.9732 ± 0.0032 | 0.9615 ± 0.0047 | 0.1123 | 0.0081 | 0.1042 | 16.5 |
| 1e-4 | 0.2 | 0.9727 ± 0.0031 | 0.9605 ± 0.0050 | 0.1147 | 0.0085 | 0.1062 | 12.6 |
| 1e-4 | 0.1 | 0.9725 ± 0.0027 | 0.9605 ± 0.0039 | 0.1152 | 0.0073 | 0.1080 | 11.5 |
| 1e-5 | 0 | 0.9707 ± 0.0030 | 0.9584 ± 0.0047 | 0.1207 | 0.0092 | 0.1114 | 6.7 |
| 1e-5 | 0.03 | 0.9705 ± 0.0033 | 0.9577 ± 0.0050 | 0.1244 | 0.0124 | 0.1121 | 7.2 |
| 1e-5 | 0.2 | 0.9701 ± 0.0028 | 0.9577 ± 0.0046 | 0.1227 | 0.0149 | 0.1078 | 5.9 |
| 1e-5 | 0.1 | 0.9701 ± 0.0031 | 0.9572 ± 0.0054 | 0.1250 | 0.0145 | 0.1105 | 5.8 |
| 0 | 0.2 | 0.9700 ± 0.0028 | 0.9574 ± 0.0048 | 0.1210 | 0.0164 | 0.1046 | 5.7 |
| 0 | 0.03 | 0.9699 ± 0.0029 | 0.9572 ± 0.0046 | 0.1264 | 0.0130 | 0.1134 | 5.7 |
| 0 | 0 | 0.9699 ± 0.0029 | 0.9572 ± 0.0047 | 0.1238 | 0.0119 | 0.1119 | 5.7 |
| 0 | 0.1 | 0.9698 ± 0.0030 | 0.9573 ± 0.0047 | 0.1212 | 0.0142 | 0.1070 | 5.7 |

**Heatmaps:**

![[grid_val_acc_heatmap.png]]

![[grid_val_loss_heatmap.png]]

![[grid_gap_heatmap.png]]


**Best combo:** L2=`1e-3` σ=`0` → val_acc CV = **0.9750 ± 0.0018**, gap = **0.0750**.

> CSV fuente: [`ejercicio3/analisis/grid_reg/grid_summary.csv`](../../ejercicio3/analisis/grid_reg/grid_summary.csv).
 (hipótesis previas, para contrastar con resultados)

Anclo estos pronósticos **antes** de ver los datos, para que la lectura post-experimento no sea ad-hoc:

1. **Paso 1 (more_digits sin reg)**: test_acc subirá a **0.93-0.95**. La clase 8 ahora se aprende → resuelve el 9.7% del gap del Ej2 casi por completo. Posible gap residual por shift de distribución de otras clases (especialmente la 5).
2. **Paso 2 best combo**: ganancia adicional **0.005-0.015** sobre el baseline del paso 1 → test_acc esperado **0.94-0.96**. Si llegamos a ≥0.98, sería sorprendente; si no, es coherente con el "techo" del MLP sin batch-norm sobre 28k samples.
3. **Adam vs regularización**: en el análisis de "Sobreajuste" del Ej2 vimos que Adam ya tenía el gap más bajo. La regularización debería mover poco la aguja sobre Adam (vs Mom/SGD que sí se beneficiarían). Si el grid muestra muy poca mejora, eso confirma esa lectura.
4. **σ útil**: ruido leve (σ=0.03) ayuda; σ=0.2 podría degradar. Esperable curva no monótona en σ.
5. **L2 útil**: λ=1e-4 es el valor "sweet spot" típico para MLPs. λ=1e-3 puede empezar a dañar.

---

## Comparativa final (al cerrar Ej3)

### Resultados — Best combo del grid de regularización

**Best combo (paso 2):** L2 = `0.001` · σ = `0` · val_acc CV = **0.9750 ± 0.0018** · gap = **0.0750**.

**Generalización externa (test):**

![[best_reg_test_confusion_matrix.png]]

| Métrica | Test (best_reg) |
| --- | --- |
| accuracy        | **0.9601 ± 0.0030** |
| macro_precision | 0.9605 ± 0.0029 |
| macro_recall    | 0.9591 ± 0.0030 |
| macro_F1        | **0.9594 ± 0.0030** |

**Métricas por clase en test:**

| clase | precision | recall | F1 | support |
| --- | --- | --- | --- | --- |
| 0 | 0.966 | 0.990 | 0.978 | 245 |
| 1 | 0.976 | 0.991 | 0.983 | 282 |
| 2 | 0.963 | 0.965 | 0.964 | 258 |
| 3 | 0.923 | 0.988 | 0.955 | 252 |
| 4 | 0.969 | 0.969 | 0.969 | 245 |
| 5 | 0.961 | 0.910 | 0.935 | 222 |
| 6 | 0.962 | 0.974 | 0.967 | 239 |
| 7 | 0.968 | 0.948 | 0.958 | 257 |
| **8** | **0.981** | **0.904** | **0.941** | 243 |
| 9 | 0.938 | 0.952 | 0.945 | 252 |



## Comparativa final (test sobre `digits_test.csv`)

| Configuración | Test accuracy | Test macro_F1 |
| --- | --- | --- |
| Ej2 (sin more_digits, sin reg) | 0.8529 ± 0.0034 | 0.8062 ± 0.0034 |
| Ej3 baseline (+more_digits, sin reg) | **0.9616 ± 0.0025** | **0.9609 ± 0.0026** |
| **Ej3 best_reg (+more_digits + L2 + σ)** | **0.9601 ± 0.0030** | **0.9594 ± 0.0030** |


**Conclusiones:**

- Sumar `more_digits.csv` aporta **+0.1087** puntos de test_acc sobre el Ej2.
- Regularización aporta **-0.0015** puntos adicionales sobre el baseline.
- Ganancia total Ej2 → Ej3 best_reg: **+0.1072** puntos de accuracy.
- ⚠️ **No se alcanzó el ≥ 98%** pedido por CompanyX. Brecha residual = 0.0199.
