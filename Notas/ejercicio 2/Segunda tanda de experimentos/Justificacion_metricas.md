# Justificación de las métricas usadas en la 2ª tanda de experimentos

Todos los experimentos cruzados de esta tanda ([Pre_LR_Batch_Opt](Pre_LR_Batch_Opt/analisis.md), [Cross_LR_Opt_Arch](Cross_LR_Opt_Arch/analisis.md), [Arch_tiebreaker](Arch_tiebreaker/analisis.md)) se ranquean por **`val_acc`** y **`macro_f1`**. Esta nota explica qué son, cómo se calculan, y **por qué son las métricas correctas** para decidir la config óptima en este problema.

---

## El contexto: digits.csv y la clase de métricas

### El dataset NO está balanceado

Según el [análisis del dataset](../analisis_dataset/preanalisis.md): de las 10 clases del problema (dígitos 0-9), 9 tienen ~1500 ejemplos y la **clase 5 tiene 271** (≈6× menos que las otras). Esto es desbalance moderado-fuerte.

| Clase | n | Proporción |
|---|---|---|
| 0 | 1480 | 11.9% |
| 1 | 1685 | 13.5% |
| ... | ... | ... |
| **5** | **271** | **2.2%** |
| ... | ... | ... |

### Lo que dice la clase de métricas/sobreajuste

La [clase de métricas y sobreajuste](../../docs/clase_metricas_sobreajuste/metricas_sobreajuste.pdf) establece dos cosas que aplican directo a este problema:

1. **Accuracy sola es engañosa cuando hay desbalance.** Un modelo trivial que predice siempre la clase mayoritaria, o que nunca predice la clase 5, tiene accuracy ≈ 97.8% (base rate del problema sin clase 5). Eso "se ve bien" pero es inútil — lo que querés es que clasifique las 10 clases bien, no las 9 fáciles.
2. **Hay que reportar el set completo:** Accuracy + Precision + Recall + F1, y cuando hay multiclase, **macro-average** por default (regla 4 del CLAUDE.md raíz).

Esa es la base teórica del por qué usamos val_acc Y macro_f1 juntas, no una sola.

---

## `val_acc` — Validation Accuracy

### Definición

**Accuracy = fracción de predicciones correctas sobre el total.**

```
acc = (predicciones correctas) / (total de predicciones)
```

En multiclase con softmax, "predicción" = `argmax(salida_softmax)` → la clase a la que el modelo le asigna mayor probabilidad.

### Cómo se mide en el repo

En `mlp/train.py` después de entrenar (con los `best_weights` restaurados gracias al fix #4):

```python
val_pred  = mlp.predict(X_val)             # devuelve argmax(softmax) por fila
val_acc_final = (val_pred == y_val_raw).mean()
```

Esto es: hago forward, tomo el argmax, comparo con la etiqueta verdadera, promedio el match (1 si acierta, 0 si no).

**Sobre qué set:** sobre el **fold de validación** del k-fold (cada fold rota cuál de los 5 subsets es validación). El `digits_test.csv` no se toca durante búsqueda de hiperparámetros (regla del TP).

**Cómo se promedia entre corridas:** la regla 3 del CLAUDE.md exige declarar el eje de promediación. En esta tanda:
- Para una cell: `val_acc_mean_seedsfolds = mean(val_acc_final)` sobre las 15 corridas (3 seeds × 5 folds) en cross_v1, o 75 (15 × 5) en el tiebreaker.
- El `std` reportado es el desvío sobre esas mismas 15/75 corridas.
- El `SEM` (error estándar de la media) = `std / √n` y es el que usamos para decidir si dos cells son distinguibles.

### Qué responde val_acc

*"De cada 100 imágenes que el modelo no vio durante entrenamiento, ¿cuántas clasificó bien?"*

### Lo que NO captura

- **No distingue qué clase erra.** Si el modelo nunca acierta la clase 5 (271 ejemplos = 2.2% del dataset), pero acierta perfecto las otras 9, la val_acc es 97.8%. Ése es exactamente el escenario que la clase de métricas advierte.
- **No es threshold-independent.** Como acá usamos argmax (no threshold sobre probabilidad), esto no es un problema directo, pero sí lo es para problemas binarios (Ej1).

---

## `macro_f1` — F1 Macro Promedio

### F1 por clase, paso por paso

Para cada clase c se cuentan:
- **TP_c** (true positives): predicciones que dicen "c" Y la etiqueta es "c".
- **FP_c** (false positives): predicciones que dicen "c" pero la etiqueta NO es "c".
- **FN_c** (false negatives): la etiqueta es "c" pero la predicción NO dijo "c".

A partir de eso:

```
precision_c = TP_c / (TP_c + FP_c)   "de todo lo que predije como c, ¿cuánto era realmente c?"
recall_c    = TP_c / (TP_c + FN_c)   "de todos los c que había, ¿cuántos detecté?"
f1_c        = 2 · (precision_c · recall_c) / (precision_c + recall_c)   media armónica
```

### Macro-average

```
macro_f1 = (1/C) · Σ_c f1_c
```

donde C = número de clases (10 acá). Es la **media aritmética del F1 por clase**, sin pesar por frecuencia. Esto significa que la clase 5 (con 271 ejemplos) **pesa exactamente lo mismo que la clase 1 (con 1685 ejemplos)** en el promedio.

### Por qué macro y no micro/weighted

- **micro_f1** = computa precision/recall/F1 sobre el conjunto agregado de TPs/FPs/FNs de todas las clases. **En multiclase con argmax, micro_f1 ≡ accuracy.** Sería redundante reportarlo.
- **weighted_f1** = promedia los F1 por clase ponderando por la frecuencia de cada clase. Hace que la clase 5 pese ~6× menos que las otras. Eso es **exactamente lo opuesto a lo que queremos** dado que la clase 5 es la difícil — pondrías al modelo a competir por aciertos en clases fáciles.
- **macro_f1** = ignora la frecuencia. **Penaliza igual los errores en clase rara y en clase frecuente.** Es el default de la cátedra y el correcto para este problema.

### Cómo se mide en el repo

En `mlp/metrics.py` (función `multiclass_metrics`):

```python
for c in range(num_classes):
    tp = ((y_pred == c) & (y_true == c)).sum()
    fp = ((y_pred == c) & (y_true != c)).sum()
    fn = ((y_pred != c) & (y_true == c)).sum()
    precision[c] = tp / (tp + fp + eps)
    recall[c]    = tp / (tp + fn + eps)
    f1[c]        = 2 * precision[c] * recall[c] / (precision[c] + recall[c] + eps)

macro_f1 = f1.mean()
```

(donde `eps = 1e-12` para evitar división por cero cuando una clase tiene 0 TPs).

### Qué responde macro_f1

*"En promedio sobre las 10 clases, ¿qué tan balanceado es el modelo entre encontrar todos los ejemplos de cada clase (recall) y no inventar predicciones erradas (precision)?"*

---

## Por qué usamos AMBAS — y no sólo una

### Caso 1: usar sólo accuracy

Falla en este problema por el desbalance. Un modelo que ignora la clase 5 puede tener accuracy ~0.97 y macro_f1 ~0.85. Si rankearas sólo por accuracy, podrías elegir un modelo peor en la métrica que captura "balance entre clases".

### Caso 2: usar sólo macro_f1

Falla porque macro_f1 no distingue entre "clase 5 tiene precision 0.5 y recall 1.0" y "precision 1.0 y recall 0.5" (ambos dan F1≈0.67). Para un usuario humano, accuracy comunica algo intuitivo ("acierta 95 de cada 100") que F1 no.

### Por eso reportamos las dos

En toda la 2ª tanda usamos val_acc como **métrica de ranking principal** y macro_f1 como **métrica de control**. Si las dos coinciden en el ranking → decisión robusta. Si discrepan, **pasa algo importante** (típicamente: el "ganador en accuracy" tiene problemas con la clase 5) y hay que mirar precision/recall por clase.

**Empíricamente en cross_v1 stage 2:** las dos métricas están correlacionadas (Pearson ≈ 0.99 entre los means de val_acc y macro_f1 en las 60 cells). Eso valida que la elección de "centro" no depende de cuál de las dos prioricemos. Ojo: que estén correlacionadas en el grid no quiere decir que sean redundantes — sí lo son **a este nivel de capacidad de los modelos**, pero podría dejar de pasar si después agregamos regularización o data augmentation.

---

## Otras métricas que reportamos pero no usamos para ranking

### `train_acc_final`

Accuracy en el fold de **train**. La reportamos para detectar overfitting (gap train−val). **NO la usamos para ranking** porque un modelo puede tener train_acc=1.0 y val_acc=0.6 (sobreajuste perfecto). Es información de diagnóstico, no de selección.

### `val_loss` (cross-entropy)

La métrica que el modelo **minimiza** durante entrenamiento. Es la "loss" del problema (no la métrica final). La usamos para:
- Early stopping: ES corta cuando val_loss deja de bajar.
- Análisis de calibración: una val_loss baja con val_acc igual a otro modelo significa que las **probabilidades** del primero están mejor calibradas (no sólo el argmax).

**Por qué no rankear por val_loss:** la cross-entropy responde una pregunta distinta a accuracy ("qué tan bien calibradas están las probabilidades"). Para "elegir el mejor clasificador" lo correcto es ranquear por la métrica de **clasificación** (acc, F1), no por la de optimización (CE). Esto es exactamente la distinción que CLAUDE.md regla 4 obliga a respetar: hay que reportar **ambos** tipos de métrica, pero no son intercambiables.

### `macro_precision`, `macro_recall`, `weighted_f1`

Reportadas en `summary.csv` por completitud y para el set completo de la regla 4. Las usamos cuando la decisión queda empatada en val_acc y macro_f1, o cuando una clase específica nos importa especialmente (ej. la clase 5 minoritaria).

---

## Resumen — por qué `val_acc + macro_f1` es la combinación correcta

1. **El dataset es desbalanceado** (clase 5 con 271 ejemplos vs ~1500 las otras). Accuracy sola sería engañosa, macro_f1 corrige eso al pesar igual cada clase.
2. **Macro vs weighted/micro** está justificada por la cátedra y por la pregunta del problema (importa cada clase, no la frecuencia).
3. **Reportar ambas** detecta inconsistencias: si una cell gana en una y pierde en la otra, hay un problema de balance entre clases que merece análisis dedicado.
4. **No rankeamos por loss** porque CE responde "qué tan calibrada está la probabilidad", no "qué tan bien clasifica".
5. **No usamos train_acc** para ranking porque expone a sobreajuste — la regla del curso es validar siempre con datos no vistos durante la búsqueda.

Toda la sesión de experimentos cruzados se ranquea con esta lógica. La conclusión final del Ej2 (`arch_shallow + Adam + LR=1e-3 + batch=64`) es la que **gana en val_acc Y empata en macro_f1** con el segundo lugar (`arch_wider`), y por Occam decidimos en favor del modelo más chico cuando el ranking estadístico no las distingue.

---

## Referencias

- [Clase de métricas y sobreajuste](../../docs/clase_metricas_sobreajuste/metricas_sobreajuste.pdf) — fundamento teórico.
- CLAUDE.md raíz, **regla 4** (set completo de métricas) y **regla 3** (explicitar eje de promedio).
- [Análisis del dataset](../analisis_dataset/preanalisis.md) — distribución de clases que motiva macro vs weighted.
- `mlp/metrics.py` — implementación de `multiclass_metrics`.
- `mlp/train.py` — dónde se computan `val_acc_final` y `macro_f1` en cada corrida.
