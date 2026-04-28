# Informe de implementación — Perceptrón lineal multi-feature

Documento que registra las **decisiones de implementación** tomadas durante el desarrollo del perceptrón lineal para el ejercicio 1 (knowledge distillation: TinyModel ≈ BigModel sobre `fraud_dataset`).

Cada sección documenta: la decisión, las alternativas consideradas y el motivo.

---

## 0. Regresion + Clasificacion

- Regresión = predecir un valor numérico continuo (ej. predecir 0.73, 0.42, 0.91). El error se mide con MSE (cuánto te alejaste delnúmero correcto).
- Clasificación = predecir una categoría discreta (ej. fraude / no-fraude, 0 / 1). El error se mide con TP/FP/recall/etc.

  Tu perceptrón lineal con activación identidad produce un número continuo (O = w · x), así que técnicamente está resolviendo un problema
  de regresión (intenta replicar big_model_fraud_probability, que es continuo). Después, al aplicar el threshold ≥ 0.5 para binarizar,
  convertís ese output continuo en una predicción de clasificación. Por eso reportamos las dos lecturas: MSE mide qué tan bien hace la
  regresión, y TP/FP/etc. miden qué tan bien clasifica una vez binarizado.

---

## 1. K-fold estratificado por `flagged_fraud`

**Decisión:** El K-fold se hace **estratificado por la columna `flagged_fraud`** — se separan las dos clases (`flagged_fraud == 0` y `flagged_fraud == 1`), se shufflean por separado, se cortan en K trozos cada una, y cada fold se arma combinando un trozo de cada clase. Así cada fold mantiene la proporción ~11.6% fraude / ~88.4% no-fraude del dataset original.

**Alternativas consideradas:**

- K-fold aleatorio simple (shuffle global + split). Más simple pero con varianza por fold mayor en datasets desbalanceados.
- Parámetro `stratify=True/False`. Descartado por overengineering — no agrega valor analítico para este TP.

**Motivo:** El dataset tiene ~11.6% de positivos (869 fraudes / 7500 filas). Con K-fold ingenuo, por azar puede tocar un fold con muchos más fraudes que otro, lo cual:

1. Ensucia la comparación entre folds (cada fold mide algo levemente distinto).
2. Aumenta la varianza de las métricas reportadas (precision/recall/F1) sin razón estadística válida.

La estratificación por `flagged_fraud` es el estándar para datasets desbalanceados y es consistente con lo que se reportará después en el análisis de generalización.

**Nota importante:** `flagged_fraud` se usa **únicamente para estratificar el sampling y para evaluar al final**. **No se usa como target de entrenamiento** (eso lo hace `big_model_fraud_probability`, según consigna del TP).

---

## 2. Target de entrenamiento, target de evaluación y umbral

**Decisión:** Separamos explícitamente tres roles que tienen las columnas del dataset:

| Rol                                              | Columna                                    | Uso                                                                                                                                          |
| ------------------------------------------------ | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| **Target de entrenamiento**                | `big_model_fraud_probability`            | Lo que el perceptrón intenta replicar (regresión continua en [0,1]). Es lo que entra en el cálculo de error/MSE durante el entrenamiento. |
| **Target de evaluación (clasificación)** | `flagged_fraud`                          | Ground truth binaria. Se compara contra la predicción binarizada para calcular TP/FP/FN/TN/accuracy/precision/recall/F1/TPR/FPR.            |
| **Umbral de binarización**                | parámetro `threshold` (default `0.5`) | Convierte el output continuo del perceptrón (`O = w · x`) en predicción binaria: `pred = 1 si O >= threshold else 0`.                 |

La función de testing devuelve **ambos tipos de métricas en una sola pasada**:

- **MSE** (regresión): `mean((O - big_model_fraud_probability)^2)` sobre el set de test. Mide qué tan bien el perceptrón replica al BigModel.
- **TP/FP/FN/TN/accuracy/precision/recall/F1/TPR/FPR** (clasificación): se aplican a `pred = (O >= threshold)` vs `flagged_fraud`. Miden si el perceptrón sirve como detector de fraude real, no solo como aproximador del BigModel.

**Alternativas consideradas:**

- **Solo MSE.** Insuficiente — el TP pide reportar métricas de clasificación contra `flagged_fraud`.
- **Umbral fijo en 0.5.** Simplifica pero perdés la posibilidad de explorar precision/recall trade-off.
- **Solo métricas de clasificación.** Tiraría a la basura la señal de qué tan bien aproximamos al BigModel.

**Motivo:**

1. La consigna del TP exige que el target de entrenamiento sea `big_model_fraud_probability` (knowledge distillation).
2. Pero la utilidad real del modelo se mide contra el ground truth `flagged_fraud`.
3. Reportar las dos lecturas en una sola pasada simplifica los notebooks de análisis y deja el threshold libre para barrer (curva ROC) en el análisis de generalización posterior.

**Notas sobre el output continuo:**
El perceptrón lineal con activación identidad (`O = w · x`) **no está acotado a [0,1]**. Puede dar valores negativos o > 1. Para el cálculo del MSE eso es fine (se compara raw). Para la binarización con threshold 0.5, también es fine — el corte funciona igual aunque el output exceda los límites. **No clipeamos**: dejamos el output crudo para que se vea fielmente la limitación del modelo lineal con identidad.

---

## 3. Normalización de features (z-score, fit-on-train-only)

**Decisión:** Estandarización **z-score por feature**, calculando `mean` y `std` **únicamente sobre el train fold** y aplicando esos mismos parámetros al test fold:

```
x_norm = (x - mean_train) / std_train
```

El bias (columna de unos) se prepende **después** de normalizar — no se normaliza.

**Alternativas consideradas:**

- **Min-max scaling a [0,1].** Sensible a outliers. Útil cuando el target tiene rango similar, pero con z-score el modelo arranca con escalas equivalentes y converge más estable.
- **Min-max scaling a [-1,1].** Bueno para targets simétricos; nuestro target está en [0,1], no aporta.
- **Sin normalización.** Inviable: `timestamp` (~1.7e9) y `account_age_days` (~3000) dominarían los gradientes y el resto de features quedaría con peso efectivo cero.

**Motivo:**

1. El dataset tiene features con escalas que difieren en **9 órdenes de magnitud** (timestamp ~1e9 vs quantity_purchased ~1e1). Sin normalización, el perceptrón lineal no converge razonablemente con un único learning rate global.
2. **Fit-on-train-only** es crítico para evitar **leak de información del test al train**: si calculamos `mean`/`std` sobre el dataset completo antes de splittear, el test contribuye a esos parámetros y las métricas de generalización quedan optimistamente sesgadas.
3. Z-score es el estándar para modelos lineales y la opción más robusta cuando hay features con outliers (ej. `amount_usd` tiene cola larga).

**Implementación práctica:**

- Función `fit_normalizer(df_train, feature_cols) -> (means, stds)` que calcula los parámetros sobre el train fold.
- Función `apply_normalizer(df, means, stds, feature_cols) -> df_normalized` que aplica la transformación tanto a train como a test.
- En folds donde `std == 0` para alguna feature (constante en el train), se reemplaza por `1.0` para evitar división por cero (la feature queda en 0 y no aporta señal).

---

## 4. Tracking de MSE por época (CSV separado por modelo)

**Decisión:** Para cada modelo entrenado, además del CSV principal con métricas finales, se guarda un **CSV separado con la curva de MSE vs época**: una fila por época, con columnas `fold`, `epoch`, `mse_train`. Esto permite graficar después la convergencia y comparar entre folds o entre configuraciones.

**Estructura tentativa del CSV de MSE-por-época:**

| fold | epoch | mse_train |
| ---- | ----- | --------- |
| 0    | 0     | 0.34      |
| 0    | 1     | 0.21      |
| ...  | ...   | ...       |
| 4    | 999   | 0.012     |

**Motivo:**

1. Permite analizar convergencia: ¿se estanca? ¿oscila? ¿hace falta más épocas o menos?
2. Permite comparar entre folds: si un fold converge mucho más lento, puede haber un problema de sampling o un outlier dominante.
3. Separar en CSV aparte (en vez de meter una columna gigante en el CSV de métricas) mantiene el CSV principal legible.

**Alternativas consideradas:**

- **Guardar también `mse_test` por época.** Costoso (1 forward pass extra por época) y no estrictamente necesario para el TP — podemos calcularlo solo al final. Si después se necesita una "learning curve" con train y test, se agrega.
- **Guardar el MSE en un solo CSV con todo lo demás.** Mezcla escalas (1 fila por fold vs 1 fila por época) y complica el análisis.

---

## 5. Estructura de outputs (carpeta por corrida + 3 CSVs)

**Decisión:** Cada invocación del script genera una carpeta única con la forma `output/<model_name>_<YYYYMMDD_HHMMSS>/`. El usuario pasa `--model-name` para identificar la corrida (ej. `--model-name baseline`, `--model-name no_timestamp`, `--model-name lr_001`). Adentro hay tres CSVs:

**(a) `metrics.csv`** — métricas agregadas por fold:

| Columna                                                           | Descripción                                                                    |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| `fold`                                                          | Identificador del fold (`0`...`K-1`, más filas finales `mean` y `std`) |
| `n_train`, `n_test`                                           | Cantidad de filas en cada split                                                 |
| `threshold`                                                     | Umbral usado para binarizar (default 0.5)                                       |
| `learning_rate`                                                 | Hiperparámetro                                                                 |
| `epochs_run`                                                    | Épocas ejecutadas (puede ser <`epochs` por convergencia)                     |
| `final_mse_train`                                               | MSE de train al final del entrenamiento                                         |
| `mse_test`                                                      | MSE de test (vs `big_model_fraud_probability`)                                |
| `tp`, `fp`, `fn`, `tn`                                    | Conteo de cada cuadrante (vs `flagged_fraud`)                                 |
| `accuracy`, `precision`, `recall`, `f1`, `tpr`, `fpr` | Métricas derivadas                                                             |

Después de las K filas de fold, se agregan dos filas de resumen: `fold = mean` y `fold = std` con la media y desviación de cada métrica numérica entre folds. Es la forma estándar de reportar K-fold CV.

**(b) `mse_history.csv`** — convergencia detallada:

| Columna       | Descripción                              |
| ------------- | ----------------------------------------- |
| `fold`      | Fold al que corresponde (`0`...`K-1`) |
| `epoch`     | Número de época                         |
| `mse_train` | MSE sobre train al final de esa época    |

Una fila por (fold, epoch). Permite graficar curvas de convergencia y compararlas entre folds.

**(c) `weights.csv`** — pesos aprendidos:

| Columna                                       | Descripción                                        |
| --------------------------------------------- | --------------------------------------------------- |
| `fold`                                      | Fold al que corresponde (`0`...`K-1`)           |
| `bias` (`w0`)                             | Término independiente                              |
| `<feature_name_1>` ... `<feature_name_n>` | Un peso por feature usada (después de exclusiones) |

Una fila por fold. Permite analizar estabilidad de los pesos entre folds (un peso que cambia mucho de signo entre folds es señal de feature poco informativa o redundante) y comparar entre modelos qué features dominan.

**Motivo:**

- `--model-name` obligatorio fuerza al usuario a etiquetar la corrida → cuando hagamos varios experimentos (con/sin timestamp, con/sin items_viewed, distintos LR), los outputs quedan auto-documentados.
- El `timestamp` en el path evita pisar corridas anteriores y mantiene un registro histórico naturalmente.
- Tres CSVs separados (métricas, historia, pesos) en lugar de uno solo: cada uno tiene su escala de filas (K, K×epochs, K) y mezclar los tres rompería la "tidy data".

---

## 6. Configuración por archivo JSON (CLI mínimo)

**Decisión:** Toda la configuración del modelo (hiperparámetros, features a excluir, columnas target/eval, K, seed, threshold, etc.) vive en un **archivo JSON**. El CLI queda reducido a tres argumentos:

```
python linear_perceptron.py \
    --config configs/baseline.json \
    --csv "data and documentation/fraud_dataset.csv" \
    [--output-dir output]
```

**Schema del JSON:**

```json
{
  "model_name": "baseline",
  "target_col": "big_model_fraud_probability",
  "eval_col": "flagged_fraud",
  "exclude_features": [
    "timestamp",
    "device_screen_resolution",
    "time_since_last_login_s"
  ],
  "k_folds": 5,
  "stratify": true,
  "random_seed": 42,
  "training": {
    "learning_rate": 0.01,
    "epochs": 1000,
    "epsilon": 1e-4
  },
  "evaluation": {
    "threshold": 0.5
  },
  "normalization": "zscore"
}
```

**Significado de cada campo:**

| Campo                      | Tipo               | Descripción                                                                                                                                                   |
| -------------------------- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_name`             | string             | Identifica la corrida. Va al nombre de la carpeta de output:`output/<model_name>_<timestamp>/`.                                                              |
| `target_col`             | string             | Columna que el perceptrón intenta predecir (regresión continua). Se excluye automáticamente del set de features.                                            |
| `eval_col`               | string             | Ground truth binario para métricas de clasificación. Se excluye automáticamente del set de features.                                                        |
| `exclude_features`       | array `<string>` | Features adicionales a excluir (sobre las descartadas por el análisis exploratorio:`timestamp`, `device_screen_resolution`, `time_since_last_login_s`). |
| `k_folds`                | int                | K del K-fold (default 5).                                                                                                                                      |
| `stratify`               | bool               | Si `true`, K-fold estratificado por `eval_col`.                                                                                                            |
| `random_seed`            | int                | Seed para shuffle e inicialización de pesos. Garantiza reproducibilidad.                                                                                      |
| `training.learning_rate` | float              | η del update rule.                                                                                                                                            |
| `training.epochs`        | int                | Tope de épocas.                                                                                                                                               |
| `training.epsilon`       | float              | Umbral de convergencia: si `mse_train < epsilon`, corta antes.                                                                                               |
| `evaluation.threshold`   | float              | Umbral de binarización del output continuo (default 0.5).                                                                                                     |
| `normalization`          | string             | `"zscore"` por ahora. Deja la puerta abierta a `"minmax"`/`"none"`.                                                                                      |

**Ubicación de los configs:** `ejercicio1/lineal_perceptron/configs/` con un archivo por experimento (ej. `baseline.json`, `lr_001.json`, `no_items_viewed.json`). Esto deja un registro autocontenido de cada modelo entrenado.

**Alternativas consideradas:**

- **Muchos flags CLI** (`--learning-rate`, `--epochs`, `--exclude`, etc.). Funcional pero tedioso para sweeps y deja la corrida sin un artefacto de config persistente.
- **JSON + flag `--seed-override`.** Descartado por overengineering — si querés cambiar el seed, editás el JSON o copiás el archivo.

**Motivo:**

1. **Reproducibilidad**: el JSON queda guardado junto al código, asociado a cada experimento. Para reproducir una corrida, basta con conservar el JSON + el commit de código.
2. **Facilita sweeps**: para barrer learning rates, hacés `cp baseline.json lr_001.json` y editás un campo. No hace falta acordarse de qué flags pasaste.
3. **Documentación viva**: cada `.json` en `configs/` es un experimento documentado. El TP final puede listar y comparar configs sin tocar el código.
4. **CLI mínimo y predecible**: 2-3 args fijos, fácil de scriptear.

---

## 7. Paralelismo entre folds (multiprocessing)

**Decisión:** Los K folds del K-fold CV se entrenan **en paralelo** usando `multiprocessing.Pool`. Los folds son independientes (no comparten estado, cada uno tiene su normalización fit-on-train, sus pesos iniciales, su historia de MSE), así que el speedup es prácticamente lineal con la cantidad de cores.

**API:**
- Por default, se usan `k_folds` workers (un proceso por fold).
- Flag CLI `--workers N` para sobreescribir (`--workers 1` fuerza ejecución serial, útil para debugging y profiling).
- La función worker top-level `_run_fold_worker` es picklable (requerido por `multiprocessing` en macOS, que usa `spawn` por default).

**Implementación:**

```python
fold_args = [(k_i, train_df, test_df, feature_cols, config, seed + k_i)
             for k_i, (train_df, test_df) in enumerate(folds)]
with mp.Pool(workers) as pool:
    results = list(pool.imap_unordered(_run_fold_worker, fold_args))
results.sort(key=lambda x: x[0])  # restaurar orden por fold_idx
```

`imap_unordered` arranca a procesar resultados a medida que terminan (en lugar de esperar al último), pero no garantiza orden de retorno → el `sort` final asegura que el CSV de métricas y el de pesos quedan en orden de fold.

**Reproducibilidad:** cada fold sigue usando `seed + k_i` como semilla (igual que en la versión serial), así que los resultados son **bit-exact** entre la corrida serial y la paralela.

**Alternativas consideradas:**
- **Lanzar los modelos en paralelo a nivel shell** (`python ... &` × 3). Funcionaría pero (a) no paraleliza dentro de un mismo modelo, (b) los outputs printeados se mezclan en stdout, (c) requiere disciplina externa al script.
- **`concurrent.futures.ProcessPoolExecutor`**. Equivalente funcional a `multiprocessing.Pool`, sin ventaja real para este caso de uso.
- **Threading.** Inútil acá: el entrenamiento es CPU-bound (NumPy + Python loop) y el GIL de CPython mata cualquier paralelismo real con threads.

**Motivo:**
1. **Speedup ~5×** con K=5 sobre máquinas con 5+ cores. Sweeps de ~12 min bajan a ~3-4 min.
2. **Cero costo conceptual**: los folds ya son independientes por construcción, no hay sincronización ni shared state.
3. **API explícita** (`--workers`): el usuario puede caer a serial si quiere debuggear con `pdb` o medir tiempo de un solo fold.

**Out of scope:** paralelismo *dentro* de un fold (ej. minibatch SGD vectorizado sobre todo el train) — sería una mejora ortogonal. Por ahora, el paralelismo entre folds es suficiente para los sweeps que estamos corriendo.

---
