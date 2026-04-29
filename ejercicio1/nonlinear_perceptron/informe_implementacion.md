# Informe de Implementacion: Perceptron No-Lineal (Sigmoide)

## 1. Cambios respecto al perceptron lineal

El perceptron no-lineal introduce dos modificaciones sobre el Adaline lineal:

1. **Activacion sigmoide**: la salida pasa de `O = w . x` a `O = sigmoid(w . x) = 1/(1+exp(-w.x))`. Esto mapea la salida al rango [0,1], que coincide con el rango del target `big_model_fraud_probability`.

2. **Regla de actualizacion**: se incorpora la derivada de la sigmoide en el delta-rule:
   - Lineal: `dw = eta * (z - O) * x`
   - No-lineal: `dw = eta * (z - O) * O*(1-O) * x`

El factor `O*(1-O)` modula el gradiente: cuando la salida esta cerca de 0 o 1 (zonas saturadas), el gradiente se achica; cuando esta cerca de 0.5, es maximo.

Se implemento la sigmoide con estabilidad numerica (mask-based, evitando overflow en `exp`).

## 2. Arquitectura del codigo

El archivo `nonlinear_perceptron.py` conserva la misma estructura del lineal:
- K-fold estratificado por `flagged_fraud`
- Normalizacion z-score (fit-on-train-only)
- Evaluacion con MSE (vs `big_model_fraud_probability`) y metricas de clasificacion (vs `flagged_fraud`)

Adicionalmente genera un **predictions.csv** con las predicciones out-of-fold (score sigmoide por muestra), necesario para el threshold sweep posterior.

---

## 3. Experimentos

### Exp 1: Baseline (lr=0.0001, 500 epochs, 5-fold)

| Metrica       | Mean    | Std     |
|---------------|---------|---------|
| MSE test      | 0.0113  | 0.0003  |
| Accuracy      | 0.7775  | 0.0060  |
| Precision     | 0.3425  | 0.0058  |
| Recall        | 1.0000  | 0.0000  |
| F1            | 0.5102  | 0.0065  |

Con threshold=0.5 (default), el modelo clasifica casi todo como fraude (recall=1.0, precision baja). Esto indica que las salidas de la sigmoide estan comprimidas hacia la zona alta: la media de los scores es 0.43, y los scores de fraude tienen mediana 0.99.

### Exp 2: Comparacion de Learning Rates

Se evaluan tres tasas de aprendizaje con el mismo setup (500 epochs, 5-fold estratificado, seed=42).

**Metricas de distillation (MSE vs `big_model_fraud_probability`):**

| Learning Rate | MSE train (mean +/- std) | MSE test (mean +/- std) | Gap train-test |
|---------------|--------------------------|-------------------------|----------------|
| 0.01          | 0.0110 +/- 0.0001        | 0.0110 +/- 0.0004       | ~0.0000        |
| 0.001         | 0.0110 +/- 0.0001        | 0.0110 +/- 0.0004       | ~0.0000        |
| 0.0001        | 0.0112 +/- 0.0001        | 0.0113 +/- 0.0003       | +0.0001        |

**Metricas operativas (clasificacion vs `flagged_fraud`, con threshold default 0.5):**

| Learning Rate | Precision | Recall | F1    |
|---------------|-----------|--------|-------|
| 0.01          | 0.335     | 1.000  | 0.502 |
| 0.001         | 0.333     | 1.000  | 0.499 |
| 0.0001        | 0.342     | 1.000  | 0.510 |

Con threshold=0.5 las metricas operativas son pobres (precision ~0.33). Esto no refleja la calidad del modelo sino un threshold inadecuado: la sigmoide concentra los scores de fraude por encima de 0.85.

**Mejor modelo por MSE test**: lr=0.001 (MSE=0.0110). La diferencia con lr=0.01 es despreciable (< 0.0001).

Todos los modelos corrieron las 500 epochs completas sin alcanzar epsilon=1e-5, indicando que 500 epochs son insuficientes para convergencia total, aunque el MSE ya esta en un plateau.

### Exp 3: Threshold Sweep

Se barre el threshold de 0.00 a 1.00 (paso 0.01) sobre las predicciones out-of-fold del modelo con lr=0.001 (mejor MSE).

**Metricas operativas post-sweep:**

| Learning Rate | AUC-ROC | Best th (F1) | Precision | Recall | F1    |
|---------------|---------|--------------|-----------|--------|-------|
| 0.01          | 0.9920  | 0.89         | 0.887     | 0.859  | 0.873 |
| 0.001         | 0.9921  | 0.89         | 0.886     | 0.861  | 0.873 |
| 0.0001        | 0.9926  | 0.86         | 0.890     | 0.862  | 0.876 |

Detalle del sweep para el modelo lr=0.001:

| Threshold | Precision | Recall | F1     | Accuracy | TP  | FP  | FN  | TN   |
|-----------|-----------|--------|--------|----------|-----|-----|-----|------|
| 0.50      | 0.333     | 1.000  | 0.499  | 0.768    | 869 | 1743| 0   | 4888 |
| 0.70      | 0.585     | 0.987  | 0.735  | 0.917    | 858 | 609 | 11  | 6022 |
| 0.80      | 0.751     | 0.952  | 0.840  | 0.958    | 827 | 274 | 42  | 6357 |
| 0.85      | 0.812     | 0.893  | 0.850  | 0.964    | 776 | 180 | 93  | 6451 |
| 0.88      | 0.864     | 0.869  | 0.866  | 0.969    | 755 | 119 | 114 | 6512 |
| **0.89**  | **0.886** | **0.861** | **0.873** | **0.971** | **748** | **96** | **121** | **6535** |
| 0.90      | 0.891     | 0.839  | 0.864  | 0.969    | 729 | 89  | 140 | 6542 |
| 0.92      | 0.915     | 0.806  | 0.857  | 0.969    | 700 | 65  | 169 | 6566 |
| **0.95**  | **0.949** | **0.746** | **0.835** | **0.966** | **648** | **35** | **221** | **6596** |
| 0.97      | 0.977     | 0.677  | 0.800  | 0.961    | 588 | 14  | 281 | 6617 |
| 0.99      | 1.000     | 0.527  | 0.690  | 0.945    | 458 | 0   | 411 | 6631 |

### Exp 4: Comparacion Lineal vs No-Lineal

**Metricas de distillation:**

| Modelo      | MSE test (mean +/- std)  | Reduccion vs lineal |
|-------------|--------------------------|---------------------|
| Lineal      | 0.0266 +/- 0.0008        | -                   |
| No-Lineal   | 0.0110 +/- 0.0004        | -59%                |

El no-lineal reduce el MSE en un 59%. La sigmoide mapea la salida a [0,1], coincidiendo con el rango natural del target. El lineal puede producir valores fuera de [0,1], penalizando el MSE.

**Metricas operativas (cada modelo con su threshold optimo por F1):**

| Metrica             | Lineal (th=0.71)  | No-Lineal (th=0.89) | Baseline deterministico |
|---------------------|-------------------|----------------------|------------------------|
| Precision           | 0.940             | 0.886                | 1.000                  |
| Recall              | 0.837             | 0.861                | 0.800                  |
| F1                  | 0.885             | 0.873                | 0.889                  |
| AUC-ROC             | 0.972             | 0.992                | -                      |

Observaciones:
- **AUC-ROC superior** (+2 puntos): el no-lineal tiene mejor capacidad de discriminacion global.
- **F1 comparable pero ligeramente inferior** (0.873 vs 0.885): la sigmoide comprime los scores, dificultando un corte limpio.
- **Recall superior** (0.861 vs 0.837): detecta mas fraudes a costa de algo mas de falsos positivos.
- **Threshold desplazado** (0.89 vs 0.71): coherente con la compresion sigmoide de las salidas.

---

## 5. Estudio de generalizacion (modelo seleccionado: lr=0.001)

### 5.1. Criterio de seleccion del modelo

Se selecciona el modelo con lr=0.001 por tener el **MSE test minimo** (0.0110), que es el criterio primario de knowledge distillation. La diferencia con lr=0.01 es despreciable, pero lr=0.001 es preferible por mayor estabilidad numerica del entrenamiento online.

### 5.2. Metricas de distillation: MSE vs `big_model_fraud_probability`

El objetivo de distillation es que el TinyModel replique la probabilidad continua del BigModel. La metrica relevante es el MSE entre la salida sigmoide y `big_model_fraud_probability`.

| Fold | MSE train | MSE test | Gap     |
|------|-----------|----------|---------|
| 0    | 0.0109    | 0.0111   | +0.0001 |
| 1    | 0.0108    | 0.0114   | +0.0006 |
| 2    | 0.0111    | 0.0102   | -0.0009 |
| 3    | 0.0109    | 0.0113   | +0.0004 |
| 4    | 0.0110    | 0.0109   | -0.0001 |
| **Mean** | **0.0110** | **0.0110** | **~0.0000** |

**No hay evidencia de overfitting**: el gap train-test es esencialmente cero (< 0.001 en todos los folds). Esto es esperable: un perceptron simple (7 parametros: 6 features + bias) tiene capacidad limitada y no puede sobreajustar 6000 muestras de entrenamiento.

La variabilidad entre folds es baja (std=0.0004), indicando que el modelo generaliza de forma estable independientemente de la particion de datos.

**Distribucion de scores por clase:**

| Clase        | Media  | Mediana | Min    | Max    |
|--------------|--------|---------|--------|--------|
| Fraude (869) | 0.9568 | 0.9916  | 0.5262 | 1.0000 |
| Legit (6631) | 0.3565 | 0.3148  | 0.0021 | 0.9877 |

La separacion entre clases es clara (mediana fraude=0.99 vs mediana legit=0.31), lo que explica el AUC de 0.992.

**MSE por subgrupo:**
- Fraude: MSE = 0.0019 (el modelo replica bien las probabilidades altas del BigModel)
- Legit: MSE = 0.0122 (mayor error en las probabilidades bajas, donde hay mas variabilidad)

### 5.3. Metricas operativas: clasificacion vs `flagged_fraud`

Estas metricas son relevantes para el uso operativo del modelo (decidir si bloquear o revisar una transaccion). No participan en el entrenamiento; `flagged_fraud` es ground truth externo.

**AUC-ROC: 0.9921**

La curva ROC se calculo desde todos los scores unicos (no solo la grilla de 101 puntos), usando el metodo trapezoidal. Un AUC de 0.992 indica discriminacion excelente: el modelo asigna un score mayor a un fraude real que a una transaccion legitima en el 99.2% de los pares posibles.

**Curva F1 vs threshold (resumen):**

El F1 es maximo en th=0.89 (F1=0.873) y se mantiene por encima de 0.85 en el rango [0.82, 0.92]. Fuera de ese rango, la precision o el recall caen rapidamente.

### 5.4. Robustez del K-fold

La evaluacion usa 5-fold estratificado, donde cada muestra aparece exactamente una vez como test. Los 7500 scores en predictions.csv cubren el dataset completo sin leakage.

Variabilidad inter-fold del MSE test: std=0.0004 (coeficiente de variacion=3.6%). Esto confirma que los resultados no dependen de una particion afortunada.

---

## 6. Recomendacion de threshold para CompanyX

El threshold optimo depende del **costo relativo** de los dos tipos de error:
- **Falso negativo (FN)**: fraude no detectado, perdida economica directa.
- **Falso positivo (FP)**: transaccion legitima bloqueada, friccion con el cliente.

### Opcion A: th=0.89 (maximo F1)

| Metrica    | Valor  |
|------------|--------|
| Precision  | 0.886  |
| Recall     | 0.861  |
| F1         | 0.873  |
| FP (7500)  | 96     |
| FN (7500)  | 121    |

- De cada 100 alertas, ~89 son fraudes reales.
- Se escapan 121 fraudes de 869 (13.9% de los fraudes no detectados).
- El equipo de revision ve 96 falsos positivos.

**Adecuado cuando**: el costo de revisar una alerta falsa es comparable al costo de perder un fraude, o cuando el equipo de revision tiene capacidad limitada.

### Opcion B: th=0.95 (alta precision)

| Metrica    | Valor  |
|------------|--------|
| Precision  | 0.949  |
| Recall     | 0.746  |
| F1         | 0.835  |
| FP (7500)  | 35     |
| FN (7500)  | 221    |

- De cada 100 alertas, ~95 son fraudes reales.
- Se escapan 221 fraudes de 869 (25.4% no detectados).
- Solo 35 falsos positivos (63% menos que th=0.89).

**Adecuado cuando**: bloquear transacciones legitimas tiene alto costo (clientes VIP, operaciones de alto valor), y se prefiere actuar solo cuando la confianza es muy alta.

### Opcion hibrida (recomendacion)

En la practica, los sistemas de fraude no toman una decision binaria. Se recomienda un esquema de dos niveles:

1. **score >= 0.95: bloqueo automatico**. Precision 94.9%, solo 35 FP cada 7500 transacciones. Riesgo de friccion minimo.
2. **0.89 <= score < 0.95: revision manual**. Estas ~135 transacciones incluyen ~100 fraudes adicionales. Un analista puede revisarlas con baja carga operativa.
3. **score < 0.89: aprobacion automatica**. El 13.9% de fraudes restante requeriria bajar el threshold hasta ~0.80, a costa de 274 FP, lo cual puede no justificarse.

Con este esquema, la cobertura efectiva seria:
- Fraudes detectados: 748 de 869 (86.1%) automaticamente o via revision
- Falsos positivos totales: ~96 (1.3% de transacciones legitimas)

---

## 7. Conclusiones

1. **Distillation**: el perceptron no-lineal replica la probabilidad del BigModel con MSE=0.0110, un 59% menor que el lineal (0.0266). La sigmoide es la activacion natural para un target en [0,1].

2. **Generalizacion**: no hay overfitting (gap train-test ~0). El modelo generaliza de forma estable con baja variabilidad inter-fold.

3. **Discriminacion**: AUC-ROC=0.992, superior al lineal (0.972). La capacidad de ordenar correctamente fraudes vs legitimos es excelente.

4. **Clasificacion**: F1=0.873 con th=0.89, comparable al lineal (0.885 con th=0.71) y al baseline deterministico (0.889). Un perceptron simple esta cerca de su limite de capacidad para este problema.

5. **Sensibilidad al threshold**: el threshold optimo (0.89) es alto porque la sigmoide comprime las salidas. Es critico calibrar el threshold post-entrenamiento; usar th=0.5 produce precision inaceptable (0.33).

6. **Robustez al learning rate**: las tres tasas evaluadas (0.01, 0.001, 0.0001) producen resultados practicamente identicos (MSE 0.0110-0.0113), indicando un problema bien condicionado.

---

## 8. Archivos generados

Para cada experimento en `output/<model_name>_<timestamp>/`:
- `metrics.csv`: metricas por fold + filas mean/std
- `mse_history.csv`: curva de entrenamiento (MSE por epoca/fold)
- `weights.csv`: pesos finales por fold
- `predictions.csv`: predicciones out-of-fold (score sigmoide)
- `config.json`: copia del config usado

Para los 4 modelos (threshold sweep):
- `threshold_sweep.csv`: metricas para 101 thresholds (0.00 a 1.00)
- `roc_curve.csv`: curva ROC desde scores unicos
- `pr_curve.csv`: curva precision-recall desde scores unicos
