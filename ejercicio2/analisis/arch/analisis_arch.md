# Análisis del sweep de arquitecturas — Ejercicio 2

**Experimento:** 4 arquitecturas × 5 seeds × 5 folds = 25 corridas por arquitectura.
**Fijo en todas:** Adam lr=0.001, batch=32, z-score, 50 épocas, early stopping patience=10.
**Datos crudos:** `raw.csv` | **Resumen:** `summary.csv`

---

## Configuraciones comparadas

| Nombre  | Arquitectura           | Capas ocultas | Parámetros aprox. |
| ------- | ---------------------- | ------------- | ----------------- |
| shallow | [784, 128, 10]         | 1             | 101.504           |
| base    | [784, 128, 64, 10]     | 2             | 109.696           |
| wider   | [784, 256, 128, 10]    | 2             | 242.304           |
| deeper  | [784, 128, 64, 32, 10] | 3             | 111.936           |

---

## Resultados (media ± std sobre 25 corridas — 5 seeds × 5 folds)

| Arquitectura | CE train final | CE val final | Accuracy val | F1 macro | Best epoch | Total épocas |
|---|---|---|---|---|---|---|
| shallow | 0.0034 ± 0.0101 | 0.2030 ± 0.0412 | 0.9576 ± 0.0061 | 0.8522 ± 0.0089 | 4.2 | 15.2 |
| base | 0.0032 ± 0.0039 | 0.2272 ± 0.0400 | 0.9554 ± 0.0059 | 0.8500 ± 0.0078 | 2.6 | 13.6 |
| wider | 0.0066 ± 0.0078 | 0.2511 ± 0.0523 | 0.9557 ± 0.0062 | 0.8505 ± 0.0075 | 2.2 | 13.2 |
| deeper | 0.0070 ± 0.0067 | 0.2498 ± 0.0335 | 0.9522 ± 0.0059 | 0.8471 ± 0.0079 | 2.1 | 13.1 |

**F1 por clase 5 (la más subrepresentada, n=271):**

| Arquitectura | F1 clase 5 |
|---|---|
| shallow | 0.8448 ± 0.0506 |
| base | 0.8406 ± 0.0427 |
| wider | 0.8428 ± 0.0423 |
| deeper | 0.8361 ± 0.0430 |

---

## Observaciones

### 1. Las arquitecturas son equivalentes entre sí

La diferencia en accuracy entre la mejor (shallow: 0.9576) y la peor (deeper: 0.9522) es de 0.54pp, menor que el std entre corridas de cualquiera (~0.006). Lo mismo ocurre en F1 macro (rango de 0.005, dentro del std de ~0.008). No hay diferencia estadísticamente significativa entre las cuatro arquitecturas para este dataset.

### 2. Sobreajuste en todas las arquitecturas

El resultado más importante del sweep no es cuál arquitectura gana, sino que **todas muestran sobreajuste claro**:

| Arquitectura | CE train final | CE val final | Ratio val/train |
|---|---|---|---|
| shallow | 0.0034 | 0.2030 | ~60× |
| base | 0.0032 | 0.2272 | ~71× |
| wider | 0.0066 | 0.2511 | ~38× |
| deeper | 0.0070 | 0.2498 | ~36× |

El error de entrenamiento es ~0.003-0.007 (muy bajo), mientras que el error de validación es ~0.20-0.25. Según la clase de regularización: cuando el error de training es mucho más bajo que el de validación y la brecha se abre, estamos en la zona de sobreajuste (mayor capacidad que la óptima).

Un dato adicional que confirma esto: el **best_epoch** es 2-4 en todos los casos. El modelo alcanza su mejor validación en las primeras épocas y luego sigue bajando el error de training mientras el de validación sube. El early stopping detiene el entrenamiento ~10 épocas después del mejor punto, y ahí el train loss ya cayó a ~0.003.

### 3. Más capacidad no ayuda — y empeora levemente

El sobreajuste se ve en la **CE de validación**: wider y deeper tienen val_loss más alto (0.25) que shallow (0.20). Agregar neuronas o capas da más capacidad al modelo para memorizar training, pero esa capacidad extra no se traduce en mejor generalización — al contrario, la val_loss sube.

Esto es consistente con la curva de capacidad de la clase: ya estamos a la derecha del punto óptimo. Agregar capacidad nos aleja más del óptimo.

### 4. ¿Qué implica para los próximos pasos?

La clase de regularización es explícita: cuando hay sobreajuste (brecha train/val abierta), la respuesta es **regularización**, no más capacidad.

El sobreajuste que vemos puede tener dos causas:
- **LR demasiado alto:** converge muy rápido (best_epoch = 2-4) a una solución que memoriza train. Un LR más chico podría llegar a un mínimo con mejor generalización.
- **Falta de regularización:** sin L2 ni dropout el modelo no tiene incentivo para aprender representaciones más generales.

Ambas cosas se van a explorar en los siguientes sweeps.

---

## Decisión de arquitectura base

Se elige **arch_base [784, 128, 64, 10]** como arquitectura base para los próximos sweeps porque:

- Tiene la justificación teórica más clara (dos niveles de jerarquía: partes → estructuras → clase)
- Sus métricas son representativas del conjunto (ni la mejor ni la peor)
- No es la más compleja (wider, deeper muestran más sobreajuste) ni la más simple

La diferencia entre architecturas es ruido estadístico para este dataset — lo que va a mover la aguja es el LR y la regularización.
