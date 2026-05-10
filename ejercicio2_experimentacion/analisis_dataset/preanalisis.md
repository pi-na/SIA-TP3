# Preánalisis del dataset — Ejercicio 2 (dígitos)

Scripts y outputs en: `ejercicio2/analisis_dataset/`

---

## 1. Estructura del dataset

| Característica | Valor |
|---|---|
| Archivo | `digits.csv` |
| Muestras totales | 12.449 |
| Features por muestra | 784 (imagen 28×28 píxeles) |
| Rango de valores | [0.0, 1.0] |
| Clases presentes | 0, 1, 2, 3, 4, 5, 6, 7, 9 |
| Clases ausentes | **8** (no hay muestras del dígito 8) |

El dataset tiene **9 clases**, no 10. El dígito 8 está completamente ausente de `digits.csv` — esto hay que tenerlo en cuenta al definir la capa de salida del MLP (9 neuronas, no 10) y al interpretar las métricas.

---

## 2. Distribución de clases

![Distribución de clases](analisis_dataset/distribucion_clases.png)

| Clase | n | Proporción |
|---|---|---|
| 0 | 1480 | 11.9% |
| 1 | 1685 | 13.5% |
| 2 | 1489 | 12.0% |
| 3 | 1532 | 12.3% |
| 4 | 1460 | 11.7% |
| **5** | **271** | **2.2%** |
| 6 | 1479 | 11.9% |
| 7 | 1566 | 12.6% |
| 9 | 1487 | 11.9% |

**Observación clave: la clase 5 está muy pocopresentada** (271 muestras vs ~1500 en el resto, 6× menos).

### Implicaciones para las métricas

La clase de métricas/sobreajuste establece que cuando el dataset está desbalanceado, **accuracy sola es engañosa**: un modelo que nunca predice "5" tendría accuracy = 97.8% (base rate de las otras clases). Por eso hay que reportar:

- **F1 macro**: media aritmética del F1 por clase, sin ponderar por frecuencia. Penaliza igual el error en la clase 5 que en el resto.
- **Precision y Recall por clase**: en particular para la clase 5, se espera que el modelo tenga menor recall (más falsos negativos) por la poca cantidad de muestras de entrenamiento.
- **Accuracy**: útil como referencia global pero no como métrica principal.

---

## 3. Imágenes del dataset

![Muestras por clase](analisis_dataset/muestras_por_clase.png)

Se observa variabilidad de escritura dentro de cada clase (distintos estilos, inclinaciones, grosores de trazo). Esta variabilidad es la que el MLP tiene que aprender a ignorar para generalizar correctamente.

---

## 4. Imagen media por clase (datos sin normalizar)

![Media por clase](analisis_dataset/media_por_clase.png)

La imagen media se obtiene promediando píxel a píxel todas las muestras de cada clase. Si las muestras de una clase están bien concentradas alrededor de una forma típica, la imagen media es nítida y reconocible. Si hay mucha variabilidad, la imagen queda borrosa.

**Observación:** todas las imágenes medias son reconocibles como el dígito correspondiente. Esto indica que dentro de cada clase las muestras comparten estructura — el MLP tiene señal suficiente para aprender.

---

## 5. Imagen media por clase tras normalización z-score

![Media normalizada](analisis_dataset/media_normalizada.png)

La normalización z-score por píxel transforma cada feature restando la media del dataset y dividiendo por el desvío estándar:

$$x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$

donde $j$ es el índice del píxel y la media y std se calculan sobre **todo el dataset de entrenamiento** (fit-on-train-only para evitar data leakage en CV).

La imagen media por clase en el espacio normalizado muestra:
- **Rojo (z > 0)**: píxeles que esta clase activa más que el promedio del dataset → zona característica de esa clase.
- **Azul (z < 0)**: píxeles que esta clase activa menos que el promedio → zona donde esta clase tiene menos tinta que la media.

Esto ilustra qué regiones del 28×28 son más informativas para distinguir cada clase, y confirma que la normalización no destruye la estructura: las formas siguen siendo reconocibles y diferenciales entre clases.

---

## 6. Decisiones que surgen del preánalisis

| Decisión                              | Justificación                                                                                                                      |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Capa de salida: 9 neuronas            | Solo hay 9 clases en el dataset                                                                                                    |
| Métrica principal: F1 macro           | Dataset desbalanceado (clase 5 con n=271) — accuracy sola es engañosa                                                              |
| Reportar Precision y Recall por clase | Para detectar si el modelo falla sistemáticamente en la clase 5                                                                    |
| Normalización z-score fit-on-train    | Evita data leakage en K-fold CV; los parámetros de normalización se calculan solo sobre el fold de train                           |
| Estratificación del K-fold            | Mantener la proporción de la clase 5 en cada fold (si no se estratifica, algunos folds podrían tener 0 o muy pocas muestras del 5) |
