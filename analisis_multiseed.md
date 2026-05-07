# Análisis sweep LR multi-seed — Ej1

## 1. Intención del experimento

Lo que cuenta `NOTAS PARA PRESENTACION NUEVA.md` es que en la presentación anterior la justificación del **learning rate** se apoyaba en una sola corrida por LR. Una sola corrida no separa dos cosas distintas: cuánto del MSE viene del LR en sí y cuánto viene del azar (inicialización de pesos uniforme + orden de muestras en online).

El experimento de `multiseed_runner.py` apunta a eso:

- 3 LRs por perceptrón × 5 seeds × 5 folds = 75 entrenamientos por modelo.
- `epochs=500`, suficiente para que las tres curvas converjan según el sweep single-seed previo.
- Para cada (lr, seed) promediamos sobre folds (K-fold cross-validation, slide 28 de la clase) y reportamos:
  - **MSE test** mean ± std → variabilidad total (seed × fold).
  - **MSE test seed-std** → variabilidad entre seeds (con folds promediados). Si esto es chico, el LR es robusto al azar.
  - **‖w‖** y **F1** para complementar.

Esto cubre el "Justificar elección del learning rate" y le da base al criterio de underfitting que se propone en las notas: comparar las métricas del perceptrón contra el baseline de 3 reglas determinísticas (precision=100%, recall=80%, acc=97.68%) calculado en `informe_analisis_dataset.pdf`.

Toda la teoría que se usa es la de la clase de "Métricas, sobreajuste y normalización":

- **Matriz de confusión** y métricas derivadas: accuracy, precision, recall, F1 (slides 10–15).
- **Underfitting = error alto en training** (slide 30). El piso de "alto" lo da el baseline.
- **K-fold cross-validation** para evaluar generalización sin depender de una partición (slides 27–29).
- **Z-score** para que todas las features entren al gradiente en escala comparable (slide 33).

## 2. Resultados

### 2.1 Perceptrón lineal (ADALINE, identidad)

| lr    | MSE test (mean ± std) | seed-std | ‖w‖ (mean) | F1 (mean)  |
| ----- | --------------------- | -------- | ---------- | ---------- |
| 1e-05 | 0.02622 ± 0.00119     | 0.00002  | 0.1945     | 0.5862     |
| 1e-04 | 0.02651 ± 0.00139     | 0.00006  | 0.2027     | **0.5930** |
| 1e-03 | 0.04599 ± 0.00616     | 0.00085  | 0.2810     | 0.5629     |

### 2.2 Perceptrón no-lineal (sigmoide)

| lr    | MSE test (mean ± std) | seed-std | ‖w‖ (mean) | F1 (mean)  |
| ----- | --------------------- | -------- | ---------- | ---------- |
| 1e-04 | 0.01128 ± 0.00052     | 0.00000  | 1.7510     | **0.5097** |
| 1e-03 | 0.01099 ± 0.00058     | 0.00000  | 1.9829     | 0.4996     |
| 1e-02 | 0.01099 ± 0.00058     | 0.00000  | 1.9792     | 0.5011     |

## 3. Análisis simple (sólo teoría de la clase)

### 3.1 Elección del LR

**Lineal.** lr=1e-05 y lr=1e-04 quedan empatados en MSE test (~0.0262, diferencia menor que el desvío entre folds). lr=1e-03 es claramente peor (MSE casi al doble) y además tiene más dispersión entre seeds (seed-std 8.5e-4 vs 6e-5). Eso indica que con lr=1e-03 el optimizador no se asienta en la misma zona en cada corrida — el azar de la inicialización pesa. Nos quedamos con **lr=1e-04**: tiene el mejor F1 promedio y la dispersión sigue siendo despreciable.

**No-lineal.** Las tres tasas terminan en MSE prácticamente idéntico (0.0110 vs 0.0113, diferencia dentro del ruido entre folds), con seed-std esencialmente 0. Es decir, el LR no cambia el punto al que converge el modelo, sólo la velocidad. La justificación pasa de MSE a **épocas hasta convergencia** (sweep single-seed previo: lr=0.01 ~30 épocas, lr=0.001 ~100). Elegir **lr=0.01** es coherente: misma calidad, menos cómputo.

### 3.2 ¿Hay underfitting?

Aplicando el criterio definido en las notas (perceptrón en train tiene que igualar al baseline de 3 reglas: acc≥97.68%, precision≈100%, recall≥80% → F1≈0.889):

- F1 lineal: 0.59. Muy por debajo de 0.89 → **underfitting**.
- F1 no-lineal: 0.51. También muy por debajo → **underfitting** (incluso peor en F1, aunque mejor en MSE).

Esto es consistente con la predicción teórica: las 3 reglas duras del dataset son saltos de ~6% a 100% en un entero (`quantity≥10`, `items≥15`, `amount>500`). Una recta + sigmoide saturada no puede reproducir esas discontinuidades, así que el modelo "promedia" la región y pierde recall sobre los fraudes claros sin ganar nada en los sutiles. El no-lineal mejora el MSE (la sigmoide lo deja saturar mejor cerca de 0/1 cuando big_model_fraud_probability está cerca de los extremos) pero el F1 no sube porque el problema no es de regresión, es de capacidad para reproducir los saltos.

### 3.3 Dispersión entre seeds

En todos los casos la seed-std es 1–2 órdenes más chica que la std fold-a-fold. Conclusión simple: el azar de la inicialización uniforme **no afecta** los resultados del Ej1 a esta escala de dataset (7500 filas, online training, 500 épocas). Reportar promedio sobre 5 seeds es suficiente para cerrar la justificación.

## 4. Qué falta

- Calcular **accuracy / precision / recall** del lineal y no-lineal con el LR elegido y compararlas explícitamente contra el baseline (acc=97.68% / prec=100% / rec=80%) en una tabla única en la slide.
- Re-evaluar el **threshold** de decisión: hoy las métricas dependen del threshold elegido (0.5 por defecto). Para hacer la comparación contra el baseline más limpia, conviene explorar el threshold como hiperparámetro y elegir el que maximiza F1, dejando documentada la decisión.
