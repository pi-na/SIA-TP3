# Justificación de K-Fold Cross-Validation con K=5 (Ej1, generalización)

Este documento responde a la corrección del profesor sobre la sección de generalización: *"No explican por qué usan K-Fold=5"*. Justifica tanto la elección del esquema K-fold (vs un único holdout) como la elección específica de K=5 (vs K=2, K=10 o leave-one-out).

---

## 1. ¿Por qué K-fold y no un único holdout 80/20?

La clase de métricas/sobreajuste (Eugenia Sol Piñeiro, transcript 00:37:43–00:39:40) introduce K-fold como respuesta directa al problema del holdout:

> *"¿Cómo sé si esa partición [80/20] me sirve? Porque no es trivial. [...] Una técnica, por ejemplo, se llama K Fold Cross Validation o validación k cruzada [...] va a tener en cuenta todos estos escenarios, no solo 1, entonces va a evitar que el modelo se adapte a una partición en específico que quizás puede haber estado particionado al azar."*

Concretamente:

- **Holdout 80/20**: una sola partición azarosa. El estimador del error de generalización depende fuertemente de qué muestras cayeron en test → varianza alta del estimador entre realizaciones del split.
- **K-fold**: rota la partición K veces → K estimaciones independientes del error → la **media reduce varianza** del estimador, y la **std entre folds** cuantifica la sensibilidad a la partición.

Para nuestro dataset con `N=7500` y clase positiva minoritaria (`flagged_fraud=1` en 11.59% de las filas), un holdout azaroso puede dejar el test con 8–14% de positivos por azar, y eso solo desplaza A/P/R/F1 sin que tenga que ver con la calidad del modelo. K-fold + estratificación elimina esa fuente de ruido.

## 2. ¿Por qué K=5 específicamente?

K-fold tiene un **trade-off bias–varianza del estimador** que depende de K. La clase no lo cuantifica (no aparece en transcripts ni en PDF), así que la elección se justifica con razonamiento estándar de la literatura adaptado a nuestro contexto:

### 2.1. Cuadro comparativo de K

| K | Train por fold | Val por fold (sobre N=7500) | Trade-off |
|---|---|---|---|
| **2** | 50% (3750) | 50% (3750) | **Bias alto del estimador**: cada modelo se entrena con la **mitad** de los datos disponibles. El error que estimás es el del modelo "sub-entrenado", no el del modelo final (que se entrena con todos los datos). El bias puede ser de varios puntos porcentuales. |
| **3** | 67% (5000) | 33% (2500) | Mejor que K=2 pero todavía con bias notable. Cada fold ve sólo 2/3 del dataset. |
| **5** | **80% (6000)** | **20% (1500)** | **Sweet spot estándar.** 80% de los datos para entrenar ≈ entrenamiento "honesto" cerca del régimen full-data. Val de 1500 muestras con ~174 positivos por fold → suficientes para que las métricas de cada fold tengan baja varianza muestral. |
| **10** | 90% (6750) | 10% (750) | Bias menor que K=5 (a costa de 2× cómputo), pero los folds de validación se reducen a 750 muestras (~87 positivos) → la varianza por fold sube, y la ganancia en bias es marginal sobre datasets de este tamaño. |
| **N = 7500** (leave-one-out) | 7499 | 1 | Bias mínimo, pero el estimador tiene **varianza enorme** (cada fold cambia una sola muestra) y cuesta `O(N)` entrenamientos → impracticable y peor estadísticamente. |

### 2.2. Por qué K=5 gana en nuestro contexto específico

1. **Tamaño del dataset (N=7500).** Con K=5 cada fold reserva 1500 muestras para validación. Esto da ~174 positivos por fold (suficientes para que Precision/Recall/F1 tengan baja varianza muestral). Con K=10 esos números se parten a la mitad y la métrica por fold se vuelve más ruidosa.

2. **Régimen de entrenamiento.** Con K=5 cada modelo se entrena con 6000 de las 7500 muestras (80%). El estimador del error de generalización corresponde a un modelo entrenado con un dataset apenas más chico que el final (que usa el 100%). El bias del estimador es chico y aceptable.

3. **Clase desbalanceada + estratificación.** Con K=5 estratificado y 869 positivos en el dataset, cada fold de validación tiene **174 positivos** — número estable. Con K=10 quedan 87 por fold, valores chicos donde la varianza muestral de Recall/Precision empieza a dominar la varianza estructural del modelo.

4. **Costo computacional.** K=5 entrena 5 modelos por configuración × 5 seeds = 25 corridas por LR. K=10 duplicaría eso a 50, lo cual ralentiza el barrido completo de LR×seed sin ganar señal real (el bias-reduction de K=5→K=10 es marginal).

5. **Std reportable.** Con 5 folds se puede reportar `mean ± std sobre folds` y tiene 5 muestras, que es el mínimo para que la std no sea ruido puro. Sumado a 5 seeds: 25 estimaciones por configuración → std robusta.

### 2.3. Cuándo elegiríamos otro K

- **N mucho más chico (ej. N=200)**: K=10 o K=20 tendría sentido, porque con K=5 los 40 ejemplos por fold ya no son estadísticamente significativos.
- **N mucho más grande (ej. N=100k)**: K=3 alcanza, porque cada fold sigue teniendo 33k ejemplos de validación — sobra para reducir varianza muestral.
- **Costo computacional altísimo (ej. modelos que tardan horas)**: K=3 o incluso un único holdout repetido con seeds distintas.
- **Búsqueda de hiperparámetros muy fina**: K=5 con muchas seeds (10+) en lugar de subir K.

Ninguno de estos escenarios aplica a nuestro caso → **K=5 es la elección defendible y estándar**.

## 3. Estratificación obligatoria

Adicional a la elección de K, los folds se construyen **estratificados** por `flagged_fraud` (binaria 0/1). Justificación según la clase (00:09:05):

> *"La idea es balancearlo justamente y que te quede al menos alguna muestra de cada clase en el train y alguna muestra de cada clase en el test."*

Con prevalencia del 11.59% de positivos, una partición sin estratificar puede dejar folds con 9% u 13% de positivos por azar — eso introduce ruido en P/R/F1 que no tiene nada que ver con el modelo. La estratificación garantiza que cada fold mantenga la prevalencia global, así la varianza entre folds refleja sólo capacidad del modelo, no azar de muestreo.

Implementado en `linear_perceptron.py:108` (`make_stratified_folds`) y replicado en `nonlinear_perceptron.py:112`.

## 4. Multi-seed sumado a K-fold

K-fold sólo controla la varianza por **partición**. Falta controlar la varianza por **inicialización** (los pesos arrancan en valores aleatorios). Por eso cada combinación se corre con 5 seeds distintas, dando **5 seeds × 5 folds = 25 estimaciones** por configuración. Esto permite separar:

- **Seed-std**: variabilidad atribuible a la inicialización de pesos.
- **Fold-std**: variabilidad atribuible a la partición de datos.

En el experimento, ambas son pequeñas (orden 10⁻³ para MSE), lo que indica que el modelo y la partición son robustos.

## 5. Resumen defendible

> *"Elegimos K=5 estratificado porque es el sweet spot estándar entre bias y varianza del estimador de generalización: con N=7500 cada fold reserva 1500 muestras de validación (~174 positivos con estratificación), suficientes para que las métricas sean estables, mientras que cada modelo se entrena con 6000 muestras (80%) — muy cerca del régimen full-data del modelo final, lo cual minimiza el bias del estimador. K=2 ó K=3 dejaría el modelo sub-entrenado respecto del final; K=10 ó leave-one-out duplicarían o multiplicarían el costo computacional para ganancia marginal en bias, a la vez que aumentarían la varianza por fold. Adicionalmente repetimos cada configuración con 5 seeds para controlar la varianza por inicialización, totalizando 25 estimaciones por punto de hiperparámetro."*
