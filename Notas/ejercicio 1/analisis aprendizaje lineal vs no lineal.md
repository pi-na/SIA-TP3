# Comparación de aprendizaje — lineal vs no-lineal (Ej1)

## Fuentes utilizadas

| Elemento | Archivo |
|---|---|
| Métricas completas a thr* (25 corridas cada modelo) | `lineal_perceptron/analisis_outputs/sweep_lr/multiseed/analisis.md` |
| Métricas completas a thr* (25 corridas no-lineal) | `nonlinear_perceptron/analisis_outputs/sweep_lr/multiseed/analisis.md` |
| Baseline de 3 reglas duras del dataset | `ejercicio1/PRESENTACION_LEER/informe_analisis_dataset.pdf` |
| Plateau / convergencia (tail-slope) | Sección "¿Convergió?" en ambos analisis multiseed |
| Norma de pesos ‖w‖ | Tablas resumen de ambos analisis multiseed |

---

## 1. Tabla comparativa de métricas

Todos los valores son **media ± std sobre 5 seeds × 5 folds = 25 corridas**, evaluados al threshold óptimo de cada modelo (thr* = argmax F1 promedio sobre las 25 corridas).

| Modelo | thr* | MSE test | Accuracy | Precision | Recall | F1 | ‖w‖ |
|---|---|---|---|---|---|---|---|
| Lineal (lr=1e-4) | 0.69 | 0.02651 ± 0.00139 | 0.9736 ± 0.0035 | 0.9100 ± 0.0210 | 0.8573 ± 0.0280 | 0.8825 ± 0.0162 | 0.2027 ± 0.0010 |
| No-lineal (lr=1e-2) | 0.89 | 0.01099 ± 0.00058 | 0.9709 ± 0.0045 | 0.8859 ± 0.0210 | 0.8594 ± 0.0271 | 0.8722 ± 0.0200 | 1.9829 ± 0.0196 |
| **Baseline 3 reglas** | — | — | 97.68% | **100%** | **80%** | **0.889** | — |

**Observación clave:** el no-lineal tiene MSE 2.4× más bajo (0.011 vs 0.026), pero el **lineal tiene F1 más alto** (0.8825 vs 0.8722). Son mejores en cosas distintas: el no-lineal ajusta mejor la probabilidad continua del BigModel; el lineal clasifica mejor el fraude binario.

---

## 2a. ¿Observan underfitting?

**Definición de la clase (PDF métricas/sobreajuste):** underfitting = error alto en training. El modelo no alcanza el error que podría alcanzar con más capacidad.

### Referencia de "error alto": el baseline de 3 reglas

El análisis del dataset encontró que el 80% del fraude está determinado por tres reglas de umbral sobre features enteras (`quantity≥10`, `items_viewed≥15`, `amount>500`). Un sistema puramente determinístico con esas 3 reglas alcanza:

| Precision | Recall | Accuracy | F1 |
|---|---|---|---|
| 100% | 80% | 97.68% | 0.889 |

Este es el **piso de comparación**: si el perceptrón no iguala estas métricas en training, hay underfitting.

### Evidencia en los modelos

**Lineal:**
- F1 = 0.8825 vs baseline 0.889 → brecha de 0.006. Muy cerca pero no alcanza.
- MSE test = 0.0265, MSE train ≈ 0.0264 (sin gap train/test) → el modelo convergió (plateau verificado en el sweep de LR), pero el error de training sigue siendo alto.
- El modelo **tiene underfitting**: la capacidad de un perceptrón lineal (una recta en el espacio de features) no alcanza para representar los saltos discretos de las 3 reglas (quantity=9 → 0% fraude, quantity=10 → 100% fraude). El modelo promedia esa transición y pierde precisión en la zona del umbral.

**No-lineal:**
- F1 = 0.8722 vs baseline 0.889 → brecha de 0.017. Mayor que el lineal.
- MSE test = 0.0110, MSE train ≈ 0.0110 (sin gap train/test). Convergió también a plateau.
- **Paradoja:** menor MSE pero menor F1. La sigmoide satura cerca de 0 y 1, lo que la hace mejor en la tarea de regresión continua (imitar la probabilidad del BigModel) pero no necesariamente en la clasificación binaria.
- También tiene underfitting: la capacidad de un único nodo sigmoide (una curva suave en una dirección del espacio de features) no alcanza para representar la función OR de 3 umbrales del BigModel.

**Conclusión:** ambos modelos tienen underfitting. La evidencia es que el error de training es alto y no baja más — no porque entrenaron poco (ambos llegan a plateau), sino porque **la capacidad de un único perceptrón es insuficiente** para aproximar la función objetivo.

---

## 2b. ¿Observan saturación de las capacidades?

Saturación de capacidad = el modelo llegó al límite de lo que puede aprender con su arquitectura. Tres evidencias convergentes:

### i. Convergencia a plateau (tail-slope ≈ 0)

El análisis de las últimas 50 épocas del sweep multi-seed muestra:

| Modelo | Δ% MSE en últimas 50 épocas |
|---|---|
| Lineal (lr=1e-4) | 0.00e+00 % (ruido float64) |
| No-lineal (lr=1e-2) | 0.00e+00 % (ruido float64) |

Ambos se detienen dentro del ruido numérico de float64. No hay más gradiente disponible: el optimizador no tiene dirección útil en la que moverse.

### ii. MSE no converge a cero — piso de error irreducible

| Modelo | MSE train final |
|---|---|
| Lineal | ~0.026 |
| No-lineal | ~0.011 |

Si hubiera capacidad sin explotar, el MSE train podría seguir bajando con más neuronas o capas. El hecho de que ambos modelos se "peguen" a un piso distinto de cero es señal de que el error residual es irreducible **para esa arquitectura** — no es falta de entrenamiento.

### iii. Estabilidad de ‖w‖ entre seeds

| Modelo | ‖w‖ (mean ± seed-std) |
|---|---|
| Lineal | 0.2027 ± 0.0006 |
| No-lineal | 1.9829 ± 0.0003 |

Seed-std en el orden de 1e-3 o menor: distintas inicializaciones convergen al mismo punto en el espacio de pesos. No hay basin-hopping ni soluciones múltiples — hay una única solución estable a la que llega el modelo.

**Conclusión:** ambos modelos están en capacidad saturada. El no-lineal satura en un MSE más bajo porque la sigmoide tiene más expresividad que la identidad para aproximar la función del BigModel. Pero ninguno puede bajar más sin cambiar la arquitectura.

---

## 2c. ¿Cuál seleccionarían para el estudio de generalización?

El enunciado pide entrenar sobre **todas las muestras** del dataset y estudiar generalización. La pregunta es qué modelo tiene más **potencial de aprendizaje** para justificar ese análisis.

### Argumento para el no-lineal

- **MSE 2.4× más bajo:** captura mejor la probabilidad continua del BigModel sobre todo el dataset.
- **Más expresivo:** la sigmoide puede representar relaciones no lineales entre features y probabilidad de fraude. Entrenar sobre todas las muestras le da más información para explotar esa expresividad.
- **El gap train/test es cero:** el no-lineal no sobreajusta pese a su mayor capacidad (‖w‖ = 2.0 vs 0.20). Esto es exactamente lo que queremos mostrar en un estudio de generalización: modelo más expresivo, misma robustez.
- **Potencial de aprendizaje:** si el dataset creciera, o si hubiera features adicionales, el no-lineal tiene más margen para mejorar que el lineal.

### Argumento para el lineal

- **F1 más alto a thr*:** para la tarea final de clasificar fraude, el lineal clasifica mejor en binario (F1=0.8825 vs 0.8722).
- **Pesos interpretables:** los pesos finales del lineal son directamente los coeficientes de cada feature en la decisión, fáciles de analizar.

### Decisión recomendada: **no-lineal**

El criterio del enunciado es "potencial de aprendizaje". El no-lineal tiene más capacidad expresiva, alcanza menor MSE (la métrica que estamos minimizando), y no sobreajusta. El estudio de generalización sobre todas las muestras es más rico con el no-lineal porque hay más para analizar: se puede mostrar cómo los pesos se estabilizan, cómo el MSE sigue siendo bajo, y cómo el modelo usa su capacidad adicional para ajustar mejor la probabilidad del BigModel sin caer en overfitting.

La paradoja F1 (lineal > no-lineal) es un hallazgo en sí mismo que merece mención en la presentación: **mejor MSE no implica mejor clasificación binaria**. Eso refuerza la importancia de reportar ambas métricas (la loss de entrenamiento y las métricas de clasificación) separadas.

---

## Pendientes para la presentación

- [ ] Plot que superponga las curvas de convergencia (MSE train por época) de ambos modelos en el mismo eje, para visualizar la diferencia en el piso de convergencia.
- [ ] Tabla comparativa unificada lineal / no-lineal / baseline en una sola slide.
- [ ] Justificar explícitamente la paradoja MSE↓ pero F1↓ del no-lineal (sección 2a).
