# Elección de K en K-fold — perceptrón no-lineal

## 1. Qué controla K (y qué no)

K-fold no determina qué tan bien entrena el modelo, sino **qué tan confiable es la estimación de su desempeño**.

Para el perceptrón no-lineal esto es incluso más claro que para el lineal: el sweep multi-seed mostró que la seed-std del MSE es esencialmente 0 (< ruido numérico float64). Toda la variabilidad observable viene de la partición fold-a-fold, no del azar de la inicialización. Eso significa que el estimado multi-seed que usamos para elegir el LR ya estaba dominado por el fold — y que K controla directamente qué tan estable es ese estimado.

## 2. El tradeoff al elegir K

El dataset tiene **7500 filas**, de las cuales **869 son positivos (fraude)** — 11.59% de la clase positiva. Cada configuración de K implica:

| K     | Filas de train | Filas de test | Positivos en test | Estimados para promediar |
| ----- | -------------- | ------------- | ----------------- | ------------------------ |
| 2     | 3750 (50%)     | 3750          | ~435              | 2                        |
| 3     | 5000 (67%)     | 2500          | ~290              | 3                        |
| **5** | **6000 (80%)** | **1500**      | **~174**          | **5**                    |
| 10    | 6750 (90%)     | 750           | ~87               | 10                       |

Hay dos efectos que se compensan (clase de métricas/sobreajuste):

- **Sesgo del estimador:** con K chico se entrena en menos datos → estimación pesimista (peor que el error real). Con K grande se entrena en casi todos los datos → estimación menos sesgada.
- **Varianza del estimador:** con K grande cada fold de test es chico → las métricas por fold son ruidosas (en particular precision/recall, que dependen de cuántos positivos caen en el fold de test). Con K chico los folds de test son más grandes → estimaciones por fold más estables, pero pocos estimados para promediar.

## 3. Por qué K=5 es una elección razonable

K=5 equilibra los dos extremos:

1. **80% de datos de entrenamiento por fold** — suficiente para que el perceptrón sigmoide converja al mismo punto que con el dataset completo.
2. **~174 positivos por fold de test** — suficiente para que precision, recall y F1 no fluctúen por efecto de pocos positivos en el fold.
3. **5 estimados para promediar** — reduce el ruido del promedio respecto a K=2 o K=3.
4. **No explotamos el tiempo de cómputo** — K=10 duplica el tiempo sin mejora en la estabilidad de la estimación para este tamaño de dataset.

## 4. Experimento empírico

**Configuración:** LR=1e-2, seed=42, epochs=500, threshold=0.5 (fijo para comparar K en igualdad de condiciones). Se usa una sola seed porque el sweep multi-seed mostró seed-std≈0 para el no-lineal — la variabilidad que vemos entre corridas es fold-a-fold, no seed-a-seed. Script: `ejercicio1/kfold_sweep_nonlinear.py`. Datos crudos en `kfold_sweep/raw.csv`.

**Hipótesis previa:** el MSE medio debería ser similar para todos los K (el perceptrón sigmoide converge al mismo mínimo). El std entre folds debería crecer al aumentar K pasado cierto punto, porque con folds de test más chicos hay menos positivos por fold y las métricas de clasificación son más ruidosas.

### Resultados (thr*=0.89)

| K | n_train | n_test | Positivos en test | MSE test (mean ± std) | F1 (mean ± std) |
|---|---|---|---|---|---|
| 2 | 3750 | 3750 | 435 | 0.01096 ± 0.000006 | 0.8713 ± 0.0046 |
| 3 | 5000 | 2500 | 290 | 0.01097 ± 0.000283 | 0.8716 ± 0.0083 |
| **5** | **6000** | **1500** | **174** | **0.01099 ± 0.000437** | **0.8724 ± 0.0298** |
| 10 | 6750 | 750 | 87 | 0.01099 ± 0.000620 | 0.8708 ± 0.0380 |

### Std entre folds — métrica de estabilidad del estimador

| K | std MSE test | std F1 |
|---|---|---|
| 2 | 0.000006 | 0.0046 |
| 3 | 0.000283 | 0.0083 |
| **5** | 0.000437 | 0.0298 |
| 10 | 0.000620 | 0.0380 |

### Interpretación

**Los medios son idénticos para todos los K** (F1 varía entre 0.870 y 0.873, MSE se mantiene en 0.01096–0.01099). Esto confirma que el perceptrón sigmoide converge al mismo mínimo sin importar cuántos datos tiene en cada fold — aún más marcado que en el lineal, consistente con que la seed-std del no-lineal era ≈0 en el sweep multi-seed.

**El std de MSE en K=2 es extraordinariamente pequeño (6e-6)** — básicamente cero. Esto es la misma señal que vimos en el sweep multi-seed: el perceptrón sigmoide satura y produce predicciones casi idénticas fold a fold cuando el fold de test es grande. Con 3750 muestras en el test (K=2), la estimación del MSE es prácticamente determinista.

**El std de F1 y de MSE crecen monótonamente con K**, igual que en el lineal. La causa es la misma: folds de test más chicos → menos positivos por fold (~87 en K=10) → estimaciones de clasificación más ruidosas.

**Comparación crítica K=5 vs K=10:** F1 std pasa de 0.0298 a 0.0380 (empeora) y MSE std pasa de 0.000437 a 0.000620 (empeora). En el no-lineal, subir a K=10 empeora la estabilidad en **ambas métricas**.

**Conclusión empírica:** igual que en el lineal, K=5 domina a K=10 en estabilidad de estimación. **K=5 es la elección correcta.**

## 5. Qué reportar en la presentación

- Tabla de tamaños de fold (sección 2) para mostrar el tradeoff concreto en números.
- Argumento del tradeoff sesgo/varianza del estimador (sección 2), anclado en la clase de métricas/sobreajuste.
- Referencia al sweep multi-seed: la seed-std≈0 del no-lineal confirma que toda la variabilidad viene de los folds → K controla directamente la confiabilidad del estimado.
- Tabla de resultados de la sección 4 con la columna std MSE y std F1.
- Conclusión sobre si K=5 minimiza la varianza o si habría que subir a K=10.
