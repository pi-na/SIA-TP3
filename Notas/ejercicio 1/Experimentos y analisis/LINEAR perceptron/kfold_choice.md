# Elección de K en K-fold — perceptrón lineal

## 1. Qué controla K (y qué no)

Una confusión frecuente: K-fold no determina qué tan bien entrena el modelo, sino **qué tan confiable es la estimación de su desempeño**.

El perceptrón lineal es un modelo de baja capacidad que converge al mismo mínimo cuadráticos independientemente del tamaño del train. Sobre 7500 filas, la diferencia entre entrenar en 3750 muestras (K=2) o en 6750 (K=10) es marginal. No hay un K que produzca un modelo "mejor" — hay un K que produce una **estimación más o menos confiable** de las métricas.

## 2. El tradeoff al elegir K

El dataset tiene **7500 filas**, de las cuales **869 son positivos (fraude)** — 11.59% de la clase positiva. Cada configuración de K implica:

| K | Filas de train | Filas de test | Positivos en test | Estimados para promediar |
|---|---|---|---|---|
| 2 | 3750 (50%) | 3750 | ~435 | 2 |
| 3 | 5000 (67%) | 2500 | ~290 | 3 |
| **5** | **6000 (80%)** | **1500** | **~174** | **5** |
| 10 | 6750 (90%) | 750 | ~87 | 10 |

Hay dos efectos que se compensan (clase de métricas/sobreajuste):

- **Sesgo del estimador:** con K chico se entrena en menos datos → la estimación del error es pesimista (peor que el error real del modelo). Con K grande se entrena en casi todos los datos → estimación menos sesgada.
- **Varianza del estimador:** con K grande cada fold de test es chico → las métricas por fold son ruidosas (en particular precision/recall, que dependen de cuántos positivos caen en el test). Con K chico los folds de test son grandes → estimaciones por fold más estables, pero pocos estimados para promediar.

## 3. Por qué K=5 es una elección razonable

K=5 equilibra los dos extremos:

1. **80% de datos de entrenamiento por fold** — suficiente para que el perceptrón converja al mismo punto que con el dataset completo.
2. **~174 positivos por fold de test** — suficiente para que precision, recall y F1 sean métricas estables (no dependientes de si cayeron 3 o 4 positivos de más en el fold por azar).
3. **5 estimados para promediar** — reduce el ruido del promedio respecto a K=2 o K=3.
4. **No explotamos el tiempo de cómputo** — 5 entrenamientos por corrida es manejable; K=10 duplica el tiempo sin mejora significativa en la estimación para este tamaño de dataset.

Regla general de la literatura (citada también en el contexto de la clase): K=5 o K=10 son las elecciones estándar para datasets de tamaño medio. Para este problema la diferencia entre K=5 y K=10 es mínima; K=5 gana por parsimonia (menos cómputo, misma confiabilidad).

## 4. Experimento empírico

**Configuración:** LR=1e-4, seed=42, epochs=500, threshold=0.69 (thr* del lineal — el que maximiza F1 promedio en el sweep multi-seed de LR). Script: `ejercicio1/kfold_sweep_linear.py`. Datos crudos en `kfold_sweep/raw.csv`.

**Hipótesis previa:** el MSE medio debería ser similar para todos los K (el modelo converge al mismo lugar). El std entre folds debería crecer al aumentar K pasado cierto punto, porque con folds de test más chicos hay menos positivos por fold y las métricas de clasificación son más ruidosas.

### Resultados (thr*=0.69)

| K     | n_train  | n_test   | Positivos en test | MSE test (mean ± std) | F1 (mean ± std)     |
| ----- | -------- | -------- | ----------------- | --------------------- | ------------------- |
| 2     | 3750     | 3750     | 435               | 0.02686 ± 0.00127     | 0.8820 ± 0.0036     |
| 3     | 5000     | 2500     | 290               | 0.02657 ± 0.00102     | 0.8825 ± 0.0089     |
| **5** | **6000** | **1500** | **174**           | **0.02658 ± 0.00076** | **0.8821 ± 0.0213** |
| 10    | 6750     | 750      | 87                | 0.02665 ± 0.00206     | 0.8809 ± 0.0250     |

### Std entre folds — métrica de estabilidad del estimador

| K | std MSE test | std F1 |
|---|---|---|
| 2 | 0.00127 | 0.0036 |
| 3 | 0.00102 | 0.0089 |
| **5** | **0.00076** | 0.0213 |
| 10 | 0.00206 | 0.0250 |

### Interpretación

**Los medios son prácticamente idénticos para todos los K** (F1 varía entre 0.880 y 0.883, MSE entre 0.0266 y 0.0269 — diferencias menores al propio std). Esto confirma la hipótesis teórica: el perceptrón lineal converge al mismo punto independientemente de cuántos datos tiene en el train de cada fold.

**El std de MSE es mínimo en K=5** (0.00076) y sube en K=10 (0.00206). Subir K no reduce la varianza de la estimación del MSE — la aumenta, porque los folds de test más chicos tienen más ruido.

**El std de F1 crece monótonamente con K.** Esto parece contraintuitivo: K=2 da el F1 más estable (std=0.0036). La razón es el tamaño del fold de test: con K=2 hay ~435 positivos por fold → cada estimación por fold es muy estable. Con K=10 hay sólo ~87 positivos → pequeñas diferencias en cuáles positivos caen en el fold mueven el F1 en ±0.02-0.03. Sin embargo, K=2 sólo produce **2 estimados** para promediar — ante un fold con partición desfavorable, el promedio queda sesgado sin posibilidad de compensar.

**Comparación crítica K=5 vs K=10:** F1 std pasa de 0.0213 a 0.0250 (diferencia de 0.0037, marginal) y MSE std empeora de 0.00076 a 0.00206. No hay ganancia medible en subir a K=10.

# **Conclusión:** 
K=5 minimiza la varianza de estimación del MSE y no es superable por K=10 en F1. K=2 da F1 más estable por fold pero con sólo 2 estimados es demasiado sensible a la suerte de la partición. **K=5 es la elección correcta.**

## 5. Qué reportar en la presentación

- Tabla de tamaños de fold (sección 2) para mostrar el tradeoff concreto en números.
- Argumento del tradeoff sesgo/varianza del estimador (sección 2), anclado en la clase de métricas/sobreajuste.
- Tabla de resultados de la sección 4 — en particular la columna std MSE y std F1 para mostrar que K=10 no mejora K=5.
- Conclusión: **K=5 minimiza la varianza de la estimación** para este dataset (7500 filas, 11.59% positivos). Aumentar a K=10 duplica el cómputo y empeora la estabilidad por folds de test demasiado chicos.
- K=5 con estratificación por `flagged_fraud` → cada fold mantiene el 11.59% de positivos → métricas de clasificación comparables entre folds.
