# Aprendizaje — Perceptrón lineal (Adaline)

## Definición operativa de "aprendizaje" en esta sección

Siguiendo la clase de perceptrón simple 

> *"El error del perceptrón se refiere al error para todo el conjunto de datos. [...] No es lo mismo aprendizaje que generalización."*

En esta sección **no se separa train/test**. El modelo se entrena sobre las N=7500 filas de `fraud_dataset.csv` y se mide el error sobre las mismas N filas. La pregunta a responder es **si el modelo es capaz de ajustar la función objetivo** (no si generaliza). Generalización se aborda en la sección siguiente con K-fold.

## Diseño experimental

| Parámetro       | Valor                                                                             | Justificación                                                                                                          |
| --------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Modelo          | Perceptrón simple, activación identidad (Adaline)                                 | Pregunta del enunciado: "perceptrón simple lineal"                                                                     |
| Target          | `big_model_fraud_probability` (continuo en `[0,1]`)                               | Distillation del BigModel                                                                                              |
| N muestras      | 7500 (todo el dataset)                                                            | No hay split en aprendizaje                                                                                            |
| Features        | 6 (excluidas: `timestamp`, `device_screen_resolution`, `time_since_last_login_s`) | Mismas exclusiones que `baseline.json`                                                                                 |
| Normalización   | z-score sobre el dataset completo (fit-on-all)                                    | Consistente con "sin split"; necesario para que LR sea comparable entre features de escalas distintas                  |
| Regla de update | Online (delta rule): `Δw = η · (z − O) · x`                                       | Como en la clase (perceptrón paso por paso, dato a dato)                                                               |
| Épocas          | 250                                                                               | Verificado empíricamente que la curva se aplana antes de 250 para todos los LR (ver `mse_convergence.png`)             |
| **Epsilon**     | **0.0**                                                                           | **No early-stop** — queremos ver la asíntota completa. Cualquier meseta observada es estructural, no de corte temprano |
| Learning rates  | `{0.1, 0.01, 0.001, 0.0001}`                                                      | Barrido log-uniforme; el "ganador" representa el potencial real del modelo                                             |
| Seeds           | 3 (`42, 43, 44`)                                                                  | Std sobre seeds reporta variabilidad por inicialización                                                                |
| Inicialización  | `uniform(-0.1, 0.1)` para los 7 pesos (6 features + bias)                         | Igual al perceptrón actual del repo                                                                                    |

**Métrica reportada por época (sobre el dataset completo):**

- `MSE_train` — promedio de `(z − O)²`. Es la loss que el Adaline minimiza y la métrica de error de regresión que define la clase de perceptrón simple (transcript 01:22:33: *"el MSE, que es el promedio de las diferencias al cuadrado"*).

**Métricas omitidas por declaración explícita** 

- Accuracy / Precision / Recall / F1 dependen de un threshold cuya **elección óptima es justamente el deliverable de la sección de generalización** (recomendación de umbral a CompanyX). Reportarlas acá con threshold=0.5 forzado sería arbitrario y contaminaría el threshold sweep posterior. Se reportan en la sección de generalización junto con AUC-ROC/AUC-PR sobre folds de validación.

## Resultados

### Resumen final (última época, media ± std sobre 3 seeds)

| LR | MSE_train final (mean ± std sobre 3 seeds) | MSE vs el mejor LR | Comentario |
|---|---|---|---|
| `0.1`    | `0.04455 ± 0.0`     | **+70.6%** | LR demasiado alto → oscila, no converge al mínimo |
| `0.01`   | `0.02791 ± 0.0`     | +6.9%  | Acercándose al óptimo |
| `0.001`  | `0.02622 ± 0.0`     | +0.4%  | Cerca del techo |
| `0.0001` | **`0.02612 ± 0.0`** | base   | **Ganador** — MSE más bajo alcanzado por el modelo |

Lectura de la 3ra columna: cuánto MSE adicional paga ese LR respecto del mejor LR del barrido (`0.04455 / 0.02612 − 1 = +70.6%`). Muestra que el orden de magnitud del LR define el régimen: 10⁻¹ falla, 10⁻²..10⁻⁴ converge al mismo orden de error.

Tabla con todos los datos: `output/aprendizaje_20260511_224303/summary.csv` (resumen por LR) y `epoch_history.csv` (12 corridas × 250 épocas).

### Plots de convergencia

![[output/aprendizaje_20260511_224303/plots/mse_convergence.png]]
MSE(epoch) por LR, banda ± std sobre seeds, eje Y lineal.


![[output/aprendizaje_20260511_224303/plots/final_metrics.png]]
bar chart MSE final por LR, eje Y lineal, error bar sólo si std>0.

## Observación clave: std = 0 entre seeds

Las 3 seeds de cada LR convergen al **mismo MSE final** (std=0 hasta 5 decimales). Esto es **esperado y defendible**:

- La loss del Adaline es **estrictamente convexa** (es un cuadrado de una transformación afín de los datos).
- Hay un único mínimo global, al que **toda inicialización razonable converge** dado suficientes épocas.
- La varianza por seed sólo aparecería si: (a) el problema tuviera múltiples mínimos locales, o (b) el LR fuera tan alto que oscilara estocásticamente.

Es un **argumento defendible para la oral**: no es que "olvidamos correr más seeds", es que el problema garantiza unicidad de solución. Un MLP del Ej2 sí va a tener std > 0 porque la loss deja de ser convexa.

## Interpretación — respuestas a las preguntas del enunciado

### a) ¿Observan underfitting?

**Sí.** La definición de la clase de métricas/sobreajuste:

> *"Básicamente no pude ajustar el problema con este modelo, y caí en underfitting. Ya acá las métricas de tren te van a dar mal."*
>
> *"Si yo evalúo mi algoritmo y tengo un error muy alto en el training set, caí en underfitting."*

Y la causa típica que da Eugenia (00:31:21):

> *"Puede que mi modelo sea demasiado simple para este conjunto de datos."*

con el ejemplo de la clase (00:31:00): *"una regresión lineal no se ajusta cuando no tengo data lineal"*, justamente como el Adaline sobre el target del BigModel.

**Aplicación del criterio a nuestro caso:**

El MSE del Adaline se estanca en `0.02612`. Es dificil juzgar si el MSE es alto:

- El target está acotado a `[0, 1]`, así que el error cuadrático puntual `(z − O)²` también está acotado a `[0, 1]`, y **MSE ∈ `[0, 1]`** sobre este problema.
- En esa escala absoluta, `MSE = 0.026` **no es "alto"**, es mas bien chiquito.

Pero la clase no define "alto" en términos numéricos absolutos: lo define **operacionalmente**, como *"no pude ajustar el problema con este modelo"*. En ese sentido, `0.026` sí es alto, y la evidencia es directa:

1. **El MSE deja de bajar.** Curva aplanada desde la época ~150, sin movimiento posterior. Std=0 entre 3 seeds → no es variabilidad por inicialización.
2. **Probamos las soluciones que la clase propone para mitigar underfitting** (*"ajustar parámetros, entrenar más épocas, más datos"*):
   - **Más épocas**: corrimos con `epsilon=0` y 250 épocas. Verificado que la curva está plana en las últimas ~100. Más épocas no la moverían.
   - **Ajustar hiperparámetros (LR)**: barrimos 4 órdenes de magnitud `{10⁻¹, 10⁻², 10⁻³, 10⁻⁴}`. Los 3 LRs que sí convergen aterrizan en `0.02612–0.02791` (rango 7%). Mejor LR no ayuda.
   - **Más datos**: usamos las 7500 muestras enteras.

1. **El modelo es estructuralmente simple para el problema**, en el sentido literal del ejemplo de la clase: un Adaline implementa `O = w·x + b` — la regresión lineal del ejemplo de Eugenia. El target del BigModel no es lineal en las 6 features (lo sabemos por el análisis del dataset y por las reglas duras del enunciado). Aplica el diagnóstico de la clase: **modelo demasiado simple para los datos**.

**Conclusión:** underfitting confirmado por el criterio operacional de la clase. El error no es alto en sentido numérico, sino en sentido funcional: *"es lo más bajo que el Adaline puede llegar, y no alcanza para ajustar el problema"*.

### b) ¿Observan saturación de las capacidades?

 **Saturación de capacidad del modelo:** la curva de MSE llega a una meseta en `0.02612` que no baja por más épocas, más seeds, ni mejor LR. Evidencia concreta:
   - 3 LRs distintos (`10⁻²`, `10⁻³`, `10⁻⁴`) terminan en MSE dentro de `0.02612–0.02791` — un rango de **7%** entre el peor y el mejor de los LRs que sí convergen.
   - Dentro de cada LR, std sobre 3 seeds = 0.
   - Más épocas sobre el ganador no bajan el MSE.

   La conclusión robusta es que **el Adaline agotó su capacidad expresiva sobre este dataset** en `MSE ≈ 0.026`.

Adicionalmente, **LR=0.1 satura en un piso aún más alto** (`MSE=0.04455`, +70.6% sobre el mejor): no es saturación de capacidad sino **mala optimización**  el LR es tan alto que el peso oscila alrededor del mínimo sin nunca asentarse. 

### c) ¿Cuál seleccionarían para generalización?

El lineal **no** se selecciona. La justificación es directamente sobre MSE:

| Modelo | MSE final (mejor LR) | Std sobre seeds |
|---|---|---|
| Lineal (Adaline)        | `0.02612` | 0.0 |
| No-lineal (sigmoide)    | `0.01095` | 0.0 |

- **Reducción absoluta de MSE** del no-lineal respecto del lineal: `0.02612 − 0.01095 = 0.01517`.
- **Reducción relativa de MSE**: `(0.02612 − 0.01095) / 0.02612 = 58.1%`. El no-lineal logra un MSE **41.9%** del que produce el lineal.
- **Cuán significativa es esa diferencia**: la peor seed del no-lineal y la mejor seed del lineal están separadas por más de `0.015` en MSE; la std intra-modelo es `0` en ambos. Hay un orden de magnitud entre la variabilidad numérica del experimento y la diferencia entre modelos.

La diferencia es **estructural** (no estadística): el no-lineal puede curvar la decisión y respetar el rango `[0,1]` del target; el lineal queda atrapado en hiperplanos rectos y predice fuera de `[0,1]` para parte del dataset (~7%, ver `slides_presentacion.tex:189`). Por capacidad de aprendizaje pura, **el no-lineal es el candidato para generalización**.

## Limitaciones y observaciones técnicas

- **LR=0.1 con z-score**: como se anticipó, este LR está en el borde de divergencia para Adaline con features z-scoreadas. No diverge, pero que oscila y no llega a converger al mismo MSE que las anteriores.
- **LR=0.0001 es el ganador, sin embargo** la mejora de 1e-3 y e-3 MSE pasa de `0.02622` a `0.02612` entre LR=10⁻³ y LR=10⁻⁴ — diferencias del 4to decimal, ya en el régimen de saturación de capacidad.
- **250 épocas suficientes**: el plot muestra que para LR ≥ 10⁻³ la curva está completamente plana desde la época ~50; para LR=10⁻⁴ se aplana hacia la época ~150-200. Defendible.

## Reproducibilidad

```bash
# Sweep:
python ejercicio1/lineal_perceptron/aprendizaje_sweep.py

# Plots (default = último run):
python ejercicio1/lineal_perceptron/plot_aprendizaje.py
```

Config: `ejercicio1/lineal_perceptron/configs/aprendizaje.json`
Wall-clock: **21.4 s** con 8 workers (`ProcessPoolExecutor`, `OMP_NUM_THREADS=1`).
Run usado para este análisis: `output/aprendizaje_20260511_224303/`.
