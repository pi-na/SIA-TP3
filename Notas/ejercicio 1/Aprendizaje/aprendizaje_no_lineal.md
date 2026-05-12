# Aprendizaje — Perceptrón no-lineal (sigmoide)

## Definición operativa de "aprendizaje" en esta sección

Siguiendo la clase de perceptrón simple 

> *"El error del perceptrón se refiere al error para todo el conjunto de datos. [...] No es lo mismo aprendizaje que generalización."*

En esta sección **no se separa train/test**. El modelo se entrena sobre las N=7500 filas de `fraud_dataset.csv` y se mide el error sobre las mismas N filas. La pregunta a responder es **si el modelo es capaz de ajustar la función objetivo** (no si generaliza).

## Diseño experimental

| Parámetro | Valor | Justificación |
|---|---|---|
| Modelo | Perceptrón simple, activación **sigmoide** `σ(z) = 1/(1+e^{-z})` | Pregunta del enunciado: "perceptrón simple no lineal con activación adecuada para el problema" — sigmoide porque el target es probabilidad en `[0,1]` y σ acota la salida al mismo rango |
| Target | `big_model_fraud_probability` (continuo en `[0,1]`) | Distillation del BigModel |
| N muestras | 7500 (todo el dataset) | No hay split en aprendizaje |
| Features | 6 (excluidas: `timestamp`, `device_screen_resolution`, `time_since_last_login_s`) | Mismas exclusiones que el lineal y `baseline.json` |
| Normalización | z-score sobre el dataset completo (fit-on-all) | Crítica para sigmoide: si las features no están en escala, `z = w·x + b` se sale rápido del rango lineal de σ y el modelo entra en saturación de activación |
| Regla de update | Online: `Δw = η · (z − O) · O·(1−O) · x` (delta rule + derivada de σ) | Implementado en `nonlinear_perceptron.py:240-256` |
| Épocas | 250 | Más que para el lineal porque `O·(1−O) ≤ 0.25` atenúa el gradiente — la sigmoide entrena más lento |
| **Epsilon** | **0.0** | **No early-stop** — queremos ver la asíntota completa. Cualquier meseta observada es estructural, no de corte temprano |
| Learning rates | `{0.1, 0.01, 0.001, 0.0001}` | Barrido log-uniforme; el "ganador" representa el potencial real del modelo |
| Seeds | 3 (`42, 43, 44`) | Std sobre seeds reporta variabilidad por inicialización |
| Inicialización | `uniform(-0.1, 0.1)` para los 7 pesos | Pesos chicos → `z` chico → σ arranca cerca de su zona lineal (`σ'(0) = 0.25`), evitando saturación de activación al inicio |

**Métrica reportada por época (sobre el dataset completo):**

- `MSE_train` — promedio de `(z − σ(w·x))²`. Es la loss que minimiza este perceptrón.

## Resultados

### Resumen final (última época, media ± std sobre 3 seeds)

| LR       | MSE_train final (mean ± std sobre 3 seeds) | MSE vs el mejor LR | Comentario                                                              |
| -------- | ------------------------------------------ | ------------------ | ----------------------------------------------------------------------- |
| `0.1`    | `0.01100 ± 0.0`                            | +0.5%              | Llegó al piso                                                           |
| `0.01`   | `0.01096 ± 0.0`                            | +0.1%              | Llegó al piso                                                           |
| `0.001`  | **`0.01095 ± 0.0`**                        | base               | **Ganador** (por márgenes mínimos)                                      |
| `0.0001` | `0.01196 ± 0.00002`                        | +9.2%              | NO llega a un piso en 250 épocas — gradiente muy chico, le faltó tiempo |

Lectura: los tres LRs altos (`10⁻¹..10⁻³`) caen dentro de un rango de **0.5%** entre sí , convergen en un mismo piso `~0.01096`. El único LR fuera del interrvalo es `10⁻⁴`, y el +9.2% de su MSE es transitorio, con mas epocas probablemente llegaba al mismo piso.

Tabla con todos los datos: `output/aprendizaje_20260511_224304/summary.csv` (resumen por LR) y `epoch_history.csv` (12 corridas × 250 épocas).

### Plots de convergencia

![[output/aprendizaje_20260511_224304/plots/mse_convergence.png]] — MSE(epoch) por LR, banda ± std sobre seeds, eje Y lineal.

![[output/aprendizaje_20260511_224304/plots/final_metrics.png]] — bar chart MSE final por LR, eje Y lineal, error bar sólo si std>0.

## Observación clave: std ≈ 0 entre seeds (con un asterisco)

Para LR ≥ 10⁻³ la std sobre 3 seeds es **exactamente 0** (5 decimales). Sólo LR=10⁻⁴ muestra std = `2 × 10⁻⁵` — y eso es porque a las 250 épocas **todavía está en transitorio**, así que captura variabilidad de "qué tan rápido va" cada seed, no de "a qué piso converge".

Por qué da std=0 al haber saturado:
- La loss `MSE = mean((z − σ(w·x))²)` no es estrictamente convexa en `w` (la sigmoide la rompe), pero **es razonable que tenga un único mínimo dominante** sobre este dataset, y todos los seeds convergen a él.

## Interpretación — respuestas a las preguntas del enunciado

### a) ¿Observan underfitting?

**Sí, pero más leve que en el lineal.** Aplicamos la definición de la clase de métricas/sobreajuste (transcript 00:32:07 y 00:40:43):

> *"Básicamente no pude ajustar el problema con este modelo, y caí en anderfitting. Ya acá las métricas de tren te van a dar mal."*
>
> *"Si yo evalúo mi algoritmo y tengo un error muy alto en el training set, caí en underfitting."*

y la causa típica (00:31:21): *"puede que mi modelo sea demasiado simple para este conjunto de datos"*.

**Aplicación a nuestro caso:**

El MSE del no-lineal se estanca en `0.01095`. Aclaración sobre la escala antes de hablar de "alto":

- Target acotado a `[0, 1]`, error cuadrático puntual en `[0, 1]`, **MSE ∈ `[0, 1]`** sobre este problema.
- `MSE = 0.011` es chico en valor absoluto sobre esa escala.

Como en el lineal, la clase no define "alto" en sentido numérico absoluto — lo define **operacionalmente** como *"no pude ajustar el problema con este modelo"*. Aplicamos el mismo criterio:

1. **El MSE deja de bajar.** Curva aplanada desde la época ~100 para los 3 LRs que convergen (`10⁻¹, 10⁻², 10⁻³` aterrizan en `0.01100, 0.01096, 0.01095` — rango de 0.5%). Std=0 entre 3 seeds.
2. **Probamos las palancas que la clase propone** (00:31:26 → 00:31:33: *"ajustar parámetros, entrenar más épocas, más datos"*):
   - **Más épocas**: 250 con `epsilon=0`. Verificado que la curva está plana en los 3 LRs altos.
   - **Ajustar hiperparámetros (LR)**: 4 órdenes de magnitud barridos. El ganador (`10⁻³`) está en el interior del rango — no es un óptimo "pegado al borde". LR=10⁻⁴ no alcanza el piso por falta de épocas, pero los 3 LRs altos coinciden en `~0.01096`.
   - **Más datos**: ya usamos las 7500 muestras enteras (no hay split en aprendizaje).
3. **El modelo sigue siendo estructuralmente simple, sólo que menos.** Un perceptrón simple no-lineal implementa `O = σ(w·x + b)` — sigue siendo **una única dirección de discriminación** en el espacio de features, sólo que pasada por una no-linealidad monotónica. No puede modelar interacciones entre features ni regiones de decisión no convexas. Es menos simple que un Adaline, pero todavía es "demasiado simple" en el sentido de la clase: el MSE no baja a cero porque el target del BigModel no es expresable como `σ` de una combinación lineal de las 6 features.

**Por qué "más leve" que en el lineal:**

El MSE residual del no-lineal (`0.01095`) es **menor** que el del lineal (`0.02612`) sobre los mismos datos con la misma loss y los mismos seeds. Eso indica que la "simplicidad estructural" del no-lineal es **menos limitante** que la del lineal para este problema. La sigmoide agrega capacidad expresiva suficiente para capturar parte de la no-linealidad del target — pero no toda. Underfitting residual: la familia `{ σ(w·x + b) }` no agota la expresividad del target. Un MLP (Ej2) tendría más margen.

**Conclusión:** underfitting confirmado pero de menor magnitud — el modelo no puede bajar más el error y sigue siendo "demasiado simple" en el sentido de la clase, sólo que la limitación estructural es más leve que la del Adaline.

### b) ¿Observan saturación de las capacidades?


**Saturación de capacidad: el MSE toca un piso en `~0.0110` que no baja por más épocas ni mejor LR. La familia `{ σ(w·x + b) : w ∈ ℝⁿ, b ∈ ℝ }` agotó su capacidad expresiva. Es el techo que el "perceptrón simple no-lineal" puede alcanzar; ir más abajo requiere un MLP. Esto **es** lo que el enunciado pregunta.

2. **LR=10⁻⁴ NO saturó** — no es saturación, es **falta de épocas**: con `lr=10⁻⁴` y el factor `O·(1−O) ≤ 0.25` en la regla de update, cada paso es minúsculo y 250 épocas son insuficientes para llegar al piso. Importante distinguirlo en la defensa: "la curva de LR=10⁻⁴ todavía está bajando al final del entrenamiento — extrapolando, llegaría al mismo piso pero a otra escala de tiempo".

### c) ¿Cuál seleccionarían para generalización?

**El no-lineal (sigmoide)** es el modelo seleccionado para el estudio de generalización por su mayor capacidad de aprendizaje.

**Comparativo defendible (sólo MSE):**

| | Lineal (Adaline) | No-lineal (sigmoide) |
|---|---|---|
| MSE final (mejor LR)        | `0.02612` | `0.01095` |
| Std sobre 3 seeds           | 0.0       | 0.0       |
| Reducción absoluta de MSE   | — | `−0.01517` |
| Ratio MSE_no_lineal / MSE_lineal | — | `0.419` |
| Reducción relativa de MSE   | — | **−58.1%** |

El no-lineal logra un MSE que es el **41.9%** del MSE del lineal. La reducción es ~140× la std intra-modelo (que es 0): la diferencia entre modelos domina por completo cualquier ruido experimental.

La diferencia es **estructural** (std=0 para ambos en el régimen saturado): el no-lineal puede curvar la decisión y respetar el rango `[0,1]` del target; el lineal queda atrapado en hiperplanos rectos y predice fuera de `[0,1]` para parte del dataset (~7%, ver `slides_presentacion.tex:189`).

## Limitaciones y observaciones técnicas

- **LR=10⁻⁴ no convergió en 250 épocas** — esto se ve claramente en el plot. Para el ganador "real" del barrido, miramos LR ∈ {10⁻¹, 10⁻², 10⁻³}, donde los tres están empatados en `MSE ≈ 0.01096` (saturaron, rango de variación entre ellos de 0.5%). El "ganador" reportado (LR=10⁻³) es marginalmente mejor por cuatro decimales.
- **El ganador NO está en el extremo del rango** (es 10⁻³, no 10⁻¹ ni 10⁻⁴), lo cual valida que el rango del barrido es razonable y no estamos perdiendo un óptimo afuera.
- **La sigmoide es la elección natural para distillation de probabilidad**: el output del BigModel está en `[0,1]`, σ(z) también está en `[0,1]`, así que estructuralmente la familia de hipótesis está bien alineada con el target. Una tanh `(-1,1)` requeriría re-escalar el target; 
- **Training mode = online**: como en la clase y consistente con el lineal. Verificable en `nonlinear_perceptron.py:240-256`.
- **Sigmoide implementada de forma numéricamente estable** (`nonlinear_perceptron.py:188-200`, rama positiva vs negativa según signo de `z`) — evita `exp(z)` para `z > 0` grande, que daría overflow.

## Reproducibilidad

```bash
# Sweep:
python ejercicio1/nonlinear_perceptron/aprendizaje_sweep.py

# Plots (default = último run):
python ejercicio1/nonlinear_perceptron/plot_aprendizaje.py
```

Config: `ejercicio1/nonlinear_perceptron/configs/aprendizaje.json`
Wall-clock: **57.6 s** con 8 workers (`ProcessPoolExecutor`, `OMP_NUM_THREADS=1`).
Run usado para este análisis: `output/aprendizaje_20260511_224304/`.
