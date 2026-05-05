# Slide 12 — Sweep de arquitectura (Ej 2, Fase 1)

## Qué significan los números entre corchetes

Cada arquitectura se nota como `[n_in, h1, h2, ..., n_out]`, donde cada número
es la cantidad de neuronas en una capa:

- `784` → input layer (28×28 píxeles aplanados a vector).
- los números intermedios → capas ocultas con activación ReLU.
- `10` → output layer con softmax (una neurona por dígito 0-9).

Ejemplos de la tabla:

| Notación | Capas |
|---|---|
| `[784, 50, 10]` | input + 1 oculta de 50 + output |
| `[784, 100, 10]` | input + 1 oculta de 100 + output |
| `[784, 100, 50, 10]` | input + 2 ocultas (100 → 50) + output |
| `[784, 128, 64, 10]` | input + 2 ocultas (128 → 64) + output |

Más números = más capas y/o más neuronas = más parámetros = más capacidad.

## Procedimiento

Se corrieron las 4 arquitecturas con el resto de hiperparámetros fijos (Adam,
lr=1e-3, batch=16, init He/Xavier según activación, early stopping con
patience=10). Cada corrida usa **un único split de validación**
(k_folds=1, 80/20 estratificado sobre `digits.csv`) — esta es la "Fase 1
exploratoria", barata y rápida para descartar opciones malas.

La métrica de comparación es `val_acc`: accuracy sobre el 20% de hold-out
**que el modelo nunca ve durante backprop**. Es decir: el modelo entrena con
el 80%, y al final de cada epoch evaluamos sobre el 20% restante para medir
generalización dentro de la distribución de `digits.csv`.

`val_acc` también es el criterio del early stopping: si no mejora durante
10 epochs seguidos, se corta el entrenamiento y se reportan los pesos del
mejor epoch.

## Resultados de Fase 1

| Arquitectura | val_acc | best_ep | tiempo |
|---|---|---|---|
| [784, 128, 64, 10] | 96.86% | 10 | 42.3 s |
| **[784, 100, 50, 10]** | **96.74%** | **7** | **27.6 s** |
| [784, 100, 10] | 96.70% | 11 | 35.1 s |
| [784, 50, 10] | 95.78% | 16 | 39.9 s |

## Por qué [784, 100, 50, 10] gana

Las tres arquitecturas más grandes están **dentro del desvío estándar entre
sí** (Δ ≈ 0.16 pp con std ≈ 0.4 pp). En términos prácticos rinden igual: la
diferencia es ruido, no señal.

Cuando varias opciones rinden igual, se aplica el principio de **simplicidad
arquitectónica** (regla de Occam): se elige la más chica y rápida.

`[784, 100, 50, 10]` gana porque:

- Tiene **menos parámetros** que `[784, 128, 64, 10]` (~84k vs ~109k).
- **Converge antes** (best_epoch = 7 vs 10).
- **Entrena 35% más rápido** (27.6 s vs 42.3 s).
- Su accuracy está dentro del ruido de la mejor.

`[784, 50, 10]` queda fuera: 0.96 pp por debajo, lo suficiente como para
considerarla **subajustada** — una sola capa de 50 neuronas no tiene
suficiente capacidad para representar bien las 10 clases.

## Conclusión

La arquitectura ganadora `[784, 100, 50, 10]` se eligió por dar la mejor
relación accuracy / costo computacional, dentro de un grupo de tres
arquitecturas en empate técnico sobre la métrica de validación.
