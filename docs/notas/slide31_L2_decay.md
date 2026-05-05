# Slide 31 — L2 / Weight Decay

## Qué es

**L2** (también llamado *weight decay* o regularización *Ridge*) es una técnica que penaliza pesos grandes en la red.

## Cómo funciona

A la función de costo original le sumás un término extra:

$$L_{total} = L_{original} + \lambda \cdot \sum_i w_i^2$$

Donde:
- `λ` (lambda) es el hiperparámetro que controla cuánto se penaliza. En el Ej 3 ganador: `λ = 1e-4`.
- La suma es sobre todos los pesos de la red (no incluye los bias, típicamente).

## Efecto en el gradiente

Al derivar el término extra respecto a `w`, aparece `2λw`. El update de cada peso queda:

$$w \leftarrow w - \eta \cdot (\nabla L_{original} + 2\lambda w)$$

El término `2λw` "tira" el peso hacia cero en cada update — de ahí el nombre **weight decay**. Cuanto más grande es el peso, más fuerte se lo empuja hacia abajo.

## Por qué regulariza

- Pesos grandes = el modelo le da mucha importancia a features específicas → memoriza patrones del train.
- Pesos chicos = decisiones más "suaves", menos sensibles a un feature individual → mejor generalización.
- Equivale a limitar la **capacidad efectiva** del modelo sin cambiar la arquitectura.

## Por qué funcionó en el Ej 3

A diferencia de dropout, L2 sí ayudó: `+0.20 pp` solo, `+0.52 pp` combinado con augmentation. Razón: L2 actúa sobre **todos** los pesos suavemente, no apaga neuronas. Es una restricción "blanda" que combina bien con augmentation porque ataca un problema diferente:

- **Augmentation** → expande la cobertura efectiva del dataset.
- **L2** → evita que el modelo se ajuste demasiado a los detalles específicos del train.

Por eso son sinérgicos: uno cubre el agujero de datos, el otro evita memorizar lo que sí hay.

## Resultados del sweep

| Config | val K=5 | test | Δ vs base_extra |
|---|---|---|---|
| base_extra_data | 0.9706 | 0.9636 | — |
| + L2 (1e-4) | 0.9758 | 0.9656 | **+0.20 pp** |
| + L2 + aug (ganador) | 0.9760 | 0.9688 | **+0.52 pp** |

## Justificación matemática

La clase no la desarrolla pero la cita: **Goodfellow Cap. 7.1.1** (análisis vía Taylor del costo, Hessiano y autovalores). La intuición formal es que L2 atenúa los pesos en las direcciones donde el Hessiano tiene autovalores chicos (direcciones poco informadas por los datos), dejando intactas las direcciones bien determinadas.
