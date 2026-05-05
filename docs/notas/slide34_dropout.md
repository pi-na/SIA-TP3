# Slide 34 — Dropout

## Qué es

**Dropout** es una técnica de regularización: durante el entrenamiento, en cada forward pass se "apaga" aleatoriamente una fracción `p` de las neuronas de una capa (se las pone a cero). En el siguiente batch se elige otro subconjunto al azar.

## Qué logra

- La red no puede depender de neuronas específicas → se ve forzada a aprender representaciones redundantes.
- Equivale aproximadamente a entrenar un ensamble de muchas sub-redes que comparten pesos.
- Reduce la co-adaptación entre neuronas (que una neurona "espere" un valor específico de otra).

## Detalle de implementación

- **En training:** cada neurona se mantiene con probabilidad `1-p` y se escala por `1/(1-p)` (inverted dropout) para que la magnitud esperada de la activación no cambie.
- **En inferencia:** dropout se desactiva, todas las neuronas activas, sin escala.

## Hiperparámetro típico

`p = 0.2` a `0.5`. En el Ej 3 probamos `p = 0.2`.

## Por qué falló en el Ej 3

Dropout ataca overfitting por **varianza del modelo** (la red memoriza el train). Nuestro problema no era ese — era **cobertura del dataset** (faltaban 8s, pocos 5s). Apagar neuronas no ayuda si lo que falta es información sobre una clase que el modelo nunca vio bien.

Por eso dropout mejoraba val (donde el K-fold no exponía el agujero de la clase 8) pero empeoraba test:

| Config | val K=5 | test |
|---|---|---|
| base_extra_data | 0.9706 | 0.9636 |
| + dropout (p=0.2) | 0.9750 | 0.9628 (**−0.08 pp**) |
| + L2 + dropout | 0.9772 | 0.9604 (**−0.32 pp**) |
| + L2 + aug (ganador) | 0.9760 | 0.9688 (**+0.52 pp**) |

**Conclusión:** dropout es una herramienta para overfitting clásico, no para problemas de cobertura. Combinar dropout con L2 + augmentation incluso degrada el test → sobre-restricción.
