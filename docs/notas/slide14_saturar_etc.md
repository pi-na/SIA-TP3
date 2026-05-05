# Slide 14 — Nota sobre "saturar" (función de activación)

Comentario originalmente incluido como `insight` en la slide 14
("Función de activación — cuál y por qué") y movido aquí para no recargar
la slide.

## ¿Qué es "saturar"?

Una activación **satura** cuando su pendiente $f'(z)$ se aproxima a $0$
para entradas grandes en magnitud.

Sigmoide y tanh saturan en sus extremos: el gradiente que llega a las
capas previas se desvanece (*vanishing gradient*) y la red deja de
aprender.

## ¿Por qué importa para la elección de activación?

- **Sigmoide / tanh** en capas ocultas profundas → riesgo de vanishing
  gradient. Por eso quedan reservadas para casos puntuales (por ejemplo
  Ej 1, donde el target es probabilidad y la red es de un solo nodo).
- **ReLU** $= \max(0, z)$: derivada $1$ en zona positiva, $0$ en
  negativa. Sin saturación en su lado activo, el gradiente fluye sin
  atenuarse a través de muchas capas. Es la activación estándar moderna
  para capas ocultas (Ej 2 y Ej 3).
- **Softmax** se usa solo en la capa de salida multiclase: convierte
  logits en una distribución de probabilidad sobre las 10 clases.

## Resumen de un párrafo (apto para presentar oralmente)

> Saturación significa que la derivada de la activación se va a cero;
> cuando eso pasa en muchas capas consecutivas el gradiente se desvanece
> y la red deja de aprender. Por eso usamos ReLU en las capas ocultas:
> en su zona activa la derivada vale 1 y el gradiente viaja sin
> atenuarse. Sigmoide y softmax quedan reservadas para las salidas, que
> es donde nos interesa interpretar el resultado como probabilidad.
