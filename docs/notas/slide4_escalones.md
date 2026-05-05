# Slide 4 — Escalones del dataset y por qué la sigmoide los captura

## Lo que muestra el gráfico

Para cada una de las tres features con regla determinística (`amount_usd > 500`,
`quantity_purchased >= 10`, `items_viewed_before_purchase >= 15`), graficamos
`big_model_fraud_probability` vs el valor de la feature, junto con la mediana
del target por bin (línea negra) y el umbral de la regla (línea roja).

La forma que aparece es una **S**:

- **Piso bajo** a la izquierda del umbral: target ≈ 0.2-0.3.
- **Subida brusca** cerca del umbral.
- **Techo plano en 1.0** a la derecha del umbral.

Verificación numérica con `amount_usd`:

| amount_usd | n | mediana(target) |
|---|---|---|
| [0, 100) | 4886 | 0.241 |
| [100, 300) | 2114 | 0.646 |
| [300, 500) | 287 | 0.975 |
| [500, 700) | 96 | 1.000 |
| [700, 1500) | 106 | 1.000 |

## Por qué un perceptrón lineal no puede

Un lineal en 1 feature ajusta `O = w·x + b` — una recta. No puede a la vez:
mantenerse en 0.2 a la izquierda, subir entre 100 y 500, y quedarse plano en
1.0 a la derecha. La mejor recta promedia la nube y termina prediciendo fuera
de [0, 1] en los extremos (verificado: ~7% del dataset cae fuera del rango,
con predicciones entre -0.44 y +2.32).

## El matiz importante: una sola S, no tres

Un perceptrón sigmoide en R⁶ NO ajusta una S por feature. Computa:

```
z = w₁·x₁ + w₂·x₂ + ... + w₆·x₆ + b      (un escalar)
O = sigmoid(z)                            (una S aplicada a ese escalar)
```

Las 6 dimensiones se colapsan en **un solo número** vía la combinación lineal,
y la sigmoide se aplica una sola vez a ese número. El output es **una S a lo
largo de la dirección `w`**, no seis S superpuestas.

## Por qué funciona igual

Las tres S marginales están **alineadas en el mismo sentido**: las tres reglas
son "feature alta → fraude probable". El perceptrón les pone pesos positivos
y `w·x` actúa como un **score combinado**: cuando alguna de las tres dispara
fuerte, `z` se vuelve grande y la sigmoide satura cerca de 1; cuando ninguna
dispara, `z` queda bajo y satura cerca de 0.

Es un OR aproximado: la sigmoide del score combinado se parece a "alguna de
las reglas se activó".

## El límite que esto impone

Como hay **una sola dirección** y **una sola S**, el perceptrón sigmoide:

- Captura bien la forma global (MSE 0.011 vs 0.027 del lineal, -59%).
- **No** modela perfectamente el OR de tres umbrales independientes. Un OR
  exacto requeriría tres "umbrales" en direcciones distintas — eso lo hace un
  MLP, donde cada neurona oculta arma una S en una dirección diferente y la
  capa de salida las combina.

Es el mismo argumento por el cual XOR no se resuelve con un perceptrón
simple: si las regiones del espacio no se separan con un único corte
lineal, una sola S no alcanza.

## Lo que se gana respecto del lineal: descomposición del MSE

| | MSE | Reducción |
|---|---|---|
| Lineal sin tocar | 0.0266 | — |
| Lineal clipeado a [0,1] post-hoc | 0.0209 | -22% |
| Sigmoide | 0.0110 | **-59%** |

De los 59 puntos de mejora, ~22 vienen de evitar predicciones fuera de [0,1]
y ~37 vienen de la capacidad de la sigmoide de modelar la forma S del target.
La justificación "el lineal predice fuera de rango y MSE penaliza eso" es
cierta pero **incompleta**: el grueso de la mejora es por la curvatura, no
por el clipping.
