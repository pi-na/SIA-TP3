# Estrategias de training

Cómo se recorre el dataset durante el entrenamiento. Tres modos según el **tamaño del batch** (cuántas muestras se usan para calcular cada update de pesos).

---

## Aclaración clave: época ≠ iteración

- **Época**: una pasada completa por TODO el dataset de entrenamiento.
- **Iteración / step**: un update de pesos (un forward + backward + step del optimizador).

El batch size determina **cuántas iteraciones entran en una época**.

---

## Mini-batch (ej. batch=32)

En UNA época recorrés **TODO el dataset**, pero lo partís en pedazos de 32. Si tenés 1000 filas:

1. Shuffleás las 1000 filas.
2. Tomás filas 1–32 → forward → loss promediada sobre esas 32 → backward → **1 update de pesos**.
3. Tomás filas 33–64 → otro update.
4. … y así hasta agotar las 1000.
5. Eso fue **1 época** = ~31 updates (1000/32, redondeando).
6. Próxima época: volvés a shufflear las **mismas 1000 filas** y repetís.

**No es que en la época 2 leas "otras 32 nuevas"** — el dataset es fijo, lo recorrés entero cada época. El batch size sólo dice de a cuántas muestras procesás por update.

---

## Online (batch = 1, SGD puro)

- 1 muestra → 1 update. En una época hacés **N updates** (N = tamaño dataset).
- Gradiente **muy ruidoso** (cada paso usa info de 1 sola fila → estimación pésima del gradiente verdadero).
- Ventaja: muchísimos updates por época; el ruido puede ayudar a escapar mínimos locales.
- Desventaja: no aprovecha vectorización (lento en wall-clock), oscila mucho, difícil de estabilizar.

---

## Full-batch (batch = N)

- Usás **las N filas completas** para calcular UN gradiente promediado → **1 update por época**.
- Gradiente **súper estable** (estimación "verdadera" del gradiente sobre todo el train set).
- Desventaja: 1 sólo update por época → converge lentísimo en wall-clock, y hay que meter todo el dataset en memoria.
- Riesgo: se queda más fácil en mínimos locales / saddle points porque no hay ruido que lo saque.

---

## Comparación rápida

| Modo | Batch | Updates/época | Ruido del gradiente | Vectorización | Memoria |
|---|---|---|---|---|---|
| Online | 1 | N | Alto | Mala | Mínima |
| Mini-batch | 32–256 (típico) | N / batch | Medio | Buena (BLAS/GPU) | Media |
| Full-batch | N | 1 | Nulo | Máxima | Alta |

---

## Por qué mini-batch ganó en la práctica

Combina lo mejor de los dos extremos:

- Gradiente razonablemente estable (promediado sobre 32–256 muestras, no sobre 1).
- Muchos updates por época (no 1 solo como full-batch).
- Aprovecha operaciones matriciales vectorizadas → mucho más rápido en wall-clock que online sample-por-sample.

---

## Conexión con la clase de optimizadores

Cuando hablamos de **SGD / Momentum / Adam** en el TP, en la práctica siempre es **mini-batch SGD**:

- La "S" (stochastic) viene de que el gradiente de cada update es una **estimación ruidosa** del gradiente verdadero (que sería el full-batch).
- Ese ruido **no es un bug, es una feature**: ayuda a escapar zonas malas del paisaje de loss (mínimos locales chatos, saddle points).
- Momentum y Adam justamente intentan **suavizar ese ruido** acumulando información de gradientes pasados, sin perder la ventaja de hacer muchos updates por época.

---

## Implicancias prácticas para el Ej2

- El batch size es un **hiperparámetro a barrer**, igual que el LR. No es algo "que viene fijo".
- Batch chico → más updates por época → puede converger en menos épocas, pero cada época es más lenta en wall-clock y el loss oscila más.
- Batch grande → menos updates por época → loss más suave pero puede necesitar más épocas (o un LR más alto para compensar).
- **Regla heurística**: si subís el batch size, normalmente conviene subir el LR (porque el gradiente es menos ruidoso, podés dar pasos más grandes con seguridad). Esto hay que verificarlo empíricamente en el sweep, no asumirlo.
- En el reporte: aclarar siempre **batch size usado** y, si se compara entre experimentos con distintos batch sizes, tener cuidado de no confundir "convergió más rápido en épocas" con "convergió más rápido en tiempo real".
