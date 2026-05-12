# Estrategias de training: online vs mini-batch vs batch

Cómo se recorre el dataset durante el entrenamiento. Tres modos según **cuántas muestras se usan para calcular cada update de pesos**. Esto sale tal cual de la clase: el tema se introduce en *Perceptrón Simple* (online) y se completa en *Perceptrón Multicapa* (las tres variantes).

---

## 0. Aclaración previa: época ≠ iteración

- **Época**: una pasada completa por TODO el dataset de entrenamiento.
- **Iteración / step**: un update de pesos (un forward + backward + step del optimizador).

El batch size es lo que **determina cuántas iteraciones entran en una época**. Es el único parámetro que distingue los tres modos.

---

## 1. La intuición (explicación de la clase)

La pregunta de fondo es muy simple: **¿después de cuántos ejemplos actualizo los pesos?**

### Online / incremental (lo que vimos primero, en Perceptrón Simple)

> *"cuando yo hago esta actualización fíjense que la estoy haciendo cada vez que veo un dato nuevo, o sea yo arranco una época, veo un dato, me fijo si al perceptron lo hizo bien y si no lo hizo bien, actualizo los pesos. Y así con el siguiente dato. (...) No es que estoy acumulando datos. Fíjense que el dato número 2 que yo veo ya está basándose o agarrando el perceptron original."* — Perceptrón Simple, [01:24:32 – 01:25:12]

Es decir: **veo un ejemplo → calculo el error → actualizo los pesos → paso al siguiente**. Es el esquema más natural si uno piensa "estoy aprendiendo de a un caso". Es un caso particular de **SGD (Stochastic Gradient Descent)**.

Analogía: un profesor que corrige cada ejercicio del alumno apenas lo termina, y le da feedback inmediato. El alumno cambia su forma de pensar después de cada ejercicio. Va a ser muy "errático" — un ejercicio raro lo manda al pasto.

### Batch / lote (el extremo opuesto)

> *"Lote o Batch vienen a ser el extremo opuesto de incremental u online. (...) cuando calcule los delta W, la fórmula de actualizar los pesos para todos los elementos del conjunto de datos y los unifico y hago una actualización de los pesos de la red. En lugar de constantemente estar actualizando todos los pesos, que eso es una operación que tiene su costo a medida que la red se vuelve más grande."* — Perceptrón Multicapa, [00:47:49 – 00:48:24]

Es decir: **paso TODOS los ejemplos del dataset → promedio el gradiente sobre todos → un solo update por época**.

Analogía: el profesor espera a que el alumno termine los 1.000 ejercicios del libro, los corrige todos, y recién al final le dice "en promedio fallás acá, cambiá esto". El feedback es súper confiable (está basado en 1.000 casos), pero el alumno tarda muchísimo en mejorar porque sólo lo recibe una vez por libro.

### Mini-batch / mini-lote (el del medio, el que se usa)

> *"Hay otras 2 variantes que en realidad se usan más, particularmente la del medio (...) lo que se suele hacer se llama mini lote o mini batch, donde yo tengo de todo mi conjunto de datos, supongo que tengo 10.000, digo, bueno, voy a separar estos 10.000 en chunks de 1.000 datos. Y para cada uno de esos 1.000 datos es que hago la actualización de los pesos y el bias. O sea, calculo el gradiente para esos 1.000, actualizo y después paso al siguiente batch."* — Perceptrón Multicapa, [00:47:40 – 00:48:58]

Es decir: parto el dataset en pedazos chicos (32, 64, 128 muestras), y para cada pedazo hago un update. **Es un compromiso explícito entre los dos extremos.**

Analogía: el profesor corrige tandas de 32 ejercicios y al final de cada tanda da feedback. El alumno mejora rápido (recibe feedback muchas veces por libro) y cada feedback está basado en suficientes ejemplos como para no ser disparatado.

> Nota del profe en la clase: *"Hay otras 2 variantes que en realidad se usan más"*. La cátedra recomienda explícitamente apartarse del esquema online del Perceptrón Simple cuando se pasa a redes más grandes.

---

## 2. Cómo se ve cada modo en una época

Imaginemos un dataset de 1.000 filas.

| Modo | Batch | Updates por época | Qué hace en 1 época |
|---|---|---|---|
| Online | 1 | 1.000 | Mira 1 fila → update. Mira la siguiente → update. ... 1.000 updates |
| Mini-batch | 32 | ~31 | Shuffle. Toma filas 1–32 → 1 update. Toma 33–64 → 1 update. ... 31 updates |
| Full-batch | 1.000 | 1 | Mira las 1.000 filas → promedia el gradiente → 1 solo update |

**Detalle importante** que tiende a confundir: en mini-batch, en la época 2 NO se leen "32 datos nuevos". El dataset es fijo. Lo que cambia es que se vuelve a shufflear las mismas 1.000 filas y se recorre entero otra vez en chunks de 32. El batch size sólo dice "de a cuántas muestras proceso por update", no "cuántas muestras conoce el modelo".

---

## 3. Trade-offs (qué se gana y qué se pierde)

| Aspecto | Online (b=1) | Mini-batch (b=32–256) | Full-batch (b=N) |
|---|---|---|---|
| Ruido del gradiente | Altísimo | Medio | Nulo |
| Updates por época | Muchos (N) | Bastantes (N/b) | Uno solo |
| Vectorización (BLAS/NumPy) | Mala — operaciones sobre vectores 1×d | Buena — matrices b×d | Máxima, pero subutilizada |
| Wall-clock por época | Lento (loop Python) | Rápido | Rápido por época, pero épocas que rinden poco |
| Memoria | Mínima | Media | Alta (todo el dataset en RAM como matriz) |
| Riesgo de mínimos locales | Bajo (el ruido lo saca) | Bajo-medio | Alto (no hay nada que lo saque de un punto de gradiente cero) |
| Estabilidad de la curva de loss | Oscila feo | Suave con ruido fino | Súper suave |

**El ruido del gradiente NO es un bug, es una feature**: ayuda a escapar zonas planas del paisaje de loss (mínimos locales chatos, saddle points). Por eso full-batch, a pesar de tener el gradiente "verdadero", suele converger peor en redes profundas.

---

## 4. Conexión con la clase de optimizadores

Cuando en el TP usamos **SGD / Momentum / Adam**, en la práctica siempre estamos hablando de **mini-batch SGD** (online y full-batch son casos límite de la misma fórmula, con b=1 y b=N respectivamente):

- La "S" (stochastic) de SGD viene precisamente de que el gradiente de cada mini-batch es una **estimación ruidosa** del gradiente verdadero (el full-batch).
- Momentum y Adam acumulan estadísticas de gradientes pasados (media móvil de g y de g²). Eso tiene sentido **porque hay ruido para promediar**. Si usás full-batch, el "ruido" desaparece y Adam pierde gran parte de su gracia: se vuelve casi un gradient descent con LR adaptativo por parámetro.

Es decir: la elección de mini-batch no es independiente del optimizador. Adam, que es nuestro default en Ej2, **asume implícitamente mini-batch**.

---

## 5. Por qué en Ej2 fuimos DIRECTO a mini-batch

Tres razones, todas justificables desde la clase:

### (a) Recomendación explícita de la cátedra

En Perceptrón Multicapa el profe dice literalmente *"Hay otras 2 variantes que en realidad se usan más, particularmente la del medio"*. Online se mostró sólo como punto de partida pedagógico en Perceptrón Simple, no como esquema a usar en la práctica con redes multicapa. Probar online en Ej2 sería retroceder a un esquema que la clase ya marcó como inferior cuando la red crece.

### (b) Costo computacional del Ej2 lo hace inviable de otra forma

Nuestro dataset es `digits.csv` con ~12.450 muestras × 784 features, red `[784, 100, 50, 10]`. Lo que dice la clase sobre el costo de updates aplicaba a redes "grandes" — la nuestra ya entra en esa categoría:

- **Online (b=1)**: 12.450 forward+backward por época, cada uno con matrices microscópicas (1×784, 1×100, …). NumPy/BLAS funcionan bien con matrices grandes, pésimo con vectores chicos en loop Python. Una sola corrida tardaría minutos o más; multiplicalo por 5 folds × N seeds × M configs del sweep y se vuelve impagable.
- **Full-batch (b=12.450)**: cabe en RAM, pero te da **1 update por época**. Adam con 1 update por época necesitaría cientos/miles de épocas para converger en problemas de esta complejidad. Además, como dije en §4, Adam pierde sentido sin ruido.
- **Mini-batch (b=16 en `base.json`)**: ~780 updates por época. Cada update es un matmul `16×784 → 16×100`, que NumPy hace en microsegundos. Wall-clock razonable, vectorización aprovechada, ruido en la cantidad justa para que Adam tenga algo que suavizar.

### (c) El TP exige experimentación con muchos hiperparámetros

CLAUDE.md y la consigna piden barrer LR, arquitectura, optimizador, etc. Para que un sweep one-at-a-time con 5 folds × varias seeds sea viable en un horizonte humano, **cada corrida individual tiene que ser barata**. Mini-batch es la única de las tres opciones que cumple eso sin sacrificar calidad de convergencia.

> Implicancia: el batch size en sí **sigue siendo un hiperparámetro a barrer** (16 vs 32 vs 64 vs 128, p. ej.), pero el barrido se hace dentro de la familia mini-batch, no comparándola contra online ni full-batch.

---

## 6. Implicancias prácticas para el reporte / defensa oral

- **Aclarar siempre el batch size usado** en cada experimento. No es un detalle: cambia la cantidad de updates por época, el ruido del gradiente y la interacción con el optimizador.
- Batch chico → más updates por época → puede converger en menos épocas, pero cada época es más lenta en wall-clock y la curva de loss oscila más.
- Batch grande → menos updates por época → curva de loss más suave, pero puede necesitar más épocas (o un LR más alto para compensar).
- **Regla heurística** (la clase la insinúa, hay que verificarla empíricamente): si subís el batch, conviene subir el LR — el gradiente es menos ruidoso, podés dar pasos más grandes con seguridad. NO asumirlo, **medirlo en el sweep**.
- Si comparamos entre experimentos con distinto batch size, hay que distinguir entre "convergió más rápido en épocas" y "convergió más rápido en wall-clock" — son métricas distintas y se pueden contradecir.
- Aplicando la regla de promedios del CLAUDE.md: cuando se reporte algo como "loss media en la época 20", aclarar que es **promedio sobre los mini-batches de esa época** — no es lo mismo que "loss en el dataset completo evaluada en el estado de los pesos al final de la época 20".
