# Apuntes: Perceptrón Simple

## 1. Contexto histórico y biológico

El estudio de redes neuronales como herramienta computacional tiene una historia que arranca en la **década del 40**, y los hitos principales que mencionó el docente son *(ver [06:53])*:

- **1943 — McCulloch y Pitts**: introducen las bases del campo formalizando matemáticamente el modelo de una **neurona artificial**. Su aporte fue tomar la neurona biológica y traducirla a una fórmula matemática y un esquema.
- **1957-1958 — Frank Rosenblatt**: propone la regla de aprendizaje, es decir, cómo esas neuronas pueden *aprender* a partir de datos. Es el nacimiento del perceptrón como herramienta práctica.
- **1960 — Widrow y Hoff**: introducen el **Adaline** (Adaptive Linear Element), que es básicamente un perceptrón con función de activación lineal (identidad) en lugar de escalón. Esto permite resolver problemas de **regresión** y no solo de clasificación binaria.
- **Década del 70 — AI Winter**: la investigación se frena por limitaciones de hardware y falta de interés.
- **Década del 80 — Resurgimiento**: aparece el algoritmo de **Backpropagation**, que es la base para entrenar redes neuronales multicapa y que seguiremos usando hoy.
- **2000s-2010s**: aparecen arquitecturas modernas (CNNs, GANs, Autoencoders, Hopfield, Transformers), y finalmente el boom de los modelos de lenguaje.

### Inspiración biológica

El perceptrón se inspira en la **neurona biológica** *(ver [16:11])*. Los componentes clave que se modelan:

- **Dendritas**: reciben estímulos de otras neuronas → se modelan como **entradas** $x_i$.
- **Sinapsis**: conexión entre neuronas, con una "fuerza" determinada por neurotransmisores e inhibidores → se modela con los **pesos sinápticos** $w_i$.
- **Soma / cuerpo de la neurona**: integra la información recibida → se modela como la **suma ponderada**.
- **Disparo (o no disparo)**: si la integración supera un umbral, la neurona dispara → se modela con la **función de activación** que devuelve un valor binario (1 / 0, o +1 / -1).

> **Intuición:** La conjetura de Donald Hebb dice *"neurons that fire together wire together"*. La idea es que si dos neuronas disparan juntas, la conexión entre ellas se refuerza. En el modelo computacional esto se representa con pesos sinápticos numéricos: una conexión más fuerte = un peso de mayor magnitud *(ver [18:03])*.

> **Intuición biomimética:** Al igual que los algoritmos genéticos se inspiran en la genética, las redes neuronales se inspiran en el cerebro. No es una copia fiel, sino una **formalización matemática** de ideas observadas en la naturaleza *(ver [05:07])*.

## 2. Arquitectura del perceptrón

### 2.1 Entradas y vector de pesos

El perceptrón recibe un vector de entradas (estímulos):

$$\mathbf{x} = (x_1, x_2, \ldots, x_n)$$

Y tiene asociado un vector de pesos sinápticos (los **parámetros libres del modelo**):

$$\mathbf{w} = (w_1, w_2, \ldots, w_n)$$

En la implementación esto son simplemente números (por ejemplo, $x_1 = 3.2$, $x_2 = 1.4$, etc.) *(ver [19:46])*.

### 2.2 Suma ponderada / net input (estado de excitación)

El procesamiento dentro del cuerpo de la neurona se modela como el **producto interno** entre entradas y pesos. Se lo llama **estado de excitación** y se denota $h$:

$$h = \sum_{i=1}^{n} w_i \, x_i$$

### 2.3 Función de activación

La salida de la neurona se obtiene aplicando una **función de activación** $\theta$ sobre el estado de excitación. En el perceptrón original (McCulloch-Pitts / Rosenblatt), $\theta$ es una **función escalón** (o signo):

$$\theta(h) = \begin{cases} +1 & \text{si } h \geq \text{umbral} \\ -1 & \text{si } h < \text{umbral} \end{cases}$$

Podés usar indistintamente $\{+1, -1\}$ o $\{1, 0\}$ siempre que sea binaria.

### 2.4 Bias / umbral

El **umbral** $\theta_{umbral}$ define el valor a partir del cual la neurona dispara. En la práctica se lo reformula como un **bias** $b = w_0$ que se suma al estado de excitación, permitiendo que el hiperplano no esté obligado a pasar por el origen *(ver [29:30])*.

$$h = \sum_{i=1}^{n} w_i \, x_i + w_0$$

**Truco de implementación** *(ver [57:24])*: en lugar de manejar el bias por separado, agregás una columna de unos al dataset ($x_0 = 1$ para todos los datos), y así:

$$h = \sum_{i=0}^{n} w_i \, x_i$$

con $w_0$ funcionando como bias automáticamente.

### 2.5 Salida

La salida final de la neurona, denotada $O$ (o $o^{\mu}$ para el dato $\mu$):

$$O = \theta\left(\sum_{i=0}^{n} w_i \, x_i\right)$$

## 3. Regla de aprendizaje

### 3.1 Aprendizaje supervisado por corrección de error

El perceptrón es un algoritmo de **aprendizaje supervisado**: para cada dato de entrada $\mathbf{x}^{\mu}$ conocemos la **salida esperada** $z^{\mu}$ (la etiqueta).

**Notación** *(ver [58:59])*:
- $P$: cantidad de datos totales.
- $\mu$: índice del dato, $\mu \in \{0, 1, \ldots, P-1\}$.
- $x^{\mu}_i$: valor de la $i$-ésima entrada del dato $\mu$.
- $z^{\mu}$: salida esperada del dato $\mu$.
- $O^{\mu}$: salida obtenida por la neurona para el dato $\mu$.

Queremos que $O^{\mu}$ coincida con $z^{\mu}$.

### 3.2 Fórmula de actualización de pesos

Rosenblatt propone actualizar los pesos **cada vez que se recibe un estímulo** (entrenamiento **online**):

$$w_i^{\text{nuevo}} = w_i^{\text{viejo}} + \Delta w_i$$

Donde el delta se aplica **solo si la salida obtenida difiere de la esperada**:

$$\Delta w_i = \begin{cases} \eta \, (z^{\mu} - O^{\mu}) \, x_i^{\mu} & \text{si } O^{\mu} \neq z^{\mu} \\ 0 & \text{si } O^{\mu} = z^{\mu} \end{cases}$$

Equivalentemente, se puede escribir de forma compacta:

$$\Delta w_i = \eta \, (z^{\mu} - O^{\mu}) \, x_i^{\mu}$$

ya que cuando $z^{\mu} = O^{\mu}$ la diferencia es 0 y no hay actualización.

### 3.3 Tasa de aprendizaje ($\eta$)

$\eta$ (eta) es la **tasa de aprendizaje** (*learning rate*). Modera el tamaño del paso que damos en la dirección de actualización *(ver [41:02])*.

- Valores típicos: $10^{-1}, 10^{-2}, 10^{-3}$.
- Si $\eta$ es **muy grande** (por ej. 10): el algoritmo se vuelve inestable, oscila y puede no converger porque "pasa de largo" el mínimo.
- Si $\eta$ es **muy chico**: tarda muchísimo en llegar y puede quedar atrapado en mínimos locales.

> **Intuición:** el learning rate está en unidades de gradiente, entonces depende mucho de la geometría de la función de costo. No se elige a ciegas: se experimenta *(ver [79:26])*.

### 3.4 Interpretación geométrica: el hiperplano separador

El perceptrón encuentra un **hiperplano de separación** entre dos clases linealmente separables *(ver [29:52])*.

- En $\mathbb{R}^2$: una **recta**.
- En $\mathbb{R}^3$: un **plano**.
- En general: un **hiperplano**.

La ecuación del hiperplano se escribe como:

$$w_0 + \sum_{i=1}^{n} w_i \, x_i = 0$$

Cada dato nuevo se clasifica según de qué lado del hiperplano cae (proyección positiva o negativa).

> **Intuición clave:** la fórmula del perceptrón de McCulloch-Pitts y la ecuación del hiperplano separador son **la misma fórmula**. Este es el puente entre el modelo biológico abstracto y la herramienta práctica para clasificación *(ver [32:35])*.

## 4. Algoritmo paso a paso

**Algoritmo del perceptrón simple (versión escalón, clasificación binaria)** *(ver [52:21])*:

1. **Inicializar** los pesos $w_1, w_2, \ldots, w_n$ con **valores random pequeños**.
2. **Inicializar** el bias $w_0$ con un valor random pequeño.
3. **Elegir** una tasa de aprendizaje $\eta$.
4. **Elegir** una cantidad máxima de épocas (epochs). Una **época** = una pasada completa por todo el dataset.
5. **Para cada época**:
   1. **Para cada dato** $\mu$ del conjunto de entrenamiento:
      1. Calcular el **estado de excitación**: $h^{\mu} = \sum_{i=0}^{n} w_i \, x_i^{\mu}$
      2. Calcular la **activación / salida**: $O^{\mu} = \theta(h^{\mu})$
      3. Calcular $\Delta w_i = \eta \, (z^{\mu} - O^{\mu}) \, x_i^{\mu}$ para todo $i$.
      4. Actualizar: $w_i \leftarrow w_i + \Delta w_i$
   2. Calcular el **error del perceptrón** sobre **todo** el dataset:
      $$E = F(w, \text{todos los datos})$$
   3. Si $E < \epsilon$ (convergencia) → **cortar**.
6. Si se alcanzó la cantidad máxima de épocas sin converger, también cortar.

> **Muy importante:** el error del perceptrón se calcula sobre **todo el conjunto de datos**, no sobre un dato individual. Esto es porque la convergencia implica que funciona bien para todo el dataset, no solo para el dato actual *(ver [56:20])*.

**Tip de implementación:** graficar cómo se van moviendo los pesos y el hiperplano a lo largo de las épocas. Ayuda mucho a detectar bugs *(ver [51:45])*.

## 5. Convergencia

El docente no enunció formalmente el teorema de convergencia de Rosenblatt, pero sí mencionó las ideas clave *(ver [43:39])*:

- Si el problema es **linealmente separable**, el algoritmo **converge** a una solución (un hiperplano que separa las clases).
- Si el problema **no es linealmente separable**, el algoritmo **no termina**: sigue oscilando entre hiperplanos hasta que se agoten las épocas.

**Cómo cortar el algoritmo** *(ver [61:49])*:
- Por **error nulo** (100% de aciertos): apropiado para clasificación binaria con datasets simples.
- Por **cantidad máxima de épocas**: salvaguarda en caso de no convergencia.
- Por **error menor a un $\epsilon$**: útil en problemas más grandes donde exigir error 0 no es realista.

> **Intuición importante:** la solución que encuentra el perceptrón es **una** solución potencial, no necesariamente la **mejor**. Pueden existir muchos hiperplanos que separen correctamente los datos, y el perceptrón se queda con el primero que encuentre (no maximiza el margen como SVM) *(ver [48:24])*.

## 6. Limitaciones

### 6.1 Separabilidad lineal

El perceptrón **solo puede resolver problemas linealmente separables** *(ver [28:01])*. Si los datos no se pueden separar con una recta/plano/hiperplano, el perceptrón no converge.

Ejemplo mencionado: si tenés un dataset donde una clase está distribuida de forma que no se puede separar con una recta, o si tenés **outliers** que caen en zonas mezcladas, el perceptrón falla.

### 6.2 Clasificación binaria únicamente

El perceptrón simple **solo maneja 2 clases** *(ver [35:19])*. Si tenés 3 o más clases, necesitás otras estrategias (por ejemplo, combinar varios perceptrones, one-vs-rest, etc., pero eso no se ve en esta clase).

### 6.3 Solución no óptima

Como mencionamos, la recta que devuelve el perceptrón puede pasar "al ras" de los datos de una clase, sin dejar margen. No garantiza la **mejor** separación.

## 7. Ejemplos trabajados en clase

### 7.1 Ejemplo conceptual: elecciones de Canadá

*(ver [22:42])* Un cliente quiere saber a qué partido (conservador o liberal) va a votar una persona en base a su **edad** e **ingreso económico normalizado** (entre 0 y 1).

- Etiqueta $z = +1$: conservador.
- Etiqueta $z = -1$: liberal.

Se asume el problema como **linealmente separable** y se usa un perceptrón simple para encontrar el hiperplano (recta) que separa ambas clases.

### 7.2 Ejemplo numérico de actualización

*(ver [45:13])* Se toma el dato $\mu = 10$ del dataset. Con los pesos actuales y el umbral actuales, se calcula:

- $h^{10} = \sum w_i \, x_i^{10}$ → da como resultado un valor (en el ejemplo, aproximadamente $0.742$).
- $O^{10} = \theta(h^{10}) = +1$ (disparó).
- Pero $z^{10} = -1$ (era liberal).
- **Como $O^{10} \neq z^{10}$**, se actualiza:
  $$\Delta w_i = \eta \, (z^{10} - O^{10}) \, x_i^{10} = \eta \, (-1 - 1) \, x_i^{10} = -2\eta \, x_i^{10}$$
- Tras actualizar, el nuevo $h^{10} \approx 0.477$ (más cercano a cambiar de signo) y el hiperplano se movió en la dirección correcta.

> **Moraleja del ejemplo:** la actualización no siempre clasifica bien al dato en un solo paso, pero **sí mueve el hiperplano en la dirección correcta**. Con suficientes iteraciones, converge *(ver [48:00])*.

### 7.3 Ejemplo de regresión con Adaline: salario vs años de experiencia

*(ver [64:19])* Queremos estimar el salario a pagar en función de los años de experiencia. Ahora la salida $z$ es un **número real**, no una clase binaria. Se modela como:

$$y = a \cdot x + b$$

que es exactamente la forma del perceptrón con función de activación identidad.

## 8. Conceptos clave para el TP

### 8.1 Qué vas a tener que implementar

Según lo que dijo el docente, el TP3 va a pedir implementar **3 tipos de perceptrón simple** *(ver [12:34], [85:59])*:

1. **Perceptrón simple escalón** (clasificación binaria con $\theta$ = función signo).
2. **Perceptrón simple lineal** (Adaline, con $\theta$ = identidad, para regresión).
3. **Perceptrón simple no lineal** (con $\theta$ = tangente hiperbólica o sigmoide logística, para regresión no lineal).

### 8.2 Perceptrón lineal (Adaline)

La función de activación es la **identidad**: $\theta(h) = h$.

La salida es directamente el estado de excitación:

$$O = \sum_{i=0}^{n} w_i \, x_i$$

**Función de costo (error cuadrático)**:

$$E(\mathbf{w}) = \frac{1}{2} \sum_{\mu} \left(z^{\mu} - O^{\mu}\right)^2$$

**Regla de actualización por descenso de gradiente**:

$$\Delta w_i = -\eta \, \frac{\partial E}{\partial w_i}$$

Haciendo la derivada (la cadena sale como se vio en *(ver [81:07])*):

$$\Delta w_i = \eta \, (z^{\mu} - O^{\mu}) \, \theta'(h^{\mu}) \, x_i^{\mu}$$

Para el caso lineal, $\theta'(h) = 1$, entonces:

$$\Delta w_i = \eta \, (z^{\mu} - O^{\mu}) \, x_i^{\mu}$$

> **Observación clave:** esta fórmula sirve para **cualquier función de activación derivable**. Solo cambia $\theta'$. Eso permite usar el mismo esquema para el no-lineal *(ver [81:12])*.

### 8.3 Perceptrón no lineal

Función de activación: **tangente hiperbólica** o **logística (sigmoide)** *(ver [85:59])*.

- Tangente hiperbólica: $\theta(h) = \tanh(\beta h)$, imagen en $(-1, 1)$.
- Logística: $\theta(h) = \frac{1}{1 + e^{-2\beta h}}$, imagen en $(0, 1)$.

> **¡Atención!** La imagen de estas funciones es acotada. Si los valores esperados $z^{\mu}$ de tu dataset no están en ese rango, **tenés que normalizarlos** antes *(ver [86:38])*.

**Parámetro $\beta$**: controla la "pendiente" de la función. Si $\beta$ es muy grande, la función se parece a un escalón; si es chico, se parece a una lineal *(ver [87:19])*.

### 8.4 Funciones de costo / métricas de performance

Cuidado con la **dualidad de nombres** *(ver [83:12])*:

- **Función de costo (la que se minimiza con gradiente)**: típicamente el error cuadrático medio $\frac{1}{2}\sum (z^{\mu}-O^{\mu})^2$.
- **Métrica de performance / rendimiento**: puede ser accuracy (clasificación), MSE (regresión), etc. Se usa para **evaluar** y como **criterio de corte**.

### 8.5 Tips de implementación

- **Agregar columna de unos** para integrar el bias en los pesos *(ver [57:24])*.
- **Inicializar pesos con valores random pequeños**, no en cero *(ver [52:49])*.
- **Graficar** la evolución del hiperplano y de los pesos para debuggear.
- **Testear el perceptrón lineal con datos generados desde una recta exacta**: tiene que ajustar perfectamente *(ver [83:52])*.
- **Experimentar con distintos learning rates** (pueden probar hasta valores absurdos como 10 para ver qué pasa).

### 8.6 Entrenamiento online vs batch

Lo visto hasta acá es **entrenamiento online**: se actualiza **después de ver cada dato**. No se acumulan los deltas *(ver [84:25])*. El esquema batch se ve en la próxima clase (backpropagation).

## 9. Glosario de términos

- **Perceptrón**: modelo matemático de una neurona artificial que clasifica datos mediante un hiperplano separador.
- **Pesos sinápticos ($w_i$)**: parámetros libres del modelo que representan la fuerza de conexión entre entradas y la neurona.
- **Bias / umbral ($w_0$, $b$, $\theta_{umbral}$)**: parámetro que permite al hiperplano no estar atado al origen.
- **Estado de excitación ($h$)**: suma ponderada de las entradas, $h = \sum w_i x_i$.
- **Función de activación ($\theta$, $g$)**: función aplicada sobre $h$ para obtener la salida (escalón, identidad, tanh, sigmoide).
- **Salida ($O$, $o^{\mu}$)**: resultado de aplicar la activación al estado de excitación.
- **Salida esperada ($z^{\mu}$)**: etiqueta verdadera del dato $\mu$ (aprendizaje supervisado).
- **Tasa de aprendizaje ($\eta$)**: controla el tamaño del paso en la actualización de pesos.
- **Época**: una pasada completa por todos los datos del conjunto de entrenamiento.
- **Aprendizaje supervisado**: se conocen las etiquetas de los datos de entrenamiento.
- **Aprendizaje no supervisado**: solo se conocen las entradas, se buscan agrupamientos/clusters.
- **Aprendizaje (training)**: etapa en la que el modelo ajusta sus pesos con datos conocidos.
- **Generalización**: habilidad del modelo de desempeñarse bien con datos **no vistos** durante el entrenamiento. Esto es lo que realmente importa *(ver [59:07])*.
- **Hiperplano de separación**: recta (en 2D), plano (en 3D), o generalización en más dimensiones que divide el espacio en dos clases.
- **Linealmente separable**: conjunto de datos que puede separarse en dos clases mediante un hiperplano.
- **Función de costo / error**: función que mide qué tan mal clasifica el perceptrón. Se busca minimizarla.
- **Gradiente**: vector que apunta en la dirección de **máximo crecimiento** de una función. Para minimizar, nos movemos en la dirección **opuesta**.
- **Online learning**: se actualizan los pesos después de cada dato visto.
- **Adaline**: perceptrón lineal (activación identidad), propuesto por Widrow y Hoff.
- **MSE (Mean Squared Error)**: $\frac{1}{P}\sum (z^{\mu}-O^{\mu})^2$, métrica común para regresión.
- **Accuracy**: proporción de datos correctamente clasificados, métrica común para clasificación.

---

## Temas mencionados sin profundizar

- **Backpropagation / retropropagación** *(ver [07:58])*: se ve en la clase 2, es el algoritmo que extiende el perceptrón a redes multicapa.
- **Deep learning** *(ver [11:21])*: redes con muchas capas. Se ve en el TP5.
- **AI Winters** *(ver [07:19])*: períodos históricos en los que la investigación en IA se frenó. No se desarrolló.
- **Métricas de evaluación y sobreajuste** *(ver [12:49])*: tema de la clase 3.
- **Aprendizaje no supervisado** *(ver [16:02])*: se ve en el TP4.
- **Inicialización de pesos: por qué random y no cero** *(ver [52:55], [88:37])*: queda como ejercicio abierto, se discute en la clase del multicapa.
- **SVM / margen máximo** *(ver [50:31])*: mencionado al pasar, no es parte de esta materia.
- **Entrenamiento batch** *(ver [84:36])*: se ve la clase que viene.