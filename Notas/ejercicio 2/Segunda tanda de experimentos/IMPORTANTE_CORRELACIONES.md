# IMPORTANTE — Correlaciones entre hiperparámetros descubiertas en el Ej2

**Status:** este archivo resume **lecciones cruzadas** que cambiaron cómo entendemos los experimentos del Ej2. Los sweeps one-at-a-time previos (Arch, LR, Optimizer) tomaron decisiones asumiendo independencia entre factores, pero el cross-experiment `cross_v1` mostró que varias de esas asunciones no se sostienen.

**Para defensa oral:** acá está la respuesta a la pregunta esperable *"¿probaste que tus hiperparámetros eran independientes? ¿qué hiciste cuando viste que no lo eran?"*.

Datos fuente:
- [Plan del cross-experiment](PLAN%20de%20todos%20los%20experimentos%20cruzados%20cross_v1.md)
- [Pre-experimento LR×Batch×Opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md) → `best_batch.json`
- [Cross-experiment LR×Opt×Arch + estrella batch](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md)
- Sweeps previos tratados como independientes: [Arch](Arquitectura.md) · [LR (1er sweep)](analisis_lr.md) · [LR_segundo_intento](LR_segundo_intento/) · [Optimizer](analisis_optimizer.md)

---

## TL;DR — las correlaciones que encontramos

| Par de factores | ¿Independientes? | Evidencia | Severidad |
|---|---|---|---|
| **LR × Optimizer** | NO | El LR óptimo varía 10× entre Adam (1e-3) y SGD (1e-2) | 🔴 Crítica |
| **LR × Batch** | NO | Adam: batch escala con LR (regla lineal de la cátedra) | 🟠 Fuerte |
| **Arch × LR** | NO | El ranking de arquitecturas cambia según el LR | 🟠 Fuerte |
| **Arch × Optimizer** | Parcial | shallow es robusta; wider sólo gana con Adam@1e-3 | 🟡 Moderada |
| **Seed → split + init** | Acoplados (código) | `random_seed` controla *fold partition* Y *init de pesos* | 🟡 Moderada |
| **patience × max_epochs** | Estructural | si patience ≥ max_epochs/2 el ES no actúa | 🟢 Suave |

---

## 1. LR × Optimizer — la más fuerte y obvia

**Hipótesis previa:** "podemos elegir el LR óptimo y después el optimizer".
**Lo que vimos:** los LRs óptimos por optimizer difieren por **un orden de magnitud**.

Datos del cross-experiment ([stage 2](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md), 15 corridas/cell):

| Optimizer | LR óptimo (todas las archs) | val_acc en óptimo |
|---|---|---|
| Adam | **1e-3** | 0.9583 (wider) / 0.9572 (shallow) |
| Momentum | **5e-3 a 1e-2** | 0.9543 (shallow @ 1e-2) |
| SGD | **1e-2** | 0.9509 (shallow @ 1e-2) |

**Implicancia metodológica:** un "sweep de LR" sin especificar optimizer es un experimento mal diseñado. Lo que descubrimos en el [LR sweep original](analisis_lr.md) ("LR=1e-3 da accuracy ~0.93 con SGD") es **falso para otros optimizadores**, porque ese sweep usó SGD only.

**Por qué pasa:** Adam normaliza el gradiente por la varianza acumulada → el "paso efectivo" en el espacio de pesos es de tamaño ~lr (independiente de la magnitud del gradiente). SGD da pasos = lr · gradiente, así que para gradientes chicos en este problema, necesita LR mucho más alto. Esto sale de la [clase de optimizadores](../../../docs/clase_optimizadores/clase%20optimizadores.pdf).

---

## 2. LR × Batch — confirmamos empíricamente la regla lineal

**Regla teórica (de la cátedra):** "doblar el batch ≈ doblar el LR" (ó √2× según el régimen).

**Lo que medimos** en el [pre-experimento LR×Batch×Opt](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md), 10 corridas/cell, arch_shallow:

| Optimizer | LR=5e-4 → best batch | LR=1e-3 → best batch | LR=5e-3 → best batch |
|---|---|---|---|
| **Adam** | **16** | **64** | **256** |
| Momentum | 16 | 16 | 16 |
| SGD | 16 | 16 | 16 |

**Adam confirma la regla casi perfectamente:** LR×2 → batch×4 (5e-4→1e-3), LR×5 → batch×4 (1e-3→5e-3). Esto es exactamente lo que predice la teoría: con un LR más alto, cada paso es más grande, y para evitar overshoot conviene promediar el gradiente sobre más muestras (batch grande).

**SGD/Momentum no muestran escalado** en el rango medido. Hipótesis: necesitan LR mucho más alto que 5e-3 para que la regla sea visible (su óptimo está en LR=1e-2, fuera del barrido). Con un sweep ampliado a LR=2e-2, 5e-2 con SGD probablemente veríamos el mismo patrón.

**Implicancia metodológica:** el [LR_segundo_intento](LR_segundo_intento/) y todos los sweeps anteriores asumían `batch_size=32` fijo. Si Adam fuese el optimizer, eso significa que para LR=5e-3 el batch debería haber sido **256, no 32** — los resultados de Adam@5e-3 con batch=32 estaban sub-óptimamente seteados.

---

## 3. Arch × LR — el ranking de arquitecturas depende del LR

**Hipótesis previa (del [Arch sweep](Arquitectura.md)):** "shallow es la arquitectura óptima".
**Lo que vimos:** shallow es óptima **sólo para algunos LRs**. Para Adam@1e-3, **wider la supera**.

Datos del cross-experiment, optimizer=Adam, ranking de arquitecturas por LR (val_acc):

| LR | 1° | 2° | 3° | 4° |
|---|---|---|---|---|
| 1e-4 | wider (0.9547) | shallow (0.9546) | base (0.9528) | deeper (0.9511) |
| 5e-4 | **shallow (0.9567)** | wider (0.9553) | base (0.9541) | deeper (0.9521) |
| 1e-3 | **wider (0.9583)** | shallow (0.9572) | base (0.9548) | deeper (0.9535) |
| 5e-3 | shallow (0.9546) | base (0.9533) | wider (0.9531) | deeper (0.9513) |
| 1e-2 | shallow (0.9472) | base (0.9465) | deeper (0.9455) | **wider (0.9450)** |

**Patrón claro: a mayor LR, peor le va a `wider`.** En LR=1e-2 wider queda último. Esto es la regla teórica *"modelos con más parámetros prefieren LRs más chicos"*: con LR=1e-2 sobre wider (~235k params) los pasos son demasiado grandes en el espacio expandido y el modelo no estabiliza.

**Implicancia metodológica:** el [Arch sweep original](Arquitectura.md) se hizo con **Adam@1e-3 fijo**, lo que casualmente es justo el LR donde shallow y wider están más cerca (0.0011 de diferencia). Si lo hubiéramos hecho con Adam@1e-2, **deeper o base habría parecido la óptima**, porque wider colapsa ahí. La conclusión "shallow es la arq óptima" del sweep one-at-a-time **estaba condicionada al LR específico que se usó como fijo**.

→ Esto es exactamente lo que motivó el cross-experiment y el [Arch tiebreaker](../Experimentos/Arch_tiebreaker/) con muchos seeds para definir entre shallow y wider con confianza estadística.

---

## 4. Arch × Optimizer — moderada, principalmente por wider

**Hipótesis:** "el ganador del Arch sweep gana con todos los optimizadores".
**Lo que vimos:** **shallow gana** con Momentum y SGD; **wider sólo gana con Adam**, y sólo en LR=1e-3.

Best arch por optimizer (cross-experiment, mejor LR de cada combo):

| Optimizer | Best arch | LR usado |
|---|---|---|
| Adam | wider (0.9583) | 1e-3 |
| Momentum | **shallow** (0.9543) | 1e-2 |
| SGD | **shallow** (0.9509) | 1e-2 |

**Lectura:** shallow es robusta a través de optimizadores; wider es una "punta especialista" que requiere Adam y un LR específico. Para defensa oral, la decisión entre wider y shallow depende del criterio: máximo absoluto (wider) vs robustez (shallow).

---

## 5. Correlaciones ocultas en el código (de la auditoría previa)

Cosas que el código trataba como una variable pero acoplaban dos cosas:

### 5.1 `seed` controla split + init simultáneamente

En `mlp/train.py`, `random_seed` rige tanto el **split estratificado en folds** como el **init de pesos**. Cuando reportamos `std` sobre N seeds, esa varianza mezcla dos fuentes (cuál fold tocó, qué init salió). No podemos decir "la varianza por init es X" sin un experimento dedicado (mismo split, distinto init).

### 5.2 Orden de batches NO depende de seed

`mlp/network.py:159` usa `BatchIterator(..., seed=epoch)` — la secuencia de minibatches en cada época es **idéntica entre seeds y folds**. Sólo cambia el init y el subset. El "ruido SGD" está sub-estimado en nuestros stds.

### 5.3 `patience × max_epochs` no son independientes en práctica

Si `patience ≥ max_epochs/2`, el ES nunca actúa efectivamente. Para Adam@5e-3 con max=30 y patience=20, el corte real es a más tardar en epoch 21 (best+patience), nunca cerca de 30.

### 5.4 Métricas finales antes del fix #4

Antes de arreglar el bug, las métricas se computaban sobre los pesos del **último** epoch si ES no disparaba, y sobre los pesos del **best** epoch si disparaba → comparábamos cosas distintas entre celdas. Fix aplicado en commit anterior: ahora siempre se restaura `best_weights`.

---

## 6. Cómo cambia esto las decisiones del Ej2

| Decisión previa | Fundamento previo | Estado tras cross-experiment + tiebreaker |
|---|---|---|
| arch = shallow | Arch sweep con Adam@1e-3 fijo | **Sostenido por Occam.** El [tiebreaker con 15 seeds × k=5](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Arch_tiebreaker/analisis.md) mostró que `wider` y `shallow` son **estadísticamente indistinguibles** (z=0.65, diff=0.0005). Misma performance, mitad de parámetros → shallow gana. |
| optimizer = Adam | Optimizer sweep con arch_base | **Sostenido**. Top-10 de cross_v1 son todos Adam. |
| LR = 1e-3 | Optimizer sweep | **Sostenido para Adam.** Para SGD/Momentum sería 1e-2. |
| batch = 32 | (default, nunca medido) | **Sub-óptimo**. `best_batch` por (opt, LR) varía: Adam@1e-3=64, Adam@5e-3=256. |

**Configuración final del Ej2:** **`arch_shallow` + Adam + LR=1e-3 + batch=64**.

### Detalle del tiebreaker

El cross_v1 con 3 seeds dejó arch_wider (0.9583) levemente arriba de arch_shallow (0.9572) — **diff=0.0011 con SEM ~0.001**, indistinguible. El [tiebreaker dedicado](Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/Arch_tiebreaker/analisis.md) amplió la muestra a **15 seeds × k=5 = 75 corridas/cell** (3 seeds del cross_v1 + 12 seeds nuevos). Resultado:

- arch_wider:   0.9581 ± 0.0049 (SEM=0.0006)
- arch_shallow: 0.9576 ± 0.0046 (SEM=0.0006)
- **diff = +0.0005, SEM(diff) = 0.0007, z = 0.65** → no distinguibles al 95% (necesitábamos |z|>1.96).

**Lección:** la diferencia de 0.0011 que vimos con 3 seeds era **ruido de muestreo**. Al ampliar a 15 seeds, la diff bajó a 0.0005 — más cerca de 0, consistente con "no hay diferencia real entre las dos arquitecturas en este problema".

---

## 7. Cómo se descubren correlaciones entre HPs — la técnica

Esta es la pregunta metodológica del TP. Hay tres niveles de rigor, escalando en costo de cómputo:

### Nivel 1 — One-at-a-time (lo que hicimos primero)

Fijás todo en valores "razonables" y barrés un factor. Repetís para cada factor.

- **Costo:** lineal en (factores × niveles).
- **Pros:** barato, fácil de explicar.
- **Contras:** **asume independencia entre factores**. Si dos HPs interactúan (como Arch×LR), tu conclusión depende del valor que fijaste para los demás.
- **Cuándo sirve:** screening inicial, cuando sabés a priori que las interacciones son débiles.

### Nivel 2 — Diseño factorial 2k

Elegís 2 niveles por factor (un "bajo" y un "alto") y corrés **todas las combinaciones**. Para 4 factores: 2⁴ = 16 cells.

- **Costo:** exponencial en factores, lineal por nivel.
- **Pros:** detecta **todas las interacciones de a pares y triples** entre los factores barridos.
- **Contras:** sólo 2 puntos por factor → no ves la curva, sólo si "aumenta o no". Si elegiste mal los niveles, te perdés el óptimo.
- **Cómo se analiza:** efecto principal del factor X = mean(X alto) − mean(X bajo). Efecto interacción X×Y = (efecto X cuando Y=alto) − (efecto X cuando Y=bajo). Si ese número es chico vs el ruido, los factores son independientes; si es grande, interactúan.
- **Cuándo sirve:** screening cuando sospechás que hay interacciones pero no sabés cuáles.

### Nivel 3 — Centro + slices selectivos (lo que hicimos en `cross_v1`)

Anclás todo en una "configuración centro" (la mejor del one-at-a-time) y desde ahí:

1. **Estrella 1D:** variás cada factor por separado, dejando los demás en el centro. Da resolución alta por factor (curvas, no barras), valida robustez del centro.
2. **Slices 3D selectivos:** en los pares que sospechás interactúan, hacés un mini-grid LR×Opt×Arch (no sólo 2D, porque el 2D condicionaría a un valor fijo de LR que sabemos es sesgado por opt). Las dimensiones del slice las elegís según teoría.

- **Costo:** intermedio. Más caro que 2k pero con mucha más resolución por factor.
- **Pros:** historia oral limpia ("este es mi best, lo perturbé en cada eje, sigue ganando, y acá están las interacciones que la teoría predecía"). Curvas, no barras.
- **Contras:** sólo medís las interacciones que **vos elegiste meter** en los slices. Si una interacción inesperada existe, no la ves.

### Lección clave: las "slices 2D" mienten

Un error tentador es hacer "slice Arch × Opt" con LR fijo. Pero como **LR×Opt están fuertemente correlacionados**, si fijás LR=1e-3 (óptimo de Adam), SGD entra a la slice con un LR sub-óptimo y siempre pierde. La conclusión "Adam le gana a SGD" sería trampa.

**La única slice 2D honesta** es una en la que las dimensiones que NO estás midiendo sean realmente independientes de las que sí. Cuando no lo son, hay que **subir a 3D** (meter el factor correlacionado adentro). Eso es lo que hicimos en `cross_v1` stage 2 con LR×Opt×Arch.

### Cómo decidir si un par interactúa cuantitativamente

Una vez tenés un grid factorial (o un slice 3D), el test es:

```
varianza_inter_celdas (entre niveles de X, fijando Y) ≠ constante a través de Y → interacción
```

O más simple: graficás `metric(X)` para cada nivel de Y (una curva por Y). Si las curvas son **paralelas**, X e Y son independientes. Si **se cruzan o cambian de pendiente**, interactúan.

**Ejemplo nuestro:** la tabla del punto 3 (Arch × LR para Adam) muestra que el ranking de archs **se reordena** al cambiar LR. Eso es un cruce de curvas → interacción Arch×LR confirmada.

### Heurísticas prácticas para el Ej2

1. **Antes de cada nuevo sweep, listá los HPs que estás fijando** y preguntate si la teoría predice interacción con el factor que vas a barrer.
2. Si sospechás interacción y el costo lo permite: subí a un grid 2D del par. Si no, **declarálo como caveat** ("nuestro resultado vale condicional a `batch=32`").
3. Cuando reportes un "óptimo", **siempre incluí el contexto fijo** ("`arch_shallow` óptima *bajo Adam@1e-3*").
4. Para defensa oral: tener al menos **un slice 3D** es lo que distingue un trabajo sólido de uno superficial. Es lo que nos da el lenguaje para decir "vimos que X e Y interactúan, acá está la evidencia".
