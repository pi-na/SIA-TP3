# Stage 1
Analisis de los resultados obtenidos en la etapa de "Pre_LR_Batch_Opt", resultados mas finos explicados en [[Notas/ejercicio 2/Segunda tanda de experimentos/Pre_LR_Batch_Opt/analisis|analisis]]

En nuestros primeros experimentos del ejercicio 2, fuimos modificando los hiperparametros 1 a la vez. Comenzamos definiendo 4 arquitecturas y variando el lr -> vemos que ENTRE las arquitecturas no hay variacion en cuanto a los lr. Ambas curvas de convergencia (train loss y val loss) muestran el mismo comportamiento.
![convergence_train](convergence_train.png)
Continuando con el experimento, fijamos el batch-size en 32, pero fuimos variando el optimizer y lr, empezamos a ver como a medida que fuimos agrandando el lr, iban empeorando los resultados para cada optimizer. Todas las corridas fueron con 5 seeds y 5 folds.
Los lr utilizados fueron 1e-4, 5e-4, 1e-3, 5e-3, 1e-2.
![convergence_lr1e-4](convergence_lr1e-4.png)
![convergence_lr1e-3](convergence_lr1e-3.png)
![convergence_lr5e-3](convergence_lr5e-3.png)
Del optimizer sweep (batch=32 fijo) sacamos una observación: Adam se desestabiliza con LR ≥ 5e-3, mientras que SGD/Momentum siguen tolerando LR  alto. Esa observación + la teoría de la cátedra (la regla LR↔batch lineal)  nos generó la sospecha de que el techo de Adam podría no ser intrínseco al  optimizador, sino al producto LR×batch. Para confirmarlo no nos alcanzaba con  el optimizer sweep — necesitábamos variar también el batch.
![[Pasted image 20260510171351.png]]

El techo de Adam (lr < 5e-3 con batch=32) no es un límite del optimizador sino del producto LR×batch. El pre-experimento confirma que Adam@5e-3 con batch=256 es estable (val_loss=0.191) mientras que con batch=16 no convergio (val_loss=0.550). La regla de escalado lineal LR↔batch de la clase de optimizadores predice exactamente esto. Para val_loss, la mejor combinación encontrada es  Adam con lr=5e-4×batch=16 o lr=1e-3×batch=64 (ambas dan val_loss ≈ 0.170). SGD y Momentum son mucho  menos sensibles al batch — su val_loss varía menos entre batch=16 y batch=256 — pero parten de valores de val_loss más altos que Adam en la zona óptima.

![[lr_batch_relationship.png]]

  En el optimizer sweep con batch=32 fijo, Adam fue el único optimizador que mostró comportamiento drásticamente distinto según el LR: estable y rápido con lr≤1e-3, inestable y oscilante con lr≥5e-3. SGD y Momentum toleraron LR altos sin colapsar — sus curvas de convergencia degradaban gradualmente, sin el quiebre abrupto que mostró Adam.
  Ese comportamiento exagerado de Adam no es un defecto — es consecuencia de su mecanismo de adaptación por parámetro. Adam ajusta internamente el tamaño de paso, lo que lo hace muy sensible al producto LR×batch. Si ese producto está fuera de rango, los pasos efectivos son demasiado grandes y el modelo no converge.
  El gráfico muestra los dos efectos en simultáneo. Para Adam con lr=5e-3 (inestable a batch=32), aumentar el batch de 16 a 256 baja la CE de 0.55 a 0.19 y sube la val_acc de 0.934 a 0.954 — las dos métricas mejoran juntas a medida que la relación LR×batch se acerca al rango adecuado. Para lr=5e-4 y lr=1e-3 (ya estables), las curvas son planas: el batch no mueve la aguja porque el LR ya era el correcto para esos valores. SGD y Momentum no requieren este análisis porque su sensibilidad al batch es mucho menor — para ellos, fijar batch=32 no introduce el mismo sesgo en la comparación.
![[ejercicio2_experimentacion/analisis/cross_v1/stage2/stage2_val_acc_vs_lr_per_opt.png]]

# Stage 2
![[grid_3d_static.png]]

> [!warning] Que significan las métricas! Importante!!
> Importantisimo tener visto [[Notas/ejercicio 2/Segunda tanda de experimentos/Cross_LR_Opt_Arch/analisis#Qué significa cada columna|explicación de métricas usadas en los experimentos]]
## Metricas, accuracy, loss, F1
Accuracy es la metrica principal -> Mas accuracy, acerté mas en la clasificación.
Pero, la accuracy puede mentir cuando las clases están desbalanceadas. Y en digits.csv la clase 5 está fuertemente minoritaria (271 ejemplos vs ~1500 las otras).                                                      

Imaginá un modelo que aprende perfectamente las 9 clases mayoritarias y nunca predice la clase 5:                                                                                                                                     
  - Acertará ~97.8% de las imágenes (todo menos las del 5.                                                              
  - val_acc ≈ 0.978 — número aparentemente excelente.                                                                    
  - Pero la clase 5 tiene recall = 0 y por lo tanto F1_5 = 0.                                                                   
  - macro_f1 = (suma de 9 F1s buenos + 1 F1 = 0) / 10 ≈ 0.85 — más bajo de lo que esperarías por la accuracy. 
![[Pasted image 20260510234716.png]]
## Resultados
Ver e incluir en la presentación: [[Notas/ejercicio 2/Segunda tanda de experimentos/Cross_LR_Opt_Arch/analisis#Configuración completa|configuración e hiperparametros fijos y variados en el experimento]].

Queriamos ver como interactuan:
LR x OPT
LR x ARCH
ARCH x OPT
LR x Batch_size (*quedó para stage 3*)

Y encontrar configuración óptima ARCH x LR x OPT

![[Pasted image 20260510235747.png]]

### LR x OPT
Mido val_acc, val_loss, macro_f1.

#### Interacción LR x OPT con ARCH
![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/lr_opt_val_acc_4panels.png]]
![[ejercicio2_experimentacion/analisis/cross_v1/lr_opt/lr_opt_val_loss_4panels.png]]
![[ejercicio2_experimentacion/analisis/cross_v1/lr_opt/lr_opt_heatmaps_4archs.png]]

**SGD:** patrón idéntico en las 4 archs. Salto grande de `1e-4` a `5e-4` (~+0.02), después plateau. **Sin interacción con arch.**

| arch    | val_acc 1e-4 | salto a 5e-4 | plateau (5e-4 → 1e-2) |
| ------- | ------------ | ------------ | --------------------- |
| shallow | 0.9244       | +0.024       | +0.002                |
| base    | 0.9263       | +0.020       | +0.001                |
| wider   | 0.9293       | +0.019       | +0.001                |
| deeper  | 0.9286       | +0.015       | +0.002                |


**Momentum:** interacción **CLARA con arch**. `shallow` y `wider` toleran LR=`1e-2` (siguen mejorando); `base` y `deeper` se rompen ahí.

| arch    | mejor LR | val_acc en 5e-3 | val_acc en 1e-2 | Δ(5e-3 → 1e-2)                |
| ------- | -------- | --------------- | --------------- | ----------------------------- |
| shallow | 1e-2     | 0.9526          | **0.9543**      | **+0.0017** (sigue mejorando) |
| wider   | 1e-2     | 0.9531          | **0.9540**      | **+0.0009** (sigue mejorando) |
| base    | 5e-3     | 0.9506          | 0.9467          | **−0.0039** (cae)             |
| deeper  | 5e-3     | 0.9500          | 0.9425          | **−0.0075** (cae más fuerte)  |

**Hipótesis estructural:** las archs que toleran LR=`1e-2` con Momentum son las que tienen la **última capa oculta más ancha** (128 neuronas). Las que se rompen tienen la última capa más estrecha (64, 32).

| arch    | última capa hidden | tolera LR=1e-2 con Momentum |
| ------- | ------------------ | --------------------------- |
| shallow | 128                | ✅ sí                        |
| wider   | 128                | ✅ sí                        |
| base    | 64                 | ❌ no                        |
| deeper  | 32                 | ❌ no                        |

La última capa oculta antes del softmax actúa como "buffer". Con más neuronas, cada peso individual pesa menos en la salida final → el modelo es más estable ante pasos de gradiente grandes. Con la última capa estrecha (deeper tiene sólo 32), cada peso afecta más al output → LR alto desestabiliza.

**Adam:** las **shapes** son idénticas en las 4 archs (pico en `1e-3`, caída después). Lo que cambia es la **magnitud** del pico y de la caída.

| arch    | val_acc pico (1e-3) | val_acc en 1e-2 | caída pico → 1e-2           |
| ------- | ------------------- | --------------- | --------------------------- |
| shallow | 0.9572              | 0.9472          | **−0.0100**                 |
| base    | 0.9548              | 0.9465          | **−0.0083**                 |
| wider   | 0.9583              | 0.9450          | **−0.0133** (la más grande) |
| deeper  | 0.9535              | 0.9455          | **−0.0080**                 |

- **Wider** tiene el pico más alto **Y** la caída más profunda. Más capacidad (235k params vs 101k de shallow) = más sensibilidad al LR alto. Coherente con la regla teórica "más parámetros → menor LR óptimo".
- **Deeper** tiene el pico más bajo (0.9535). No por mayor caída sino por menor pico — coincide con la observación del Arch sweep: deeper sufre por la profundidad (gradientes peor propagados sin batch-norm).
- **Shallow** y **Base** están en el medio, con curvas casi idénticas.

#### val_loss CE — lo que la accuracy esconde
![[Notas/ejercicio 2/Segunda tanda de experimentos/Cross_LR_Opt_Arch/lr_opt_val_loss_4panels.png]]

El plot de val_loss captura algo que la accuracy diluye: el **quiebre de Adam@1e-2 es vertical**. Adam pasa de val_loss ≈ 0.17 (en su óptimo en 1e-3) a ≈ 0.25 (en 1e-2), un **50% peor** en sólo un orden de magnitud de LR. La accuracy del mismo punto sólo cae de 0.957 a 0.947 (~1 pp).

| arch    | val_loss CE Adam@1e-3 | val_loss CE Adam@1e-2 | empeoramiento relativo |
| ------- | --------------------- | --------------------- | ---------------------- |
| shallow | 0.170                 | 0.249                 | +46%                   |
| base    | 0.175                 | 0.215                 | +23%                   |
| wider   | 0.170                 | 0.228                 | +34%                   |
| deeper  | 0.182                 | 0.221                 | +21%                   |

→ El CE **detecta el modelo descalibrándose antes de que la accuracy lo refleje**. Cuando un modelo se rompe pero el argmax todavía aterriza en la clase correcta a veces, la accuracy no cambia mucho pero las probabilidades quedan totalmente desordenadas — la cross-entropy se entera de eso.

Para SGD: la curva baja (LR bajo, no convergió) y se estabiliza. SGD@1e-4 tiene val_loss ≈ 0.27–0.28 en las 4 archs, consistente con "no llegó al mínimo en presupuesto razonable".

#### Sobreajuste (gap val_loss − train_loss)

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/lr_opt_overfit_gap_4panels.png]]

Gap = `val_loss CE − train_loss CE`. Cuanto más alto, más memoriza train sin generalizar a val.

**Resultado contraintuitivo:** **Adam tiene el gap MÁS BAJO** consistentemente (~0.15), mientras Momentum y SGD están en ~0.17–0.18 cuando llegan al mínimo. Uno esperaría que el optimizer más agresivo (Adam) memorizara más, pero pasa lo contrario.

| optimizer | gap típico en óptimo  | comportamiento                                                          |
| --------- | --------------------- | ----------------------------------------------------------------------- |
| Adam      | ~0.15                 | estable, el menor                                                       |
| Momentum  | ~0.17–0.18            | sube con LR alto (1e-2 → 0.20+)                                         |
| SGD       | ~0.17 (en LR ≥ 5e-4)  | el de 1e-4 da ~0.07 pero es artefacto: ni train ni val convergieron     |

**Hipótesis:** Adam llega al mínimo de val_loss en 3–5 épocas y ES lo corta ahí (restaurando best_weights). Momentum y SGD necesitan más épocas para converger, durante las cuales el modelo sigue ajustando pesos a train sin mejorar val → mayor gap.

**Implicancia para el siguiente paso:** un futuro experimento de regularización (Pack C) tiene **menos margen para mejorar Adam** que Mom/SGD, porque Adam ya overfittea poco. La regularización movería más la aguja en Momentum o SGD si decidiéramos usarlos.

#### Conclusiones cualitativas

1. El factor definitorio es la arquitectura. Las 4 archs llegan a un techo ≈ **0.957–0.958** con la combinación correcta. **Más capacidad no es lo que falta.**

2. **El factor definitorio SI es la combinacion LR x OPT**. Diferencia entre la mejor cell (`wider+Adam@1e-3` = 0.9583) y la peor (`shallow+SGD@1e-4` = 0.9244) ≈ **3.4 pp**. Casi todo eso es atribuible a HP, no a arch. **Elegir bien LR y opt importa 4× más que elegir bien la arquitectura** en este problema.

3. **Adam domina pero es frágil.** Tiene el techo más alto (0.958) pero el rango operativo más estrecho (LR ∈ [5e-4, 1e-3]). SGD tolera LR ∈ [5e-4, 1e-2] sin romperse, pero su techo es ~0.951. Trade-off: *Adam es para cuando sabés el LR; SGD/Momentum perdonan errores en HP*.

4. **Adam overfittea MENOS, no más**. Esto refuerza que Adam@1e-3 sea el ganador: combina mejor accuracy, mejor calibración y menor sobreajuste.

5. **Hay un "ceiling" de ~0.96 sin regularización.** Ningún modelo del grid supera 0.96 val_acc. Para subir eso necesitás **más datos (Ej3)** o augmentation. Esa es la motivación natural del siguiente experimento.

6. **val_loss y accuracy NO son redundantes.** La accuracy mide aciertos del argmax; CE mide calibración de probabilidades. En este grid hay configuraciones con accuracy casi igual pero CE muy distinta (Adam@1e-2 acc=0.947 vs Adam@1e-3 acc=0.957, pero CE=0.25 vs 0.17 — caída de accuracy de 1pp esconde quiebre de CE de 50%).

7. **Top-10 dominado por Adam.** Sólo una celda no-Adam aparece en el top-10 (`shallow+momentum+lr=1e-2` en posición 9). **El optimizer importa más que la arch para llegar al top.** SGD no llega al top-10 con ninguna combinación.

### LR x ARCH

Misma data del stage 2 (60 cells = 5 LR × 3 opt × 4 arch) que la sección LR×OPT, rotada para responder una pregunta distinta: **dentro de cada optimizer, ¿el LR óptimo es el mismo en las 4 arquitecturas, o se desplaza?**

En la vista anterior fijábamos arch y comparábamos opts (4 paneles × 3 curvas). Acá fijamos opt y comparamos archs (3 paneles × 4 curvas). Si dentro de un panel las 4 curvas pican en la misma columna de LR, no hay interacción LR×ARCH para ese optimizer.

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/lr_arch_val_acc_3panels.png]]

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/lr_arch_heatmaps_3opts.png]]

**Best LR por (opt, arch)** — extraído de la tabla completa del stage 2. val_acc reportado como media ± std sobre **3 seeds × 5 folds = 15 corridas/celda**. val_loss CE = media sobre las mismas 15 corridas.

| opt          | arch    | best LR  | val_acc (15 corridas) | macro_f1 (15 corridas) | val_loss CE | best_epoch |
| ------------ | ------- | -------- | --------------------- | ---------------------- | ----------- | ---------- |
| **SGD**      | shallow | **1e-2** | 0.9509 ± 0.0053       | 0.8444                 | 0.1942      | 18.6       |
| **SGD**      | base    | **1e-2** | 0.9476 ± 0.0047       | 0.8407                 | 0.2061      | 9.1        |
| **SGD**      | wider   | **1e-2** | 0.9498 ± 0.0053       | 0.8445                 | 0.2019      | 8.9        |
| **SGD**      | deeper  | **1e-2** | 0.9463 ± 0.0036       | 0.8394                 | 0.2153      | 5.9        |
| **Momentum** | shallow | **1e-2** | 0.9543 ± 0.0062       | 0.8499                 | 0.2341      | 4.7        |
| **Momentum** | base    | **5e-3** | 0.9506 ± 0.0039       | 0.8437                 | 0.2121      | 3.0        |
| **Momentum** | wider   | **1e-2** | 0.9540 ± 0.0061       | 0.8478                 | 0.2191      | 3.2        |
| **Momentum** | deeper  | **5e-3** | 0.9500 ± 0.0072       | 0.8429                 | 0.2154      | 2.8        |
| **Adam**     | shallow | **1e-3** | 0.9572 ± 0.0041       | 0.8521                 | 0.1701      | 5.7        |
| **Adam**     | base    | **1e-3** | 0.9548 ± 0.0044       | 0.8493                 | 0.1751      | 3.5        |
| **Adam**     | wider   | **1e-3** | 0.9583 ± 0.0036       | 0.8537                 | 0.1701      | 3.4        |
| **Adam**     | deeper  | **1e-3** | 0.9535 ± 0.0078       | 0.8476                 | 0.1821      | 3.3        |

> **CSV fuente:** [`ejercicio2_experimentacion/analisis/cross_v1/lr_arch/best_lr_per_opt_arch.csv`](../../ejercicio2_experimentacion/analisis/cross_v1/lr_arch/best_lr_per_opt_arch.csv) — es la tabla reducida (12 filas) que sale de tomar, por cada (opt, arch), la celda con val_acc media más alta dentro de las 5 que varían LR. Incluye `best_lr`, `val_acc_mean_seeds_folds`, `val_acc_std_seeds_folds`, `macro_f1_mean_seeds_folds`, `val_loss_CE_mean_seeds_folds` y `best_epoch_mean`. Se genera con `scripts/cross_v1/plot_lr_arch.py` y es lo que usé para escribir la tabla de arriba.

#### Lectura por optimizer

**Adam — sin interacción LR×ARCH.** Las 4 archs pican en `1e-3`. La columna del óptimo es la misma; lo único que cambia es la **altura** del pico (wider 0.9583 > shallow 0.9572 > base 0.9548 > deeper 0.9535). Esto es coherente con lo que vimos en LR×OPT: las shapes de Adam son idénticas en las 4 archs, sólo cambia magnitud. La hipótesis previa de "Adam pica en 1e-3 en las 4 archs" se confirma.

**SGD — sin interacción LR×ARCH (pero con caveat).** Las 4 archs pican formalmente en `1e-2`, pero la curva es prácticamente plana desde `5e-4` en adelante (Δ entre `5e-4` y `1e-2` ≈ 0.002 en las 4 archs). El "óptimo en 1e-2" es real pero **marginal**: la diferencia con `5e-3` está dentro del SEM. SGD muestra el mismo plateau alto en las 4 arquitecturas. No hay shift del LR óptimo entre archs, y la única celda claramente sub-óptima es `1e-4` (no convergida en presupuesto).

**Momentum — interacción CLARA con arch.** Acá sí se desplaza el óptimo:

| arch    | best LR | val_acc en 5e-3 | val_acc en 1e-2 | Δ(5e-3 → 1e-2)                |
| ------- | ------- | --------------- | --------------- | ----------------------------- |
| shallow | **1e-2** | 0.9526 | **0.9543** | **+0.0017** (sigue mejorando) |
| wider   | **1e-2** | 0.9531 | **0.9540** | **+0.0009** (sigue mejorando) |
| base    | **5e-3** | **0.9506** | 0.9467 | **−0.0039** (cae)             |
| deeper  | **5e-3** | **0.9500** | 0.9425 | **−0.0075** (cae más fuerte)  |

El shift de `1e-2` a `5e-3` separa exactamente las archs por **última capa oculta**: `shallow` y `wider` (128 neuronas antes del softmax) toleran `1e-2`; `base` (64) y `deeper` (32) se rompen ahí. Es la misma observación de "última capa oculta como buffer" descrita en LR×OPT — acá lo confirma desde el ángulo "best LR per (opt, arch)".

#### Conexión con `IMPORTANTE_CORRELACIONES.md` (sección 3)

En la sección "Arch × LR" de [[IMPORTANTE_CORRELACIONES]] reportamos que **el ranking de archs cambia con el LR para Adam**: en `1e-3` wider es 1° y wider está 4° en `1e-2`. Eso no se contradice con lo de acá. Son dos cortes distintos de la misma grilla:

- **Acá** (LR×ARCH) preguntamos "fijo opt, ¿se mueve el **LR óptimo** al cambiar de arch?" → respuesta: no para Adam ni SGD, sí para Momentum.
- **Ahí** (IMPORTANTE_CORRELACIONES §3) preguntamos "fijo opt, ¿se mueve el **ranking de archs** al cambiar de LR?" → respuesta: sí para Adam (wider colapsa con LR alto).

Las dos cosas son consistentes: aunque el LR óptimo de Adam sea `1e-3` en las 4 archs, **fuera** de ese óptimo la pendiente de caída es distinta por arch (wider cae más fuerte, ver tabla de LR×OPT panel Adam). Es interacción de **magnitudes**, no de **ubicación del óptimo**.

#### Conclusiones de LR x ARCH

1. **El LR óptimo es estable entre archs para Adam (`1e-3`) y SGD (`1e-2`).** Esto es buena noticia metodológica: para esos dos optimizers se podría haber hecho un "sweep LR sobre una sola arch" y la decisión hubiera generalizado. El sweep LR original (que se hizo SGD-only) era válido en ese sentido, aunque por otra razón (LR×OPT) no generalizaba a Adam.

2. **Momentum sí tiene interacción.** Para Momentum, "el LR óptimo es 1e-2" depende de la arch. En `base` y `deeper` el óptimo cae a `5e-3`. Para Momentum **no se puede separar el sweep de LR del de arch**.

3. **El shift de Momentum está estructuralmente explicado.** Las archs que toleran `1e-2` con Momentum son las que tienen la última capa oculta más ancha (128). Las que se rompen tienen la última capa más estrecha (64, 32). El mecanismo (cada peso pesa más en la salida cuando la capa es estrecha → LR alto desestabiliza) es el mismo que ya identificamos en LR×OPT, sólo que ahora lo vemos también desde "best LR per arch".

4. **La configuración decidida (`shallow + Adam + 1e-3`) sigue justificada.** Adam@`1e-3` es óptimo en las 4 archs, así que la elección del LR no quedaría condicionada al ARCH específico que terminamos eligiendo. Eso refuerza que el centro es robusto al swap de arch dentro del rango medido.

### ARCH x OPT (marginalizado sobre LR)

Hasta acá miramos slices con la dimensión LR explícita (LR×OPT con arch fijo en cada panel; LR×ARCH con opt fijo en cada panel). Esta vista colapsa LR para responder una pregunta más simple: **dada una (arch, opt), si elegís el LR óptimo para esa combinación, ¿cuánto da?** Y de ahí: ¿hay arquitecturas que sean **especialistas** de un optimizer?

#### Por qué "best-over-LR" y no "mean-over-LR"

Marginalizar = colapsar la dimensión LR para tener un solo número por celda (arch, opt). Hay dos formas, y sólo una es honesta acá:

| Forma | Qué responde | Honesto? |
| ----- | ------------ | -------- |
| **Best-over-LR** (lo que usamos) | "Si elegís el LR óptimo para esta (arch, opt), cuánto da" | ✅ refleja el potencial real |
| Mean-over-LR | "Promedio sobre los 5 LRs medidos" | ❌ contaminado por LRs sub-óptimos; depende de qué LRs incluí en el barrido |

Esto también evita la trampa de las "slices 2D con LR fijo" descrita en [[IMPORTANTE_CORRELACIONES]] §7: si fijás LR=`1e-3` (óptimo de Adam), SGD entra con LR sub-óptimo y la comparación queda sesgada. Tomando best-over-LR cada combo entra con su mejor chance.

#### Heatmap

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/arch_opt_best_lr_heatmap.png]]

Cada celda = val_acc del mejor LR para esa combinación (arch, opt), reportada como media ± std sobre **15 corridas** (3 seeds × 5 folds). ★ marca la mejor arch dentro de cada columna (opt fijo); recuadro rojo marca el máximo global. La columna Adam es visiblemente la más amarilla — los 4 valores de Adam (0.9535 a 0.9583) están por encima de los 4 mejores de Momentum (0.9500 a 0.9543) y de SGD (0.9463 a 0.9509).

#### Lectura

**1. Adam domina las 4 filas.** Para las 4 arquitecturas, el mejor optimizer es Adam por entre +0.0029 (shallow: 0.9572 vs 0.9543) y +0.0042 (deeper: 0.9535 vs 0.9500). No hay arquitectura que prefiera Momentum o SGD sobre Adam. Esto sostiene la decisión "Adam" hecha en el optimizer sweep previo, ahora con la diferencia de que el optimizer sweep usó arch_base fijo y acá lo vemos en las 4 archs.

**2. Shallow es robusta entre optimizers; wider es especialista de Adam.**

| arch    | best opt | val_acc                             | comportamiento                                                                                                                            |
| ------- | -------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| shallow | Adam     | 0.9572 (★ con Mom 0.9543, ★ con SGD 0.9509) | gana en SGD y Momentum; muy cerca de wider con Adam (diff 0.0011, indistinguible — ver tiebreaker)                                       |
| base    | Adam     | 0.9548                              | sin podio en ningún opt — siempre 3°                                                                                                      |
| wider   | Adam     | **0.9583** (máx global)             | gana con Adam, **pero queda 2°/3° con SGD y Momentum**. Su rendimiento depende del optimizer                                              |
| deeper  | Adam     | 0.9535                              | última en las 3 columnas                                                                                                                  |

→ Shallow se comporta como una **arquitectura "todo terreno"**: ★ en SGD y ★ en Momentum, y segunda por 0.0011 en Adam. Wider tiene el techo absoluto más alto pero **sólo cuando combina con Adam**. Esta es exactamente la lectura de [[IMPORTANTE_CORRELACIONES]] §4, ahora visualizada en una sola figura.

**3. Deeper es la única arch consistentemente peor.** Última en las 3 columnas. Coherente con la observación de los sweeps previos: tener 3 capas ocultas sin batch-norm penaliza la propagación de gradientes y no la compensa la mayor profundidad. Es la única evidencia clara contra una arquitectura.

#### Conexión con la decisión final

El máximo global del grid es `wider + Adam@1e-3` con val_acc=0.9583. La configuración que terminamos eligiendo es `shallow + Adam@1e-3` con val_acc=0.9572. La diferencia (+0.0011) **no es estadísticamente significativa** según el [Arch tiebreaker](Segunda%20tanda%20de%20experimentos/Arch_tiebreaker/analisis.md) hecho con 15 seeds × k=5 (z=0.65). Con dos archs estadísticamente empatadas, **Occam decide shallow** (~101k params vs ~235k de wider).

Lectura en el lenguaje de esta sección: dentro del especialista (Adam), shallow y wider están empatadas; cambiando a Mom/SGD shallow gana sólo. Eso refuerza la elección de shallow desde dos ángulos: paridad con wider bajo el opt elegido, y dominancia bajo cualquier otro opt.

> **CSV fuente:** misma tabla que en LR×ARCH ([`best_lr_per_opt_arch.csv`](../../ejercicio2_experimentacion/analisis/cross_v1/lr_arch/best_lr_per_opt_arch.csv)) — el heatmap no es más que esa tabla reorganizada en matriz 4×3.