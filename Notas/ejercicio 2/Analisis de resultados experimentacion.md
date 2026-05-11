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

### Comparativa de Optimizadores

Las secciones anteriores miraron interacciones (LR×OPT, LR×ARCH, ARCH×OPT). Esta cierra el análisis del optimizer con una vista **head-to-head**: ¿cuál es la mejor configuración de cada familia (SGD, Momentum, Adam) en el grid del stage 2, y qué se gana o pierde con cada una?

Para cada familia tomamos su mejor version sobre el stage 2 (mejor (arch, LR) por val_acc media). Es la misma operación de "best-over-LR" que en ARCH×OPT, pero llevada un nivel más arriba: aquí marginalizamos también sobre arch. Sólo nos quedan 3 filas, una por familia, para comparar lado a lado.

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/family_comparison_bars.png]]

| Familia      | best (arch @ LR)        | val_acc (15 corridas) | val_loss CE | best_epoch | gap (val−train) | sensibilidad al LR                               |
| ------------ | ----------------------- | --------------------- | ----------- | ---------- | --------------- | ------------------------------------------------ |
| **SGD**      | `shallow @ 1e-2`        | 0.9509 ± 0.0053       | 0.194       | 18.6       | 0.163           | plana (tolera LR ∈ [5e-4, 1e-2] sin romper)      |
| **Momentum** | `shallow @ 1e-2`        | 0.9543 ± 0.0062       | 0.234       | 4.7        | 0.215           | media (rompe en base/deeper @ 1e-2)              |
| **Adam**     | `wider   @ 1e-3`        | 0.9583 ± 0.0036       | 0.170       | 3.4        | 0.155           | aguda (ventana óptima estrecha en LR ∈ [5e-4, 1e-3]) |

**Lectura por métrica:**

- **val_acc** — Adam gana (+0.004 sobre Momentum, +0.007 sobre SGD). Su error estándar (SEM = std/√15 ≈ 0.0009) hace que la diferencia con SGD sea estadísticamente sólida; con Momentum es más justa pero igual a favor de Adam.
- **val_loss CE** — Adam también gana (0.170 vs 0.194 SGD vs 0.234 Mom). Acá el ranking cambia: Momentum @1e-2 calibra peor que SGD, porque a LR alto Momentum entrega probabilidades menos confiables aunque acierte el argmax. Ya lo vimos en LR×OPT: cuando un modelo "se rompe" el CE lo capta antes que la accuracy.
- **best_epoch** — Adam y Momentum convergen en 3–5 ép, SGD necesita 18.6. Es ~4× más lento que Adam en wall-clock.
- **gap (val−train)** — Adam tiene el gap más bajo (0.155), SGD intermedio (0.163), Momentum el más alto (0.215). El gap alto de Momentum en su mejor punto es coherente con la observación de LR×OPT: a `1e-2` Momentum llega rápido al óptimo de val pero sigue empujando train hacia 0 → memoriza más.

**Conclusión de la sección:** Adam gana en las 3 métricas que importan para decisión (accuracy, calibración, sobreajuste) **y** además es más rápido que SGD. La única ventaja real de Mom/SGD sería **robustez al HP** (toleran rangos más anchos de LR) — útil si no se hizo HP search, pero acá ese costo ya está pagado.

> **CSVs fuente:**
> - [`ejercicio2_experimentacion/analisis/cross_v1/family_comparison/best_of_family.csv`](../../ejercicio2_experimentacion/analisis/cross_v1/family_comparison/best_of_family.csv) — 3 filas, una por familia, con la mejor (arch, LR) y todas las métricas del head-to-head de arriba (incluye `overfit_gap`, `train_loss` y el LR ganador).
> - Generado por [`scripts/cross_v1/plot_family_comparison.py`](../../ejercicio2_experimentacion/scripts/cross_v1/plot_family_comparison.py), que también produce el plot de barras.

### Configuración óptima para este análisis

Las secciones anteriores midieron las interacciones; esta consolida la decisión. El método tiene 4 pasos: top-N → test estadístico → criterios de desempate → ganador.

#### Paso 1 — Top 10 del grid

Las 10 mejores celdas del stage 2 ordenadas por val_acc media (sobre 15 corridas = 3 seeds × 5 folds). Tabla extraída del [stage 2 completo](Segunda%20tanda%20de%20experimentos/Cross_LR_Opt_Arch/analisis.md#Top%20configs):

| #   | arch    | opt      | LR   | val_acc (±std)  | macro_f1 (±std) | val_loss CE | train_loss | gap (val−train) | best_epoch |
| --- | ------- | -------- | ---- | --------------- | --------------- | ----------- | ---------- | --------------- | ---------- |
| 1   | wider   | Adam     | 1e-3 | 0.9583 ± 0.0036 | 0.8537 ± 0.0050 | 0.1701      | 0.0148     | 0.1553          | 3.4        |
| 2   | shallow | Adam     | 1e-3 | 0.9572 ± 0.0041 | 0.8521 ± 0.0067 | 0.1701      | 0.0180     | 0.1521          | 5.7        |
| 3   | shallow | Adam     | 5e-4 | 0.9567 ± 0.0049 | 0.8518 ± 0.0075 | 0.1695      | 0.0203     | 0.1493          | 4.8        |
| 4   | wider   | Adam     | 5e-4 | 0.9553 ± 0.0050 | 0.8503 ± 0.0071 | 0.1775      | 0.0318     | 0.1457          | 2.2        |
| 5   | base    | Adam     | 1e-3 | 0.9548 ± 0.0044 | 0.8493 ± 0.0050 | 0.1751      | 0.0229     | 0.1521          | 3.5        |
| 6   | wider   | Adam     | 1e-4 | 0.9547 ± 0.0047 | 0.8500 ± 0.0060 | 0.1761      | 0.0187     | 0.1574          | 7.9        |
| 7   | shallow | Adam     | 1e-4 | 0.9546 ± 0.0044 | 0.8491 ± 0.0073 | 0.1746      | 0.0252     | 0.1494          | 15.9       |
| 8   | shallow | Adam     | 5e-3 | 0.9546 ± 0.0046 | 0.8493 ± 0.0069 | 0.1926      | 0.0242     | 0.1684          | 3.2        |
| 9   | shallow | Momentum | 1e-2 | 0.9543 ± 0.0062 | 0.8499 ± 0.0081 | 0.2341      | 0.0196     | 0.2146          | 4.7        |
| 10  | base    | Adam     | 5e-4 | 0.9541 ± 0.0050 | 0.8483 ± 0.0072 | 0.1741      | 0.0266     | 0.1475          | 2.9        |

**Observaciones:** 9 de las 10 son Adam; la única excepción es `shallow + momentum @ 1e-2` en la posición 9. Las primeras 4 están dentro de 0.003 puntos de val_acc — todas dentro de ~3·SEM, prácticamente empatadas.

> **Comparativa interactiva (60 celdas, sorteable y filtrable):** [`comparativa.html`](../../Notas/ejercicio%202/Segunda%20tanda%20de%20experimentos/comparativa_total/comparativa.html) (también accesible localmente en `Notas/ejercicio 2/Segunda tanda de experimentos/comparativa_total/comparativa.html`). El HTML tiene las 60 filas del stage 2 con todas las columnas (arch, opt, LR, val_acc±std, macro_f1±std, val_loss CE, train_loss, gap, best_epoch). Click en cualquier columna para sortear; pills preconfigurados arriba (🏆 mejor val_acc · ⚖️ mejor macro_f1 · 🎯 menor val_loss CE · ⚡ más rápido · 🚫 menos sobreajuste · 🧩 agrupar por arch · 🧪 agrupar por opt · 🔄 orden original); chips de abajo filtran por opt / arch / LR.

#### Paso 2 — ¿Las top-N son estadísticamente distinguibles?

Con `n = 15` corridas por celda, el error estándar de la media (SEM) sobre val_acc es ≈ `std / √15 ≈ 0.001`. Para distinguir dos celdas al 95% se necesita |z| > 1.96 con `z = (μ₁ − μ₂) / √(SEM₁² + SEM₂²) ≈ Δ/0.0014`. Concretamente: **diferencias < 0.003 son indistinguibles**.

Las dos primeras celdas — `wider+Adam@1e-3 = 0.9583` y `shallow+Adam@1e-3 = 0.9572` — están en Δ = 0.0011, claramente dentro de ese margen. Por eso lanzamos el [Arch tiebreaker](Segunda%20tanda%20de%20experimentos/Arch_tiebreaker/analisis.md) con **15 seeds × k=5 = 75 corridas/celda**, que amplió el N por 5 y permitía distinguir Δ ≥ 0.0014. Resultado:

- `arch_wider`:   0.9581 ± 0.0049  (SEM = 0.0006)
- `arch_shallow`: 0.9576 ± 0.0046  (SEM = 0.0006)
- **Δ = +0.0005, z = 0.65** → no distinguibles al 95% (necesitábamos |z| > 1.96).

La diferencia de 0.0011 que se veía con 3 seeds **era ruido de muestreo**. Conclusión: con la mejor evidencia disponible, las top-2 son **estadísticamente equivalentes**. Hay que recurrir a criterios de desempate.

#### Paso 3 — Criterios de desempate

Cuando hay empate estadístico, se elige por criterios secundarios. Estos son los disponibles y a quién favorece cada uno:

| criterio                       | favorece a                  | razón                                                                            |
| ------------------------------ | --------------------------- | -------------------------------------------------------------------------------- |
| **Occam** (cantidad de params) | shallow (~101k) sobre wider (~235k) | menos params → menos riesgo de overfitting; defensa oral simple; menor compute   |
| **velocidad** (best_epoch)     | Adam (3.4 ep) sobre Mom (4.7) / SGD (18.6) | menor wall-clock, menor compute                                                  |
| **calibración** (val_loss CE)  | Adam (0.170)                | val_loss más baja en su óptimo → probabilidades mejor calibradas                 |
| **sobreajuste** (gap)          | Adam (0.155)                | gap más bajo → mejor generalización                                              |
| **robustez al HP**             | SGD / Momentum              | toleran rangos más anchos de LR; valor sólo si no se hizo HP search              |
| **robustez al swap de arch**   | shallow                     | ★ en SGD y Momentum, empatada con wider en Adam → "todo terreno"                  |

#### Paso 4 — Decisión

**Para nuestro caso: Occam (params) + las 3 ventajas internas de Adam (calibración, velocidad, sobreajuste bajo) + robustez de shallow → la configuración elegida es:**

> **`arch_shallow` + Adam + LR=`1e-3` + batch=`64`**

Resumen del recorrido de la decisión:

1. Del grid 3D (LR × Opt × Arch), Adam domina las 4 archs → el optimizer es Adam.
2. Dentro de Adam, las top-2 (wider, shallow) están estadísticamente empatadas según el tiebreaker (z=0.65, n=75).
3. Entre dos archs empatadas, Occam elige la de menor capacidad → shallow.
4. El LR óptimo de Adam es `1e-3` en las 4 archs (LR×ARCH) → LR fijo en `1e-3`.
5. El batch óptimo para Adam@`1e-3` viene del [pre-experimento LR×Batch×Opt](Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md) → 64 (confirmado por la estrella batch del stage 2b, ver siguiente sección).

Esta es la **decisión-con-criterios-explícitos** que faltaba documentar como tal en una sola sección del .md. La configuración misma ya estaba en `IMPORTANTE_CORRELACIONES.md` y en `Cross_LR_Opt_Arch/analisis.md`, pero el árbol que llega a ella estaba disperso.

## Stage 2b -- Comparación de batch size con la mejor combinacion del stage 2

**Intención.** El batch_size óptimo para Adam@`1e-3` se decidió en el [pre-experimento LR×Batch×Opt](Segunda%20tanda%20de%20experimentos/Pre_LR_Batch_Opt/analisis.md) con sólo 3 puntos (16, 64, 256). Acá queremos confirmar que ese batch=64 es realmente el óptimo (y no un máximo entre puntos demasiado espaciados) **sobre el centro decidido** (`shallow + Adam + 1e-3`). Una "estrella" es eso: dejar todo fijo en el centro y variar **un solo factor** con resolución alta.

**Configuración.** 5 valores de batch (16, 32, 64, 128, 256), 3 seeds (42, 7, 13), k-folds=5 → 15 corridas/celda. Resto fijo en el centro.

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/stage2b_val_acc_vs_batch.png]]

| batch | val_acc (15 corridas) | macro_f1        | val_loss CE | best_epoch |
| ----- | --------------------- | --------------- | ----------- | ---------- |
| 16    | 0.9540 ± 0.0036       | 0.8477 ± 0.0061 | 0.1785      | 2.3        |
| 32    | 0.9568 ± 0.0054       | 0.8510 ± 0.0079 | 0.1700      | 3.9        |
| **64**    | **0.9572 ± 0.0041**       | **0.8521 ± 0.0067** | **0.1701**      | 5.7        |
| 128   | 0.9556 ± 0.0045       | 0.8502 ± 0.0069 | 0.1742      | 7.8        |
| 256   | 0.9548 ± 0.0045       | 0.8493 ± 0.0073 | 0.1771      | 11.2       |

**Análisis.** La curva es **unimodal con pico claro en batch=64**. A los costados:

- **Batches chicos (16):** val_acc cae 0.003 y val_loss sube 0.5%. Hipótesis: el ruido SGD es alto por minibatch (sólo 16 muestras), las direcciones de gradiente son más erráticas y el modelo no termina de afinar.
- **Batches grandes (256):** val_acc cae 0.002 y `best_epoch` sube a 11.2 (vs 5.7 en batch=64). Hipótesis: gradiente menos ruidoso pero más sesgado al óptimo de train → necesita más épocas para alcanzar val óptimo. Coherente con la regla LR×batch de la cátedra (con LR=`1e-3` fijo, batch=256 deja el producto LR×batch en un régimen sub-óptimo en el extremo "demasiado promediado").

**Conclusión.** `batch=64` es robustamente el óptimo del centro: gana en val_acc, val_loss y macro_f1 simultáneamente. La diferencia con batch=32 está dentro del SEM (Δ=0.0004), así que ambos serían defendibles, pero 64 es lo que ya teníamos decidido del pre-experimento y la estrella lo confirma. **Decisión final del Ej2 sostenida:** `shallow + Adam + 1e-3 + batch=64`.

---

# Convergencia y generalización de la configuración óptima

Una vez congelada la configuración (`shallow + Adam@1e-3 + batch=64`), queda medir **cómo aprende** y **qué tan bien generaliza**. Esta sección reporta tres cosas, en orden:

1. **Convergencia** sobre el CV interno: cómo bajan `train_loss` y `val_loss` por época, y a partir de qué época `val_loss` deja de bajar (overfit).
2. **Generalización interna** (CV sobre `digits.csv`): las 4 métricas (accuracy, precision macro, recall macro, F1 macro) + CE en val vs train del fold, promediadas sobre las 15 corridas (3 seeds × 5 folds) del stage 2.
3. **Generalización externa** sobre `digits_test.csv`: corremos `ejercicio3/final_eval.py` con la config congelada y 3 seeds, evaluando una sola vez sobre el set de producción que no se tocó durante HP search.

**Datos fuente:**
- Convergencia (15 epoch_history.csv): `ejercicio2_experimentacion/output/cross_v1/stage2/stage2_arch_shallow_adam_lr1e-3_bs64_seed{42,7,13}/.../epoch_history.csv`. Agregados en `ejercicio2_experimentacion/analisis/cross_v1/optimal_config/optimal_convergence_table.csv`.
- CV interno (3 summary.csv): `ejercicio2_experimentacion/output/cross_v1/stage2/stage2_arch_shallow_adam_lr1e-3_bs64_seed{42,7,13}/summary.csv`. Agregado en `ejercicio2_experimentacion/analisis/cross_v1/optimal_config/cv_internal_summary.csv`.
- Test (3 corridas): `ejercicio2_experimentacion/output/final_eval_ej2/final_eval_final_ej2_seed{42,7,13}_*/test_metrics.csv` + `test_confusion_matrix.csv`. Config: `ejercicio2_experimentacion/configs/final_config_ej2.json`. Agregado en `optimal_config/test_summary.csv` y `test_per_class.csv`.
- Scripts: `ejercicio2_experimentacion/scripts/cross_v1/plot_optimal_convergence.py` y `analyze_optimal_generalization.py`.

## A. Convergencia (sobre CV interno)

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/optimal_convergence.png]]

**Cómo se construyó:** para cada una de las 15 corridas del CV interno (3 seeds × 5 folds) cargamos su `epoch_history.csv` y, **para cada época**, promediamos las 4 series (`train_loss`, `val_loss`, `train_acc`, `val_acc`) sobre todas las corridas que llegaron a esa época. Las bandas son ± 1 std a nivel de época. El plot se corta donde menos de 8 de las 15 corridas siguen vivas (las demás ya cortó early stopping con `patience=20`).

**Lo que se ve:**

- **train_loss** baja muy rápido: de 0.17 (epoch 0) a ≈ 0.02 en epoch 10. En la última época con cobertura, está ≈ 0.005 (modelo memorizando train completo).
- **val_loss** baja a su mínimo de ≈ 0.170 alrededor del **epoch 5.7** (`best_epoch` promedio sobre las 15 corridas, std ± 1.6) y a partir de ahí sube ligeramente — el modelo sigue ajustando train pero deja de mejorar en val.
- La separación clara entre train_loss y val_loss arranca alrededor del epoch 4-5: a partir de ese punto, lo que aprende sobre train **ya no transfiere a val**. Es el "techo" del modelo en este dataset sin regularización.
- **train_acc** llega a 1.0 (memoria perfecta) alrededor del epoch 10; **val_acc** se estabiliza en ≈ 0.957.

**Lectura defensiva.** El early stopping con `patience=20` sobre val_loss es **suficientemente paciente** para no cortar artificialmente: en las 15 corridas, val_loss alcanzó su mínimo y se quedó 20 épocas chequeando si bajaba más. No bajó. La convergencia en 5-6 épocas no es un artefacto de cortar temprano; es la "velocidad real" de Adam@`1e-3` con este dataset.

**Cuándo cortó early stopping y por qué el eje X termina cerca de la época 25.** Cada corrida del CV interno terminó cuando ES disparó, no por agotar `max_epochs`. ES corta cuando val_loss no mejora durante `patience=20` épocas consecutivas después del último mínimo, así que la **época de corte ≈ best_epoch + patience**. Con `best_epoch` distribuido en `[3, 9]` (min/max) y media `5.7 ± 1.6` sobre las 15 corridas, la **época de corte** quedó distribuida en `[23, 29]` con media **25.7 ± 1.6** — exactamente `best_epoch + 20`. Resumen numérico:

| Estadístico                | best_epoch (sobre 15 corridas) | stop_epoch (corte por ES, sobre 15 corridas) |
| -------------------------- | ------------------------------ | -------------------------------------------- |
| mean                       | 5.7                            | 25.7                                         |
| std                        | 1.6                            | 1.6                                          |
| min                        | 3                              | 23                                           |
| max                        | 9                              | 29                                           |
| Llegaron a max_epochs=40   | —                              | 0 / 15                                       |

**Ninguna** corrida llegó al límite duro `max_epochs=40` (configurado en el stage 2 para Adam@`1e-3`). El plot se trunca alrededor del epoch 25 porque a partir de ahí **menos de 8 de las 15 series siguen vivas** (algunas cortaron antes en epoch 23-24), y el promedio + banda de std deja de ser estable con tan pocas muestras — por eso recortamos visualmente ahí. La parte después de la época 25 existe en algunas curvas individuales pero promediar 2-3 contra 15 sería engañoso a nivel de bandas.

**Implicancia metodológica:** el `patience=20` auditado en el plan de cross_v1 está bien calibrado para este problema. Si lo hubiéramos puesto en 10, algunas corridas con `best_epoch=9` habrían cortado prematuramente (9 + 10 = epoch 19, con el riesgo de tirar a la basura mejoras tardías). Si lo hubiéramos puesto en 50, no cambiaría nada: ya verificamos que después del epoch ~6 val_loss sólo sube, así que más paciencia sería puro desperdicio de compute (~30 épocas extras por corrida × 15 corridas).

## B. Generalización interna (CV sobre `digits.csv`)

El CV interno usa **k-folds estratificado con k=5** sobre `digits.csv`: cada corrida entrena con 4 folds (~9956 samples) y valida sobre el 5° (~2493 samples). Reportamos las 4 métricas + CE del **fold de validación**, promediadas sobre las 15 corridas. Para train, sólo se almacenó accuracy y CE (las 4 métricas no se guardan en el `summary.csv` durante stage 2 — diseño del runner). El gap de accuracy y CE alcanza para diagnosticar overfit.

| Métrica                  | Train (CV, 15 corridas) | Val (CV, 15 corridas) | Gap val − train         |
| ------------------------ | ----------------------- | --------------------- | ----------------------- |
| **accuracy**             | 0.9979 ± 0.0023         | 0.9572 ± 0.0041       | −0.041                  |
| **macro_precision**      | (no almacenada)         | 0.8546 ± 0.0062       | —                       |
| **macro_recall**         | (no almacenada)         | 0.8502 ± 0.0078       | —                       |
| **macro_F1**             | (no almacenada)         | 0.8521 ± 0.0067       | —                       |
| **CE loss**              | 0.0180 ± 0.0104         | 0.1701 ± 0.0170       | +0.152                  |
| **best_epoch**           | —                       | 5.7 ± 1.6             | —                       |

> **Promedios:** todos los valores son media sobre **15 corridas = 3 seeds (42, 7, 13) × 5 folds estratificados**. El ± reportado es desviación estándar sobre esas 15 corridas (no SEM). El SEM correspondiente sería std/√15 ≈ std/3.87.

**Interpretación.**

- El modelo **memoriza train casi perfectamente** (train_acc = 0.9979): tiene capacidad de sobra para este dataset.
- En val baja a **0.957 accuracy** con 0.852 macro_F1. La brecha entre accuracy y macro_F1 (~10 puntos) es **estructural** del dataset: la clase 5 está fuertemente sub-representada en `digits.csv` (271 ejemplos vs ~1500 las otras), por lo que su precision/recall son bajos y arrastran el macro_F1 hacia abajo aunque la accuracy se vea bien (ya lo discutimos en el bloque de Métricas al inicio del Stage 2).
- El **gap train→val** es **+0.041 en accuracy** y **+0.152 en CE**: overfit moderado. Es lo esperado para un MLP con ~101k parámetros entrenando sobre ~10k samples sin regularización.

**Diagnóstico defensivo.** El val_acc de 0.957 **no es el techo de Adam ni del LR** — es el techo de este dataset sin regularización. Lo confirman tres números: train_acc=0.9979 (capacidad de sobra), CE gap=+0.152 (memoriza), best_epoch=5.7 con patience=20 (no es corte temprano artificial).

## C. Generalización externa (sobre `digits_test.csv`)

**Procedimiento.** Tomamos la config congelada (`final_config_ej2.json`: arch=[784, 128, 10], Adam@`1e-3`, batch=64, epochs=40, patience=20, sin regularización), y la corrimos **3 veces** con seeds 42, 7, 13 usando `ejercicio3/final_eval.py`. Cada corrida:

1. Entrena con todo `digits.csv` (12 449 samples) usando un split 90/10 para early stopping (no k-folds — esto es producción, no validacion). Usamos 90/10 para maximizar training. Hacemos split estratificado (y le afecta el seed)
2. Predice sobre `digits_test.csv` (2497 samples) **una sola vez** con los `best_weights` restaurados.
3. Reporta accuracy, macro precision/recall/F1, weighted F1, y matriz de confusión.

**Resultado agregado** (mean ± std sobre 3 seeds):

| Métrica              | Val CV (referencia interna) | **Test** (digits_test.csv)   | Δ (val CV − test) |
| -------------------- | --------------------------- | ---------------------------- | ----------------- |
| **accuracy**         | 0.9572 ± 0.0041             | **0.8529 ± 0.0034**          | **+0.1043**       |
| **macro_precision**  | 0.8546 ± 0.0062             | **0.7706 ± 0.0039**          | +0.0840           |
| **macro_recall**     | 0.8502 ± 0.0078             | **0.8485 ± 0.0034**          | +0.0017           |
| **macro_F1**         | 0.8521 ± 0.0067             | **0.8062 ± 0.0034**          | +0.0459           |
| **weighted_F1**      | —                           | **0.8103 ± 0.0033**          | —                 |

**Esto es una caída enorme en accuracy: 10 puntos.** Antes de interpretarla, hay que entender por qué ocurre.

### MALA Distribución de clases 

Conteo de clases en cada CSV:

| clase | `digits.csv` (train) | `digits_test.csv` (test) | `more_digits.csv` (Ej3, no usado acá) |
| ----- | -------------------- | ------------------------ | ------------------------------------- |
| 0     | 1480                 | 245                      | 1776                                  |
| 1     | 1685                 | 283                      | 2022                                  |
| 2     | 1489                 | 258                      | 1787                                  |
| 3     | 1532                 | 252                      | 1839                                  |
| 4     | 1460                 | 245                      | 1752                                  |
| 5     | **271** (sub-repr)   | 223                      | 542                                   |
| 6     | 1479                 | 239                      | 1775                                  |
| 7     | 1566                 | 257                      | 1879                                  |
| **8** | **0 (ausente!)**     | 243 (9.7% del test)      | 585                                   |
| 9     | 1487                 | 252                      | 1784                                  |
| **N** | **12 449**           | **2497**                 | 15 741                                |

**Hallazgo clave: `digits.csv` no contiene ningún ejemplo de la clase 8.** El modelo nunca vio un 8 durante entrenamiento. Pero `digits_test.csv` tiene **243 ejemplos de clase 8 (9.7% del test)**. Esos 243 ejemplos son necesariamente mal clasificados: precision=0, recall=0, F1=0 para la clase 8 (verificado en la tabla per-clase de abajo).

### Confusion matrix sobre test

![[Segunda tanda de experimentos/Cross_LR_Opt_Arch/optimal_test_confusion_matrix.png]]

Cada celda muestra `recall normalizado por fila` (cuenta absoluta entre paréntesis), promediada sobre las 3 seeds. La fila 8 está enmarcada en rojo: todos los 8s reales se reparten entre las clases que el modelo SÍ vio (sobre todo 3 y 9, los visualmente más parecidos a un 8). La diagonal de la fila 8 es 0.

### Métricas por clase en test (mean sobre 3 seeds)

| clase | precision         | recall            | F1                | support (test) |
| ----- | ----------------- | ----------------- | ----------------- | -------------- |
| 0     | 0.884             | 0.995             | 0.936             | 245            |
| 1     | 0.942             | 0.988             | 0.964             | 283            |
| 2     | 0.862             | 0.943             | 0.900             | 258            |
| 3     | 0.727             | 0.983             | 0.835             | 252            |
| 4     | 0.885             | 0.961             | 0.921             | 245            |
| 5     | 0.789             | 0.803             | 0.796             | 223            |
| 6     | 0.897             | 0.960             | 0.927             | 239            |
| 7     | 0.934             | 0.938             | 0.936             | 257            |
| **8** | **0.000**         | **0.000**         | **0.000**         | 243            |
| 9     | 0.786             | 0.915             | 0.845             | 252            |

→ La clase 8 mete un **0 en cada una de las tres métricas macro**. Como el macro promedia las 10 clases con peso igual, un 0 puro arrastra el macro_F1 macro hacia abajo de forma agresiva. Pero hay un detalle más fino: **precision** cae más que **recall** porque, además de no poder predecir 8, el modelo "vomita" otros dígitos (típicamente 3 y 9) sobre los 8s reales — eso infla los falsos positivos de las clases 3 y 9 y baja sus precisions también.

### Que pasa si exlcuimos la clase 8?? La que no tiene representación

Si **excluimos del test los casos donde el ground truth es clase 8** (242–243 ejemplos por corrida, ≈ 9.7% del set), la accuracy sube a:

> **Test acc excluyendo clase 8 = 0.9448 ± 0.0030** (sobre las 9 clases que sí están en train)

Comparado con val_acc CV = 0.9572 ± 0.0041, el **gap residual real es sólo +0.0124** (~1.2 puntos), perfectamente consistente con un ligero shift de distribución entre `digits.csv` y `digits_test.csv` en las clases que **sí** están en ambos (la clase 5, por ejemplo, está al 2.2% en train y 8.9% en test — el modelo entrenó con ese desbalance y al evaluarse en una distribución más balanceada pierde un poco).

### Lectura defensiva

- El CV interno **no mintió** sobre lo que medía: predijo bien la generalización **dentro de la distribución de `digits.csv`** (~0.957). El gap residual al test (excluyendo la clase ausente) es de sólo 1.2 puntos, completamente razonable.
- Lo que falla **no es el modelo ni la metodología de HP search** — es que `digits.csv` está **incompleto** respecto al universo que `digits_test.csv` representa. Sin clase 8 en train, ese 9.7% del test es **imposible** de acertar.
- Esto es exactamente lo que **motiva el Ej3**: `more_digits.csv` tiene 585 ejemplos de clase 8 (y refuerza la 5 con 542 más). Sumarlo al entrenamiento debería cerrar la mayor parte de la caída.

**Pronóstico para Ej3** (testeable en la próxima corrida):
- Si la única causa del gap fuera la clase 8 ausente, sumar `more_digits.csv` debería llevar el test_acc a ~0.94–0.95.
- Si hay además un shift residual de distribución (clase 5 sub-representada en train, clases más difíciles, etc.), podríamos quedar en 0.92–0.94 incluso con la clase 8 cubierta. La diferencia entre estos dos escenarios es lo que cuantifica el Ej3.

**Síntesis:** la convergencia es sana (5-6 épocas, no es corte prematuro). La generalización **dentro de la distribución de digits.csv es del 95.7%**, consistente con el techo esperable de un MLP sin regularización sobre 12.5k samples. La generalización **fuera de la distribución de digits.csv** cae a 85.3% por una razón **estructural del dataset** (falta una clase), no del modelo. La intervención correcta no es "más HP search" ni "más optimizador", es **más datos** — exactamente la motivación de Ej3.
