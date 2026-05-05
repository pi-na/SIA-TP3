1 a 5
6 a 8
10 a 13
14 a 17
18 a 21
22 a 26
28 y 29
30 a 32
33 y 34
35


---

# Análisis de dificultad, tiempo y distribución

**Total**: 36 slides. Las slides 9 (divider Ej 2), 27 (divider Ej 3) y
36 (Gracias) no están asignadas a ningún bloque — quien esté hablando
antes/después las cubre.

> **Cambios desde la versión original (33 slides):**
> - **+1 en bloque A**: nueva slide "intuición de cómo ajusta la sigmoide"
>   (foto + insight) entre el análisis del dataset y el comparativo MSE
>   del Ej 1 → bloque A pasa a 5 slides (1–5).
> - **+1 en bloque D**: nueva slide "ReLU y Softmax — visualización"
>   (dos gráficos representativos + mini-explicación) inmediatamente
>   después de la slide de función de activación → bloque D pasa a
>   4 slides (14–17). Asignada a **Tomas**.
> - **+1 en bloque F**: nueva slide "Distribución de clases en train vs
>   test" (tabla de conteos por clase + hallazgos) entre la matriz base
>   y el cachetazo → bloque F pasa a 5 slides (22–26). Asignada a
>   **Tomas**. Explica numéricamente por qué el drop val→test es de
>   exactamente $\sim 10$ pp (clase 8 ausente en \file{digits.csv}).
> - El resto de los bloques mantiene contenido pero queda corrido
>   según corresponda.

**Tiempo objetivo**: 20 minutos de presentación + Q&A. Promedio ~1 min por slide.

## Tabla por bloque

| Bloque | Slides | # | Contenido | Dificultad | Q&A risk | Tiempo |
|---|---|---|---|---|---|---|
| **A** | 1–5    | 5 | Portada + setup Ej 1 + análisis del dataset (3 reglas) + intuición sigmoide | Media | Bajo–Medio | 4 min |
| **B** | 6–8    | 3 | Lineal vs no-lineal MSE + threshold sweep + recomendación de umbral | Media-Alta | Medio | 3 min |
| **C** | 10–13  | 4 | Setup Ej 2 + fases + sweep arquitectura + sweep optimizador | Media | Medio | 3 min |
| **D** | 14–17  | 4 | **Decisiones de diseño**: activación + visualización ReLU/softmax + inicialización + optimizador | **Alta** | **Alto** | 4 min |
| **E** | 18–21  | 4 | Learning rate + dos tablas LR + convergencia y bias | Media | Medio | 3 min |
| **F** | 22–26  | 5 | Curvas + matriz base + distribución de clases + final eval (cachetazo) + matrices val/test | **Alta** | **Alto** | 4 min |
| **G** | 28–29  | 2 | Hipótesis Ej 3 + el movimiento dominó (+10 pp) | Baja-Media | Bajo | 2 min |
| **H** | 30–32  | 3 | Matriz ganador + tabla de resultados Pack C + comparación visual | Media | Medio | 3 min |
| **I** | 33–34  | 2 | Curvas del ganador + qué ayudó / qué no | **Alta** | **Alto** | 3 min |
| **J** | 35     | 1 | Por qué no llegamos al 98 % | Media | Medio | 2 min |

### Por qué cada bloque tiene esa dificultad

- **Bloque D (decisiones)** — alta. Cualquier pregunta sobre matemática
  (vanishing gradient, fórmula de He vs Xavier, defaults de Adam) puede ir
  profundo. Hay que saber defender por qué ReLU + softmax y por qué Adam.
  La nueva slide de visualización ayuda a anclar la intuición en las
  formas de ReLU y softmax antes de entrar a init y optimizador.
- **Bloque F (cachetazo + shift)** — alta. Introduce el concepto de
  *distribution shift*. Q&A puede pedir tipos (covariate, label), por qué no
  es overfitting, por qué la clase 8 colapsó.
- **Bloque I (qué ayudó / qué no)** — alta. La pregunta natural es "por qué
  dropout perdió en test si ganó en val", que requiere entender shift y por
  qué reduce varianza al fold pero no al shift.
- **Bloque B (threshold)** — media-alta. Pueden preguntar precision/recall
  trade-off, ROC, calibración.
- **Bloques A, C, E, H, J** — media. Mucho contenido factual y tablas; Q&A
  manejable con conocimiento del config.
- **Bloque G (dominó)** — baja-media. Es el momento "ganador" del TP, fácil
  de presentar; la pregunta más difícil es por qué se confirmó la hipótesis.

## Distribución propuesta (4 integrantes)

Criterio: Tomas y Katia toman los bloques de **alta dificultad / alto Q&A risk**.
Nicolas toma bloques medios donde el guion lleva el peso. Mateo toma los bloques
con más respaldo visual (gráficos / números grandes) y menor superficie técnica.

### Tomas — 2 bloques, 9 slides (~9 min)
- **Bloque D** (14–17): Decisiones de diseño (activación + viz ReLU/softmax + init + optimizador)
- **Bloque F** (22–26): Curvas, matriz base, distribución de clases, **cachetazo** y matrices val vs test

> Le toca defender los dos puntos donde el Q&A puede ir más profundo
> (matemática de las decisiones y distribution shift). La slide nueva
> de visualización ReLU/softmax (15) aporta intuición visual antes de
> que aparezcan las fórmulas de He/Xavier en la 16. Sigue siendo el
> presentador con mayor carga técnica de Q&A.

### Katia — 3 bloques, 7 slides (~7 min)
- **Bloque B** (6–8): Lineal vs no-lineal + threshold sweep + recomendación
- **Bloque H** (30–32): Matriz ganador + tabla resultados + comparación visual
- **Bloque J** (35): Por qué no llegamos al 98 %

> Cierra la presentación, que es donde hay que vender la historia. Maneja
> threshold (Q&A: ROC/precision-recall) y la tabla de Pack C (Q&A: cómo se
> implementa cada técnica). Slide final pide carisma para cerrar con una
> hipótesis honesta sin sonar a derrota.

### Nicolas — 2 bloques, 8 slides (~8 min)
- **Bloque C** (10–13): Setup Ej 2 + fases + sweeps de arquitectura y optimizador
- **Bloque E** (18–21): Learning rate + tablas Fase 1/Fase 2 + convergencia/bias

> Casi todo es leer tablas y gráficos. El guion va a tener todo el contenido
> necesario. Si le preguntan algo difícil sobre learning rate (por ej. sobre
> el lr adaptativo del paper de optimizadores) Tomas o Katia rescatan: ya lo
> manejan por estar a cargo de los bloques D / F.

### Mateo — 3 bloques, 9 slides (~7 min)
- **Bloque A** (1–5): Portada + setup Ej 1 + análisis del dataset + intuición sigmoide
- **Bloque G** (28–29): Hipótesis Ej 3 + el **dominó** del +10 pp
- **Bloque I** (33–34): Curvas del modelo ganador + qué ayudó / qué no

> Le tocan el arranque (portada + setup), el punto alto visual del Ej 3
> (el dominó del +10 pp se vende solo) y el cierre de Pack C donde sólo hay
> que leer los bullets de qué ayudó y qué no.
> **Sobre el bloque I**: cuando Mateo lo presente, Tomas ya habrá explicado
> distribution shift en el bloque F, así que el bullet de ``dropout: gana en
> val, pierde en test'' se lee verbatim y se puede derivar a Tomas si
> profundizan. La slide de curvas (32) es factual: ``best\_ep ~10 vs 5 por
> L2 + aug, sin overfitting''.
> **Sobre la slide 5 (intuición sigmoide)**: es la slide nueva del bloque A.
> El insight box explica por qué el modelo no-lineal va a ganar en el MSE
> que viene en el bloque B. Mateo la lee verbatim; si profundizan en por
> qué el lineal sale del rango $[0,1]$, deriva a Katia.
> **Q&A sobre slide 4 (análisis del dataset)**: la tabla muestra TP=695,
> FN=174, FP=0. Mateo lee esos números literales (de ahí sale el ``80\%
> del fraude con tres if y cero FP'') y deriva al equipo si profundizan en
> la elección de features.

## Plan de Q&A

| Tipo de pregunta | Quién contesta primero |
|---|---|
| Detalle matemático (gradiente, Adam, init) | **Tomas** |
| Distribution shift / tipos / por qué dropout no transfirió | **Tomas** |
| Threshold / precision-recall / ROC / costos | **Katia** |
| Pack C, cada técnica de regularización, cómo se implementa | **Katia** |
| Tabla de sweeps (LR, arch, optimizador): valores y elección | **Nicolas** (con guion) → Tomas si profundizan |
| Análisis del dataset / por qué excluir features | **Mateo** lee la tabla → **Katia** rescata |
| Próximos pasos / por qué no llegamos al 98 % | **Katia** |

## Resumen de balance

| Persona | # bloques | # slides | Tiempo aprox | Carga Q&A |
|---|---|---|---|---|
| Tomas    | 2 | 9 | ~9 min | Alta (la lleva) |
| Katia    | 3 | 7 | ~7 min | Alta (la lleva) |
| Nicolas  | 2 | 8 | ~8 min | Media (con guion) |
| Mateo    | 3 | 9 | ~7 min | Baja (visual / scripted, con rescate) |
