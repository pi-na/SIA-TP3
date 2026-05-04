1 a 4
5 a 7
9 a 12
13 a 15
16 a 19
20 a 23
25 y 26
27 a 29
30 y 31
32


---

# Análisis de dificultad, tiempo y distribución

**Total**: 33 slides (se eliminaron la conclusión del análisis del Ej 1 y la
slide de "Conclusiones del TP"). Las slides 8 (divider Ej 2), 24 (divider Ej 3)
y 33 (Gracias) no están asignadas a ningún bloque — quien esté hablando
antes/después las cubre.

**Tiempo objetivo**: 20 minutos de presentación + Q&A. Promedio ~1 min por slide.

## Tabla por bloque

| Bloque | Slides | # | Contenido | Dificultad | Q&A risk | Tiempo |
|---|---|---|---|---|---|---|
| **A** | 1–4    | 4 | Portada + setup Ej 1 + análisis del dataset (3 reglas, 80% derivable de la tabla) | Media | Bajo–Medio | 3 min |
| **B** | 5–7    | 3 | Lineal vs no-lineal MSE + threshold sweep + recomendación de umbral | Media-Alta | Medio | 3 min |
| **C** | 9–12   | 4 | Setup Ej 2 + fases + sweep arquitectura + sweep optimizador | Media | Medio | 3 min |
| **D** | 13–15  | 3 | **Decisiones de diseño**: activación + inicialización + optimizador | **Alta** | **Alto** | 4 min |
| **E** | 16–19  | 4 | Learning rate + dos tablas LR + convergencia y bias | Media | Medio | 3 min |
| **F** | 20–23  | 4 | Curvas + matriz base + final eval (cachetazo) + matriz test | **Alta** | **Alto** | 3 min |
| **G** | 25–26  | 2 | Hipótesis Ej 3 + el movimiento dominó (+10 pp) | Baja-Media | Bajo | 2 min |
| **H** | 27–29  | 3 | Matriz ganador + tabla de resultados Pack C + comparación visual | Media | Medio | 3 min |
| **I** | 30–31  | 2 | Curvas del ganador + qué ayudó / qué no | **Alta** | **Alto** | 3 min |
| **J** | 32     | 1 | Por qué no llegamos al 98 % | Media | Medio | 2 min |

### Por qué cada bloque tiene esa dificultad

- **Bloque D (decisiones)** — alta. Cualquier pregunta sobre matemática
  (vanishing gradient, fórmula de He vs Xavier, defaults de Adam) puede ir
  profundo. Hay que saber defender por qué ReLU + softmax y por qué Adam.
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

### Tomas — 2 bloques, 7 slides (~7 min)
- **Bloque D** (13–15): Decisiones de diseño (activación / init / optimizador)
- **Bloque F** (20–23): Curvas, matriz base, **cachetazo** y matriz final

> Le toca defender los dos puntos donde el Q&A puede ir más profundo
> (matemática de las decisiones y distribution shift). Sigue siendo el
> presentador con mayor carga técnica de Q&A.

### Katia — 3 bloques, 7 slides (~7 min)
- **Bloque B** (5–7): Lineal vs no-lineal + threshold sweep + recomendación
- **Bloque H** (27–29): Matriz ganador + tabla resultados + comparación visual
- **Bloque J** (32): Por qué no llegamos al 98 %

> Cierra la presentación, que es donde hay que vender la historia. Maneja
> threshold (Q&A: ROC/precision-recall) y la tabla de Pack C (Q&A: cómo se
> implementa cada técnica). Slide final pide carisma para cerrar con una
> hipótesis honesta sin sonar a derrota.

### Nicolas — 2 bloques, 8 slides (~8 min)
- **Bloque C** (9–12): Setup Ej 2 + fases + sweeps de arquitectura y optimizador
- **Bloque E** (16–19): Learning rate + tablas Fase 1/Fase 2 + convergencia/bias

> Casi todo es leer tablas y gráficos. El guion va a tener todo el contenido
> necesario. Si le preguntan algo difícil sobre learning rate (por ej. sobre
> el lr adaptativo del paper de optimizadores) Tomas o Katia rescatan: ya lo
> manejan por estar a cargo de los bloques D / F.

### Mateo — 3 bloques, 8 slides (~7 min)
- **Bloque A** (1–4): Portada + setup Ej 1 + análisis del dataset
- **Bloque G** (25–26): Hipótesis Ej 3 + el **dominó** del +10 pp
- **Bloque I** (30–31): Curvas del modelo ganador + qué ayudó / qué no

> Le tocan el arranque (portada + setup), el punto alto visual del Ej 3
> (el dominó del +10 pp se vende solo) y el cierre de Pack C donde sólo hay
> que leer los bullets de qué ayudó y qué no.
> **Sobre el bloque I**: cuando Mateo lo presente, Tomas ya habrá explicado
> distribution shift en el bloque F, así que el bullet de ``dropout: gana en
> val, pierde en test'' se lee verbatim y se puede derivar a Tomas si
> profundizan. Slide 30 (curvas) es factual: ``best\_ep ~10 vs 5 por L2 +
> aug, sin overfitting''.
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
| Tomas    | 2 | 7 | ~7 min | Alta (la lleva) |
| Katia    | 3 | 7 | ~7 min | Alta (la lleva) |
| Nicolas  | 2 | 8 | ~8 min | Media (con guion) |
| Mateo    | 3 | 8 | ~7 min | Baja (visual / scripted, con rescate) |
