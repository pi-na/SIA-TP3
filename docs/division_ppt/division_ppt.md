1 a 5
6 a 8
10 a 13
14 a 16
17 a 20
21 a 24
26 y 27
28 a 30
31 y 32
33 y 34


---

# Análisis de dificultad, tiempo y distribución

**Total**: 35 slides. Las slides 9 (divider Ej 2), 25 (divider Ej 3) y 35 (Gracias) no
están asignadas a ningún bloque — quien esté hablando antes/después las cubre.

**Tiempo objetivo**: 20 minutos de presentación + Q&A. Promedio ~1 min por slide.

## Tabla por bloque

| Bloque | Slides | # | Contenido | Dificultad | Q&A risk | Tiempo |
|---|---|---|---|---|---|---|
| **A** | 1–5    | 5 | Portada + setup Ej 1 + análisis del dataset (3 reglas) + cuadrito 80% | Media | Bajo–Medio | 3 min |
| **B** | 6–8    | 3 | Lineal vs no-lineal MSE + threshold sweep + recomendación de umbral | Media-Alta | Medio | 3 min |
| **C** | 10–13  | 4 | Setup Ej 2 + fases + sweep arquitectura + sweep optimizador | Media | Medio | 3 min |
| **D** | 14–16  | 3 | **Decisiones de diseño**: activación + inicialización + optimizador | **Alta** | **Alto** | 4 min |
| **E** | 17–20  | 4 | Learning rate + dos tablas LR + convergencia y bias | Media | Medio | 3 min |
| **F** | 21–24  | 4 | Curvas + matriz base + final eval (cachetazo) + matriz test | **Alta** | **Alto** | 3 min |
| **G** | 26–27  | 2 | Hipótesis Ej 3 + el movimiento dominó (+10 pp) | Baja-Media | Bajo | 2 min |
| **H** | 28–30  | 3 | Matriz ganador + tabla de resultados Pack C + comparación visual | Media | Medio | 3 min |
| **I** | 31–32  | 2 | Curvas del ganador + qué ayudó / qué no | **Alta** | **Alto** | 3 min |
| **J** | 33–34  | 2 | Por qué no llegamos al 98 % + conclusiones | Media | Medio | 3 min |

### Por qué cada bloque tiene esa dificultad

- **Bloque D (decisiones)** — alta. Cualquier pregunta sobre matemática (vanishing
  gradient, fórmula de He vs Xavier, defaults de Adam) puede ir profundo. Hay que
  saber defender por qué ReLU + softmax y por qué Adam.
- **Bloque F (cachetazo + shift)** — alta. Introduce el concepto de
  *distribution shift*. Q&A puede pedir tipos (covariate, label), por qué no es
  overfitting, por qué la clase 8 colapsó.
- **Bloque I (qué ayudó / qué no)** — alta. La pregunta natural es "por qué
  dropout perdió en test si ganó en val", que requiere entender shift y por qué
  reduce varianza al fold pero no al shift.
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
- **Bloque D** (14–16): Decisiones de diseño (activación / init / optimizador)
- **Bloque F** (21–24): Curvas, matriz base, **cachetazo** y matriz final

> Le toca defender los dos puntos donde el Q&A puede ir más profundo
> (matemática de las decisiones y distribution shift). Sigue siendo el
> presentador con mayor carga técnica de Q&A.

### Katia — 3 bloques, 8 slides (~8 min)
- **Bloque B** (6–8): Lineal vs no-lineal + threshold sweep + recomendación
- **Bloque H** (28–30): Matriz ganador + tabla resultados + comparación visual
- **Bloque J** (33–34): Por qué no 98 % + conclusiones

> Cierra la presentación, que es donde hay que vender la historia. Maneja
> threshold (Q&A: ROC/precision-recall) y la tabla de Pack C (Q&A: cómo se
> implementa cada técnica). Las conclusiones piden carisma.

### Nicolas — 2 bloques, 8 slides (~8 min)
- **Bloque C** (10–13): Setup Ej 2 + fases + sweeps de arquitectura y optimizador
- **Bloque E** (17–20): Learning rate + tablas Fase 1/Fase 2 + convergencia/bias

> Casi todo es leer tablas y gráficos. El guion va a tener todo el contenido
> necesario. Si le preguntan algo difícil sobre learning rate (por ej. sobre
> el lr adaptativo del paper de optimizadores) Tomas o Katia rescatan: ya lo
> manejan por estar a cargo de los bloques D / F.

### Mateo — 3 bloques, 9 slides (~8 min)
- **Bloque A** (1–5): Portada + setup Ej 1 + análisis del dataset + cuadrito 80 %
- **Bloque G** (26–27): Hipótesis Ej 3 + el **dominó** del +10 pp
- **Bloque I** (31–32): Curvas del modelo ganador + qué ayudó / qué no

> Le tocan el arranque (portada + setup), el punto alto visual del Ej 3
> (el dominó del +10 pp se vende solo) y el cierre de Pack C donde sólo hay
> que leer los bullets de qué ayudó y qué no.
> **Sobre el bloque I**: cuando Mateo lo presente, Tomas ya habrá explicado
> distribution shift en el bloque F, así que el bullet de ``dropout: gana en
> val, pierde en test'' se lee verbatim y se puede derivar a Tomas si
> profundizan. Slide 31 (curvas) es factual: ``best\_ep ~10 vs 5 por L2 +
> aug, sin overfitting''.
> **Q&A**: si le preguntan algo sobre el análisis del dataset (slide 4),
> puede contestar lo que está literal en el cuadrito (3 reglas, 80 %, FP=0)
> y derivar al resto del equipo si profundizan.

## Plan de Q&A

| Tipo de pregunta | Quién contesta primero |
|---|---|
| Detalle matemático (gradiente, Adam, init) | **Tomas** |
| Distribution shift / tipos / por qué dropout no transfirió | **Tomas** |
| Threshold / precision-recall / ROC / costos | **Katia** |
| Pack C, cada técnica de regularización, cómo se implementa | **Katia** |
| Tabla de sweeps (LR, arch, optimizador): valores y elección | **Nicolas** (con guion) → Tomas si profundizan |
| Análisis del dataset / por qué excluir features | **Mateo** lee el cuadrito → **Katia** rescata |
| Conclusión / próximos pasos | **Katia** |

## Resumen de balance

| Persona | # bloques | # slides | Tiempo aprox | Carga Q&A |
|---|---|---|---|---|
| Tomas    | 2 | 7 | ~7 min | Alta (la lleva) |
| Katia    | 3 | 8 | ~8 min | Alta (la lleva) |
| Nicolas  | 2 | 8 | ~8 min | Media (con guion) |
| Mateo    | 3 | 9 | ~8 min | Baja (visual / scripted, con rescate) |
