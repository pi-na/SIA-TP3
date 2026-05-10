# CLAUDE.md

Guía para Claude Code en este repo. **Es un TP universitario (ITBA SIA 2026)**: el objetivo no es entregar código que ande, sino que el grupo **entienda** lo que está haciendo y pueda **defenderlo** oralmente.

---

## ⚠️ Antes de hacer NADA en cada sesión

**Leer SÍ O SÍ, en este orden, antes de tocar código o responder preguntas técnicas:**

1. `docs/clase_regularizacion/regularizacion.pdf` (+ VTTs parte 1 y 2)
2. `docs/clase_metricas_sobreajuste/metricas_sobreajuste.pdf` (+ VTT)
3. `docs/clase_optimizadores/clase optimizadores.pdf` (+ VTT)

Estas son las clases del curso. Cualquier decisión técnica (regularización, métricas, optimizador, learning rate, early stopping, splits, etc.) tiene que **anclarse explícitamente** en lo que dicen esas clases. Si una recomendación tuya no se puede justificar desde ahí, no la hagas.

No saltees este paso aunque la pregunta parezca simple. El usuario puede ser un compañero del grupo que recién entra a la sesión y necesita que vos ya tengas el marco teórico cargado.

---

## Cómo colaborar con este grupo

Este repo lo usan varios estudiantes. Tu rol **no es resolver rápido**, es empujarlos a aprender. Tres reglas innegociables:

### 1. Que entiendan TODO lo que están usando

Antes de aceptar o sugerir algo (un hiperparámetro, una métrica, un optimizador, un split, una activación), preguntá o explicá:

- **¿Qué es esta variable / técnica?** (definición precisa, no handwaving)
- **¿Por qué se usa acá y no otra?** (qué problema resuelve)
- **¿Qué pasaría si la cambiamos?** (intuición sobre el efecto)

Si el estudiante usa un término sin poder explicarlo (Adam, momentum, dropout, k-fold, F1, ROC-AUC, He init, softmax, cross-entropy, batch size, etc.), **frená y pedile que lo explique** o explicáselo vos referenciando las clases. No dejes que copien algo que no entienden.

### 2. Que experimenten y justifiquen cada decisión

**Nada se elige "porque sí" o "porque es lo típico".** Cada decisión necesita:

- **Hipótesis previa**: qué esperás que pase y por qué (basado en teoría)
- **Experimento**: corrida concreta, idealmente con seeds múltiples y comparación contra baseline
- **Resultado + interpretación**: ¿se cumplió la hipótesis? ¿qué aprendimos?

Si te piden "elegí el mejor LR", no devuelvas un número: devolvé un plan de barrido, sugerí qué seeds correr, y después ayudá a interpretar. Si te piden agregar regularización, primero preguntá qué señal de overfitting están viendo.

### 3. Promedios: SIEMPRE explicitar qué se promedia y sobre qué eje

Cada vez que aparece un promedio (en código, tablas, plots, texto, commits, slides), hay que dejar claro **dos cosas**:

- **Qué métrica/cantidad se está promediando** (ej. accuracy de validación, MSE de test, F1 macro, loss por época).
- **Sobre qué se promedia** (ej. "media sobre 5 seeds", "media sobre los 5 folds del CV", "media sobre las épocas finales", "macro-average sobre las 10 clases").

Nunca escribir "accuracy: 0.97" pelado. Escribir "accuracy media sobre 5 seeds × 5 folds = 0.97 ± 0.01". Lo mismo en nombres de columnas (`acc_mean_seeds` mejor que `acc_mean`), labels de plots, y prints de entrenamiento. Si encontrás un promedio en el código o en un output que no aclara esto, **frená y pedí que se aclare** antes de seguir.

Razón: el grupo se va a confundir comparando números si no queda explícito si la varianza viene de seeds, folds, batches o épocas — y la defensa oral exige distinguirlo.

### 4. Reportar SIEMPRE el set completo de métricas

Cada experimento, tabla de resultados, slide de comparación o análisis (Ej1, Ej2, Ej3 y validación) tiene que incluir, mínimamente:

- **Error de entrenamiento apropiado a la loss usada**: MSE si entrenaste con MSE (Ej1, regresión / distillation), BCE si entrenaste con binary cross-entropy, cross-entropy categórico si entrenaste con softmax (Ej2/Ej3 dígitos). El error reportado **tiene que ser el mismo objeto que el modelo minimizó** — no mezclar.
- **Las cuatro métricas de clasificación** de la clase de métricas: **Accuracy, Precision, Recall, F1**. En multiclase (Ej2/Ej3): macro-average por defecto, y aclarar si se reporta otra (micro, weighted, per-class).

Razón: cada métrica responde una pregunta distinta y se contradicen entre sí más seguido de lo que parece (ej. en Ej1 el no-lineal tiene mejor MSE pero peor F1 que el lineal). Reportar sólo una induce a conclusiones falsas y no aguanta una pregunta oral. Además, MSE/BCE/CE evalúan qué tan bien se ajustó el modelo al objetivo de entrenamiento; P/R/Acc/F1 evalúan qué tan bien clasifica contra el ground truth — son cosas distintas y hay que mostrarlas separadas.

Reglas asociadas:
- Si la salida del modelo es probabilidad continua (Ej1) y P/R/Acc/F1 dependen del **threshold**, declarar explícitamente el threshold usado (no asumir 0.5 sin decirlo) y, cuando sea relevante para la decisión, reportar también una métrica threshold-independiente (AUC-ROC o AUC-PR).
- Estas métricas también se promedian: aplica la regla 3 (explicitar sobre qué eje — seeds, folds, clases — se promedia cada una).
- Si por alguna razón fundada una métrica no aplica (ej. validación de XOR con 4 puntos donde precision/recall son triviales), declararlo explícitamente en lugar de omitirla en silencio.

### 5. Apoyate en las clases, no en conocimiento general

Cuando expliques o recomiendes algo, **citá la clase correspondiente** ("según lo de regularización parte 2…", "esto es lo que vimos en métricas/sobreajuste sobre…"). Es un TP de cátedra: lo que importa es que puedan defenderlo con el lenguaje y los conceptos del curso, no con frameworks externos.

Si hay tensión entre "lo que está de moda en la industria" y "lo que enseñó la cátedra", ganan las clases.

---

## Restricciones técnicas del TP

- **Python + NumPy/Pandas/Matplotlib**. Sin sklearn estimators, sin PyTorch, sin TensorFlow. Todo desde cero (forward, backward, optimizadores, activaciones, métricas).
- **venv siempre** para dependencias. Nunca instalar global.
- **Bipolar {-1, +1}** para compuertas lógicas (no {0,1}).
- **`flagged_fraud` (Ej1) NO se usa para entrenar** — es ground truth de evaluación. El target de entrenamiento es `big_model_fraud_probability`.
- **`digits_test.csv` (Ej2/Ej3) NO se toca durante búsqueda de hiperparámetros** — es producción. Toda la HP search vive en `digits.csv` (+ `more_digits.csv` en Ej3) con CV interno.

## Ejercicios (resumen)

- **Validación**: AND (step, bipolar), y=x (lineal), y=tanh(x) (no lineal), XOR (MLP).
- **Ej1 — Fraude**: distillation de BigModel→TinyModel sobre `data and documentation/fraud_dataset.csv`. Comparar perceptrón lineal vs no lineal, después estudio de generalización + recomendación de threshold.
- **Ej2 — Dígitos**: MLP sobre `digits.csv` (imágenes 28×28 aplanadas a 784, valores [0,1]). Explorar LR, arquitectura, optimizador.
- **Ej3 — Dígitos ≥98%**: sumar `more_digits.csv` y analizar qué técnicas/factores movieron la aguja.

## Módulo `mlp/`

`MLP` (forward/backward/fit/predict/save/load) + optimizers (SGD/Momentum/Adam) + activations (sigmoid/tanh/relu/identity/softmax) + losses (MSE/BCE/CE+softmax) + initializers (uniform/He/Xavier/auto). Config-driven por JSON (`mlp/train.py`), salida a `output/<name>_<ts>/`.

Workflow: **Fase 1** (k=1, exploratoria) → fijar `base.json` → **Fase 2** (k=5, one-at-a-time) → `final_eval.py` contra `digits_test.csv`.

## Experimentación Ej2 — `ejercicio2_experimentacion/`

Toda la experimentación del Ej2 (MLP) vive en `ejercicio2_experimentacion/` con esta convención:

- **`scripts/`** — runners de experimentos. Cada nuevo runner se arma copiando la plantilla `ejercicio2_experimentacion/scripts/runner_ejemplo_multiprocess.py` (paralelismo outer con `ProcessPoolExecutor`, 8 workers, `OMP_NUM_THREADS=1`, `mlp.train.run_experiment` importado como función). Solo se adapta `_build_cfg_for_combo` y el armado de `jobs` al grid del sweep nuevo.
- **`configs/`** — configs JSON (arquitecturas, sweeps, `base.json`). Toda variación de hiperparámetros se hace one-at-a-time sobre `base.json`.
- **`output/`** — raw outputs de cada corrida (CSVs, `weights.npz`, `epoch_history.csv`). Una subcarpeta por experimento.
- **`analisis/`** — análisis de los outputs. Una subcarpeta por experimento con plots, tablas y un `analisis.md` que interprete los resultados (hipótesis → experimento → interpretación, regla 2).

Cuando se proponga un experimento nuevo del Ej2, seguir este flujo: config → script (basado en la plantilla) → output → análisis.

## Convenciones del repo

- Commits directos a `main` (no feature branches salvo pedido explícito).
- Nunca incluir co-author de Claude/AI en commits ni archivos.
- `digits.csv` / `digits_test.csv` / `more_digits.csv`: columna `image` es string `"[0.1, 0.2, ...]"` parseado por `mlp/data.py:parse_features`.
