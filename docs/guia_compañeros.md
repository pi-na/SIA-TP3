# Guía para compañeros — TP3 (Perceptrón Multicapa)

Esta guía explica cómo correr un experimento end-to-end, desde levantar el repo
hasta tener un plot listo para slides. Asume Python básico y línea de comandos.

## Setup (una sola vez)

```bash
git clone git@github.com:pi-na/SIA-TP3.git
cd SIA-TP3
# Opción A: instalar deps system-wide
pip install --user numpy pandas matplotlib pytest
# Opción B: venv
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib pytest
```

Si usás venv, prependé `.venv/bin/` a los comandos `python3` de abajo.

## Verificar que el engine funciona

```bash
python3 -m pytest mlp/tests/ ejercicio0/tests/ -v
```

Debe pasar todo verde (78+ tests).

## Anatomía de un experimento

Un experimento es: **un `config.json` → corrida → carpeta de output → plots**.

### 1. Mirar un config existente

```bash
cat ejercicio2/configs/base.json
```

Campos clave:
- `architecture.layer_sizes`: `[784, 100, 50, 10]` = 784 inputs (28×28), hidden 100, hidden 50, 10 outputs.
- `architecture.activations`: una activación por transición. La última debe ser `softmax` para clasificación multiclase.
- `architecture.initializer`: `"auto" | "uniform" | "he" | "xavier"` (`auto` elige según activación).
- `training.optimizer.name`: `"sgd" | "momentum" | "adam"`.
- `training.optimizer.lr`: tasa de aprendizaje.
- `training.batch_size`: tamaño de mini-batch.
- `split.k_folds`: 1 = train/val split simple (Fase 1 exploratoria), 5 = 5-fold CV (Fase 2).

### 2. Crear un config nuevo

Copiá uno existente y cambiá lo que querés barrer:

```bash
cp ejercicio2/configs/base.json ejercicio2/configs/sweeps_fase2/lr_005.json
# Editar lr_005.json: cambiar model_name y training.optimizer.lr
```

### 3. Correr

```bash
python3 -m mlp.train \
    --config ejercicio2/configs/sweeps_fase2/lr_005.json \
    --csv-root . \
    --output-dir ejercicio2/output \
    --workers 5
```

`--workers N` paraleliza los folds (sólo útil si k_folds > 1).

Output: carpeta `ejercicio2/output/<model_name>_<timestamp>/` con 5 CSVs + npz.

### 4. Inspeccionar resultados

```python
import pandas as pd
run_dir = "ejercicio2/output/lr_005_20260502_143022"
print(pd.read_csv(f"{run_dir}/run_summary.csv"))   # métricas por fold + mean/std
print(pd.read_csv(f"{run_dir}/epoch_history.csv").head())  # evolución por época
```

### 5. Plotear

```bash
python3 ejercicio2/analisis/plot_learning_curves.py \
    --run-dir ejercicio2/output/lr_005_<ts> \
    --out ejercicio2/presentacion/foo.png
```

Plots terminan en `ejercicio2/presentacion/` (carpeta separada para slides).

## Workflow disciplinado

**No tirar configs random.** Orden:

1. **Fase 1 — exploratoria, k_folds=1**: barrer arch → opt → LR → batch. Elegir ganador de cada uno antes del siguiente. Termina con `base.json`.
2. **Fase 2 — one-at-a-time, k_folds=5**: con `base.json` fijo, variar **un solo HP** a la vez. Da comparativas limpias.
3. **Final eval**: una vez que `base.json` está congelado, `final_eval.py` lo entrena con todo el train set y reporta sobre `digits_test.csv` UNA VEZ.

## Reglas no negociables

- No usar `digits_test.csv` durante búsqueda de HP — es "producción".
- No commitear `weights.npz` y `predictions.csv` si pesan mucho (chequear `.gitignore`).
- Sí commitear `run_summary.csv` y `epoch_history.csv` (livianos, permiten recrear plots).
- Cada experimento tiene un `model_name` distinto en el config para que la carpeta no se pise.

## Si algo se rompe

1. Correr los tests: `python3 -m pytest mlp/tests/ -v`. Si fallan, hay un bug del engine.
2. Si los tests pasan pero un experimento da resultados raros, mirar `run_summary.csv` para ver si convergió y `epoch_history.csv` para curvas.
3. Si nada de eso, prompteame con la traza de error o el CSV problemático.
