# Guía para compañeros — TP3 (Perceptrón Multicapa)

Esta guía tiene dos partes:
1. **Qué hicimos y cómo presentarlo** (el contenido del TP).
2. **Cómo correr un experimento desde cero** (la parte práctica).

---

## Parte 1 — Cómo explicar el TP

### La idea central (1 minuto)

> "Construimos un módulo `mlp/` reutilizable y desde ahí resolvemos los 3 ejercicios cambiando solo configs JSON. La filosofía: separar el **engine** (NumPy puro, sin frameworks) del **experimento** (configs + análisis + plots), para que cada experimento sea: una corrida → una carpeta con CSVs → un plot."

### Estructura del repo (30s)

```
mlp/                     ← engine: forward, backward, fit, predict, optimizers, losses
ejercicio0/              ← validación: AND (step), XOR (MLP), linear, tanh
ejercicio1/              ← fraud (Tomás)
ejercicio2/              ← dígitos: configs/, output/, analisis/, presentacion/, final_eval
ejercicio3/              ← dígitos ≥98%: igual estructura + Pack C (regularización)
docs/guia_compañeros.md  ← este archivo
```

### El workflow disciplinado de Ej2 — la parte conceptual fuerte (3 min)

> "Hicimos dos fases para no caer en sweep aleatorio:"

- **Fase 1 (k_folds=1, exploratoria):** sweeps **secuenciales** — primero arquitectura, después optimizador, después learning rate, después batch size. Cada sub-fase usa el ganador del anterior. **Esto reduce el espacio de búsqueda de O(n⁴) a O(4·n).**
- **Fase 2 (k_folds=5, validación):** **one-at-a-time** sobre `base.json`. Variar un solo HP a la vez con K-fold sirve para mostrar comparativas limpias en la presentación.
- **Final eval:** UNA SOLA VEZ contra `digits_test.csv`. No se toca antes.

### Resultado clave de Ej2 (1 min)

> "val K=5 = 96.22%, pero **test = 86.30%** — un drop de 10pp. Eso fue una sorpresa y motivó Ej3: el problema no era el modelo, era distribution shift entre `digits.csv` y `digits_test.csv`."

### Resultado clave de Ej3 (2 min)

> "Solo agregar `more_digits.csv` (sin tocar nada más) llevó test de 86.30% → 96.36%. **+10pp por más datos.** Ese es el take-away principal."
>
> "Después intentamos Pack C (L2, dropout, augmentation gaussiana, lr_schedule, wider arch) y sus combinaciones. El ganador fue **L2 + augmentation gaussiana σ=0.05 → 96.88%**. Aug y L2 sumaron, dropout no transfirió a test, wider sobreajustó."
>
> "**No llegamos al target del 98%.** Hipótesis: el shift residual no se captura con noise isotrópico — habría que augmentar con rotaciones/desplazamientos, no random noise."

### Tabla resumen para la slide

| Config | val K=5 | test (digits_test.csv) |
|---|---:|---:|
| Ej2 base | 96.22% | 86.30% |
| Ej3 +more_digits.csv | 97.06% | 96.36% (+10pp) |
| Ej3 +L2 | 97.58% | 96.56% |
| **Ej3 +L2+aug (ganador)** | 97.60% | **96.88%** |
| Ej3 +L2+dropout | 97.72% | 96.04% |
| Ej3 +wider+L2+aug | 97.77% | 96.40% |

### Qué decir si te preguntan "¿por qué no llegaron al 98%?"

> "Probamos las 4 técnicas Pack C del enunciado y combinaciones. Lo que no probamos fue augmentation por **transformaciones geométricas** (rotación, traslación, elastic). El plan era gaussian noise como augmentation por simpleza, pero ese tipo de noise no captura el shift de estilos de escritura entre train y test. Si tuviéramos otra iteración, eso es lo que agregaríamos."

### Demo en vivo (si querés mostrar el flujo)

```bash
# 1. Mostrar tests verde
python3 -m pytest mlp/tests/ ejercicio0/tests/ -v

# 2. Mostrar un config base
cat ejercicio2/configs/base.json

# 3. Correr un experimento chico (XOR, ~5 segundos)
python3 -m mlp.train --config ejercicio0/configs/xor_2_2_1.json \
    --csv-root . --output-dir /tmp/xor_demo --workers 1

# 4. Mostrar los plots ya generados
ls ejercicio2/presentacion/
ls ejercicio3/presentacion/
```

### Plots más importantes para la presentación

**Ej2** (`ejercicio2/presentacion/`):
- `01_curvas_aprendizaje_base.png` — convergencia del modelo base, K-fold=5
- `02_sweep_lr.png` — barras de val_acc por LR (incluye lr=0.1 y lr=10 que divergen)
- `03_sweep_arch.png` — comparativa de 4 arquitecturas
- `04_confusion_matrix_base.png` — matriz 10×10 del val set
- `05_per_class_metrics_base.png` — precision/recall/f1 por dígito
- `06_sweep_opt.png` — SGD vs Momentum vs Adam
- `07_sweep_init.png` — He vs Xavier vs Uniform
- `08_confusion_matrix_final.png` — matriz sobre digits_test.csv (post final_eval)
- `09_per_class_metrics_final.png` — métricas por clase en test

**Ej3** (`ejercicio3/presentacion/`):
- `01_comparacion_ej2_vs_ej3.png` — bars side-by-side con línea roja en 98%
- `02_confusion_matrix_ej3.png` — matriz del modelo ganador en test
- `03_per_class_metrics_ej3.png` — métricas por clase
- `04_curvas_aprendizaje_ej3.png` — curvas K=5 del l2_aug

### Qué cumplió y qué no la consigna

| Item | Status |
|---|---|
| Validación step perceptron AND (bipolar) | ✅ |
| Validación lineal y no-lineal 1D | ✅ |
| Validación MLP XOR [2,2,1] y [2,3,2,1] | ✅ |
| Ej1 fraud detection | ✅ (Tomás) |
| Ej2 dígitos con MLP from-scratch (NumPy) | ✅ |
| Ej2 explorar LR variants | ✅ |
| Ej2 explorar arquitecturas | ✅ |
| Ej2 explorar optimizadores | ✅ |
| Ej2 final_eval vs digits_test.csv (one-shot) | ✅ |
| Ej3 usar more_digits.csv | ✅ |
| Ej3 target ≥98% | ❌ (mejor 96.88%) |
| Ej3 análisis de qué ayudó / qué no | ✅ |

Único faltante: el 98%. Está documentado honestamente con hipótesis en `ejercicio3/README.md`.

---

## Parte 2 — Cómo correr un experimento

### Setup (una sola vez)

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

### Verificar que el engine funciona

```bash
python3 -m pytest mlp/tests/ ejercicio0/tests/ -v
```

Debe pasar todo verde (78 tests).

### Anatomía de un experimento

Un experimento es: **un `config.json` → corrida → carpeta de output → plots**.

#### 1. Mirar un config existente

```bash
cat ejercicio2/configs/base.json
```

Campos clave:
- `architecture.layer_sizes`: `[784, 100, 50, 10]` = 784 inputs (28×28), hidden 100, hidden 50, 10 outputs.
- `architecture.activations`: una activación por transición. La última debe ser `softmax` para multiclass.
- `architecture.initializer`: `"auto" | "uniform" | "he" | "xavier"` (`auto` elige según activación).
- `training.optimizer.name`: `"sgd" | "momentum" | "adam"`.
- `training.optimizer.lr`: tasa de aprendizaje.
- `training.batch_size`: tamaño de mini-batch.
- `split.k_folds`: 1 = train/val split simple (Fase 1 exploratoria), 5 = 5-fold CV (Fase 2).
- `regularization`: L2, dropout, augmentation, lr_schedule (Pack C de Ej3).

#### 2. Crear un config nuevo

Copiá uno existente y cambiá lo que querés barrer:

```bash
cp ejercicio2/configs/base.json ejercicio2/configs/sweeps_fase2/lr_005.json
# Editar lr_005.json: cambiar model_name y training.optimizer.lr
```

#### 3. Correr

```bash
python3 -m mlp.train \
    --config ejercicio2/configs/sweeps_fase2/lr_005.json \
    --csv-root . \
    --output-dir ejercicio2/output \
    --workers 5
```

`--workers N` paraleliza los folds (sólo útil si `k_folds > 1`). **Si en una corrida larga el job termina silenciosamente con `exit 0` pero no produce CSVs, bajá a `--workers 1`** — es un issue de multiprocessing bajo carga, no del modelo.

Output: carpeta `ejercicio2/output/<model_name>_<timestamp>/` con 5 CSVs + npz.

#### 4. Inspeccionar resultados

```python
import pandas as pd
run_dir = "ejercicio2/output/lr_005_20260502_143022"
print(pd.read_csv(f"{run_dir}/run_summary.csv"))   # métricas por fold + mean/std
print(pd.read_csv(f"{run_dir}/epoch_history.csv").head())  # evolución por época
```

#### 5. Plotear

```bash
python3 ejercicio2/analisis/plot_learning_curves.py \
    --run-dir ejercicio2/output/lr_005_<ts> \
    --out ejercicio2/presentacion/foo.png

python3 ejercicio2/analisis/plot_sweep.py \
    --run-dirs $(ls -d ejercicio2/output/fase2_lr_*) \
    --metric val_acc_final --label-by lr \
    --out ejercicio2/presentacion/bar.png
```

Plots terminan en `ejercicio2/presentacion/` (carpeta separada para slides).

### Workflow disciplinado (Ej2)

**No tirar configs random.** Orden:

1. **Fase 1 — exploratoria, k_folds=1**: barrer arch → opt → LR → batch. Elegir ganador de cada uno antes del siguiente. Termina con `base.json`.
2. **Fase 2 — one-at-a-time, k_folds=5**: con `base.json` fijo, variar **un solo HP** a la vez. Da comparativas limpias.
3. **Final eval**: una vez que `base.json` está congelado, `final_eval.py` lo entrena con todo el train set y reporta sobre `digits_test.csv` UNA VEZ.

```bash
# Final eval Ej2
python3 ejercicio2/final_eval.py \
    --config ejercicio2/configs/base.json \
    --csv-root . \
    --output-dir ejercicio2/output_final
```

### Workflow Ej3 (Pack C)

```bash
# 1. Base con more_digits.csv (concatena con digits.csv)
python3 -m mlp.train --config ejercicio3/configs/base_extra_data.json \
    --csv-root . --output-dir ejercicio3/output --workers 5

# 2. Si val_acc < 0.98, activar Pack C de a uno o combinados
python3 -m mlp.train --config ejercicio3/configs/pack_c/l2_aug.json \
    --csv-root . --output-dir ejercicio3/output --workers 1

# 3. Final eval con el ganador
python3 ejercicio3/final_eval.py --config ejercicio3/configs/pack_c/l2_aug.json \
    --csv-root . --output-dir ejercicio3/output_final
```

Pack C disponible en el engine vía `regularization` del config:
- `"l2": 1e-4` — weight decay (no penaliza bias)
- `"dropout": 0.2` — inverted dropout en hidden layers (training-only)
- `"augmentation": {"type": "gaussian_noise", "sigma": 0.05}` — noise pre-batch
- `"lr_schedule": {"type": "step", "decay": 0.5, "every": 10}` — step decay del LR

### Reglas no negociables

- No usar `digits_test.csv` durante búsqueda de HP — es "producción".
- No commitear `weights.npz` y `predictions.csv` si pesan mucho (chequear `.gitignore`).
- Sí commitear `run_summary.csv` y `epoch_history.csv` (livianos, permiten recrear plots).
- Cada experimento tiene un `model_name` distinto en el config para que la carpeta no se pise.

### Si algo se rompe

1. Correr los tests: `python3 -m pytest mlp/tests/ -v`. Si fallan, hay un bug del engine.
2. Si los tests pasan pero un experimento da resultados raros, mirar `run_summary.csv` para ver si convergió y `epoch_history.csv` para curvas.
3. Si nada de eso, prompteame con la traza de error o el CSV problemático.

### Si `final_eval.py` da `ModuleNotFoundError: No module named 'mlp'`

El script ya tiene `sys.path.insert(0, repo_root)` así que debería funcionar desde cualquier directorio. Si no, correr con `PYTHONPATH=. python3 ...`.
