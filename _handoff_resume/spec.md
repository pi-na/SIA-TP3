# Diseño — Completar TP3 (Perceptrón Simple y Multicapa)

| | |
|---|---|
| **Fecha** | 2026-05-01 |
| **Autor** | Nicolás (con asistencia de Claude) |
| **Estado** | Aprobado, pendiente de plan de implementación |
| **Materia** | Sistemas de Inteligencia Artificial — ITBA, 1Q 2026 |
| **Repo** | `git@github.com:pi-na/SIA-TP3.git` |
| **Plan referenciado** | (a generar con `writing-plans` después de aprobado este spec) |

---

## 1. Contexto

El TP3 ya tiene resuelto el **Ejercicio 1** (knowledge distillation con perceptrón simple lineal y no lineal sobre `fraud_dataset.csv`) en estado de presentación. Resta:

1. **Validación faltante** — perceptrón escalón (AND) y MLP (XOR con arquitecturas `[2,2,1]` y `[2,3,2,1]`).
2. **Ejercicio 2** — clasificación de dígitos escritos a mano (28×28) con MLP, explorando variantes de tasa de aprendizaje, arquitectura y mecanismos de optimización.
3. **Ejercicio 3** — alcanzar `accuracy ≥ 98%` sobre `digits_test.csv` aprovechando `more_digits.csv` como datos adicionales.

Los tres bloques comparten infraestructura: el MLP genérico, el pipeline de entrenamiento config-driven, el logging en CSV, y los scripts de plotting post-hoc.

### 1.1 Restricciones del enunciado

- Implementación **desde cero con NumPy** (sin frameworks de alto nivel: nada de PyTorch, TensorFlow, Keras, ni estimadores de scikit-learn).
- Recomendaciones del enunciado que adoptamos como requisitos:
  - Operaciones matriciales (no loops sobre neuronas individuales).
  - Reportar progreso durante el entrenamiento.
  - Configuración extensible y persistible.
  - Save/load de modelos para continuar entrenando.
  - **División de responsabilidades**: el programa de entrenamiento almacena información de cada experimento; el análisis (plots, tablas) se hace por separado.

### 1.2 Restricciones del usuario y del compañero

Aporte explícito del compañero del usuario, internalizado como restricciones de diseño:

- **Engine puro**: el programa toma `config.json` y produce CSVs ricos + carpeta por corrida. Cero plots adentro del programa.
- **Plotting post-hoc**: scripts separados que toman CSVs y producen PNGs. Los gráficos se generan por separado prompteando a Claude desde los CSVs.
- **Plots en carpeta separada y bien etiquetados** para que vayan directo a la presentación.
- **Workflow disciplinado**:
  1. Hacer que el programa funcione antes de cualquier sweep.
  2. **Fase 1**: búsqueda exploratoria de configuración base.
  3. **Fase 2**: variar **un solo hiperparámetro** a la vez con el resto fijo.

### 1.3 Filosofía de calidad

El usuario eligió la opción **"híbrido pragmático"**: cumplir el "como mínimo" del enunciado (LR, arquitectura, optimización) con set pre-elegido de variantes, y un análisis profundo del modelo ganador. La vara es presentable a un evaluador exigente.

---

## 2. Alcance

### 2.1 In scope

**Capacidades técnicas del MLP** (Pack B "estándar académico"):
- Activaciones: sigmoide, tanh, ReLU, identidad, softmax (output multiclase).
- Optimizadores: SGD vanilla, SGD con Momentum (β=0.9), Adam (β₁=0.9, β₂=0.999, ε=1e-8).
- Loss: MSE (regresión), BCE (clasificación binaria), cross-entropy (multiclase, fusionado con softmax para estabilidad numérica).
- Inicialización: uniform (default fallback), He (auto para ReLU), Xavier (auto para tanh/sigmoide).
- Mini-batch SGD configurable.
- Early stopping por mejor `val_acc`.
- Save/load de modelo (pesos + arquitectura).
- K-fold estratificado (consistente con Ej1) **o** train/val split simple, configurable.

**Pack C — módulos enchufables opcionales** (sólo se activan si Fase 2 no alcanza ≥98% en Ej3):
- Regularización L2 (weight decay).
- Dropout en capas ocultas.
- Learning rate scheduling (step decay).
- Data augmentation: ruido gaussiano sobre el vector de pixels.

**Implementaciones específicas**:
- Perceptrón escalón (signo) para validar AND, en `ejercicio0/step_perceptron.py`.
- Validación XOR del MLP genérico vía configs en `ejercicio0/configs/`.
- Pipeline completo de Ej2: configs, sweeps, final eval.
- Pipeline completo de Ej3: configs adicionales con `more_digits.csv`, opcional Pack C.
- Scripts de análisis y plots por ejercicio.

**Documentación**:
- READMEs por ejercicio y por módulo.
- `docs/guia_compañeros.md` con walkthrough end-to-end.
- Actualización de `CLAUDE.md` y `README.md` raíz.

### 2.2 Out of scope

- **No tocamos** `ejercicio0/linear_perceptron.py` ni `ejercicio0/nonlinear_perceptron.py` (validación 1D existente y testeada).
- **No tocamos nada** dentro de `ejercicio1/` (presentación-ready, mate ya commiteó su contribución, no merece riesgo de regresión).
- **No refactorizamos** los perceptrones del Ej1 hacia el MLP genérico, aunque conceptualmente sean un caso particular (MLP con 0 capas ocultas). Razones:
  - Ej1 tiene su propio storytelling (knowledge distillation) y material de presentación.
  - El refactor agregaría riesgo sin beneficio académico claro.
  - Ej1 quedaría acoplado a `mlp/`, dificultando explicarlo aisladamente.
- **No implementamos** otros optimizadores (RMSProp, AdaGrad), otras activaciones (LeakyReLU, GELU), ni regularización avanzada (Batch Normalization, Layer Normalization). Pack B se considera suficiente para el scope académico.
- **No implementamos** logger estructurado, monitoring real-time, ni dashboards. El reporte de progreso es `print` durante entrenamiento + CSVs al final.

---

## 3. Arquitectura

### 3.1 Layout del repositorio

```
SIA-TP3/
├── mlp/                              # NUEVO — módulo genérico del MLP
│   ├── __init__.py
│   ├── network.py                    # clase MLP (forward, backward, fit, predict, save, load)
│   ├── activations.py                # funciones puras + ACTIVATIONS dict
│   ├── losses.py                     # funciones de loss + gradientes
│   ├── optimizers.py                 # SGD, Momentum, Adam (clases con .step())
│   ├── initializers.py               # uniform, he, xavier (auto-pick por activación)
│   ├── data.py                       # K-fold estratificado, train/val split, mini-batch iter
│   ├── metrics.py                    # accuracy, precision/recall/f1 multi-class, conf matrix
│   ├── train.py                      # CLI entry-point: config.json → output dir
│   ├── README.md                     # API + cómo agregar optimizador/activación
│   └── tests/
│       ├── conftest.py
│       ├── test_xor.py               # regression test: [2,2,1] y [2,3,2,1] convergen
│       ├── test_optimizers.py        # SGD/Momentum/Adam updates correctos
│       ├── test_activations.py       # gradientes vs diferenciación numérica
│       ├── test_losses.py            # gradientes vs diferenciación numérica
│       └── test_save_load.py         # round-trip preserva pesos y config
│
├── ejercicio0/                       # validación
│   ├── linear_perceptron.py          # EXISTENTE — no se toca
│   ├── nonlinear_perceptron.py       # EXISTENTE — no se toca
│   ├── generate_linear_dataset.py    # EXISTENTE
│   ├── generate_tanh_dataset.py      # EXISTENTE
│   ├── tests/                        # EXISTENTE + nuevo test_step_perceptron.py
│   ├── step_perceptron.py            # NUEVO — AND gate, ~50 LOC
│   ├── step_dataset.csv              # NUEVO — AND con bipolar {-1, 1}
│   ├── configs/                      # NUEVO — XOR para validar el MLP
│   │   ├── xor_2_2_1.json
│   │   └── xor_2_3_2_1.json
│   └── README.md                     # actualizado con sección de step + XOR
│
├── ejercicio1/                       # INTACTO
│
├── ejercicio2/                       # NUEVO — clasificación de dígitos
│   ├── configs/
│   │   ├── base.json                 # output de Fase 1 (config base elegido)
│   │   ├── sweeps_fase1/             # exploración coarse-to-fine
│   │   │   ├── arch_*.json
│   │   │   ├── opt_*.json
│   │   │   ├── lr_*.json
│   │   │   └── batch_*.json
│   │   └── sweeps_fase2/             # one-at-a-time desde base.json
│   │       ├── lr_*.json
│   │       ├── arch_*.json
│   │       ├── opt_*.json
│   │       └── init_*.json
│   ├── output/                       # poblada por mlp/train.py — una carpeta por corrida
│   ├── analisis/                     # scripts de plotting (post-hoc)
│   │   ├── plot_learning_curves.py
│   │   ├── plot_sweep.py
│   │   ├── plot_confusion_matrix.py
│   │   └── plot_per_class_metrics.py
│   ├── presentacion/                 # plots etiquetados, listos para slides
│   ├── final_eval.py                 # entrena con todo digits.csv → eval digits_test.csv
│   └── README.md                     # workflow Fase 1 → Fase 2 → final_eval
│
├── ejercicio3/                       # NUEVO — target ≥98%
│   ├── configs/
│   │   ├── base_extra_data.json      # base.json del Ej2 + more_digits.csv
│   │   └── pack_c/                   # opcionales si no llega
│   │       ├── dropout.json
│   │       ├── l2.json
│   │       ├── lr_schedule.json
│   │       └── augmentation.json
│   ├── output/
│   ├── analisis/                     # scripts adicionales: comparación Ej2 vs Ej3
│   ├── presentacion/
│   ├── final_eval.py                 # final eval para el modelo del Ej3
│   └── README.md
│
├── docs/
│   ├── superpowers/specs/
│   │   └── 2026-05-01-tp3-completion-design.md   # ESTE ARCHIVO
│   └── guia_compañeros.md            # walkthrough end-to-end del workflow
│
├── data and documentation/           # EXISTENTE — datasets
├── README.md                         # actualizado con índice global del repo
└── CLAUDE.md                         # actualizado con info del módulo mlp/
```

**Criterio de organización**: el MLP es una librería usada por los ejercicios. Cada ejercicio es **autocontenido** (configs propios, output propio, análisis propio, presentación propia) y se puede explicar aisladamente. La única dependencia externa de cada ejercicio es `mlp/`.

### 3.2 API del MLP (`mlp/network.py`)

```python
class MLP:
    def __init__(
        self,
        layer_sizes: list[int],            # ej. [784, 100, 10]
        activations: list[str],            # len = len(layer_sizes) - 1
        loss: str,                         # "mse" | "bce" | "cross_entropy"
        optimizer: Optimizer,              # instancia de SGD/Momentum/Adam
        initializer: str = "auto",         # "auto" | "uniform" | "he" | "xavier"
        seed: int | None = None,
        regularization: dict | None = None,
    ): ...

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, list]:
        """Forward pass. Devuelve (output, cache) para backward.

        cache: lista de tuplas (z_l, a_l) por capa, evita recomputar pre-activaciones.
        """

    def backward(self, X, y_true, cache) -> list[np.ndarray]:
        """Backprop. Devuelve gradientes por capa (mismo shape que weights)."""

    def fit(
        self,
        X_train, y_train,
        X_val, y_val,
        epochs: int,
        batch_size: int,
        early_stopping_patience: int | None = None,
        callback=None,
    ) -> dict:
        """Loop principal. Llama callback(epoch, metrics_dict) cada época
        si callback no es None. Devuelve history (lista de dicts por época)."""

    def predict(self, X) -> np.ndarray: ...
    def predict_proba(self, X) -> np.ndarray: ...
    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> "MLP": ...
```

**Convenciones internas**:
- `weights[l]` tiene shape `(n_l, n_{l-1} + 1)` — la primera columna es el bias (bias trick).
- Las entradas a `forward` se prependean con una columna de unos automáticamente.
- `cache` se reusa entre forward y backward para no recomputar.
- `fit` usa `BatchIterator` de `mlp/data.py` para mini-batch shuffleado por época.

### 3.3 Optimizadores (`mlp/optimizers.py`)

```python
class Optimizer:
    """Interface base. Cada subclase mantiene su state interno."""
    def step(self, weights: list[np.ndarray], grads: list[np.ndarray]) -> None: ...

class SGD(Optimizer):
    def __init__(self, lr: float): ...

class Momentum(Optimizer):
    def __init__(self, lr: float, beta: float = 0.9):
        self.velocity: list[np.ndarray] | None = None  # lazy init en primer step()

class Adam(Optimizer):
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m: list | None = None  # primer momento
        self.v: list | None = None  # segundo momento
        self.t: int = 0             # timestep para bias correction
```

### 3.4 Activaciones y losses (`mlp/activations.py`, `mlp/losses.py`)

Activaciones como funciones puras (no clases):

```python
def sigmoid(z): ...
def sigmoid_grad(z, a):  # a = sigmoid(z) si ya está computado
    return a * (1 - a)

def tanh(z): ...
def tanh_grad(z, a):
    return 1 - a ** 2

def relu(z): ...
def relu_grad(z, a):
    return (z > 0).astype(z.dtype)

def identity(z): return z
def identity_grad(z, a): return np.ones_like(z)

def softmax(z):
    """Estable: resta el max por fila antes de exp."""
    z_shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh": (tanh, tanh_grad),
    "relu": (relu, relu_grad),
    "identity": (identity, identity_grad),
    "softmax": (softmax, None),  # grad va junto con cross_entropy
}
```

Losses (`mlp/losses.py`):

```python
def mse(y_true, y_pred) -> float: ...
def mse_grad(y_true, y_pred) -> np.ndarray: ...

def bce(y_true, y_pred) -> float:
    """Clip para evitar log(0)."""

def bce_grad(y_true, y_pred): ...

def cross_entropy(y_true_onehot, y_pred_softmax) -> float:
    """Clip de y_pred para evitar log(0)."""

def cross_entropy_grad_with_softmax(y_true_onehot, y_pred_softmax):
    """Truco numérico: ∂CE/∂z = y_pred - y_true cuando y_pred = softmax(z).
    Esto evita inestabilidad de calcular softmax_grad y CE_grad por separado."""
    return y_pred_softmax - y_true_onehot
```

### 3.5 Inicializadores (`mlp/initializers.py`)

```python
def init_uniform(shape, rng, scale=0.1):
    return rng.uniform(-scale, scale, size=shape)

def init_he(shape, rng):
    """He: std = sqrt(2/fan_in). Apropiado para ReLU."""
    fan_in = shape[1] - 1  # -1 por la columna de bias
    return rng.normal(0, np.sqrt(2.0 / fan_in), size=shape)

def init_xavier(shape, rng):
    """Xavier: std = sqrt(1/fan_in). Apropiado para tanh/sigmoide."""
    fan_in = shape[1] - 1
    return rng.normal(0, np.sqrt(1.0 / fan_in), size=shape)

def auto_pick(activation_name: str) -> str:
    return {"relu": "he", "tanh": "xavier", "sigmoid": "xavier",
            "identity": "uniform", "softmax": "xavier"}[activation_name]
```

### 3.6 Schema del `config.json`

```json
{
  "model_name": "base_ej2",
  "dataset": {
    "csv_path": "data and documentation/digits.csv",
    "feature_col": "image",
    "target_col": "label",
    "num_classes": 10,
    "extra_csv_paths": []
  },
  "split": {
    "k_folds": 5,
    "stratify": true,
    "val_fraction_if_k1": 0.2,
    "random_seed": 42
  },
  "preprocessing": {
    "normalization": "none",
    "one_hot_targets": true
  },
  "architecture": {
    "layer_sizes": [784, 100, 10],
    "activations": ["relu", "softmax"],
    "initializer": "auto"
  },
  "training": {
    "loss": "cross_entropy",
    "optimizer": {"name": "adam", "lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
    "epochs": 50,
    "batch_size": 64,
    "early_stopping_patience": 10
  },
  "regularization": {
    "l2": 0.0,
    "dropout": 0.0,
    "lr_schedule": null,
    "augmentation": null
  }
}
```

**Notas del schema**:
- `extra_csv_paths`: para Ej3, lista de CSVs adicionales que se concatenan a `csv_path` para entrenar (`more_digits.csv`).
- `k_folds=1` ⇒ split simple usando `val_fraction_if_k1`. Mismo pipeline.
- `normalization: "none"` para dígitos (pixels ya en [0,1]). `"zscore"` y `"minmax"` disponibles para datasets futuros.
- `initializer: "auto"` ⇒ `auto_pick` decide por activación de cada capa.
- `regularization.dropout: 0.0` ⇒ no se aplica. `> 0` activa dropout en hidden layers.
- `regularization.lr_schedule: null` ⇒ LR fija. `{"type": "step", "decay": 0.5, "every": 20}` activa step decay.
- `regularization.augmentation: null` ⇒ sin augmentación. `{"type": "gaussian_noise", "sigma": 0.05}` agrega ruido a inputs durante entrenamiento.
- **Parsing de features**: si `feature_col` contiene strings con arrays (formato del `digits.csv`: `"[0.0, 0.1, ..., 0.9]"`), `mlp/data.py` los parsea automáticamente a `np.ndarray` de shape `(P, n_features)`. Para columnas ya numéricas (como `fraud_dataset.csv`), se usan directamente.
- **Validaciones en `load_config`**:
  - `len(activations) == len(layer_sizes) - 1` (una activación por transición).
  - `loss == "cross_entropy"` ⇒ última activación debe ser `"softmax"` y `one_hot_targets: true`.
  - `loss == "bce"` ⇒ última activación debe ser `"sigmoid"` y `one_hot_targets: false`.
  - `loss == "mse"` ⇒ activación libre, `one_hot_targets: false`.
  - `optimizer.name in {"sgd", "momentum", "adam"}`.
  - `k_folds >= 1`.

### 3.6.1 Criterio de selección del ganador en cada sweep

Para todos los sweeps de Fases 1 y 2:
- **Criterio primario**: mejor `val_acc` (mean across folds si K>1).
- **Desempate**: mejor `val_loss` (mean across folds).
- **Segundo desempate**: menor `time_seconds` (preferir el más rápido si todo lo demás empata).

Para Ej3 el criterio cambia a: **`val_acc` (mean across folds) ≥ 0.98** como gate, no como ranking. El primer config que cruza ese umbral es el ganador (no se sigue optimizando para no over-fit el config a un val set).

### 3.7 Estructura de salida (`output/<model_name>_<timestamp>/`)

| Archivo | Una fila por... | Columnas clave |
|---|---|---|
| `config.json` | (copia íntegra del input) | — |
| `run_summary.csv` | fold + filas `mean`/`std` | `fold, n_train, n_val, total_epochs, best_epoch, train_loss_final, val_loss_final, train_acc_final, val_acc_final, macro_precision, macro_recall, macro_f1, weighted_f1, precision_0..9, recall_0..9, f1_0..9, time_seconds` |
| `epoch_history.csv` | (fold, epoch) | `fold, epoch, time_elapsed_s, train_loss, train_acc, val_loss, val_acc, lr_actual` |
| `confusion_matrix.csv` | (fold, true_label, pred_label) | `fold, true_label, pred_label, count` |
| `predictions.csv` | (fold, row_id out-of-fold) | `fold, row_id, true_label, pred_label, score_0..9` |
| `weights.npz` | — | `npz` con arrays `fold0_W0, fold0_W1, …, fold4_W2` y `meta` con arquitectura |

**Decisiones**:
- Métricas por época en `epoch_history.csv` para diagnosticar over/underfitting visualmente (item del apunte de métricas).
- `predictions.csv` con scores de softmax (no sólo predicción) para análisis de errores (cuándo el modelo está "casi-correcto").
- `confusion_matrix.csv` en formato stacked (no NxN) — más fácil de leer en pandas con `pd.pivot_table`.
- `run_summary.csv` con métricas por clase (`precision_0..9`, etc.) → la "MONTON de columnas" del compañero del usuario.

---

## 4. Workflow de experimentación

### 4.1 Fase 0 — Implementación (gate técnica)

1. Implementar `mlp/` completo con tests pytest.
2. Implementar `step_perceptron.py` en Ej0 con tests.
3. Crear configs XOR en Ej0 (`[2,2,1]` y `[2,3,2,1]`).
4. **Gate**: `pytest mlp/tests/ -v` y `pytest ejercicio0/tests/ -v` verdes; XOR converge en `<500` épocas; AND clasifica con error 0.

**Sin esta gate verde no se arrancan sweeps.**

### 4.2 Fase 1 — Búsqueda de `base.json` (Ej2, exploratoria, `k_folds=1`)

Búsqueda **coarse-to-fine secuencial** (no grid completo). Cada sweep usa el ganador del anterior.

| Sweep | Variantes | Otros HP fijos | Output |
|---|---|---|---|
| **1.1 Arquitectura** | `[784,50,10]`, `[784,100,10]`, `[784,128,64,10]`, `[784,100,50,10]` | adam, lr=0.001, batch=64, epochs=50 | mejor arch |
| **1.2 Optimizador** | SGD(lr=0.01), Momentum(lr=0.01), Adam(lr=0.001) | arch ganadora | mejor opt |
| **1.3 Learning rate** | {0.0001, 0.0005, 0.001, 0.005, 0.01} (escalas adaptadas al opt ganador) | opt ganador | mejor LR |
| **1.4 Batch size** | {16, 32, 64, 128} | LR ganadora | mejor batch |

**Total: ~16 corridas**. Cada una usa `k_folds=1` (split simple 80/20) para iterar rápido.

**Output**: `ejercicio2/configs/base.json` con la combinación ganadora, justificado por tabla comparativa generada de los CSVs.

### 4.3 Fase 2 — One-at-a-time (Ej2, comparativas limpias, `k_folds=5`)

Con `base.json` fijo, **un experimento por hiperparámetro variando sólo ese HP**:

| Sweep | Variantes | Output |
|---|---|---|
| **2.1 LR** | 5 valores incluyendo extremos para mostrar over/under-shooting | curva val_acc vs LR + curvas de aprendizaje |
| **2.2 Arquitectura** | 4 variantes (superficial → profunda) | tabla comparativa |
| **2.3 Optimizador** | SGD, Momentum, Adam | curvas de convergencia comparadas |
| **2.4 Inicializador** | uniform, He, Xavier | curvas de pérdida iniciales (primeras épocas) |

**Total: ~15 corridas con `k_folds=5`** → mean ± std en las tablas de la presentación.

### 4.4 Fase 3 — Ej3 (target ≥98%)

1. Tomar `base.json` ganador del Ej2.
2. Crear `ejercicio3/configs/base_extra_data.json` agregando `more_digits.csv` a `extra_csv_paths`. Correr.
3. Si `val_acc ≥ 98%` → ir a `final_eval.py`.
4. Si no, **activar Pack C secuencialmente** (un componente por vez, midiendo el delta):
   - `regularization.l2 = 1e-4`
   - `regularization.dropout = 0.2` (en hidden layers)
   - `regularization.lr_schedule = {"type": "step", "decay": 0.5, "every": 20}`
   - `regularization.augmentation = {"type": "gaussian_noise", "sigma": 0.05}`
5. **Final eval**: `final_eval.py` entrena un modelo con todo `digits.csv ∪ more_digits.csv` (sin K-fold) usando el config ganador, evalúa una sola vez sobre `digits_test.csv`, reporta:
   - Accuracy global
   - Confusion matrix 10×10
   - Métricas por clase (precision, recall, f1)
   - Comparación contra el final eval del Ej2 (cuánto mejoramos)

**Importante**: `digits_test.csv` se evalúa **una única vez** al final, después de que el config ganador está congelado. No se usa durante búsqueda de HP. Esto preserva el rol de "datos de producción" que pide el enunciado.

### 4.5 Fase 4 — Presentación

- Scripts en `ejercicioN/analisis/` que toman uno o varios CSVs de `output/` y escriben PNGs en `ejercicioN/presentacion/`.
- Naming convention: `NN_descripcion_corta.png` (`01_curvas_aprendizaje_base.png`, `02_sweep_lr_val_accuracy.png`, …).
- El usuario prompta a Claude desde los CSVs para generar los plots — los scripts son guías, no contratos rígidos. Cada gráfico se puede personalizar prompteando.

---

## 5. Tests y validación

### 5.1 Tests automatizados (`mlp/tests/` con pytest)

| Test | Qué valida | Tolerancia |
|---|---|---|
| `test_xor.py` | XOR converge a `train_loss < 0.1` en ≤500 epochs con `[2,2,1]` y `[2,3,2,1]` | acepta si al menos 2 de 5 seeds convergen (`[2,2,1]` es notoriamente sensible a init) |
| `test_optimizers.py` | SGD/Momentum/Adam reducen loss en problema cuadrático toy en 50 steps | loss final < loss inicial × 0.1 |
| `test_activations.py` | Cada `act_grad(z, a)` matchea diferenciación numérica `(act(z+ε) - act(z-ε)) / (2ε)` | `‖analítico - numérico‖ < 1e-5` |
| `test_losses.py` | Igual que activaciones para `mse_grad`, `bce_grad`, `cross_entropy_grad_with_softmax` | idem |
| `test_save_load.py` | `MLP.save` + `MLP.load` preserva pesos y arquitectura; predicciones idénticas | `np.allclose` |

**Test del step perceptron en Ej0** (`ejercicio0/tests/test_step_perceptron.py`):
- AND con bipolar `{-1, 1}` converge en ≤20 epochs con error 0.

### 5.2 Validación manual (no automatizada)

- Smoke test sobre `digits.csv`: 1 corrida con `k_folds=1`, 5 epochs, `val_acc > 0.5`. (Si esto falla, hay un bug grueso; no es un test pero es gate de Fase 1.)
- Inspección visual de curvas de aprendizaje del primer sweep para confirmar que el rango de epochs y LR son razonables.

### 5.3 Gate de Fase 0

```bash
pytest mlp/tests/ -v && pytest ejercicio0/tests/ -v
```

Sin verde, no se arranca Fase 1.

---

## 6. Documentación

### 6.1 Audiencia

- **Compañeros del equipo del usuario** (estudiantes de SIA): necesitan correr experimentos sin entender los internals del MLP.
- **Evaluador** (docente / corrector): lee READMEs para entender qué hizo cada quién.
- **LLMs futuros** (continuidad): `CLAUDE.md` actualizado con el nuevo módulo.

### 6.2 Archivos de docs

| Archivo | Audiencia | Contenido |
|---|---|---|
| `mlp/README.md` | dev del módulo | API del MLP, schema del config, cómo agregar optimizador/activación/inicializador, cómo correr los tests |
| `ejercicio0/README.md` (update) | compañeros + evaluador | qué validamos (AND, lineal 1D, no-lineal 1D, XOR), cómo correr cada cosa |
| `ejercicio2/README.md` | compañeros + evaluador | descripción del problema, workflow Fase 1 → Fase 2 → final_eval, cómo interpretar cada CSV, lista de configs y qué resultado dieron |
| `ejercicio3/README.md` | compañeros + evaluador | qué cambia respecto a Ej2, cómo activar Pack C, resultado final |
| `docs/guia_compañeros.md` | nuevo integrante | end-to-end: setup, correr una corrida básica, agregar un experimento nuevo, generar un plot, preparar la presentación |
| `CLAUDE.md` (update) | LLMs | actualización del estado del proyecto + módulo `mlp/` + dataset format de digits |
| `README.md` raíz (update) | landing | índice global del repo con links a cada ejercicio |

### 6.3 Convenciones para docs

- READMEs en español (consistente con el repo actual y con la materia).
- Cada README arranca con "Para qué sirve esto" (1 párrafo, 2-3 oraciones).
- Snippets de código copiables directamente, con paths reales (no `<PLACEHOLDER>`).
- Tabla de hiperparámetros: `nombre, valor default, rango razonable, qué efecto tiene`.
- Capturas o referencias a CSVs reales para ilustrar outputs (no ejemplos sintéticos).

### 6.4 Cuándo se escribe cada doc

- `mlp/README.md`: junto con `mlp/` (Fase 0).
- `ejercicio0/README.md` update: junto con `step_perceptron.py` (Fase 0).
- `ejercicio2/README.md`: stub al inicio de Fase 1, completado después de Fase 2.
- `ejercicio3/README.md`: durante Fase 3.
- `docs/guia_compañeros.md`: **después** de Fase 4, basada en la experiencia real del workflow (no especulativa).

---

## 7. Deliverables y milestones

| Milestone | Deliverable | Criterio de aceptación |
|---|---|---|
| **M0** | `mlp/` + `step_perceptron.py` + tests | `pytest mlp/tests/` y `pytest ejercicio0/tests/` verdes; XOR converge; AND clasifica con error 0 |
| **M1** | Fase 1 completa, `base.json` consolidado | 16 corridas en `ejercicio2/output/`; `base.json` justificado con tabla comparativa; smoke test post-base verde |
| **M2** | Fase 2 completa, tablas comparativas | 15 corridas con K-fold=5; plots Fase 2 generados en `ejercicio2/presentacion/` |
| **M3** | Ej3 ≥98% sobre `digits_test.csv` | `ejercicio3/final_eval.py` reporta accuracy ≥ 0.98; matriz de confusión y métricas por clase generadas |
| **M4** | Presentación armada | `presentacion/` por ejercicio con plots etiquetados (`NN_descripcion.png`); cobertura mínima: curvas de aprendizaje, sweeps de Fase 2, matriz de confusión final, comparación Ej2 vs Ej3 |
| **M5** | Documentación completa | Todos los READMEs actualizados, `guia_compañeros.md` con walkthrough end-to-end, `CLAUDE.md` actualizado |

---

## 8. Riesgos y mitigaciones

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| 1 | Ej3 no llega a 98% incluso con Pack C | media | alto | Probar augmentación más agresiva (rotaciones pequeñas además de ruido), aumentar capacidad del modelo (más capas/neuronas), o entrenar más epochs con LR más chica al final |
| 2 | K-fold con MLPs grandes en Fase 2 es lento (5× tiempo) | alta | medio | Paralelizar folds vía `multiprocessing.Pool` (ya hecho en Ej1, replicar el patrón); reducir epochs si el early stopping confirma convergencia antes |
| 3 | XOR con `[2,2,1]` no converge consistentemente | alta | bajo | Test acepta si al menos 2 de 5 seeds convergen; el TP recomienda calcular a mano para esta arch precisamente porque es educativa |
| 4 | Memory blow-up con `more_digits.csv` (95MB) en Ej3 | media | medio | Cargar como `float32` en vez de `float64` (reduce a la mitad); si sigue siendo problema, mini-batch streaming desde disco |
| 5 | Bug sutil en backprop (no convergencia) | media | alto | Test de gradientes vs diferenciación numérica en `test_activations.py` y `test_losses.py`; XOR como integration test |
| 6 | Inconsistencia entre Fase 1 (k=1) y Fase 2 (k=5) | baja | medio | Re-run del config ganador de Fase 1 con K-fold=5 antes de empezar Fase 2 para confirmar que sigue siendo mejor |
| 7 | Scripts de plotting cambian de spec | baja | bajo | Plots se generan post-hoc desde CSVs; cualquier plot es regenerable sin re-entrenar |

---

## 9. Decisiones explícitas tomadas en este spec

1. **Pack B** como capacidades técnicas core; Pack C como módulos enchufables opcionales.
2. **`mlp/` a nivel raíz** del repo, no dentro de un ejercicio específico.
3. **Ej1 intacto**: no se refactoriza, no se subsume con `mlp/`.
4. **Clase MLP con optimizadores enchufables** (estilo C en la discusión), no monolítico ni mini-framework.
5. **K-fold configurable** (`k_folds=1` ⇒ split simple). Fase 1 usa `k_folds=1` para velocidad; Fase 2 usa `k_folds=5` para rigor.
6. **Cero plots dentro del programa de entrenamiento.** Plotting es post-hoc desde CSVs.
7. **Workflow disciplinado**: Fase 0 (impl) → Fase 1 (búsqueda base) → Fase 2 (one-at-a-time) → Fase 3 (Ej3) → Fase 4 (presentación).
8. **`run_summary.csv` con métricas por clase** (las "MONTON de columnas") para análisis profundo.
9. **`epoch_history.csv` con métricas train+val por época** para diagnóstico over/underfitting visual.
10. **Documentación al final** de cada fase, no especulativa al inicio.

---

## 10. Open questions / future work (fuera de scope inmediato)

- **Calibración del modelo**: el opcional teórico de Ej1 podría aplicar también al Ej2/Ej3. Si sobra tiempo, agregar un análisis de calibración (reliability diagrams).
- **Robustez al ruido**: el opcional práctico del enunciado para Ej2/Ej3. Pack C ya implementa ruido gaussiano para entrenamiento; medir robustez = evaluar el modelo final con ruido en `digits_test.csv` a distintos σ.
- **Interpretabilidad**: el otro opcional. Métodos de atribución (saliency maps) sobre el MLP final. Implementación: gradiente del logit ganador respecto al input. Plot por dígito mostrando qué pixels más contribuyen.
- **Otros optimizadores**: RMSProp, AdaGrad. Fácil de agregar como subclases de `Optimizer`. No los incluimos por scope.
- **Otras activaciones**: LeakyReLU, GELU. Mismo razonamiento.
- **Refactorización futura del Ej1**: si en algún momento se quiere consolidar todo bajo `mlp/`, el código del Ej1 podría migrar a usar `MLP(layer_sizes=[n,1], activations=["sigmoid"])`. Out of scope por riesgo de regresión.

---

## 11. Firmas y aprobación

- **Diseñado por**: Claude (Opus 4.7) en sesión de brainstorming con Nicolás
- **Aprobado por**: pendiente — Nicolás revisa este documento antes de generar el plan de implementación
- **Próximo paso**: invocar `superpowers:writing-plans` para producir el plan de implementación detallado milestone por milestone.
