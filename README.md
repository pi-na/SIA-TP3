# SIA TP3 — Perceptrones (ITBA, 1Q 2026)

Implementación de perceptrones desde cero con NumPy. Este README cubre los scripts del **Ejercicio 0 (validación)**: perceptrón lineal y perceptrón no lineal.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib pytest
```

## Tests

```bash
.venv/bin/pytest -v
```

15 tests cubren generación de datasets, entrenamiento, convergencia y pipeline de salida.

---

## Perceptrón Lineal (Adaline)

Activación identidad, regla de actualización online.

### Generar dataset

```bash
.venv/bin/python generate_linear_dataset.py --output linear_dataset.csv --seed 42
```

Genera 50 puntos de `y = 3x + 2 + ruido`, con `x ∈ [-10, 10]`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--output` | `linear_dataset.csv` | Path de salida |
| `--n_points` | `50` | Cantidad de puntos |
| `--seed` | None | Semilla del RNG |

### Entrenar

```bash
.venv/bin/python linear_perceptron.py --csv linear_dataset.csv --learning_rate 0.001 --epochs 1000
```

| Flag | Default | Descripción |
|------|---------|-------------|
| `--csv` | (requerido) | CSV de entrada con columnas `x`, `y` |
| `--learning_rate` | `0.01` | η |
| `--epochs` | `1000` | Máximo de épocas |
| `--epsilon` | `1e-4` | Umbral de MSE para corte temprano |

**Salida:** carpeta `output_<csv_basename>_<timestamp>/` con:
- `weights.csv`: `w0` (bias), `w1` (pendiente), `mse` final
- `plot.png`: scatter de los datos + recta aprendida

---

## Perceptrón No Lineal

Activación `tanh(β·h)`, derivada `β·(1 − tanh²(β·h))` en la actualización, targets normalizados a `(−1, 1)` antes de entrenar.

### Generar dataset

```bash
.venv/bin/python generate_tanh_dataset.py --output tanh_dataset.csv --seed 42
```

Genera 50 puntos exactos de `y = tanh(x)` (sin ruido), con `x ∈ [-5, 5]`.

| Flag | Default | Descripción |
|------|---------|-------------|
| `--output` | `tanh_dataset.csv` | Path de salida |
| `--n_points` | `50` | Cantidad de puntos |
| `--seed` | None | Semilla del RNG |

### Entrenar

```bash
.venv/bin/python nonlinear_perceptron.py --csv tanh_dataset.csv --learning_rate 0.01 --epochs 5000 --beta 1.0
```

| Flag | Default | Descripción |
|------|---------|-------------|
| `--csv` | (requerido) | CSV de entrada con columnas `x`, `y` |
| `--learning_rate` | `0.01` | η |
| `--epochs` | `5000` | Máximo de épocas |
| `--epsilon` | `1e-6` | Umbral de MSE (en escala normalizada) |
| `--beta` | `1.0` | Pendiente de la tanh |

**Salida:** carpeta `output_<csv_basename>_<timestamp>/` con:
- `weights.csv`: `w0`, `w1`, `beta`, `mse` final, `z_min`, `z_max` (los dos últimos para denormalizar)
- `plot.png`: scatter de los datos + curva aprendida (denormalizada al rango original)

### Algoritmo

1. Normalizar `z` a `(−1, 1)`: `z_norm = 2·(z − z_min) / (z_max − z_min) − 1`
2. Por cada punto μ (online):
   - `h = w·x`
   - `O = tanh(β·h)`
   - `Δw = η · (z_norm − O) · β·(1 − O²) · x`
3. Calcular MSE en escala normalizada al final de cada época. Cortar si `MSE < ε`.

Para graficar, denormalizar la salida: `y = (O + 1)·(z_max − z_min)/2 + z_min`.
