# Non-Linear Perceptron Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a non-linear perceptron with `tanh(β·h)` activation that fits 50 points from `y=tanh(x)`, validating the algorithm before graded exercises.

**Architecture:** Two scripts following the same pattern as the existing linear perceptron: a dataset generator (`generate_tanh_dataset.py`) producing a CSV with 50 noiseless points, and a trainer (`nonlinear_perceptron.py`) that normalizes targets to `(-1, 1)`, performs online gradient descent with the tanh activation derivative, and saves weights/plot to a timestamped output folder. Targets are normalized before training (because tanh's image is bounded in `(-1, 1)`) and denormalized when plotting the learned curve.

**Tech Stack:** Python 3, NumPy, Pandas, Matplotlib, pytest. Use the existing `.venv`.

---

## File Structure

- **Create** `generate_tanh_dataset.py` — CLI that writes a CSV of 50 points sampled from `y=tanh(x)` with `x ∈ [-5, 5]`. Mirrors `generate_linear_dataset.py`.
- **Create** `nonlinear_perceptron.py` — CLI trainer with `train_perceptron`, `run_and_save`, `main`. Mirrors `linear_perceptron.py` but adds: target normalization, `beta` parameter, tanh activation + derivative, denormalization for plotting.
- **Create** `tests/test_generate_tanh_dataset.py` — tests for shape, x range, y=tanh(x) match.
- **Create** `tests/test_nonlinear_perceptron.py` — tests for training convergence, output artifacts, normalization round-trip.

Each file has a single responsibility (data generation vs. training vs. tests). No shared module — the linear and non-linear perceptrons are kept as independent validation scripts to match the academic exercise structure.

---

## Task 1: Dataset generator — `generate_tanh_dataset.py`

**Files:**
- Create: `generate_tanh_dataset.py`
- Test: `tests/test_generate_tanh_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_generate_tanh_dataset.py`:

```python
import numpy as np
from generate_tanh_dataset import generate_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_points=50, seed=42)
    assert df.shape == (50, 2)
    assert list(df.columns) == ["x", "y"]


def test_generate_dataset_x_range():
    df = generate_dataset(n_points=50, seed=42)
    assert df["x"].min() >= -5.0
    assert df["x"].max() <= 5.0


def test_generate_dataset_y_equals_tanh_x():
    """Dataset is noiseless: y must equal tanh(x) exactly."""
    df = generate_dataset(n_points=50, seed=42)
    np.testing.assert_allclose(df["y"].values, np.tanh(df["x"].values), atol=1e-12)


def test_generate_dataset_seed_reproducibility():
    df1 = generate_dataset(n_points=50, seed=123)
    df2 = generate_dataset(n_points=50, seed=123)
    np.testing.assert_array_equal(df1["x"].values, df2["x"].values)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_generate_tanh_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'generate_tanh_dataset'`.

- [ ] **Step 3: Implement `generate_tanh_dataset.py`**

Create `generate_tanh_dataset.py`:

```python
import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_points=50, seed=None):
    """Generate y = tanh(x) dataset (no noise — exact function for validation)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, n_points)
    y = np.tanh(x)
    return pd.DataFrame({"x": x, "y": y})


def main():
    parser = argparse.ArgumentParser(description="Generate y=tanh(x) dataset")
    parser.add_argument("--output", default="tanh_dataset.csv", help="Output CSV path")
    parser.add_argument("--n_points", type=int, default=50, help="Number of data points")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    df = generate_dataset(n_points=args.n_points, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output} ({len(df)} points)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_generate_tanh_dataset.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add generate_tanh_dataset.py tests/test_generate_tanh_dataset.py
git commit -m "feat: add tanh dataset generator for nonlinear perceptron validation"
```

---

## Task 2: Trainer — convergence test and `train_perceptron`

**Files:**
- Create: `nonlinear_perceptron.py`
- Test: `tests/test_nonlinear_perceptron.py`

- [ ] **Step 1: Write the failing convergence test**

Create `tests/test_nonlinear_perceptron.py`:

```python
import numpy as np
import pandas as pd
from nonlinear_perceptron import train_perceptron


def test_perceptron_learns_tanh_normalized_target():
    """
    For a linearly separable target (z=ax), the non-linear perceptron with
    tanh activation should drive normalized MSE to a small value.
    Use y = x to make sure the model can fit it after target normalization.
    """
    x = np.linspace(-2, 2, 40)
    y = x  # linear target — easy for the model
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate=0.05, epochs=5000, epsilon=1e-6, beta=1.0
    )

    assert mse_history[-1] < 1e-2, f"final MSE too high: {mse_history[-1]}"


def test_perceptron_mse_decreases():
    """MSE should drop substantially over training."""
    x = np.linspace(-5, 5, 50)
    y = np.tanh(x)
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate=0.01, epochs=200, epsilon=1e-10, beta=1.0
    )

    assert mse_history[-1] < mse_history[0], "MSE should decrease"


def test_normalization_bounds_returned():
    """train_perceptron must return z_min, z_max so callers can denormalize."""
    x = np.linspace(-5, 5, 50)
    y = np.tanh(x)
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate=0.01, epochs=10, epsilon=1e-10, beta=1.0
    )

    assert z_min == y.min()
    assert z_max == y.max()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest tests/test_nonlinear_perceptron.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'nonlinear_perceptron'`.

- [ ] **Step 3: Implement `train_perceptron`**

Create `nonlinear_perceptron.py`:

```python
# nonlinear_perceptron.py
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_perceptron(df, learning_rate=0.01, epochs=5000, epsilon=1e-6, beta=1.0):
    """
    Train a non-linear perceptron with tanh activation using online learning.

    Targets are normalized to (-1, 1) before training because tanh's image is bounded.

    Args:
        df: DataFrame with columns 'x' and 'y'
        learning_rate: eta
        epochs: max epochs
        epsilon: MSE convergence threshold (computed on normalized targets)
        beta: steepness parameter for tanh activation

    Returns:
        weights: array [w0 (bias), w1 (slope)]
        mse_history: list of MSE per epoch (on normalized scale)
        z_min: min of original target (for denormalization)
        z_max: max of original target (for denormalization)
    """
    x_raw = df["x"].values
    z = df["y"].values
    P = len(z)

    # Normalize targets to (-1, 1): z_norm = 2*(z - z_min)/(z_max - z_min) - 1
    z_min = z.min()
    z_max = z.max()
    z_norm = 2.0 * (z - z_min) / (z_max - z_min) - 1.0

    # Prepend x_0 = 1 for bias trick
    X = np.column_stack([np.ones(P), x_raw])

    rng = np.random.default_rng()
    weights = rng.uniform(-0.1, 0.1, size=2)

    mse_history = []

    for epoch in range(epochs):
        for mu in range(P):
            h_mu = np.dot(weights, X[mu])
            O_mu = np.tanh(beta * h_mu)
            theta_prime = beta * (1.0 - np.tanh(beta * h_mu) ** 2)

            error = z_norm[mu] - O_mu
            weights = weights + learning_rate * error * theta_prime * X[mu]

        # MSE over the entire dataset (on normalized scale)
        predictions = np.tanh(beta * (X @ weights))
        mse = np.mean((z_norm - predictions) ** 2)
        mse_history.append(mse)

        if mse < epsilon:
            break

    return weights, mse_history, z_min, z_max
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest tests/test_nonlinear_perceptron.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add nonlinear_perceptron.py tests/test_nonlinear_perceptron.py
git commit -m "feat: implement nonlinear perceptron training with tanh activation"
```

---

## Task 3: Output pipeline — `run_and_save`

**Files:**
- Modify: `nonlinear_perceptron.py`
- Test: `tests/test_nonlinear_perceptron.py`

- [ ] **Step 1: Add the failing pipeline test**

Append to `tests/test_nonlinear_perceptron.py`:

```python
import os


def test_run_and_save_creates_output(tmp_path):
    """Full pipeline: train + save should create weights.csv and plot.png."""
    from nonlinear_perceptron import run_and_save

    x = np.linspace(-5, 5, 50)
    y = np.tanh(x)
    csv_path = tmp_path / "test_tanh.csv"
    pd.DataFrame({"x": x, "y": y}).to_csv(csv_path, index=False)

    output_dir = tmp_path / "output_test"
    run_and_save(
        csv_path=str(csv_path),
        learning_rate=0.01,
        epochs=200,
        epsilon=1e-6,
        beta=1.0,
        output_dir=str(output_dir),
    )

    assert os.path.isfile(output_dir / "weights.csv")
    assert os.path.isfile(output_dir / "plot.png")

    wdf = pd.read_csv(output_dir / "weights.csv")
    for col in ["w0", "w1", "beta", "mse", "z_min", "z_max"]:
        assert col in wdf.columns, f"missing column {col}"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest tests/test_nonlinear_perceptron.py::test_run_and_save_creates_output -v`
Expected: FAIL with `ImportError: cannot import name 'run_and_save'`.

- [ ] **Step 3: Implement `run_and_save`**

Append to `nonlinear_perceptron.py`:

```python
def run_and_save(csv_path, learning_rate, epochs, epsilon, beta, output_dir):
    """Train non-linear perceptron and save weights + plot to output_dir."""
    df = pd.read_csv(csv_path)
    weights, mse_history, z_min, z_max = train_perceptron(
        df, learning_rate, epochs, epsilon, beta
    )

    os.makedirs(output_dir, exist_ok=True)

    # Save weights + normalization bounds
    wdf = pd.DataFrame({
        "w0": [weights[0]],
        "w1": [weights[1]],
        "beta": [beta],
        "mse": [mse_history[-1]],
        "z_min": [z_min],
        "z_max": [z_max],
    })
    wdf.to_csv(os.path.join(output_dir, "weights.csv"), index=False)

    # Plot: dataset (original scale) + learned curve (denormalized)
    x = df["x"].values
    y = df["y"].values
    x_line = np.linspace(x.min(), x.max(), 200)
    h_line = weights[0] + weights[1] * x_line
    o_norm = np.tanh(beta * h_line)
    # Denormalize: z = (o_norm + 1) * (z_max - z_min) / 2 + z_min
    y_line = (o_norm + 1.0) * (z_max - z_min) / 2.0 + z_min

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Datos", alpha=0.7)
    plt.plot(x_line, y_line, "r-", linewidth=2, label="Perceptrón no lineal (tanh)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(
        f"Perceptrón No Lineal (β={beta}) — MSE final: {mse_history[-1]:.6f} "
        f"({len(mse_history)} épocas)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Weights: w0={weights[0]:.4f}, w1={weights[1]:.4f}, beta={beta}")
    print(f"MSE final: {mse_history[-1]:.6f} ({len(mse_history)} epochs)")
    print(f"Output saved to {output_dir}")
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest tests/test_nonlinear_perceptron.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add nonlinear_perceptron.py tests/test_nonlinear_perceptron.py
git commit -m "feat: add output pipeline for nonlinear perceptron (weights CSV + plot)"
```

---

## Task 4: CLI entrypoint

**Files:**
- Modify: `nonlinear_perceptron.py`

- [ ] **Step 1: Implement `main()` and `__main__` guard**

Append to `nonlinear_perceptron.py`:

```python
def main():
    parser = argparse.ArgumentParser(description="Non-Linear Perceptron (tanh activation)")
    parser.add_argument("--csv", required=True, help="Path to input CSV with columns x, y")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=5000, help="Max epochs (default: 5000)")
    parser.add_argument("--epsilon", type=float, default=1e-6, help="MSE convergence threshold (default: 1e-6)")
    parser.add_argument("--beta", type=float, default=1.0, help="tanh steepness parameter (default: 1.0)")
    args = parser.parse_args()

    csv_basename = os.path.splitext(os.path.basename(args.csv))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{csv_basename}_{timestamp}"

    run_and_save(
        args.csv, args.learning_rate, args.epochs, args.epsilon, args.beta, output_dir
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test the CLI end-to-end**

Run:

```bash
.venv/bin/python generate_tanh_dataset.py --output tanh_dataset.csv --seed 42
.venv/bin/python nonlinear_perceptron.py --csv tanh_dataset.csv --learning_rate 0.01 --epochs 2000 --beta 1.0
```

Expected:
- `tanh_dataset.csv` has 50 rows.
- A folder `output_tanh_dataset_<timestamp>/` is created with `weights.csv` and `plot.png`.
- Console prints final weights and MSE.

- [ ] **Step 3: Run the full test suite**

Run: `.venv/bin/pytest -v`
Expected: all tests pass (existing linear-perceptron tests + the 4 new generator tests + the 4 new non-linear perceptron tests = 11 total).

- [ ] **Step 4: Commit**

```bash
git add nonlinear_perceptron.py
git commit -m "feat: add CLI entrypoint for nonlinear perceptron"
```

---

## Self-Review Notes

- **Spec coverage:** generator (Task 1), normalization + tanh + derivative + online learning + epsilon stop (Task 2), weights.csv with all six columns and denormalized plot (Task 3), CLI flags `--csv --learning_rate --epochs --epsilon --beta` and timestamped output folder (Task 4). All spec items mapped.
- **No placeholders:** every step has either complete code or an exact command + expected outcome.
- **Type consistency:** `train_perceptron` signature returns `(weights, mse_history, z_min, z_max)` everywhere it's referenced; `run_and_save` takes `(csv_path, learning_rate, epochs, epsilon, beta, output_dir)` and is called with the same kwargs in tests and `main()`.
