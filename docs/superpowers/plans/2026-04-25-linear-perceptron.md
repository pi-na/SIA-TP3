# Linear Perceptron (Exercise 0 - Validation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a linear perceptron from scratch to fit 50 points from y=3x+2, validating the Adaline algorithm before graded exercises.

**Architecture:** Two standalone scripts — one generates the dataset CSV, the other trains the perceptron and produces output. No shared modules needed; each script is self-contained with argparse CLI.

**Tech Stack:** Python 3, NumPy, Pandas, Matplotlib

---

## File Structure

| File | Responsibility |
|------|---------------|
| `generate_linear_dataset.py` | Generate 50-point CSV from y=3x+2+noise |
| `linear_perceptron.py` | Train linear perceptron, save weights + plot |
| `tests/test_generate_dataset.py` | Tests for dataset generation |
| `tests/test_linear_perceptron.py` | Tests for perceptron training logic |

---

### Task 1: Dataset Generator

**Files:**
- Create: `tests/test_generate_dataset.py`
- Create: `generate_linear_dataset.py`

- [ ] **Step 1: Write failing test for dataset generation**

```python
# tests/test_generate_dataset.py
import os
import tempfile
import pandas as pd
import numpy as np
from generate_linear_dataset import generate_dataset


def test_generate_dataset_shape_and_columns():
    df = generate_dataset(n_points=50, seed=42)
    assert df.shape == (50, 2)
    assert list(df.columns) == ["x", "y"]


def test_generate_dataset_x_range():
    df = generate_dataset(n_points=50, seed=42)
    assert df["x"].min() >= -10.0
    assert df["x"].max() <= 10.0


def test_generate_dataset_y_follows_linear_trend():
    """With known seed, y should be close to 3x+2."""
    df = generate_dataset(n_points=1000, seed=42)
    # Fit a line with numpy and check coefficients are close
    coeffs = np.polyfit(df["x"], df["y"], 1)
    # slope ~3, intercept ~2 (with 1000 points, noise averages out)
    assert abs(coeffs[0] - 3.0) < 0.2
    assert abs(coeffs[1] - 2.0) < 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python -m pytest tests/test_generate_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'generate_linear_dataset'`

- [ ] **Step 3: Implement generate_linear_dataset.py**

```python
# generate_linear_dataset.py
import argparse
import numpy as np
import pandas as pd


def generate_dataset(n_points=50, seed=None):
    """Generate y = 3x + 2 + noise dataset."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-10, 10, n_points)
    noise = rng.normal(0, 0.5, n_points)
    y = 3 * x + 2 + noise
    return pd.DataFrame({"x": x, "y": y})


def main():
    parser = argparse.ArgumentParser(description="Generate linear dataset")
    parser.add_argument("--output", default="linear_dataset.csv", help="Output CSV path")
    parser.add_argument("--n_points", type=int, default=50, help="Number of data points")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    df = generate_dataset(n_points=args.n_points, seed=args.seed)
    df.to_csv(args.output, index=False)
    print(f"Dataset saved to {args.output} ({len(df)} points)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python -m pytest tests/test_generate_dataset.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add generate_linear_dataset.py tests/test_generate_dataset.py
git commit -m "feat: add linear dataset generator for exercise 0 validation"
```

---

### Task 2: Linear Perceptron — Core Training Logic

**Files:**
- Create: `tests/test_linear_perceptron.py`
- Create: `linear_perceptron.py`

- [ ] **Step 1: Write failing test for perceptron training**

```python
# tests/test_linear_perceptron.py
import numpy as np
import pandas as pd
from linear_perceptron import train_perceptron


def test_perceptron_learns_exact_line():
    """Given y=3x+2 with no noise, perceptron should learn w0~2, w1~3."""
    x = np.linspace(-5, 5, 30)
    y = 3 * x + 2
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history = train_perceptron(df, learning_rate=0.001, epochs=500, epsilon=1e-6)

    # weights[0] = bias (w0 ~ 2), weights[1] = slope (w1 ~ 3)
    assert abs(weights[0] - 2.0) < 0.5, f"bias w0={weights[0]}, expected ~2"
    assert abs(weights[1] - 3.0) < 0.5, f"slope w1={weights[1]}, expected ~3"


def test_perceptron_mse_decreases():
    """MSE should generally decrease over training."""
    x = np.linspace(-5, 5, 30)
    y = 3 * x + 2
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history = train_perceptron(df, learning_rate=0.001, epochs=100, epsilon=1e-10)

    assert mse_history[-1] < mse_history[0], "MSE should decrease"


def test_perceptron_early_stop():
    """With exact data and enough epochs, should converge before max epochs."""
    x = np.linspace(-1, 1, 20)  # small range for fast convergence
    y = 2 * x + 1
    df = pd.DataFrame({"x": x, "y": y})

    weights, mse_history = train_perceptron(df, learning_rate=0.01, epochs=5000, epsilon=1e-4)

    assert len(mse_history) < 5000, f"Should converge early, got {len(mse_history)} epochs"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python -m pytest tests/test_linear_perceptron.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'linear_perceptron'`

- [ ] **Step 3: Implement the train_perceptron function**

```python
# linear_perceptron.py
import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def train_perceptron(df, learning_rate=0.01, epochs=1000, epsilon=1e-4):
    """
    Train a linear perceptron (Adaline) using online learning.

    Args:
        df: DataFrame with columns 'x' and 'y'
        learning_rate: eta
        epochs: max epochs
        epsilon: MSE convergence threshold

    Returns:
        weights: array [w0 (bias), w1 (slope)]
        mse_history: list of MSE per epoch
    """
    # Extract data
    x_raw = df["x"].values
    z = df["y"].values  # expected output
    P = len(z)

    # Prepend x_0 = 1 for bias trick -> X shape: (P, 2)
    X = np.column_stack([np.ones(P), x_raw])

    # Initialize weights with small random values
    rng = np.random.default_rng()
    weights = rng.uniform(-0.1, 0.1, size=2)

    mse_history = []

    for epoch in range(epochs):
        # Online learning: iterate over each data point
        for mu in range(P):
            # Excitation (h_mu) — identity activation so O_mu = h_mu
            O_mu = np.dot(weights, X[mu])

            # Update each weight
            error = z[mu] - O_mu
            weights = weights + learning_rate * error * X[mu]

        # Compute MSE over entire dataset
        predictions = X @ weights
        mse = np.mean((z - predictions) ** 2)
        mse_history.append(mse)

        if mse < epsilon:
            break

    return weights, mse_history
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python -m pytest tests/test_linear_perceptron.py -v`
Expected: 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add linear_perceptron.py tests/test_linear_perceptron.py
git commit -m "feat: implement linear perceptron training with online learning"
```

---

### Task 3: CLI + Output (weights CSV, plot, output folder)

**Files:**
- Modify: `linear_perceptron.py`

- [ ] **Step 1: Write failing test for output generation**

Add to `tests/test_linear_perceptron.py`:

```python
import os
import tempfile


def test_run_and_save_creates_output(tmp_path):
    """Full pipeline: train + save should create weights.csv and plot.png."""
    from linear_perceptron import run_and_save

    # Create a small test CSV
    x = np.linspace(-5, 5, 20)
    y = 3 * x + 2
    csv_path = tmp_path / "test_data.csv"
    pd.DataFrame({"x": x, "y": y}).to_csv(csv_path, index=False)

    output_dir = tmp_path / "output_test"
    run_and_save(
        csv_path=str(csv_path),
        learning_rate=0.001,
        epochs=500,
        epsilon=1e-4,
        output_dir=str(output_dir),
    )

    assert os.path.isfile(output_dir / "weights.csv")
    assert os.path.isfile(output_dir / "plot.png")

    # Check weights.csv content
    wdf = pd.read_csv(output_dir / "weights.csv")
    assert "w0" in wdf.columns
    assert "w1" in wdf.columns
    assert "mse" in wdf.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python -m pytest tests/test_linear_perceptron.py::test_run_and_save_creates_output -v`
Expected: FAIL — `ImportError: cannot import name 'run_and_save'`

- [ ] **Step 3: Implement run_and_save and CLI main**

Add to `linear_perceptron.py`:

```python
def run_and_save(csv_path, learning_rate, epochs, epsilon, output_dir):
    """Train perceptron and save results to output_dir."""
    df = pd.read_csv(csv_path)
    weights, mse_history = train_perceptron(df, learning_rate, epochs, epsilon)

    os.makedirs(output_dir, exist_ok=True)

    # Save weights
    wdf = pd.DataFrame({"w0": [weights[0]], "w1": [weights[1]], "mse": [mse_history[-1]]})
    wdf.to_csv(os.path.join(output_dir, "weights.csv"), index=False)

    # Plot
    x = df["x"].values
    y = df["y"].values
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = weights[0] + weights[1] * x_line

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, label="Datos", alpha=0.7)
    plt.plot(x_line, y_line, "r-", linewidth=2, label=f"Perceptrón: y = {weights[1]:.3f}x + {weights[0]:.3f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Perceptrón Lineal — MSE final: {mse_history[-1]:.6f} ({len(mse_history)} épocas)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "plot.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Weights: w0={weights[0]:.4f}, w1={weights[1]:.4f}")
    print(f"MSE final: {mse_history[-1]:.6f} ({len(mse_history)} epochs)")
    print(f"Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Linear Perceptron (Adaline)")
    parser.add_argument("--csv", required=True, help="Path to input CSV with columns x, y")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate (default: 0.01)")
    parser.add_argument("--epochs", type=int, default=1000, help="Max epochs (default: 1000)")
    parser.add_argument("--epsilon", type=float, default=1e-4, help="MSE convergence threshold (default: 1e-4)")
    args = parser.parse_args()

    csv_basename = os.path.splitext(os.path.basename(args.csv))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_{csv_basename}_{timestamp}"

    run_and_save(args.csv, args.learning_rate, args.epochs, args.epsilon, output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python -m pytest tests/ -v`
Expected: 7 tests PASS (3 from dataset + 4 from perceptron)

- [ ] **Step 5: Commit**

```bash
git add linear_perceptron.py tests/test_linear_perceptron.py
git commit -m "feat: add CLI and output generation (weights CSV + plot)"
```

---

### Task 4: End-to-End Validation

**Files:** None (manual verification)

- [ ] **Step 1: Generate the dataset**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python generate_linear_dataset.py --output linear_dataset.csv --seed 42`
Expected: `Dataset saved to linear_dataset.csv (50 points)`

- [ ] **Step 2: Train the perceptron**

Run: `cd /Users/tomaspinausig/code/SIA-TP3 && python linear_perceptron.py --csv linear_dataset.csv --learning_rate 0.001 --epochs 2000 --epsilon 1e-4`
Expected: Output showing w0 close to 2, w1 close to 3, MSE < 1. An `output_linear_dataset_*` folder created with `weights.csv` and `plot.png`.

- [ ] **Step 3: Verify output files**

Check that `output_linear_dataset_*/weights.csv` exists and has reasonable values.
Open `output_linear_dataset_*/plot.png` and confirm the red line fits the scatter data.

- [ ] **Step 4: Commit generated dataset**

```bash
git add linear_dataset.csv
git commit -m "chore: add generated linear dataset for validation"
```
