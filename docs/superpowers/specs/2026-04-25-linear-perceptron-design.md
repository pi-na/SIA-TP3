# Linear Perceptron - Exercise 0 (Validation)

## Goal

Implement a linear perceptron (Adaline) from scratch to fit a linear dataset of 50 points. This validates the perceptron implementation before tackling graded exercises.

## Files

### `generate_linear_dataset.py`

Script to generate a CSV dataset:

- **Function**: `y = 3x + 2 + noise`, where noise ~ N(0, 0.5)
- **Points**: 50, with x sampled uniformly from [-10, 10]
- **Output**: CSV with columns `x`, `y`
- **CLI**: `--output` flag for output path (default: `linear_dataset.csv`)

### `linear_perceptron.py`

Main perceptron script, callable from CLI:

- `--csv` (required): path to the input CSV
- `--learning_rate` (default: 0.01): eta
- `--epochs` (default: 1000): max epochs
- `--epsilon` (default: 1e-4): MSE convergence threshold

## Algorithm

Following the pseudocode from class notes (online learning):

```
1. Load CSV, extract x as feature and y as z_mu (expected output)
2. Prepend x_0 = 1 column to input data (bias trick)
3. Initialize weights = small random values (shape: 2 — w0=bias, w1=slope)
4. For each epoch:
   5. For each data point mu:
      a. Compute h_mu = w0*x0_mu + w1*x1_mu  (excitation)
      b. Compute O_mu = h_mu  (identity activation)
      c. For each weight i:
         delta_w_i = learning_rate * (z_mu - O_mu) * x_i_mu
         w_i = w_i + delta_w_i
   6. Compute MSE over entire dataset
   7. If MSE < epsilon: break
8. If max epochs reached: stop
```

Key details:
- **Activation function**: identity (O = h)
- **Error metric**: MSE = (1/P) * sum((z_mu - O_mu)^2)
- **Learning**: online (update after each data point)
- **No ML frameworks** — only NumPy for math

## Output

Creates folder `output_{csv_basename}_{YYYYMMDD_HHMMSS}/` containing:

1. **`weights.csv`**: final weights (w0, w1) and final MSE
2. **`plot.png`**: scatter plot of dataset points + the learned linear function overlaid

## Dependencies

- NumPy (matrix ops)
- Pandas (CSV loading)
- Matplotlib (plotting)
