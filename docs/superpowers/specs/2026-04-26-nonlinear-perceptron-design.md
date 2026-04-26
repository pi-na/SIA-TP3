# Non-Linear Perceptron - Exercise 0 (Validation)

## Goal

Implement a non-linear perceptron with tanh activation from scratch to fit 50 points from y=tanh(x), validating the algorithm before graded exercises.

## Files

### `generate_tanh_dataset.py`

Script to generate a CSV dataset:

- **Function**: `y = tanh(x)` (no noise — exact function for validation)
- **Points**: 50, with x sampled uniformly from [-5, 5]
- **Output**: CSV with columns `x`, `y`
- **CLI**: `--output` flag for output path (default: `tanh_dataset.csv`), `--n_points`, `--seed`

### `nonlinear_perceptron.py`

Main perceptron script, callable from CLI:

- `--csv` (required): path to the input CSV
- `--learning_rate` (default: 0.01): eta
- `--epochs` (default: 5000): max epochs
- `--epsilon` (default: 1e-6): MSE convergence threshold
- `--beta` (default: 1.0): steepness parameter for tanh activation

## Algorithm

Same online learning structure as the linear perceptron, with these differences:

```
1. Load CSV, extract x as feature and y as z_mu (expected output)
2. Normalize z_mu to (-1, 1) range: z_norm = 2*(z - z_min)/(z_max - z_min) - 1
   Store z_min, z_max for denormalization later.
3. Prepend x_0 = 1 column to input data (bias trick)
4. Initialize weights = small random values (shape: 2 — w0=bias, w1=slope)
5. For each epoch:
   6. For each data point mu:
      a. Compute h_mu = w0*x0_mu + w1*x1_mu  (excitation)
      b. Compute O_mu = tanh(beta * h_mu)     (non-linear activation)
      c. Compute theta_prime = beta * (1 - tanh(beta * h_mu)^2)
      d. For each weight i:
         delta_w_i = learning_rate * (z_norm_mu - O_mu) * theta_prime * x_i_mu
         w_i = w_i + delta_w_i
   7. Compute MSE over entire dataset (on normalized values)
   8. If MSE < epsilon: break
9. If max epochs reached: stop
```

Key details:
- **Activation function**: tanh(β * h), image in (-1, 1)
- **Activation derivative**: β * (1 - tanh²(β * h))
- **Error metric**: MSE = (1/P) * sum((z_norm_mu - O_mu)^2)
- **Normalization**: targets normalized to (-1, 1) before training, denormalized for output/plot
- **Learning**: online (update after each data point)

## Output

Creates folder `output_{csv_basename}_{YYYYMMDD_HHMMSS}/` containing:

1. **`weights.csv`**: final weights (w0, w1), beta, final MSE, z_min, z_max (for denormalization)
2. **`plot.png`**: scatter plot of original dataset points + the learned curve (denormalized) overlaid

## Dependencies

- NumPy, Pandas, Matplotlib
