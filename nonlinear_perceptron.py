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

    # Prepend x_0 = 1 for bias trick -> X shape: (P, 2)
    X = np.column_stack([np.ones(P), x_raw])

    # Initialize weights with small random values
    rng = np.random.default_rng()
    weights = rng.uniform(-0.1, 0.1, size=2)

    mse_history = []

    for epoch in range(epochs):
        # Online learning: iterate over each data point
        for mu in range(P):
            h_mu = np.dot(weights, X[mu])
            O_mu = np.tanh(beta * h_mu)
            theta_prime = beta * (1.0 - O_mu ** 2)

            error = z_norm[mu] - O_mu
            weights = weights + learning_rate * error * theta_prime * X[mu]

        # MSE over the entire dataset (on normalized scale)
        predictions = np.tanh(beta * (X @ weights))
        mse = np.mean((z_norm - predictions) ** 2)
        mse_history.append(mse)

        if mse < epsilon:
            break

    return weights, mse_history, z_min, z_max


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
