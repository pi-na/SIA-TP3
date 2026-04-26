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
