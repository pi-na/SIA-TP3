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
