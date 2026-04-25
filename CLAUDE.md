# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University assignment (ITBA - Sistemas de Inteligencia Artificial, 2026) implementing perceptrons from scratch. The project has three exercises plus validation tasks, all centered on building and evaluating simple and multilayer perceptrons **without** high-level ML frameworks (no sklearn estimators, no PyTorch/TensorFlow layers — use NumPy for matrix operations).

## Exercises

### Validation (not graded, but important for correctness)
- **Step perceptron**: AND gate with bipolar inputs ({-1,1}), expected output {-1,-1,-1,1}
- **Linear perceptron**: fit ~50 samples from y=x
- **Non-linear perceptron**: fit ~50 samples from y=tanh(x)
- **Multilayer perceptron**: XOR gate, architectures [2,2,1] and [2,3,2,1]

### Exercise 1 — Fraud detection (Knowledge Distillation)
- Dataset: `data and documentation/fraud_dataset.csv` (transactions.csv in the assignment)
- Goal: train a TinyModel (simple perceptron) to replicate BigModel's fraud probability output
- **Target column for training**: `big_model_fraud_probability` (float in [0,1])
- **`flagged_fraud` column MUST NOT be used during training** — it is ground truth for evaluation only
- Features: timestamp, amount_usd, quantity_purchased, session_duration_seconds, days_since_last_purchase, account_age_days, device_screen_resolution, time_since_last_login_s, items_viewed_before_purchase
- Compare linear vs non-linear simple perceptron (underfitting, capacity saturation)
- Then do a generalization study on the selected perceptron (metrics, train/test split strategy, fraud threshold recommendation)

### Exercise 2 — Digit classification (MLP)
- Training: `data and documentation/digits.csv` — columns: `label`, `image` (stringified flat array of 784 floats, 28x28 pixels, values in [0,1])
- Test: `data and documentation/digits_test.csv` (same format — treat as production data, do NOT use for hyperparameter tuning)
- Use `data and documentation/digit_dataset_loader.py` to load and visualize
- Classify digits 0-9 with a multilayer perceptron
- Must explore: learning rate variants, architecture variants, optimization mechanisms

### Exercise 3 — Improved digit classification (target >= 98% accuracy)
- Additional training data: `data and documentation/more_digits.csv`
- Test set: same `digits_test.csv` as Exercise 2
- Analyze what techniques and what external factors improved performance

## Tech Stack

- **Language**: Python 3
- **Core dependencies**: NumPy (matrix ops), Pandas (data loading), Matplotlib (plotting)
- Implementations must be from scratch — perceptron logic, backpropagation, activation functions, etc.

## Key Implementation Notes

- Use **bipolar representation** (-1/+1) for logical gates, not binary (0/1)
- Activation functions needed: step (sign), linear (identity), sigmoid/logistic (output in [0,1] for fraud probability), tanh, optionally ReLU
- For Exercise 1, the output must be in [0,1] — use an appropriate activation (e.g., sigmoid/logistic)
- Digit images are 784-dimensional vectors (28x28 flattened), pixel values in [0,1]
- Use matrix operations (not loops over neurons) for performance
- The assignment recommends: progress reporting during training, extensible config (save/load), model serialization, and separation of experiment logging from analysis/plotting

## Data Loading

```python
# Fraud dataset
import pandas as pd
df = pd.read_csv("data and documentation/fraud_dataset.csv")

# Digit datasets — use the provided loader
from digit_dataset_loader import load_dataset, plot_sample
df = load_dataset("data and documentation/digits.csv")
# df["image"] is a numpy array of shape (784,), df["label"] is the digit
```
