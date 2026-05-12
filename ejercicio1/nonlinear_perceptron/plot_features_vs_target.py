"""Plot de features clave vs probabilidad — target arriba, predicción abajo.

Para cada una de las 3 features con umbrales (amount_usd, quantity_purchased,
items_viewed_before_purchase), agrega los datos por intervalos de la feature
y promedia la probabilidad de fraude en cada intervalo.

Layout (2×3):
- Fila superior:  3 paneles con el target (mean del BigModel por intervalo).
- Fila inferior:  3 paneles con la predicción del no-lineal (mean por intervalo).

Definición de intervalos:
- quantity_purchased, items_viewed_before_purchase: un intervalo por valor entero.
- amount_usd: intervalos de ancho fijo (50 USD).

El no-lineal usa LR=10⁻³, seed=42, 250 épocas, training online sobre todo el
dataset z-scoreado (mismo setup ganador del sweep de aprendizaje).

Uso:
    python plot_features_vs_target.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nonlinear_perceptron import (  # noqa: E402
    train_perceptron, fit_normalizer, apply_normalizer, sigmoid,
)

CSV = (HERE / ".." / ".." / "data and documentation" /
       "fraud_dataset.csv").resolve()
TARGET = "big_model_fraud_probability"
EVAL = "flagged_fraud"
EXCLUDE = ["timestamp", "device_screen_resolution", "time_since_last_login_s"]

LR = 1e-3
SEED = 42
EPOCHS = 250

FEATURES_TO_PLOT = [
    "amount_usd",
    "quantity_purchased",
    "items_viewed_before_purchase",
]

AMOUNT_BIN_WIDTH = 50.0  # USD por intervalo en amount_usd

OUT_DIR = HERE / "output" / "aprendizaje_20260511_224304" / "plots"


def aggregate_by_intervals(x_raw: np.ndarray, values: np.ndarray, feat: str):
    """Agrega `values` (target o predicción) por intervalos de la feature.

    Devuelve (centers, means, counts, width, edges_label).
    """
    if feat == "amount_usd":
        x_min = float(np.floor(x_raw.min() / AMOUNT_BIN_WIDTH) * AMOUNT_BIN_WIDTH)
        x_max = float(np.ceil(x_raw.max() / AMOUNT_BIN_WIDTH) * AMOUNT_BIN_WIDTH)
        edges = np.arange(x_min, x_max + AMOUNT_BIN_WIDTH, AMOUNT_BIN_WIDTH)
        centers = (edges[:-1] + edges[1:]) / 2
        width = AMOUNT_BIN_WIDTH * 0.95
    else:
        x_int = x_raw.astype(int)
        vmin, vmax = int(x_int.min()), int(x_int.max())
        edges = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
        centers = np.arange(vmin, vmax + 1, dtype=float)
        width = 0.95

    idx = np.clip(np.digitize(x_raw, edges) - 1, 0, len(centers) - 1)
    means = np.full(len(centers), np.nan)
    counts = np.zeros(len(centers), dtype=int)
    for i in range(len(centers)):
        mask = idx == i
        counts[i] = int(mask.sum())
        if counts[i] > 0:
            means[i] = float(values[mask].mean())
    return centers, means, counts, width


def main() -> None:
    df = pd.read_csv(CSV)
    reserved = {TARGET, EVAL, *EXCLUDE}
    feature_cols = [c for c in df.columns if c not in reserved]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    means, stds = fit_normalizer(df, feature_cols)
    df_norm = apply_normalizer(df, means, stds, feature_cols)

    print(f"Entrenando no-lineal: lr={LR}, seed={SEED}, epochs={EPOCHS}, "
          f"N={len(df_norm)} (full dataset)…")
    weights, mse_history, _ = train_perceptron(
        df_norm, feature_cols, TARGET,
        learning_rate=LR, epochs=EPOCHS, epsilon=0.0, seed=SEED, df_test=None,
    )
    print(f"  MSE final: {mse_history[-1]:.5f}")

    X = np.column_stack([np.ones(len(df_norm)),
                         df_norm[feature_cols].to_numpy()])
    preds = sigmoid(X @ weights)
    z = df[TARGET].to_numpy()

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), facecolor="white",
                             sharey=True)

    for col, feat in enumerate(FEATURES_TO_PLOT):
        x_raw = df[feat].to_numpy()
        centers, target_means, counts, width = aggregate_by_intervals(
            x_raw, z, feat,
        )
        _, pred_means, _, _ = aggregate_by_intervals(x_raw, preds, feat)
        valid = counts > 0

        x_label = feat + ("  (intervalos de 50 USD)" if feat == "amount_usd"
                          else "  (intervalos enteros)")

        # Fila superior: target del BigModel
        ax_top = axes[0, col]
        ax_top.set_facecolor("white")
        ax_top.bar(centers[valid], target_means[valid], width=width,
                   alpha=0.85, color="#1f4e79", edgecolor="#1f4e79",
                   linewidth=0.4)
        ax_top.set_ylabel("P(fraude) — media en el intervalo")
        ax_top.set_ylim(0, 1.05)
        ax_top.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax_top.set_title(f"target (BigModel) — {feat}\n"
                         f"({int(valid.sum())} intervalos con datos, "
                         f"N total = {int(counts.sum())})")

        # Fila inferior: predicción del no-lineal
        ax_bot = axes[1, col]
        ax_bot.set_facecolor("white")
        ax_bot.bar(centers[valid], pred_means[valid], width=width,
                   alpha=0.85, color="#d95f0e", edgecolor="#d95f0e",
                   linewidth=0.4)
        ax_bot.set_xlabel(x_label)
        ax_bot.set_ylabel("P(fraude) — media en el intervalo")
        ax_bot.set_ylim(0, 1.05)
        ax_bot.grid(True, alpha=0.25, linestyle="--", axis="y")
        ax_bot.set_title(f"predicción no-lineal — {feat}")

    fig.suptitle(
        f"Probabilidad de fraude agregada por intervalo de la feature\n"
        f"Arriba: target del BigModel  ·  Abajo: predicción del perceptrón no-lineal  "
        f"(LR={LR}, seed={SEED}, {EPOCHS} épocas — MSE final={mse_history[-1]:.5f})",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "features_target_vs_prediccion_por_intervalo.png"
    fig.savefig(out_path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"Plot guardado en: {out_path}")


if __name__ == "__main__":
    main()
