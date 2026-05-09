"""Plot de generalización — lee raw_history.csv y genera gap_train_test.png.

Requiere que run_generalizacion.py haya corrido primero.

Uso:
    python plot_generalizacion.py [--input PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT        = Path(__file__).resolve().parent
DEFAULT_IN  = ROOT / "analisis_generalizacion" / "raw_history.csv"
DEFAULT_OUT = ROOT / "analisis_generalizacion" / "gap_train_test.png"
CONFIG_PATH = ROOT / "analisis_generalizacion" / "config.json"

COLORS = {"train": "#2196F3", "test": "#FF5722"}
LABELS = {
    "linear":    "Lineal  (lr=1e-4)",
    "nonlinear": "No-lineal  (lr=1e-2)",
}


def aggregate(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.groupby("epoch")[col].agg(["mean", "std"]).reset_index()


def plot_perceptron(ax: plt.Axes, df: pd.DataFrame, name: str) -> None:
    train_agg = aggregate(df, "mse_train")
    epochs  = train_agg["epoch"].values
    m_train = train_agg["mean"].values
    s_train = train_agg["std"].values

    ax.plot(epochs, m_train, color=COLORS["train"], lw=1.8,
            label="MSE train (media 25 corridas)")
    ax.fill_between(epochs, m_train - s_train, m_train + s_train,
                    color=COLORS["train"], alpha=0.18, label="±1 std train")

    if "mse_test" in df.columns and df["mse_test"].notna().any():
        test_agg = aggregate(df, "mse_test")
        m_test = test_agg["mean"].values
        s_test = test_agg["std"].values
        ax.plot(epochs, m_test, color=COLORS["test"], lw=1.8, ls="--",
                label="MSE test (media 25 corridas)")
        ax.fill_between(epochs, m_test - s_test, m_test + s_test,
                        color=COLORS["test"], alpha=0.18, label="±1 std test")
    else:
        cfg = json.loads(CONFIG_PATH.read_text())
        lr  = cfg["perceptrons"][name]["lr"]
        raw_path = (ROOT / "lineal_perceptron" / "analisis_outputs"
                    / "sweep_lr" / "multiseed" / "raw.csv")
        raw  = pd.read_csv(raw_path)
        best = raw[raw["lr"].astype(float).round(6) == round(lr, 6)]
        m_ref = best["mse_test"].mean()
        s_ref = best["mse_test"].std()
        ax.axhline(m_ref, color=COLORS["test"], lw=1.8, ls="--",
                   label=f"MSE test final (media 25 corridas) = {m_ref:.5f}")
        ax.axhspan(m_ref - s_ref, m_ref + s_ref,
                   color=COLORS["test"], alpha=0.12, label="±1 std test final")

    final = m_train[-1]
    ax.set_title(f"{LABELS[name]}\nMSE train final = {final:.5f}", fontsize=10)
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    perceptrons = ["linear", "nonlinear"]

    cfg  = json.loads(CONFIG_PATH.read_text())
    n_seeds  = len(cfg["seeds"])
    n_folds  = cfg["k_folds"]
    n_total  = n_seeds * n_folds

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, name in zip(axes, perceptrons):
        sub = df[df["perceptron"] == name]
        plot_perceptron(ax, sub, name)

    fig.suptitle(
        "Curvas de convergencia — MSE train y test por época\n"
        f"Media ± 1 std sobre {n_seeds} seeds × {n_folds} folds = {n_total} corridas por modelo",
        fontsize=10,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Plot guardado: {args.output}")


if __name__ == "__main__":
    main()
