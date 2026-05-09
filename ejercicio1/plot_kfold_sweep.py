"""Plot del K-fold sweep — lee raw.csv y summary.csv, genera kfold_sweep.png.

Uso:
    python plot_kfold_sweep.py --perceptron linear
    python plot_kfold_sweep.py --perceptron nonlinear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

OUT_ROOTS = {
    "linear":    ROOT / "lineal_perceptron"    / "analisis_outputs" / "kfold_sweep",
    "nonlinear": ROOT / "nonlinear_perceptron" / "analisis_outputs" / "kfold_sweep",
}

COLORS = {"linear": "#cce5ff", "nonlinear": "#ffe0cc"}
TITLES = {
    "linear":    "K-fold sweep — perceptrón lineal",
    "nonlinear": "K-fold sweep — perceptrón no-lineal",
}
METRICS = [
    ("mse_test",  "MSE test"),
    ("f1",        "F1"),
    ("precision", "Precision"),
    ("recall",    "Recall"),
]


def plot(perceptron: str) -> None:
    out_root = OUT_ROOTS[perceptron]
    raw     = pd.read_csv(out_root / "raw.csv")
    summary = pd.read_csv(out_root / "summary.csv")
    k_values = sorted(raw["k"].unique())

    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 5))
    color = COLORS[perceptron]

    for ax, (col, label) in zip(axes, METRICS):
        data = [raw[raw["k"] == k][col].values for k in k_values]
        bp = ax.boxplot(data, tick_labels=[str(k) for k in k_values],
                        widths=0.5, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        means = [summary[summary["k"] == k][f"{col}_mean"].values[0] for k in k_values]
        ax.scatter(range(1, len(k_values) + 1), means,
                   color="tab:red", zorder=5, s=50, label="media")

        ax.set_title(label)
        ax.set_xlabel("K (número de folds)")
        ax.grid(True, alpha=0.3, axis="y")
        if col == "mse_test":
            ax.legend(fontsize=8)

    thr_col = [c for c in summary.columns if "threshold" in c]
    cfg_path = out_root / "config.json"
    import json
    cfg = json.loads(cfg_path.read_text())
    subtitle = (f"seed={cfg['seed']}, epochs={cfg['epochs']}, "
                f"thr={cfg['threshold']}\n"
                "Boxplot: distribución entre folds. Punto rojo: media.")
    fig.suptitle(f"{TITLES[perceptron]}\n{subtitle}", fontsize=10)
    fig.tight_layout()

    out = out_root / "kfold_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Plot guardado: {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perceptron", choices=["linear", "nonlinear"], required=True)
    args = parser.parse_args()
    plot(args.perceptron)


if __name__ == "__main__":
    main()
