"""Plots de convergencia para experimentos de aprendizaje (perceptrón lineal).

Lee output/aprendizaje_<ts>/epoch_history.csv y genera:
- mse_convergence.png:  MSE(epoch) por LR, banda ± std sobre seeds, eje Y log.
- r2_convergence.png:   R²(epoch) por LR, banda ± std sobre seeds, eje Y lineal.
- final_metrics.png:    bar chart con MSE final y R² final por LR (mean ± std).

Uso:
    python plot_aprendizaje.py [--run-dir output/aprendizaje_<ts>]
    (por default: el último output/aprendizaje_* en orden alfabético).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TITLE_MODEL = "Perceptrón lineal (Adaline)"


def _find_latest_run() -> Path:
    runs = sorted((HERE / "output").glob("aprendizaje_*"))
    if not runs:
        raise SystemExit("No hay output/aprendizaje_* — corré aprendizaje_sweep.py primero.")
    return runs[-1]


def _agg_band(history: pd.DataFrame, value_col: str
              ) -> dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Por LR: devuelve (epochs, mean_over_seeds, std_over_seeds)."""
    out = {}
    for lr in sorted(history["lr"].unique(), reverse=True):
        sub = history[history["lr"] == lr]
        pivot = sub.pivot(index="epoch", columns="seed", values=value_col)
        epochs = pivot.index.to_numpy()
        mean = pivot.mean(axis=1).to_numpy()
        std = pivot.std(axis=1, ddof=0).to_numpy()
        out[lr] = (epochs, mean, std)
    return out


def plot_mse(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    bands = _agg_band(history, "mse_train_fulldataset")
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(bands)))
    for color, (lr, (epochs, mean, std)) in zip(cmap, bands.items()):
        ax.plot(epochs, mean, lw=1.8, color=color, label=f"lr = {lr:g}")
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel("época")
    ax.set_ylabel("MSE_train (full dataset, sin split)")
    ax.set_title(f"{TITLE_MODEL} — convergencia de MSE\n"
                 f"media ± std sobre 3 seeds, training online")
    ax.grid(True, which="both", alpha=0.25, linestyle="--")
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_r2(history: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="white")
    ax.set_facecolor("white")
    bands = _agg_band(history, "r2_train_fulldataset")
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(bands)))
    for color, (lr, (epochs, mean, std)) in zip(cmap, bands.items()):
        ax.plot(epochs, mean, lw=1.8, color=color, label=f"lr = {lr:g}")
        ax.fill_between(epochs, mean - std, mean + std,
                        color=color, alpha=0.18, linewidth=0)
    ax.axhline(0, color="#888", lw=0.8, linestyle=":", label="baseline trivial (predecir media)")
    ax.set_xlabel("época")
    ax.set_ylabel("R²_train (full dataset, sin split)")
    ax.set_title(f"{TITLE_MODEL} — convergencia de R²\n"
                 f"media ± std sobre 3 seeds, training online")
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white")
    plt.close(fig)


def plot_final_bars(summary: pd.DataFrame, out_path: Path) -> None:
    """Bar chart de MSE final por LR. Error bar sólo si std>0. Eje Y lineal."""
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor="white")
    ax.set_facecolor("white")
    ax.grid(True, alpha=0.25, linestyle="--", axis="y")

    summary = summary.sort_values("lr", ascending=False).reset_index(drop=True)
    labels = [f"{lr:g}" for lr in summary["lr"]]
    x = np.arange(len(labels))

    mse_mean = summary["mse_final_mean_seeds"].to_numpy()
    mse_std  = summary["mse_final_std_seeds"].to_numpy()
    # NaN oculta la error bar; sólo se dibuja donde std>0.
    mse_err = np.where(mse_std > 0, mse_std, np.nan)

    ax.bar(x, mse_mean, yerr=mse_err, capsize=4,
           color="#3b78c0", edgecolor="#1f4e79")
    for xi, v in zip(x, mse_mean):
        ax.text(xi, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("MSE_train final (media sobre 3 seeds)")
    ax.set_title(f"{TITLE_MODEL} — MSE final por LR  "
                 f"(error bar sólo si std>0)")
    ax.set_ylim(0, max(mse_mean) * 1.18)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Default: el último output/aprendizaje_*")
    args = parser.parse_args()

    run_dir = args.run_dir or _find_latest_run()
    print(f"Reading: {run_dir}")
    history = pd.read_csv(run_dir / "epoch_history.csv")
    summary = pd.read_csv(run_dir / "summary.csv")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plot_mse(history, plots_dir / "mse_convergence.png")
    plot_r2(history, plots_dir / "r2_convergence.png")
    plot_final_bars(summary, plots_dir / "final_metrics.png")
    print(f"Plots guardados en: {plots_dir}")


if __name__ == "__main__":
    main()
