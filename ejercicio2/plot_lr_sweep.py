"""Genera plots del sweep de learning rates — Ejercicio 2.

Lee analisis/lr/epoch_history.csv y analisis/lr/summary.csv.
Produce:
  - convergence_train.png : curvas de train_loss por época, una línea por LR, un subplot por arquitectura
  - convergence_val.png   : curvas de val_loss por época
  - convergence_gap.png   : curvas de (val_loss - train_loss) por época (brecha de sobreajuste)
  - final_metrics.png     : barras de val_acc y macro_f1 por LR, agrupadas por arquitectura

Cada curva de época es la media ± std sobre las 25 corridas (5 seeds × 5 folds).

Uso:
    python ejercicio2/plot_lr_sweep.py [--data-dir PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
EJ2     = ROOT / "ejercicio2"
DEFAULT = EJ2 / "analisis" / "lr"

LR_LABELS = {
    0.0001: "1e-4",
    0.0005: "5e-4",
    0.001:  "1e-3",
    0.005:  "5e-3",
    0.01:   "1e-2",
}

ARCH_ORDER = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]
ARCH_TITLES = {
    "arch_shallow": "shallow [784→128→10]",
    "arch_base":    "base [784→128→64→10]",
    "arch_wider":   "wider [784→256→128→10]",
    "arch_deeper":  "deeper [784→128→64→32→10]",
}

COLORS = plt.cm.tab10.colors


def load(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    history = pd.read_csv(data_dir / "epoch_history.csv")
    summary = pd.read_csv(data_dir / "summary.csv")
    return history, summary


def plot_convergence(history: pd.DataFrame, col: str, ylabel: str, title_prefix: str,
                     out_path: Path) -> None:
    """Curvas de `col` por época. Un subplot por arquitectura, una línea por LR."""
    archs = [a for a in ARCH_ORDER if a in history["arch"].unique()]
    lrs   = sorted(history["lr"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharey=False)
    axes = axes.flatten()

    for ax_idx, arch in enumerate(archs):
        ax = axes[ax_idx]
        sub_arch = history[history["arch"] == arch]

        for lr_idx, lr in enumerate(lrs):
            sub = sub_arch[sub_arch["lr"] == lr]
            grouped = sub.groupby("epoch")[col]
            mean = grouped.mean()
            std  = grouped.std()
            label = LR_LABELS.get(lr, str(lr))
            color = COLORS[lr_idx % len(COLORS)]

            ax.plot(mean.index, mean.values, label=f"lr={label}", color=color, linewidth=1.5)
            ax.fill_between(mean.index,
                            mean.values - std.values,
                            mean.values + std.values,
                            alpha=0.15, color=color)

        ax.set_title(ARCH_TITLES.get(arch, arch), fontsize=10)
        ax.set_xlabel("Época")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    for ax_idx in range(len(archs), len(axes)):
        axes[ax_idx].set_visible(False)

    fig.suptitle(f"{title_prefix}\n(media ± std sobre 25 corridas — 5 seeds × 5 folds)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot_final_metrics(summary: pd.DataFrame, out_path: Path) -> None:
    """Barras de val_acc y macro_f1 por LR, agrupadas por arquitectura."""
    archs = [a for a in ARCH_ORDER if a in summary["arch"].unique()]
    lrs   = sorted(summary["lr"].unique())
    n_lrs = len(lrs)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    metrics = [
        ("val_acc_final_mean", "val_acc_final_std", "Accuracy val (media ± std sobre 25 corridas)"),
        ("macro_f1_mean",      "macro_f1_std",      "F1 macro val (media ± std sobre 25 corridas)"),
    ]

    for ax, (mean_col, std_col, ylabel) in zip(axes, metrics):
        x = np.arange(len(archs))
        width = 0.8 / n_lrs

        for lr_idx, lr in enumerate(lrs):
            sub = summary[summary["lr"] == lr]
            means, stds = [], []
            for arch in archs:
                row = sub[sub["arch"] == arch]
                means.append(row[mean_col].values[0] if len(row) else np.nan)
                stds.append(row[std_col].values[0]  if len(row) else np.nan)

            offset = (lr_idx - n_lrs / 2 + 0.5) * width
            label = f"lr={LR_LABELS.get(lr, str(lr))}"
            ax.bar(x + offset, means, width, yerr=stds, label=label,
                   color=COLORS[lr_idx % len(COLORS)], capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_TITLES.get(a, a).split(" ")[0] for a in archs])
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Métricas finales por LR y arquitectura\n(media ± std sobre 25 corridas — 5 seeds × 5 folds)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT)
    args = parser.parse_args()

    history, summary = load(args.data_dir)

    # Curvas de convergencia
    plot_convergence(
        history, col="train_loss",
        ylabel="Cross-entropy (train)",
        title_prefix="Convergencia train loss por LR y arquitectura",
        out_path=args.data_dir / "convergence_train.png",
    )
    plot_convergence(
        history, col="val_loss",
        ylabel="Cross-entropy (val)",
        title_prefix="Convergencia val loss por LR y arquitectura",
        out_path=args.data_dir / "convergence_val.png",
    )

    # Brecha de sobreajuste
    history = history.copy()
    history["gap"] = history["val_loss"] - history["train_loss"]
    plot_convergence(
        history, col="gap",
        ylabel="val_loss − train_loss",
        title_prefix="Brecha de sobreajuste por LR y arquitectura",
        out_path=args.data_dir / "convergence_gap.png",
    )

    # Métricas finales en barras
    plot_final_metrics(summary, out_path=args.data_dir / "final_metrics.png")


if __name__ == "__main__":
    main()
