"""Plots de val_acc para análisis de optimizador × LR y batch_size.

Plot 1: val_acc vs épocas — optimizer sweep (media sobre 25 corridas, 5 seeds × 5 folds)
         Un subplot por LR, una línea por optimizador.

Plot 2: val_acc final vs batch_size — Pre_LR_Batch experiment
         Una línea por (optimizer, LR), separadas por optimizador.

Uso:
    python ejercicio2_experimentacion/plot_val_acc_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
EJ2_EXP = ROOT / "ejercicio2_experimentacion"
OPT_DIR = EJ2_EXP / "analisis" / "optimizer"
NOTES_DIR = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Pre_LR_Batch_Opt"

LR_LABELS = {
    0.0001: "1e-4",
    0.0005: "5e-4",
    0.001:  "1e-3",
    0.005:  "5e-3",
    0.01:   "1e-2",
}

OPT_COLORS = {"sgd": "tab:blue", "momentum": "tab:orange", "adam": "tab:green"}
OPT_LABELS = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}

LR_COLORS = {
    0.0005: "tab:blue",
    0.001:  "tab:orange",
    0.005:  "tab:green",
}


def plot_val_acc_vs_epochs(history: pd.DataFrame, out_path: Path) -> None:
    """val_acc vs épocas por optimizador, un subplot por LR."""
    lrs  = sorted(history["lr"].unique())
    opts = ["sgd", "momentum", "adam"]

    n_cols = len(lrs)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5), sharey=True)

    for ax, lr in zip(axes, lrs):
        lr_label = LR_LABELS.get(lr, str(lr))
        for opt in opts:
            sub = history[(history["optimizer"] == opt) & (history["lr"] == lr)]
            if sub.empty:
                continue
            grouped = sub.groupby("epoch")["val_acc"]
            mean = grouped.mean()
            std  = grouped.std()
            color = OPT_COLORS[opt]

            ax.plot(mean.index, mean.values, label=OPT_LABELS[opt],
                    color=color, linewidth=1.8)
            ax.fill_between(mean.index,
                            mean.values - std.values,
                            mean.values + std.values,
                            alpha=0.15, color=color)

        ax.set_title(f"lr = {lr_label}", fontsize=10)
        ax.set_xlabel("Época")
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    axes[0].set_ylabel("Accuracy val (media ± std sobre 25 corridas)")
    axes[0].legend(fontsize=9)

    fig.suptitle(
        "val_acc por época — comparación de optimizadores\n"
        "arch_base [784→128→64→10] | media ± std sobre 25 corridas (5 seeds × 5 folds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def plot_val_acc_vs_batch(agg: pd.DataFrame, out_path: Path) -> None:
    """val_acc final vs batch_size, separado por optimizador."""
    opts  = ["sgd", "momentum", "adam"]
    lrs   = sorted(agg["lr"].unique())
    batches = sorted(agg["batch"].unique())

    fig, axes = plt.subplots(1, len(opts), figsize=(5 * len(opts), 5), sharey=True)

    for ax, opt in zip(axes, opts):
        for lr in lrs:
            sub = agg[(agg["opt"] == opt) & (agg["lr"] == lr)].sort_values("batch")
            if sub.empty:
                continue
            lr_label = LR_LABELS.get(lr, str(lr))
            color = LR_COLORS.get(lr, "gray")
            ax.errorbar(
                sub["batch"], sub["val_acc_final_mean"],
                yerr=sub["val_acc_final_std"],
                label=f"lr={lr_label}", color=color,
                marker="o", linewidth=1.8, capsize=4,
            )

        ax.set_title(OPT_LABELS[opt], fontsize=11)
        ax.set_xlabel("Batch size")
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    axes[0].set_ylabel("Accuracy val final (media ± std sobre 10 corridas)")

    fig.suptitle(
        "val_acc final vs batch_size por optimizador y LR\n"
        "arch_shallow | media ± std sobre 10 corridas (2 seeds × 5 folds)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out_path}")


def main() -> None:
    # Plot 1: val_acc vs épocas (optimizer sweep)
    history = pd.read_csv(OPT_DIR / "epoch_history.csv")
    plot_val_acc_vs_epochs(
        history,
        out_path=OPT_DIR / "val_acc_vs_epochs.png",
    )

    # Plot 2: val_acc vs batch_size (Pre_LR_Batch)
    agg = pd.read_csv(NOTES_DIR / "stage1_agg.csv")
    plot_val_acc_vs_batch(
        agg,
        out_path=NOTES_DIR / "val_acc_vs_batch.png",
    )


if __name__ == "__main__":
    main()
