"""Genera plots del sweep de optimizadores — Ejercicio 2.

Combina los datos de SGD (analisis/lr/epoch_history.csv, arch_base)
con los de Momentum y Adam (analisis/optimizer/epoch_history.csv).

Produce un plot por LR con 3 líneas (SGD, Momentum, Adam),
mostrando train loss y val loss por época.

Uso:
    python ejercicio2/plot_optimizer_sweep.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT    = Path(__file__).resolve().parent.parent
EJ2     = ROOT / "ejercicio2"
LR_DIR  = EJ2 / "analisis" / "lr"
OPT_DIR = EJ2 / "analisis" / "optimizer"

LR_LABELS = {
    0.0001: "1e-4",
    0.0005: "5e-4",
    0.001:  "1e-3",
    0.005:  "5e-3",
    0.01:   "1e-2",
}

OPT_COLORS = {"sgd": "tab:blue", "momentum": "tab:orange", "adam": "tab:green"}
OPT_LABELS = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}


def load_combined() -> pd.DataFrame:
    # SGD: filtrar solo arch_base
    sgd = pd.read_csv(LR_DIR / "epoch_history.csv")
    sgd = sgd[sgd["arch"] == "arch_base"].copy()
    sgd["optimizer"] = "sgd"
    sgd = sgd.drop(columns=["arch"], errors="ignore")

    # Momentum y Adam
    opt = pd.read_csv(OPT_DIR / "epoch_history.csv")

    combined = pd.concat([sgd, opt], ignore_index=True)
    return combined


def plot_convergence(history: pd.DataFrame, out_dir: Path) -> None:
    lrs  = sorted(history["lr"].unique())
    opts = ["sgd", "momentum", "adam"]

    for lr in lrs:
        lr_label = LR_LABELS.get(lr, str(lr))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        for col, ax, ylabel in [
            ("train_loss", axes[0], "Cross-entropy (train)"),
            ("val_loss",   axes[1], "Cross-entropy (val)"),
        ]:
            for opt in opts:
                sub = history[(history["optimizer"] == opt) & (history["lr"] == lr)]
                if sub.empty:
                    continue
                grouped = sub.groupby("epoch")[col]
                mean = grouped.mean()
                std  = grouped.std()
                color = OPT_COLORS[opt]
                label = OPT_LABELS[opt]

                ax.plot(mean.index, mean.values, label=label, color=color, linewidth=1.8)
                ax.fill_between(mean.index,
                                mean.values - std.values,
                                mean.values + std.values,
                                alpha=0.15, color=color)

            ax.set_xlabel("Época")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

        fig.suptitle(
            f"Convergencia por optimizador — lr={lr_label}\n"
            f"arch_base [784→128→64→10] | media ± std sobre 25 corridas (5 seeds × 5 folds)",
            fontsize=11,
        )
        fig.tight_layout()
        out_path = out_dir / f"convergence_lr{lr_label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Guardado: {out_path}")


def main() -> None:
    history = load_combined()
    plot_convergence(history, OPT_DIR)


if __name__ == "__main__":
    main()
