"""Plot: relación LR × batch_size sobre val_loss y val_acc.

Muestra dos efectos opuestos según el LR:
- Adam@5e-3 (LR demasiado alto para batch chico): batch↑ → CE↓, val_acc↑
- Adam@5e-4 (LR óptimo para batch chico): batch↑ → CE↑, val_acc↓

Uso:
    python ejercicio2_experimentacion/plot_lr_batch_relationship.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Pre_LR_Batch_Opt"

LR_STYLES = {
    0.0005: {"color": "tab:blue",  "label": "lr=5e-4", "marker": "o"},
    0.001:  {"color": "tab:orange","label": "lr=1e-3", "marker": "s"},
    0.005:  {"color": "tab:green", "label": "lr=5e-3", "marker": "^"},
}


def main() -> None:
    agg = pd.read_csv(NOTES_DIR / "stage1_agg.csv")
    adam = agg[agg["opt"] == "adam"].sort_values(["lr", "batch"])
    batches = sorted(adam["batch"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for metric, ylabel, ax in [
        ("val_loss_final_mean",  "Cross-entropy val (media sobre 10 corridas)", axes[0]),
        ("val_acc_final_mean",   "Accuracy val (media sobre 10 corridas)",      axes[1]),
    ]:
        std_col = metric.replace("_mean", "_std")
        for lr, style in LR_STYLES.items():
            sub = adam[adam["lr"] == lr]
            ax.errorbar(
                sub["batch"], sub[metric],
                yerr=sub[std_col],
                label=style["label"],
                color=style["color"],
                marker=style["marker"],
                linewidth=2, markersize=7, capsize=5,
            )
        ax.set_xlabel("Batch size")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(batches)
        ax.set_xticklabels([str(b) for b in batches])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    axes[0].set_title("CE val vs batch_size")
    axes[1].set_title("val_acc vs batch_size")

    fig.suptitle(
        "Efecto del batch_size según el LR — Adam, arch_shallow\n"
        "media ± std sobre 10 corridas (2 seeds × 5 folds)",
        fontsize=12,
    )
    fig.tight_layout()

    out = NOTES_DIR / "lr_batch_relationship.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Guardado: {out}")


if __name__ == "__main__":
    main()
