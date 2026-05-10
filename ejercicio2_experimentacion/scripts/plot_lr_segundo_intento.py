"""Plots del segundo sweep LR (arch_shallow, 500 épocas, SGD básico, sin ES).

Genera 3 plots en el directorio destino (default: analisis/lr_segundo_intento/):
  - convergence_train.png  (train CE vs época, una curva por LR, media sobre 25)
  - convergence_val.png    (val CE vs época, una curva por LR)
  - final_metrics.png      (acc, F1, val_loss, train_loss vs LR — barras)

Uso:
    python ejercicio2_experimentacion/scripts/plot_lr_segundo_intento.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR  = ROOT / "ejercicio2_experimentacion" / "output" / "lr_segundo_intento"
OUT_DIR  = ROOT / "ejercicio2_experimentacion" / "analisis" / "lr_segundo_intento"
NOTES_DIR = ROOT / "Notas" / "ejercicio 2" / "Experimentos y analisis" / "LR_segundo_intento"

OUT_DIR.mkdir(parents=True, exist_ok=True)
NOTES_DIR.mkdir(parents=True, exist_ok=True)

LR_COLORS = {
    1e-4: "#1f77b4",
    5e-4: "#ff7f0e",
    1e-3: "#2ca02c",
    5e-3: "#d62728",
    1e-2: "#9467bd",
}
LR_LABEL = {1e-4: "1e-4", 5e-4: "5e-4", 1e-3: "1e-3", 5e-3: "5e-3", 1e-2: "1e-2"}


def lr_key(x: float) -> float:
    for k in LR_COLORS:
        if abs(x - k) < 1e-12:
            return k
    return x


def main() -> None:
    raw = pd.read_csv(RAW_DIR / "raw.csv")
    history = pd.read_csv(RAW_DIR / "epoch_history.csv")
    summary = pd.read_csv(RAW_DIR / "summary.csv")

    lrs_sorted = sorted(raw["lr"].unique())

    # ---------- convergence (train + val) ----------
    for split, label in [("train", "Train CE"), ("val", "Val CE")]:
        col = f"{split}_loss"
        if col not in history.columns:
            col = f"{split}_loss_epoch"
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for lr in lrs_sorted:
            sub = history[history["lr"] == lr]
            # promedio sobre seeds×folds por época
            grp = sub.groupby("epoch")[col].agg(["mean", "std"])
            color = LR_COLORS.get(lr_key(lr), None)
            ax.plot(grp.index, grp["mean"], label=f"lr={LR_LABEL.get(lr_key(lr), lr)}",
                    color=color, linewidth=1.6)
            ax.fill_between(grp.index, grp["mean"] - grp["std"], grp["mean"] + grp["std"],
                            alpha=0.15, color=color)
        ax.set_xlabel("Época")
        ax.set_ylabel(f"{label} (media ± std sobre 5 seeds × 5 folds)")
        ax.set_title(f"{label} por época — arch_shallow, SGD, 500 épocas, sin ES")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Learning rate")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"convergence_{split}.png", dpi=140)
        fig.savefig(NOTES_DIR / f"convergence_{split}.png", dpi=140)
        plt.close(fig)

    # ---------- final metrics ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [
        ("val_acc_final", "Val accuracy (mean ± std, 25 corridas)"),
        ("macro_f1",      "F1 macro (mean ± std, 25 corridas)"),
        ("val_loss_final", "Val CE loss (mean ± std, 25 corridas)"),
        ("train_loss_final", "Train CE loss (mean ± std, 25 corridas)"),
    ]
    x = np.arange(len(lrs_sorted))
    xticklabels = [LR_LABEL.get(lr_key(v), str(v)) for v in lrs_sorted]
    for ax, (m, title) in zip(axes.flatten(), metrics):
        means = [summary[(summary["lr"] == lr)][f"{m}_mean_seedsfolds"].iloc[0] for lr in lrs_sorted]
        stds  = [summary[(summary["lr"] == lr)][f"{m}_std_seedsfolds"].iloc[0] for lr in lrs_sorted]
        colors = [LR_COLORS.get(lr_key(lr), "#888") for lr in lrs_sorted]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Learning rate")
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        if "loss" in m:
            ax.set_yscale("log")
    fig.suptitle("Métricas finales (época 500) — arch_shallow, SGD básico, sin early stopping",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "final_metrics.png", dpi=140)
    fig.savefig(NOTES_DIR / "final_metrics.png", dpi=140)
    plt.close(fig)

    print(f"Plots guardados en:\n  {OUT_DIR}\n  {NOTES_DIR}")


if __name__ == "__main__":
    main()
