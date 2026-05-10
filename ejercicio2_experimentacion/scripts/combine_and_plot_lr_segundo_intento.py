"""Combina los datos rescatados (fase 1, crash) con los de fase 2 y genera
plots + tablas para el análisis del LR_segundo_intento.

Outputs (en analisis/lr_segundo_intento/ y Notas/.../LR_segundo_intento/):
  - raw_combined.csv, summary_combined.csv, epoch_history_combined.csv
  - convergence_train.png, convergence_val.png
  - final_metrics.png  (val_acc, F1, val_loss, train_loss vs LR)
  - val_acc_vs_epoch.png  (heat-line de val_acc por época, una curva por LR)
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent.parent
RESCUED = ROOT / "ejercicio2_experimentacion" / "output" / "lr_segundo_intento"
PHASE2  = RESCUED / "phase2"
OUT_AN  = ROOT / "ejercicio2_experimentacion" / "analisis" / "lr_segundo_intento"
NOTES   = ROOT / "Notas" / "ejercicio 2" / "Experimentos y analisis" / "LR_segundo_intento"
OUT_AN.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

LR_COLORS = {1e-4: "#1f77b4", 5e-4: "#ff7f0e", 1e-3: "#2ca02c", 5e-3: "#d62728", 1e-2: "#9467bd"}
LR_LABEL  = {1e-4: "1e-4", 5e-4: "5e-4", 1e-3: "1e-3", 5e-3: "5e-3", 1e-2: "1e-2"}
EPOCHS_BY_LR = {1e-4: 500, 5e-4: 500, 1e-3: 250, 5e-3: 150, 1e-2: 150}

METRIC_COLS = [
    "total_epochs", "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]

def lr_key(x: float) -> float:
    for k in LR_COLORS:
        if abs(x - k) < 1e-12:
            return k
    return x

def main() -> None:
    raw_a = pd.read_csv(RESCUED / "raw.csv")
    his_a = pd.read_csv(RESCUED / "epoch_history.csv")
    raw_b = pd.read_csv(PHASE2 / "raw.csv")
    his_b = pd.read_csv(PHASE2 / "epoch_history.csv")

    raw = pd.concat([raw_a, raw_b], ignore_index=True)
    his = pd.concat([his_a, his_b], ignore_index=True)

    raw.to_csv(OUT_AN / "raw_combined.csv", index=False)
    his.to_csv(OUT_AN / "epoch_history_combined.csv", index=False)

    # Summary combinado
    rows = []
    arch_name = raw["arch"].iloc[0]
    for lr in sorted(raw["lr"].unique()):
        sub = raw[(raw["arch"] == arch_name) & (raw["lr"] == lr)]
        row = {"arch": arch_name, "lr": lr,
               "epochs_planned": EPOCHS_BY_LR.get(lr_key(lr)),
               "n_seeds": sub["seed"].nunique(),
               "n_corridas": len(sub)}
        for col in METRIC_COLS:
            if col in sub.columns:
                row[f"{col}_mean_seedsfolds"] = sub[col].mean()
                row[f"{col}_std_seedsfolds"] = sub[col].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_AN / "summary_combined.csv", index=False)
    summary.to_csv(NOTES / "summary_combined.csv", index=False)

    lrs_sorted = sorted(raw["lr"].unique())

    # ---------- convergence (train + val) ----------
    for split, label in [("train", "Train CE"), ("val", "Val CE")]:
        col = f"{split}_loss"
        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for lr in lrs_sorted:
            sub = his[his["lr"] == lr]
            grp = sub.groupby("epoch")[col].agg(["mean", "std"])
            color = LR_COLORS.get(lr_key(lr))
            ax.plot(grp.index, grp["mean"], label=f"lr={LR_LABEL.get(lr_key(lr), lr)}",
                    color=color, linewidth=1.6)
            ax.fill_between(grp.index, grp["mean"]-grp["std"], grp["mean"]+grp["std"],
                            alpha=0.15, color=color)
        ax.set_xlabel("Época")
        ax.set_ylabel(f"{label} (media ± std seeds×folds)")
        ax.set_title(f"{label} por época — arch_shallow, SGD, sin ES, sin reg")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(title="Learning rate")
        fig.tight_layout()
        fig.savefig(OUT_AN / f"convergence_{split}.png", dpi=140)
        fig.savefig(NOTES / f"convergence_{split}.png", dpi=140)
        plt.close(fig)

    # ---------- val_acc vs epoch ----------
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for lr in lrs_sorted:
        sub = his[his["lr"] == lr]
        grp = sub.groupby("epoch")["val_acc"].agg(["mean", "std"])
        color = LR_COLORS.get(lr_key(lr))
        ax.plot(grp.index, grp["mean"], label=f"lr={LR_LABEL.get(lr_key(lr), lr)}",
                color=color, linewidth=1.6)
        ax.fill_between(grp.index, grp["mean"]-grp["std"], grp["mean"]+grp["std"],
                        alpha=0.15, color=color)
    ax.set_xlabel("Época")
    ax.set_ylabel("Val accuracy (media ± std seeds×folds)")
    ax.set_title("Val accuracy por época — arch_shallow, SGD, sin ES")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Learning rate")
    fig.tight_layout()
    fig.savefig(OUT_AN / "val_acc_vs_epoch.png", dpi=140)
    fig.savefig(NOTES / "val_acc_vs_epoch.png", dpi=140)
    plt.close(fig)

    # ---------- final metrics bars ----------
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8))
    metrics = [
        ("val_acc_final", "Val accuracy"),
        ("macro_f1",      "F1 macro"),
        ("val_loss_final", "Val CE loss"),
        ("train_loss_final", "Train CE loss"),
    ]
    x = np.arange(len(lrs_sorted))
    xticklabels = [f"{LR_LABEL.get(lr_key(v), str(v))}\n({EPOCHS_BY_LR.get(lr_key(v))} ep)" for v in lrs_sorted]
    for ax, (m, title) in zip(axes.flatten(), metrics):
        means = [summary[summary["lr"] == lr][f"{m}_mean_seedsfolds"].iloc[0] for lr in lrs_sorted]
        stds  = [summary[summary["lr"] == lr][f"{m}_std_seedsfolds"].iloc[0] for lr in lrs_sorted]
        colors = [LR_COLORS.get(lr_key(lr), "#888") for lr in lrs_sorted]
        ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_xticks(x); ax.set_xticklabels(xticklabels)
        ax.set_xlabel("Learning rate (épocas)")
        ax.set_title(f"{title} (mean ± std)")
        ax.grid(True, axis="y", alpha=0.3)
        if "loss" in m:
            ax.set_yscale("log")
    fig.suptitle("Métricas finales — arch_shallow, SGD básico, sin ES, sin regularización",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_AN / "final_metrics.png", dpi=140)
    fig.savefig(NOTES / "final_metrics.png", dpi=140)
    plt.close(fig)

    print("=== Summary combinado ===")
    print(summary[["lr", "epochs_planned", "n_seeds",
                   "val_acc_final_mean_seedsfolds", "val_acc_final_std_seedsfolds",
                   "macro_f1_mean_seedsfolds",
                   "val_loss_final_mean_seedsfolds",
                   "train_loss_final_mean_seedsfolds"]].to_string(index=False))


if __name__ == "__main__":
    main()
