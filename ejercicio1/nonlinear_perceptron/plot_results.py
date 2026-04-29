"""Genera los 3 plots del mejor modelo no-lineal a partir de CSVs existentes.

Uso:
    python plot_results.py --run-dir output/lr_001_nonlinear_<timestamp> --out analisis_outputs
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_threshold_sweep(sweep_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(sweep_df["threshold"], sweep_df["precision"], label="Precision", linewidth=1.8)
    ax.plot(sweep_df["threshold"], sweep_df["recall"],    label="Recall",    linewidth=1.8)
    ax.plot(sweep_df["threshold"], sweep_df["f1"],        label="F1",        linewidth=2.0)

    best_idx = sweep_df["f1"].idxmax()
    best = sweep_df.loc[best_idx]
    ax.axvline(best["threshold"], color="gray", linestyle="--", linewidth=1.2,
               label=f"Best th={best['threshold']:.2f} (F1={best['f1']:.3f})")

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold sweep — Perceptrón no-lineal (lr=0.001)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "threshold_sweep.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_roc(roc_df: pd.DataFrame, auc: float, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(roc_df["fpr"], roc_df["tpr"], linewidth=2.0,
            label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Curva ROC — Perceptrón no-lineal (lr=0.001)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "roc_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_pr(preds_df: pd.DataFrame, out_dir: Path) -> None:
    thresholds = np.arange(0.0, 1.001, 0.005)
    precisions, recalls = [], []
    y_true = preds_df["flagged_fraud"].to_numpy()
    scores  = preds_df["score"].to_numpy()
    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 1.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        precisions.append(prec)
        recalls.append(rec)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(recalls, precisions, linewidth=2.0)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Curva Precision-Recall — Perceptrón no-lineal (lr=0.001)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "pr_curve.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", default=Path("analisis_outputs"), type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    sweep_df = pd.read_csv(args.run_dir / "threshold_sweep.csv")
    roc_df   = pd.read_csv(args.run_dir / "roc_curve.csv")
    preds_df = pd.read_csv(args.run_dir / "predictions.csv")

    auc = float(np.trapz(roc_df["tpr"], roc_df["fpr"]))
    # trapz computes a negative value when fpr is sorted descending — fix sign
    auc = abs(auc)

    print(f"AUC-ROC: {auc:.4f}")

    plot_threshold_sweep(sweep_df, args.out)
    plot_roc(roc_df, auc, args.out)
    plot_pr(preds_df, args.out)

    print("Done.")


if __name__ == "__main__":
    main()
