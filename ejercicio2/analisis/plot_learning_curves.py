"""Plot train/val loss y accuracy por época, una curva por fold + mean."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Curvas de aprendizaje")
    args = parser.parse_args()

    df = pd.read_csv(args.run_dir / "epoch_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for fold in df["fold"].unique():
        sub = df[df["fold"] == fold]
        axes[0].plot(sub["epoch"], sub["train_loss"], alpha=0.4, color="C0")
        axes[0].plot(sub["epoch"], sub["val_loss"], alpha=0.4, color="C1")
        axes[1].plot(sub["epoch"], sub["train_acc"], alpha=0.4, color="C0")
        axes[1].plot(sub["epoch"], sub["val_acc"], alpha=0.4, color="C1")

    mean = df.groupby("epoch")[["train_loss", "val_loss", "train_acc", "val_acc"]].mean().reset_index()
    axes[0].plot(mean["epoch"], mean["train_loss"], color="C0", linewidth=2.5, label="train (mean)")
    axes[0].plot(mean["epoch"], mean["val_loss"], color="C1", linewidth=2.5, label="val (mean)")
    axes[1].plot(mean["epoch"], mean["train_acc"], color="C0", linewidth=2.5, label="train (mean)")
    axes[1].plot(mean["epoch"], mean["val_acc"], color="C1", linewidth=2.5, label="val (mean)")

    axes[0].set_xlabel("Época"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set_xlabel("Época"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.suptitle(args.title)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
