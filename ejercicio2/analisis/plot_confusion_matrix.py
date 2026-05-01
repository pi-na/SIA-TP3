"""Heatmap de matriz de confusión (10x10 para dígitos)."""
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
    parser.add_argument("--fold", default="all", help="all=sumar, o índice de fold")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Matriz de confusión")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument(
        "--cm-file",
        default="confusion_matrix.csv",
        help="Nombre del CSV de la matriz (e.g. test_confusion_matrix.csv para final_eval)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.run_dir / args.cm_file)
    if "fold" in df.columns and args.fold != "all":
        df = df[df["fold"] == int(args.fold)]
    pivot = df.pivot_table(index="true_label", columns="pred_label",
                            values="count", aggfunc="sum", fill_value=0)
    pivot = pivot.reindex(index=range(args.num_classes), columns=range(args.num_classes), fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(args.num_classes))
    ax.set_yticks(range(args.num_classes))
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(args.title)
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            ax.text(j, i, int(pivot.values[i, j]), ha="center", va="center",
                    color="white" if pivot.values[i, j] > pivot.values.max() / 2 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
