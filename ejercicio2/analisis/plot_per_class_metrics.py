"""Bar chart de precision/recall/f1 por clase."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--title", default="Métricas por clase")
    parser.add_argument(
        "--summary-file",
        default="run_summary.csv",
        help="CSV con métricas (run_summary.csv para sweeps, test_metrics.csv para final_eval)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.run_dir / args.summary_file)
    if "fold" in df.columns:
        mean_row = df[df["fold"].astype(str) == "mean"]
        row = mean_row.iloc[0] if len(mean_row) > 0 else df.iloc[0]
    else:
        row = df.iloc[0]

    precisions = [float(row[f"precision_{c}"]) for c in range(args.num_classes)]
    recalls = [float(row[f"recall_{c}"]) for c in range(args.num_classes)]
    f1s = [float(row[f"f1_{c}"]) for c in range(args.num_classes)]

    x = np.arange(args.num_classes)
    width = 0.27
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, precisions, width, label="Precision")
    ax.bar(x, recalls, width, label="Recall")
    ax.bar(x + width, f1s, width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in range(args.num_classes)])
    ax.set_xlabel("Clase (dígito)")
    ax.set_ylabel("Score")
    ax.set_title(args.title)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
