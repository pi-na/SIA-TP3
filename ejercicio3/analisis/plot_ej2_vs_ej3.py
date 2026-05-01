"""Bar plot comparando test_accuracy y test_macro_f1 entre Ej2 y Ej3."""
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
    parser.add_argument("--ej2-final-dir", required=True, type=Path)
    parser.add_argument("--ej3-final-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    m2 = pd.read_csv(args.ej2_final_dir / "test_metrics.csv").iloc[0]
    m3 = pd.read_csv(args.ej3_final_dir / "test_metrics.csv").iloc[0]

    metrics = ["test_accuracy", "test_macro_f1", "test_weighted_f1",
               "test_macro_precision", "test_macro_recall"]
    ej2_vals = [float(m2[k]) for k in metrics]
    ej3_vals = [float(m3[k]) for k in metrics]

    x = np.arange(len(metrics))
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, ej2_vals, width, label="Ej2 (digits.csv)")
    ax.bar(x + width/2, ej3_vals, width, label="Ej3 (+more_digits.csv)")
    ax.axhline(0.98, color="red", linestyle="--", linewidth=1, label="Target ≥98%")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("test_", "") for m in metrics], rotation=20)
    ax.set_ylabel("Score")
    ax.set_title("Comparación Ej2 vs Ej3 (test set = digits_test.csv)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
