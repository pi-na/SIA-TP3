"""Comparar val_acc/val_loss/macro_f1 entre múltiples corridas (sweep)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def extract_label(run_dir: Path, label_by: str) -> str:
    cfg = json.loads((run_dir / "config.json").read_text())
    if label_by == "lr":
        return f"lr={cfg['training']['optimizer']['lr']}"
    if label_by == "arch":
        return f"arch={cfg['architecture']['layer_sizes']}"
    if label_by == "optimizer":
        return f"opt={cfg['training']['optimizer']['name']}"
    if label_by == "init":
        return f"init={cfg['architecture']['initializer']}"
    if label_by == "batch":
        return f"bs={cfg['training']['batch_size']}"
    return cfg["model_name"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dirs", nargs="+", required=True, type=Path)
    parser.add_argument("--metric", default="val_acc_final")
    parser.add_argument("--label-by", default="lr",
                        choices=["lr", "arch", "optimizer", "init", "batch", "model_name"])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--title", default="Sweep")
    args = parser.parse_args()

    rows = []
    for run in args.run_dirs:
        df = pd.read_csv(run / "run_summary.csv")
        mean_row = df[df["fold"].astype(str) == "mean"]
        std_row = df[df["fold"].astype(str) == "std"]
        if len(mean_row) > 0:
            metric_mean = float(mean_row.iloc[0][args.metric])
            metric_std = float(std_row.iloc[0][args.metric]) if len(std_row) > 0 else 0.0
        else:
            metric_mean = float(df.iloc[0][args.metric])
            metric_std = 0.0
        rows.append({"label": extract_label(run, args.label_by),
                     "mean": metric_mean, "std": metric_std})

    rows = sorted(rows, key=lambda r: r["mean"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = [r["label"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    ax.bar(labels, means, yerr=stds, capsize=5)
    ax.set_ylabel(args.metric)
    ax.set_title(args.title)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
