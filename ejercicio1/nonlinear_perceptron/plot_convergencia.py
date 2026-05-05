"""Plots de convergencia (MSE train vs test por epoca) para sweep LR.

Genera:
    convergencia/sweep_all.png         -- los 3 LRs juntos (mean +/- std)
    convergencia/sweep_all_zoom50.png  -- los 3 LRs juntos, primeras 50 epocas
    convergencia/lr_<X>.png            -- un plot por LR (train + test, mean +/- std)

Uso:
    python plot_convergencia.py [--out analisis_outputs]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LR_RUNS = {
    "0.01":   "output/sweep_lr_singleseed/lr_0_01",
    "0.001":  "output/sweep_lr_singleseed/lr_0_001",
    "0.0001": "output/sweep_lr_singleseed/lr_0_0001",
}

COLORS = {"0.01": "tab:blue", "0.001": "tab:orange", "0.0001": "tab:green"}


def load_history_stats(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "mse_history.csv")
    g = df.groupby("epoch")
    out = pd.DataFrame({
        "epoch": g["mse_train"].mean().index,
        "train_mean": g["mse_train"].mean().values,
        "train_std":  g["mse_train"].std().values,
        "test_mean":  g["mse_test"].mean().values,
        "test_std":   g["mse_test"].std().values,
    })
    return out


def plot_combined(out_path: Path, base_dir: Path, max_epoch: int | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for label, run_subdir in LR_RUNS.items():
        hist = load_history_stats(base_dir / run_subdir)
        if max_epoch is not None:
            hist = hist[hist["epoch"] <= max_epoch]
        color = COLORS[label]
        ax.plot(hist["epoch"], hist["train_mean"], color=color, linestyle="-",
                linewidth=1.8, label=f"lr={label} (train)")
        ax.fill_between(hist["epoch"],
                        hist["train_mean"] - hist["train_std"],
                        hist["train_mean"] + hist["train_std"],
                        color=color, alpha=0.18, linewidth=0)
        ax.plot(hist["epoch"], hist["test_mean"], color=color, linestyle="--",
                linewidth=1.5, label=f"lr={label} (test)")
        ax.fill_between(hist["epoch"],
                        hist["test_mean"] - hist["test_std"],
                        hist["test_mean"] + hist["test_std"],
                        color=color, alpha=0.10, linewidth=0)

    suffix = f" (zoom epocas 0-{max_epoch})" if max_epoch else ""
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE (mean +/- std entre 5 folds)")
    ax.set_title(f"Convergencia train vs test - sigmoid - sweep LR{suffix}")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_single_lr(out_path: Path, base_dir: Path, lr_label: str) -> None:
    hist = load_history_stats(base_dir / LR_RUNS[lr_label])
    color = COLORS[lr_label]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(hist["epoch"], hist["train_mean"], color=color, linestyle="-",
            linewidth=2.0, label="MSE train (mean)")
    ax.fill_between(hist["epoch"],
                    hist["train_mean"] - hist["train_std"],
                    hist["train_mean"] + hist["train_std"],
                    color=color, alpha=0.20, linewidth=0, label="train +/- std")
    ax.plot(hist["epoch"], hist["test_mean"], color=color, linestyle="--",
            linewidth=2.0, label="MSE test (mean)")
    ax.fill_between(hist["epoch"],
                    hist["test_mean"] - hist["test_std"],
                    hist["test_mean"] + hist["test_std"],
                    color=color, alpha=0.10, linewidth=0, label="test +/- std")

    final_train = hist["train_mean"].iloc[-1]
    final_test = hist["test_mean"].iloc[-1]
    ax.set_xlabel("Epoca")
    ax.set_ylabel("MSE (mean +/- std entre 5 folds)")
    ax.set_title(
        f"Convergencia - sigmoid - lr={lr_label}  "
        f"(final train={final_train:.5f}, test={final_test:.5f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=Path("analisis_outputs"), type=Path)
    parser.add_argument(
        "--base-dir", default=Path(__file__).parent, type=Path,
    )
    args = parser.parse_args()

    out_dir = args.out / "sweep_lr" / "singleseed"
    per_lr_dir = out_dir / "per_lr"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_lr_dir.mkdir(parents=True, exist_ok=True)

    plot_combined(out_dir / "convergencia.png", args.base_dir)
    plot_combined(out_dir / "convergencia_zoom50.png", args.base_dir, max_epoch=50)
    for lr_label in LR_RUNS:
        safe = lr_label.replace(".", "_")
        plot_single_lr(per_lr_dir / f"lr_{safe}.png", args.base_dir, lr_label)


if __name__ == "__main__":
    main()
