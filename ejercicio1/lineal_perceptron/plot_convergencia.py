"""Plots de convergencia (MSE train por epoca) para sweep LR del perceptron lineal.

El lineal solo loggea mse_train (no mse_test). Genera:
    analisis_outputs/sweep_lr_convergencia.png         -- 3 LRs juntos (mean +/- std)
    analisis_outputs/sweep_lr_convergencia_zoom.png    -- zoom primeras 300 epocas

Uso:
    python plot_convergencia.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

LR_RUNS = {
    "0.001":  "output/sweep_lr_singleseed/lr_0_001",
    "0.0001": "output/sweep_lr_singleseed/lr_0_0001",
    "1e-05":  "output/sweep_lr_singleseed/lr_1e-05",
}
COLORS = {"0.001": "tab:red", "0.0001": "tab:blue", "1e-05": "tab:green"}


def load_history_stats(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "mse_history.csv")
    g = df.groupby("epoch")["mse_train"]
    return pd.DataFrame({
        "epoch": g.mean().index,
        "mean": g.mean().values,
        "std":  g.std().values,
    })


def plot(out_path: Path, base_dir: Path, max_epoch: int | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax_idx, max_e in enumerate([300, None] if max_epoch is None else [max_epoch, max_epoch]):
        ax = axes[ax_idx]
        for label, run_subdir in LR_RUNS.items():
            hist = load_history_stats(base_dir / run_subdir)
            if max_e is not None:
                hist = hist[hist["epoch"] <= max_e]
            color = COLORS[label]
            ax.plot(hist["epoch"], hist["mean"], color=color, linewidth=1.8, label=f"lr={label}")
            ax.fill_between(hist["epoch"],
                            hist["mean"] - hist["std"],
                            hist["mean"] + hist["std"],
                            color=color, alpha=0.20, linewidth=0)
        title = "Convergencia (zoom: primeras 300 epocas)" if max_e == 300 else "Convergencia (todas las epocas)"
        ax.set_xlabel("Epoca")
        ax.set_ylabel("MSE train (mean +/- std entre 5 folds)")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main():
    base_dir = Path(__file__).parent
    out_dir = base_dir / "analisis_outputs" / "sweep_lr" / "singleseed"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot(out_dir / "convergencia.png", base_dir)


if __name__ == "__main__":
    main()
