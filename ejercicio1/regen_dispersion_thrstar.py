"""Regenera `dispersion.png` evaluando Acc/Prec/Rec/F1 al `thr*` de cada LR.

El boxplot original (en `multiseed_runner.py:plot_dispersion`) usaba las columnas
de `raw.csv`, que están a thr=0.5 — caso saturado/engañoso. Las tablas del
`analisis.md` ya están a thr*, este script alinea el plot con esas tablas.

Fuentes:
  - raw.csv                  -> mse_test, wnorm (no dependen del threshold)
  - threshold_summary.csv    -> thr* por lr
  - threshold_sweep_raw.csv  -> Acc/Prec/Rec/F1 por (lr, seed, fold, threshold)

No toca analisis.md (tiene ediciones manuales).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DIRS = {
    "linear":    ROOT / "lineal_perceptron"   / "analisis_outputs" / "sweep_lr" / "multiseed",
    "nonlinear": ROOT / "nonlinear_perceptron" / "analisis_outputs" / "sweep_lr" / "multiseed",
}


def build_at_thrstar(d: Path, lr_filter: list[float] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    raw = pd.read_csv(d / "raw.csv")
    thr_sum = pd.read_csv(d / "threshold_summary.csv")
    thr_raw = pd.read_csv(d / "threshold_sweep_raw.csv")

    raw["lr_f"] = raw["lr"].astype(float)
    thr_sum["lr_f"] = thr_sum["lr"].astype(float)
    thr_raw["lr_f"] = thr_raw["lr"].astype(float)

    if lr_filter is not None:
        keep = lambda x: any(np.isclose(x, lr, atol=1e-12) for lr in lr_filter)
        raw = raw[raw["lr_f"].map(keep)].copy()
        thr_sum = thr_sum[thr_sum["lr_f"].map(keep)].copy()
        thr_raw = thr_raw[thr_raw["lr_f"].map(keep)].copy()
        if raw.empty or thr_sum.empty:
            available = sorted(set(pd.read_csv(d / "raw.csv")["lr"].astype(float)))
            raise ValueError(f"Ningún LR del filtro {lr_filter} existe en {d}. Disponibles: {available}")

    thr_map = dict(zip(thr_sum["lr_f"], thr_sum["thr_star"]))

    # Pick rows in thr_raw matching thr* (within tolerance) per lr
    pieces = []
    for lr_f, thr in thr_map.items():
        sub = thr_raw[(thr_raw["lr_f"] == lr_f) &
                      (np.isclose(thr_raw["threshold"], thr, atol=1e-6))]
        pieces.append(sub)
    thr_at = pd.concat(pieces, ignore_index=True)

    # Merge per-fold MSE + wnorm from raw with per-fold metrics at thr*
    keys = ["lr_f", "seed", "fold"]
    base = raw[keys + ["mse_test", "wnorm"]]
    cls = thr_at[keys + ["accuracy", "precision", "recall", "f1"]]
    df = base.merge(cls, on=keys, how="inner")
    # Re-attach lr label (string) preserving raw order/values
    label_map = dict(zip(raw["lr_f"], raw["lr"].astype(str)))
    df["lr"] = df["lr_f"].map(label_map)

    per_seed = (df.groupby(["lr", "seed"])
                  [["mse_test", "wnorm", "accuracy", "precision", "recall", "f1"]]
                  .mean()
                  .reset_index())
    return df, per_seed, {label_map[k]: v for k, v in thr_map.items()}


def plot(df: pd.DataFrame, per_seed: pd.DataFrame, thr_map: dict[str, float],
         out_path: Path, title: str) -> None:
    metrics = [
        ("mse_test",  "MSE test (no depende del threshold)"),
        ("accuracy",  "Accuracy @ thr*"),
        ("precision", "Precision @ thr*"),
        ("recall",    "Recall @ thr*"),
        ("f1",        "F1 @ thr*"),
        ("wnorm",     "‖w‖ (L2)"),
    ]
    lrs = sorted(df["lr"].unique(), key=lambda s: float(s))
    tick_labels = [f"{lr}\nthr*={thr_map[lr]:.2f}" for lr in lrs]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, (col, label) in zip(axes, metrics):
        data = [df[df["lr"] == lr][col].values for lr in lrs]
        bp = ax.boxplot(data, tick_labels=tick_labels, widths=0.5, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cce5ff")
            patch.set_alpha(0.6)
        for i, lr in enumerate(lrs, start=1):
            vals = per_seed[per_seed["lr"] == lr][col].values
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                       color="tab:red", s=30, zorder=3,
                       label="mean por seed (sobre folds)" if i == 1 else None)
        ax.set_title(label)
        ax.set_xlabel("learning rate")
        ax.grid(True, alpha=0.3, axis="y")
        if col == "mse_test":
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{title}\nBoxplot: distribución sobre 5 seeds × 5 folds (n=25). "
        "Acc/Prec/Rec/F1 evaluados al thr* de cada LR (mismo criterio que las tablas). "
        "Puntos rojos: media por seed sobre folds (n=5)."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--perceptron", choices=["linear", "nonlinear", "both"], default="both")
    p.add_argument("--lrs", nargs="+", type=float, default=None,
                   help="LRs a incluir (ej: --lrs 0.0001 0.001). Default: todos.")
    args = p.parse_args()

    targets = ["linear", "nonlinear"] if args.perceptron == "both" else [args.perceptron]
    for name in targets:
        d = DIRS[name]
        print(f"=== {name} ===")
        df, per_seed, thr_map = build_at_thrstar(d, lr_filter=args.lrs)
        print(f"  thr* por lr: {thr_map}")
        print(f"  filas (lr, seed, fold) tras merge: {len(df)}")
        plot(df, per_seed, thr_map,
             d / "dispersion.png",
             f"Sweep LR multi-seed - {name}")


if __name__ == "__main__":
    main()
