"""Convergencia de la configuracion optima del Ej2 (cross_v1 stage 2 main).

Carga los epoch_history.csv de las 3 seeds (42, 7, 13) x 5 folds = 15 corridas
de la celda shallow + Adam@1e-3 + bs64 y promedia por epoca.

Output:
  - optimal_convergence.png  (2 paneles: loss / accuracy, con bandas mean ± std)
  - optimal_convergence_table.csv  (epoch, mean/std de las 4 series)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
STAGE_DIR = ROOT / "ejercicio2_experimentacion" / "output" / "cross_v1" / "stage2"
OUT = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "optimal_config"
NOTES = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Cross_LR_Opt_Arch"
OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

BG="#ffffff"; TEXT="#1a1a1a"; LABEL="#555555"; GRID="#cccccc"
COLOR_TRAIN = "#1c4e8f"  # azul oscuro
COLOR_VAL   = "#c92a2a"  # rojo
COLOR_BAND_TRAIN = "#a8c2e0"
COLOR_BAND_VAL   = "#f1aaa4"

SEEDS = [42, 7, 13]
CELL  = "stage2_arch_shallow_adam_lr1e-3_bs64"


def load_all_epoch_histories() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        # buscar el subdir con timestamp
        seed_dir = STAGE_DIR / f"{CELL}_seed{seed}"
        candidates = list(seed_dir.glob(f"{CELL}_seed{seed}_*/epoch_history.csv"))
        if not candidates:
            raise FileNotFoundError(f"No epoch_history para seed={seed} en {seed_dir}")
        df = pd.read_csv(candidates[0])
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def aggregate_by_epoch(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada epoca, promedia sobre las (seed, fold) que llegaron a esa epoca."""
    cols = ["train_loss", "val_loss", "train_acc", "val_acc"]
    agg = df.groupby("epoch")[cols].agg(["mean", "std", "count"]).reset_index()
    # aplanar columnas multinivel
    agg.columns = ["epoch"] + [f"{c}_{stat}" for c in cols for stat in ["mean","std","count"]]
    return agg


def best_epoch_stats(df: pd.DataFrame) -> dict:
    """argmin de val_loss por (seed, fold); mean/std/min/max sobre las 15 series."""
    best = df.groupby(["seed","fold"])["val_loss"].idxmin()
    epochs = df.loc[best, "epoch"].to_numpy()
    return {"mean": float(np.mean(epochs)),
            "std":  float(np.std(epochs)),
            "min":  int(np.min(epochs)),
            "max":  int(np.max(epochs))}


def stop_epoch_stats(df: pd.DataFrame) -> dict:
    """Ultima epoca registrada por (seed, fold) = donde corto el training
    (sea por early stopping o max_epochs)."""
    stops = df.groupby(["seed","fold"])["epoch"].max().to_numpy()
    return {"mean": float(np.mean(stops)),
            "std":  float(np.std(stops)),
            "min":  int(np.min(stops)),
            "max":  int(np.max(stops))}


def plot_convergence(agg: pd.DataFrame, best_stats: dict) -> None:
    best_ep = best_stats["mean"]
    best_std = best_stats["std"]
    # cortar a las épocas donde aún hay >= 8 series vivas (>=50% del total)
    keep = agg[agg["train_loss_count"] >= 8].copy()
    epochs = keep["epoch"].to_numpy()

    fig, (axL, axA) = plt.subplots(1, 2, figsize=(14, 5.3), facecolor=BG)

    # --- Loss panel ---
    axL.set_facecolor(BG)
    tl_m = keep["train_loss_mean"].to_numpy(); tl_s = keep["train_loss_std"].to_numpy()
    vl_m = keep["val_loss_mean"].to_numpy();   vl_s = keep["val_loss_std"].to_numpy()
    axL.fill_between(epochs, tl_m-tl_s, tl_m+tl_s, color=COLOR_BAND_TRAIN, alpha=0.4, linewidth=0)
    axL.fill_between(epochs, vl_m-vl_s, vl_m+vl_s, color=COLOR_BAND_VAL,   alpha=0.4, linewidth=0)
    axL.plot(epochs, tl_m, color=COLOR_TRAIN, linewidth=2, label="train_loss CE", marker="o", markersize=3.5)
    axL.plot(epochs, vl_m, color=COLOR_VAL,   linewidth=2, label="val_loss CE",   marker="s", markersize=3.5)
    axL.axvline(best_ep, color="#555555", linestyle="--", linewidth=1.2, alpha=0.7)
    axL.text(best_ep + 0.4, axL.get_ylim()[1]*0.93,
              f"best_epoch promedio = {best_ep:.1f}\n(std = {best_std:.1f} sobre 15 corridas)",
              color="#333", fontsize=9.5, linespacing=1.2)
    axL.set_xlabel("epoch", color=TEXT, fontsize=11)
    axL.set_ylabel("cross-entropy", color=TEXT, fontsize=11)
    axL.set_title("Loss de entrenamiento vs validación", color=TEXT, fontsize=12, fontweight="bold")
    axL.grid(True, alpha=0.25, color=LABEL, linewidth=0.5)
    axL.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    for spine in axL.spines.values(): spine.set_color(GRID)
    axL.tick_params(colors=LABEL, labelsize=9)

    # --- Accuracy panel ---
    axA.set_facecolor(BG)
    ta_m = keep["train_acc_mean"].to_numpy(); ta_s = keep["train_acc_std"].to_numpy()
    va_m = keep["val_acc_mean"].to_numpy();   va_s = keep["val_acc_std"].to_numpy()
    axA.fill_between(epochs, ta_m-ta_s, ta_m+ta_s, color=COLOR_BAND_TRAIN, alpha=0.4, linewidth=0)
    axA.fill_between(epochs, va_m-va_s, va_m+va_s, color=COLOR_BAND_VAL,   alpha=0.4, linewidth=0)
    axA.plot(epochs, ta_m, color=COLOR_TRAIN, linewidth=2, label="train_acc", marker="o", markersize=3.5)
    axA.plot(epochs, va_m, color=COLOR_VAL,   linewidth=2, label="val_acc",   marker="s", markersize=3.5)
    axA.axvline(best_ep, color="#555555", linestyle="--", linewidth=1.2, alpha=0.7)
    axA.set_xlabel("epoch", color=TEXT, fontsize=11)
    axA.set_ylabel("accuracy", color=TEXT, fontsize=11)
    axA.set_title("Accuracy de entrenamiento vs validación", color=TEXT, fontsize=12, fontweight="bold")
    axA.grid(True, alpha=0.25, color=LABEL, linewidth=0.5)
    axA.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)
    for spine in axA.spines.values(): spine.set_color(GRID)
    axA.tick_params(colors=LABEL, labelsize=9)

    fig.suptitle("Convergencia de la configuración óptima · shallow + Adam@1e-3 + batch=64",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.05,
             "Curvas: media sobre 15 corridas (3 seeds × 5 folds del CV interno) por época. "
             "Bandas: ± 1 std a nivel de época. "
             "Solo se grafican las épocas donde ≥ 8 de las 15 series seguían entrenando "
             "(las demás cortó early stopping con patience=20). "
             "Línea vertical: best_epoch promedio (época con val_loss mínimo, promediada sobre las 15 corridas).",
             color=LABEL, ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "optimal_convergence.png", dpi=160, facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "optimal_convergence.png", dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'optimal_convergence.png'}")


def main():
    df = load_all_epoch_histories()
    print(f"Loaded {len(df)} rows from {df['seed'].nunique()} seeds x {df['fold'].nunique()} folds")
    print(f"Epochs reached per (seed,fold): "
          f"min={df.groupby(['seed','fold']).size().min()}, "
          f"max={df.groupby(['seed','fold']).size().max()}, "
          f"mean={df.groupby(['seed','fold']).size().mean():.1f}")
    agg = aggregate_by_epoch(df)
    agg.to_csv(OUT / "optimal_convergence_table.csv", index=False)
    print(f"saved {OUT/'optimal_convergence_table.csv'}")
    best_stats = best_epoch_stats(df)
    stop_stats = stop_epoch_stats(df)
    print(f"best_epoch: mean={best_stats['mean']:.2f} std={best_stats['std']:.2f} "
          f"min={best_stats['min']} max={best_stats['max']}")
    print(f"stop_epoch (ultima registrada): mean={stop_stats['mean']:.2f} "
          f"std={stop_stats['std']:.2f} min={stop_stats['min']} max={stop_stats['max']}")
    print(f"max_epochs config = 40, patience = 20")
    print(f"Cuantas llegaron al max_epochs? "
          f"{(df.groupby(['seed','fold'])['epoch'].max() >= 39).sum()} / 15")
    plot_convergence(agg, best_stats)


if __name__ == "__main__":
    main()
