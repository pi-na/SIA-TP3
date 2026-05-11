"""Plots dedicados a la interacción LR x ARCH del stage 2 del cross_v1.

Es la misma data del stage 2 que `plot_lr_opt.py`, rotada: en vez de
"4 paneles por arch, 3 curvas por opt", hacemos "3 paneles por opt,
4 curvas por arch". Responde: para cada optimizer, el LR optimo es el
mismo en las 4 archs o se desplaza?

Genera:
  1. lr_arch_val_acc_3panels.png - 3 paneles (uno por opt), val_acc vs LR,
     4 curvas por panel (una por arch). Errorbars = std sobre 15 corridas
     (3 seeds x 5 folds).
  2. lr_arch_heatmaps_3opts.png - 3 heatmaps (uno por opt), filas=ARCH,
     columnas=LR, color=val_acc. Vista de mapa para shifts.
  3. best_lr_per_opt_arch.csv - tabla cuantitativa: por cada (opt, arch),
     el LR ganador y su val_acc media (sobre 15 corridas) +- std.

Outputs en ejercicio2_experimentacion/analisis/cross_v1/lr_arch/.
Copias a Notas/ejercicio 2/Segunda tanda de experimentos/Cross_LR_Opt_Arch/
con prefijo lr_arch_*.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CSV  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "stage2" / "stage2_summary.csv"
OUT  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "lr_arch"
NOTES = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Cross_LR_Opt_Arch"
OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

# --- estilo light (fondo blanco) ---
BG    = "#ffffff"
TEXT  = "#1a1a1a"
LABEL = "#555555"
GRID  = "#cccccc"

ARCH_ORDER = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]
OPT_ORDER  = ["sgd", "momentum", "adam"]
LR_ORDER   = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LR_LABEL   = ["1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
OPT_LABEL  = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}
ARCH_LABEL = {
    "arch_shallow": "shallow",
    "arch_base":    "base",
    "arch_wider":   "wider",
    "arch_deeper":  "deeper",
}
ARCH_COLORS = {
    "arch_shallow": "#58a6ff",  # azul
    "arch_base":    "#f78166",  # naranja
    "arch_wider":   "#3fb950",  # verde
    "arch_deeper":  "#a371f7",  # violeta
}
ARCH_MARKERS = {
    "arch_shallow": "o",
    "arch_base":    "s",
    "arch_wider":   "^",
    "arch_deeper":  "D",
}


def plot_3panels_val_acc(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), facecolor=BG, sharey=True)
    for ax, opt in zip(axes, OPT_ORDER):
        ax.set_facecolor(BG)
        sub = df[df["opt"] == opt]
        for arch in ARCH_ORDER:
            s = sub[sub["arch"] == arch].sort_values("lr")
            ax.errorbar(s["lr"], s["val_acc_final_mean"],
                        yerr=s["val_acc_final_std"],
                        marker=ARCH_MARKERS[arch], color=ARCH_COLORS[arch],
                        label=ARCH_LABEL[arch], linewidth=2,
                        markersize=8, capsize=4, capthick=1.2,
                        markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks(LR_ORDER); ax.set_xticklabels(LR_LABEL,
                                                     color=TEXT, fontsize=9)
        ax.set_xlabel("Learning rate", color=TEXT, fontsize=10)
        ax.set_title(OPT_LABEL[opt], color=TEXT, fontsize=13,
                     fontweight="bold", pad=8)
        ax.grid(True, alpha=0.25, color=LABEL, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)
        ax.set_ylim(0.92, 0.965)

    axes[0].set_ylabel("val_acc (media ± std sobre 3 seeds × 5 folds = 15 corridas)",
                       color=TEXT, fontsize=10)
    axes[0].legend(loc="lower right", facecolor=BG, edgecolor=GRID,
                   labelcolor=TEXT, fontsize=10, framealpha=0.9, title="arch",
                   title_fontsize=10)

    fig.suptitle("val_acc vs Learning Rate, por optimizer y arquitectura · stage 2 del cross_v1",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
             "Para cada optimizer, 4 curvas (una por arquitectura) muestran cómo cambia val_acc al variar LR. "
             "Si dentro de un panel las 4 curvas pican en la misma columna → el LR óptimo de ese optimizer NO depende de arch.",
             color=LABEL, ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "lr_arch_val_acc_3panels.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "lr_arch_val_acc_3panels.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'lr_arch_val_acc_3panels.png'}")


def plot_heatmaps_3opts(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=BG)
    vmin = df["val_acc_final_mean"].min()
    vmax = df["val_acc_final_mean"].max()
    cmap = plt.cm.viridis

    for ax, opt in zip(axes, OPT_ORDER):
        ax.set_facecolor(BG)
        sub = df[df["opt"] == opt]
        mat = np.full((len(ARCH_ORDER), len(LR_ORDER)), np.nan)
        for i, arch in enumerate(ARCH_ORDER):
            for j, lr in enumerate(LR_ORDER):
                row = sub[(sub["arch"] == arch) &
                          (np.isclose(sub["lr"], lr))]
                if not row.empty:
                    mat[i, j] = row["val_acc_final_mean"].iloc[0]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        # Marcar el mejor LR por fila (arch) con asterisco
        for i in range(len(ARCH_ORDER)):
            if np.all(np.isnan(mat[i])):
                continue
            best_j = int(np.nanargmax(mat[i]))
            for j in range(len(LR_ORDER)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                txt_color = "white" if v < (vmin + vmax)/2 + 0.005 else "black"
                marker = "*" if j == best_j else ""
                ax.text(j, i, f"{v:.3f}"[1:] + marker, ha="center", va="center",
                        color=txt_color, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_LABEL, color=TEXT, fontsize=9)
        ax.set_yticks(range(len(ARCH_ORDER)))
        ax.set_yticklabels([ARCH_LABEL[a] for a in ARCH_ORDER],
                           color=TEXT, fontsize=10)
        ax.set_xlabel("Learning rate", color=TEXT, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Arquitectura", color=TEXT, fontsize=10)
        ax.set_title(OPT_LABEL[opt], color=TEXT, fontsize=13,
                     fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)

    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02, aspect=24)
    cbar.set_label("val_acc (media sobre 15 corridas)", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)
    cbar.outline.set_edgecolor(GRID)

    fig.suptitle("Heatmaps ARCH × LR por optimizer · stage 2 del cross_v1",
                 color=TEXT, fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.04,
             "Cada heatmap es la grilla ARCH × LR para UN optimizer. El asterisco marca el mejor LR de cada fila (arch). "
             "Si los asteriscos de un panel están en la misma columna → no hay shift del LR óptimo entre archs para ese optimizer.",
             color=LABEL, ha="center", fontsize=10, style="italic")
    fig.savefig(OUT / "lr_arch_heatmaps_3opts.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "lr_arch_heatmaps_3opts.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'lr_arch_heatmaps_3opts.png'}")


def best_lr_table(df: pd.DataFrame) -> pd.DataFrame:
    """Para cada (opt, arch), encuentra el LR con maxima val_acc media y reporta
    val_acc +- std y val_loss CE en ese LR. Promedios sobre 3 seeds x 5 folds = 15 corridas."""
    rows = []
    for opt in OPT_ORDER:
        for arch in ARCH_ORDER:
            sub = df[(df["opt"] == opt) & (df["arch"] == arch)]
            if sub.empty:
                continue
            best = sub.loc[sub["val_acc_final_mean"].idxmax()]
            rows.append({
                "opt": opt,
                "arch": arch,
                "best_lr": best["lr"],
                "val_acc_mean_seeds_folds": best["val_acc_final_mean"],
                "val_acc_std_seeds_folds": best["val_acc_final_std"],
                "macro_f1_mean_seeds_folds": best["macro_f1_mean"],
                "val_loss_CE_mean_seeds_folds": best["val_loss_final_mean"],
                "best_epoch_mean": best["best_epoch_mean"],
            })
    return pd.DataFrame(rows)


def main() -> None:
    df = pd.read_csv(CSV)
    plot_3panels_val_acc(df)
    plot_heatmaps_3opts(df)
    tab = best_lr_table(df)
    csv_path = OUT / "best_lr_per_opt_arch.csv"
    tab.to_csv(csv_path, index=False)
    print(f"saved {csv_path}")
    print("\nBest LR por (opt, arch):")
    print(tab.to_string(index=False))
    print(f"\nGenerados en:\n  {OUT}\n  {NOTES}")


if __name__ == "__main__":
    main()
