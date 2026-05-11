"""Plots dedicados a la interacción LR × OPT del stage 2 del cross_v1.

Genera:
  1. lr_opt_val_acc_4panels.png — 4 paneles (uno por arch) con val_acc vs LR,
     3 curvas por panel (sgd/momentum/adam). Para evaluar si la interacción
     LR×OPT es robusta a arch.
  2. lr_opt_heatmaps_4archs.png — 4 heatmaps (uno por arch) LR×OPT con
     val_acc como color. Vista de mapa para ranking por celda.

Outputs en ejercicio2_experimentacion/analisis/cross_v1/lr_opt/.
Copias a Notas/ejercicio 2/Segunda tanda de experimentos/Cross_LR_Opt_Arch/
con prefijo lr_opt_*.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CSV  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "stage2" / "stage2_summary.csv"
OUT  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "lr_opt"
NOTES = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Cross_LR_Opt_Arch"
OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

# --- estilo light (fondo blanco) ---
BG       = "#ffffff"
TEXT     = "#1a1a1a"
LABEL    = "#555555"
GRID     = "#cccccc"

ARCH_ORDER = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]
OPT_ORDER  = ["sgd", "momentum", "adam"]
LR_ORDER   = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LR_LABEL   = ["1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
OPT_LABEL  = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}
OPT_COLORS = {"sgd": "#58a6ff", "momentum": "#f78166", "adam": "#3fb950"}
OPT_MARKERS = {"sgd": "o", "momentum": "s", "adam": "^"}


def plot_4panels(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), facecolor=BG, sharey=True)
    for ax, arch in zip(axes, ARCH_ORDER):
        ax.set_facecolor(BG)
        sub = df[df["arch"] == arch]
        for opt in OPT_ORDER:
            s = sub[sub["opt"] == opt].sort_values("lr")
            ax.errorbar(s["lr"], s["val_acc_final_mean"],
                         yerr=s["val_acc_final_std"],
                         marker=OPT_MARKERS[opt], color=OPT_COLORS[opt],
                         label=OPT_LABEL[opt], linewidth=2,
                         markersize=8, capsize=4, capthick=1.2,
                         markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks(LR_ORDER); ax.set_xticklabels(LR_LABEL,
                                                       color=TEXT, fontsize=9)
        ax.set_xlabel("Learning rate", color=TEXT, fontsize=10)
        ax.set_title(arch.replace("arch_", "").upper(),
                      color=TEXT, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.25, color=LABEL, linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)
        ax.set_ylim(0.92, 0.965)

    axes[0].set_ylabel("val_acc (media ± std sobre 3 seeds × 5 folds)",
                        color=TEXT, fontsize=10)
    axes[0].legend(loc="lower right", facecolor=BG, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=10, framealpha=0.9)

    fig.suptitle("val_acc vs Learning Rate, por arquitectura y optimizer · stage 2 del cross_v1",
                  color=TEXT, fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
              "Para cada arquitectura, 3 curvas (una por optimizer) muestran cómo cambia val_acc al variar LR. "
              "Si las 4 sub-figuras se ven iguales → la interacción LR×OPT es robusta a la arquitectura.",
              color=LABEL, ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "lr_opt_val_acc_4panels.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    # Copy to notes
    fig.savefig(NOTES / "lr_opt_val_acc_4panels.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'lr_opt_val_acc_4panels.png'}")


def plot_4heatmaps(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(20, 5.5), facecolor=BG)
    vmin = df["val_acc_final_mean"].min()
    vmax = df["val_acc_final_mean"].max()
    cmap = plt.cm.viridis

    for ax, arch in zip(axes, ARCH_ORDER):
        ax.set_facecolor(BG)
        sub = df[df["arch"] == arch]
        mat = np.full((len(OPT_ORDER), len(LR_ORDER)), np.nan)
        for i, opt in enumerate(OPT_ORDER):
            for j, lr in enumerate(LR_ORDER):
                row = sub[(sub["opt"] == opt) &
                          (np.isclose(sub["lr"], lr))]
                if not row.empty:
                    mat[i, j] = row["val_acc_final_mean"].iloc[0]
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        # Annotate cells with val_acc
        for i in range(len(OPT_ORDER)):
            for j in range(len(LR_ORDER)):
                v = mat[i, j]
                if not np.isnan(v):
                    txt_color = "white" if v < (vmin + vmax)/2 + 0.005 else "black"
                    ax.text(j, i, f"{v:.3f}"[1:], ha="center", va="center",
                            color=txt_color, fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(LR_ORDER)))
        ax.set_xticklabels(LR_LABEL, color=TEXT, fontsize=9)
        ax.set_yticks(range(len(OPT_ORDER)))
        ax.set_yticklabels([OPT_LABEL[o] for o in OPT_ORDER],
                            color=TEXT, fontsize=10)
        ax.set_xlabel("Learning rate", color=TEXT, fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Optimizer", color=TEXT, fontsize=10)
        ax.set_title(arch.replace("arch_", "").upper(),
                      color=TEXT, fontsize=12, fontweight="bold", pad=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)

    # Colorbar compartido a la derecha
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02, aspect=28)
    cbar.set_label("val_acc (media seeds×folds)", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)
    cbar.outline.set_edgecolor(GRID)

    fig.suptitle("Heatmaps LR × OPT por arquitectura · stage 2 del cross_v1",
                  color=TEXT, fontsize=15, fontweight="bold", y=1.0)
    fig.text(0.5, -0.02,
              "Cada heatmap es la grilla LR × OPT para UNA arquitectura. "
              "Compará las 4 grillas para evaluar si el patrón LR×OPT depende de arch.",
              color=LABEL, ha="center", fontsize=10, style="italic")
    fig.savefig(OUT / "lr_opt_heatmaps_4archs.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "lr_opt_heatmaps_4archs.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'lr_opt_heatmaps_4archs.png'}")


def plot_val_loss_4panels(df: pd.DataFrame) -> None:
    """val_loss CE vs LR, 4 paneles (uno por arch), 3 curvas por opt.
    Capta la inestabilidad de Adam@LR alto que la accuracy oculta."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), facecolor=BG, sharey=True)
    for ax, arch in zip(axes, ARCH_ORDER):
        ax.set_facecolor(BG)
        sub = df[df["arch"] == arch]
        for opt in OPT_ORDER:
            s = sub[sub["opt"] == opt].sort_values("lr")
            ax.errorbar(s["lr"], s["val_loss_final_mean"],
                         yerr=s["val_loss_final_std"],
                         marker=OPT_MARKERS[opt], color=OPT_COLORS[opt],
                         label=OPT_LABEL[opt], linewidth=2,
                         markersize=8, capsize=4, capthick=1.2,
                         markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(LR_ORDER); ax.set_xticklabels(LR_LABEL,
                                                       color=TEXT, fontsize=9)
        ax.set_xlabel("Learning rate", color=TEXT, fontsize=10)
        ax.set_title(arch.replace("arch_", "").upper(),
                      color=TEXT, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.25, color=LABEL, linewidth=0.5, which="both")
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)

    axes[0].set_ylabel("val_loss CE (escala log)", color=TEXT, fontsize=10)
    axes[0].legend(loc="upper left", facecolor=BG, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=10, framealpha=0.9)

    fig.suptitle("val_loss CE vs Learning Rate, por arquitectura y optimizer · stage 2 del cross_v1",
                  color=TEXT, fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
              "val_loss = cross-entropy en validación (escala log). Capta el 'quiebre' de Adam@LR alto "
              "más dramáticamente que la accuracy: cuando un modelo se desestabiliza, asigna probabilidades "
              "erradas con alta confianza → CE explota.",
              color=LABEL, ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "lr_opt_val_loss_4panels.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "lr_opt_val_loss_4panels.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'lr_opt_val_loss_4panels.png'}")


def plot_overfit_gap_4panels(df: pd.DataFrame) -> None:
    """Gap = val_loss − train_loss vs LR, 4 paneles (uno por arch), 3 curvas por opt.
    Mide la magnitud del sobreajuste por celda."""
    df = df.copy()
    df["overfit_gap"] = df["val_loss_final_mean"] - df["train_loss_final_mean"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), facecolor=BG, sharey=True)
    for ax, arch in zip(axes, ARCH_ORDER):
        ax.set_facecolor(BG)
        sub = df[df["arch"] == arch]
        for opt in OPT_ORDER:
            s = sub[sub["opt"] == opt].sort_values("lr")
            ax.plot(s["lr"], s["overfit_gap"],
                     marker=OPT_MARKERS[opt], color=OPT_COLORS[opt],
                     label=OPT_LABEL[opt], linewidth=2,
                     markersize=8, markeredgecolor="black", markeredgewidth=0.6)
        ax.set_xscale("log")
        ax.set_xticks(LR_ORDER); ax.set_xticklabels(LR_LABEL,
                                                       color=TEXT, fontsize=9)
        ax.set_xlabel("Learning rate", color=TEXT, fontsize=10)
        ax.set_title(arch.replace("arch_", "").upper(),
                      color=TEXT, fontsize=12, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.25, color=LABEL, linewidth=0.5)
        ax.axhline(0, color="black", linewidth=0.6, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)

    axes[0].set_ylabel("Gap = val_loss CE − train_loss CE", color=TEXT, fontsize=10)
    axes[0].legend(loc="upper left", facecolor=BG, edgecolor=GRID,
                    labelcolor=TEXT, fontsize=10, framealpha=0.9)

    fig.suptitle("Sobreajuste (gap val_loss − train_loss) vs LR, por arquitectura y optimizer",
                  color=TEXT, fontsize=15, fontweight="bold", y=1.02)
    fig.text(0.5, -0.03,
              "Gap alto = el modelo memoriza train mejor de lo que generaliza a val. "
              "Útil para identificar dónde un futuro experimento de regularización (Pack C) "
              "podría mover la aguja.",
              color=LABEL, ha="center", fontsize=10, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "lr_opt_overfit_gap_4panels.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "lr_opt_overfit_gap_4panels.png", dpi=160,
                 facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'lr_opt_overfit_gap_4panels.png'}")


def main() -> None:
    df = pd.read_csv(CSV)
    plot_4panels(df)
    plot_4heatmaps(df)
    plot_val_loss_4panels(df)
    plot_overfit_gap_4panels(df)
    print(f"\nGenerados en:\n  {OUT}\n  {NOTES}")


if __name__ == "__main__":
    main()
