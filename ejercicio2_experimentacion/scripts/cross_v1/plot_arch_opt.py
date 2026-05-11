"""Plot dedicado a la interaccion ARCH x OPT del stage 2 del cross_v1,
marginalizando sobre LR via best-over-LR (no mean-over-LR; ver discusion
en analisis.md / IMPORTANTE_CORRELACIONES.md sec. 7).

Para cada combo (arch, opt) toma la celda con val_acc mas alta entre los 5 LRs
disponibles del stage 2. La idea es ver, una sola figura, si alguna arquitectura
es "especialista" de un optimizer.

Output unico:
  - arch_opt_best_lr_heatmap.png  (heatmap 4 archs x 3 opts con val_acc
    como color y best_lr anotado en cada celda)

El CSV de respaldo es best_lr_per_opt_arch.csv (generado por plot_lr_arch.py).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CSV  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "stage2" / "stage2_summary.csv"
OUT  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "arch_opt"
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
ARCH_LABEL = {"arch_shallow": "shallow",
              "arch_base":    "base",
              "arch_wider":   "wider",
              "arch_deeper":  "deeper"}
OPT_LABEL  = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}


def lr_to_label(lr: float) -> str:
    # Formato consistente con el resto del repo
    mapping = {1e-4: "1e-4", 5e-4: "5e-4", 1e-3: "1e-3",
               5e-3: "5e-3", 1e-2: "1e-2"}
    for k, v in mapping.items():
        if np.isclose(lr, k):
            return v
    return f"{lr:.0e}"


def main() -> None:
    df = pd.read_csv(CSV)

    # Best-over-LR por (arch, opt)
    val_mat  = np.full((len(ARCH_ORDER), len(OPT_ORDER)), np.nan)
    std_mat  = np.full_like(val_mat, np.nan)
    lr_mat   = np.empty((len(ARCH_ORDER), len(OPT_ORDER)), dtype=object)
    for i, arch in enumerate(ARCH_ORDER):
        for j, opt in enumerate(OPT_ORDER):
            sub = df[(df["arch"] == arch) & (df["opt"] == opt)]
            if sub.empty:
                continue
            row = sub.loc[sub["val_acc_final_mean"].idxmax()]
            val_mat[i, j] = row["val_acc_final_mean"]
            std_mat[i, j] = row["val_acc_final_std"]
            lr_mat[i, j]  = lr_to_label(row["lr"])

    fig, ax = plt.subplots(figsize=(8.5, 6.0), facecolor=BG)
    ax.set_facecolor(BG)

    vmin = float(np.nanmin(val_mat))
    vmax = float(np.nanmax(val_mat))
    im = ax.imshow(val_mat, cmap=plt.cm.viridis, vmin=vmin, vmax=vmax, aspect="auto")

    # Best por columna (mejor arch para cada opt) y best por fila (mejor opt
    # para cada arch). Marcamos el global con un recuadro.
    best_per_col = np.nanargmax(val_mat, axis=0)
    best_per_row = np.nanargmax(val_mat, axis=1)
    global_i, global_j = np.unravel_index(np.nanargmax(val_mat), val_mat.shape)

    for i in range(len(ARCH_ORDER)):
        for j in range(len(OPT_ORDER)):
            v   = val_mat[i, j]
            s   = std_mat[i, j]
            lr  = lr_mat[i, j]
            if np.isnan(v):
                continue
            txt_color = "white" if v < (vmin + vmax) / 2 + 0.003 else "black"
            tag = ""
            if best_per_col[j] == i:
                tag += " ★"
            label = f"{v:.4f}\n± {s:.4f}\n@LR={lr}{tag}"
            ax.text(j, i, label, ha="center", va="center",
                    color=txt_color, fontsize=10, fontweight="bold",
                    linespacing=1.3)

    # Recuadro al maximo global
    rect = plt.Rectangle((global_j - 0.48, global_i - 0.48), 0.96, 0.96,
                          fill=False, edgecolor="#ff6b6b", linewidth=2.6)
    ax.add_patch(rect)

    ax.set_xticks(range(len(OPT_ORDER)))
    ax.set_xticklabels([OPT_LABEL[o] for o in OPT_ORDER],
                       color=TEXT, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(ARCH_ORDER)))
    ax.set_yticklabels([ARCH_LABEL[a] for a in ARCH_ORDER],
                       color=TEXT, fontsize=11, fontweight="bold")
    ax.set_xlabel("Optimizer", color=TEXT, fontsize=11)
    ax.set_ylabel("Arquitectura", color=TEXT, fontsize=11)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.tick_params(colors=LABEL, labelsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("val_acc best-over-LR (media sobre 15 corridas)",
                   color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)
    cbar.outline.set_edgecolor(GRID)

    fig.suptitle("ARCH × OPT (best-over-LR) · stage 2 del cross_v1",
                 color=TEXT, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, -0.02,
             "Cada celda = val_acc del mejor LR para esa combinacion (arch, opt). "
             "★ = mejor arch dentro de su columna (opt fijo).  "
             "Recuadro rojo = maximo global. val_acc reportada como media ± std sobre 3 seeds × 5 folds = 15 corridas.",
             color=LABEL, ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "arch_opt_best_lr_heatmap.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "arch_opt_best_lr_heatmap.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'arch_opt_best_lr_heatmap.png'}")
    print(f"saved {NOTES/'arch_opt_best_lr_heatmap.png'}")

    # Tabla auxiliar para verificacion en stdout
    print("\nMatriz val_acc best-over-LR (filas=arch, cols=opt):")
    print(pd.DataFrame(val_mat, index=[ARCH_LABEL[a] for a in ARCH_ORDER],
                                columns=[OPT_LABEL[o] for o in OPT_ORDER]))
    print("\nMejor arch por opt (★):")
    for j, opt in enumerate(OPT_ORDER):
        i = best_per_col[j]
        print(f"  {OPT_LABEL[opt]:10s} -> {ARCH_LABEL[ARCH_ORDER[i]]:8s} "
              f"({val_mat[i,j]:.4f} @LR={lr_mat[i,j]})")
    print(f"\nMaximo global: {ARCH_LABEL[ARCH_ORDER[global_i]]} + "
          f"{OPT_LABEL[OPT_ORDER[global_j]]} = {val_mat[global_i,global_j]:.4f} "
          f"@LR={lr_mat[global_i, global_j]}")


if __name__ == "__main__":
    main()
