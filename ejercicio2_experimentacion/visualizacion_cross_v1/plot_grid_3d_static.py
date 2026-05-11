"""Visualización del grid 3D del stage 2 del cross-experiment cross_v1.

Estrategia: hibridar 3D + 2D para captar la complejidad sin caer en una sola
representación que oculte cosas:
  - Panel principal: cube 3D con 60 esferas, una por celda, colorgrado por val_acc.
    Top-3 celdas etiquetadas. Ejes anotados con los valores reales.
  - Panel inferior izquierdo: 3 heatmaps faceted (un panel por optimizer), rows=arch, cols=LR.
    Es la "proyección" del cubo cuando se aplana el eje opt.
  - Panel inferior derecho: ranking horizontal de top-12 cells.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "analisis" / "cross_v1" / "stage2" / "stage2_summary.csv"
OUT = HERE / "grid_3d_static.png"

BG = "#0d1117"
TEXT = "#e6edf3"
LABEL = "#8b949e"
GRID = "#30363d"

# Orden de ejes
ARCH_ORDER = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]
OPT_ORDER  = ["sgd", "momentum", "adam"]
LR_ORDER   = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LR_LABEL   = ["1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
ARCH_LABEL = [a.replace("arch_", "") for a in ARCH_ORDER]
OPT_LABEL  = ["SGD", "Momentum", "Adam"]


def main() -> None:
    df = pd.read_csv(CSV)
    # index maps
    arch_idx = {a: i for i, a in enumerate(ARCH_ORDER)}
    opt_idx  = {o: i for i, o in enumerate(OPT_ORDER)}
    lr_idx   = {round(l, 6): i for i, l in enumerate(LR_ORDER)}
    df["xi"] = df["lr"].round(6).map(lr_idx)
    df["yi"] = df["opt"].map(opt_idx)
    df["zi"] = df["arch"].map(arch_idx)

    vmin = df["val_acc_final_mean"].min()
    vmax = df["val_acc_final_mean"].max()
    cmap = plt.cm.viridis

    fig = plt.figure(figsize=(18, 12), facecolor=BG)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.5, 1.0], width_ratios=[1.6, 1.0, 1.0],
                           hspace=0.18, wspace=0.25,
                           left=0.04, right=0.97, top=0.89, bottom=0.06)

    # ========== Panel principal: 3D cube ==========
    ax = fig.add_subplot(gs[0, :2], projection="3d", facecolor=BG)
    # axis colors
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_edgecolor(GRID)
        axis.pane.set_alpha(0.2)
        axis.label.set_color(TEXT)
        axis._axinfo['grid']['color'] = GRID
        axis._axinfo['grid']['linewidth'] = 0.4

    sizes = 80 + 1400 * (df["val_acc_final_mean"] - vmin) / (vmax - vmin)
    sc = ax.scatter(df["xi"], df["yi"], df["zi"], c=df["val_acc_final_mean"],
                     cmap=cmap, s=sizes, vmin=vmin, vmax=vmax,
                     edgecolors="white", linewidths=0.5, alpha=0.95)

    # Wireframe of the cube for visual structure
    edges = []
    nx, ny, nz = len(LR_ORDER), len(OPT_ORDER), len(ARCH_ORDER)
    # Lines along z (arch) for each (lr, opt)
    for xi in range(nx):
        for yi in range(ny):
            edges.append([(xi, yi, 0), (xi, yi, nz - 1)])
    # Lines along y (opt) for each (lr, arch)
    for xi in range(nx):
        for zi in range(nz):
            edges.append([(xi, 0, zi), (xi, ny - 1, zi)])
    # Lines along x (lr) for each (opt, arch)
    for yi in range(ny):
        for zi in range(nz):
            edges.append([(0, yi, zi), (nx - 1, yi, zi)])
    lc = Line3DCollection(edges, colors=GRID, linewidths=0.35, alpha=0.45)
    ax.add_collection3d(lc)

    # Top-3 highlight (anillo amarillo en el cubo, sin texto adentro)
    top = df.nlargest(3, "val_acc_final_mean")
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        ax.scatter([r["xi"]], [r["yi"]], [r["zi"]], s=sizes[r.name]*1.6,
                    facecolors="none", edgecolors="#f9e64f", linewidths=2.5, zorder=20)

    # Axes ticks
    ax.set_xticks(range(nx)); ax.set_xticklabels(LR_LABEL, color=TEXT, fontsize=9)
    ax.set_yticks(range(ny)); ax.set_yticklabels(OPT_LABEL, color=TEXT, fontsize=9)
    ax.set_zticks(range(nz)); ax.set_zticklabels(ARCH_LABEL, color=TEXT, fontsize=9)
    ax.set_xlabel("Learning rate", color=TEXT, fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Optimizer", color=TEXT, fontsize=11, fontweight="bold", labelpad=10)
    ax.set_zlabel("Arquitectura", color=TEXT, fontsize=11, fontweight="bold", labelpad=10)
    ax.tick_params(colors=TEXT)
    ax.view_init(elev=18, azim=-58)

    # Colorbar
    # Columna derecha dividida en 3 zonas verticales: colorbar / top-3 / cómo-leer.
    gs_right = gs[0, 2].subgridspec(3, 1, height_ratios=[1.2, 1.4, 1.0], hspace=0.35)

    # --- Colorbar (zona superior) ---
    cax_cb = fig.add_subplot(gs_right[0, 0])
    cax_cb.set_facecolor(BG); cax_cb.axis("off")
    cb = fig.colorbar(sc, ax=cax_cb, fraction=0.55, pad=0.02, aspect=14, location="left")
    cb.set_label("val_acc (media seeds×folds)", color=TEXT, fontsize=10)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=8.5)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=TEXT)
    cb.outline.set_edgecolor(GRID)

    # --- Top-3 legend (zona media) ---
    cax_top3 = fig.add_subplot(gs_right[1, 0])
    cax_top3.set_facecolor(BG); cax_top3.axis("off")
    top3_lines = ["TOP-3 CONFIGS", ""]
    medals = ["#1", "#2", "#3"]
    for rank, (_, r) in enumerate(top.iterrows(), 1):
        top3_lines.append(
            f"{medals[rank-1]}  {r['arch'].replace('arch_','')} · "
            f"{OPT_LABEL[int(r['yi'])]} @ LR={LR_LABEL[int(r['xi'])]}"
        )
        top3_lines.append(
            f"      val_acc = {r['val_acc_final_mean']:.4f} ± {r['val_acc_final_std']:.4f}"
        )
        top3_lines.append("")
    cax_top3.text(0.5, 0.5, "\n".join(top3_lines).strip(),
                  transform=cax_top3.transAxes, color="#f9e64f", fontsize=9,
                  ha="center", va="center", linespacing=1.5, family="monospace",
                  bbox=dict(facecolor="#161b22", edgecolor="#f9e64f",
                            boxstyle="round,pad=0.6", linewidth=1.2))

    # --- "Cómo leer el cubo" (zona inferior) ---
    cax_howto = fig.add_subplot(gs_right[2, 0])
    cax_howto.set_facecolor(BG); cax_howto.axis("off")
    cax_howto.text(0.5, 0.5,
                   "Cómo leer el cubo\n"
                   "• X = LR (de bajo a alto)\n"
                   "• Y = Optimizer\n"
                   "• Z = Arquitectura\n"
                   "• Tamaño y color = val_acc\n"
                   "• ⊙ amarillo = top-3 cells",
                   transform=cax_howto.transAxes, color=TEXT, fontsize=8.5,
                   ha="center", va="center", linespacing=1.5,
                   bbox=dict(facecolor="#161b22", edgecolor=GRID, boxstyle="round,pad=0.5"))

    # ========== Panel inferior izquierdo: 3 heatmaps faceted por opt ==========
    gs_hm = gs[1, :2].subgridspec(1, 3, wspace=0.32)
    for i, opt in enumerate(OPT_ORDER):
        ax_h = fig.add_subplot(gs_hm[0, i], facecolor=BG)
        sub = df[df["opt"] == opt]
        mat = np.full((nz, nx), np.nan)
        for _, r in sub.iterrows():
            mat[int(r["zi"]), int(r["xi"])] = r["val_acc_final_mean"]
        im = ax_h.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax_h.set_xticks(range(nx)); ax_h.set_xticklabels(LR_LABEL, color=TEXT, fontsize=8)
        ax_h.set_yticks(range(nz)); ax_h.set_yticklabels(ARCH_LABEL, color=TEXT, fontsize=8)
        ax_h.set_title(OPT_LABEL[i], color=TEXT, fontsize=11, fontweight="bold")
        for spine in ax_h.spines.values():
            spine.set_color(GRID)
        # annotate values
        for zi in range(nz):
            for xi in range(nx):
                v = mat[zi, xi]
                if not np.isnan(v):
                    c = "white" if v < (vmin + vmax) / 2 + 0.005 else "#0d1117"
                    ax_h.text(xi, zi, f"{v:.3f}"[1:], ha="center", va="center",
                                color=c, fontsize=7.5)
        if i == 0:
            ax_h.set_ylabel("Arquitectura", color=TEXT, fontsize=9)
        if i == 1:
            ax_h.set_xlabel("Learning rate", color=TEXT, fontsize=9)
        ax_h.tick_params(colors=LABEL)

    # ========== Panel inferior derecho: ranking top-12 ==========
    ax_r = fig.add_subplot(gs[1, 2], facecolor=BG)
    top12 = df.nlargest(12, "val_acc_final_mean").iloc[::-1]
    labels = [f"{r.arch.replace('arch_','')[:6]}·{r.opt[:3]}·{LR_LABEL[int(r.xi)]}"
              for r in top12.itertuples()]
    colors_bar = [cmap((v - vmin) / (vmax - vmin)) for v in top12["val_acc_final_mean"]]
    ax_r.barh(range(len(top12)), top12["val_acc_final_mean"],
                xerr=top12["val_acc_final_std"], color=colors_bar, edgecolor="white",
                linewidth=0.5, capsize=2.5, ecolor=LABEL)
    ax_r.set_yticks(range(len(top12))); ax_r.set_yticklabels(labels, color=TEXT, fontsize=8.5, family="monospace")
    ax_r.set_xlim(0.945, 0.962)
    ax_r.set_xlabel("val_acc (mean ± std)", color=TEXT, fontsize=9)
    ax_r.set_title("Top-12 configuraciones", color=TEXT, fontsize=11, fontweight="bold")
    ax_r.tick_params(colors=LABEL)
    for spine in ax_r.spines.values():
        spine.set_color(GRID)
    ax_r.grid(True, axis="x", alpha=0.2, color=LABEL)
    # annotate values
    for i, (v, s) in enumerate(zip(top12["val_acc_final_mean"], top12["val_acc_final_std"])):
        ax_r.text(v + s + 0.0003, i, f"{v:.4f}", color=TEXT, fontsize=7.5, va="center")

    # Title (con margen claro entre header y subheader)
    fig.suptitle("Stage 2 del Cross-experimento · Grid 3D LR × Optimizer × Arquitectura",
                  color=TEXT, fontsize=18, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925,
              "60 celdas · 3 seeds × 5 folds por celda = 900 corridas · batch heredado de stage 1 · ES patience=20",
              color=LABEL, ha="center", fontsize=10.5, style="italic")

    fig.savefig(OUT, dpi=160, facecolor=BG)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
