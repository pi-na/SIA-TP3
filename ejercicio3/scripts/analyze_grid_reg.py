"""Analisis del paso 2 (grid de regularizacion L2 x sigma).

Outputs:
  ejercicio3/analisis/grid_reg/{
    grid_summary.csv  (16 filas: l2, sigma, val_acc, gap, etc),
    val_acc_heatmap.png  (4x4 con val_acc ± std en cada celda),
    gap_heatmap.png      (4x4 con gap val-train),
    val_loss_heatmap.png (4x4 con val_loss CE),
    best_combo_info.json (la mejor combo elegida),
    grid_results.md      (snippet markdown)
  }
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CV_DIR    = ROOT / "ejercicio3" / "output" / "grid_reg"
OUT       = ROOT / "ejercicio3" / "analisis" / "grid_reg"
NOTES_PNG = ROOT / "Notas" / "ejercicio 3"
OUT.mkdir(parents=True, exist_ok=True)
NOTES_PNG.mkdir(parents=True, exist_ok=True)

BG="#ffffff"; TEXT="#1a1a1a"; LABEL="#555555"; GRID="#cccccc"

SEEDS = [42, 7, 13]
L2_VALUES    = [0.0, 1e-5, 1e-4, 1e-3]
SIGMA_VALUES = [0.0, 0.03, 0.1, 0.2]


def _l2_label(l2: float) -> str:
    if l2 == 0.0: return "0"
    return f"{l2:.0e}".replace("e-0", "e-")


def _sigma_label(s: float) -> str:
    if s == 0.0: return "0"
    return f"{s:.2f}".rstrip("0").rstrip(".")


def load_grid() -> pd.DataFrame:
    rows = []
    for l2 in L2_VALUES:
        for sigma in SIGMA_VALUES:
            seed_dfs = []
            for seed in SEEDS:
                cell_id = f"l2_{_l2_label(l2)}_sigma_{_sigma_label(sigma)}_seed{seed}"
                p = CV_DIR / cell_id / "summary.csv"
                if not p.exists():
                    print(f"WARN: {p} missing")
                    continue
                df = pd.read_csv(p); df["seed"] = seed
                seed_dfs.append(df)
            if not seed_dfs:
                continue
            agg = pd.concat(seed_dfs, ignore_index=True)
            rows.append({
                "l2": l2, "sigma": sigma,
                "l2_label": _l2_label(l2), "sigma_label": _sigma_label(sigma),
                "n": len(agg),
                "val_acc_mean":   agg["val_acc_final"].mean(),
                "val_acc_std":    agg["val_acc_final"].std(),
                "train_acc_mean": agg["train_acc_final"].mean(),
                "macro_f1_mean":  agg["macro_f1"].mean(),
                "macro_f1_std":   agg["macro_f1"].std(),
                "val_loss_mean":  agg["val_loss_final"].mean(),
                "train_loss_mean":agg["train_loss_final"].mean(),
                "gap":            agg["val_loss_final"].mean() - agg["train_loss_final"].mean(),
                "best_epoch_mean": agg["best_epoch"].mean(),
            })
    return pd.DataFrame(rows)


def make_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    mat = np.full((len(L2_VALUES), len(SIGMA_VALUES)), np.nan)
    for i, l2 in enumerate(L2_VALUES):
        for j, sigma in enumerate(SIGMA_VALUES):
            row = df[(np.isclose(df["l2"], l2)) & (np.isclose(df["sigma"], sigma))]
            if not row.empty:
                mat[i, j] = row[col].iloc[0]
    return mat


def plot_heatmap(df: pd.DataFrame, value_col: str, std_col: str | None,
                 title: str, cbar_label: str, cmap: str, out_name: str,
                 reverse_cmap: bool = False, fmt: str = ".4f") -> None:
    mat = make_matrix(df, value_col)
    std_mat = make_matrix(df, std_col) if std_col else None
    fig, ax = plt.subplots(figsize=(8.5, 6.8), facecolor=BG)
    ax.set_facecolor(BG)
    cm = plt.cm.get_cmap(cmap)
    if reverse_cmap: cm = cm.reversed()
    vmin, vmax = float(np.nanmin(mat)), float(np.nanmax(mat))
    im = ax.imshow(mat, cmap=cm, vmin=vmin, vmax=vmax, aspect="auto")

    best_i, best_j = np.unravel_index(np.nanargmax(mat) if not reverse_cmap else np.nanargmin(mat), mat.shape)
    for i in range(len(L2_VALUES)):
        for j in range(len(SIGMA_VALUES)):
            v = mat[i, j]
            if np.isnan(v): continue
            v_norm = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            color = "white" if (v_norm > 0.55 and not reverse_cmap) or (v_norm < 0.45 and reverse_cmap) else TEXT
            txt = f"{v:{fmt}}"
            if std_mat is not None and not np.isnan(std_mat[i, j]):
                txt += f"\n± {std_mat[i,j]:.4f}"
            if i == best_i and j == best_j:
                txt += " ★"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=10, linespacing=1.1,
                    fontweight="bold" if (i == best_i and j == best_j) else "normal")

    ax.set_xticks(range(len(SIGMA_VALUES)))
    ax.set_xticklabels([f"σ={_sigma_label(s)}" for s in SIGMA_VALUES],
                       color=TEXT, fontsize=10)
    ax.set_yticks(range(len(L2_VALUES)))
    ax.set_yticklabels([f"L2={_l2_label(l2)}" for l2 in L2_VALUES],
                       color=TEXT, fontsize=10)
    ax.set_xlabel("σ (gaussian noise augmentation)", color=TEXT, fontsize=11)
    ax.set_ylabel("L2 (weight decay)", color=TEXT, fontsize=11)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=LABEL, labelsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(cbar_label, color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)
    cbar.outline.set_edgecolor(GRID)
    fig.suptitle(title, color=TEXT, fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, -0.02,
             "Cada celda: media (± std) sobre 15 corridas (3 seeds × 5 folds). ★ marca el extremo (mejor/menor según métrica).",
             color=LABEL, ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=160, facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES_PNG / f"grid_{out_name}", dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


def main():
    print("=== Analizando paso 2 grid de regularizacion ===\n")
    df = load_grid()
    if df.empty:
        print("ERROR: no data del grid. Abortando.")
        return
    df = df.sort_values("val_acc_mean", ascending=False).reset_index(drop=True)
    df.to_csv(OUT / "grid_summary.csv", index=False)
    print(f"Loaded {len(df)} combos\n")
    print(df[["l2_label","sigma_label","val_acc_mean","val_acc_std",
              "macro_f1_mean","val_loss_mean","gap","best_epoch_mean"]].to_string(index=False))

    plot_heatmap(df, "val_acc_mean", "val_acc_std",
                 "Ej3 GRID — val_acc por (L2, σ) — best-over-LR fijo en 1e-3",
                 "val_acc (15 corridas)", "viridis",
                 "val_acc_heatmap.png", reverse_cmap=False, fmt=".4f")
    plot_heatmap(df, "val_loss_mean", None,
                 "Ej3 GRID — val_loss CE por (L2, σ)",
                 "val_loss CE (15 corridas)", "viridis",
                 "val_loss_heatmap.png", reverse_cmap=True, fmt=".4f")
    plot_heatmap(df, "gap", None,
                 "Ej3 GRID — gap (val_loss − train_loss) por (L2, σ)",
                 "gap CE (val − train)", "viridis",
                 "gap_heatmap.png", reverse_cmap=True, fmt=".4f")

    # best combo
    best = df.iloc[0]
    best_info = {
        "l2":    float(best["l2"]),
        "sigma": float(best["sigma"]),
        "val_acc_mean": float(best["val_acc_mean"]),
        "val_acc_std":  float(best["val_acc_std"]),
        "macro_f1_mean": float(best["macro_f1_mean"]),
        "val_loss_mean": float(best["val_loss_mean"]),
        "gap":           float(best["gap"]),
    }
    (OUT / "best_combo_info.json").write_text(json.dumps(best_info, indent=2))
    print(f"\nBest combo: L2={best['l2_label']} σ={best['sigma_label']} → val_acc={best['val_acc_mean']:.4f} ± {best['val_acc_std']:.4f}")

    # Markdown snippet
    md = ["### Resultados paso 2 — Grid de regularización (L2 × σ)\n",
          f"Grid 4×4 = 16 combinaciones × 3 seeds × 5 folds = **{16*15} corridas CV**.\n",
          "**Tabla agregada (15 corridas/combo, ordenada por val_acc):**\n",
          "| L2 | σ | val_acc (±std) | macro_F1 (±std) | val_loss CE | train_loss | gap | best_epoch |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, r in df.iterrows():
        md.append(f"| {r['l2_label']} | {r['sigma_label']} | "
                  f"{r['val_acc_mean']:.4f} ± {r['val_acc_std']:.4f} | "
                  f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
                  f"{r['val_loss_mean']:.4f} | {r['train_loss_mean']:.4f} | "
                  f"{r['gap']:.4f} | {r['best_epoch_mean']:.1f} |")
    md.append("\n**Heatmaps:**\n")
    md.append("![[grid_val_acc_heatmap.png]]\n")
    md.append("![[grid_val_loss_heatmap.png]]\n")
    md.append("![[grid_gap_heatmap.png]]\n")
    md.append("\n**Best combo:** L2=`{l2}` σ=`{s}` → val_acc CV = **{v:.4f} ± {vs:.4f}**, gap = **{g:.4f}**.\n"
              .format(l2=best['l2_label'], s=best['sigma_label'],
                      v=best['val_acc_mean'], vs=best['val_acc_std'], g=best['gap']))
    md.append(f"> CSV fuente: [`ejercicio3/analisis/grid_reg/grid_summary.csv`](../../ejercicio3/analisis/grid_reg/grid_summary.csv).\n")

    (OUT / "grid_results.md").write_text("\n".join(md))
    print(f"\nsaved {OUT/'grid_results.md'}")


if __name__ == "__main__":
    main()
