"""Comparativa head-to-head entre las 3 familias de optimizadores
(SGD vanilla, SGD+Momentum, Adam) en su mejor configuracion respectiva
sobre el grid del stage 2.

Para cada opt tomamos la celda (arch, lr) con val_acc mas alta. Se grafican
4 metricas en 4 subpaneles compactos para que las escalas no compitan:

  1. val_acc          (alto = mejor)
  2. val_loss CE      (bajo = mejor, mejor calibracion)
  3. best_epoch       (bajo = mas rapido, mejor)
  4. overfit gap      (bajo = menos memoriza, mejor)

Output:
  - family_comparison_bars.png  (1x4 paneles agrupados por familia)

Comparte data con best_lr_per_opt_arch.csv pero acompaña UN nivel de
agregacion mas alto: best-of-family, no best-per-(arch,opt).
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
CSV  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "stage2" / "stage2_summary.csv"
OUT  = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "family_comparison"
NOTES = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Cross_LR_Opt_Arch"
OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

BG    = "#ffffff"
TEXT  = "#1a1a1a"
LABEL = "#555555"
GRID  = "#cccccc"

OPT_ORDER  = ["sgd", "momentum", "adam"]
OPT_LABEL  = {"sgd": "SGD", "momentum": "Momentum", "adam": "Adam"}
OPT_COLORS = {"sgd": "#58a6ff", "momentum": "#f78166", "adam": "#3fb950"}


def lr_to_label(lr: float) -> str:
    mapping = {1e-4: "1e-4", 5e-4: "5e-4", 1e-3: "1e-3", 5e-3: "5e-3", 1e-2: "1e-2"}
    for k, v in mapping.items():
        if np.isclose(lr, k):
            return v
    return f"{lr:.0e}"


def best_of_family(df: pd.DataFrame) -> pd.DataFrame:
    """Best (arch, lr) por opt segun val_acc media."""
    df = df.copy()
    df["overfit_gap"] = df["val_loss_final_mean"] - df["train_loss_final_mean"]
    rows = []
    for opt in OPT_ORDER:
        sub = df[df["opt"] == opt]
        best = sub.loc[sub["val_acc_final_mean"].idxmax()]
        rows.append({
            "opt": opt,
            "arch": best["arch"],
            "lr": best["lr"],
            "lr_label": lr_to_label(best["lr"]),
            "val_acc": best["val_acc_final_mean"],
            "val_acc_std": best["val_acc_final_std"],
            "macro_f1": best["macro_f1_mean"],
            "val_loss": best["val_loss_final_mean"],
            "train_loss": best["train_loss_final_mean"],
            "overfit_gap": best["overfit_gap"],
            "best_epoch": best["best_epoch_mean"],
        })
    return pd.DataFrame(rows).set_index("opt").loc[OPT_ORDER].reset_index()


def plot_bars(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6), facecolor=BG)
    metrics = [
        ("val_acc",    "val_acc",        "alto = mejor",  "val_acc_std", None),
        ("val_loss",   "val_loss CE",    "bajo = mejor",  None,          None),
        ("best_epoch", "best_epoch",     "bajo = mas rápido", None,      None),
        ("overfit_gap","gap CE (val−train)", "bajo = menos memoriza", None, None),
    ]
    x = np.arange(len(OPT_ORDER))
    for ax, (col, title, hint, err_col, _) in zip(axes, metrics):
        ax.set_facecolor(BG)
        colors = [OPT_COLORS[o] for o in tab["opt"]]
        yerr = tab[err_col] if err_col else None
        bars = ax.bar(x, tab[col], color=colors, edgecolor="black", linewidth=0.7,
                      yerr=yerr, capsize=5,
                      error_kw=dict(elinewidth=1.0, ecolor="#222"))
        for bar, val in zip(bars, tab[col]):
            if col == "best_epoch":
                lbl = f"{val:.1f}"
            elif col == "val_acc":
                lbl = f"{val:.4f}"
            else:
                lbl = f"{val:.3f}"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    lbl, ha="center", va="bottom", fontsize=9,
                    color=TEXT, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([OPT_LABEL[o] for o in tab["opt"]],
                           color=TEXT, fontsize=10, fontweight="bold")
        ax.set_title(title, color=TEXT, fontsize=12, fontweight="bold", pad=6)
        ax.text(0.5, -0.18, hint, ha="center", va="top",
                transform=ax.transAxes, color=LABEL, fontsize=9, style="italic")
        ax.grid(True, alpha=0.25, color=LABEL, linewidth=0.5, axis="y")
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)
        # margen arriba para que entren los labels
        ymax = ax.get_ylim()[1]
        ax.set_ylim(top=ymax * 1.08)

    # zoom para val_acc (rango chico)
    axes[0].set_ylim(0.94, 0.965)

    # anotacion del best (arch, lr) por familia debajo del titulo principal
    configs = " · ".join(
        f"{OPT_LABEL[r['opt']]}: {r['arch'].replace('arch_','')}@{r['lr_label']}"
        for _, r in tab.iterrows()
    )
    fig.suptitle("Comparativa head-to-head entre familias de optimizadores · best-of-family sobre el stage 2",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.04)
    fig.text(0.5, 0.97, "Configuración ganadora por familia (arch @ LR): " + configs,
             color=LABEL, ha="center", fontsize=10, style="italic")
    fig.text(0.5, -0.06,
             "Cada barra = mejor configuración de esa familia en el grid del stage 2 "
             "(60 cells = 5 LR × 3 opt × 4 arch). Promedios sobre 3 seeds × 5 folds = 15 corridas/celda. "
             "Errorbars de val_acc = std sobre las 15 corridas.",
             color=LABEL, ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT / "family_comparison_bars.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "family_comparison_bars.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'family_comparison_bars.png'}")


def main() -> None:
    df = pd.read_csv(CSV)
    tab = best_of_family(df)
    print("\nBest-of-family:")
    print(tab.to_string(index=False))
    csv_path = OUT / "best_of_family.csv"
    tab.to_csv(csv_path, index=False)
    print(f"\nsaved {csv_path}")
    plot_bars(tab)


if __name__ == "__main__":
    main()
