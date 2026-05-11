"""Agrega y reporta convergencia + generalizacion de la config optima del Ej2.

(A) CV interno: lee los 3 summary.csv del stage 2 (3 seeds x 5 folds = 15 corridas)
    y reporta train vs val con las 4 metricas + losses.
(B) Test (digits_test.csv): lee los 3 test_metrics.csv y test_confusion_matrix.csv
    de final_eval_ej2 y reporta mean ± std + plot de confusion matrix.
(C) Comparacion CV interno vs Test (gap de generalizacion fuera de muestra).
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
STAGE_DIR = ROOT / "ejercicio2_experimentacion" / "output" / "cross_v1" / "stage2"
FINAL_DIR = ROOT / "ejercicio2_experimentacion" / "output" / "final_eval_ej2"
OUT = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1" / "optimal_config"
NOTES = ROOT / "Notas" / "ejercicio 2" / "Segunda tanda de experimentos" / "Cross_LR_Opt_Arch"
OUT.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

BG="#ffffff"; TEXT="#1a1a1a"; LABEL="#555555"; GRID="#cccccc"

SEEDS = [42, 7, 13]
CELL  = "stage2_arch_shallow_adam_lr1e-3_bs64"


def aggregate_cv_internal() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        p = STAGE_DIR / f"{CELL}_seed{seed}" / "summary.csv"
        df = pd.read_csv(p)
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def aggregate_test() -> tuple[pd.DataFrame, np.ndarray, list[np.ndarray]]:
    rows = []
    cms = []
    for s in SEEDS:
        p_m = glob.glob(str(FINAL_DIR / f"final_eval_final_ej2_seed{s}_*/test_metrics.csv"))[0]
        rows.append(pd.read_csv(p_m).iloc[0])
        p_cm = glob.glob(str(FINAL_DIR / f"final_eval_final_ej2_seed{s}_*/test_confusion_matrix.csv"))[0]
        df = pd.read_csv(p_cm)
        cm = np.zeros((10, 10), dtype=int)
        for _, r in df.iterrows():
            cm[int(r["true_label"]), int(r["pred_label"])] = int(r["count"])
        cms.append(cm)
    return pd.DataFrame(rows), np.mean(cms, axis=0), cms


def plot_confusion_matrix(cm_mean: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.2), facecolor=BG)
    ax.set_facecolor(BG)
    row_totals = cm_mean.sum(axis=1, keepdims=True)
    safe_totals = np.where(row_totals == 0, 1, row_totals)
    norm_cm = cm_mean / safe_totals
    im = ax.imshow(norm_cm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(10):
        for j in range(10):
            v_norm = norm_cm[i, j]
            v_abs  = cm_mean[i, j]
            color = "white" if v_norm > 0.55 else TEXT
            # Mostrar TODAS las celdas (incluidas las de valor 0)
            txt = f"{v_norm:.2f}\n({v_abs:.1f})"
            ax.text(j, i, txt, ha="center", va="center",
                    color=color, fontsize=7.5, linespacing=1.05,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10)); ax.set_xticklabels(range(10), color=TEXT)
    ax.set_yticks(range(10)); ax.set_yticklabels(range(10), color=TEXT)
    ax.set_xlabel("Predicted label", color=TEXT, fontsize=11)
    ax.set_ylabel("True label", color=TEXT, fontsize=11)
    ax.tick_params(colors=LABEL, labelsize=10)
    for spine in ax.spines.values(): spine.set_color(GRID)

    # destacar fila clase 8 (ausente en train)
    rect = plt.Rectangle((-0.5, 7.5), 10, 1, fill=False,
                          edgecolor="#c92a2a", linewidth=2.4)
    ax.add_patch(rect)
    ax.text(10.2, 8, "← clase 8 ausente\n   en digits.csv",
            color="#c92a2a", fontsize=10, va="center", fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cbar.set_label("recall (normalizado por fila)", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)
    cbar.outline.set_edgecolor(GRID)

    fig.suptitle("Matriz de confusión sobre digits_test.csv — config óptima",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, -0.02,
             "Cada celda: recall normalizado por fila (cuenta absoluta entre paréntesis). Promedio sobre 3 seeds (42, 7, 13). "
             "La fila 8 está toda en 0 porque la clase 8 no existe en digits.csv (train) — el modelo nunca la vio.",
             color=LABEL, ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "optimal_test_confusion_matrix.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES / "optimal_test_confusion_matrix.png", dpi=160,
                facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT/'optimal_test_confusion_matrix.png'}")


def main():
    # ---- CV interno
    cv = aggregate_cv_internal()
    print(f"CV interno: {len(cv)} corridas ({cv['seed'].nunique()} seeds x "
          f"{cv['fold'].nunique()} folds)")
    cv_summary = {
        "n_corridas": len(cv),
        "train_acc_mean":  cv["train_acc_final"].mean(),
        "train_acc_std":   cv["train_acc_final"].std(),
        "train_loss_mean": cv["train_loss_final"].mean(),
        "train_loss_std":  cv["train_loss_final"].std(),
        "val_acc_mean":  cv["val_acc_final"].mean(),
        "val_acc_std":   cv["val_acc_final"].std(),
        "val_macro_precision_mean": cv["macro_precision"].mean(),
        "val_macro_precision_std":  cv["macro_precision"].std(),
        "val_macro_recall_mean":    cv["macro_recall"].mean(),
        "val_macro_recall_std":     cv["macro_recall"].std(),
        "val_macro_f1_mean":        cv["macro_f1"].mean(),
        "val_macro_f1_std":         cv["macro_f1"].std(),
        "val_loss_mean": cv["val_loss_final"].mean(),
        "val_loss_std":  cv["val_loss_final"].std(),
        "best_epoch_mean": cv["best_epoch"].mean(),
        "best_epoch_std":  cv["best_epoch"].std(),
    }
    pd.DataFrame([cv_summary]).to_csv(OUT / "cv_internal_summary.csv", index=False)
    print(f"saved {OUT/'cv_internal_summary.csv'}")

    # ---- Test
    test_df, cm_mean, cms = aggregate_test()
    print(f"\nTest: {len(test_df)} corridas (seeds {SEEDS})")
    test_summary = {
        "n_seeds": len(test_df),
        "test_acc_mean":  test_df["test_accuracy"].mean(),
        "test_acc_std":   test_df["test_accuracy"].std(),
        "test_macro_precision_mean": test_df["test_macro_precision"].mean(),
        "test_macro_precision_std":  test_df["test_macro_precision"].std(),
        "test_macro_recall_mean":    test_df["test_macro_recall"].mean(),
        "test_macro_recall_std":     test_df["test_macro_recall"].std(),
        "test_macro_f1_mean": test_df["test_macro_f1"].mean(),
        "test_macro_f1_std":  test_df["test_macro_f1"].std(),
        "test_weighted_f1_mean": test_df["test_weighted_f1"].mean(),
        "test_weighted_f1_std":  test_df["test_weighted_f1"].std(),
    }
    pd.DataFrame([test_summary]).to_csv(OUT / "test_summary.csv", index=False)
    print(f"saved {OUT/'test_summary.csv'}")

    # tabla por clase
    per_class = []
    for c in range(10):
        per_class.append({
            "class": c,
            "support_test": int(cm_mean[c].sum()),
            "precision_mean": test_df[f"precision_{c}"].mean(),
            "precision_std":  test_df[f"precision_{c}"].std(),
            "recall_mean":    test_df[f"recall_{c}"].mean(),
            "recall_std":     test_df[f"recall_{c}"].std(),
            "f1_mean":        test_df[f"f1_{c}"].mean(),
            "f1_std":         test_df[f"f1_{c}"].std(),
        })
    per_class_df = pd.DataFrame(per_class)
    per_class_df.to_csv(OUT / "test_per_class.csv", index=False)
    print(f"saved {OUT/'test_per_class.csv'}")

    # accuracy excluyendo clase 8 (no presente en train)
    acc_no8 = []
    for cm in cms:
        mask_rows = np.arange(10) != 8
        total_non8 = cm[mask_rows].sum()
        correct_non8 = sum(cm[i,i] for i in range(10) if i != 8)
        acc_no8.append(correct_non8 / total_non8)
    print(f"\nTest acc excluyendo casos donde GT=8: "
          f"{np.mean(acc_no8):.4f} ± {np.std(acc_no8):.4f}")

    # ---- Plot
    plot_confusion_matrix(cm_mean)

    # ---- Resumen impreso
    print("\n=== Reporte final ===")
    print(f"CV interno (val, sobre 15 corridas del stage 2):")
    print(f"  acc       = {cv_summary['val_acc_mean']:.4f} ± {cv_summary['val_acc_std']:.4f}")
    print(f"  precision = {cv_summary['val_macro_precision_mean']:.4f} ± {cv_summary['val_macro_precision_std']:.4f}")
    print(f"  recall    = {cv_summary['val_macro_recall_mean']:.4f} ± {cv_summary['val_macro_recall_std']:.4f}")
    print(f"  F1 macro  = {cv_summary['val_macro_f1_mean']:.4f} ± {cv_summary['val_macro_f1_std']:.4f}")
    print(f"  CE        = {cv_summary['val_loss_mean']:.4f} ± {cv_summary['val_loss_std']:.4f}")
    print(f"\nTest (sobre 3 seeds, full train + eval en digits_test.csv):")
    print(f"  acc       = {test_summary['test_acc_mean']:.4f} ± {test_summary['test_acc_std']:.4f}")
    print(f"  precision = {test_summary['test_macro_precision_mean']:.4f} ± {test_summary['test_macro_precision_std']:.4f}")
    print(f"  recall    = {test_summary['test_macro_recall_mean']:.4f} ± {test_summary['test_macro_recall_std']:.4f}")
    print(f"  F1 macro  = {test_summary['test_macro_f1_mean']:.4f} ± {test_summary['test_macro_f1_std']:.4f}")
    print(f"\nDelta (CV val - test): acc = {cv_summary['val_acc_mean']-test_summary['test_acc_mean']:+.4f}")
    print(f"Test acc excluyendo clase 8: {np.mean(acc_no8):.4f} (gap residual = {cv_summary['val_acc_mean']-np.mean(acc_no8):+.4f})")


if __name__ == "__main__":
    main()
