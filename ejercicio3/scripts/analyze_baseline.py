"""Analisis del paso 1 (baseline Ej3 con more_digits.csv).

Replica el analisis A/B/C del Ej2:
  A) Convergencia: plot train/val loss y acc vs epoch, agregado sobre 15 corridas.
  B) Generalizacion interna: tabla con las 4 metricas + losses sobre CV.
  C) Generalizacion externa: tabla test, matriz de confusion, per-clase.

Outputs:
  ejercicio3/analisis/baseline/{
    optimal_convergence.png, optimal_convergence_table.csv,
    cv_internal_summary.csv,
    test_summary.csv, test_per_class.csv,
    test_confusion_matrix.png,
    baseline_results.md  (snippet markdown para inyectar en la nota)
  }
"""
from __future__ import annotations
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CV_DIR     = ROOT / "ejercicio3" / "output" / "baseline"
FINAL_DIR  = ROOT / "ejercicio3" / "output" / "final_eval" / "baseline"
OUT        = ROOT / "ejercicio3" / "analisis" / "baseline"
NOTES_PNG  = ROOT / "Notas" / "ejercicio 3"
OUT.mkdir(parents=True, exist_ok=True)
NOTES_PNG.mkdir(parents=True, exist_ok=True)

BG="#ffffff"; TEXT="#1a1a1a"; LABEL="#555555"; GRID="#cccccc"
COLOR_TRAIN = "#1c4e8f"; COLOR_VAL = "#c92a2a"
COLOR_BAND_TRAIN = "#a8c2e0"; COLOR_BAND_VAL = "#f1aaa4"

SEEDS = [42, 7, 13]


# ============================================================
# A) Convergencia
# ============================================================

def load_epoch_histories() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        seed_dir = CV_DIR / f"baseline_ej3_seed{seed}"
        history = seed_dir / "history.csv"
        if not history.exists():
            print(f"WARN: {history} not found")
            continue
        df = pd.read_csv(history)
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def best_epoch_stats(df: pd.DataFrame) -> dict:
    best_idx = df.groupby(["seed","fold"])["val_loss"].idxmin()
    eps = df.loc[best_idx, "epoch"].to_numpy()
    return {"mean": float(np.mean(eps)), "std": float(np.std(eps)),
            "min": int(np.min(eps)), "max": int(np.max(eps))}


def stop_epoch_stats(df: pd.DataFrame) -> dict:
    stops = df.groupby(["seed","fold"])["epoch"].max().to_numpy()
    return {"mean": float(np.mean(stops)), "std": float(np.std(stops)),
            "min": int(np.min(stops)), "max": int(np.max(stops))}


def plot_convergence(df: pd.DataFrame, best_stats: dict) -> None:
    cols = ["train_loss", "val_loss", "train_acc", "val_acc"]
    agg = df.groupby("epoch")[cols].agg(["mean","std","count"]).reset_index()
    agg.columns = ["epoch"] + [f"{c}_{s}" for c in cols for s in ["mean","std","count"]]
    agg.to_csv(OUT / "optimal_convergence_table.csv", index=False)

    keep = agg[agg["train_loss_count"] >= 8].copy()
    epochs = keep["epoch"].to_numpy()

    fig, (axL, axA) = plt.subplots(1, 2, figsize=(14, 5.3), facecolor=BG)
    for ax in (axL, axA):
        ax.set_facecolor(BG)
        for sp in ax.spines.values(): sp.set_color(GRID)
        ax.tick_params(colors=LABEL, labelsize=9)
        ax.grid(True, alpha=0.25, color=LABEL, linewidth=0.5)

    tlm, tls = keep["train_loss_mean"].to_numpy(), keep["train_loss_std"].to_numpy()
    vlm, vls = keep["val_loss_mean"].to_numpy(), keep["val_loss_std"].to_numpy()
    axL.fill_between(epochs, tlm-tls, tlm+tls, color=COLOR_BAND_TRAIN, alpha=0.4, lw=0)
    axL.fill_between(epochs, vlm-vls, vlm+vls, color=COLOR_BAND_VAL,   alpha=0.4, lw=0)
    axL.plot(epochs, tlm, color=COLOR_TRAIN, lw=2, label="train_loss CE", marker="o", ms=3.5)
    axL.plot(epochs, vlm, color=COLOR_VAL,   lw=2, label="val_loss CE",   marker="s", ms=3.5)
    axL.axvline(best_stats["mean"], color="#555555", ls="--", lw=1.2, alpha=0.7)
    axL.text(best_stats["mean"] + 0.4, axL.get_ylim()[1]*0.93,
             f"best_epoch promedio = {best_stats['mean']:.1f}\n(std = {best_stats['std']:.1f})",
             color="#333", fontsize=9.5, linespacing=1.2)
    axL.set_xlabel("epoch", color=TEXT, fontsize=11)
    axL.set_ylabel("cross-entropy", color=TEXT, fontsize=11)
    axL.set_title("Loss de entrenamiento vs validación", color=TEXT, fontsize=12, fontweight="bold")
    axL.legend(loc="upper right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    tam, tas = keep["train_acc_mean"].to_numpy(), keep["train_acc_std"].to_numpy()
    vam, vas = keep["val_acc_mean"].to_numpy(), keep["val_acc_std"].to_numpy()
    axA.fill_between(epochs, tam-tas, tam+tas, color=COLOR_BAND_TRAIN, alpha=0.4, lw=0)
    axA.fill_between(epochs, vam-vas, vam+vas, color=COLOR_BAND_VAL,   alpha=0.4, lw=0)
    axA.plot(epochs, tam, color=COLOR_TRAIN, lw=2, label="train_acc", marker="o", ms=3.5)
    axA.plot(epochs, vam, color=COLOR_VAL,   lw=2, label="val_acc",   marker="s", ms=3.5)
    axA.axvline(best_stats["mean"], color="#555555", ls="--", lw=1.2, alpha=0.7)
    axA.set_xlabel("epoch", color=TEXT, fontsize=11)
    axA.set_ylabel("accuracy", color=TEXT, fontsize=11)
    axA.set_title("Accuracy de entrenamiento vs validación", color=TEXT, fontsize=12, fontweight="bold")
    axA.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    fig.suptitle("Ej3 BASELINE — Convergencia · shallow + Adam@1e-3 + bs64 + more_digits",
                 color=TEXT, fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.05,
             "Media sobre 15 corridas (3 seeds × 5 folds). Bandas: ± 1 std a nivel de época. "
             "Se grafican épocas donde ≥ 8 de las 15 series seguían vivas. "
             "Vertical: best_epoch promedio.",
             color=LABEL, ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "optimal_convergence.png", dpi=160, facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES_PNG / "baseline_optimal_convergence.png", dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# B) Generalizacion interna (CV)
# ============================================================

def aggregate_cv_internal() -> dict:
    rows = []
    for seed in SEEDS:
        p = CV_DIR / f"baseline_ej3_seed{seed}" / "summary.csv"
        if not p.exists():
            print(f"WARN: {p} not found")
            continue
        df = pd.read_csv(p); df["seed"] = seed
        rows.append(df)
    cv = pd.concat(rows, ignore_index=True)
    s = {
        "n_corridas": len(cv),
        "train_acc_mean":  cv["train_acc_final"].mean(),
        "train_acc_std":   cv["train_acc_final"].std(),
        "train_loss_mean": cv["train_loss_final"].mean(),
        "train_loss_std":  cv["train_loss_final"].std(),
        "val_acc_mean":    cv["val_acc_final"].mean(),
        "val_acc_std":     cv["val_acc_final"].std(),
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
    pd.DataFrame([s]).to_csv(OUT / "cv_internal_summary.csv", index=False)
    return s


# ============================================================
# C) Generalizacion externa (test)
# ============================================================

def aggregate_test() -> tuple[dict, np.ndarray, list[np.ndarray], pd.DataFrame]:
    rows = []; cms = []
    for s in SEEDS:
        ms = glob.glob(str(FINAL_DIR / f"final_eval_*seed{s}_*/test_metrics.csv"))
        if not ms:
            print(f"WARN: no final_eval for seed={s}")
            continue
        rows.append(pd.read_csv(ms[0]).iloc[0])
        cm_p = glob.glob(str(FINAL_DIR / f"final_eval_*seed{s}_*/test_confusion_matrix.csv"))[0]
        cm_df = pd.read_csv(cm_p)
        cm = np.zeros((10,10), dtype=int)
        for _, r in cm_df.iterrows():
            cm[int(r["true_label"]), int(r["pred_label"])] = int(r["count"])
        cms.append(cm)
    test_df = pd.DataFrame(rows)
    cm_mean = np.mean(cms, axis=0)
    test_summary = {
        "n_seeds": len(test_df),
        "test_acc_mean":             test_df["test_accuracy"].mean(),
        "test_acc_std":              test_df["test_accuracy"].std(),
        "test_macro_precision_mean": test_df["test_macro_precision"].mean(),
        "test_macro_precision_std":  test_df["test_macro_precision"].std(),
        "test_macro_recall_mean":    test_df["test_macro_recall"].mean(),
        "test_macro_recall_std":     test_df["test_macro_recall"].std(),
        "test_macro_f1_mean":        test_df["test_macro_f1"].mean(),
        "test_macro_f1_std":         test_df["test_macro_f1"].std(),
        "test_weighted_f1_mean":     test_df["test_weighted_f1"].mean(),
        "test_weighted_f1_std":      test_df["test_weighted_f1"].std(),
    }
    pd.DataFrame([test_summary]).to_csv(OUT / "test_summary.csv", index=False)
    # per-class
    per = []
    for c in range(10):
        per.append({
            "class": c,
            "support_test":    int(cm_mean[c].sum()),
            "precision_mean":  test_df[f"precision_{c}"].mean(),
            "precision_std":   test_df[f"precision_{c}"].std(),
            "recall_mean":     test_df[f"recall_{c}"].mean(),
            "recall_std":      test_df[f"recall_{c}"].std(),
            "f1_mean":         test_df[f"f1_{c}"].mean(),
            "f1_std":          test_df[f"f1_{c}"].std(),
        })
    per_df = pd.DataFrame(per)
    per_df.to_csv(OUT / "test_per_class.csv", index=False)
    return test_summary, cm_mean, cms, per_df


def plot_confusion_matrix(cm_mean: np.ndarray, suffix: str = "baseline") -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.2), facecolor=BG)
    ax.set_facecolor(BG)
    row_totals = cm_mean.sum(axis=1, keepdims=True)
    safe = np.where(row_totals == 0, 1, row_totals)
    norm = cm_mean / safe
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(10):
        for j in range(10):
            v_norm = norm[i, j]; v_abs = cm_mean[i, j]
            color = "white" if v_norm > 0.55 else TEXT
            ax.text(j, i, f"{v_norm:.2f}\n({v_abs:.1f})", ha="center", va="center",
                    color=color, fontsize=7.5, linespacing=1.05,
                    fontweight="bold" if i == j else "normal")
    ax.set_xticks(range(10)); ax.set_xticklabels(range(10), color=TEXT)
    ax.set_yticks(range(10)); ax.set_yticklabels(range(10), color=TEXT)
    ax.set_xlabel("Predicted label", color=TEXT, fontsize=11)
    ax.set_ylabel("True label", color=TEXT, fontsize=11)
    for sp in ax.spines.values(): sp.set_color(GRID)
    ax.tick_params(colors=LABEL, labelsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.04)
    cbar.set_label("recall (normalizado por fila)", color=TEXT, fontsize=10)
    cbar.ax.yaxis.set_tick_params(color=TEXT, labelsize=9)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=TEXT)
    cbar.outline.set_edgecolor(GRID)
    fig.suptitle(f"Ej3 {suffix.upper()} — Matriz de confusión sobre digits_test.csv",
                 color=TEXT, fontsize=13, fontweight="bold", y=0.99)
    fig.text(0.5, -0.02,
             "Cada celda: recall normalizado por fila (cuenta absoluta entre paréntesis). "
             "Promedio sobre 3 seeds.",
             color=LABEL, ha="center", fontsize=9.5, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / f"test_confusion_matrix_{suffix}.png", dpi=160, facecolor=BG, bbox_inches="tight")
    fig.savefig(NOTES_PNG / f"{suffix}_test_confusion_matrix.png", dpi=160, facecolor=BG, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("=== Analizando paso 1 baseline Ej3 ===\n")

    # A) Convergencia
    df = load_epoch_histories()
    if df.empty:
        print("ERROR: no epoch_histories. Abortando.")
        return
    print(f"A) Loaded {len(df)} rows, {df['seed'].nunique()} seeds × {df['fold'].nunique()} folds")
    best_stats = best_epoch_stats(df)
    stop_stats = stop_epoch_stats(df)
    print(f"   best_epoch: mean={best_stats['mean']:.2f} ± {best_stats['std']:.2f}")
    print(f"   stop_epoch: mean={stop_stats['mean']:.2f} ± {stop_stats['std']:.2f}")
    plot_convergence(df, best_stats)

    # B) CV interno
    cv = aggregate_cv_internal()
    print(f"\nB) CV interno (15 corridas):")
    print(f"   train_acc = {cv['train_acc_mean']:.4f} ± {cv['train_acc_std']:.4f}")
    print(f"   val_acc   = {cv['val_acc_mean']:.4f} ± {cv['val_acc_std']:.4f}")
    print(f"   val_f1    = {cv['val_macro_f1_mean']:.4f} ± {cv['val_macro_f1_std']:.4f}")
    print(f"   val_loss  = {cv['val_loss_mean']:.4f}")

    # C) Test
    test_s, cm_mean, cms, per_df = aggregate_test()
    print(f"\nC) Test (3 seeds):")
    print(f"   test_acc       = {test_s['test_acc_mean']:.4f} ± {test_s['test_acc_std']:.4f}")
    print(f"   test_macro_f1  = {test_s['test_macro_f1_mean']:.4f} ± {test_s['test_macro_f1_std']:.4f}")
    print(f"   test_precision = {test_s['test_macro_precision_mean']:.4f}")
    print(f"   test_recall    = {test_s['test_macro_recall_mean']:.4f}")
    # acc sin clase 8
    acc_no8 = []
    for cm in cms:
        mask = np.arange(10) != 8
        total = cm[mask].sum()
        if total > 0:
            correct = sum(cm[i,i] for i in range(10) if i != 8)
            acc_no8.append(correct / total)
    if acc_no8:
        print(f"   test_acc excl clase 8 = {np.mean(acc_no8):.4f} ± {np.std(acc_no8):.4f}")
    plot_confusion_matrix(cm_mean, suffix="baseline")

    # Snippet markdown
    md = ["### Resultados paso 1 — Baseline Ej3 con `more_digits.csv`\n",
          "**A) Convergencia** (sobre 15 corridas = 3 seeds × 5 folds del CV interno).\n",
          "![[baseline_optimal_convergence.png]]\n",
          f"- `best_epoch` promedio = **{best_stats['mean']:.1f} ± {best_stats['std']:.1f}** (range `[{best_stats['min']}, {best_stats['max']}]`).",
          f"- `stop_epoch` (corte ES) promedio = **{stop_stats['mean']:.1f} ± {stop_stats['std']:.1f}** (range `[{stop_stats['min']}, {stop_stats['max']}]`).",
          f"- `max_epochs=50`: {(df.groupby(['seed','fold'])['epoch'].max() >= 49).sum()}/15 corridas llegaron al límite duro.\n",
          "**B) Generalización interna (CV)**.\n",
          "| Métrica | Train (CV, 15 corridas) | Val (CV, 15 corridas) |",
          "| --- | --- | --- |",
          f"| accuracy        | {cv['train_acc_mean']:.4f} ± {cv['train_acc_std']:.4f} | **{cv['val_acc_mean']:.4f} ± {cv['val_acc_std']:.4f}** |",
          f"| macro_precision | (no almacenada) | {cv['val_macro_precision_mean']:.4f} ± {cv['val_macro_precision_std']:.4f} |",
          f"| macro_recall    | (no almacenada) | {cv['val_macro_recall_mean']:.4f} ± {cv['val_macro_recall_std']:.4f} |",
          f"| macro_F1        | (no almacenada) | **{cv['val_macro_f1_mean']:.4f} ± {cv['val_macro_f1_std']:.4f}** |",
          f"| CE loss         | {cv['train_loss_mean']:.4f} ± {cv['train_loss_std']:.4f} | {cv['val_loss_mean']:.4f} ± {cv['val_loss_std']:.4f} |",
          f"| best_epoch      | — | {cv['best_epoch_mean']:.1f} ± {cv['best_epoch_std']:.1f} |\n",
          "**C) Generalización externa (test sobre `digits_test.csv`)**.\n",
          "![[baseline_test_confusion_matrix.png]]\n",
          "| Métrica | Val CV (interno) | **Test** (digits_test.csv) | Δ (val CV − test) |",
          "| --- | --- | --- | --- |",
          f"| accuracy        | {cv['val_acc_mean']:.4f} ± {cv['val_acc_std']:.4f} | **{test_s['test_acc_mean']:.4f} ± {test_s['test_acc_std']:.4f}** | {cv['val_acc_mean']-test_s['test_acc_mean']:+.4f} |",
          f"| macro_precision | {cv['val_macro_precision_mean']:.4f} | {test_s['test_macro_precision_mean']:.4f} ± {test_s['test_macro_precision_std']:.4f} | {cv['val_macro_precision_mean']-test_s['test_macro_precision_mean']:+.4f} |",
          f"| macro_recall    | {cv['val_macro_recall_mean']:.4f} | {test_s['test_macro_recall_mean']:.4f} ± {test_s['test_macro_recall_std']:.4f} | {cv['val_macro_recall_mean']-test_s['test_macro_recall_mean']:+.4f} |",
          f"| macro_F1        | {cv['val_macro_f1_mean']:.4f} | **{test_s['test_macro_f1_mean']:.4f} ± {test_s['test_macro_f1_std']:.4f}** | {cv['val_macro_f1_mean']-test_s['test_macro_f1_mean']:+.4f} |\n",
          "**Métricas por clase en test** (mean sobre 3 seeds):\n",
          "| clase | precision | recall | F1 | support |",
          "| --- | --- | --- | --- | --- |"]
    for _, r in per_df.iterrows():
        bold = "**" if r['class'] == 8 else ""
        md.append(f"| {bold}{int(r['class'])}{bold} | "
                  f"{bold}{r['precision_mean']:.3f}{bold} | "
                  f"{bold}{r['recall_mean']:.3f}{bold} | "
                  f"{bold}{r['f1_mean']:.3f}{bold} | "
                  f"{int(r['support_test'])} |")
    md.append("")
    if acc_no8:
        md.append(f"\n**Test acc excluyendo clase 8** = {np.mean(acc_no8):.4f} ± {np.std(acc_no8):.4f}\n")
    md.append("\n**Comparación con Ej2 (sin `more_digits.csv`)**:\n")
    md.append("| Configuración | Test acc | Test macro_F1 |")
    md.append("| --- | --- | --- |")
    md.append("| Ej2 (sin more_digits, sin reg) | 0.8529 ± 0.0034 | 0.8062 ± 0.0034 |")
    md.append(f"| **Ej3 baseline (+more_digits, sin reg)** | **{test_s['test_acc_mean']:.4f} ± {test_s['test_acc_std']:.4f}** | **{test_s['test_macro_f1_mean']:.4f} ± {test_s['test_macro_f1_std']:.4f}** |")
    delta_acc = test_s['test_acc_mean'] - 0.8529
    delta_f1 = test_s['test_macro_f1_mean'] - 0.8062
    md.append(f"| Δ (Ej3 − Ej2) | **{delta_acc:+.4f}** | **{delta_f1:+.4f}** |\n")

    (OUT / "baseline_results.md").write_text("\n".join(md))
    print(f"\nsaved {OUT/'baseline_results.md'}")


if __name__ == "__main__":
    main()
