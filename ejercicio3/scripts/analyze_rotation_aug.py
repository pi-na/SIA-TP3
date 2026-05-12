"""Análisis del Experimento A — augmentación rotacional pura.

Lee los outputs de:
  - ejercicio3/output/rotation_aug/{rot10_seed*,rot15_seed*}/summary.csv  (CV)
  - ejercicio3/output/final_eval/rotation_aug/{rot10,rot15}/final_eval_*/test_metrics.csv

Y genera:
  - ejercicio3/analisis/rotation_aug/cv_summary.csv
  - ejercicio3/analisis/rotation_aug/test_summary.csv
  - ejercicio3/analisis/rotation_aug/test_per_class.csv
  - PNGs: comparison_val_acc, comparison_test_acc, convergence_per_config,
          gap_comparison, confusion_matrix_best, per_class_metrics_best.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

CV_DIR        = ROOT / "ejercicio3" / "output" / "rotation_aug"
TEST_DIR      = ROOT / "ejercicio3" / "output" / "final_eval" / "rotation_aug"
ANALYSIS_DIR  = ROOT / "ejercicio3" / "analisis" / "rotation_aug"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

# Baselines previos para comparar (vienen del paso 1 y paso 2 del Ej3).
BASELINE_TEST_ACC      = 0.9616  # Ej3 baseline (+more_digits, sin reg)
BASELINE_TEST_F1       = 0.9609
BEST_REG_TEST_ACC      = 0.9601  # L2=1e-3, sigma=0 (best CV del grid paso 2)
BEST_REG_TEST_F1       = 0.9594
BASELINE_CV_VAL_ACC    = 0.9699  # Ej3 baseline CV val_acc
BEST_REG_CV_VAL_ACC    = 0.9750  # L2=1e-3 CV val_acc

# Tags de los 2 configs nuevos
CONFIGS = ["rot10", "rot15"]
SEEDS   = [42, 7, 13]


# ============================================================
# CV aggregation
# ============================================================

def load_cv_summary(config_tag: str) -> pd.DataFrame:
    """Lee los summary.csv de los 3 seeds de un config, concatena."""
    dfs = []
    for seed in SEEDS:
        cell_dir = CV_DIR / f"{config_tag}_seed{seed}"
        s_path = cell_dir / "summary.csv"
        if not s_path.exists():
            print(f"  AVISO: no encuentro {s_path}")
            continue
        df = pd.read_csv(s_path)
        df["seed"] = seed
        df["config"] = config_tag
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def aggregate_cv() -> pd.DataFrame:
    rows = []
    for tag in CONFIGS:
        df = load_cv_summary(tag)
        if df.empty:
            continue
        n = len(df)
        # Columnas que sabemos existen del run_summary.csv del MLP
        row = {
            "config": tag,
            "n_corridas":          n,
            "val_acc_mean":        df["val_acc_final"].mean(),
            "val_acc_std":         df["val_acc_final"].std(ddof=0),
            "train_acc_mean":      df["train_acc_final"].mean() if "train_acc_final" in df else np.nan,
            "macro_f1_mean":       df["macro_f1"].mean() if "macro_f1" in df else np.nan,
            "macro_f1_std":        df["macro_f1"].std(ddof=0) if "macro_f1" in df else np.nan,
            "val_loss_mean":       df["val_loss_final"].mean() if "val_loss_final" in df else np.nan,
            "train_loss_mean":     df["train_loss_final"].mean() if "train_loss_final" in df else np.nan,
            "best_epoch_mean":     df["best_epoch"].mean() if "best_epoch" in df else np.nan,
            "best_epoch_std":      df["best_epoch"].std(ddof=0) if "best_epoch" in df else np.nan,
        }
        row["gap_val_minus_train"] = row["val_loss_mean"] - row["train_loss_mean"]
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Test aggregation
# ============================================================

def load_test_metrics(config_tag: str) -> pd.DataFrame:
    """Lee los test_metrics.csv de las 3 corridas final_eval de un config."""
    cfg_dir = TEST_DIR / config_tag
    rows = []
    if not cfg_dir.exists():
        return pd.DataFrame()
    for sub in sorted(cfg_dir.glob("final_eval_*")):
        m_path = sub / "test_metrics.csv"
        if not m_path.exists():
            continue
        # extraer seed del path
        m = re.search(r"seed(\d+)", sub.name)
        seed = int(m.group(1)) if m else -1
        df = pd.read_csv(m_path)
        df["config"] = config_tag
        df["seed"] = seed
        df["run_dir"] = str(sub)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def aggregate_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (summary_test, per_class_test) agregados sobre 3 seeds."""
    all_rows = []
    all_per_class = []
    for tag in CONFIGS:
        df = load_test_metrics(tag)
        if df.empty:
            continue
        row = {
            "config": tag,
            "n_seeds": len(df),
            "test_acc_mean":           df["test_accuracy"].mean(),
            "test_acc_std":            df["test_accuracy"].std(ddof=0),
            "test_macro_f1_mean":      df["test_macro_f1"].mean(),
            "test_macro_f1_std":       df["test_macro_f1"].std(ddof=0),
            "test_macro_precision_mean": df["test_macro_precision"].mean(),
            "test_macro_precision_std":  df["test_macro_precision"].std(ddof=0),
            "test_macro_recall_mean":  df["test_macro_recall"].mean(),
            "test_macro_recall_std":   df["test_macro_recall"].std(ddof=0),
            "test_weighted_f1_mean":   df["test_weighted_f1"].mean(),
            "epochs_run_mean":         df["epochs_run"].mean() if "epochs_run" in df else np.nan,
        }
        all_rows.append(row)
        # Per-class
        per_class = []
        for c in range(10):
            per_class.append({
                "config": tag,
                "class": c,
                "precision_mean":  df[f"precision_{c}"].mean(),
                "precision_std":   df[f"precision_{c}"].std(ddof=0),
                "recall_mean":     df[f"recall_{c}"].mean(),
                "recall_std":      df[f"recall_{c}"].std(ddof=0),
                "f1_mean":         df[f"f1_{c}"].mean(),
                "f1_std":          df[f"f1_{c}"].std(ddof=0),
            })
        all_per_class.extend(per_class)
    return pd.DataFrame(all_rows), pd.DataFrame(all_per_class)


def load_test_confusion(config_tag: str) -> np.ndarray:
    """Lee las matrices de confusión de las 3 seeds del config y promedia."""
    cfg_dir = TEST_DIR / config_tag
    cms = []
    for sub in sorted(cfg_dir.glob("final_eval_*")):
        cm_path = sub / "test_confusion_matrix.csv"
        if not cm_path.exists():
            continue
        df = pd.read_csv(cm_path)
        cm = np.zeros((10, 10), dtype=np.float64)
        for _, row in df.iterrows():
            cm[int(row["true_label"]), int(row["pred_label"])] = row["count"]
        cms.append(cm)
    if not cms:
        return np.zeros((10, 10))
    return np.mean(cms, axis=0)


# ============================================================
# Plots
# ============================================================

def plot_comparison_val_acc(cv_df: pd.DataFrame, out_path: Path):
    """Bar chart de val_acc CV, con baselines previas como referencia."""
    fig, ax = plt.subplots(figsize=(9, 5))
    # baselines (no tienen std-bar visible)
    labels = ["Ej3 baseline\n(+more_digits)", "L2=1e-3 σ=0\n(grid_reg best)"]
    vals = [BASELINE_CV_VAL_ACC, BEST_REG_CV_VAL_ACC]
    errs = [0, 0]
    colors = ["#7f8c8d", "#3498db"]
    for tag in CONFIGS:
        row = cv_df[cv_df["config"] == tag]
        if row.empty:
            labels.append(f"{tag}\n(no data)"); vals.append(0); errs.append(0)
            colors.append("#bdc3c7")
        else:
            labels.append(f"{tag}\n±std")
            vals.append(float(row["val_acc_mean"].iloc[0]))
            errs.append(float(row["val_acc_std"].iloc[0]))
            colors.append("#e67e22" if tag == "rot10" else "#c0392b")
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=4, alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, (v, e) in enumerate(zip(vals, errs)):
        if v > 0:
            ax.text(i, v + (e or 0) + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("val_acc (CV interno)", fontsize=11)
    ax.set_title("Val_acc CV — rotación pura vs baselines previas\n"
                 "(mean ± std sobre 3 seeds × 5 folds = 15 corridas)",
                 fontsize=11)
    ax.set_ylim(0.95, max(vals) + 0.012)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_comparison_test_acc(test_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Ej3 baseline\n(+more_digits)", "L2=1e-3 σ=0\n(grid_reg best)"]
    vals = [BASELINE_TEST_ACC, BEST_REG_TEST_ACC]
    errs = [0, 0]
    colors = ["#7f8c8d", "#3498db"]
    for tag in CONFIGS:
        row = test_df[test_df["config"] == tag]
        if row.empty:
            labels.append(f"{tag}\n(no data)"); vals.append(0); errs.append(0)
            colors.append("#bdc3c7")
        else:
            labels.append(f"{tag}\n±std")
            vals.append(float(row["test_acc_mean"].iloc[0]))
            errs.append(float(row["test_acc_std"].iloc[0]))
            colors.append("#e67e22" if tag == "rot10" else "#c0392b")
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=errs, color=colors, capsize=4, alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, (v, e) in enumerate(zip(vals, errs)):
        if v > 0:
            ax.text(i, v + (e or 0) + 0.002, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("test_acc (digits_test.csv)", fontsize=11)
    ax.set_title("Test_acc sobre digits_test.csv — rotación pura vs baselines\n"
                 "(mean ± std sobre 3 seeds; modelo entrenado con digits.csv + more_digits.csv)",
                 fontsize=11)
    ax.set_ylim(0.94, max(vals) + 0.015)
    ax.axhline(0.98, color="#27ae60", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Objetivo CompanyX = 0.98")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_convergence_per_config(out_path: Path):
    """Curvas train_loss y val_loss por época, agregadas sobre las 15 corridas
    de cada config (3 seeds × 5 folds)."""
    fig, axes = plt.subplots(1, len(CONFIGS), figsize=(6 * len(CONFIGS), 4.5), sharey=True)
    if len(CONFIGS) == 1:
        axes = [axes]
    for ax, tag in zip(axes, CONFIGS):
        hist_dfs = []
        for seed in SEEDS:
            h_path = CV_DIR / f"{tag}_seed{seed}" / "history.csv"
            if not h_path.exists():
                continue
            h = pd.read_csv(h_path)
            h["seed"] = seed
            hist_dfs.append(h)
        if not hist_dfs:
            ax.set_title(f"{tag} (no data)")
            continue
        H = pd.concat(hist_dfs, ignore_index=True)
        # Agrupar por epoch (sobre folds × seeds = 15 series)
        agg = H.groupby("epoch").agg(
            train_loss_mean=("train_loss", "mean"),
            train_loss_std =("train_loss", "std"),
            val_loss_mean  =("val_loss",   "mean"),
            val_loss_std   =("val_loss",   "std"),
            n=("train_loss", "count"),
        ).reset_index()
        # Cortar donde sigan vivas al menos 8 corridas
        agg = agg[agg["n"] >= 8].copy()
        ax.plot(agg["epoch"], agg["train_loss_mean"], "-", color="#2980b9", label="train_loss")
        ax.fill_between(agg["epoch"], agg["train_loss_mean"] - agg["train_loss_std"],
                        agg["train_loss_mean"] + agg["train_loss_std"], alpha=0.25, color="#2980b9")
        ax.plot(agg["epoch"], agg["val_loss_mean"], "-", color="#c0392b", label="val_loss")
        ax.fill_between(agg["epoch"], agg["val_loss_mean"] - agg["val_loss_std"],
                        agg["val_loss_mean"] + agg["val_loss_std"], alpha=0.25, color="#c0392b")
        ax.set_xlabel("epoch")
        ax.set_title(f"{tag} — convergencia (15 corridas)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Cross-entropy loss")
    fig.suptitle("Convergencia por config — augmentación rotacional pura", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_gap_comparison(cv_df: pd.DataFrame, out_path: Path):
    """Comparar gap val−train CE entre baselines y rotation configs."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Ej3 baseline\n(+more_digits)", "L2=1e-3 σ=0\n(grid_reg best)"]
    # gaps de baselines (sacados del grid_summary.csv)
    vals = [0.1119, 0.0750]   # de ejercicio3/analisis/grid_reg/grid_summary.csv
    colors = ["#7f8c8d", "#3498db"]
    for tag in CONFIGS:
        row = cv_df[cv_df["config"] == tag]
        if row.empty:
            labels.append(f"{tag}\n(no data)"); vals.append(0); colors.append("#bdc3c7")
        else:
            labels.append(tag)
            vals.append(float(row["gap_val_minus_train"].iloc[0]))
            colors.append("#e67e22" if tag == "rot10" else "#c0392b")
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.6)
    for i, v in enumerate(vals):
        if v > 0:
            ax.text(i, v + 0.003, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("gap = val_loss CE − train_loss CE", fontsize=11)
    ax.set_title("Gap val − train (CV) — menor gap = menos memorización\n"
                 "(mean sobre 3 seeds × 5 folds = 15 corridas)", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, title: str, out_path: Path):
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicción"); ax.set_ylabel("ground truth")
    ax.set_title(title, fontsize=11)
    for i in range(10):
        for j in range(10):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({int(count)})", ha="center", va="center",
                    fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


def plot_per_class_metrics(per_class_df: pd.DataFrame, best_config: str, out_path: Path):
    df = per_class_df[per_class_df["config"] == best_config].copy().sort_values("class")
    classes = df["class"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.27
    x = np.arange(len(classes))
    ax.bar(x - w, df["precision_mean"], w, yerr=df["precision_std"],
           label="precision", color="#3498db", capsize=3, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x,     df["recall_mean"],    w, yerr=df["recall_std"],
           label="recall",    color="#2ecc71", capsize=3, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.bar(x + w, df["f1_mean"],        w, yerr=df["f1_std"],
           label="F1",        color="#e67e22", capsize=3, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([str(int(c)) for c in classes])
    ax.set_xlabel("clase"); ax.set_ylabel("métrica")
    ax.set_title(f"Per-class metrics en test ({best_config}) — mean ± std sobre 3 seeds",
                 fontsize=11)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0.7, 1.02)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("Agregando CV...")
    cv_df = aggregate_cv()
    if not cv_df.empty:
        cv_df.to_csv(ANALYSIS_DIR / "cv_summary.csv", index=False)
        print(cv_df.to_string(index=False))
    else:
        print("  AVISO: no hay datos de CV todavia")

    print("\nAgregando test...")
    test_df, per_class_df = aggregate_test()
    if not test_df.empty:
        test_df.to_csv(ANALYSIS_DIR / "test_summary.csv", index=False)
        per_class_df.to_csv(ANALYSIS_DIR / "test_per_class.csv", index=False)
        print(test_df.to_string(index=False))
    else:
        print("  AVISO: no hay datos de test todavia")

    print("\nGenerando plots...")
    if not cv_df.empty:
        plot_comparison_val_acc(cv_df, ANALYSIS_DIR / "comparison_val_acc.png")
        plot_gap_comparison(cv_df, ANALYSIS_DIR / "gap_comparison.png")
        plot_convergence_per_config(ANALYSIS_DIR / "convergence_per_config.png")
    if not test_df.empty:
        plot_comparison_test_acc(test_df, ANALYSIS_DIR / "comparison_test_acc.png")
        # Best config por test_acc_mean
        best = test_df.sort_values("test_acc_mean", ascending=False).iloc[0]
        best_tag = best["config"]
        print(f"\nBest config en test: {best_tag} (test_acc_mean={best['test_acc_mean']:.4f})")
        cm = load_test_confusion(best_tag)
        plot_confusion_matrix(
            cm, f"Confusion matrix — {best_tag} (mean sobre 3 seeds)",
            ANALYSIS_DIR / "confusion_matrix_best.png",
        )
        plot_per_class_metrics(per_class_df, best_tag,
                               ANALYSIS_DIR / "per_class_metrics_best.png")
    print("\nOK")


if __name__ == "__main__":
    main()
