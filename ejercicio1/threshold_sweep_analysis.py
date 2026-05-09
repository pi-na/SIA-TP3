"""Sweep de threshold post-training sobre los runs ya entrenados.

Para cada perceptrón (linear / nonlinear), reconstruye las predicciones
continuas en el fold de test usando los pesos guardados (no re-entrena),
y barre el threshold de decisión sobre una grilla densa.

Por qué tiene sentido este sweep separado:
- LR vive *dentro* del entrenamiento (cambia los pesos).
- Threshold vive *después* (sólo decide cómo binarizar la salida continua).
Por eso elegir LR con la métrica de la loss (MSE) y después barrer threshold
sobre las predicciones del modelo entrenado es coherente con la teoría
(clase de optimizadores: el LR optimiza la loss; clase de métricas: P/R/F1
dependen del threshold). Un sweep gratis: no se vuelve a entrenar.

Outputs por perceptrón en `analisis_outputs/sweep_lr/multiseed/`:
- `threshold_sweep_raw.csv` — una fila por (lr, seed, fold, threshold).
- `threshold_curves.png`    — métricas vs threshold (banda sobre seeds×folds).
- `pr_curve.png`            — curva Precision–Recall.
- `threshold_summary.csv`   — mejor threshold global por lr (max F1 promedio).
- Actualiza `analisis.md` con un bloque "Sweep de threshold".
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT.parent / "data and documentation" / "fraud_dataset.csv"

PERCEPTRONS = {
    "linear": {
        "out_root": ROOT / "lineal_perceptron"   / "output"           / "sweep_lr_multiseed",
        "analisis": ROOT / "lineal_perceptron"   / "analisis_outputs" / "sweep_lr" / "multiseed",
        "activation": "identity",
    },
    "nonlinear": {
        "out_root": ROOT / "nonlinear_perceptron" / "output"           / "sweep_lr_multiseed",
        "analisis": ROOT / "nonlinear_perceptron" / "analisis_outputs" / "sweep_lr" / "multiseed",
        "activation": "sigmoid",
    },
}

THRESHOLD_GRID = np.round(np.linspace(0.01, 0.99, 99), 4)
DISCRETE_THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]


# ---------- Reconstrucción determinística (mismo algoritmo que los runners) ----------

def make_stratified_folds(df: pd.DataFrame, k: int, eval_col: str, seed: int):
    pos_idx = df.index[df[eval_col] == 1].to_numpy().copy()
    neg_idx = df.index[df[eval_col] == 0].to_numpy().copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    pos_chunks = np.array_split(pos_idx, k)
    neg_chunks = np.array_split(neg_idx, k)
    folds = []
    for k_i in range(k):
        test_idx = np.concatenate([pos_chunks[k_i], neg_chunks[k_i]])
        train_idx = np.concatenate(
            [pos_chunks[j] for j in range(k) if j != k_i]
            + [neg_chunks[j] for j in range(k) if j != k_i]
        )
        folds.append((train_idx, test_idx))
    return folds


def sigmoid_stable(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[neg])
    out[neg] = e / (1.0 + e)
    return out


def predict_continuous(weights: np.ndarray, X_test_norm: np.ndarray, activation: str) -> np.ndarray:
    P = X_test_norm.shape[0]
    X_aug = np.column_stack([np.ones(P), X_test_norm])
    h = X_aug @ weights
    return h if activation == "identity" else sigmoid_stable(h)


# ---------- Métricas ----------

def confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    total = tp + fp + fn + tn
    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(accuracy=accuracy, precision=precision, recall=recall, f1=f1)


# ---------- Reconstrucción de predicciones por (lr, seed, fold) ----------

def collect_predictions(perceptron: str, info: dict) -> pd.DataFrame:
    """Devuelve DataFrame con (lr, seed, fold, sample_idx, y_true, y_pred_cont)."""
    df_full = pd.read_csv(CSV_PATH)
    rows = []
    run_dirs = sorted(info["out_root"].glob("*_seed*"))
    if not run_dirs:
        raise FileNotFoundError(
            f"[{perceptron}] no hay run dirs en {info['out_root']}. "
            "Corré primero el sweep multi-seed."
        )
    for rd in run_dirs:
        cfg_path = rd / "config.json"
        w_path   = rd / "weights.csv"
        if not (cfg_path.exists() and w_path.exists()):
            print(f"  WARN: {rd.name} incompleto, salteo.")
            continue
        import json
        cfg = json.loads(cfg_path.read_text())
        seed       = int(cfg["random_seed"])
        k_folds    = int(cfg["k_folds"])
        target_col = cfg["target_col"]
        eval_col   = cfg["eval_col"]
        excluded   = set(cfg["exclude_features"])
        lr_label   = str(cfg["training"]["learning_rate"])
        feature_cols = [c for c in df_full.columns if c not in ({target_col, eval_col} | excluded)]

        weights_df = pd.read_csv(w_path)
        # Garantizar orden de columnas igual al de feature_cols
        for fc in feature_cols:
            if fc not in weights_df.columns:
                raise RuntimeError(f"feature {fc} no está en {w_path}")

        folds = make_stratified_folds(df_full, k_folds, eval_col, seed)

        for fold_i, (train_idx, test_idx) in enumerate(folds):
            train_df = df_full.loc[train_idx]
            test_df  = df_full.loc[test_idx]
            means = train_df[feature_cols].mean().to_numpy()
            stds  = train_df[feature_cols].std(ddof=0).to_numpy()
            stds  = np.where(stds == 0, 1.0, stds)
            X_test_norm = (test_df[feature_cols].to_numpy() - means) / stds
            wrow = weights_df[weights_df["fold"] == fold_i].iloc[0]
            w = np.array([wrow["bias"]] + [wrow[fc] for fc in feature_cols], dtype=float)
            y_pred_cont = predict_continuous(w, X_test_norm, info["activation"])
            y_true_cls = test_df[eval_col].to_numpy().astype(int)
            for s_i, (yt, yp) in enumerate(zip(y_true_cls, y_pred_cont)):
                rows.append(dict(
                    lr=lr_label, seed=seed, fold=fold_i,
                    sample_idx=int(test_df.index[s_i]),
                    y_true=int(yt), y_pred_cont=float(yp),
                ))
    return pd.DataFrame(rows)


# ---------- Sweep threshold ----------

def sweep_threshold(preds: pd.DataFrame) -> pd.DataFrame:
    """Para cada (lr, seed, fold, threshold) computa Acc/Prec/Rec/F1."""
    out = []
    for (lr, seed, fold), g in preds.groupby(["lr", "seed", "fold"]):
        y_true = g["y_true"].to_numpy()
        y_cont = g["y_pred_cont"].to_numpy()
        for thr in THRESHOLD_GRID:
            y_pred = (y_cont >= thr).astype(int)
            m = confusion_metrics(y_true, y_pred)
            out.append(dict(lr=lr, seed=int(seed), fold=int(fold), threshold=float(thr), **m))
    return pd.DataFrame(out)


# ---------- Plots ----------

def plot_threshold_curves(sweep: pd.DataFrame, out_path: Path, title: str) -> None:
    lrs = sorted(sweep["lr"].unique(), key=lambda s: float(s))
    metrics = [("f1", "F1"), ("precision", "Precision"),
               ("recall", "Recall"), ("accuracy", "Accuracy")]
    colors = {"f1": "tab:purple", "precision": "tab:blue",
              "recall": "tab:orange", "accuracy": "tab:green"}
    fig, axes = plt.subplots(1, len(lrs), figsize=(5.5 * len(lrs), 4.6), sharey=True)
    if len(lrs) == 1:
        axes = [axes]
    for ax, lr in zip(axes, lrs):
        g = sweep[sweep["lr"] == lr]
        for col, label in metrics:
            agg = g.groupby("threshold")[col].agg(["mean", "std"])
            ax.plot(agg.index, agg["mean"], color=colors[col], label=label, linewidth=2)
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=colors[col], alpha=0.15)
        # Marca threshold por defecto
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
        # Marca threshold óptimo F1 (global, sobre la curva de F1 promedio)
        f1_curve = g.groupby("threshold")["f1"].mean()
        thr_star = float(f1_curve.idxmax())
        ax.axvline(thr_star, color="tab:red", linestyle=":", linewidth=1.5,
                   label=f"thr* (max F1) = {thr_star:.2f}")
        ax.set_title(f"lr = {lr}")
        ax.set_xlabel("threshold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    axes[0].set_ylabel("métrica")
    fig.suptitle(
        f"{title}\nLínea: media sobre 5 seeds × 5 folds (n=25). "
        "Banda: ±1 std sobre seeds×folds. Línea gris a thr=0.5; punteado rojo en thr* que maximiza F1 promedio."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def plot_pr_curve(sweep: pd.DataFrame, out_path: Path, title: str) -> None:
    """Curva PR: para cada lr, traza precision vs recall (mean sobre seeds×folds, banda en F1)."""
    lrs = sorted(sweep["lr"].unique(), key=lambda s: float(s))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for lr in lrs:
        g = sweep[sweep["lr"] == lr]
        agg = g.groupby("threshold").agg(prec_mean=("precision", "mean"),
                                          rec_mean=("recall", "mean"),
                                          f1_mean=("f1", "mean"))
        ax.plot(agg["rec_mean"], agg["prec_mean"], label=f"lr = {lr}", linewidth=2)
        # Marca thr* (max F1)
        thr_star = float(agg["f1_mean"].idxmax())
        ax.scatter([agg.loc[thr_star, "rec_mean"]], [agg.loc[thr_star, "prec_mean"]],
                   marker="*", s=120, zorder=5,
                   label=f"   thr*={thr_star:.2f} (lr={lr})")
    ax.set_xlabel("Recall (media sobre 5 seeds × 5 folds)")
    ax.set_ylabel("Precision (media sobre 5 seeds × 5 folds)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    ax.set_title(f"{title}\nCurva Precision–Recall por LR (estrella = thr* max F1)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


# ---------- Resúmenes ----------

def threshold_summary(sweep: pd.DataFrame) -> pd.DataFrame:
    """Por lr: thr* global que maximiza F1 promedio + métricas a thr*."""
    rows = []
    for lr, g in sweep.groupby("lr"):
        f1_curve = g.groupby("threshold")["f1"].mean()
        thr_star = float(f1_curve.idxmax())
        at_star = g[g["threshold"] == thr_star]
        rows.append(dict(
            lr=lr,
            thr_star=thr_star,
            f1_mean=at_star["f1"].mean(),         f1_std=at_star["f1"].std(),
            prec_mean=at_star["precision"].mean(), prec_std=at_star["precision"].std(),
            rec_mean=at_star["recall"].mean(),     rec_std=at_star["recall"].std(),
            acc_mean=at_star["accuracy"].mean(),   acc_std=at_star["accuracy"].std(),
        ))
    return pd.DataFrame(rows).sort_values("lr").reset_index(drop=True)


def discrete_thresholds_table(sweep: pd.DataFrame) -> pd.DataFrame:
    """Métricas a thresholds 0.1, 0.3, 0.5, 0.7, 0.9 (mean ± std sobre seeds×folds)."""
    rows = []
    for lr, g in sweep.groupby("lr"):
        for thr in DISCRETE_THRESHOLDS:
            sub = g[np.isclose(g["threshold"], thr)]
            if sub.empty:
                continue
            rows.append(dict(
                lr=lr, threshold=thr,
                accuracy_mean=sub["accuracy"].mean(),   accuracy_std=sub["accuracy"].std(),
                precision_mean=sub["precision"].mean(), precision_std=sub["precision"].std(),
                recall_mean=sub["recall"].mean(),       recall_std=sub["recall"].std(),
                f1_mean=sub["f1"].mean(),               f1_std=sub["f1"].std(),
            ))
    return pd.DataFrame(rows).sort_values(["lr", "threshold"]).reset_index(drop=True)


# ---------- Inserción en analisis.md ----------

THRESHOLD_BLOCK_MARKER = "## Sweep de threshold (post-training)"

def render_threshold_block(perceptron: str, summary: pd.DataFrame,
                            discrete: pd.DataFrame) -> str:
    lines = []
    lines.append(THRESHOLD_BLOCK_MARKER)
    lines.append("")
    lines.append(
        "El threshold de decisión vive **post-training**: no cambia los pesos del perceptrón, "
        "sólo decide cómo binarizar la salida continua. Por eso este sweep no requiere re-entrenar — "
        "se reconstruyen las predicciones de cada (lr, seed, fold) a partir de los pesos guardados "
        "y se evalúan métricas sobre una grilla densa de thresholds."
    )
    lines.append("")
    lines.append("![Curvas threshold](threshold_curves.png)")
    lines.append("")
    lines.append("![Curva Precision-Recall](pr_curve.png)")
    lines.append("")
    lines.append("### Threshold óptimo por LR (max F1 promedio sobre 5 seeds × 5 folds)")
    lines.append("")
    lines.append("Por LR se elige un threshold global (un solo número, no por fold) que maximiza el F1 medio sobre las 25 corridas. Después se reportan las métricas a ese threshold.")
    lines.append("")
    lines.append("| lr | thr* | F1 (mean ± std) | Precision (mean ± std) | Recall (mean ± std) | Accuracy (mean ± std) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['lr']} | {r['thr_star']:.2f} "
            f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} "
            f"| {r['prec_mean']:.4f} ± {r['prec_std']:.4f} "
            f"| {r['rec_mean']:.4f} ± {r['rec_std']:.4f} "
            f"| {r['acc_mean']:.4f} ± {r['acc_std']:.4f} |"
        )
    lines.append("")
    lines.append("### Métricas a thresholds discretos (mean ± std sobre 5 seeds × 5 folds)")
    lines.append("")
    lines.append("| lr | threshold | Accuracy | Precision | Recall | F1 |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in discrete.iterrows():
        lines.append(
            f"| {r['lr']} | {r['threshold']:.2f} "
            f"| {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f} "
            f"| {r['precision_mean']:.4f} ± {r['precision_std']:.4f} "
            f"| {r['recall_mean']:.4f} ± {r['recall_std']:.4f} "
            f"| {r['f1_mean']:.4f} ± {r['f1_std']:.4f} |"
        )
    lines.append("")
    lines.append("### Datos crudos")
    lines.append("")
    lines.append("- `threshold_sweep_raw.csv` — una fila por (lr, seed, fold, threshold) con Acc/Prec/Rec/F1.")
    lines.append("- `threshold_summary.csv` — thr* global por lr y métricas a thr*.")
    lines.append("")
    return "\n".join(lines)


def upsert_threshold_block(analisis_md: Path, block: str) -> None:
    text = analisis_md.read_text()
    if THRESHOLD_BLOCK_MARKER in text:
        # Reemplazar bloque existente (hasta el siguiente '## ' o EOF)
        before, _, rest = text.partition(THRESHOLD_BLOCK_MARKER)
        # Buscamos el siguiente header de nivel 2 a partir del bloque
        rest_after_marker = rest.split("\n", 1)[1] if "\n" in rest else ""
        next_h2 = rest_after_marker.find("\n## ")
        tail = rest_after_marker[next_h2:] if next_h2 >= 0 else ""
        new = before + block + ("\n" + tail.lstrip("\n") if tail else "")
    else:
        new = text.rstrip() + "\n\n" + block + "\n"
    analisis_md.write_text(new)
    print(f"  saved: {analisis_md}")


# ---------- Driver ----------

def run_perceptron(name: str) -> None:
    info = PERCEPTRONS[name]
    info["analisis"].mkdir(parents=True, exist_ok=True)

    print(f"\n=== {name}: reconstruyendo predicciones desde pesos guardados ===")
    preds = collect_predictions(name, info)
    print(f"  {len(preds)} predicciones (= n_samples × n_folds × n_seeds × n_lrs / k_folds_per_run)")

    print(f"=== {name}: barriendo threshold sobre {len(THRESHOLD_GRID)} valores ===")
    sweep = sweep_threshold(preds)
    sweep.to_csv(info["analisis"] / "threshold_sweep_raw.csv", index=False)

    summary  = threshold_summary(sweep)
    discrete = discrete_thresholds_table(sweep)
    summary.to_csv(info["analisis"] / "threshold_summary.csv", index=False)

    plot_threshold_curves(sweep, info["analisis"] / "threshold_curves.png",
                          f"Sweep de threshold post-training - {name}")
    plot_pr_curve(sweep, info["analisis"] / "pr_curve.png",
                  f"Sweep de threshold post-training - {name}")

    block = render_threshold_block(name, summary, discrete)
    upsert_threshold_block(info["analisis"] / "analisis.md", block)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--perceptron", choices=["linear", "nonlinear", "both"], default="both")
    args = p.parse_args()
    targets = ["linear", "nonlinear"] if args.perceptron == "both" else [args.perceptron]
    for t in targets:
        run_perceptron(t)


if __name__ == "__main__":
    main()
