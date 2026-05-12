"""Agregación de métricas K-fold por LR para el perceptrón no-lineal.

Pool 25 corridas (5 seeds × 5 folds) por LR, hace threshold sweep sobre las
predicciones pooled, encuentra thr* (max F1 sobre los 25 runs promediados) y
reporta MSE_test, Accuracy, Precision, Recall, F1 — mean ± std sobre las 25
corridas evaluadas al thr* propio del LR.

Sirve para responder: "¿LR=1e-2 gana sólo en MSE o también en las demás
métricas comparado con LR=1e-3 y LR=1e-4?"

Uso:
    python lr_kfold_comparativo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "output" / "sweep_lr_multiseed"

LRS = [0.0001, 0.001, 0.01]
SEEDS = [7, 13, 21, 42, 99]
N_FOLDS = 5

THRESHOLDS = np.linspace(0.01, 0.99, 99)


def metrics_at(y_true: np.ndarray, score: np.ndarray, thr: float) -> dict:
    pred = (score >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    total = tp + fp + fn + tn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def load_run(lr: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    lr_tag = str(lr) if lr != 0.0001 else "0.0001"
    # Naming convention: <lr>_seed<seed>
    run_dir = OUTPUT_ROOT / f"{lr_tag}_seed{seed}"
    if not run_dir.exists():
        # Try alternate format
        alt = OUTPUT_ROOT / f"{lr:g}_seed{seed}"
        if alt.exists():
            run_dir = alt
    metrics = pd.read_csv(run_dir / "metrics.csv")
    preds = pd.read_csv(run_dir / "predictions.csv")
    return metrics, preds


def compute_for_lr(lr: float) -> dict:
    """Para un LR: pool runs, sweep thresholds, find thr*, report metrics."""
    all_metrics_rows = []
    all_preds = []
    for seed in SEEDS:
        metrics, preds = load_run(lr, seed)
        # Skip mean/std summary rows in metrics.csv (kept as strings)
        metrics_runs = metrics[~metrics["fold"].astype(str).isin(["mean", "std"])]
        for _, r in metrics_runs.iterrows():
            all_metrics_rows.append({
                "seed": seed,
                "fold": int(float(r["fold"])),
                "mse_test": float(r["mse_test"]),
                "final_mse_train": float(r["final_mse_train"]),
            })
        preds = preds.copy()
        preds["seed"] = seed
        all_preds.append(preds)

    metrics_df = pd.DataFrame(all_metrics_rows)
    preds_df = pd.concat(all_preds, ignore_index=True)

    # Threshold sweep: para cada threshold, evaluar F1 promedio sobre los 25 runs
    # (cada run = una combinación (seed, fold)).
    f1_by_thr = []
    for thr in THRESHOLDS:
        f1s = []
        for (seed, fold), grp in preds_df.groupby(["seed", "fold"]):
            m = metrics_at(grp["flagged_fraud"].to_numpy(),
                           grp["score"].to_numpy(), thr)
            f1s.append(m["f1"])
        f1_by_thr.append((thr, float(np.mean(f1s)), float(np.std(f1s, ddof=0))))

    f1_df = pd.DataFrame(f1_by_thr, columns=["threshold", "f1_mean", "f1_std"])
    thr_star = float(f1_df.loc[f1_df["f1_mean"].idxmax(), "threshold"])

    # Métricas finales por run al thr*
    run_rows = []
    for (seed, fold), grp in preds_df.groupby(["seed", "fold"]):
        m = metrics_at(grp["flagged_fraud"].to_numpy(),
                       grp["score"].to_numpy(), thr_star)
        run_rows.append({"seed": seed, "fold": fold, **m})
    runs_at_thr = pd.DataFrame(run_rows)

    # Combinar con MSE
    combined = runs_at_thr.merge(metrics_df, on=["seed", "fold"])

    def agg(col):
        return combined[col].mean(), combined[col].std(ddof=0)

    return {
        "lr": lr,
        "thr_star": thr_star,
        "n_runs": len(combined),
        "mse_test_mean": agg("mse_test")[0],
        "mse_test_std":  agg("mse_test")[1],
        "mse_train_mean": agg("final_mse_train")[0],
        "mse_train_std":  agg("final_mse_train")[1],
        "accuracy_mean": agg("accuracy")[0],
        "accuracy_std":  agg("accuracy")[1],
        "precision_mean": agg("precision")[0],
        "precision_std":  agg("precision")[1],
        "recall_mean": agg("recall")[0],
        "recall_std":  agg("recall")[1],
        "f1_mean": agg("f1")[0],
        "f1_std":  agg("f1")[1],
    }


def main() -> None:
    print(f"Pool: 5 seeds × 5 folds = 25 corridas por LR. Threshold óptimo "
          f"calculado maximizando F1 promedio sobre las 25 corridas.")
    print()
    rows = []
    for lr in LRS:
        print(f"Procesando LR={lr}…", flush=True)
        try:
            row = compute_for_lr(lr)
            rows.append(row)
            print(f"  thr*={row['thr_star']:.2f}  "
                  f"MSE_test={row['mse_test_mean']:.5f} ± {row['mse_test_std']:.5f}  "
                  f"F1={row['f1_mean']:.4f} ± {row['f1_std']:.4f}  "
                  f"Acc={row['accuracy_mean']:.4f}  "
                  f"P={row['precision_mean']:.4f}  R={row['recall_mean']:.4f}")
        except Exception as e:
            print(f"  ERROR: {e}")

    df = pd.DataFrame(rows)
    out_path = HERE / "analisis_outputs" / "lr_kfold_comparativo.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print()
    print(f"Comparativo guardado en: {out_path}")
    print()
    cols = ["lr", "thr_star", "mse_test_mean", "mse_test_std",
            "accuracy_mean", "precision_mean", "recall_mean", "f1_mean"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
