"""Threshold sweep y curvas ROC/PR sobre predictions.csv del mejor modelo.

Genera 3 CSVs:
  - threshold_sweep.csv: metricas para thresholds 0.00 a 1.00 (step 0.01)
  - roc_curve.csv: curva ROC calculada desde todos los scores unicos
  - pr_curve.csv: curva precision-recall desde todos los scores unicos
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_roc_from_scores(scores: np.ndarray, y_true: np.ndarray):
    """Calcula curva ROC desde scores ordenados (sin sklearn).

    Recorre todos los thresholds unicos (scores descendentes),
    agrega puntos (0,0) y (1,1).
    Returns: (fprs, tprs, thresholds) arrays.
    """
    # Unique thresholds sorted descending
    thresholds = np.sort(np.unique(scores))[::-1]

    P = int(y_true.sum())
    N = len(y_true) - P

    tprs = [0.0]
    fprs = [0.0]
    ths = [float(thresholds[0]) + 1e-9]  # above max => all negative

    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tpr = tp / P if P > 0 else 0.0
        fpr = fp / N if N > 0 else 0.0
        tprs.append(tpr)
        fprs.append(fpr)
        ths.append(float(th))

    # Endpoint (1,1) — threshold = -inf
    tprs.append(1.0)
    fprs.append(1.0)
    ths.append(0.0)

    return np.array(fprs), np.array(tprs), np.array(ths)


def compute_pr_from_scores(scores: np.ndarray, y_true: np.ndarray):
    """Calcula curva precision-recall desde scores ordenados.

    Returns: (recalls, precisions, thresholds) arrays.
    """
    thresholds = np.sort(np.unique(scores))[::-1]

    P = int(y_true.sum())

    recalls = []
    precisions = []
    ths = []

    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        ths.append(float(th))

    return np.array(recalls), np.array(precisions), np.array(ths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.predictions)
    scores = df["score"].to_numpy()
    y_true = df["flagged_fraud"].to_numpy().astype(int)

    # --- 1) Threshold sweep: 0.00 a 1.00 en pasos de 0.01 ---
    thresholds = np.round(np.arange(0.0, 1.01, 0.01), 2)
    rows = []
    for th in thresholds:
        y_pred = (scores >= th).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        total = tp + fp + fn + tn

        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        fpr_val = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        rows.append({
            "threshold": float(th),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "accuracy": acc, "precision": prec, "recall": rec,
            "f1": f1, "tpr": rec, "fpr": fpr_val,
        })

    sweep_df = pd.DataFrame(rows)
    sweep_path = args.output_dir / "threshold_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)

    # --- 2) Curva ROC desde scores unicos ---
    fprs, tprs, roc_ths = compute_roc_from_scores(scores, y_true)
    roc_df = pd.DataFrame({"fpr": fprs, "tpr": tprs, "threshold": roc_ths})
    roc_path = args.output_dir / "roc_curve.csv"
    roc_df.to_csv(roc_path, index=False)

    # AUC via trapezoidal
    order = np.argsort(fprs)
    auc = float(np.trapz(tprs[order], fprs[order]))

    # --- 3) Curva PR desde scores unicos ---
    recalls, precisions, pr_ths = compute_pr_from_scores(scores, y_true)
    pr_df = pd.DataFrame({
        "recall": recalls, "precision": precisions, "threshold": pr_ths,
    })
    pr_path = args.output_dir / "pr_curve.csv"
    pr_df.to_csv(pr_path, index=False)

    # --- Summary ---
    best_idx = sweep_df["f1"].idxmax()
    best = sweep_df.iloc[best_idx]
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Best F1: threshold={best['threshold']:.2f}, "
          f"prec={best['precision']:.4f}, rec={best['recall']:.4f}, "
          f"f1={best['f1']:.4f}")
    print(f"Saved: {sweep_path}, {roc_path}, {pr_path}")


if __name__ == "__main__":
    main()
