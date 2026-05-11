"""Evaluación de los modelos entrenados con dataset_{10,25,50,100} contra
digits_test.csv (set de "producción", no se tocó durante el entrenamiento).

Carga cada weights_fracXXX.npz, aplica z-score con las means/stds del TRAIN
correspondiente (sin leakage) y computa el set completo de métricas:
CE-loss + accuracy + macro-precision/recall/f1 (regla 4 de CLAUDE.md).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlp.data import parse_features  # noqa: E402
from mlp.metrics import multiclass_metrics  # noqa: E402
from mlp.network import MLP  # noqa: E402
from mlp.optimizers import build_optimizer  # noqa: E402


OUT_DIR = ROOT / "ejercicio2_experimentacion" / "output" / "dataset_size_sweep"
TEST_CSV = ROOT / "data and documentation" / "digits_test.csv"
FRACTIONS = [0.10, 0.25, 0.50, 1.00]


def _cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    """CE multiclase: mean(-log(p_correcta)). Igual objeto que usamos para entrenar."""
    eps = 1e-12
    p_correct = probs[np.arange(len(y)), y.astype(int)]
    return float(-np.log(np.clip(p_correct, eps, 1.0)).mean())


def main() -> None:
    # Test set
    df_test = pd.read_csv(TEST_CSV)
    X_test_raw = parse_features(df_test, ["image"])
    y_test = df_test["label"].to_numpy().astype(int)
    n_classes = int(y_test.max() + 1)

    print(f"Test set: {TEST_CSV.name}  ·  N={len(df_test)}  ·  clases={n_classes}")
    print(f"Distribución test:\n{pd.Series(y_test).value_counts().sort_index().to_string()}\n")

    # Cargar val_acc desde summary.csv para comparación lado-a-lado
    summary_train = pd.read_csv(OUT_DIR / "summary.csv")

    rows = []
    cm_rows = []
    pred_rows = []

    for frac in FRACTIONS:
        tag = f"frac{int(frac*100):03d}"
        npz = np.load(OUT_DIR / f"weights_{tag}.npz", allow_pickle=False)
        meta = json.loads(str(npz["meta"]))

        layer_sizes = meta["layer_sizes"]
        activations = meta["activations"]
        means = np.array(npz["means"], dtype=np.float64)
        stds  = np.array(npz["stds"],  dtype=np.float64)
        weights = [np.array(npz[f"W{i}"]) for i in range(len(layer_sizes) - 1)]

        # Rebuild MLP shell; el optimizer y la seed acá no importan (solo predicción).
        mlp = MLP(
            layer_sizes=layer_sizes,
            activations=activations,
            loss=meta["loss"],
            optimizer=build_optimizer("adam", lr=1e-3),
            initializer="auto",
            seed=0,
            regularization={"l2": 0.0, "dropout": 0.0, "lr_schedule": None, "augmentation": None},
        )
        mlp.weights = weights

        # Z-score con stats del TRAIN de esta fracción
        X_test = (X_test_raw - means) / stds

        probs = mlp.predict_proba(X_test)
        preds = mlp.predict(X_test)

        ce = _cross_entropy(probs, y_test)
        m = multiclass_metrics(y_test, preds, n_classes)

        val_acc_train = summary_train.loc[
            summary_train["fraction"] == frac, "val_acc_final"].iloc[0]
        val_loss_train = summary_train.loc[
            summary_train["fraction"] == frac, "val_loss_final"].iloc[0]

        row = {
            "fraction": frac,
            "n_train": int(meta["n_train"]),
            "n_test": int(len(y_test)),
            "val_acc (de la corrida)": val_acc_train,
            "val_loss_CE (de la corrida)": val_loss_train,
            "test_acc": m["accuracy"],
            "test_loss_CE": ce,
            "macro_precision": m["macro_precision"],
            "macro_recall": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
            "gap_val_to_test_acc": val_acc_train - m["accuracy"],
        }
        for c in range(n_classes):
            row[f"precision_{c}"] = float(m["precision"][c])
            row[f"recall_{c}"]    = float(m["recall"][c])
            row[f"f1_{c}"]        = float(m["f1"][c])
        rows.append(row)

        cm = m["confusion_matrix"]
        for t in range(n_classes):
            for p in range(n_classes):
                cm_rows.append({
                    "fraction": frac, "true_label": t,
                    "pred_label": p, "count": int(cm[t, p]),
                })

        for i in range(len(y_test)):
            pred_rows.append({
                "fraction": frac, "idx": i,
                "true_label": int(y_test[i]), "pred_label": int(preds[i]),
                "score_max": float(probs[i].max()),
            })

        print(f"=== {tag}  (N_train={meta['n_train']}) ===")
        print(f"  val_acc={val_acc_train:.4f}  →  test_acc={m['accuracy']:.4f}   "
              f"(Δ={val_acc_train - m['accuracy']:+.4f})")
        print(f"  val_loss={val_loss_train:.4f}  →  test_loss_CE={ce:.4f}")
        print(f"  macro_P={m['macro_precision']:.4f}  macro_R={m['macro_recall']:.4f}  "
              f"macro_F1={m['macro_f1']:.4f}  weighted_F1={m['weighted_f1']:.4f}")
        print()

    pd.DataFrame(rows).to_csv(OUT_DIR / "test_metrics.csv", index=False)
    pd.DataFrame(cm_rows).to_csv(OUT_DIR / "test_confusion.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT_DIR / "test_predictions.csv", index=False)
    print(f"Guardado: {OUT_DIR / 'test_metrics.csv'}")
    print(f"Guardado: {OUT_DIR / 'test_confusion.csv'}")
    print(f"Guardado: {OUT_DIR / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
