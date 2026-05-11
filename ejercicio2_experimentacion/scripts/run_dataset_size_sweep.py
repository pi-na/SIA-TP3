"""Sweep de TAMAÑO DE DATASET (BALANCEADO) — Ej2.

Pregunta experimental:
    ¿Cómo cambia el desempeño del MLP óptimo cuando entrenamos con datasets
    PEQUEÑOS pero PERFECTAMENTE BALANCEADOS por clase?

Diseño:
    1. Pool = concat(digits.csv, more_digits.csv).  N=28.190 filas, 10 clases,
       fuertemente desbalanceado: clase 8 = 585 (mínimo); clase 1 = 3.707
       (máximo). digits.csv NO contiene la clase 8 — la aporta more_digits.

    2. La clase MINORITARIA (8, con 585 ejemplares) pone el techo: el pool
       BALANCEADO MÁXIMO sin sobremuestrear es 585 × 10 = 5.850 filas
       (585 por clase). Llamamos a esta cantidad N_BAL_FULL.

    3. dataset_10, dataset_25, dataset_50 toman 10/25/50 % de N_BAL_FULL,
       repartido EN PARTES IGUALES por clase. dataset_100 = pool balanceado
       máximo (585 por clase, 5.850 total).

        dataset_10  → 58/clase   ·   580 total
        dataset_25  → 146/clase  ·  1.460 total
        dataset_50  → 292/clase  ·  2.920 total
        dataset_100 → 585/clase  ·  5.850 total (máximo balanceado)

    4. Sobre cada subconjunto: split estratificado 90/10 train/val (queda
       balanceado en train y val).

    5. Entrenamiento con la config óptima del Ej2 (final_config_ej2.json):
       [784,128,10] relu→softmax, CE-loss, Adam(lr=1e-3), bs=64, z-score,
       epochs=40, ES patience=20.

Single seed (decisión del grupo, no replicación). Por lo tanto **NO HAY std
sobre seeds** — la variabilidad por inicialización/shuffle no se cuantifica
acá. El "promedio" sólo aplica a macro-precision/recall/f1 (promedio sobre
las 10 clases). Ver Notas/testing_datasets_inventados/.

Output: ejercicio2_experimentacion/output/dataset_size_sweep/
    - summary.csv         : una fila por fracción con todas las métricas finales
    - epoch_history.csv   : curvas por época (con columna `fraction`)
    - config_used.json    : la config exacta que se levantó
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mlp.data import parse_features, train_val_split  # noqa: E402
from mlp.train import run_fold  # noqa: E402


FRACTIONS = [0.10, 0.25, 0.50, 1.00]
SEED = 42  # única seed (subsample + split + init MLP)
CONFIG_PATH = ROOT / "ejercicio2_experimentacion" / "configs" / "final_config_ej2.json"
OUTPUT_DIR = ROOT / "ejercicio2_experimentacion" / "output" / "dataset_size_sweep"


def balanced_subsample(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    """Subsample BALANCEADO: misma cantidad por clase = ⌊min_class * fraction⌋.

    La clase minoritaria fija el techo. fraction=1.0 ⇒ min_class por clase
    (pool balanceado máximo). Devuelve índices del array `y`.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    counts = {cls: int(np.sum(y == cls)) for cls in classes}
    min_count = min(counts.values())
    n_per_class = max(1, int(round(min_count * fraction)))

    pieces = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0].copy()
        rng.shuffle(cls_idx)
        pieces.append(cls_idx[:n_per_class])
    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out


def main() -> None:
    cfg = json.loads(CONFIG_PATH.read_text())
    cfg["split"]["random_seed"] = SEED
    cfg["split"]["val_fraction_if_k1"] = 0.1
    cfg["split"]["stratify"] = True

    print(f"Config base: {CONFIG_PATH.name}")
    print(f"  arch:       {cfg['architecture']['layer_sizes']}  "
          f"{cfg['architecture']['activations']}")
    print(f"  optimizer:  {cfg['training']['optimizer']}")
    print(f"  epochs:     {cfg['training']['epochs']}  "
          f"batch_size={cfg['training']['batch_size']}  "
          f"ES patience={cfg['training']['early_stopping_patience']}")
    print(f"  normaliz.:  {cfg['preprocessing']['normalization']}")
    print()

    # Pool = digits + more_digits
    df1 = pd.read_csv(ROOT / "data and documentation" / "digits.csv")
    df2 = pd.read_csv(ROOT / "data and documentation" / "more_digits.csv")
    pool = pd.concat([df1, df2], ignore_index=True)
    X_pool = parse_features(pool, ["image"])
    y_pool = pool["label"].to_numpy()
    print(f"Pool concatenado: N={len(pool)}, clases={sorted(np.unique(y_pool).tolist())}")
    counts = pd.Series(y_pool).value_counts().sort_index()
    print(f"Distribución por clase:\n{counts.to_string()}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "config_used.json").write_text(json.dumps(cfg, indent=2))

    all_summaries = []
    all_histories = []

    for frac in FRACTIONS:
        tag = f"frac{int(frac*100):03d}"
        print(f"=== Fracción {frac:.2f} ({tag}) ===")

        sub_idx = balanced_subsample(y_pool, frac, seed=SEED)
        X = X_pool[sub_idx]
        y = y_pool[sub_idx]

        train_idx, val_idx = train_val_split(
            y, val_fraction=0.1, stratify=True, seed=SEED,
        )
        sub_counts = pd.Series(y).value_counts().sort_index()
        print(f"  N_total={len(y)}  N_train={len(train_idx)}  N_val={len(val_idx)}")
        print(f"  clases en subsample: {sub_counts.to_dict()}")

        t0 = time.time()
        summary, history, weights = run_fold(
            cfg, X, y, train_idx, val_idx, fold_idx=0, fold_seed=SEED,
        )
        elapsed = time.time() - t0

        # Guardar pesos + stats de normalización del TRAIN de esta fracción
        # (necesarias para re-aplicar z-score a digits_test.csv sin leakage).
        X_train = X[train_idx]
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0); stds[stds == 0] = 1.0
        weight_dict = {f"W{i}": W for i, W in enumerate(weights)}
        np.savez_compressed(
            OUTPUT_DIR / f"weights_{tag}.npz",
            means=means, stds=stds,
            meta=json.dumps({
                "fraction": frac,
                "layer_sizes": cfg["architecture"]["layer_sizes"],
                "activations": cfg["architecture"]["activations"],
                "loss": cfg["training"]["loss"],
                "normalization": cfg["preprocessing"]["normalization"],
                "n_train": int(len(train_idx)),
                "seed": SEED,
            }),
            **weight_dict,
        )

        summary_row = {
            "fraction": frac,
            "n_total": len(y),
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            **summary,
            "wall_seconds": elapsed,
        }
        all_summaries.append(summary_row)
        for h in history:
            all_histories.append({"fraction": frac, **h})

        print(f"  → val_acc={summary['val_acc_final']:.4f}  "
              f"val_loss={summary['val_loss_final']:.4f}  "
              f"macro_f1={summary['macro_f1']:.4f}  "
              f"best_epoch={summary['best_epoch']}  "
              f"total_epochs_run={summary['total_epochs_run']}  "
              f"({elapsed:.1f}s)\n")

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(OUTPUT_DIR / "summary.csv", index=False)
    print(f"Guardado: {OUTPUT_DIR / 'summary.csv'}")

    history_df = pd.DataFrame(all_histories)
    history_df.to_csv(OUTPUT_DIR / "epoch_history.csv", index=False)
    print(f"Guardado: {OUTPUT_DIR / 'epoch_history.csv'}")

    print("\n=== Resumen ===")
    cols = ["fraction", "n_train", "val_acc_final", "val_loss_final",
            "train_acc_final", "train_loss_final",
            "macro_precision", "macro_recall", "macro_f1",
            "best_epoch", "total_epochs_run"]
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
