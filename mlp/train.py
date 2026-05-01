"""CLI: config.json → output dir con CSVs + weights.npz."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlp.activations import ACTIVATIONS
from mlp.data import (
    parse_features, stratified_kfold, train_val_split, BatchIterator,
)
from mlp.losses import LOSSES
from mlp.metrics import multiclass_metrics
from mlp.network import MLP
from mlp.optimizers import build_optimizer


REQUIRED_TOP = {"model_name", "dataset", "split", "preprocessing",
                "architecture", "training", "regularization"}


def load_and_validate_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    missing = REQUIRED_TOP - set(cfg)
    if missing:
        raise ValueError(f"Config inválido: faltan campos {sorted(missing)}")

    arch = cfg["architecture"]
    if len(arch["activations"]) != len(arch["layer_sizes"]) - 1:
        raise ValueError(
            f"activations debe tener {len(arch['layer_sizes'])-1} elementos "
            f"(uno por transición), got {len(arch['activations'])}"
        )

    loss = cfg["training"]["loss"]
    if loss not in LOSSES:
        raise ValueError(f"loss desconocido: {loss!r}")
    if loss == "cross_entropy" and arch["activations"][-1] != "softmax":
        raise ValueError("loss='cross_entropy' requiere última activación='softmax'")
    if loss == "bce" and arch["activations"][-1] != "sigmoid":
        raise ValueError("loss='bce' requiere última activación='sigmoid'")

    opt_name = cfg["training"]["optimizer"]["name"]
    if opt_name not in {"sgd", "momentum", "adam"}:
        raise ValueError(f"optimizer desconocido: {opt_name!r}")

    if cfg["split"]["k_folds"] < 1:
        raise ValueError("k_folds debe ser >= 1")
    return cfg


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    return np.eye(num_classes)[y.astype(int)]


def run_fold(
    cfg: dict, X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray,
    fold_idx: int, fold_seed: int,
) -> tuple[dict, list[dict], list[np.ndarray]]:
    """Entrena un fold. Devuelve (summary, history, weights)."""
    arch = cfg["architecture"]
    train_cfg = cfg["training"]
    num_classes = cfg["dataset"]["num_classes"]
    one_hot = cfg["preprocessing"]["one_hot_targets"]

    X_train, y_train_raw = X[train_idx], y[train_idx]
    X_val, y_val_raw = X[val_idx], y[val_idx]

    # Normalización opcional (z-score fit-on-train)
    norm = cfg["preprocessing"]["normalization"]
    if norm == "zscore":
        means = X_train.mean(axis=0)
        stds = X_train.std(axis=0)
        stds[stds == 0] = 1.0
        X_train = (X_train - means) / stds
        X_val = (X_val - means) / stds
    elif norm == "minmax":
        mins = X_train.min(axis=0)
        rng_ = X_train.max(axis=0) - mins
        rng_[rng_ == 0] = 1.0
        X_train = (X_train - mins) / rng_
        X_val = (X_val - mins) / rng_
    elif norm != "none":
        raise ValueError(f"normalization desconocida: {norm!r}")

    # Targets one-hot si aplica
    y_train = _one_hot(y_train_raw, num_classes) if one_hot else y_train_raw.reshape(-1, 1).astype(float)
    y_val = _one_hot(y_val_raw, num_classes) if one_hot else y_val_raw.reshape(-1, 1).astype(float)

    # Build optimizer + MLP
    opt_cfg = dict(train_cfg["optimizer"])
    opt_name = opt_cfg.pop("name")
    optimizer = build_optimizer(opt_name, **opt_cfg)
    mlp = MLP(
        layer_sizes=arch["layer_sizes"],
        activations=arch["activations"],
        loss=train_cfg["loss"],
        optimizer=optimizer,
        initializer=arch["initializer"],
        seed=fold_seed,
        regularization=cfg["regularization"],
    )

    # Entrenar
    t0 = time.time()
    history_compact = []

    def on_epoch(epoch, m):
        # Evaluar también accuracy en cada época (para epoch_history.csv)
        train_pred = mlp.predict(X_train)
        val_pred = mlp.predict(X_val)
        train_acc = float((train_pred == y_train_raw).mean())
        val_acc = float((val_pred == y_val_raw).mean())
        history_compact.append({
            "epoch": epoch,
            "time_elapsed_s": time.time() - t0,
            "train_loss": m["train_loss"],
            "val_loss": m["val_loss"],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "lr_actual": opt_cfg.get("lr", 0.0),
        })

    mlp.fit(
        X_train, y_train, X_val, y_val,
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        early_stopping_patience=train_cfg.get("early_stopping_patience"),
        callback=on_epoch,
    )
    elapsed = time.time() - t0

    # Métricas finales sobre val
    val_pred = mlp.predict(X_val)
    final = multiclass_metrics(y_val_raw, val_pred, num_classes)
    train_pred = mlp.predict(X_train)
    train_acc_final = float((train_pred == y_train_raw).mean())

    summary = {
        "fold": fold_idx,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "total_epochs": len(history_compact),
        "best_epoch": int(np.argmin([h["val_loss"] for h in history_compact])),
        "train_loss_final": history_compact[-1]["train_loss"],
        "val_loss_final": history_compact[-1]["val_loss"],
        "train_acc_final": train_acc_final,
        "val_acc_final": final["accuracy"],
        "macro_precision": final["macro_precision"],
        "macro_recall": final["macro_recall"],
        "macro_f1": final["macro_f1"],
        "weighted_f1": final["weighted_f1"],
        "time_seconds": elapsed,
    }
    for c in range(num_classes):
        summary[f"precision_{c}"] = float(final["precision"][c])
        summary[f"recall_{c}"] = float(final["recall"][c])
        summary[f"f1_{c}"] = float(final["f1"][c])

    return summary, history_compact, mlp.weights


def _build_folds(cfg: dict, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    s = cfg["split"]
    if s["k_folds"] >= 2:
        return stratified_kfold(y, k=s["k_folds"], seed=s["random_seed"])
    # k_folds == 1 → split simple
    train, val = train_val_split(
        y, val_fraction=s["val_fraction_if_k1"],
        stratify=s["stratify"], seed=s["random_seed"],
    )
    return [(train, val)]


def run_experiment(cfg: dict, csv_root: Path, output_dir: Path) -> Path:
    """Corre todos los folds, escribe CSVs en output_dir/<model_name>_<ts>/."""
    # Cargar dataset principal + extras
    df = pd.read_csv(csv_root / cfg["dataset"]["csv_path"])
    for extra in cfg["dataset"].get("extra_csv_paths", []):
        df_extra = pd.read_csv(csv_root / extra)
        df = pd.concat([df, df_extra], ignore_index=True)

    target_col = cfg["dataset"]["target_col"]
    feature_col = cfg["dataset"]["feature_col"]
    feature_cols = [feature_col] if isinstance(feature_col, str) else list(feature_col)
    X = parse_features(df, feature_cols)
    y = df[target_col].to_numpy()

    folds = _build_folds(cfg, y)

    # Carpeta de output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"{cfg['model_name']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    all_summaries, all_histories, all_predictions, all_cm_rows = [], [], [], []
    weight_dict_for_save = {}

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_seed = cfg["split"]["random_seed"] + fold_idx
        print(f"  fold {fold_idx}: starting…")
        summary, history, weights = run_fold(
            cfg, X, y, train_idx, val_idx, fold_idx, fold_seed,
        )
        all_summaries.append(summary)
        for h in history:
            all_histories.append({"fold": fold_idx, **h})

        # Predicciones out-of-fold
        # (recomputar sobre val_idx con los pesos finales)
        from mlp.network import MLP as _MLP
        # Reconstruir MLP simple para predict_proba (usa los weights finales)
        # — alternativamente devolver el mlp desde run_fold. Lo hacemos así por simplicidad.
        # Aquí asumimos que run_fold ya validó sobre val. Para predicciones detalladas:
        # cargar X_val, normalizar, forward.
        norm = cfg["preprocessing"]["normalization"]
        X_val = X[val_idx]
        if norm == "zscore":
            X_train_for_norm = X[train_idx]
            means, stds = X_train_for_norm.mean(0), X_train_for_norm.std(0)
            stds[stds == 0] = 1.0
            X_val = (X_val - means) / stds
        elif norm == "minmax":
            X_train_for_norm = X[train_idx]
            mins = X_train_for_norm.min(0)
            rng_ = X_train_for_norm.max(0) - mins
            rng_[rng_ == 0] = 1.0
            X_val = (X_val - mins) / rng_

        opt_cfg = dict(cfg["training"]["optimizer"])
        opt_name = opt_cfg.pop("name")
        from mlp.optimizers import build_optimizer as _bo
        tmp_mlp = _MLP(
            layer_sizes=cfg["architecture"]["layer_sizes"],
            activations=cfg["architecture"]["activations"],
            loss=cfg["training"]["loss"],
            optimizer=_bo(opt_name, **opt_cfg),
            initializer=cfg["architecture"]["initializer"],
            seed=fold_seed,
            regularization=cfg["regularization"],
        )
        tmp_mlp.weights = weights  # inyectar pesos entrenados
        scores = tmp_mlp.predict_proba(X_val)
        preds = tmp_mlp.predict(X_val)
        n_classes = cfg["dataset"]["num_classes"]
        for i, row_id in enumerate(val_idx):
            row = {
                "fold": fold_idx, "row_id": int(row_id),
                "true_label": int(y[row_id]), "pred_label": int(preds[i]),
            }
            if scores.shape[1] == n_classes:
                for c in range(n_classes):
                    row[f"score_{c}"] = float(scores[i, c])
            else:
                row["score_0"] = float(scores[i, 0])
            all_predictions.append(row)

        # Confusion matrix stacked
        cm = multiclass_metrics(y[val_idx], preds, n_classes)["confusion_matrix"]
        for t in range(n_classes):
            for p in range(n_classes):
                all_cm_rows.append({
                    "fold": fold_idx, "true_label": t,
                    "pred_label": p, "count": int(cm[t, p]),
                })

        # Save weights for this fold
        for layer_idx, W in enumerate(weights):
            weight_dict_for_save[f"fold{fold_idx}_W{layer_idx}"] = W

        print(f"  fold {fold_idx}: val_acc={summary['val_acc_final']:.4f} "
              f"macro_f1={summary['macro_f1']:.4f}")

    # Filas mean/std al final del summary
    summaries_df = pd.DataFrame(all_summaries)
    numeric = summaries_df.select_dtypes(include=[np.number]).columns
    mean_row = {"fold": "mean", **summaries_df[numeric].mean().to_dict()}
    std_row = {"fold": "std", **summaries_df[numeric].std(ddof=0).to_dict()}
    summaries_df = pd.concat(
        [summaries_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )

    # Escribir CSVs
    summaries_df.to_csv(run_dir / "run_summary.csv", index=False)
    pd.DataFrame(all_histories).to_csv(run_dir / "epoch_history.csv", index=False)
    pd.DataFrame(all_predictions).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(all_cm_rows).to_csv(run_dir / "confusion_matrix.csv", index=False)
    np.savez_compressed(run_dir / "weights.npz",
                        meta=json.dumps({
                            "layer_sizes": cfg["architecture"]["layer_sizes"],
                            "activations": cfg["architecture"]["activations"],
                            "loss": cfg["training"]["loss"],
                        }),
                        **weight_dict_for_save)

    # Resumen a stdout
    print()
    print(f"=== Resumen ===")
    cols = ["fold", "val_acc_final", "macro_f1", "weighted_f1"]
    summary_view = summaries_df[summaries_df["fold"].isin(["mean", "std"])]
    print(summary_view[cols].to_string(index=False))
    print(f"\nOutput: {run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="MLP training engine")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--csv-root", default=Path("."), type=Path,
                        help="Root path for relative csv_path in config")
    args = parser.parse_args()

    cfg = load_and_validate_config(args.config)
    print(f"Modelo: {cfg['model_name']}")
    print(f"Arquitectura: {cfg['architecture']['layer_sizes']} "
          f"({cfg['architecture']['activations']})")
    print(f"Optimizer: {cfg['training']['optimizer']}")
    print(f"K-folds: {cfg['split']['k_folds']}")
    print()
    run_experiment(cfg, args.csv_root, args.output_dir)


if __name__ == "__main__":
    main()
