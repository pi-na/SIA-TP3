"""Final eval: entrenar con TODO digits.csv y evaluar UNA VEZ sobre digits_test.csv.

No usar durante búsqueda de HP. Sólo cuando el config ganador esté congelado.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from mlp.data import parse_features
from mlp.metrics import multiclass_metrics
from mlp.network import MLP
from mlp.optimizers import build_optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--csv-root", default=Path("."), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--test-csv",
        default="data and documentation/digits_test.csv",
        help="CSV de producción (no se toca durante tuning)",
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    feature_col = cfg["dataset"]["feature_col"]
    target_col = cfg["dataset"]["target_col"]
    num_classes = cfg["dataset"]["num_classes"]

    train_csvs = [cfg["dataset"]["csv_path"]] + cfg["dataset"].get("extra_csv_paths", [])
    df_train = pd.concat([pd.read_csv(args.csv_root / p) for p in train_csvs], ignore_index=True)
    feature_cols = [feature_col] if isinstance(feature_col, str) else list(feature_col)
    X_train = parse_features(df_train, feature_cols)
    y_train_raw = df_train[target_col].to_numpy()

    df_test = pd.read_csv(args.csv_root / args.test_csv)
    X_test = parse_features(df_test, feature_cols)
    y_test = df_test[target_col].to_numpy()

    norm = cfg["preprocessing"]["normalization"]
    if norm == "zscore":
        means, stds = X_train.mean(0), X_train.std(0)
        stds[stds == 0] = 1.0
        X_train = (X_train - means) / stds
        X_test = (X_test - means) / stds
    elif norm == "minmax":
        mins = X_train.min(0)
        rng_ = X_train.max(0) - mins
        rng_[rng_ == 0] = 1.0
        X_train = (X_train - mins) / rng_
        X_test = (X_test - mins) / rng_

    y_train_oh = np.eye(num_classes)[y_train_raw.astype(int)]

    opt_cfg = dict(cfg["training"]["optimizer"])
    opt_name = opt_cfg.pop("name")
    optimizer = build_optimizer(opt_name, **opt_cfg)
    mlp = MLP(
        layer_sizes=cfg["architecture"]["layer_sizes"],
        activations=cfg["architecture"]["activations"],
        loss=cfg["training"]["loss"],
        optimizer=optimizer,
        initializer=cfg["architecture"]["initializer"],
        seed=cfg["split"]["random_seed"],
        regularization=cfg["regularization"],
    )

    n = len(X_train)
    rng = np.random.default_rng(cfg["split"]["random_seed"])
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(n * 0.1)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    print(f"Train final: {len(tr_idx)} samples, val (early stopping): {len(val_idx)}")
    print(f"Test (production): {len(X_test)} samples")

    history = mlp.fit(
        X_train[tr_idx], y_train_oh[tr_idx],
        X_train[val_idx], y_train_oh[val_idx],
        epochs=cfg["training"]["epochs"],
        batch_size=cfg["training"]["batch_size"],
        early_stopping_patience=cfg["training"].get("early_stopping_patience"),
    )

    test_pred = mlp.predict(X_test)
    metrics = multiclass_metrics(y_test, test_pred, num_classes)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"final_eval_{cfg['model_name']}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mlp.save(run_dir / "model.npz")
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    report = {
        "test_accuracy": metrics["accuracy"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        **{f"precision_{c}": float(metrics["precision"][c]) for c in range(num_classes)},
        **{f"recall_{c}": float(metrics["recall"][c]) for c in range(num_classes)},
        **{f"f1_{c}": float(metrics["f1"][c]) for c in range(num_classes)},
        "n_train": len(tr_idx),
        "n_val": len(val_idx),
        "n_test": len(X_test),
        "epochs_run": len(history),
    }
    pd.DataFrame([report]).to_csv(run_dir / "test_metrics.csv", index=False)

    cm = metrics["confusion_matrix"]
    cm_rows = [{"true_label": t, "pred_label": p, "count": int(cm[t, p])}
               for t in range(num_classes) for p in range(num_classes)]
    pd.DataFrame(cm_rows).to_csv(run_dir / "test_confusion_matrix.csv", index=False)

    scores = mlp.predict_proba(X_test)
    preds_df = pd.DataFrame({
        "row_id": np.arange(len(X_test)),
        "true_label": y_test.astype(int),
        "pred_label": test_pred.astype(int),
    })
    for c in range(num_classes):
        preds_df[f"score_{c}"] = scores[:, c]
    preds_df.to_csv(run_dir / "test_predictions.csv", index=False)

    print()
    print(f"=== Final Eval Report ===")
    print(f"Test accuracy: {metrics['accuracy']:.4f}")
    print(f"Test macro F1: {metrics['macro_f1']:.4f}")
    print(f"Test weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Output: {run_dir}")


if __name__ == "__main__":
    main()
