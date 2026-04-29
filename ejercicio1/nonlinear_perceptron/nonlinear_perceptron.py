"""Perceptron no-lineal (sigmoide) multi-feature con K-fold estratificado.

Entrena un perceptron con activacion sigmoide para replicar el output continuo
del BigModel (`big_model_fraud_probability`) sobre el dataset de fraude.

Diferencias con el perceptron lineal:
    1. Activacion: O = 1 / (1 + exp(-w . x))  en vez de O = w . x
    2. Update rule: dw = eta * (z - O) * O*(1-O) * x  (derivada de la sigmoide)

Pipeline:
    1. Carga CSV y config JSON.
    2. Arma K folds estratificados por `eval_col` (`flagged_fraud`).
    3. Para cada fold: normaliza features (z-score, fit-on-train-only),
       entrena el perceptron online, evalua MSE + metricas de clasificacion.
    4. Escribe 4 CSVs en `output/<model_name>_<timestamp>/`:
         - metrics.csv      (una fila por fold + filas mean/std)
         - mse_history.csv  (una fila por (fold, epoch))
         - weights.csv      (una fila por fold)
         - predictions.csv  (predicciones out-of-fold)

Uso:
    python nonlinear_perceptron.py \\
        --config configs/baseline.json \\
        --csv ../../data/fraud_dataset.csv \\
        [--output-dir output]

"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------- Config & loading ----------------------------- #

REQUIRED_CONFIG_KEYS = {
    "model_name", "target_col", "eval_col", "exclude_features",
    "k_folds", "stratify", "random_seed",
    "training", "evaluation", "normalization",
}
REQUIRED_TRAINING_KEYS = {"learning_rate", "epochs", "epsilon"}
REQUIRED_EVAL_KEYS = {"threshold"}


def load_config(path: Path) -> dict:
    with open(path) as f:
        config = json.load(f)
    missing = REQUIRED_CONFIG_KEYS - set(config)
    if missing:
        raise ValueError(f"Config invalido: faltan campos {sorted(missing)}")
    missing_t = REQUIRED_TRAINING_KEYS - set(config["training"])
    if missing_t:
        raise ValueError(f"Config.training: faltan campos {sorted(missing_t)}")
    missing_e = REQUIRED_EVAL_KEYS - set(config["evaluation"])
    if missing_e:
        raise ValueError(f"Config.evaluation: faltan campos {sorted(missing_e)}")
    if config["normalization"] != "zscore":
        raise NotImplementedError(
            f"normalization='{config['normalization']}' no implementado. "
            "Solo 'zscore' esta soportado en esta iteracion."
        )
    return config


def load_csv(csv_path: Path, config: dict) -> tuple[pd.DataFrame, list[str]]:
    """Carga el CSV y devuelve (df, feature_cols).

    feature_cols = todas las columnas numericas excepto target_col, eval_col,
    y las listadas en exclude_features.
    """
    df = pd.read_csv(csv_path)

    target_col = config["target_col"]
    eval_col = config["eval_col"]
    exclude = set(config["exclude_features"])

    if target_col not in df.columns:
        raise ValueError(f"target_col='{target_col}' no esta en el CSV.")
    if eval_col not in df.columns:
        raise ValueError(f"eval_col='{eval_col}' no esta en el CSV.")

    # Warning sobre exclude_features inexistentes (permisivo, no falla)
    missing_excludes = exclude - set(df.columns)
    if missing_excludes:
        print(f"WARNING: exclude_features ignoradas (no estan en el CSV): "
              f"{sorted(missing_excludes)}", file=sys.stderr)

    reserved = {target_col, eval_col} | exclude
    feature_cols = [c for c in df.columns if c not in reserved]

    # Validar que todas las features sean numericas
    non_numeric = [c for c in feature_cols
                   if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(
            f"Features no numericas: {non_numeric}. "
            "Excluilas via 'exclude_features' o preprocesalas antes."
        )

    return df, feature_cols


# ----------------------------- Stratified K-fold ---------------------------- #

def make_stratified_folds(
    df: pd.DataFrame, k: int, eval_col: str, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold estratificado por eval_col (binaria 0/1).

    Cada fold conserva la proporcion original de clases.
    Returns list of (train_indices, test_indices) using df.index values.
    """
    pos_idx = df.index[df[eval_col] == 1].to_numpy().copy()
    neg_idx = df.index[df[eval_col] == 0].to_numpy().copy()

    if len(pos_idx) < k or len(neg_idx) < k:
        raise ValueError(
            f"k_folds={k} > min(|pos|, |neg|) = "
            f"{min(len(pos_idx), len(neg_idx))}. "
            "Reduci k_folds o agrega datos."
        )

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


def make_simple_folds(
    df: pd.DataFrame, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """K-fold aleatorio (sin estratificar) -- fallback para stratify=False."""
    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    chunks = np.array_split(idx, k)
    folds = []
    for k_i in range(k):
        test_idx = chunks[k_i]
        train_idx = np.concatenate([chunks[j] for j in range(k) if j != k_i])
        folds.append((train_idx, test_idx))
    return folds


# ----------------------------- Normalization -------------------------------- #

def fit_normalizer(
    df_train: pd.DataFrame, feature_cols: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula mean/std por feature sobre el train. std=0 -> 1.0."""
    means = df_train[feature_cols].mean().to_numpy()
    stds = df_train[feature_cols].std(ddof=0).to_numpy()
    stds = np.where(stds == 0, 1.0, stds)
    return means, stds


def apply_normalizer(
    df: pd.DataFrame, means: np.ndarray, stds: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Devuelve copia de df con feature_cols normalizadas (z-score)."""
    out = df.copy()
    out[feature_cols] = (df[feature_cols].to_numpy() - means) / stds
    return out


# ----------------------------- Sigmoid activation --------------------------- #

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoide numericamente estable (mask-based, no evalua ambas ramas)."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    pos = x >= 0
    neg = ~pos
    # Rama positiva: 1/(1+exp(-x)) — safe porque -x <= 0, exp no explota
    exp_neg = np.exp(-x[pos])
    out[pos] = 1.0 / (1.0 + exp_neg)
    # Rama negativa: exp(x)/(1+exp(x)) — safe porque x < 0, exp no explota
    exp_pos = np.exp(x[neg])
    out[neg] = exp_pos / (1.0 + exp_pos)
    return out


# ----------------------------- Perceptron training -------------------------- #

def train_perceptron(
    df_train: pd.DataFrame, feature_cols: list[str], target_col: str,
    learning_rate: float, epochs: int, epsilon: float, seed: int,
) -> tuple[np.ndarray, list[float]]:
    """Perceptron no-lineal (sigmoide) online learning multi-feature.

    Activacion: O = sigmoid(w . x)
    Update:     dw = eta * (z - O) * O*(1-O) * x

    Returns:
        weights: shape (n_features + 1,) -- weights[0] es bias.
        mse_history: lista con MSE de train al final de cada epoca.
    """
    X_raw = df_train[feature_cols].to_numpy()
    z = df_train[target_col].to_numpy()
    P, n = X_raw.shape
    X = np.column_stack([np.ones(P), X_raw])  # bias trick

    rng = np.random.default_rng(seed)
    weights = rng.uniform(-0.1, 0.1, size=n + 1)

    mse_history = []
    for epoch in range(epochs):
        for mu in range(P):
            h = weights @ X[mu]
            O = float(sigmoid(np.array([h]))[0])
            deriv = O * (1.0 - O)
            weights += learning_rate * (z[mu] - O) * deriv * X[mu]
        # Predictions for full training set
        H = X @ weights
        predictions = sigmoid(H)
        mse = float(np.mean((z - predictions) ** 2))
        mse_history.append(mse)
        if mse < epsilon:
            break
    return weights, mse_history


# ----------------------------- Metrics -------------------------------------- #

def compute_metrics(
    weights: np.ndarray, df: pd.DataFrame, feature_cols: list[str],
    target_col: str, eval_col: str, threshold: float,
) -> tuple[dict, np.ndarray]:
    """MSE (regresion vs target_col) + metricas de clasificacion (vs eval_col).

    Devuelve (dict_metrics, scores) donde scores son las salidas sigmoid.
    """
    P = len(df)
    X = np.column_stack([np.ones(P), df[feature_cols].to_numpy()])
    H = X @ weights
    O = sigmoid(H)

    y_true_reg = df[target_col].to_numpy()
    y_true_cls = df[eval_col].to_numpy().astype(int)
    y_pred_cls = (O >= threshold).astype(int)

    mse = float(np.mean((O - y_true_reg) ** 2))

    tp = int(((y_pred_cls == 1) & (y_true_cls == 1)).sum())
    fp = int(((y_pred_cls == 1) & (y_true_cls == 0)).sum())
    fn = int(((y_pred_cls == 0) & (y_true_cls == 1)).sum())
    tn = int(((y_pred_cls == 0) & (y_true_cls == 0)).sum())
    total = tp + fp + fn + tn

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0  # = TPR
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )

    metrics = {
        "mse": mse,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tpr": recall,
        "fpr": fpr,
    }
    return metrics, O


# ----------------------------- Fold orchestration --------------------------- #

def train_and_test_fold(
    train_df: pd.DataFrame, test_df: pd.DataFrame,
    test_original_indices: np.ndarray,
    feature_cols: list[str], config: dict, fold_seed: int, fold_idx: int,
) -> tuple[dict, list[float], np.ndarray, list[dict]]:
    """Entrena en train_df, evalua en test_df.

    Devuelve (metrics, mse_hist, weights, prediction_rows).
    """
    target_col = config["target_col"]
    eval_col = config["eval_col"]
    threshold = config["evaluation"]["threshold"]
    lr = config["training"]["learning_rate"]
    epochs = config["training"]["epochs"]
    epsilon = config["training"]["epsilon"]

    # Normalize fit-on-train-only
    means, stds = fit_normalizer(train_df, feature_cols)
    train_norm = apply_normalizer(train_df, means, stds, feature_cols)
    test_norm = apply_normalizer(test_df, means, stds, feature_cols)

    weights, mse_history = train_perceptron(
        train_norm, feature_cols, target_col,
        learning_rate=lr, epochs=epochs, epsilon=epsilon, seed=fold_seed,
    )

    test_metrics, scores = compute_metrics(
        weights, test_norm, feature_cols, target_col, eval_col, threshold,
    )

    # Build prediction rows for this fold
    prediction_rows = []
    y_bigmodel = test_df[target_col].to_numpy()
    flagged = test_df[eval_col].to_numpy()
    for i in range(len(test_df)):
        prediction_rows.append({
            "fold": fold_idx,
            "row_id": int(test_original_indices[i]),
            "y_bigmodel": float(y_bigmodel[i]),
            "flagged_fraud": int(flagged[i]),
            "score": float(scores[i]),
        })

    fold_metrics = {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "threshold": threshold,
        "learning_rate": lr,
        "epochs_run": len(mse_history),
        "final_mse_train": mse_history[-1],
        "mse_test": test_metrics["mse"],
        "tp": test_metrics["tp"],
        "fp": test_metrics["fp"],
        "fn": test_metrics["fn"],
        "tn": test_metrics["tn"],
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "tpr": test_metrics["tpr"],
        "fpr": test_metrics["fpr"],
    }
    return fold_metrics, mse_history, weights, prediction_rows


def _run_fold_worker(args):
    """Worker para multiprocessing.Pool. Devuelve (fold_idx, metrics, hist, w, preds).

    Top-level para que sea picklable en macOS (spawn).
    """
    (fold_idx, train_df, test_df, test_original_indices,
     feature_cols, config, fold_seed) = args
    fold_metrics, mse_history, weights, prediction_rows = train_and_test_fold(
        train_df, test_df, test_original_indices,
        feature_cols, config, fold_seed, fold_idx,
    )
    return fold_idx, fold_metrics, mse_history, weights, prediction_rows


def train_and_test_dataset(
    df: pd.DataFrame, feature_cols: list[str], config: dict,
    workers: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """K-fold completo. Devuelve (metrics_df, mse_history_df, weights_df, predictions_df).

    Args:
        workers: cantidad de procesos paralelos. None = k_folds (paralelo total).
                 1 = ejecucion serial (util para debugging).
    """
    k = config["k_folds"]
    seed = config["random_seed"]
    eval_col = config["eval_col"]

    if config["stratify"]:
        folds = make_stratified_folds(df, k, eval_col, seed)
    else:
        folds = make_simple_folds(df, k, seed)

    fold_args = []
    for k_i, (train_idx, test_idx) in enumerate(folds):
        train_df = df.loc[train_idx].reset_index(drop=True)
        test_df = df.loc[test_idx].reset_index(drop=True)
        fold_args.append((
            k_i, train_df, test_df, test_idx,
            feature_cols, config, seed + k_i,
        ))

    if workers is None:
        workers = k
    workers = max(1, min(workers, k))

    if workers > 1:
        print(f"  (running {k} folds in parallel with {workers} workers)")
        with mp.Pool(workers) as pool:
            results = list(pool.imap_unordered(_run_fold_worker, fold_args))
    else:
        results = [_run_fold_worker(a) for a in fold_args]

    results.sort(key=lambda x: x[0])

    all_metrics = []
    all_history = []
    all_weights = []
    all_predictions = []
    for fold_idx, fold_metrics, mse_hist, weights, pred_rows in results:
        fold_metrics = {"fold": fold_idx, **fold_metrics}
        all_metrics.append(fold_metrics)

        for epoch_idx, mse_val in enumerate(mse_hist):
            all_history.append({
                "fold": fold_idx, "epoch": epoch_idx, "mse_train": mse_val,
            })

        weights_row = {"fold": fold_idx, "bias": float(weights[0])}
        for fname, w in zip(feature_cols, weights[1:]):
            weights_row[fname] = float(w)
        all_weights.append(weights_row)

        all_predictions.extend(pred_rows)

        print(
            f"  fold {fold_idx}: mse_test={fold_metrics['mse_test']:.4f}  "
            f"acc={fold_metrics['accuracy']:.4f}  "
            f"prec={fold_metrics['precision']:.4f}  "
            f"rec={fold_metrics['recall']:.4f}  "
            f"f1={fold_metrics['f1']:.4f}  "
            f"({fold_metrics['epochs_run']} epochs)"
        )

    metrics_df = pd.DataFrame(all_metrics)

    # Filas resumen: mean / std sobre columnas numericas (excepto fold)
    numeric_cols = [c for c in metrics_df.columns if c != "fold"]
    mean_row = {"fold": "mean", **metrics_df[numeric_cols].mean().to_dict()}
    std_row = {"fold": "std", **metrics_df[numeric_cols].std(ddof=0).to_dict()}
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([mean_row, std_row])], ignore_index=True
    )

    mse_history_df = pd.DataFrame(all_history)
    weights_df = pd.DataFrame(all_weights)
    predictions_df = pd.DataFrame(all_predictions)
    return metrics_df, mse_history_df, weights_df, predictions_df


# ----------------------------- Main ----------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output-dir", default=Path("output"), type=Path)
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Procesos paralelos (default = k_folds). 1 = serial.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df, feature_cols = load_csv(args.csv, config)

    print(f"Modelo: {config['model_name']}")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Filas: {len(df)} (positivos en '{config['eval_col']}': "
          f"{int((df[config['eval_col']] == 1).sum())})")
    print(f"K folds: {config['k_folds']}  stratify: {config['stratify']}")
    print(f"Training: lr={config['training']['learning_rate']}, "
          f"epochs={config['training']['epochs']}, "
          f"epsilon={config['training']['epsilon']}")
    print()

    metrics_df, mse_history_df, weights_df, predictions_df = train_and_test_dataset(
        df, feature_cols, config, workers=args.workers,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / f"{config['model_name']}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    metrics_df.to_csv(run_dir / "metrics.csv", index=False)
    mse_history_df.to_csv(run_dir / "mse_history.csv", index=False)
    weights_df.to_csv(run_dir / "weights.csv", index=False)
    predictions_df.to_csv(run_dir / "predictions.csv", index=False)

    # Copiar el config al output dir (reproducibilidad)
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print(f"=== Resumen K-fold ===")
    summary = metrics_df[metrics_df["fold"].isin(["mean", "std"])]
    cols = ["fold", "mse_test", "accuracy", "precision", "recall", "f1"]
    print(summary[cols].to_string(index=False))
    print()
    print(f"Outputs en: {run_dir}")


if __name__ == "__main__":
    main()
