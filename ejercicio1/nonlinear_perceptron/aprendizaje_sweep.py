"""Runner paralelo para experimentos de APRENDIZAJE — perceptrón no-lineal (sigmoide).

Entrena el perceptrón no-lineal sobre TODO el dataset (sin train/test split, sin
folds), barriendo learning_rate × seed. Por época registra MSE y R² sobre el
mismo dataset de entrenamiento.

Decisiones (consistentes con ejercicio2_experimentacion/scripts/runner_ejemplo_multiprocess.py):

1. Paralelismo OUTER (combos LR × seed) con ProcessPoolExecutor.
2. OMP_NUM_THREADS=1 en cada worker — evita contención BLAS con 8 procesos.
3. train_perceptron se importa como función — sin overhead de subprocess.
4. N_WORKERS=8 por default (M1 8 perf cores).

Salida en output/aprendizaje_<ts>/:
- epoch_history.csv: una fila por (lr, seed, epoch) con mse_train y r2_train.
- summary.csv:       por LR, mean ± std del último epoch sobre seeds.
- parameters.csv:    tabla con los hiperparámetros de cada combo (lr, seed, ...).
- config.json:       snapshot del sweep config.

Uso:
    python aprendizaje_sweep.py \\
        --sweep-config configs/aprendizaje.json \\
        --csv "../../data and documentation/fraud_dataset.csv"
"""

from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from nonlinear_perceptron import (  # noqa: E402
    train_perceptron, fit_normalizer, apply_normalizer,
)

N_WORKERS_DEFAULT = 8
DEFAULT_SWEEP = HERE / "configs" / "aprendizaje.json"
DEFAULT_CSV = (HERE / ".." / ".." / "data and documentation" /
               "fraud_dataset.csv").resolve()


def _prepare_data(csv_path: Path, sweep: dict
                  ) -> tuple[pd.DataFrame, list[str], float]:
    df = pd.read_csv(csv_path)
    target_col = sweep["target_col"]
    eval_col = sweep["eval_col"]
    exclude = set(sweep["exclude_features"])
    reserved = {target_col, eval_col} | exclude
    feature_cols = [c for c in df.columns if c not in reserved]

    means, stds = fit_normalizer(df, feature_cols)
    df_norm = apply_normalizer(df, means, stds, feature_cols)
    var_z = float(df[target_col].var(ddof=0))
    return df_norm, feature_cols, var_z


def run_combo(csv_path_str: str, sweep_json: str, lr: float, seed: int):
    sweep = json.loads(sweep_json)
    df_norm, feature_cols, var_z = _prepare_data(Path(csv_path_str), sweep)

    # train_perceptron del no-lineal devuelve 3 cosas; df_test=None.
    weights, mse_history, _ = train_perceptron(
        df_norm, feature_cols, sweep["target_col"],
        learning_rate=lr, epochs=sweep["epochs"],
        epsilon=sweep["epsilon"], seed=seed,
        df_test=None,
    )
    r2_history = [1.0 - mse / var_z for mse in mse_history]
    return lr, seed, mse_history, r2_history, len(feature_cols)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    args = parser.parse_args()

    sweep = json.loads(args.sweep_config.read_text())
    sweep_json = json.dumps(sweep)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = HERE / "output" / f"aprendizaje_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.sweep_config, out_dir / "config.json")

    jobs = [(str(args.csv), sweep_json, lr, seed)
            for lr in sweep["learning_rates"]
            for seed in sweep["seeds"]]
    total = len(jobs)
    print(f"=== APRENDIZAJE NO-LINEAL (sigmoide) ===")
    print(f"CSV: {args.csv}")
    print(f"LRs: {sweep['learning_rates']}  ·  seeds: {sweep['seeds']}  ·  "
          f"epochs: {sweep['epochs']}  ·  epsilon: {sweep['epsilon']}")
    print(f"Total combos: {total}  ·  workers: {args.workers}  ·  "
          f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
    print()

    t0 = time.time()
    rows = []
    n_features_seen = None
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_combo, *job): job for job in jobs}
        for fut in as_completed(futs):
            lr, seed, mse_hist, r2_hist, n_features = fut.result()
            n_features_seen = n_features
            for epoch_idx, (mse, r2) in enumerate(zip(mse_hist, r2_hist)):
                rows.append({"lr": lr, "seed": seed, "epoch": epoch_idx,
                             "mse_train_fulldataset": mse,
                             "r2_train_fulldataset": r2})
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[{done:2d}/{total}]  lr={lr:<8g}  seed={seed}  "
                  f"final_mse={mse_hist[-1]:.5f}  "
                  f"final_r2={r2_hist[-1]:+.4f}  "
                  f"·  elapsed={elapsed:5.1f}s  eta={eta:5.1f}s")

    history_df = pd.DataFrame(rows)
    history_df.to_csv(out_dir / "epoch_history.csv", index=False)

    last = (history_df.sort_values(["lr", "seed", "epoch"])
            .groupby(["lr", "seed"]).tail(1))
    summary_rows = []
    for lr in sweep["learning_rates"]:
        sub = last[last["lr"] == lr]
        summary_rows.append({
            "lr": lr,
            "n_seeds": len(sub),
            "epochs": sweep["epochs"],
            "mse_final_mean_seeds": sub["mse_train_fulldataset"].mean(),
            "mse_final_std_seeds":  sub["mse_train_fulldataset"].std(ddof=0),
            "r2_final_mean_seeds":  sub["r2_train_fulldataset"].mean(),
            "r2_final_std_seeds":   sub["r2_train_fulldataset"].std(ddof=0),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    params_rows = []
    for lr in sweep["learning_rates"]:
        for seed in sweep["seeds"]:
            params_rows.append({
                "lr": lr,
                "seed": seed,
                "epochs": sweep["epochs"],
                "epsilon": sweep["epsilon"],
                "training_mode": sweep["training_mode"],
                "activation": sweep["activation"],
                "normalization": sweep["normalization"],
                "target_col": sweep["target_col"],
                "n_features": n_features_seen,
                "n_samples": "full_dataset",
                "split": "none",
            })
    pd.DataFrame(params_rows).to_csv(out_dir / "parameters.csv", index=False)

    print()
    print(f"Wall-clock total: {time.time() - t0:.1f}s")
    print(f"Outputs: {out_dir}")
    print()
    print("=== Summary (mean ± std sobre seeds, último epoch) ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))


if __name__ == "__main__":
    main()
