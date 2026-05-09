"""K-fold sweep — perceptrón lineal.

Lee la configuración del experimento desde config.json y corre el perceptrón
con cada K en k_values. Guarda raw.csv y summary.csv.

Para generar el plot:
    python plot_kfold_sweep.py --perceptron linear

Uso:
    python kfold_sweep_linear.py [--config PATH]

Config por defecto: lineal_perceptron/analisis_outputs/kfold_sweep/config.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT       = Path(__file__).resolve().parent
CSV_PATH   = ROOT.parent / "data and documentation" / "fraud_dataset.csv"
SCRIPT     = ROOT / "lineal_perceptron" / "linear_perceptron.py"
DEFAULT_CFG = ROOT / "lineal_perceptron" / "analisis_outputs" / "kfold_sweep" / "config.json"
OUT_ROOT   = ROOT / "lineal_perceptron" / "analisis_outputs" / "kfold_sweep"
RUNS_DIR   = OUT_ROOT / "_runs"

METRIC_COLS = ["mse_test", "accuracy", "precision", "recall", "f1"]


def make_run_config(base_cfg_path: Path, k: int, seed: int, epochs: int,
                    threshold: float, model_name: str) -> Path:
    cfg = json.loads(base_cfg_path.read_text())
    cfg["k_folds"]                    = k
    cfg["random_seed"]                = seed
    cfg["model_name"]                 = model_name
    cfg["training"]["epochs"]         = epochs
    cfg["evaluation"]["threshold"]    = threshold
    tmp = Path(tempfile.mkstemp(suffix=".json", prefix=f"kfold_k{k}_")[1])
    tmp.write_text(json.dumps(cfg, indent=2))
    return tmp


def run_k(base_cfg_path: Path, k: int, seed: int, epochs: int,
          threshold: float) -> Path:
    model_name = f"kfold_sweep_k{k}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    target_dir = RUNS_DIR / model_name
    if target_dir.exists():
        shutil.rmtree(target_dir)

    cfg_path = make_run_config(base_cfg_path, k, seed, epochs, threshold, model_name)
    cmd = [sys.executable, str(SCRIPT),
           "--config", str(cfg_path),
           "--csv",    str(CSV_PATH),
           "--output-dir", str(RUNS_DIR)]
    print(f"  K={k} ...", end=" ", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cfg_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(f"K={k} falló:\n{proc.stdout}\n{proc.stderr}")

    generated = sorted(RUNS_DIR.glob(f"{model_name}_*"))
    if not generated:
        raise RuntimeError(f"No se encontró output para K={k}")
    generated[-1].rename(target_dir)
    print("ok")
    return target_dir


def collect_metrics(k: int, run_dir: Path) -> pd.DataFrame:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    metrics = metrics[~metrics["fold"].isin(["mean", "std"])].copy()
    metrics["fold"] = metrics["fold"].astype(int)
    metrics["k"] = k
    metrics["positivos_test"] = metrics["tp"] + metrics["fn"]
    return metrics[["k", "fold", "n_train", "n_test", "positivos_test"] + METRIC_COLS]


def summary_table(df: pd.DataFrame, k_values: list[int]) -> pd.DataFrame:
    rows = []
    for k in k_values:
        sub = df[df["k"] == k]
        row = {"k": k,
               "n_folds": len(sub),
               "n_train_mean": sub["n_train"].mean(),
               "n_test_mean":  sub["n_test"].mean(),
               "positivos_test_mean": sub["positivos_test"].mean()}
        for m in METRIC_COLS:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_std"]  = sub[m].std(ddof=0)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    args = parser.parse_args()

    exp = json.loads(args.config.read_text())
    base_cfg_path = (args.config.parent / exp["base_config"]).resolve()
    k_values  = exp["k_values"]
    epochs    = exp["epochs"]
    seed      = exp["seed"]
    threshold = exp["threshold"]

    print(f"=== K-fold sweep — lineal (K ∈ {k_values}, seed={seed}, "
          f"epochs={epochs}, thr={threshold}) ===")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    frames = []
    for k in k_values:
        run_dir = run_k(base_cfg_path, k, seed, epochs, threshold)
        frames.append(collect_metrics(k, run_dir))

    df      = pd.concat(frames, ignore_index=True)
    summary = summary_table(df, k_values)

    df.to_csv(OUT_ROOT / "raw.csv", index=False)
    summary.to_csv(OUT_ROOT / "summary.csv", index=False)
    print(f"\nGuardado: {OUT_ROOT / 'raw.csv'}")
    print(f"Guardado: {OUT_ROOT / 'summary.csv'}")
    print("\nPara generar el plot: python plot_kfold_sweep.py --perceptron linear")
    print(summary[["k", "mse_test_mean", "mse_test_std", "f1_mean", "f1_std"]].to_string(index=False))


if __name__ == "__main__":
    main()
