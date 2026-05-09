"""Runner del estudio de generalización — lineal vs no-lineal.

Lee analisis_generalizacion/config.json, corre 5 seeds × 5 folds = 25 corridas
para cada perceptrón al mejor LR, y guarda raw_history.csv con las curvas de
MSE por época.

Para generar el plot después:
    python plot_generalizacion.py

Uso:
    python run_generalizacion.py [--config PATH]
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

ROOT        = Path(__file__).resolve().parent
CSV_PATH    = ROOT.parent / "data and documentation" / "fraud_dataset.csv"
DEFAULT_CFG = ROOT / "analisis_generalizacion" / "config.json"
OUT_DIR     = ROOT / "analisis_generalizacion"

SCRIPTS = {
    "linear":    ROOT / "lineal_perceptron"    / "linear_perceptron.py",
    "nonlinear": ROOT / "nonlinear_perceptron" / "nonlinear_perceptron.py",
}


def make_run_config(base_cfg_path: Path, seed: int, epochs: int,
                    k_folds: int, model_name: str) -> Path:
    cfg = json.loads(base_cfg_path.read_text())
    cfg["random_seed"]        = seed
    cfg["training"]["epochs"] = epochs
    cfg["k_folds"]            = k_folds
    cfg["model_name"]         = model_name
    tmp = Path(tempfile.mkstemp(suffix=".json", prefix="gen_")[1])
    tmp.write_text(json.dumps(cfg, indent=2))
    return tmp


def run_seed(perceptron: str, base_cfg_path: Path, seed: int,
             epochs: int, k_folds: int, tmp_root: Path) -> Path:
    model_name = f"gen_{perceptron}_seed{seed}"
    out_dir = tmp_root / perceptron / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = make_run_config(base_cfg_path, seed, epochs, k_folds, model_name)
    cmd = [sys.executable, str(SCRIPTS[perceptron]),
           "--config", str(cfg_path),
           "--csv",    str(CSV_PATH),
           "--output-dir", str(out_dir)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cfg_path.unlink(missing_ok=True)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{perceptron} seed={seed} falló:\n{proc.stdout}\n{proc.stderr}")

    subdirs = list(out_dir.iterdir())
    assert len(subdirs) == 1, f"Esperaba 1 subdir, encontré {subdirs}"
    return subdirs[0] / "mse_history.csv"


def collect_all(exp: dict, tmp_root: Path) -> pd.DataFrame:
    seeds   = exp["seeds"]
    epochs  = exp["epochs"]
    k_folds = exp["k_folds"]
    frames  = []

    for perceptron, pcfg in exp["perceptrons"].items():
        base_cfg_path = (DEFAULT_CFG.parent / pcfg["base_config"]).resolve()
        print(f"\n[{perceptron}] {len(seeds)} seeds × {k_folds} folds …")
        for seed in seeds:
            print(f"  seed={seed} ...", end=" ", flush=True)
            hist_path = run_seed(perceptron, base_cfg_path, seed,
                                 epochs, k_folds, tmp_root)
            df = pd.read_csv(hist_path)
            df["perceptron"] = perceptron
            df["seed"]       = seed
            frames.append(df)
            print("ok")

    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CFG)
    args = parser.parse_args()

    exp = json.loads(args.config.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(tempfile.mkdtemp(prefix="gen_run_"))
    try:
        df = collect_all(exp, tmp_root)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out = OUT_DIR / "raw_history.csv"
    df.to_csv(out, index=False)
    print(f"\nGuardado: {out}")
    print(f"Filas: {len(df)}  |  Columnas: {list(df.columns)}")
    print("\nPara generar el plot: python plot_generalizacion.py")


if __name__ == "__main__":
    main()
