"""Paso 1 Ej3: baseline con more_digits.csv sumado.

Corre:
  (A) CV interno: 3 seeds x 5 folds con la config baseline_ej3.json
  (B) final_eval x 3 seeds sobre digits_test.csv

Outputs en ejercicio3/output/baseline/ y ejercicio3/output/final_eval/baseline/
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_cv_paralelo import run_cells  # noqa: E402

CONFIG_PATH = ROOT / "ejercicio3" / "configs" / "baseline_ej3.json"
FINAL_CONFIG_PATH = ROOT / "ejercicio3" / "configs" / "final_config_ej3_baseline.json"
OUT_CV     = ROOT / "ejercicio3" / "output" / "baseline"
OUT_FINAL  = ROOT / "ejercicio3" / "output" / "final_eval" / "baseline"

SEEDS = [42, 7, 13]


def step_A_cv():
    """(A) CV interno: 3 seeds x 5 folds = 15 corridas."""
    base_cfg = json.loads(CONFIG_PATH.read_text())
    cells = []
    for seed in SEEDS:
        cfg = json.loads(json.dumps(base_cfg))  # deep copy
        cfg["split"]["random_seed"] = seed
        cfg["model_name"] = f"baseline_ej3_seed{seed}"
        cells.append({"cell_id": f"baseline_ej3_seed{seed}", "cfg": cfg})
    n_ok, n_fail = run_cells(cells, OUT_CV, workers=3, label="baseline_cv")
    print(f"\n[step A] CV done: {n_ok} ok, {n_fail} fail")
    return n_ok, n_fail


def step_B_final():
    """(B) final_eval: 3 seeds, full train + eval en digits_test.csv."""
    OUT_FINAL.mkdir(parents=True, exist_ok=True)
    base_cfg = json.loads(FINAL_CONFIG_PATH.read_text())
    print(f"\n[step B] final_eval x {len(SEEDS)} seeds")
    t0 = time.time()
    for seed in SEEDS:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["split"]["random_seed"] = seed
        cfg["model_name"] = f"final_ej3_baseline_seed{seed}"
        cfg_path = OUT_FINAL / f"config_seed{seed}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log_path = OUT_FINAL / f"log_seed{seed}.txt"
        print(f"  seed={seed} starting...", flush=True)
        ts = time.time()
        with open(log_path, "w") as lf:
            r = subprocess.run([
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "ejercicio3" / "final_eval.py"),
                "--config", str(cfg_path),
                "--output-dir", str(OUT_FINAL),
            ], stdout=lf, stderr=subprocess.STDOUT)
        print(f"  seed={seed} done in {time.time()-ts:.0f}s rc={r.returncode}",
              flush=True)
    print(f"[step B] final_eval done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    t0 = time.time()
    step_A_cv()
    step_B_final()
    print(f"\n[step 1 baseline] TOTAL: {time.time()-t0:.0f}s")
