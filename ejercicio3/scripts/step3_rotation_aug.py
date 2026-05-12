"""Paso 3 Ej3: experimento A — augmentación rotacional PURA.

Sobre la baseline +more_digits (paso 1 del Ej3): shallow [784,128,10] +
Adam@1e-3 + batch=64 + ES patience=20, SIN L2 ni σ. Sólo agrega rotación
aleatoria por minibatch a ±max_angle°.

Configs:
  rot10  — max_angle = 10°  (Simard 2003 baseline)
  rot15  — max_angle = 15°  (Cireşan 2010 affine)

Total: 2 configs × 3 seeds = 6 cells (cada cell hace k=5 internamente)
        = 30 corridas CV.

Outputs en ejercicio3/output/rotation_aug/<cell_id>/.
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
from run_cv_paralelo import run_cells  # noqa: E402

CONFIG_DIR  = ROOT / "ejercicio3" / "configs" / "rotation_aug"
OUT_DIR     = ROOT / "ejercicio3" / "output" / "rotation_aug"

SEEDS = [42, 7, 13]
CONFIGS = [
    ("rot10", CONFIG_DIR / "rot10.json"),
    ("rot15", CONFIG_DIR / "rot15.json"),
]


def build_cells() -> list[dict]:
    cells = []
    for tag, cfg_path in CONFIGS:
        base_cfg = json.loads(cfg_path.read_text())
        for seed in SEEDS:
            cfg = json.loads(json.dumps(base_cfg))
            cfg["split"]["random_seed"] = seed
            cell_id = f"{tag}_seed{seed}"
            cfg["model_name"] = cell_id
            cells.append({"cell_id": cell_id, "cfg": cfg})
    return cells


def main():
    cells = build_cells()
    print(f"Experimento A — rotación pura: {len(CONFIGS)} configs × "
          f"{len(SEEDS)} seeds = {len(cells)} cells\n")
    t0 = time.time()
    n_ok, n_fail = run_cells(cells, OUT_DIR, workers=8, label="rotation_aug")
    print(f"\n[step 3 rotation_aug] TOTAL: {time.time()-t0:.0f}s "
          f"(ok={n_ok}, fail={n_fail})")


if __name__ == "__main__":
    main()
