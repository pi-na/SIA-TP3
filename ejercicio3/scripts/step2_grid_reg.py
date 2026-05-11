"""Paso 2 Ej3: grid de regularizacion (L2 x gaussian_noise).

Sin dropout, sin LR schedule. Sobre el centro shallow + Adam@1e-3 + bs64
con more_digits sumado.

Grid:
  L2    = [0, 1e-5, 1e-4, 1e-3]
  sigma = [0, 0.03, 0.1, 0.2]
Total: 16 combinaciones x 3 seeds = 48 cells (cada cell hace k=5 internamente)

La celda (l2=0, sigma=0) se OMITE porque coincide con el baseline del paso 1.
Quedan 15 combinaciones nuevas x 3 seeds = 45 cells.

Outputs en ejercicio3/output/grid_reg/<cell_id>/.
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

BASE_CONFIG = ROOT / "ejercicio3" / "configs" / "baseline_ej3.json"
OUT_DIR     = ROOT / "ejercicio3" / "output" / "grid_reg"

SEEDS = [42, 7, 13]
L2_VALUES    = [0.0, 1e-5, 1e-4, 1e-3]
SIGMA_VALUES = [0.0, 0.03, 0.1, 0.2]


def _l2_label(l2: float) -> str:
    if l2 == 0.0:
        return "0"
    return f"{l2:.0e}".replace("e-0", "e-")


def _sigma_label(s: float) -> str:
    if s == 0.0:
        return "0"
    return f"{s:.2f}".rstrip("0").rstrip(".")


def build_cells() -> list[dict]:
    base_cfg = json.loads(BASE_CONFIG.read_text())
    cells = []
    for l2 in L2_VALUES:
        for sigma in SIGMA_VALUES:
            if l2 == 0.0 and sigma == 0.0:
                # baseline ya esta en paso 1; lo replicamos para tener una
                # corrida con el mismo runner y poder agregar en la misma tabla
                pass
            for seed in SEEDS:
                cfg = json.loads(json.dumps(base_cfg))
                cfg["split"]["random_seed"] = seed
                cfg["regularization"]["l2"] = l2
                if sigma > 0.0:
                    cfg["regularization"]["augmentation"] = {
                        "type": "gaussian_noise", "sigma": sigma
                    }
                else:
                    cfg["regularization"]["augmentation"] = None
                cell_id = (f"l2_{_l2_label(l2)}_sigma_{_sigma_label(sigma)}"
                           f"_seed{seed}")
                cfg["model_name"] = cell_id
                cells.append({"cell_id": cell_id, "cfg": cfg})
    return cells


def main():
    cells = build_cells()
    print(f"Grid: {len(L2_VALUES)} x {len(SIGMA_VALUES)} = "
          f"{len(L2_VALUES)*len(SIGMA_VALUES)} combos x {len(SEEDS)} seeds = "
          f"{len(cells)} cells\n")
    t0 = time.time()
    n_ok, n_fail = run_cells(cells, OUT_DIR, workers=8, label="grid_reg")
    print(f"\n[step 2 grid_reg] TOTAL: {time.time()-t0:.0f}s "
          f"(ok={n_ok}, fail={n_fail})")


if __name__ == "__main__":
    main()
