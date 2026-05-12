"""Final eval para rot10 y rot15 (Experimento A del Ej3).

Para cada uno de los 2 configs, corre final_eval.py con 3 seeds (42, 7, 13)
sobre digits_test.csv. NO se toca el test durante la búsqueda de HP — esto
es producción.

Outputs en ejercicio3/output/final_eval/rotation_aug/.
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
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

CONFIGS = [
    ("rot10", ROOT / "ejercicio3" / "configs" / "rotation_aug" / "rot10.json"),
    ("rot15", ROOT / "ejercicio3" / "configs" / "rotation_aug" / "rot15.json"),
]
SEEDS = [42, 7, 13]
OUT_BASE = ROOT / "ejercicio3" / "output" / "final_eval" / "rotation_aug"
FINAL_EVAL = ROOT / "ejercicio3" / "final_eval.py"


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs = []
    for tag, cfg_path in CONFIGS:
        base_cfg = json.loads(cfg_path.read_text())
        for seed in SEEDS:
            cfg = json.loads(json.dumps(base_cfg))
            cfg["split"]["random_seed"] = seed
            cfg["model_name"] = f"{tag}_seed{seed}"

            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                json.dump(cfg, f, indent=2)
                tmp_path = Path(f.name)

            out_dir = OUT_BASE / tag
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"--- {tag} seed={seed} ---", flush=True)
            t1 = time.time()
            res = subprocess.run(
                [sys.executable, str(FINAL_EVAL),
                 "--config", str(tmp_path),
                 "--csv-root", str(ROOT),
                 "--output-dir", str(out_dir)],
                capture_output=True, text=True,
            )
            elapsed = time.time() - t1
            tmp_path.unlink(missing_ok=True)
            if res.returncode != 0:
                print(f"FAILED ({elapsed:.0f}s):\n{res.stderr}", flush=True)
                runs.append((tag, seed, False))
            else:
                # parse last "Test accuracy:" line from stdout for quick view
                for line in res.stdout.splitlines():
                    if "Test accuracy" in line or "Test macro" in line:
                        print(f"  {line}", flush=True)
                print(f"  done in {elapsed:.0f}s", flush=True)
                runs.append((tag, seed, True))

    n_ok = sum(1 for _, _, ok in runs if ok)
    print(f"\n[final_eval rotation_aug] TOTAL: {time.time()-t0:.0f}s "
          f"(ok={n_ok}/{len(runs)})")


if __name__ == "__main__":
    main()
