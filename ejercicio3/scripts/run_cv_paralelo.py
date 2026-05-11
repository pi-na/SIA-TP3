"""Runner paralelo para Ej3: corre una lista de configs en paralelo.

Cada config se identifica por un cell_id. El worker corre run_experiment
(CV interno con k=5 dentro del worker, paralelismo OUTER entre cells).

Uso programatico: from ejercicio3.scripts.run_cv_paralelo import run_cells

Cells = lista de dicts con: {"cell_id": str, "cfg": dict} donde cfg ya tiene
todos los hiperparametros listos para mlp.train.run_experiment.
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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from mlp.train import run_experiment  # noqa: E402

CSV_ROOT = ROOT


def run_one_cell(args: dict) -> dict:
    """Worker: corre 1 cell (= 1 config con k=5 folds adentro).
    Persiste summary.csv y history.csv en out_dir/<cell_id>/."""
    cell_id  = args["cell_id"]
    cfg      = args["cfg"]
    out_root = Path(args["out_root"])
    cell_dir = out_root / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = run_experiment(cfg, csv_root=CSV_ROOT, output_dir=cell_dir, workers=1)
        s = pd.read_csv(run_dir / "run_summary.csv")
        s = s[~s["fold"].isin(["mean", "std"])].copy()
        s["fold"] = s["fold"].astype(int)
        s.to_csv(cell_dir / "summary.csv", index=False)
        h = pd.read_csv(run_dir / "epoch_history.csv")
        h.to_csv(cell_dir / "history.csv", index=False)
        return {"cell_id": cell_id, "ok": True, "run_dir": str(run_dir)}
    except Exception as e:
        tb = traceback.format_exc()
        (cell_dir / "error.log").write_text(f"{type(e).__name__}: {e}\n\n{tb}")
        return {"cell_id": cell_id, "ok": False, "error": str(e)}


def run_cells(cells: list[dict], out_root: Path, workers: int = 8,
              label: str = "cells") -> tuple[int, int]:
    out_root.mkdir(parents=True, exist_ok=True)
    total = len(cells)
    print(f"[{label}] {total} cells, {workers} workers", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    payloads = [{"cell_id": c["cell_id"], "cfg": c["cfg"], "out_root": str(out_root)}
                for c in cells]
    with ProcessPoolExecutor(max_workers=workers) as exe:
        futs = {exe.submit(run_one_cell, p): p["cell_id"] for p in payloads}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            if r["ok"]:
                n_ok += 1
            else:
                n_fail += 1
                with (out_root / "errors.log").open("a") as f:
                    f.write(f"{r['cell_id']}: {r['error']}\n")
            elapsed = time.time() - t0
            eta = elapsed / i * (total - i) if i > 0 else 0
            print(f"[{label}] {i}/{total} done (ok={n_ok}, fail={n_fail}) "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    print(f"[{label}] DONE in {time.time()-t0:.0f}s (ok={n_ok}, fail={n_fail})",
          flush=True)
    return n_ok, n_fail
