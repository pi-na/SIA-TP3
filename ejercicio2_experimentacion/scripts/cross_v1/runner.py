"""Universal cell-list runner para cross_v1.

Toma una lista de cells (cada una con todos sus hiperparámetros explícitos)
y las corre en paralelo con ProcessPoolExecutor (paralelismo OUTER, k-folds
secuenciales adentro de cada worker, OMP=1).

Robustez:
- Cada cell persiste su raw output a output/<stage>/<cell_id>/ ANTES de devolver
  el summary al master, así un crash no borra nada.
- Errores por cell se loguean en errors.log y NO matan el pipeline.
- run_cells() devuelve (n_ok, n_fail) y el caller decide qué hacer.
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

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
from mlp.train import run_experiment  # noqa: E402

CSV_ROOT = ROOT


def _cell_to_cfg(cell: dict) -> dict:
    """Construye el dict de config para una cell. Cell debe contener todos los
    hiperparams explícitos."""
    cfg = json.loads(Path(cell["arch_config_path"]).read_text())
    cfg["training"]["optimizer"] = {"name": cell["optimizer"], "lr": float(cell["lr"])}
    cfg["training"]["epochs"] = int(cell["max_epochs"])
    cfg["training"]["batch_size"] = int(cell["batch_size"])
    cfg["training"]["early_stopping_patience"] = int(cell["patience"]) if cell.get("patience") else None
    cfg["regularization"] = {"l2": 0.0, "dropout": 0.0, "lr_schedule": None, "augmentation": None}
    cfg["split"]["random_seed"] = int(cell["seed"])
    cfg["split"]["k_folds"] = 5
    cfg["split"]["stratify"] = True
    cfg["preprocessing"] = {"normalization": "zscore", "one_hot_targets": True}
    cfg["model_name"] = cell["id"]
    return cfg


def run_one_cell(cell: dict, out_root_str: str) -> dict:
    """Worker. Corre una cell y persiste a disco. Devuelve summary parcial."""
    out_root = Path(out_root_str)
    cell_dir = out_root / cell["id"]
    cell_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cell_to_cfg(cell)
    run_dir = run_experiment(cfg, csv_root=CSV_ROOT, output_dir=cell_dir, workers=1)
    s = pd.read_csv(run_dir / "run_summary.csv")
    s = s[~s["fold"].isin(["mean", "std"])].copy()
    s["fold"] = s["fold"].astype(int)
    h = pd.read_csv(run_dir / "epoch_history.csv")
    s_path = cell_dir / "summary.csv"
    h_path = cell_dir / "history.csv"
    s.to_csv(s_path, index=False)
    h.to_csv(h_path, index=False)
    return {"id": cell["id"], "summary_path": str(s_path), "history_path": str(h_path)}


def run_cells(cells: list[dict], out_dir: Path, workers: int = 8,
              status_file: Path | None = None, label: str = "stage") -> tuple[int, int]:
    """Corre una lista de cells en paralelo. Devuelve (n_ok, n_fail)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    errors_log = out_dir / "errors.log"
    total = len(cells)
    print(f"[{label}] starting {total} cells with {workers} workers", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_one_cell, c, str(out_dir)): c for c in cells}
        for fut in as_completed(futs):
            cell = futs[fut]
            try:
                fut.result()
                n_ok += 1
            except Exception:
                n_fail += 1
                with errors_log.open("a") as f:
                    f.write(f"=== {cell['id']} ===\n")
                    f.write(traceback.format_exc())
                    f.write("\n")
            done = n_ok + n_fail
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done else 0
            line = (f"[{label}] {done:3d}/{total}  ok={n_ok} fail={n_fail}  "
                    f"last={cell['id']}  elapsed={elapsed:.0f}s eta={eta:.0f}s")
            print(line, flush=True)
            if status_file is not None:
                status_file.write_text(line + "\n")
    print(f"[{label}] DONE  ok={n_ok}  fail={n_fail}  wall={time.time()-t0:.1f}s", flush=True)
    return n_ok, n_fail


def consolidate(out_dir: Path) -> None:
    """Junta todos los summary/history en raw.csv y epoch_history.csv."""
    raws, hists = [], []
    for d in sorted(out_dir.iterdir()):
        if not d.is_dir():
            continue
        sp = d / "summary.csv"
        hp = d / "history.csv"
        if sp.exists() and hp.exists():
            raws.append(pd.read_csv(sp))
            hists.append(pd.read_csv(hp))
    if not raws:
        print(f"[consolidate] WARN: no summaries in {out_dir}", flush=True)
        return
    pd.concat(raws, ignore_index=True).to_csv(out_dir / "raw.csv", index=False)
    pd.concat(hists, ignore_index=True).to_csv(out_dir / "epoch_history.csv", index=False)
    print(f"[consolidate] wrote {out_dir/'raw.csv'} ({len(raws)} cells)", flush=True)
