"""Runner del segundo sweep de learning rates — Ejercicio 2.

Diferencias respecto al primer sweep LR (`run_lr_sweep.py`):

- **Una sola arquitectura**: la óptima del arch sweep (`arch_shallow`,
  784→128→10). Justificación en `Notas/ejercicio 2/Experimentos y analisis/
  Arch/Arquitectura.md`.
- **500 épocas** (vs 50). Con SGD básico y LR chicos, 50 no alcanzaba para
  converger (ver `analisis_lr.md` del primer sweep).
- **Sin early stopping, sin regularización** (ya estaba así, lo dejamos
  explícito).
- **Paralelismo OUTER multiprocess** (basado en
  `runner_ejemplo_multiprocess.py`): cada combinación (arch × lr × seed) en
  un proceso, k-folds secuenciales adentro, `OMP_NUM_THREADS=1`.

LRs explorados: los mismos del primer sweep — [1e-4, 5e-4, 1e-3, 5e-3, 1e-2].
Seeds: [42, 7, 13, 21, 99]. K-folds: 5. Total: 1 × 5 × 5 × 5 = 125 corridas
de 500 épocas cada una.

Uso:
    python ejercicio2_experimentacion/scripts/run_lr_segundo_intento.py [--workers 8]
"""

from __future__ import annotations

# IMPORTANTE: estas env vars deben setearse ANTES de importar numpy.
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
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from mlp.train import run_experiment  # noqa: E402

EJ2      = ROOT / "ejercicio2_experimentacion"
CSV_ROOT = ROOT

DEFAULT_SWEEP = EJ2 / "configs" / "sweeps" / "lr_segundo_intento" / "sweep_config.json"
N_WORKERS_DEFAULT = 8

METRIC_COLS = [
    "total_epochs",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]


def _build_cfg_for_combo(arch_cfg_path: Path, lr: float, seed: int) -> dict:
    cfg = json.loads(arch_cfg_path.read_text())
    cfg["training"]["optimizer"] = {"name": "sgd", "lr": lr}
    cfg["training"]["early_stopping_patience"] = None
    cfg["training"]["epochs"] = 500
    cfg["regularization"] = {"l2": 0.0, "dropout": 0.0, "lr_schedule": None, "augmentation": None}
    cfg["split"]["random_seed"] = seed

    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    cfg["model_name"] = f"{arch_name}_{lr_tag}_seed{seed}"
    return cfg


def run_combo(arch_cfg_path_str: str, lr: float, seed: int, tmp_root_str: str
              ) -> tuple[str, str, float, int, pd.DataFrame, pd.DataFrame]:
    arch_cfg_path = Path(arch_cfg_path_str)
    tmp_root = Path(tmp_root_str)
    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")

    cfg = _build_cfg_for_combo(arch_cfg_path, lr, seed)

    out_dir = tmp_root / f"{arch_name}_{lr_tag}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dir = run_experiment(cfg, csv_root=CSV_ROOT, output_dir=out_dir, workers=1)

    summary_df = pd.read_csv(run_dir / "run_summary.csv")
    summary_df = summary_df[~summary_df["fold"].isin(["mean", "std"])].copy()
    summary_df["fold"] = summary_df["fold"].astype(int)

    history_df = pd.read_csv(run_dir / "epoch_history.csv")

    return arch_name, lr_tag, lr, seed, summary_df, history_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT)
    args = parser.parse_args()

    sweep   = json.loads(args.sweep_config.read_text())
    seeds   = sweep["seeds"]
    lrs     = sweep["learning_rates"]
    archs   = sweep["arch_configs"]
    out_dir = ROOT / sweep["output_dir"]
    cfg_dir = args.sweep_config.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="lr2_sweep_"))

    jobs = []
    for arch_file in archs:
        arch_cfg_path = cfg_dir / arch_file
        for lr in lrs:
            for seed in seeds:
                jobs.append((str(arch_cfg_path), lr, seed, str(tmp_root)))

    total = len(jobs)
    print(f"Total combinaciones: {total}  ·  workers: {args.workers}  "
          f"·  OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}", flush=True)
    t0 = time.time()

    all_summary_rows = []
    all_history_rows = []
    done = 0

    try:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(run_combo, *job): job for job in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                try:
                    arch_name, lr_tag, lr, seed, summary_df, history_df = fut.result()
                except Exception:
                    print(f"\n[FALLO] {job[0]}  lr={job[1]}  seed={job[2]}\n"
                          f"{traceback.format_exc()}", flush=True)
                    raise

                summary_df.insert(0, "arch", arch_name)
                summary_df.insert(1, "lr", lr)
                summary_df.insert(2, "seed", seed)
                all_summary_rows.append(summary_df)

                history_df.insert(0, "arch", arch_name)
                history_df.insert(1, "lr", lr)
                history_df.insert(2, "seed", seed)
                all_history_rows.append(history_df)

                done += 1
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"[{done:3d}/{total}]  {arch_name}  {lr_tag}  seed={seed}  "
                      f"·  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s", flush=True)
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    raw = pd.concat(all_summary_rows, ignore_index=True)
    raw.to_csv(out_dir / "raw.csv", index=False)
    print(f"\nGuardado: {out_dir / 'raw.csv'}", flush=True)

    history = pd.concat(all_history_rows, ignore_index=True)
    history.to_csv(out_dir / "epoch_history.csv", index=False)
    print(f"Guardado: {out_dir / 'epoch_history.csv'}", flush=True)

    rows = []
    for arch_file in archs:
        arch_name = Path(arch_file).stem
        for lr in lrs:
            sub = raw[(raw["arch"] == arch_name) & (raw["lr"] == lr)]
            row = {"arch": arch_name, "lr": lr, "n_corridas": len(sub)}
            for col in METRIC_COLS:
                if col in sub.columns:
                    row[f"{col}_mean_seedsfolds"] = sub[col].mean()
                    row[f"{col}_std_seedsfolds"]  = sub[col].std()
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Guardado: {out_dir / 'summary.csv'}", flush=True)
    print(f"\nWall-clock total: {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
