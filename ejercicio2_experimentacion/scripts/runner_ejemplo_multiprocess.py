"""Plantilla de runner paralelo — Ejercicio 2.

Versión paralela del patrón usado en run_lr_sweep.py / run_arch_sweep.py /
run_optimizer_sweep.py. Sirve como base para reescribir esos 3 runners.

Decisiones clave (justificadas en discusión con el grupo):

1. Paralelismo OUTER, no inner.
   Las combinaciones del grid (arch × lr × seed) corren en paralelo en un
   ProcessPoolExecutor. Cada worker ejecuta los k-folds SECUENCIALMENTE
   (workers=1 dentro de run_experiment). Razón: con grids de cientos de
   combinaciones, el paralelismo afuera escala mejor que el de adentro
   (5 folds), y se evita anidar procesos.

2. mlp.train se importa como función, no se invoca por subprocess.
   Esto elimina el costo de arrancar Python + importar numpy una vez por
   combinación. La función relevante es mlp.train.run_experiment.

3. OMP_NUM_THREADS=1 (y MKL/OpenBLAS) en cada worker.
   Si no se limita, cada uno de los 8 workers intenta usar todos los cores
   vía BLAS multi-thread → contención brutal. Para MLPs chicos
   (784→hidden→10) los matmuls son demasiado pequeños para que BLAS
   multi-thread pague: gana "muchos workers, 1 thread c/u".

4. N_WORKERS = 8 por default.
   Pensado para la M1 (8 perf cores). Bajar a 6 si se quiere dejar margen
   para usar la máquina, subir en Ryzen 5800X3D (8 cores físicos también
   → 8 sigue siendo razonable; ojo con confundir threads lógicos).

Uso:
    python ejercicio2_experimentacion/scripts/runner_ejemplo_multiprocess.py \\
        --sweep-config ejercicio2_experimentacion/configs/sweeps/lr/sweep_config.json \\
        [--workers 8]
"""

from __future__ import annotations

# IMPORTANTE: estas env vars deben setearse ANTES de importar numpy. En macOS
# ProcessPoolExecutor usa "spawn" por default, así que los children re-ejecutan
# el módulo desde cero y heredan estos valores antes de su propio import numpy.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS Accelerate
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

# mlp.train importa numpy adentro; al estar las env vars seteadas arriba,
# cualquier subproceso que re-ejecute el módulo las verá a tiempo.
ROOT     = Path(__file__).resolve().parent.parent.parent  # repo root
sys.path.insert(0, str(ROOT))
from mlp.train import run_experiment  # noqa: E402

EJ2      = ROOT / "ejercicio2_experimentacion"
CSV_ROOT = ROOT

DEFAULT_SWEEP = EJ2 / "configs" / "sweeps" / "lr" / "sweep_config.json"
N_WORKERS_DEFAULT = 8

METRIC_COLS = [
    "total_epochs",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]


def _build_cfg_for_combo(arch_cfg_path: Path, lr: float, seed: int) -> dict:
    """Construye el dict de config para una combinación. Mismo patrón que
    el run_one() original pero sin escribir a disco ni lanzar subprocess."""
    cfg = json.loads(arch_cfg_path.read_text())
    cfg["training"]["optimizer"] = {"name": "sgd", "lr": lr}
    cfg["training"]["early_stopping_patience"] = None
    cfg["split"]["random_seed"] = seed

    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    cfg["model_name"] = f"{arch_name}_{lr_tag}_seed{seed}"
    return cfg


def run_combo(arch_cfg_path_str: str, lr: float, seed: int, tmp_root_str: str
              ) -> tuple[str, str, float, int, pd.DataFrame, pd.DataFrame]:
    """Worker: corre UNA combinación (arch, lr, seed) con sus 5 folds
    secuenciales. Devuelve (arch_name, lr_tag, lr, seed, summary_df, history_df).

    Importante: argumentos picklables (str en vez de Path) y todo top-level
    para que ProcessPoolExecutor los pueda spawnear sin problemas.
    """
    arch_cfg_path = Path(arch_cfg_path_str)
    tmp_root = Path(tmp_root_str)
    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")

    cfg = _build_cfg_for_combo(arch_cfg_path, lr, seed)

    # Cada worker escribe en su propio out_dir para evitar colisiones.
    out_dir = tmp_root / f"{arch_name}_{lr_tag}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # workers=1 → folds secuenciales dentro de este worker (paralelismo outer).
    run_dir = run_experiment(cfg, csv_root=CSV_ROOT, output_dir=out_dir, workers=1)

    summary_df = pd.read_csv(run_dir / "run_summary.csv")
    summary_df = summary_df[~summary_df["fold"].isin(["mean", "std"])].copy()
    summary_df["fold"] = summary_df["fold"].astype(int)

    history_df = pd.read_csv(run_dir / "epoch_history.csv")

    return arch_name, lr_tag, lr, seed, summary_df, history_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--workers", type=int, default=N_WORKERS_DEFAULT,
                        help=f"Procesos paralelos sobre el grid (default {N_WORKERS_DEFAULT}).")
    args = parser.parse_args()

    sweep   = json.loads(args.sweep_config.read_text())
    seeds   = sweep["seeds"]
    lrs     = sweep["learning_rates"]
    archs   = sweep["arch_configs"]
    out_dir = ROOT / sweep["output_dir"]
    cfg_dir = args.sweep_config.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="multiproc_sweep_"))

    # Construir lista plana de jobs.
    jobs = []
    for arch_file in archs:
        arch_cfg_path = cfg_dir / arch_file
        for lr in lrs:
            for seed in seeds:
                jobs.append((str(arch_cfg_path), lr, seed, str(tmp_root)))

    total = len(jobs)
    print(f"Total combinaciones: {total}  ·  workers: {args.workers}  "
          f"·  OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")
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
                except Exception as e:
                    print(f"\n[FALLO] {job[0]}  lr={job[1]}  seed={job[2]}\n"
                          f"{traceback.format_exc()}")
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
                      f"·  elapsed={elapsed:6.1f}s  eta={eta:6.1f}s")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # Consolidar CSVs (mismo formato que los runners actuales).
    raw = pd.concat(all_summary_rows, ignore_index=True)
    raw.to_csv(out_dir / "raw.csv", index=False)
    print(f"\nGuardado: {out_dir / 'raw.csv'}")

    history = pd.concat(all_history_rows, ignore_index=True)
    history.to_csv(out_dir / "epoch_history.csv", index=False)
    print(f"Guardado: {out_dir / 'epoch_history.csv'}")

    # summary: mean ± std por (arch, lr) sobre las 25 corridas (5 seeds × 5 folds).
    rows = []
    for arch_file in archs:
        arch_name = Path(arch_file).stem
        for lr in lrs:
            sub = raw[(raw["arch"] == arch_name) & (raw["lr"] == lr)]
            row = {"arch": arch_name, "lr": lr, "n_corridas": len(sub)}
            for col in METRIC_COLS:
                if col in sub.columns:
                    # NOTE: media sobre las 25 corridas (5 seeds × 5 folds).
                    # Si querés varianza solo de seeds, agrupá primero por seed.
                    row[f"{col}_mean_seedsfolds"] = sub[col].mean()
                    row[f"{col}_std_seedsfolds"]  = sub[col].std()
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Guardado: {out_dir / 'summary.csv'}")
    print(f"\nWall-clock total: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
