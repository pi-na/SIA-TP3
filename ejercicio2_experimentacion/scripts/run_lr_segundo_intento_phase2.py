"""Fase 2 del segundo sweep LR — corre solo lo que faltó después del crash.

LRs: 1e-3 (250 ep), 5e-3 (150 ep), 1e-2 (150 ep).  Seeds: 3 por config.
LRs bajos (1e-4, 5e-4) NO se tocan: ya tenemos los datos rescatados.

Diferencias respecto a `run_lr_segundo_intento.py`:
- Lee `epochs_by_lr` del sweep_config (épocas distintas por LR).
- Persiste cada combo a `output_dir/<combo>/...` ANTES de devolverlo al master,
  así un crash no borra nada (lección del crash anterior).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
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

EJ2 = ROOT / "ejercicio2_experimentacion"
CSV_ROOT = ROOT
DEFAULT_SWEEP = EJ2 / "configs" / "sweeps" / "lr_segundo_intento" / "sweep_config_phase2.json"

METRIC_COLS = [
    "total_epochs",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]


def _build_cfg(arch_cfg_path: Path, lr: float, seed: int, epochs: int) -> dict:
    cfg = json.loads(arch_cfg_path.read_text())
    cfg["training"]["optimizer"] = {"name": "sgd", "lr": lr}
    cfg["training"]["early_stopping_patience"] = None
    cfg["training"]["epochs"] = int(epochs)
    cfg["regularization"] = {"l2": 0.0, "dropout": 0.0, "lr_schedule": None, "augmentation": None}
    cfg["split"]["random_seed"] = seed
    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    cfg["model_name"] = f"{arch_name}_{lr_tag}_seed{seed}"
    return cfg


def run_combo(arch_cfg_path_str: str, lr: float, seed: int, epochs: int, out_root_str: str):
    arch_cfg_path = Path(arch_cfg_path_str)
    out_root = Path(out_root_str)
    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    cfg = _build_cfg(arch_cfg_path, lr, seed, epochs)

    combo_dir = out_root / f"{arch_name}_{lr_tag}_seed{seed}"
    combo_dir.mkdir(parents=True, exist_ok=True)

    run_dir = run_experiment(cfg, csv_root=CSV_ROOT, output_dir=combo_dir, workers=1)

    s = pd.read_csv(run_dir / "run_summary.csv")
    s = s[~s["fold"].isin(["mean", "std"])].copy()
    s["fold"] = s["fold"].astype(int)

    h = pd.read_csv(run_dir / "epoch_history.csv")
    return arch_name, lr_tag, lr, seed, s, h


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    sweep = json.loads(args.sweep_config.read_text())
    seeds = sweep["seeds"]
    epochs_by_lr = {float(k): int(v) for k, v in sweep["epochs_by_lr"].items()}
    archs = sweep["arch_configs"]
    out_dir = ROOT / sweep["output_dir"]
    cfg_dir = args.sweep_config.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    for arch_file in archs:
        arch_cfg_path = cfg_dir / arch_file
        for lr, ep in epochs_by_lr.items():
            for seed in seeds:
                jobs.append((str(arch_cfg_path), lr, seed, ep, str(out_dir)))

    total = len(jobs)
    print(f"Total combos: {total}  ·  workers: {args.workers}", flush=True)
    t0 = time.time()
    summaries, histories = [], []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_combo, *job): job for job in jobs}
        for fut in as_completed(futs):
            try:
                arch_name, lr_tag, lr, seed, s, h = fut.result()
            except Exception:
                print(traceback.format_exc(), flush=True)
                raise
            s.insert(0, "arch", arch_name); s.insert(1, "lr", lr); s.insert(2, "seed", seed)
            h.insert(0, "arch", arch_name); h.insert(1, "lr", lr); h.insert(2, "seed", seed)
            summaries.append(s); histories.append(h)
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(f"[{done:2d}/{total}] {arch_name} {lr_tag} seed={seed}  "
                  f"elapsed={elapsed:.1f}s eta={eta:.1f}s", flush=True)

    raw = pd.concat(summaries, ignore_index=True)
    raw.to_csv(out_dir / "raw.csv", index=False)
    history = pd.concat(histories, ignore_index=True)
    history.to_csv(out_dir / "epoch_history.csv", index=False)

    rows = []
    arch_name = raw["arch"].iloc[0]
    for lr in sorted(raw["lr"].unique()):
        sub = raw[(raw["arch"] == arch_name) & (raw["lr"] == lr)]
        row = {"arch": arch_name, "lr": lr, "n_corridas": len(sub), "n_seeds": sub["seed"].nunique()}
        for col in METRIC_COLS:
            if col in sub.columns:
                row[f"{col}_mean_seedsfolds"] = sub[col].mean()
                row[f"{col}_std_seedsfolds"]  = sub[col].std()
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nWall-clock total: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
