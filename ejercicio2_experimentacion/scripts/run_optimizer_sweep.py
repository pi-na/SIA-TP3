"""Runner del sweep de optimizadores — Ejercicio 2.

Lee configs/sweeps/optimizer/sweep_config.json, corre cada combinación
(optimizador × lr × seed) con 5 folds, y guarda raw.csv y epoch_history.csv.

Para generar los plots después:
    python ejercicio2/plot_optimizer_sweep.py

Uso:
    python ejercicio2/run_optimizer_sweep.py [--sweep-config PATH]
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

ROOT     = Path(__file__).resolve().parent.parent
EJ2      = ROOT / "ejercicio2"
CSV_ROOT = ROOT

DEFAULT_SWEEP = EJ2 / "configs" / "sweeps" / "optimizer" / "sweep_config.json"

METRIC_COLS = [
    "total_epochs",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]


def run_one(arch_cfg_path: Path, opt_cfg: dict, lr: float, seed: int,
            epochs: int, tmp_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = json.loads(arch_cfg_path.read_text())
    cfg["training"]["optimizer"] = {**opt_cfg, "lr": lr}
    cfg["training"]["early_stopping_patience"] = None
    cfg["training"]["epochs"] = epochs
    cfg["split"]["random_seed"] = seed

    opt_name = opt_cfg["name"]
    lr_tag   = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    cfg["model_name"] = f"opt_{opt_name}_{lr_tag}_seed{seed}"

    tmp_cfg = tmp_root / f"{opt_name}_{lr_tag}_seed{seed}.json"
    tmp_cfg.write_text(json.dumps(cfg, indent=2))

    out_dir = tmp_root / "runs"
    out_dir.mkdir(exist_ok=True)

    proc = subprocess.run(
        [sys.executable, "-m", "mlp.train",
         "--config", str(tmp_cfg),
         "--csv-root", str(CSV_ROOT),
         "--output-dir", str(out_dir)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Falló {opt_name} lr={lr} seed={seed}:\n{proc.stderr}"
        )

    run_dirs = sorted(out_dir.glob(f"{cfg['model_name']}_*"))
    if not run_dirs:
        raise RuntimeError(f"No se encontró output para {opt_name} lr={lr} seed={seed}")
    run_dir = run_dirs[-1]

    summary_df = pd.read_csv(run_dir / "run_summary.csv")
    summary_df = summary_df[~summary_df["fold"].isin(["mean", "std"])].copy()
    summary_df["fold"] = summary_df["fold"].astype(int)

    history_df = pd.read_csv(run_dir / "epoch_history.csv")

    return summary_df, history_df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    args = parser.parse_args()

    sweep    = json.loads(args.sweep_config.read_text())
    seeds    = sweep["seeds"]
    epochs   = sweep["epochs"]
    lrs      = sweep["learning_rates"]
    opts     = sweep["optimizers"]
    out_dir  = ROOT / sweep["output_dir"]
    cfg_dir  = args.sweep_config.parent
    arch_cfg = cfg_dir / sweep["arch_config"]

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="opt_sweep_"))

    all_summary_rows = []
    all_history_rows = []

    total = len(opts) * len(lrs) * len(seeds)
    done  = 0

    try:
        for opt_cfg in opts:
            opt_name = opt_cfg["name"]
            for lr in lrs:
                lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
                print(f"\n[{opt_name}  {lr_tag}]  5 seeds × 5 folds …")
                for seed in seeds:
                    print(f"  seed={seed} …", end=" ", flush=True)
                    summary_df, history_df = run_one(arch_cfg, opt_cfg, lr, seed, epochs, tmp_root)

                    summary_df.insert(0, "optimizer", opt_name)
                    summary_df.insert(1, "lr", lr)
                    summary_df.insert(2, "seed", seed)
                    all_summary_rows.append(summary_df)

                    history_df.insert(0, "optimizer", opt_name)
                    history_df.insert(1, "lr", lr)
                    history_df.insert(2, "seed", seed)
                    all_history_rows.append(history_df)

                    done += 1
                    print(f"ok  ({done}/{total})")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    raw = pd.concat(all_summary_rows, ignore_index=True)
    raw.to_csv(out_dir / "raw.csv", index=False)
    print(f"\nGuardado: {out_dir / 'raw.csv'}")

    history = pd.concat(all_history_rows, ignore_index=True)
    history.to_csv(out_dir / "epoch_history.csv", index=False)
    print(f"Guardado: {out_dir / 'epoch_history.csv'}")

    rows = []
    for opt_cfg in opts:
        opt_name = opt_cfg["name"]
        for lr in lrs:
            sub = raw[(raw["optimizer"] == opt_name) & (raw["lr"] == lr)]
            row = {"optimizer": opt_name, "lr": lr, "n_corridas": len(sub)}
            for col in METRIC_COLS:
                if col in sub.columns:
                    row[f"{col}_mean"] = sub[col].mean()
                    row[f"{col}_std"]  = sub[col].std()
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Guardado: {out_dir / 'summary.csv'}")
    print("\nResumen:")
    print(summary[["optimizer", "lr", "val_acc_final_mean", "val_acc_final_std",
                   "macro_f1_mean", "macro_f1_std",
                   "val_loss_final_mean"]].to_string(index=False))
    print("\nPara generar los plots: python ejercicio2/plot_optimizer_sweep.py")


if __name__ == "__main__":
    main()
