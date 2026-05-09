"""Runner del sweep de learning rates — Ejercicio 2.

Lee configs/sweeps/lr/sweep_config.json, corre cada combinación
(arquitectura × lr × seed) con 5 folds = 25 corridas por combinación,
y guarda raw.csv y epoch_history.csv en analisis/lr/.

Para generar los plots después:
    python ejercicio2/plot_lr_sweep.py

Uso:
    python ejercicio2/run_lr_sweep.py [--sweep-config PATH]
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

DEFAULT_SWEEP = EJ2 / "configs" / "sweeps" / "lr" / "sweep_config.json"

METRIC_COLS = [
    "total_epochs",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]


def run_one(arch_cfg_path: Path, lr: float, seed: int, tmp_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Corre una arquitectura con un lr y seed dados. Devuelve (summary_df, history_df)."""
    cfg = json.loads(arch_cfg_path.read_text())
    cfg["training"]["optimizer"] = {"name": "sgd", "lr": lr}
    cfg["training"]["early_stopping_patience"] = None
    cfg["split"]["random_seed"] = seed

    arch_name = arch_cfg_path.stem
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    cfg["model_name"] = f"{arch_name}_{lr_tag}_seed{seed}"

    tmp_cfg = tmp_root / f"{arch_name}_{lr_tag}_seed{seed}.json"
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
            f"Falló {arch_name} lr={lr} seed={seed}:\n{proc.stderr}"
        )

    run_dirs = sorted(out_dir.glob(f"{cfg['model_name']}_*"))
    if not run_dirs:
        raise RuntimeError(f"No se encontró output para {arch_name} lr={lr} seed={seed}")
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

    sweep   = json.loads(args.sweep_config.read_text())
    seeds   = sweep["seeds"]
    lrs     = sweep["learning_rates"]
    archs   = sweep["arch_configs"]
    out_dir = ROOT / sweep["output_dir"]
    cfg_dir = args.sweep_config.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="lr_sweep_"))

    all_summary_rows = []
    all_history_rows = []

    total = len(archs) * len(lrs) * len(seeds)
    done  = 0

    try:
        for arch_file in archs:
            arch_cfg_path = cfg_dir / arch_file
            arch_name = arch_cfg_path.stem
            for lr in lrs:
                lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
                print(f"\n[{arch_name}  {lr_tag}]  5 seeds × 5 folds …")
                for seed in seeds:
                    print(f"  seed={seed} …", end=" ", flush=True)
                    summary_df, history_df = run_one(arch_cfg_path, lr, seed, tmp_root)

                    summary_df.insert(0, "arch", arch_name)
                    summary_df.insert(1, "lr", lr)
                    summary_df.insert(2, "seed", seed)
                    all_summary_rows.append(summary_df)

                    history_df.insert(0, "arch", arch_name)
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

    # summary: mean ± std por (arch, lr) sobre las 25 corridas
    rows = []
    for arch_file in archs:
        arch_name = Path(arch_file).stem
        for lr in lrs:
            sub = raw[(raw["arch"] == arch_name) & (raw["lr"] == lr)]
            row = {"arch": arch_name, "lr": lr, "n_corridas": len(sub)}
            for col in METRIC_COLS:
                if col in sub.columns:
                    row[f"{col}_mean"] = sub[col].mean()
                    row[f"{col}_std"]  = sub[col].std()
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Guardado: {out_dir / 'summary.csv'}")
    print("\nResumen (val_acc y macro_f1):")
    print(summary[["arch", "lr", "val_acc_final_mean", "val_acc_final_std",
                   "macro_f1_mean", "macro_f1_std",
                   "val_loss_final_mean"]].to_string(index=False))
    print("\nPara generar los plots: python ejercicio2/plot_lr_sweep.py")


if __name__ == "__main__":
    main()
