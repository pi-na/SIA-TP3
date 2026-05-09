"""Runner del sweep de arquitecturas — Ejercicio 2.

Lee configs/sweeps/arch/sweep_config.json, corre cada arquitectura con
5 seeds × 5 folds = 25 corridas, y guarda raw.csv y summary.csv.

Para generar los plots después:
    python ejercicio2/plot_arch_sweep.py

Uso:
    python ejercicio2/run_arch_sweep.py [--sweep-config PATH]
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

ROOT       = Path(__file__).resolve().parent.parent   # raíz del repo
EJ2        = ROOT / "ejercicio2"
CSV_ROOT   = ROOT

DEFAULT_SWEEP = EJ2 / "configs" / "sweeps" / "arch" / "sweep_config.json"

METRIC_COLS = [
    "total_epochs", "best_epoch",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]


def run_one(arch_cfg_path: Path, seed: int, tmp_root: Path) -> pd.DataFrame:
    """Corre una arquitectura con una seed. Devuelve run_summary como DataFrame."""
    cfg = json.loads(arch_cfg_path.read_text())
    cfg["split"]["random_seed"] = seed
    cfg["model_name"] = f"{cfg['model_name']}_seed{seed}"

    tmp_cfg = tmp_root / f"{arch_cfg_path.stem}_seed{seed}.json"
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
            f"Falló {arch_cfg_path.stem} seed={seed}:\n{proc.stderr}"
        )

    run_dirs = sorted(out_dir.glob(f"{cfg['model_name']}_*"))
    if not run_dirs:
        raise RuntimeError(f"No se encontró output para {arch_cfg_path.stem} seed={seed}")
    summary_path = run_dirs[-1] / "run_summary.csv"
    df = pd.read_csv(summary_path)
    df = df[~df["fold"].isin(["mean", "std"])].copy()
    df["fold"] = df["fold"].astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-config", type=Path, default=DEFAULT_SWEEP)
    args = parser.parse_args()

    sweep = json.loads(args.sweep_config.read_text())
    seeds        = sweep["seeds"]
    arch_files   = sweep["configs"]
    out_dir      = ROOT / sweep["output_dir"]
    cfg_dir      = args.sweep_config.parent

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="arch_sweep_"))

    all_rows = []
    try:
        for arch_file in arch_files:
            arch_cfg_path = cfg_dir / arch_file
            arch_name = arch_cfg_path.stem
            print(f"\n[{arch_name}]  5 seeds × 5 folds …")
            for seed in seeds:
                print(f"  seed={seed} …", end=" ", flush=True)
                df = run_one(arch_cfg_path, seed, tmp_root)
                df.insert(0, "arch", arch_name)
                df.insert(1, "seed", seed)
                all_rows.append(df)
                print("ok")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    raw = pd.concat(all_rows, ignore_index=True)
    raw.to_csv(out_dir / "raw.csv", index=False)
    print(f"\nGuardado: {out_dir / 'raw.csv'}")

    # summary: mean ± std por arquitectura sobre las 25 corridas
    rows = []
    for arch in [Path(f).stem for f in arch_files]:
        sub = raw[raw["arch"] == arch]
        row = {"arch": arch, "n_corridas": len(sub)}
        for col in METRIC_COLS:
            if col in sub.columns:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_std"]  = sub[col].std()
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Guardado: {out_dir / 'summary.csv'}")
    print("\nResumen:")
    print(summary[["arch", "val_acc_final_mean", "val_acc_final_std",
                   "macro_f1_mean", "macro_f1_std",
                   "val_loss_final_mean", "best_epoch_mean"]].to_string(index=False))
    print("\nPara generar los plots: python ejercicio2/plot_arch_sweep.py")


if __name__ == "__main__":
    main()
