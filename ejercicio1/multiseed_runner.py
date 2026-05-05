"""Runner multi-seed para sweep LR (lineal y no-lineal).

Para cada perceptron y cada (lr, seed), corre el script existente con
`epochs=500` (suficiente para convergencia segun single-seed sweep) y
`random_seed` variando. Paraleliza los (lr, seed) externamente.

Cada subprocess usa --workers=1 (folds secuenciales internamente);
la paralelizacion ocurre en el outer loop.

Outputs:
    output/sweep_lr_multiseed_<perceptron>/<lr>_<seed>/  -- runs crudos
    analisis_outputs/sweep_lr/multiseed/                 -- agregados + plots

Uso:
    python multiseed_runner.py --perceptron {linear,nonlinear,both}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT.parent / "data and documentation" / "fraud_dataset.csv"

SEEDS = [42, 7, 13, 21, 99]
EPOCHS_OVERRIDE = 500

PERCEPTRONS = {
    "linear": {
        "script":   ROOT / "lineal_perceptron" / "linear_perceptron.py",
        "configs": {
            "0.001":  ROOT / "lineal_perceptron" / "configs" / "lr_001.json",
            "0.0001": ROOT / "lineal_perceptron" / "configs" / "lr_0001.json",
            "1e-05":  ROOT / "lineal_perceptron" / "configs" / "lr_00001.json",
        },
        "out_root":   ROOT / "lineal_perceptron" / "output" / "sweep_lr_multiseed",
        "analisis":   ROOT / "lineal_perceptron" / "analisis_outputs" / "sweep_lr" / "multiseed",
    },
    "nonlinear": {
        "script":   ROOT / "nonlinear_perceptron" / "nonlinear_perceptron.py",
        "configs": {
            "0.01":   ROOT / "nonlinear_perceptron" / "configs" / "lr_01.json",
            "0.001":  ROOT / "nonlinear_perceptron" / "configs" / "lr_001.json",
            "0.0001": ROOT / "nonlinear_perceptron" / "configs" / "lr_0001.json",
        },
        "out_root":   ROOT / "nonlinear_perceptron" / "output" / "sweep_lr_multiseed",
        "analisis":   ROOT / "nonlinear_perceptron" / "analisis_outputs" / "sweep_lr" / "multiseed",
    },
}


def make_temp_config(base_path: Path, seed: int, epochs: int, model_name: str) -> Path:
    cfg = json.loads(base_path.read_text())
    cfg["random_seed"] = seed
    cfg["training"]["epochs"] = epochs
    cfg["model_name"] = model_name
    tmp = Path(tempfile.mkstemp(suffix=".json", prefix="ms_cfg_")[1])
    tmp.write_text(json.dumps(cfg, indent=2))
    return tmp


def run_one(args) -> tuple[str, str, Path]:
    perceptron, script, base_cfg, lr_label, seed, out_root = args
    model_name = f"{lr_label}_seed{seed}"
    cfg_path = make_temp_config(Path(base_cfg), seed, EPOCHS_OVERRIDE, model_name)
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    target_dir = out_root / model_name
    if target_dir.exists():
        shutil.rmtree(target_dir)

    # Run with workers=1 to leave concurrency to outer pool
    # Cada subprocess usa default workers (k_folds=5 en paralelo internamente).
    # El outer pool corre pocos subprocesses concurrentes para no overcommit.
    cmd = [
        sys.executable, str(script),
        "--config", str(cfg_path),
        "--csv", str(CSV_PATH),
        "--output-dir", str(out_root),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cfg_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"[{perceptron}/{lr_label}/seed={seed}] failed:\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

    # Find the generated dir (script appends timestamp)
    generated = sorted(out_root.glob(f"{model_name}_*"))
    if not generated:
        raise RuntimeError(f"No output dir for {model_name} in {out_root}")
    generated_dir = generated[-1]
    final_dir = out_root / model_name
    generated_dir.rename(final_dir)

    elapsed = time.time() - t0
    print(f"  [{perceptron}] lr={lr_label} seed={seed}  done in {elapsed:.1f}s -> {final_dir.name}")
    return perceptron, lr_label, final_dir


def aggregate(perceptron: str, runs: list[tuple[str, int, Path]]) -> pd.DataFrame:
    rows = []
    for lr_label, seed, run_dir in runs:
        m = pd.read_csv(run_dir / "metrics.csv")
        m = m[~m["fold"].isin(["mean", "std"])].copy()
        for c in m.columns:
            try: m[c] = pd.to_numeric(m[c])
            except Exception: pass
        w = pd.read_csv(run_dir / "weights.csv")
        feat_cols = [c for c in w.columns if c not in ("fold", "bias")]
        wnorm = np.sqrt((w[feat_cols] ** 2).sum(axis=1))
        for i, row in m.iterrows():
            rows.append({
                "perceptron": perceptron,
                "lr": lr_label,
                "seed": seed,
                "fold": int(row["fold"]),
                "mse_train": row["final_mse_train"],
                "mse_test":  row["mse_test"],
                "accuracy":  row["accuracy"],
                "precision": row["precision"],
                "recall":    row["recall"],
                "f1":        row["f1"],
                "wnorm":     float(wnorm.iloc[i]),
                "bias":      float(w["bias"].iloc[i]),
            })
    return pd.DataFrame(rows)


def summary_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two views:
    - per_seed: average across folds for each (lr, seed) -> shape (lrs*seeds, *)
    - aggregate: mean+/-std across all (seeds, folds) and across seeds-only.
    """
    per_seed = (
        df.groupby(["lr", "seed"])
          .agg(mse_test=("mse_test", "mean"),
               mse_train=("mse_train", "mean"),
               wnorm=("wnorm", "mean"),
               accuracy=("accuracy", "mean"),
               f1=("f1", "mean"))
          .reset_index()
    )
    # Across all (seed,fold) -- total spread
    agg_all = (
        df.groupby("lr")
          .agg(mse_test_mean=("mse_test", "mean"),
               mse_test_std=("mse_test", "std"),
               wnorm_mean=("wnorm", "mean"),
               wnorm_std=("wnorm", "std"),
               f1_mean=("f1", "mean"),
               f1_std=("f1", "std"))
    )
    # Across seeds only (using per-seed averages) -- inter-seed dispersion
    agg_seed = (
        per_seed.groupby("lr")
                .agg(mse_test_seedmean=("mse_test", "mean"),
                     mse_test_seedstd=("mse_test", "std"),
                     wnorm_seedmean=("wnorm", "mean"),
                     wnorm_seedstd=("wnorm", "std"),
                     f1_seedmean=("f1", "mean"),
                     f1_seedstd=("f1", "std"))
    )
    summary = agg_all.join(agg_seed)
    return per_seed, summary


def plot_dispersion(per_seed: pd.DataFrame, df: pd.DataFrame, out_path: Path, title: str) -> None:
    metrics = [("mse_test", "MSE test"), ("wnorm", "||w||"), ("f1", "F1")]
    lrs = sorted(df["lr"].unique(), key=lambda s: float(s))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (col, label) in zip(axes, metrics):
        # boxplot por lr usando todos los (seed, fold)
        data = [df[df["lr"] == lr][col].values for lr in lrs]
        bp = ax.boxplot(data, tick_labels=lrs, widths=0.5, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#cce5ff")
            patch.set_alpha(0.6)
        # overlay per-seed averages
        for i, lr in enumerate(lrs, start=1):
            vals = per_seed[per_seed["lr"] == lr][col].values
            jitter = (np.random.RandomState(0).rand(len(vals)) - 0.5) * 0.15
            ax.scatter(np.full_like(vals, i, dtype=float) + jitter, vals,
                       color="tab:red", s=30, zorder=3, label="mean por seed" if i == 1 else None)
        ax.set_title(label)
        ax.set_xlabel("learning rate")
        ax.grid(True, alpha=0.3, axis="y")
        if col == "mse_test":
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def write_doc(perceptron: str, df: pd.DataFrame, per_seed: pd.DataFrame,
              summary: pd.DataFrame, out_dir: Path) -> None:
    n_seeds = df["seed"].nunique()
    n_folds = df["fold"].nunique()
    seeds_list = sorted(df["seed"].unique())

    lines = []
    lines.append(f"# Sweep LR multi-seed — perceptrón {perceptron}")
    lines.append("")
    lines.append(
        f"Experimento: {n_seeds} seeds × {df['lr'].nunique()} LRs × "
        f"{n_folds} folds = {len(df)} entrenamientos. "
        f"`epochs={EPOCHS_OVERRIDE}` (suficiente para convergencia segun single-seed). "
        f"Seeds: {seeds_list}."
    )
    lines.append("")
    lines.append("![Dispersion](dispersion.png)")
    lines.append("")
    lines.append("## Resumen agregado por LR")
    lines.append("")
    lines.append("**Total (todos los seeds × folds):**")
    lines.append("")
    lines.append("| lr | MSE test (mean ± std) | ‖w‖ (mean ± std) | F1 (mean ± std) |")
    lines.append("|---|---|---|---|")
    for lr, row in summary.iterrows():
        lines.append(
            f"| {lr} | {row['mse_test_mean']:.5f} ± {row['mse_test_std']:.5f} "
            f"| {row['wnorm_mean']:.4f} ± {row['wnorm_std']:.4f} "
            f"| {row['f1_mean']:.4f} ± {row['f1_std']:.4f} |"
        )
    lines.append("")
    lines.append("**Dispersion entre seeds** (cada celda usa el promedio sobre folds de cada seed):")
    lines.append("")
    lines.append("| lr | MSE test seed-std | ‖w‖ seed-std | F1 seed-std |")
    lines.append("|---|---|---|---|")
    for lr, row in summary.iterrows():
        lines.append(
            f"| {lr} | {row['mse_test_seedstd']:.5f} "
            f"| {row['wnorm_seedstd']:.4f} "
            f"| {row['f1_seedstd']:.4f} |"
        )
    lines.append("")
    lines.append("## Per-seed (mean sobre folds)")
    lines.append("")
    lines.append("| lr | seed | MSE test | ‖w‖ | F1 |")
    lines.append("|---|---|---|---|---|")
    for _, r in per_seed.sort_values(["lr", "seed"]).iterrows():
        lines.append(
            f"| {r['lr']} | {int(r['seed'])} | {r['mse_test']:.5f} "
            f"| {r['wnorm']:.4f} | {r['f1']:.4f} |"
        )
    lines.append("")
    lines.append("## Datos crudos")
    lines.append("")
    lines.append("- `raw.csv` — una fila por (lr, seed, fold) con todas las metricas + ‖w‖.")
    lines.append("- `per_seed.csv` — agregado por (lr, seed).")
    lines.append("- `summary.csv` — agregado por lr.")
    (out_dir / "analisis.md").write_text("\n".join(lines))
    print(f"  saved: {out_dir / 'analisis.md'}")


def run_perceptron(name: str, max_workers: int) -> None:
    info = PERCEPTRONS[name]
    info["analisis"].mkdir(parents=True, exist_ok=True)
    info["out_root"].mkdir(parents=True, exist_ok=True)

    jobs = []
    for lr_label, base_cfg in info["configs"].items():
        for seed in SEEDS:
            jobs.append((name, str(info["script"]), str(base_cfg),
                         lr_label, seed, str(info["out_root"])))

    print(f"\n=== {name}: {len(jobs)} jobs, {max_workers} concurrentes ===")
    t0 = time.time()
    completed = []
    if max_workers == 1:
        for j in jobs:
            completed.append(run_one(j))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futs = [pool.submit(run_one, j) for j in jobs]
            for f in as_completed(futs):
                completed.append(f.result())
    elapsed = time.time() - t0
    print(f"=== {name}: terminado en {elapsed:.1f}s ===")

    # Aggregate
    runs = [(lr, int(d.name.split("seed")[1]), d) for (_, lr, d) in completed]
    df = aggregate(name, runs)
    df.to_csv(info["analisis"] / "raw.csv", index=False)

    per_seed, summary = summary_tables(df)
    per_seed.to_csv(info["analisis"] / "per_seed.csv", index=False)
    summary.to_csv(info["analisis"] / "summary.csv")

    plot_dispersion(per_seed, df, info["analisis"] / "dispersion.png",
                    f"Sweep LR multi-seed - {name}")
    write_doc(name, df, per_seed, summary, info["analisis"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--perceptron", choices=["linear", "nonlinear", "both"], default="both")
    # Cada subprocess usa ~5 cores internamente (folds en paralelo).
    # Default outer = cpu_count // 5 para no overcommit.
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 5))
    args = p.parse_args()

    targets = ["linear", "nonlinear"] if args.perceptron == "both" else [args.perceptron]
    for t in targets:
        run_perceptron(t, args.workers)


if __name__ == "__main__":
    main()
