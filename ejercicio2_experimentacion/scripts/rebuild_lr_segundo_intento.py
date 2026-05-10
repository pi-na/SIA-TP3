"""Reconstruye raw.csv / epoch_history.csv / summary.csv desde el tmpdir
rescatado de un corte abrupto del sweep.

Solo incluye combinaciones (lr, seed) que tengan run_summary.csv + epoch_history.csv
completos. Es defensivo: si una carpeta está vacía o incompleta, se omite.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
SRC  = ROOT / "ejercicio2_experimentacion" / "output" / "lr_segundo_intento" / "_rescued_tmp"
OUT  = ROOT / "ejercicio2_experimentacion" / "output" / "lr_segundo_intento"

LR_TAG_TO_FLOAT = {
    "lr1e-4": 1e-4, "lr5e-4": 5e-4, "lr1e-3": 1e-3, "lr5e-3": 5e-3, "lr1e-2": 1e-2,
}

METRIC_COLS = [
    "total_epochs",
    "train_loss_final", "val_loss_final",
    "train_acc_final", "val_acc_final",
    "macro_precision", "macro_recall", "macro_f1",
]

PAT = re.compile(r"^(arch_\w+?)_(lr[0-9e\-]+)_seed(\d+)$")

def main() -> None:
    summaries, histories = [], []
    skipped = []
    for combo_dir in sorted(SRC.iterdir()):
        if not combo_dir.is_dir():
            continue
        m = PAT.match(combo_dir.name)
        if not m:
            skipped.append((combo_dir.name, "no match"))
            continue
        arch_name, lr_tag, seed_s = m.group(1), m.group(2), m.group(3)
        seed = int(seed_s)
        lr = LR_TAG_TO_FLOAT.get(lr_tag)
        if lr is None:
            skipped.append((combo_dir.name, f"unknown lr {lr_tag}"))
            continue

        # Cada combo tiene un subdir <model_name>_<ts>/
        run_dirs = [d for d in combo_dir.iterdir() if d.is_dir()]
        if not run_dirs:
            skipped.append((combo_dir.name, "empty (in-flight)"))
            continue
        rd = run_dirs[0]
        s_csv = rd / "run_summary.csv"
        h_csv = rd / "epoch_history.csv"
        if not (s_csv.exists() and h_csv.exists()):
            skipped.append((combo_dir.name, "missing csvs"))
            continue

        s = pd.read_csv(s_csv)
        s = s[~s["fold"].isin(["mean", "std"])].copy()
        s["fold"] = s["fold"].astype(int)
        s.insert(0, "arch", arch_name)
        s.insert(1, "lr", lr)
        s.insert(2, "seed", seed)
        summaries.append(s)

        h = pd.read_csv(h_csv)
        h.insert(0, "arch", arch_name)
        h.insert(1, "lr", lr)
        h.insert(2, "seed", seed)
        histories.append(h)

    raw = pd.concat(summaries, ignore_index=True)
    raw.to_csv(OUT / "raw.csv", index=False)

    history = pd.concat(histories, ignore_index=True)
    history.to_csv(OUT / "epoch_history.csv", index=False)

    rows = []
    arch_name = raw["arch"].iloc[0]
    for lr in sorted(raw["lr"].unique()):
        sub = raw[(raw["arch"] == arch_name) & (raw["lr"] == lr)]
        n_seeds = sub["seed"].nunique()
        row = {"arch": arch_name, "lr": lr, "n_corridas": len(sub), "n_seeds": n_seeds}
        for col in METRIC_COLS:
            if col in sub.columns:
                row[f"{col}_mean_seedsfolds"] = sub[col].mean()
                row[f"{col}_std_seedsfolds"]  = sub[col].std()
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    print("=== Reconstruido ===")
    print(summary[["lr", "n_corridas", "n_seeds",
                   "val_acc_final_mean_seedsfolds", "val_acc_final_std_seedsfolds",
                   "macro_f1_mean_seedsfolds", "val_loss_final_mean_seedsfolds"]].to_string(index=False))
    print(f"\nfilas raw: {len(raw)}  ·  filas history: {len(history)}")
    if skipped:
        print("\nOmitidos:")
        for n, r in skipped:
            print(f"  - {n}  ({r})")

if __name__ == "__main__":
    main()
