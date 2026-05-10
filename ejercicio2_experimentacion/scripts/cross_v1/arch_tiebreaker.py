"""Arch tiebreaker: 4 archs × Adam × {5e-4, 1e-3} × 12 seeds NUEVOS × k=5.

Justificación: en cross_v1 stage2 las top configs eran arch_wider y arch_shallow
con adam@1e-3 (diff 0.0011, SEM ~0.001). Con 3 seeds no se distingue. Acá
corremos 12 seeds NUEVOS por cell → 12×5=60 corridas/cell, SEM ~0.0005,
suficiente para resolver la diferencia.

Batches heredados del cross_v1 best_batch.json: adam@5e-4 → 16, adam@1e-3 → 64.
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
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runner import run_cells, consolidate  # noqa: E402

OUT = ROOT / "ejercicio2_experimentacion" / "output" / "arch_tiebreaker"
ANL = ROOT / "ejercicio2_experimentacion" / "analisis" / "arch_tiebreaker"
NOTES = ROOT / "Notas" / "ejercicio 2" / "Experimentos" / "Arch_tiebreaker"
ARCH_DIR = ROOT / "ejercicio2_experimentacion" / "configs" / "sweeps" / "arch"
OUT.mkdir(parents=True, exist_ok=True)
ANL.mkdir(parents=True, exist_ok=True)
NOTES.mkdir(parents=True, exist_ok=True)

ARCHS = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]
LRS = [5e-4, 1e-3]
BATCH_BY_LR = {5e-4: 16, 1e-3: 64}
MAX_EP_BY_LR = {5e-4: 40, 1e-3: 40}
PATIENCE = 20
# 12 seeds nuevos (distintos de {42, 7, 13} usados en cross_v1)
NEW_SEEDS = [1, 2, 3, 5, 8, 11, 17, 23, 31, 41, 53, 67]


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def build_cells():
    cells = []
    for arch in ARCHS:
        for lr in LRS:
            bs = BATCH_BY_LR[lr]
            for seed in NEW_SEEDS:
                cells.append({
                    "id": f"tiebreak_{arch}_adam_lr{_lr_tag(lr)}_bs{bs}_seed{seed}",
                    "arch_config_path": str(ARCH_DIR / f"{arch}.json"),
                    "lr": lr, "batch_size": bs,
                    "optimizer": "adam", "seed": seed,
                    "max_epochs": MAX_EP_BY_LR[lr],
                    "patience": PATIENCE,
                })
    return cells


def parse_id(cid: str) -> dict:
    parts = cid.split("_")
    arch = "_".join(parts[1:3])
    opt = parts[3]
    lr_tag = parts[4][2:]
    bs = int(parts[5][2:])
    seed = int(parts[6][4:])
    lr_map = {"5e-4": 5e-4, "1e-3": 1e-3}
    return {"arch": arch, "opt": opt, "lr": lr_map[lr_tag], "batch": bs, "seed": seed}


def main():
    cells = build_cells()
    print(f"Total cells: {len(cells)}  (4 archs × 2 LRs × 12 seeds = 96)", flush=True)
    t0 = time.time()
    n_ok, n_fail = run_cells(cells, OUT, workers=8, status_file=OUT / "STATUS.txt", label="tiebreak")
    print(f"DONE ok={n_ok} fail={n_fail} wall={time.time()-t0:.1f}s", flush=True)
    consolidate(OUT)

    # ---------- Combinar con cross_v1 stage 2 (3 seeds) para tener 15 seeds totales ----------
    rows_new = []
    for d in sorted(OUT.iterdir()):
        if not d.is_dir() or not d.name.startswith("tiebreak_"):
            continue
        sp = d / "summary.csv"
        if not sp.exists():
            continue
        tags = parse_id(d.name)
        s = pd.read_csv(sp)
        for k, v in tags.items():
            s[k] = v
        rows_new.append(s)
    df_new = pd.concat(rows_new, ignore_index=True) if rows_new else pd.DataFrame()
    df_new["source"] = "tiebreaker_12seeds"

    # cross_v1 stage 2 (3 seeds {42, 7, 13}) — sólo las cells que coinciden
    s2_dir = ROOT / "ejercicio2_experimentacion" / "output" / "cross_v1" / "stage2"
    rows_old = []
    for d in sorted(s2_dir.iterdir()):
        if not d.is_dir():
            continue
        sp = d / "summary.csv"
        if not sp.exists():
            continue
        # parse stage2_arch_NAME_opt_lrTAG_bsBS_seedSS
        parts = d.name.split("_")
        if parts[3] != "adam" or parts[4][2:] not in ("5e-4", "1e-3"):
            continue
        arch = "_".join(parts[1:3])
        lr_tag = parts[4][2:]
        bs = int(parts[5][2:])
        seed = int(parts[6][4:])
        lr_map = {"5e-4": 5e-4, "1e-3": 1e-3}
        s = pd.read_csv(sp)
        s["arch"] = arch; s["opt"] = "adam"; s["lr"] = lr_map[lr_tag]
        s["batch"] = bs; s["seed"] = seed; s["source"] = "cross_v1_3seeds"
        rows_old.append(s)
    df_old = pd.concat(rows_old, ignore_index=True) if rows_old else pd.DataFrame()

    df_all = pd.concat([df_new, df_old], ignore_index=True)
    df_all.to_csv(OUT / "raw_combined.csv", index=False)

    # Aggregate
    agg = df_all.groupby(["arch", "lr"]).agg(
        val_acc_mean=("val_acc_final", "mean"),
        val_acc_std=("val_acc_final", "std"),
        val_acc_sem=("val_acc_final", lambda x: x.std() / np.sqrt(len(x))),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        val_loss_mean=("val_loss_final", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        n=("val_acc_final", "count"),
    ).reset_index()
    agg.to_csv(OUT / "summary_combined.csv", index=False)
    agg.to_csv(NOTES / "summary_combined.csv", index=False)

    # Plot: bar plot val_acc por arch para cada LR con error bars (95% CI ≈ 1.96·SEM)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, lr in zip(axes, LRS):
        sub = agg[agg["lr"] == lr].set_index("arch").reindex(ARCHS)
        x = np.arange(len(ARCHS))
        ax.bar(x, sub["val_acc_mean"], yerr=1.96 * sub["val_acc_sem"], capsize=4,
               color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"], edgecolor="black")
        ax.set_xticks(x); ax.set_xticklabels([a.replace("arch_", "") for a in ARCHS])
        ax.set_title(f"adam @ lr={_lr_tag(lr)}  (n={int(sub['n'].iloc[0])} corridas/arch)")
        ax.set_ylabel("val_acc (mean ± 95% CI)")
        ax.grid(True, axis="y", alpha=0.3)
        for i, (m, s) in enumerate(zip(sub["val_acc_mean"], sub["val_acc_sem"])):
            ax.text(i, m + 1.96*s + 0.0005, f"{m:.4f}", ha="center", fontsize=8)
    fig.suptitle("Arch tiebreaker — val_acc por arquitectura (combina 3 seeds cross_v1 + 12 seeds nuevos)")
    fig.tight_layout()
    fig.savefig(ANL / "tiebreaker_val_acc.png", dpi=140)
    fig.savefig(NOTES / "tiebreaker_val_acc.png", dpi=140)
    plt.close(fig)

    # Notes
    md = []
    md.append("# Arch tiebreaker — alta resolución\n\n")
    md.append("**Objetivo:** resolver el empate estadístico entre `arch_wider` y `arch_shallow` "
              "que apareció en `cross_v1` stage 2 (diferencia 0.0011 con SEM ~0.001).\n\n")
    md.append("## Configuración\n\n")
    md.append("| Parámetro | Valor |\n|---|---|\n"
              "| Optimizer | adam (β1=0.9, β2=0.999, ε=1e-8) |\n"
              "| LRs probados | 5e-4, 1e-3 |\n"
              "| Batch size | 16 (para LR=5e-4), 64 (para LR=1e-3) — heredado de cross_v1 best_batch |\n"
              "| Arquitecturas | shallow, base, wider, deeper |\n"
              "| Seeds NUEVOS | [1, 2, 3, 5, 8, 11, 17, 23, 31, 41, 53, 67] (12) |\n"
              "| Seeds heredados de cross_v1 | [42, 7, 13] (3) |\n"
              "| Total seeds combinados | 15 → 75 corridas/cell con k=5 |\n"
              "| k-folds | 5 estratificado |\n"
              "| max_epochs | 40 |\n"
              "| patience | 20 sobre val_loss |\n"
              "| Loss | cross_entropy |\n"
              "| Preprocessing | zscore + one-hot |\n"
              "| Regularización | ninguna |\n\n")

    md.append("## Resultados — combinando 3 seeds previos + 12 nuevos = 15 seeds × 5 folds = 75 corridas/cell\n\n")
    md.append("SEM ≈ std/√75 ≈ std/8.66 → con std~0.005, SEM ≈ 0.0006. Distingue diffs ≥0.0012 al 95%.\n\n")
    md.append("| arch | LR | val_acc mean | std | **SEM** | macro_f1 | val_loss | best_epoch | n |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in agg.iterrows():
        md.append(f"| {r['arch']} | {_lr_tag(r['lr'])} | "
                  f"**{r['val_acc_mean']:.4f}** | {r['val_acc_std']:.4f} | "
                  f"{r['val_acc_sem']:.4f} | "
                  f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
                  f"{r['val_loss_mean']:.4f} | "
                  f"{r['best_epoch_mean']:.1f} | {int(r['n'])} |\n")

    md.append("\n![tiebreaker val_acc](tiebreaker_val_acc.png)\n\n")

    # Decide winner
    winner = agg.nlargest(1, "val_acc_mean").iloc[0]
    md.append("## Conclusión\n\n")
    md.append(f"**Ganador del tiebreaker:** `{winner['arch']}` + adam + LR=`{_lr_tag(winner['lr'])}`\n\n")
    md.append(f"- val_acc: {winner['val_acc_mean']:.4f} ± {winner['val_acc_std']:.4f} (SEM={winner['val_acc_sem']:.4f})\n")
    md.append(f"- macro_f1: {winner['macro_f1_mean']:.4f}\n\n")

    # Comparison wider vs shallow
    w = agg[(agg["arch"] == "arch_wider") & (agg["lr"] == 1e-3)]
    s = agg[(agg["arch"] == "arch_shallow") & (agg["lr"] == 1e-3)]
    if not w.empty and not s.empty:
        diff = w["val_acc_mean"].iloc[0] - s["val_acc_mean"].iloc[0]
        sem_diff = float(np.sqrt(w["val_acc_sem"].iloc[0]**2 + s["val_acc_sem"].iloc[0]**2))
        z = diff / sem_diff if sem_diff > 0 else 0
        md.append("### wider vs shallow (LR=1e-3)\n\n")
        md.append(f"- diff = {diff:+.4f}\n")
        md.append(f"- SEM(diff) = {sem_diff:.4f}\n")
        md.append(f"- z-score ≈ {z:.2f}  → "
                  f"{'estadísticamente distintos al 95%' if abs(z) > 1.96 else 'NO distinguibles al 95% (queda empate aún con 15 seeds)'}\n")

    md.append("\n## Limitaciones\n\n")
    md.append("- Sólo se midió Adam con LR ∈ {5e-4, 1e-3}, batch heredado de cross_v1. No se varió nada más.\n")
    md.append("- Las 3 seeds de cross_v1 corrieron con `max_epochs` heredado de la auditoría (40); las 12 nuevas usan los mismos parámetros — son combinables directamente.\n")
    md.append("- Si el resultado es 'no distinguibles', la elección final se decide por Occam (el modelo más chico) o por otros criterios (val_loss, best_epoch, tiempo).\n")

    (NOTES / "analisis.md").write_text("".join(md))
    print(f"[notes] wrote {NOTES / 'analisis.md'}", flush=True)


if __name__ == "__main__":
    main()
