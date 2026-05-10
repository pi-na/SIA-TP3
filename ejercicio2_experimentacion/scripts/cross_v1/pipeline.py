"""Orchestrator del pipeline cross_v1: stage1 → decide → stage2 → stage2b →
combine + plots + notas + commit + push.

Pensado para correr de noche desatendido. Defensivo: cada stage es
independiente, errores en una cell no matan al pipeline, y al despertar
hay un commit pusheado con todo lo que se logró completar.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import run_cells, consolidate  # noqa: E402

OUT = ROOT / "ejercicio2_experimentacion" / "output" / "cross_v1"
ANL = ROOT / "ejercicio2_experimentacion" / "analisis" / "cross_v1"
CFG = ROOT / "ejercicio2_experimentacion" / "configs" / "sweeps" / "cross_v1"
ARCH_DIR = ROOT / "ejercicio2_experimentacion" / "configs" / "sweeps" / "arch"
NOTES_PRE = ROOT / "Notas" / "ejercicio 2" / "Experimentos" / "Pre_LR_Batch_Opt"
NOTES_MAIN = ROOT / "Notas" / "ejercicio 2" / "Experimentos" / "Cross_LR_Opt_Arch"

OUT.mkdir(parents=True, exist_ok=True)
ANL.mkdir(parents=True, exist_ok=True)
NOTES_PRE.mkdir(parents=True, exist_ok=True)
NOTES_MAIN.mkdir(parents=True, exist_ok=True)

STATUS = OUT / "STATUS.txt"
PIPELINE_LOG = OUT / "pipeline.log"

# ---------- Hiperparams compartidos ----------
PATIENCE = 20

MAX_EPOCHS = {
    # (opt, lr) -> max_epochs
    ("sgd",      1e-4): 200,   # ajustado: ya sabemos que no converge en 600 → reportarla así
    ("sgd",      5e-4): 300,   # ajustado de 500 (alcanza con margen para best_ep+patience)
    ("sgd",      1e-3): 200,
    ("sgd",      5e-3): 100,
    ("sgd",      1e-2):  80,
    ("momentum", 1e-4): 250,
    ("momentum", 5e-4): 150,
    ("momentum", 1e-3):  80,
    ("momentum", 5e-3):  40,
    ("momentum", 1e-2):  40,
    ("adam",     1e-4):  60,
    ("adam",     5e-4):  40,
    ("adam",     1e-3):  40,
    ("adam",     5e-3):  30,
    ("adam",     1e-2):  30,
}
ALL_LRS = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
ALL_OPTS = ["sgd", "momentum", "adam"]
ALL_ARCHS = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]

STAGE1_LRS = [5e-4, 1e-3, 5e-3]
STAGE1_BATCHES = [16, 64, 256]
STAGE1_SEEDS = [42, 7]

STAGE2_SEEDS = [42, 7, 13]

STAGE2B_BATCHES = [16, 32, 64, 128, 256]
STAGE2B_SEEDS = [42, 7, 13]
STAGE2B_LR = 1e-3
STAGE2B_OPT = "adam"
STAGE2B_ARCH = "arch_shallow"

WORKERS = 8


def _lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def log(msg: str) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with PIPELINE_LOG.open("a") as f:
        f.write(line + "\n")


def status(msg: str) -> None:
    STATUS.write_text(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}\n")


# ---------- Stage 1 ----------
def build_stage1_cells() -> list[dict]:
    cells = []
    arch_path = str(ARCH_DIR / "arch_shallow.json")
    for opt in ALL_OPTS:
        for lr in STAGE1_LRS:
            for bs in STAGE1_BATCHES:
                for seed in STAGE1_SEEDS:
                    cells.append({
                        "id": f"stage1_{opt}_lr{_lr_tag(lr)}_bs{bs}_seed{seed}",
                        "arch_config_path": arch_path,
                        "lr": lr, "batch_size": bs,
                        "optimizer": opt, "seed": seed,
                        "max_epochs": MAX_EPOCHS[(opt, lr)],
                        "patience": PATIENCE,
                    })
    return cells


# ---------- Stage 2 main ----------
def closest_stage1_lr(lr: float) -> float:
    return min(STAGE1_LRS, key=lambda x: abs(x - lr))


def build_stage2_cells(best_batch_per_opt_lr: dict) -> list[dict]:
    cells = []
    for arch in ALL_ARCHS:
        arch_path = str(ARCH_DIR / f"{arch}.json")
        for opt in ALL_OPTS:
            for lr in ALL_LRS:
                stage1_lr = lr if lr in STAGE1_LRS else closest_stage1_lr(lr)
                key = f"{opt}|{stage1_lr}"
                bs = int(best_batch_per_opt_lr.get(key, 32))
                for seed in STAGE2_SEEDS:
                    cells.append({
                        "id": f"stage2_{arch}_{opt}_lr{_lr_tag(lr)}_bs{bs}_seed{seed}",
                        "arch_config_path": arch_path,
                        "lr": lr, "batch_size": bs,
                        "optimizer": opt, "seed": seed,
                        "max_epochs": MAX_EPOCHS[(opt, lr)],
                        "patience": PATIENCE,
                    })
    return cells


# ---------- Stage 2b estrella batch ----------
def build_stage2b_cells() -> list[dict]:
    cells = []
    arch_path = str(ARCH_DIR / f"{STAGE2B_ARCH}.json")
    for bs in STAGE2B_BATCHES:
        for seed in STAGE2B_SEEDS:
            cells.append({
                "id": f"stage2b_{STAGE2B_ARCH}_{STAGE2B_OPT}_lr{_lr_tag(STAGE2B_LR)}_bs{bs}_seed{seed}",
                "arch_config_path": arch_path,
                "lr": STAGE2B_LR, "batch_size": bs,
                "optimizer": STAGE2B_OPT, "seed": seed,
                "max_epochs": MAX_EPOCHS[(STAGE2B_OPT, STAGE2B_LR)],
                "patience": PATIENCE,
            })
    return cells


# ---------- Decide best batch ----------
def decide_best_batch(stage1_dir: Path, out_json: Path) -> dict:
    """Lee stage1/raw.csv y elige el batch ganador por (opt, lr) según
    val_acc_final medio sobre seeds×folds."""
    import pandas as pd
    raw = pd.read_csv(stage1_dir / "raw.csv")
    # Necesitamos extraer (opt, lr, batch) del id ya que raw.csv no los trae como cols
    # Pero run_summary tiene fold y demás — necesito agregar tags. Usamos el cell_id parseado.
    # Mejor: leemos cell_dir / summary.csv y enriquecemos con tags. Simplificación:
    # parseamos id desde una columna 'model' si existe, o desde el cell dirname.
    # Como raw.csv ya está consolidado pero sin id, releemos cell dirs:
    rows = []
    for d in sorted(stage1_dir.iterdir()):
        if not d.is_dir() or not d.name.startswith("stage1_"):
            continue
        sp = d / "summary.csv"
        if not sp.exists():
            continue
        # parse id: stage1_<opt>_lr<tag>_bs<bs>_seed<seed>
        parts = d.name.split("_")
        opt = parts[1]
        lr_tag = parts[2][2:]  # quita 'lr'
        bs = int(parts[3][2:])
        seed = int(parts[4][4:])
        # convertir lr_tag de vuelta a float
        lr_map = {_lr_tag(lr): lr for lr in STAGE1_LRS}
        lr = lr_map[lr_tag]
        s = pd.read_csv(sp)
        for _, r in s.iterrows():
            rows.append({"opt": opt, "lr": lr, "batch": bs, "seed": seed,
                         "fold": r["fold"], "val_acc_final": r["val_acc_final"],
                         "macro_f1": r["macro_f1"]})
    df = pd.DataFrame(rows)
    df.to_csv(stage1_dir / "raw_tagged.csv", index=False)
    # Agregar por (opt, lr, batch)
    agg = df.groupby(["opt", "lr", "batch"]).agg(
        val_acc_mean=("val_acc_final", "mean"),
        val_acc_std=("val_acc_final", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        n=("val_acc_final", "count"),
    ).reset_index()
    agg.to_csv(stage1_dir / "agg_by_batch.csv", index=False)
    # Ganador por (opt, lr): mayor val_acc_mean
    best = {}
    for (opt, lr), grp in agg.groupby(["opt", "lr"]):
        winner = grp.sort_values("val_acc_mean", ascending=False).iloc[0]
        best[f"{opt}|{lr}"] = int(winner["batch"])
    out_json.write_text(json.dumps(best, indent=2))
    log(f"best_batch decided: {best}")
    return best


# ---------- Plots + notas ----------
def make_plots_and_notes() -> None:
    """Llama scripts auxiliares para generar plots y notas."""
    from plot_and_notes import build_all  # noqa: E402
    build_all(OUT, ANL, NOTES_PRE, NOTES_MAIN, MAX_EPOCHS, ALL_LRS, ALL_OPTS,
              ALL_ARCHS, STAGE1_LRS, STAGE1_BATCHES, STAGE1_SEEDS,
              STAGE2_SEEDS, STAGE2B_BATCHES, STAGE2B_SEEDS,
              STAGE2B_LR, STAGE2B_OPT, STAGE2B_ARCH, PATIENCE)


# ---------- Commit + push ----------
def git_commit_push() -> None:
    repo = ROOT
    msg = (
        "cross_v1: experimento cruzado LR×Opt×Arch + estrella batch + pre LR×Batch×Opt\n\n"
        "- Stage1: pre LR×Batch×Opt sobre arch_shallow para decidir batch óptimo por (opt, LR).\n"
        "- Stage2: grid 3D 5 LR × 3 opt × 4 arch con batch heredado de stage1.\n"
        "- Stage2b: estrella batch alrededor del centro (shallow+Adam@1e-3).\n"
        "- Notas en Notas/ejercicio 2/Experimentos/{Pre_LR_Batch_Opt,Cross_LR_Opt_Arch}/.\n"
        "- ES patience=20, fix #4 (best_weights restaurados siempre) aplicado al MLP.\n"
    )
    try:
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
        subprocess.run(["git", "commit", "-m", msg], cwd=repo, check=True)
        log("git commit OK")
        try:
            subprocess.run(["git", "push"], cwd=repo, check=True)
            log("git push OK")
        except subprocess.CalledProcessError as e:
            log(f"git push FAILED (commit local OK): {e}")
    except subprocess.CalledProcessError as e:
        log(f"git commit FAILED: {e}")


# ---------- Main ----------
def main() -> None:
    log("=== cross_v1 pipeline START ===")
    status("starting")

    # Stage 1
    log("Stage 1: pre LR×Batch×Opt")
    s1_dir = OUT / "stage1"
    cells = build_stage1_cells()
    log(f"Stage 1 cells: {len(cells)}  (jobs)")
    status(f"Stage 1 running (0/{len(cells)})")
    n_ok, n_fail = run_cells(cells, s1_dir, workers=WORKERS, status_file=STATUS, label="stage1")
    log(f"Stage 1 done: ok={n_ok} fail={n_fail}")
    consolidate(s1_dir)

    # Decide best batch
    log("Deciding best batch per (opt, lr)")
    try:
        best_batch = decide_best_batch(s1_dir, OUT / "best_batch.json")
    except Exception:
        log(f"decide_best_batch FAILED:\n{traceback.format_exc()}")
        log("Falling back to batch=32 for all cells.")
        best_batch = {}

    # Stage 2 main
    log("Stage 2 main: 5 LR × 3 opt × 4 arch")
    s2_dir = OUT / "stage2"
    cells = build_stage2_cells(best_batch)
    log(f"Stage 2 cells: {len(cells)}")
    status(f"Stage 2 running (0/{len(cells)})")
    n_ok, n_fail = run_cells(cells, s2_dir, workers=WORKERS, status_file=STATUS, label="stage2")
    log(f"Stage 2 done: ok={n_ok} fail={n_fail}")
    consolidate(s2_dir)

    # Stage 2b
    log("Stage 2b: estrella batch")
    s2b_dir = OUT / "stage2b"
    cells = build_stage2b_cells()
    log(f"Stage 2b cells: {len(cells)}")
    status(f"Stage 2b running (0/{len(cells)})")
    n_ok, n_fail = run_cells(cells, s2b_dir, workers=WORKERS, status_file=STATUS, label="stage2b")
    log(f"Stage 2b done: ok={n_ok} fail={n_fail}")
    consolidate(s2b_dir)

    # Plots + notas
    log("Building plots + notas")
    status("plots+notas")
    try:
        make_plots_and_notes()
        log("plots + notas OK")
    except Exception:
        log(f"plots+notas FAILED:\n{traceback.format_exc()}")

    # Commit + push
    log("git commit + push")
    status("commit+push")
    git_commit_push()

    log("=== cross_v1 pipeline END ===")
    status("DONE")


if __name__ == "__main__":
    main()
