"""Genera plots y notas del experimento cross_v1.

Llamado por pipeline.py al final.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LR_COLORS = {1e-4: "#1f77b4", 5e-4: "#ff7f0e", 1e-3: "#2ca02c", 5e-3: "#d62728", 1e-2: "#9467bd"}
LR_LABEL = {1e-4: "1e-4", 5e-4: "5e-4", 1e-3: "1e-3", 5e-3: "5e-3", 1e-2: "1e-2"}
OPT_MARKERS = {"sgd": "o", "momentum": "s", "adam": "^"}
OPT_COLORS = {"sgd": "#1f77b4", "momentum": "#ff7f0e", "adam": "#2ca02c"}
ARCH_COLORS = {"arch_shallow": "#1f77b4", "arch_base": "#ff7f0e",
               "arch_wider": "#2ca02c", "arch_deeper": "#d62728"}


def _parse_cell_id(cid: str) -> dict:
    """Parsea cell IDs como stage2_arch_shallow_adam_lr1e-3_bs64_seed42."""
    parts = cid.split("_")
    stage = parts[0]
    if stage in ("stage2", "stage2b"):
        arch = "_".join(parts[1:3])
        opt = parts[3]
        lr_tag = parts[4][2:]
        bs = int(parts[5][2:])
        seed = int(parts[6][4:])
    else:  # stage1
        arch = "arch_shallow"
        opt = parts[1]
        lr_tag = parts[2][2:]
        bs = int(parts[3][2:])
        seed = int(parts[4][4:])
    lr_map = {"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "5e-3": 5e-3, "1e-2": 1e-2}
    return {"stage": stage, "arch": arch, "opt": opt, "lr": lr_map[lr_tag],
            "batch": bs, "seed": seed, "id": cid}


def _load_stage(stage_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (raw_tagged, history_tagged) leyendo cada cell_dir."""
    rows_raw, rows_hist = [], []
    for d in sorted(stage_dir.iterdir()):
        if not d.is_dir():
            continue
        sp = d / "summary.csv"
        hp = d / "history.csv"
        if not (sp.exists() and hp.exists()):
            continue
        tags = _parse_cell_id(d.name)
        s = pd.read_csv(sp)
        h = pd.read_csv(hp)
        for k, v in tags.items():
            s[k] = v
            h[k] = v
        rows_raw.append(s)
        rows_hist.append(h)
    if not rows_raw:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(rows_raw, ignore_index=True), pd.concat(rows_hist, ignore_index=True)


def _agg(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Agrega media y std por grupos."""
    out = df.groupby(by).agg(**{
        f"{c}_mean": (c, "mean") for c in cols
    } | {
        f"{c}_std": (c, "std") for c in cols
    }).reset_index()
    out["n"] = df.groupby(by).size().values
    return out


# =================== STAGE 1 PLOTS ===================
def plot_stage1(out_dir: Path, raw: pd.DataFrame) -> None:
    if raw.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _agg(raw, ["opt", "lr", "batch"], ["val_acc_final", "macro_f1", "val_loss_final", "best_epoch"])

    # 1) heatmap por opt: rows=lr, cols=batch
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
        sub = agg[agg["opt"] == opt].pivot(index="lr", columns="batch", values="val_acc_final_mean")
        sub = sub.reindex(index=sorted(sub.index, reverse=True))
        im = ax.imshow(sub.values, cmap="viridis", aspect="auto", vmin=0.85, vmax=0.97)
        ax.set_xticks(range(len(sub.columns))); ax.set_xticklabels([str(c) for c in sub.columns])
        ax.set_yticks(range(len(sub.index))); ax.set_yticklabels([LR_LABEL.get(v, str(v)) for v in sub.index])
        ax.set_xlabel("batch_size"); ax.set_ylabel("lr")
        ax.set_title(f"{opt} — val_acc mean (seeds×folds)")
        for i, lr in enumerate(sub.index):
            for j, b in enumerate(sub.columns):
                v = sub.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < 0.93 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Stage 1 — val_acc por (lr, batch) — arch_shallow, 2 seeds × 5 folds")
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_heatmap_val_acc.png", dpi=140)
    plt.close(fig)

    # 2) curvas: para cada (opt, lr), val_acc vs batch
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
        sub = agg[agg["opt"] == opt]
        for lr in sorted(sub["lr"].unique()):
            s = sub[sub["lr"] == lr].sort_values("batch")
            ax.errorbar(s["batch"], s["val_acc_final_mean"], yerr=s["val_acc_final_std"],
                        marker="o", label=f"lr={LR_LABEL.get(lr, lr)}",
                        color=LR_COLORS.get(lr), capsize=3)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("batch_size"); ax.set_ylabel("val_acc (mean ± std)")
        ax.set_title(f"{opt}")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Stage 1 — val_acc vs batch_size por LR")
    fig.tight_layout()
    fig.savefig(out_dir / "stage1_val_acc_vs_batch.png", dpi=140)
    plt.close(fig)

    agg.to_csv(out_dir / "stage1_summary.csv", index=False)


# =================== STAGE 2 PLOTS ===================
def plot_stage2(out_dir: Path, raw: pd.DataFrame, history: pd.DataFrame) -> None:
    if raw.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _agg(raw, ["arch", "opt", "lr"], ["val_acc_final", "macro_f1", "val_loss_final",
                                            "train_loss_final", "best_epoch"])
    agg.to_csv(out_dir / "stage2_summary.csv", index=False)

    # 1) Por arch: una fila × 3 cols (uno por opt). Cada plot: val_acc vs LR.
    archs = sorted(raw["arch"].unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
        for arch in archs:
            sub = agg[(agg["opt"] == opt) & (agg["arch"] == arch)].sort_values("lr")
            ax.errorbar(sub["lr"], sub["val_acc_final_mean"], yerr=sub["val_acc_final_std"],
                        marker="o", label=arch.replace("arch_", ""),
                        color=ARCH_COLORS.get(arch), capsize=3)
        ax.set_xscale("log"); ax.set_xlabel("lr")
        ax.set_title(f"{opt}")
        ax.grid(True, alpha=0.3); ax.legend()
    axes[0].set_ylabel("val_acc (mean ± std)")
    fig.suptitle("Stage 2 — val_acc vs LR por (arch, opt) — 3 seeds × 5 folds")
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_val_acc_vs_lr_per_opt.png", dpi=140)
    plt.close(fig)

    # 2) Heatmap arch × (opt, lr) val_acc
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
        sub = agg[agg["opt"] == opt].pivot(index="arch", columns="lr", values="val_acc_final_mean")
        sub = sub.reindex(index=archs)
        im = ax.imshow(sub.values, cmap="viridis", aspect="auto", vmin=0.85, vmax=0.97)
        ax.set_xticks(range(len(sub.columns)))
        ax.set_xticklabels([LR_LABEL.get(c, str(c)) for c in sub.columns])
        ax.set_yticks(range(len(sub.index)))
        ax.set_yticklabels([a.replace("arch_", "") for a in sub.index])
        ax.set_title(f"{opt}")
        for i in range(len(sub.index)):
            for j in range(len(sub.columns)):
                v = sub.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            color="white" if v < 0.93 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("Stage 2 — val_acc heatmap por (arch, lr) y opt")
    fig.tight_layout()
    fig.savefig(out_dir / "stage2_heatmap_arch_lr.png", dpi=140)
    plt.close(fig)

    # 3) Curvas de convergencia val_loss por opt (sólo arch_shallow para limpieza)
    if not history.empty:
        sh = history[history["arch"] == "arch_shallow"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for ax, opt in zip(axes, ["sgd", "momentum", "adam"]):
            sub = sh[sh["opt"] == opt]
            for lr in sorted(sub["lr"].unique()):
                s = sub[sub["lr"] == lr].groupby("epoch")["val_loss"].mean()
                ax.plot(s.index, s.values, label=f"lr={LR_LABEL.get(lr, lr)}",
                        color=LR_COLORS.get(lr), linewidth=1.4)
            ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("val CE (mean)")
            ax.set_title(f"{opt} — arch_shallow")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle("Stage 2 — convergencia val_loss por opt y LR (arch_shallow)")
        fig.tight_layout()
        fig.savefig(out_dir / "stage2_convergence_shallow.png", dpi=140)
        plt.close(fig)


# =================== STAGE 2B PLOTS ===================
def plot_stage2b(out_dir: Path, raw: pd.DataFrame) -> None:
    if raw.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = _agg(raw, ["batch"], ["val_acc_final", "macro_f1", "val_loss_final", "best_epoch"])
    agg.to_csv(out_dir / "stage2b_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    s = agg.sort_values("batch")
    ax.errorbar(s["batch"], s["val_acc_final_mean"], yerr=s["val_acc_final_std"],
                marker="o", capsize=4, color="#2ca02c")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch_size"); ax.set_ylabel("val_acc (mean ± std)")
    ax.set_title("Stage 2b — estrella batch (shallow + Adam@1e-3, 3 seeds × 5 folds)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stage2b_val_acc_vs_batch.png", dpi=140)
    plt.close(fig)


# =================== NOTAS ===================
def write_pre_note(notes_dir: Path, anl_dir: Path, raw1: pd.DataFrame,
                   best_batch: dict, MAX_EPOCHS: dict,
                   STAGE1_LRS, STAGE1_BATCHES, STAGE1_SEEDS, PATIENCE) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)
    # Copiar plots a la carpeta de notas
    for plot in ["stage1_heatmap_val_acc.png", "stage1_val_acc_vs_batch.png"]:
        src = anl_dir / "stage1" / plot
        if src.exists():
            (notes_dir / plot).write_bytes(src.read_bytes())
    if raw1.empty:
        (notes_dir / "analisis.md").write_text("# Pre LR×Batch×Opt — sin datos\nstage 1 no produjo resultados.")
        return
    agg = _agg(raw1, ["opt", "lr", "batch"], ["val_acc_final", "macro_f1", "val_loss_final"])
    agg.to_csv(notes_dir / "stage1_agg.csv", index=False)

    md = []
    md.append("# Pre-experimento: LR × Batch × Optimizer\n")
    md.append("**Objetivo:** decidir el `batch_size` óptimo por (optimizer, learning rate) "
              "para usar como hiperparámetro heredado en el grid principal del Cross_LR_Opt_Arch.\n")
    md.append("## Configuración completa\n")
    md.append("Parámetros **explícitos** de la corrida (todos los que no se varían son fijos):\n")
    md.append("| Parámetro | Valor |\n|---|---|\n"
              f"| Arquitectura | `arch_shallow` = `[784, 128, 10]`, activations `[relu, softmax]`, init `auto` (He) |\n"
              f"| Loss | `cross_entropy` |\n"
              f"| Preprocessing | `zscore`, `one_hot_targets=true` |\n"
              f"| Split | k-folds=5 estratificado, val_fraction_if_k1=0.2 |\n"
              f"| Regularización | l2=0, dropout=0, sin lr_schedule, sin augmentation |\n"
              f"| Early stopping | patience={PATIENCE} sobre val_loss (CE), restaura best_weights al cortar |\n"
              f"| Seeds | {STAGE1_SEEDS} |\n"
              f"| Optimizers | sgd, momentum (β=0.9), adam (β1=0.9, β2=0.999, ε=1e-8) — defaults del módulo |\n")

    md.append("Factores variados:\n")
    md.append(f"- LR: {[LR_LABEL.get(lr, lr) for lr in STAGE1_LRS]}\n")
    md.append(f"- Batch size: {STAGE1_BATCHES}\n")
    md.append(f"- Optimizer: sgd, momentum, adam\n\n")

    md.append("`max_epochs` por (opt, LR) (auditado previamente):\n\n")
    md.append("| optimizer | LR=5e-4 | 1e-3 | 5e-3 |\n|---|---|---|---|\n")
    for opt in ["sgd", "momentum", "adam"]:
        md.append(f"| {opt} | {MAX_EPOCHS[(opt, 5e-4)]} | {MAX_EPOCHS[(opt, 1e-3)]} | {MAX_EPOCHS[(opt, 5e-3)]} |\n")
    md.append("\nTotal cells: 3 LR × 3 batch × 3 opt = 27. Con 2 seeds = **54 jobs × 5 folds = 270 corridas**.\n\n")

    md.append("## Resultados crudos — val_acc media ± std (sobre 2 seeds × 5 folds = 10 corridas)\n\n")
    md.append("| opt | LR | batch | val_acc | macro_f1 | val_loss CE |\n|---|---|---|---|---|---|\n")
    for opt in ["sgd", "momentum", "adam"]:
        for lr in STAGE1_LRS:
            for bs in STAGE1_BATCHES:
                row = agg[(agg["opt"] == opt) & (agg["lr"] == lr) & (agg["batch"] == bs)]
                if row.empty:
                    md.append(f"| {opt} | {LR_LABEL.get(lr, lr)} | {bs} | – | – | – |\n")
                else:
                    r = row.iloc[0]
                    md.append(f"| {opt} | {LR_LABEL.get(lr, lr)} | {bs} | "
                              f"{r['val_acc_final_mean']:.4f} ± {r['val_acc_final_std']:.4f} | "
                              f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
                              f"{r['val_loss_final_mean']:.4f} |\n")
    md.append("\n")

    md.append("## Decisión: best `batch_size` por (opt, LR)\n\n")
    md.append("Criterio: máxima val_acc media sobre 10 corridas.\n\n")
    md.append("| optimizer | LR=5e-4 | 1e-3 | 5e-3 |\n|---|---|---|---|\n")
    for opt in ["sgd", "momentum", "adam"]:
        cells = []
        for lr in STAGE1_LRS:
            cells.append(str(best_batch.get(f"{opt}|{lr}", "?")))
        md.append(f"| {opt} | {cells[0]} | {cells[1]} | {cells[2]} |\n")
    md.append("\n*Para LR fuera del set de etapa 1 (1e-4, 1e-2), el grid principal hereda el batch del LR más cercano dentro del mismo optimizer.*\n\n")

    md.append("## Plots\n\n")
    md.append("![Heatmap val_acc por (lr, batch) y opt](stage1_heatmap_val_acc.png)\n\n")
    md.append("![Curvas val_acc vs batch por LR y opt](stage1_val_acc_vs_batch.png)\n\n")

    md.append("## Limitaciones\n\n")
    md.append("- 2 seeds × 5 folds = 10 corridas: SEM ≈ 0.0019 (suficiente para decidir batch, no para reportar diferencias finas).\n")
    md.append("- Hecho **sólo sobre `arch_shallow`**. Asumimos que el batch óptimo no depende fuertemente de la arquitectura para hereparlo en el grid principal.\n")
    md.append("- Sólo 3 LR; los LRs extremos (1e-4, 1e-2) no se midieron acá y heredan del LR más cercano.\n")
    md.append("- No exploramos batches < 16 ni > 256.\n")

    (notes_dir / "analisis.md").write_text("".join(md))


def write_main_note(notes_dir: Path, anl_dir: Path, raw2: pd.DataFrame, raw2b: pd.DataFrame,
                    best_batch: dict, MAX_EPOCHS: dict,
                    ALL_LRS, ALL_OPTS, ALL_ARCHS, STAGE2_SEEDS,
                    STAGE2B_BATCHES, STAGE2B_SEEDS, STAGE2B_LR, STAGE2B_OPT, STAGE2B_ARCH,
                    PATIENCE) -> None:
    notes_dir.mkdir(parents=True, exist_ok=True)
    for plot in ["stage2_val_acc_vs_lr_per_opt.png", "stage2_heatmap_arch_lr.png",
                 "stage2_convergence_shallow.png", "stage2b_val_acc_vs_batch.png"]:
        src = anl_dir / "stage2" / plot
        if not src.exists():
            src = anl_dir / "stage2b" / plot
        if src.exists():
            (notes_dir / plot).write_bytes(src.read_bytes())

    if raw2.empty:
        (notes_dir / "analisis.md").write_text("# Cross LR×Opt×Arch — sin datos\nstage 2 no produjo resultados.")
        return

    agg2 = _agg(raw2, ["arch", "opt", "lr"], ["val_acc_final", "macro_f1", "val_loss_final",
                                              "train_loss_final", "best_epoch"])
    agg2.to_csv(notes_dir / "stage2_agg.csv", index=False)
    if not raw2b.empty:
        agg2b = _agg(raw2b, ["batch"], ["val_acc_final", "macro_f1", "val_loss_final", "best_epoch"])
        agg2b.to_csv(notes_dir / "stage2b_agg.csv", index=False)

    md = []
    md.append("# Cross-experimento: LR × Optimizer × Arquitectura\n")
    md.append("**Objetivo:** validar la elección de hiperparámetros del Ej2 testeando el supuesto de "
              "independencia entre los factores LR, optimizer y arquitectura, y reportar la mejor "
              "configuración global con interacciones explícitas.\n\n")

    md.append("## Configuración completa\n\n")
    md.append("**Hiperparams fijos en TODAS las celdas:**\n\n")
    md.append("| Parámetro | Valor |\n|---|---|\n"
              "| Loss | `cross_entropy` |\n"
              "| Preprocessing | `zscore`, `one_hot_targets=true` |\n"
              "| Split | k-folds=5 estratificado |\n"
              "| Regularización | l2=0, dropout=0, sin lr_schedule, sin augmentation |\n"
              f"| Early stopping | patience={PATIENCE} sobre val_loss (CE), restaura best_weights al cortar |\n"
              "| Inicialización | `auto` (He para ReLU; Xavier para tanh/sigmoid) |\n"
              "| Output | softmax + cross_entropy combinados (regla de la cátedra) |\n"
              f"| Seeds (stage 2 main) | {STAGE2_SEEDS} |\n"
              f"| Seeds (stage 2b) | {STAGE2B_SEEDS} |\n"
              "| Optimizer hyperparams | sgd: solo lr · momentum: lr, β=0.9 · adam: lr, β1=0.9, β2=0.999, ε=1e-8 |\n")

    md.append("\n**Factores variados (stage 2 main):**\n\n")
    md.append(f"- LR: {[LR_LABEL.get(lr, lr) for lr in ALL_LRS]}\n")
    md.append(f"- Optimizer: {ALL_OPTS}\n")
    md.append(f"- Arquitectura: {ALL_ARCHS}\n")
    md.append("- Batch size: heredado del pre-experimento (Pre_LR_Batch_Opt) por (opt, LR)\n\n")

    md.append("**`batch_size` por celda (resultado del pre-experimento):**\n\n")
    md.append("| optimizer | LR=1e-4 (heredado) | 5e-4 | 1e-3 | 5e-3 | 1e-2 (heredado) |\n|---|---|---|---|---|---|\n")
    for opt in ALL_OPTS:
        bsmap = []
        for lr in ALL_LRS:
            stage1_lr = lr if lr in [5e-4, 1e-3, 5e-3] else (5e-4 if lr == 1e-4 else 5e-3)
            bsmap.append(str(best_batch.get(f"{opt}|{stage1_lr}", "?")))
        md.append(f"| {opt} | " + " | ".join(bsmap) + " |\n")

    md.append("\n**`max_epochs` por (opt, LR) (con ES patience=20, el corte real lo decide la curva):**\n\n")
    md.append("| optimizer | 1e-4 | 5e-4 | 1e-3 | 5e-3 | 1e-2 |\n|---|---|---|---|---|---|\n")
    for opt in ALL_OPTS:
        md.append(f"| {opt} | " + " | ".join(str(MAX_EPOCHS[(opt, lr)]) for lr in ALL_LRS) + " |\n")
    md.append("\n*Nota: `SGD@1e-4` capeado a 200 ep — auditoría previa estableció que no converge "
              "dentro de presupuesto razonable; se reporta como referencia de 'LR demasiado bajo' para SGD.*\n\n")

    md.append("**Arquitecturas comparadas:**\n\n")
    md.append("| arch | layer_sizes | hidden layers | params aprox. |\n|---|---|---|---|\n"
              "| arch_shallow | [784, 128, 10] | 1 | ~101k |\n"
              "| arch_base    | [784, 128, 64, 10] | 2 | ~109k |\n"
              "| arch_wider   | [784, 256, 128, 10] | 2 | ~235k |\n"
              "| arch_deeper  | [784, 128, 64, 32, 10] | 3 | ~111k |\n")

    md.append("\nTotal stage 2 main: 5 × 3 × 4 = 60 cells × 3 seeds = **180 jobs × 5 folds = 900 corridas**.\n\n")

    md.append("## Resultados — val_acc media ± std (sobre 3 seeds × 5 folds = 15 corridas)\n\n")
    md.append("| arch | opt | LR | val_acc | macro_f1 | val_loss CE | train_loss CE | best_epoch |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for arch in ALL_ARCHS:
        for opt in ALL_OPTS:
            for lr in ALL_LRS:
                r = agg2[(agg2["arch"] == arch) & (agg2["opt"] == opt) & (agg2["lr"] == lr)]
                if r.empty:
                    md.append(f"| {arch} | {opt} | {LR_LABEL.get(lr, lr)} | – | – | – | – | – |\n")
                else:
                    r = r.iloc[0]
                    md.append(f"| {arch} | {opt} | {LR_LABEL.get(lr, lr)} | "
                              f"{r['val_acc_final_mean']:.4f} ± {r['val_acc_final_std']:.4f} | "
                              f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
                              f"{r['val_loss_final_mean']:.4f} | "
                              f"{r['train_loss_final_mean']:.4f} | "
                              f"{r['best_epoch_mean']:.1f} |\n")

    md.append("\n## Top configs (ordenadas por val_acc)\n\n")
    top = agg2.nlargest(10, "val_acc_final_mean")
    md.append("| # | arch | opt | LR | val_acc | macro_f1 | best_epoch |\n|---|---|---|---|---|---|---|\n")
    for i, r in enumerate(top.itertuples(), 1):
        md.append(f"| {i} | {r.arch} | {r.opt} | {LR_LABEL.get(r.lr, r.lr)} | "
                  f"{r.val_acc_final_mean:.4f} ± {r.val_acc_final_std:.4f} | "
                  f"{r.macro_f1_mean:.4f} | {r.best_epoch_mean:.1f} |\n")

    if not raw2b.empty:
        md.append("\n## Stage 2b — Estrella batch alrededor del centro\n\n")
        md.append(f"Centro: `{STAGE2B_ARCH}` + `{STAGE2B_OPT}` + LR=`{LR_LABEL.get(STAGE2B_LR, STAGE2B_LR)}`. "
                  f"Batches probados: {STAGE2B_BATCHES}. Seeds: {STAGE2B_SEEDS}. k=5.\n\n")
        md.append("| batch | val_acc | macro_f1 | val_loss CE | best_epoch |\n|---|---|---|---|---|\n")
        for bs in STAGE2B_BATCHES:
            r = agg2b[agg2b["batch"] == bs]
            if r.empty:
                md.append(f"| {bs} | – | – | – | – |\n")
            else:
                r = r.iloc[0]
                md.append(f"| {bs} | {r['val_acc_final_mean']:.4f} ± {r['val_acc_final_std']:.4f} | "
                          f"{r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f} | "
                          f"{r['val_loss_final_mean']:.4f} | {r['best_epoch_mean']:.1f} |\n")

    md.append("\n## Plots\n\n")
    md.append("![val_acc vs LR por (arch, opt)](stage2_val_acc_vs_lr_per_opt.png)\n\n")
    md.append("![Heatmap val_acc por arch × LR para cada opt](stage2_heatmap_arch_lr.png)\n\n")
    md.append("![Convergencia val_loss por opt y LR (arch_shallow)](stage2_convergence_shallow.png)\n\n")
    if not raw2b.empty:
        md.append("![Estrella batch (centro: shallow + Adam@1e-3)](stage2b_val_acc_vs_batch.png)\n\n")

    # Conclusión auto-generada (mejor cell)
    if not agg2.empty:
        winner = agg2.nlargest(1, "val_acc_final_mean").iloc[0]
        md.append("## Configuración óptima encontrada\n\n")
        md.append(f"**`{winner['arch']}` + `{winner['opt']}` + LR=`{LR_LABEL.get(winner['lr'], winner['lr'])}`**\n\n")
        md.append(f"- val_acc: {winner['val_acc_final_mean']:.4f} ± {winner['val_acc_final_std']:.4f}\n")
        md.append(f"- macro_f1: {winner['macro_f1_mean']:.4f} ± {winner['macro_f1_std']:.4f}\n")
        md.append(f"- val_loss: {winner['val_loss_final_mean']:.4f}\n")
        md.append(f"- best_epoch promedio: {winner['best_epoch_mean']:.1f}\n\n")

    md.append("## Limitaciones / caveats\n\n")
    md.append("- **`SGD@1e-4` capeado a 200 ep**: ya sabemos por la auditoría previa que no converge en presupuesto razonable; se incluye para mostrar la curva 'LR demasiado bajo' pero la celda **no está convergida**.\n")
    md.append("- **`batch_size` heredado del pre-experimento sobre `arch_shallow`**: asumimos que el batch óptimo no depende fuertemente de la arquitectura. Es una suposición razonable pero no medida — si el grid muestra que el ranking de archs cambia mucho entre opts, vale la pena rever esto.\n")
    md.append("- **3 seeds × 5 folds = 15 corridas/celda**: SEM ≈ 0.0016 sobre val_acc. Distingue diferencias ≥0.005 con confianza pero no menores.\n")
    md.append(f"- **Patience={PATIENCE}**: auditoría previa mostró que cubre con ~3× la subida transitoria máxima observada en los sweeps anteriores. Para `SGD@1e-4` (descenso lento, monótono) el patience no dispara espuriamente porque cada epoch mejora estrictamente.\n")
    md.append("- **No se varió L2/dropout/data augmentation**: este experimento no testea regularización, sólo optimización. La regularización se atacaría en un experimento siguiente sobre el centro encontrado acá.\n")
    md.append("- **No se varió la inicialización**: `auto` selecciona He para ReLU, sin variantes.\n")
    md.append("- **No se varió la activación**: ReLU + softmax fijo. La activación es factor del Pack B, fuera del scope.\n")
    md.append("- **batch_size en stage 2b**: medido SÓLO en el centro `shallow + Adam@1e-3`. Si quisiéramos certificar que el efecto del batch generaliza a otras celdas, haría falta un mini-grid 2D adicional batch×opt o batch×arch.\n")

    (notes_dir / "analisis.md").write_text("".join(md))


def build_all(OUT, ANL, NOTES_PRE, NOTES_MAIN, MAX_EPOCHS, ALL_LRS, ALL_OPTS,
              ALL_ARCHS, STAGE1_LRS, STAGE1_BATCHES, STAGE1_SEEDS,
              STAGE2_SEEDS, STAGE2B_BATCHES, STAGE2B_SEEDS,
              STAGE2B_LR, STAGE2B_OPT, STAGE2B_ARCH, PATIENCE) -> None:
    OUT = Path(OUT); ANL = Path(ANL)
    raw1, _   = _load_stage(OUT / "stage1")
    raw2, his2 = _load_stage(OUT / "stage2")
    raw2b, _  = _load_stage(OUT / "stage2b")

    plot_stage1(ANL / "stage1", raw1)
    plot_stage2(ANL / "stage2", raw2, his2)
    plot_stage2b(ANL / "stage2b", raw2b)

    bb_path = OUT / "best_batch.json"
    best_batch = json.loads(bb_path.read_text()) if bb_path.exists() else {}

    write_pre_note(NOTES_PRE, ANL, raw1, best_batch, MAX_EPOCHS,
                   STAGE1_LRS, STAGE1_BATCHES, STAGE1_SEEDS, PATIENCE)
    write_main_note(NOTES_MAIN, ANL, raw2, raw2b, best_batch, MAX_EPOCHS,
                    ALL_LRS, ALL_OPTS, ALL_ARCHS, STAGE2_SEEDS,
                    STAGE2B_BATCHES, STAGE2B_SEEDS, STAGE2B_LR, STAGE2B_OPT, STAGE2B_ARCH,
                    PATIENCE)
