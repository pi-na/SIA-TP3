"""Plots para el sweep de tamaño de dataset (Notas/testing_datasets_inventados)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "ejercicio2_experimentacion" / "output" / "dataset_size_sweep"
DEST = ROOT / "Notas" / "testing_datasets_inventados" / "plots"
DEST.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(OUT / "summary.csv")
history = pd.read_csv(OUT / "epoch_history.csv")

# Excluir filas mean/std si las hubiera
summary = summary[summary["fold"].astype(str).str.match(r"^\d+$|^0$")].copy()
summary["fraction"] = summary["fraction"].astype(float)

FRACS  = sorted(summary["fraction"].unique())
COLORS = {0.10: "#d62728", 0.25: "#ff7f0e", 0.50: "#1f77b4", 1.00: "#2ca02c"}
LABELS = {f: f"{int(f*100)}% (N_train={int(summary[summary.fraction==f].n_train.iloc[0])})" for f in FRACS}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "#222",
    "axes.labelcolor": "#222", "xtick.color": "#222", "ytick.color": "#222",
    "text.color": "#222", "font.size": 11, "axes.grid": True,
    "grid.color": "#dddddd", "grid.linewidth": 0.6,
})


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=12, color="#111")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1 — Curvas de convergencia: train/val loss y val_acc por época
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

ax = axes[0]
for f in FRACS:
    h = history[history["fraction"] == f].sort_values("epoch")
    c = COLORS[f]
    ax.plot(h["epoch"], h["train_loss"], color=c, linestyle="--", alpha=0.55, linewidth=1.4)
    ax.plot(h["epoch"], h["val_loss"],   color=c, linestyle="-",  linewidth=2.0,
            label=LABELS[f])
ax.plot([], [], color="#666", linestyle="--", label="train (dashed)")
ax.plot([], [], color="#666", linestyle="-",  label="val (solid)")
ax.set_yscale("log")
_style(ax, "Convergencia — cross-entropy loss (escala log)", "Época", "Loss")
ax.legend(fontsize=9, frameon=False)

ax = axes[1]
for f in FRACS:
    h = history[history["fraction"] == f].sort_values("epoch")
    ax.plot(h["epoch"], h["val_acc"], color=COLORS[f], linewidth=2.0, label=LABELS[f])
_style(ax, "Convergencia — accuracy de validación", "Época", "val_acc")
ax.set_ylim(0.5, 1.005)
ax.legend(fontsize=9, frameon=False, loc="lower right")

fig.tight_layout()
fig.savefig(DEST / "convergencia.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'convergencia.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — Learning curve (val_acc vs N_train) + gap train-val
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

n_train = summary["n_train"].to_numpy()
val_acc = summary["val_acc_final"].to_numpy()
train_acc = summary["train_acc_final"].to_numpy()

ax = axes[0]
ax.plot(n_train, train_acc, "o-", color="#888", linewidth=1.8, markersize=8, label="train_acc")
ax.plot(n_train, val_acc,   "o-", color="#1f77b4", linewidth=2.3, markersize=9, label="val_acc")
for x, y in zip(n_train, val_acc):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(8, -14),
                fontsize=9, color="#1f77b4")
ax.set_xscale("log")
_style(ax, "Learning curve  —  accuracy vs N_train (log-x)", "N_train (log)", "accuracy")
ax.set_ylim(0.85, 1.005); ax.legend(fontsize=10, frameon=False, loc="lower right")

ax = axes[1]
gap = train_acc - val_acc
ax.bar([f"{int(f*100)}%\nN={int(summary[summary.fraction==f].n_train.iloc[0])}" for f in FRACS],
       gap, color=[COLORS[f] for f in FRACS], edgecolor="#222", alpha=0.85)
for i, g in enumerate(gap):
    ax.text(i, g + 0.001, f"{g:.3f}", ha="center", fontsize=10, color="#222")
_style(ax, "Gap de generalización  (train_acc − val_acc)", "Fracción del pool usada", "gap")
ax.set_ylim(0, max(gap) * 1.25)

fig.tight_layout()
fig.savefig(DEST / "learning_curve_y_gap.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'learning_curve_y_gap.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — F1 por clase × fracción (heatmap)
# ─────────────────────────────────────────────────────────────────────────────
f1_cols = [f"f1_{c}" for c in range(10)]
M = summary[f1_cols].to_numpy()  # shape (4 fracs, 10 clases)
fig, ax = plt.subplots(figsize=(10, 3.6))
im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.0)
ax.set_xticks(range(10)); ax.set_xticklabels(range(10))
ax.set_yticks(range(len(FRACS)))
ax.set_yticklabels([f"{int(f*100)}%" for f in FRACS])
ax.set_xlabel("Clase (dígito)"); ax.set_ylabel("Fracción del pool")
ax.set_title("F1 por clase  —  validación", color="#111")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                color="#111" if M[i, j] > 0.78 else "#fff", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="F1")
fig.tight_layout()
fig.savefig(DEST / "f1_por_clase.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'f1_por_clase.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — best_epoch y total_epochs_run vs fracción
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4.2))
x = np.arange(len(FRACS)); w = 0.35
ax.bar(x - w/2, summary["best_epoch"],       w, color="#1f77b4", label="best_epoch (min val_loss)", edgecolor="#222")
ax.bar(x + w/2, summary["total_epochs_run"], w, color="#bbb",    label="total_epochs (ES dispara aquí)", edgecolor="#222")
for i, (b, t) in enumerate(zip(summary["best_epoch"], summary["total_epochs_run"])):
    ax.text(i - w/2, b + 0.3, str(int(b)), ha="center", fontsize=10)
    ax.text(i + w/2, t + 0.3, str(int(t)), ha="center", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"{int(f*100)}%" for f in FRACS])
_style(ax, "Convergencia temporal  —  best_epoch y stop por ES (patience=20)",
       "Fracción del pool", "Época")
ax.legend(fontsize=10, frameon=False)
fig.tight_layout()
fig.savefig(DEST / "epocas.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'epocas.png'}")
