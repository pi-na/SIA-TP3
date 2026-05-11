"""Plots de la evaluación en digits_test.csv para el sweep de tamaño de dataset."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT  = ROOT / "ejercicio2_experimentacion" / "output" / "dataset_size_sweep"
DEST = ROOT / "Notas" / "testing_datasets_inventados" / "plots_test"
DEST.mkdir(parents=True, exist_ok=True)

tm = pd.read_csv(OUT / "test_metrics.csv")
cm = pd.read_csv(OUT / "test_confusion.csv")

FRACS = sorted(tm["fraction"].unique())
LABELS = [f"{int(f*100)}%\nN_train={int(tm[tm.fraction==f].n_train.iloc[0])}\nN_val={[50,140,290,580][i]}" for i, f in enumerate(FRACS)]

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
# Plot 1 — Bar: val_acc vs test_acc + intervalo de confianza Wilson 95%
# ─────────────────────────────────────────────────────────────────────────────
def wilson_ci(p, n, z=1.96):
    """Intervalo Wilson 95% para proporciones (más estable que Wald con N chico)."""
    if n == 0: return (p, p)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z * np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return (center - half, center + half)

val_n = [50, 140, 290, 580]
test_n = 2497

val_acc  = tm["val_acc (de la corrida)"].to_numpy()
test_acc = tm["test_acc"].to_numpy()
val_ci  = [wilson_ci(p, n) for p, n in zip(val_acc, val_n)]
test_ci = [wilson_ci(p, test_n) for p in test_acc]
val_err  = np.array([[a - lo, hi - a] for a, (lo, hi) in zip(val_acc, val_ci)]).T
test_err = np.array([[a - lo, hi - a] for a, (lo, hi) in zip(test_acc, test_ci)]).T

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(FRACS)); w = 0.38
bars1 = ax.bar(x - w/2, val_acc,  w, color="#1f77b4", edgecolor="#222",
               yerr=val_err,  capsize=4, ecolor="#222", alpha=0.9, label="val (set chico, IC95%)")
bars2 = ax.bar(x + w/2, test_acc, w, color="#2ca02c", edgecolor="#222",
               yerr=test_err, capsize=4, ecolor="#222", alpha=0.9, label="test (N=2497, IC95%)")
for xi, a in zip(x - w/2, val_acc):  ax.text(xi, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
for xi, a in zip(x + w/2, test_acc): ax.text(xi, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=9)
ax.set_ylim(0.6, 1.005)
_style(ax, "Accuracy: validación (set chico) vs test (digits_test.csv, N=2497)",
       "", "accuracy")
ax.legend(fontsize=10, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(DEST / "val_vs_test_acc.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'val_vs_test_acc.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2 — F1 por clase en TEST (heatmap)
# ─────────────────────────────────────────────────────────────────────────────
f1_cols = [f"f1_{c}" for c in range(10)]
M = tm[f1_cols].to_numpy()
fig, ax = plt.subplots(figsize=(10, 3.6))
im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=1.0)
ax.set_xticks(range(10)); ax.set_xticklabels(range(10))
ax.set_yticks(range(len(FRACS)))
ax.set_yticklabels([f"{int(f*100)}%" for f in FRACS])
ax.set_xlabel("Clase (dígito)"); ax.set_ylabel("Fracción del pool balanceado")
ax.set_title("F1 por clase  —  TEST (digits_test.csv, N=2497)", color="#111")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                color="#111" if M[i, j] > 0.78 else "#fff", fontsize=9)
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="F1")
fig.tight_layout()
fig.savefig(DEST / "f1_por_clase_test.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'f1_por_clase_test.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3 — Matriz de confusión: dataset_10 vs dataset_100 lado a lado
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, frac, title in zip(axes, [0.10, 1.00], ["dataset_10 (N_train=530)", "dataset_100 (N_train=5270)"]):
    sub = cm[cm["fraction"] == frac].pivot(index="true_label", columns="pred_label", values="count").fillna(0).astype(int)
    sub = sub.reindex(index=range(10), columns=range(10), fill_value=0)
    M = sub.to_numpy()
    M_row = M / M.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(M_row, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xlabel("predicho"); ax.set_ylabel("real")
    ax.set_title(title, fontsize=11, color="#111")
    for i in range(10):
        for j in range(10):
            v = M[i, j]
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8,
                    color="#111" if M_row[i, j] < 0.5 else "#fff")
fig.suptitle("Matrices de confusión en TEST (counts; row-normalized en color)",
             fontsize=12, color="#111")
fig.tight_layout()
fig.savefig(DEST / "confusion_10_vs_100.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'confusion_10_vs_100.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot extra — Comparación contra el modelo final del Ej2 (digits.csv, sin clase 8)
# ─────────────────────────────────────────────────────────────────────────────
# Números tomados de Notas/ejercicio 2/Analisis de resultados experimentacion.md
# Sección "Generalización externa (sobre digits_test.csv)" — 3 seeds.
EJ2_TEST = {
    "name": "Ej2 final\n(digits.csv\nshallow Adam@1e-3 bs=64\nN_train=12 449, 9 clases)",
    "test_acc": 0.8529, "test_macro_f1": 0.8062, "test_macro_P": 0.7706,
    "test_macro_R": 0.8485,
}

mine_acc = tm["test_acc"].to_numpy()
mine_f1  = tm["macro_f1"].to_numpy()
mine_labels = [f"my dataset_{int(f*100):d}\nN_train={int(tm[tm.fraction==f].n_train.iloc[0])}, 10 clases (balanceado)"
               for f in FRACS]

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

ax = axes[0]
all_labels = mine_labels + [EJ2_TEST["name"]]
all_acc = np.concatenate([mine_acc, [EJ2_TEST["test_acc"]]])
colors = ["#a6cee3", "#1f77b4", "#0c4a6e", "#0a2540", "#d62728"]
bars = ax.bar(range(len(all_labels)), all_acc, color=colors, edgecolor="#222")
for i, a in enumerate(all_acc):
    ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=10)
ax.set_xticks(range(len(all_labels))); ax.set_xticklabels(all_labels, fontsize=8.5)
ax.set_ylim(0.75, 1.005)
_style(ax, "Test accuracy  —  mis 4 modelos balanceados  vs  Ej2 final",
       "", "test_acc (digits_test.csv, N=2497)")

ax = axes[1]
all_f1 = np.concatenate([mine_f1, [EJ2_TEST["test_macro_f1"]]])
ax.bar(range(len(all_labels)), all_f1, color=colors, edgecolor="#222")
for i, a in enumerate(all_f1):
    ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=10)
ax.set_xticks(range(len(all_labels))); ax.set_xticklabels(all_labels, fontsize=8.5)
ax.set_ylim(0.70, 1.005)
_style(ax, "Test macro-F1  —  mis 4 modelos balanceados  vs  Ej2 final",
       "", "macro_F1 (digits_test.csv)")

fig.suptitle("La clase 8 es decisiva: el modelo del Ej2 nunca la vio (digits.csv no la tiene).",
             fontsize=11, color="#444", y=1.01)
fig.tight_layout()
fig.savefig(DEST / "comparacion_vs_ej2.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"OK  {DEST / 'comparacion_vs_ej2.png'}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4 — Curva val_acc, test_acc y train_acc vs N_train
# ─────────────────────────────────────────────────────────────────────────────
summary_train = pd.read_csv(OUT / "summary.csv")
n_train  = summary_train["n_train"].to_numpy()
train_acc = summary_train["train_acc_final"].to_numpy()
val_acc_v = summary_train["val_acc_final"].to_numpy()
test_acc_v = tm["test_acc"].to_numpy()

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(n_train, train_acc,  "o--", color="#888",    linewidth=1.5, markersize=8, label="train_acc (set chico, mismas filas)")
ax.plot(n_train, val_acc_v,  "o-",  color="#1f77b4", linewidth=2.2, markersize=9, label="val_acc (set chico)")
ax.plot(n_train, test_acc_v, "o-",  color="#2ca02c", linewidth=2.5, markersize=10, label="test_acc (digits_test, N=2497)")
for x, y in zip(n_train, test_acc_v):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, 8), fontsize=9, color="#2ca02c")
for x, y in zip(n_train, val_acc_v):
    ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(6, -16), fontsize=9, color="#1f77b4")
ax.set_xscale("log")
_style(ax, "Train vs Val vs Test  —  accuracy vs N_train (log-x)", "N_train (log)", "accuracy")
ax.set_ylim(0.70, 1.01); ax.legend(fontsize=10, frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(DEST / "train_val_test_vs_n.png", dpi=140)
plt.close(fig)
print(f"OK  {DEST / 'train_val_test_vs_n.png'}")
