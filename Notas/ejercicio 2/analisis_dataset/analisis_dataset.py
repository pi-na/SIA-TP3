"""Análisis exploratorio del dataset digits.csv.

Genera cuatro figuras:
  1. distribucion_clases.png   — conteo de muestras por clase
  2. muestras_por_clase.png    — 8 muestras aleatorias por clase
  3. media_por_clase.png       — imagen media (píxel a píxel) por clase, datos crudos
  4. media_normalizada.png     — imagen media por clase, datos z-score normalizados

Uso:
    python analisis_dataset.py

Outputs en: ejercicio2/analisis_dataset/
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT     = Path(__file__).resolve().parent
DATA     = ROOT.parent.parent / "data and documentation" / "digits.csv"
OUT_DIR  = ROOT
IMG_SIZE = 28

SEED = 42
N_SAMPLES = 8  # muestras a mostrar por clase en la figura 2


# --------------------------------------------------------------------------- #
# Carga y parseo                                                               #
# --------------------------------------------------------------------------- #

def load() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(DATA)
    labels = df["label"].values.astype(int)
    images = np.stack([np.array(json.loads(s), dtype=np.float32)
                       for s in df["image"]])
    return images, labels


# --------------------------------------------------------------------------- #
# Fig 1 — distribución de clases                                               #
# --------------------------------------------------------------------------- #

def plot_distribucion(labels: np.ndarray, classes: list[int]) -> None:
    counts = [np.sum(labels == c) for c in classes]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar([str(c) for c in classes], counts, color="#2196F3", alpha=0.8)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Clase (dígito)")
    ax.set_ylabel("Cantidad de muestras")
    ax.set_title(f"Distribución de clases — digits.csv  (total={len(labels)})")
    ax.set_ylim(0, max(counts) * 1.12)
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(np.mean(counts), color="tab:red", ls="--", lw=1.2,
               label=f"media = {np.mean(counts):.0f}")
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "distribucion_clases.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


# --------------------------------------------------------------------------- #
# Fig 2 — muestras aleatorias por clase                                        #
# --------------------------------------------------------------------------- #

def plot_muestras(images: np.ndarray, labels: np.ndarray,
                  classes: list[int]) -> None:
    rng = np.random.default_rng(SEED)
    n_cls = len(classes)

    fig, axes = plt.subplots(n_cls, N_SAMPLES,
                             figsize=(N_SAMPLES * 1.2, n_cls * 1.3))
    fig.suptitle(f"Muestras aleatorias por clase ({N_SAMPLES} por clase)",
                 fontsize=11)

    for row, cls in enumerate(classes):
        idx = np.where(labels == cls)[0]
        chosen = rng.choice(idx, size=N_SAMPLES, replace=False)
        for col, i in enumerate(chosen):
            ax = axes[row, col]
            ax.imshow(images[i].reshape(IMG_SIZE, IMG_SIZE),
                      cmap="gray_r", vmin=0, vmax=1)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(str(cls), fontsize=10, rotation=0,
                              labelpad=14, va="center")

    fig.tight_layout()
    out = OUT_DIR / "muestras_por_clase.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


# --------------------------------------------------------------------------- #
# Fig 3 — imagen media por clase (datos crudos)                                #
# --------------------------------------------------------------------------- #

def plot_media(images: np.ndarray, labels: np.ndarray,
               classes: list[int], suffix: str = "",
               title_prefix: str = "") -> None:
    n_cls = len(classes)
    ncols = n_cls
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 1.6, 2.2))
    fig.suptitle(f"{title_prefix}Imagen media por clase — píxel a píxel\n"
                 f"(media sobre todas las muestras de cada clase)",
                 fontsize=10)

    for ax, cls in zip(axes, classes):
        mask = labels == cls
        mean_img = images[mask].mean(axis=0).reshape(IMG_SIZE, IMG_SIZE)
        im = ax.imshow(mean_img, cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"{cls}\n(n={mask.sum()})", fontsize=8)
        ax.axis("off")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04,
                 label="intensidad media")
    fig.tight_layout()
    out = OUT_DIR / f"media_por_clase{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


# --------------------------------------------------------------------------- #
# Fig 4 — imagen media tras z-score por píxel                                  #
# --------------------------------------------------------------------------- #

def plot_media_normalizada(images: np.ndarray, labels: np.ndarray,
                           classes: list[int]) -> None:
    mean_px = images.mean(axis=0)
    std_px  = images.std(axis=0)
    std_px  = np.where(std_px == 0, 1.0, std_px)  # evitar div/0 en píxeles constantes
    images_z = (images - mean_px) / std_px

    n_cls = len(classes)
    fig, axes = plt.subplots(1, n_cls, figsize=(n_cls * 1.6, 2.4))
    fig.suptitle("Imagen media por clase — tras z-score por píxel\n"
                 "(normalización fit sobre todo el dataset)",
                 fontsize=10)

    vmin = images_z.min()
    vmax = images_z.max()

    for ax, cls in zip(axes, classes):
        mask = labels == cls
        mean_img = images_z[mask].mean(axis=0).reshape(IMG_SIZE, IMG_SIZE)
        im = ax.imshow(mean_img, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
        ax.set_title(f"{cls}\n(n={mask.sum()})", fontsize=8)
        ax.axis("off")

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04,
                 label="z-score medio")
    fig.tight_layout()
    out = OUT_DIR / "media_normalizada.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Guardado: {out}")


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    print("Cargando datos …")
    images, labels = load()
    classes = sorted(np.unique(labels).tolist())
    print(f"  {len(images)} muestras  |  clases: {classes}")
    print(f"  píxeles por imagen: {images.shape[1]}  ({IMG_SIZE}×{IMG_SIZE})")
    print(f"  rango de valores: [{images.min():.3f}, {images.max():.3f}]")
    print()

    print("Conteos por clase:")
    for c in classes:
        n = np.sum(labels == c)
        print(f"  {c}: {n}")
    print()

    plot_distribucion(labels, classes)
    plot_muestras(images, labels, classes)
    plot_media(images, labels, classes)
    plot_media_normalizada(images, labels, classes)

    print("\nListo.")


if __name__ == "__main__":
    main()
