"""Smoke test visual de la augmentación rotacional.

Carga unas pocas imágenes de digits.csv, las rota a ±10° y ±15°, y guarda un
PNG comparativo para verificar a ojo que la rotación produce dígitos legibles.

Salida: ejercicio3/analisis/rotation_aug/rotation_samples_demo.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from mlp.data import parse_features
from mlp.augmentation import apply_random_rotation

CSV = ROOT / "data and documentation" / "digits.csv"
OUT = ROOT / "ejercicio3" / "analisis" / "rotation_aug" / "rotation_samples_demo.png"


def main():
    rng = np.random.default_rng(42)
    df = pd.read_csv(CSV)
    X = parse_features(df, ["image"])
    y = df["label"].to_numpy()

    # 8 muestras: una por clase entre las que están en digits.csv (no la 8)
    classes_present = [c for c in range(10) if c != 8]
    sample_idx = []
    for c in classes_present[:8]:
        # primera ocurrencia
        idx = int(np.where(y == c)[0][0])
        sample_idx.append(idx)
    X_samples = X[sample_idx]              # shape (8, 784) — sin z-score, [0,1]
    y_samples = y[sample_idx]

    # Aplicar rotaciones; usamos los ÁNGULOS FIJOS ±10 y ±15 (no random)
    # para visualizar el extremo del rango. Para eso pasamos un seed fijo,
    # pero la función toma random uniform. Hacemos un wrapper rápido que
    # fuerza el ángulo, para el smoke test.
    from mlp.augmentation import _rotate_image_bilinear

    def rotate_batch_fixed(X, angle):
        B = X.shape[0]
        imgs = X.reshape(B, 28, 28)
        out = np.empty_like(imgs)
        for i in range(B):
            out[i] = _rotate_image_bilinear(imgs[i], angle, fill=0.0)
        return out.reshape(B, 784)

    cases = [
        ("original",  X_samples),
        ("rot -15°",  rotate_batch_fixed(X_samples, -15.0)),
        ("rot -10°",  rotate_batch_fixed(X_samples, -10.0)),
        ("rot +10°",  rotate_batch_fixed(X_samples,  10.0)),
        ("rot +15°",  rotate_batch_fixed(X_samples,  15.0)),
    ]

    # También testeamos random rotation con ángulos sorteados, ±15°
    np.random.seed(7)
    X_rand_rot15 = apply_random_rotation(X_samples, 15.0)
    cases.append(("random ±15°", X_rand_rot15))

    n_rows = len(cases)
    n_cols = len(sample_idx)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.4))
    if n_rows == 1:
        axes = axes[None, :]
    for r, (label, batch) in enumerate(cases):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(batch[c].reshape(28, 28), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"clase {y_samples[c]}", fontsize=9)
            if c == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, labelpad=42, ha="right")
    fig.suptitle("Smoke test — rotación bilineal (apply_random_rotation)", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"OK  ->  {OUT}")


if __name__ == "__main__":
    main()
