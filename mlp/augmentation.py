"""Augmentaciones de datos para el MLP.

Implementaciones desde cero en NumPy puro (sin scipy/PIL) para cumplir con el
mandato "todo desde cero" del TP. Las funciones operan sobre arrays flat
(B, 784) provenientes de imágenes 28x28 z-scoreadas a nivel dataset.

Convención de relleno (fill=0.0):
    Después de z-score por píxel `(x - μ_j) / σ_j`, el valor 0.0 representa la
    media del dataset para ese píxel. Rellenar zonas fuera de imagen con 0.0
    equivale a inyectar "el valor promedio del píxel" — neutral y consistente
    con la distribución del set.
"""
from __future__ import annotations

import numpy as np


IMG_SIDE = 28  # asume entradas 28x28 = 784


def _rotate_image_bilinear(
    img: np.ndarray, angle_deg: float, fill: float = 0.0
) -> np.ndarray:
    """Rotar una imagen 2D alrededor de su centro con interpolación bilineal.

    Args:
        img: array (H, W) — una imagen.
        angle_deg: ángulo en grados. Positivo = anti-horario.
        fill: valor para pixeles cuya fuente cae fuera de la imagen.

    Returns:
        array (H, W) — la imagen rotada, mismo dtype.
    """
    H, W = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    theta = np.deg2rad(angle_deg)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Para cada pixel de salida (y, x), calcular la coordenada fuente
    # (x_src, y_src) aplicando la rotación inversa (-theta) alrededor del centro.
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    dx = xx - cx
    dy = yy - cy
    x_src = cx + dx * cos_t + dy * sin_t
    y_src = cy - dx * sin_t + dy * cos_t

    # Bilinear: tomar los 4 vecinos enteros (x0, y0), (x1, y0), (x0, y1), (x1, y1)
    x0 = np.floor(x_src).astype(np.int64)
    x1 = x0 + 1
    y0 = np.floor(y_src).astype(np.int64)
    y1 = y0 + 1

    wx = (x_src - x0).astype(img.dtype)
    wy = (y_src - y0).astype(img.dtype)

    # Indexamos con clip para evitar IndexError; el resultado fuera-de-imagen
    # lo reemplazamos luego con `fill` usando la máscara `in_bounds`.
    x0c = np.clip(x0, 0, W - 1)
    x1c = np.clip(x1, 0, W - 1)
    y0c = np.clip(y0, 0, H - 1)
    y1c = np.clip(y1, 0, H - 1)

    v00 = img[y0c, x0c]
    v01 = img[y0c, x1c]
    v10 = img[y1c, x0c]
    v11 = img[y1c, x1c]

    interp = (
        (1.0 - wx) * (1.0 - wy) * v00
        + wx * (1.0 - wy) * v01
        + (1.0 - wx) * wy * v10
        + wx * wy * v11
    )

    in_bounds = (x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)
    return np.where(in_bounds, interp, fill).astype(img.dtype)


def apply_random_rotation(
    X: np.ndarray,
    max_angle_deg: float,
    img_side: int = IMG_SIDE,
    fill: float = 0.0,
) -> np.ndarray:
    """Aplicar rotación aleatoria por muestra a un batch de imágenes flat.

    Cada fila de X se interpreta como una imagen img_side x img_side aplanada.
    Para cada muestra se sortea un ángulo uniforme en [-max_angle, +max_angle]
    y se rota con interpolación bilineal centrada.

    Args:
        X: array (B, img_side*img_side) — batch de imágenes z-scoreadas, flat.
        max_angle_deg: máximo ángulo absoluto de rotación (grados).
        img_side: lado de la imagen (28 por defecto para digits).
        fill: valor de relleno para pixeles fuera de imagen.

    Returns:
        array (B, img_side*img_side) — batch rotado, mismo dtype que X.
    """
    if max_angle_deg <= 0:
        return X
    B = X.shape[0]
    expected = img_side * img_side
    if X.shape[1] != expected:
        raise ValueError(
            f"apply_random_rotation: X.shape[1]={X.shape[1]} != "
            f"img_side*img_side={expected}"
        )

    # Usamos el estado global de np.random para mantener la convención existente
    # con gaussian_noise en network.py (que usa np.random.normal directamente).
    angles = np.random.uniform(-max_angle_deg, max_angle_deg, size=B)

    imgs = X.reshape(B, img_side, img_side)
    rotated = np.empty_like(imgs)
    for i in range(B):
        rotated[i] = _rotate_image_bilinear(imgs[i], angles[i], fill=fill)
    return rotated.reshape(B, expected)
