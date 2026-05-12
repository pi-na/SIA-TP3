"""Análisis de rotación natural del dataset Ej3 (digits.csv + more_digits.csv).

Mide el ángulo de orientación del eje principal de tinta de cada imagen
(via momentos de imagen) y la excentricidad (cuán alargada vs. circular).

Modos:
  --first-pass: genera eccentricity_histogram.html para decidir umbral de
                excentricidad por debajo del cual el ángulo no es confiable.
  --full: genera analisis_rotacion.html completo (requiere ECC_THRESHOLD
          editado en este archivo después del first-pass).

Uso:
    .venv/bin/python ejercicio3/analisis_dataset/analisis_rotacion.py --first-pass
    # mirá eccentricity_histogram.html, decidí umbral, editá ECC_THRESHOLD abajo
    .venv/bin/python ejercicio3/analisis_dataset/analisis_rotacion.py --full
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent  # SIA-TP3/

sys.path.insert(0, str(ROOT))
from mlp.data import parse_features  # noqa: E402

DIGITS_CSV = ROOT / "data and documentation" / "digits.csv"
MORE_DIGITS_CSV = ROOT / "data and documentation" / "more_digits.csv"

FIRST_PASS_HTML = HERE / "eccentricity_histogram.html"
FULL_HTML = HERE / "analisis_rotacion.html"
STATS_CSV = HERE / "rotation_stats.csv"

# -----------------------------------------------------------------------------
# Constantes del análisis
# -----------------------------------------------------------------------------
IMG_SIZE = 28
PERCENTILES = [5, 15, 35, 50, 65, 85, 95]

# Umbral de excentricidad: clases con e_mean < ECC_THRESHOLD se marcan como
# "ángulo no confiable". Decidir mirando eccentricity_histogram.html después
# del first-pass. None = no decidido todavía → --full falla con mensaje claro.
ECC_THRESHOLD: float | None = 0.40

# -----------------------------------------------------------------------------
# Carga del dataset combinado
# -----------------------------------------------------------------------------
def load_combined() -> pd.DataFrame:
    """Lee digits.csv + more_digits.csv y devuelve DataFrame con columnas:
    label (int), image (np.ndarray (784,) float [0,1]), source (str)."""
    frames = []
    for path, src in [(DIGITS_CSV, "digits"), (MORE_DIGITS_CSV, "more_digits")]:
        df = pd.read_csv(path)
        X = parse_features(df, ["image"])  # (N, 784)
        out = pd.DataFrame({
            "label": df["label"].astype(int).values,
            "image": list(X),
            "source": src,
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


# -----------------------------------------------------------------------------
# Cómputo vectorizado de momentos
# -----------------------------------------------------------------------------
def compute_moments(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (theta_deg, eccentricity) para cada imagen en X.

    X: (N, 784) float, valores en [0, 1].
    theta_deg ∈ [-90, 90]: ángulo del eje principal de la nube de tinta,
        medido desde el eje horizontal. Una "1" vertical da θ ≈ ±90°.
    eccentricity ∈ [0, 1]: cuán alargada es la nube. 0 = circular (ángulo
        indefinido), 1 = línea perfecta.
    """
    N = X.shape[0]
    img = X.reshape(N, IMG_SIZE, IMG_SIZE)  # (N, 28, 28)

    # Grilla de coordenadas. y = fila (crece hacia abajo), x = columna.
    y_grid, x_grid = np.meshgrid(np.arange(IMG_SIZE), np.arange(IMG_SIZE), indexing="ij")

    M = img.sum(axis=(1, 2))  # (N,)
    M_safe = np.where(M > 0, M, 1.0)

    x_bar = (img * x_grid[None]).sum(axis=(1, 2)) / M_safe
    y_bar = (img * y_grid[None]).sum(axis=(1, 2)) / M_safe

    dx = x_grid[None] - x_bar[:, None, None]
    dy = y_grid[None] - y_bar[:, None, None]

    mu20 = (img * dx * dx).sum(axis=(1, 2)) / M_safe
    mu02 = (img * dy * dy).sum(axis=(1, 2)) / M_safe
    mu11 = (img * dx * dy).sum(axis=(1, 2)) / M_safe

    theta_rad = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    theta_deg = np.degrees(theta_rad)

    trace = mu20 + mu02
    trace_safe = np.where(trace > 0, trace, 1.0)
    ecc = np.sqrt((mu20 - mu02) ** 2 + 4.0 * mu11 ** 2) / trace_safe

    # Sanitizar muestras vacías (no debería pasar pero por las dudas).
    theta_deg = np.where(M > 0, theta_deg, 0.0)
    ecc = np.where(M > 0, ecc, 0.0)

    return theta_deg, ecc


def center_theta_per_class(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega columna theta_centered: θ de cada muestra centrado en la
    mediana circular (período 180°) de su clase. 0° = orientación típica
    de esa clase, ±X° = cuánto se aparta esa muestra de la típica.

    También agrega theta_class_ref con el ángulo de referencia de la clase
    (para reporte, no para plots).
    """
    df = df.copy()
    df["theta_centered"] = np.nan
    df["theta_class_ref"] = np.nan

    for cls, sub in df.groupby("label"):
        theta_rad = np.radians(sub["theta"].values)
        # Período 180° → trabajar con 2θ que tiene período 360°.
        z = np.exp(1j * 2.0 * theta_rad)
        ref_rad = 0.5 * np.angle(z.mean())  # en (-π/2, π/2]
        ref_deg = float(np.degrees(ref_rad))

        # Shift y wrap a (-90°, 90°]
        shifted = sub["theta"].values - ref_deg
        shifted = ((shifted + 90.0) % 180.0) - 90.0

        df.loc[sub.index, "theta_centered"] = shifted
        df.loc[sub.index, "theta_class_ref"] = ref_deg

    return df


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------
def fig_to_base64(fig) -> str:
    """matplotlib Figure → base64 PNG embebible en <img src="data:..."> ."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def build_demo_grid(df: pd.DataFrame, demo_class: int) -> tuple[str, int, float]:
    """Grilla 1×7 de la clase demo, ordenada por theta_centered ascendente,
    para validar visualmente que ordenar por θ produce una secuencia rotada.
    Devuelve (base64_png, n_samples_clase, ecc_mean_clase).
    """
    sub = df[df["label"] == demo_class].copy()
    sub = sub.sort_values("theta_centered")
    qs = np.array(PERCENTILES) / 100.0
    idx_at_q = (qs * (len(sub) - 1)).round().astype(int)
    picks = sub.iloc[idx_at_q].reset_index(drop=True)

    fig, axes = plt.subplots(1, len(picks), figsize=(2 * len(picks), 2.6))
    for ax, (_, row) in zip(axes, picks.iterrows()):
        ax.imshow(np.asarray(row["image"]).reshape(IMG_SIZE, IMG_SIZE), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"θ = {row['theta_centered']:+.1f}°", fontsize=10)
    fig.suptitle(
        f"Validación visual — clase {demo_class} (la de mayor excentricidad media), "
        f"ordenada por θ ascendente",
        fontsize=11,
    )
    fig.tight_layout()
    return fig_to_base64(fig), len(sub), float(sub["eccentricity"].mean())


def build_class_card_png(df: pd.DataFrame, cls: int) -> str:
    """PNG combinado para la tarjeta de una clase:
       fila 1: imagen media + boxplot horizontal de θ_centered
       fila 2: 7 muestras a percentiles fijos con θ debajo
    """
    sub = df[df["label"] == cls]
    sub_sorted = sub.sort_values("theta_centered")
    qs = np.array(PERCENTILES) / 100.0
    idx_at_q = (qs * (len(sub_sorted) - 1)).round().astype(int)
    picks = sub_sorted.iloc[idx_at_q].reset_index(drop=True)

    mean_img = np.stack(sub["image"].values).mean(axis=0).reshape(IMG_SIZE, IMG_SIZE)

    fig = plt.figure(figsize=(14, 4.8))
    gs = fig.add_gridspec(2, len(picks), height_ratios=[1.0, 1.0])

    ax_mean = fig.add_subplot(gs[0, 0:2])
    ax_mean.imshow(mean_img, cmap="gray_r")
    ax_mean.set_xticks([])
    ax_mean.set_yticks([])
    ax_mean.set_title("Imagen media", fontsize=10)

    ax_box = fig.add_subplot(gs[0, 2:])
    ax_box.boxplot(
        sub["theta_centered"].values,
        vert=False,
        widths=0.6,
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    ax_box.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    ax_box.set_xlabel("θ centrado en la mediana de la clase (°)", fontsize=9)
    ax_box.set_yticks([])
    ax_box.set_title(f"Distribución de θ — clase {cls} (n = {len(sub)})", fontsize=10)

    for j, (_, row) in enumerate(picks.iterrows()):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(np.asarray(row["image"]).reshape(IMG_SIZE, IMG_SIZE), cmap="gray_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"p{PERCENTILES[j]}\nθ={row['theta_centered']:+.1f}°", fontsize=9)

    fig.tight_layout()
    return fig_to_base64(fig)


def build_eccentricity_box_plotly(df: pd.DataFrame) -> str:
    """Boxplot interactivo de excentricidad por clase (first-pass)."""
    fig = go.Figure()
    classes = sorted(df["label"].unique())
    for cls in classes:
        vals = df.loc[df["label"] == cls, "eccentricity"].values
        fig.add_trace(go.Box(
            y=vals,
            name=str(cls),
            boxpoints="outliers",
            marker=dict(size=3),
            line=dict(width=1.2),
        ))
    fig.update_layout(
        title="Excentricidad por clase — mirá esto para decidir el umbral",
        xaxis_title="clase",
        yaxis_title="excentricidad e ∈ [0, 1]",
        yaxis=dict(range=[0, 1]),
        height=480,
        showlegend=False,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


def build_theta_box_plotly(df: pd.DataFrame, ecc_threshold: float,
                            class_e_mean: dict[int, float]) -> str:
    """Boxplot interactivo de θ centrado por clase, color por confiabilidad."""
    fig = go.Figure()
    classes = sorted(df["label"].unique())
    for cls in classes:
        vals = df.loc[df["label"] == cls, "theta_centered"].values
        confiable = class_e_mean[cls] >= ecc_threshold
        color = "#3366cc" if confiable else "#aaaaaa"
        fig.add_trace(go.Box(
            y=vals,
            name=str(cls),
            boxpoints="outliers",
            marker=dict(size=3, color=color),
            line=dict(color=color, width=1.2),
            fillcolor=color,
            opacity=0.6,
        ))
    fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title=("θ centrado en la mediana de cada clase — "
               "azul = clase confiable (ē ≥ umbral), gris = baja confianza"),
        xaxis_title="clase",
        yaxis_title="θ centrado (°)",
        height=520,
        showlegend=False,
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# -----------------------------------------------------------------------------
# Stats agregadas
# -----------------------------------------------------------------------------
def class_stats(df: pd.DataFrame, ecc_threshold: float) -> pd.DataFrame:
    rows = []
    for cls, sub in df.groupby("label"):
        theta_c = sub["theta_centered"].values
        ecc = sub["eccentricity"].values
        sources = sub["source"].value_counts().to_dict()
        rows.append({
            "clase": cls,
            "n": len(sub),
            "e_mean": float(ecc.mean()),
            "theta_class_ref": float(sub["theta_class_ref"].iloc[0]),
            "theta_centered_median": float(np.median(theta_c)),
            "theta_centered_std": float(np.std(theta_c, ddof=0)),
            "theta_centered_iqr": float(np.percentile(theta_c, 75) - np.percentile(theta_c, 25)),
            "theta_centered_p5": float(np.percentile(theta_c, 5)),
            "theta_centered_p95": float(np.percentile(theta_c, 95)),
            "n_digits": sources.get("digits", 0),
            "n_more_digits": sources.get("more_digits", 0),
            "confiable": bool(ecc.mean() >= ecc_threshold),
        })
    return pd.DataFrame(rows).sort_values("clase").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Textos del HTML
# -----------------------------------------------------------------------------
DESCRIPTION_HTML = """
<h2>1. Descripción de la implementación</h2>

<p>Para cada imagen del dataset queremos saber cuánto está inclinada el dígito
respecto a su orientación típica. La idea es tratar la tinta como una "nube"
de puntos en 2D: si esa nube es alargada (como la barra vertical de un "1"),
tiene un eje principal claro, y el ángulo de ese eje es la orientación del
dígito.</p>

<p><b>Cálculo por imagen 28×28:</b></p>
<ol>
  <li><b>Centro de masa</b> (x̄, ȳ): promedio de las posiciones de los píxeles,
      ponderado por su intensidad.</li>
  <li><b>Momentos centrados</b>: cuánto se dispersa la tinta horizontalmente
      (μ₂₀), verticalmente (μ₀₂) y la correlación horizontal-vertical (μ₁₁).</li>
  <li><b>Ángulo θ del eje principal</b>:
      <code>θ = ½ · atan2(2·μ₁₁, μ₂₀ − μ₀₂)</code></li>
  <li><b>Excentricidad e</b> (cuán "alargada" vs. "redonda" es la nube):
      <code>e = √[(μ₂₀ − μ₀₂)² + 4·μ₁₁²] / (μ₂₀ + μ₀₂)</code> ∈ [0, 1].</li>
</ol>

<p><b>¿Por qué medimos e?</b> Para clases simétricas (0, 8), el "eje principal"
no está bien definido — cualquier rotación de un círculo da el mismo resultado.
En esos casos, e ≈ 0 y el ángulo medido es ruido. Marcamos como
<em>no confiable</em> las clases con excentricidad media por debajo del umbral
elegido (ver sección 3).</p>

<p><b>Centrado por clase.</b> El ángulo crudo θ está medido respecto al eje
horizontal, así que todas las "1" caen cerca de ±90°. Como nos interesa la
variación <em>relativa</em> a la orientación típica de cada clase, en los
plots reportamos <b>θ centrado</b>: cada muestra se compara con la mediana
circular (período 180°) de su clase. 0° = orientación típica de esa clase,
±X° = cuánto se aparta esa muestra.</p>
"""

DEMO_INTRO = """
<h3>1.1 Validación visual del método</h3>
<p>Antes de creerle a los boxplots de las próximas secciones, mostramos que
ordenar muestras por θ ascendente produce a ojo una secuencia visualmente
rotada. La clase de la demo se elige automáticamente: la de mayor
excentricidad media (la que tiene el ángulo más confiable).</p>
"""

CONCLUSION_INTRO = """
<h2>4. Conclusión: rango propuesto para augmentation</h2>
<p>Basado en la variación natural medida en el dataset combinado,
proponemos un rango de rotación que cubra ~2σ de la variación natural sin
entrar en zona de riesgo de cambio de clase (cota a priori ±15° para evitar
confusión 6↔9 a partir de rotaciones grandes).</p>
"""


# -----------------------------------------------------------------------------
# Modo --first-pass
# -----------------------------------------------------------------------------
def first_pass(df: pd.DataFrame, output_path: Path) -> None:
    box_html = build_eccentricity_box_plotly(df)

    # Tabla resumen de e por clase
    rows = []
    for cls, sub in df.groupby("label"):
        e = sub["eccentricity"].values
        rows.append({
            "clase": cls,
            "n": len(sub),
            "e_mean": e.mean(),
            "e_median": float(np.median(e)),
            "e_p5": float(np.percentile(e, 5)),
            "e_p95": float(np.percentile(e, 95)),
        })
    tbl = pd.DataFrame(rows).sort_values("clase")
    tbl_html = tbl.to_html(index=False, float_format="%.3f",
                            classes="stats-table", border=0)

    html = f"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8">
<title>First-pass: excentricidad por clase</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1100px;
          margin: 2rem auto; padding: 0 1.5rem; color: #222; line-height: 1.55; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  table.stats-table {{ border-collapse: collapse; margin: 1rem 0; }}
  table.stats-table th, table.stats-table td {{ padding: .35rem .9rem;
          border-bottom: 1px solid #ddd; text-align: right; }}
  table.stats-table th {{ background: #f4f4f4; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
  .note {{ background: #fff8dc; border-left: 4px solid #d4a000;
           padding: .7rem 1rem; margin: 1.2rem 0; }}
</style>
</head><body>
<h1>First-pass: excentricidad por clase</h1>
<p>Dataset combinado <code>digits.csv + more_digits.csv</code>
({len(df):,} muestras). Para cada imagen calculamos la excentricidad de la
nube de tinta (0 = circular, 1 = línea perfecta). Las clases con excentricidad
baja tienen ángulo de orientación no confiable.</p>

<div class="note">
  <b>Cómo usar esto:</b> mirá el plot y la tabla. Elegí un umbral de e_mean
  por debajo del cual la clase queda marcada como "ángulo no confiable" en
  el análisis completo. Después editá la constante <code>ECC_THRESHOLD</code>
  en el script y corré con <code>--full</code>.
</div>

{box_html}

<h2>Estadísticas por clase</h2>
{tbl_html}

</body></html>
"""
    output_path.write_text(html, encoding="utf-8")


# -----------------------------------------------------------------------------
# Modo --full
# -----------------------------------------------------------------------------
def full_report(df: pd.DataFrame, ecc_threshold: float, output_path: Path) -> None:
    df = center_theta_per_class(df)
    stats = class_stats(df, ecc_threshold)
    stats.to_csv(STATS_CSV, index=False)

    class_e_mean = dict(zip(stats["clase"], stats["e_mean"]))

    # Clase demo: la confiable con mayor e_mean
    confiables = stats[stats["confiable"]]
    if len(confiables) == 0:
        demo_class = int(stats.loc[stats["e_mean"].idxmax(), "clase"])
        print(f"  WARNING: ninguna clase pasa el umbral; usando clase {demo_class} para demo")
    else:
        demo_class = int(confiables.loc[confiables["e_mean"].idxmax(), "clase"])
    print(f"  clase demo (mayor e_mean): {demo_class}")

    demo_b64, demo_n, demo_e = build_demo_grid(df, demo_class)

    # Per-class cards
    cards_html_parts = []
    for cls in sorted(df["label"].unique()):
        row = stats[stats["clase"] == cls].iloc[0]
        warns = []
        if not row["confiable"]:
            warns.append("⚠ Ángulo no confiable (e_mean &lt; umbral)")
        if row["n_digits"] == 0:
            warns.append("⚠ Solo presente en <code>more_digits.csv</code>")
        warn_html = ("<div class='warn'>" + " · ".join(warns) + "</div>") if warns else ""

        card_b64 = build_class_card_png(df, cls)
        cards_html_parts.append(f"""
<div class="card">
  <div class="card-header">
    <h3>Clase {cls}</h3>
    <div class="stats-line">
      n = {row['n']:,} &nbsp;·&nbsp;
      ē = {row['e_mean']:.3f} &nbsp;·&nbsp;
      θ̃<sub>centrado</sub> = {row['theta_centered_median']:+.2f}° &nbsp;·&nbsp;
      σ<sub>θ</sub> = {row['theta_centered_std']:.2f}° &nbsp;·&nbsp;
      IQR = {row['theta_centered_iqr']:.2f}° &nbsp;·&nbsp;
      ref. clase = {row['theta_class_ref']:+.1f}°
    </div>
    {warn_html}
  </div>
  <img src="data:image/png;base64,{card_b64}" alt="clase {cls}">
</div>
""")
    cards_html = "\n".join(cards_html_parts)

    # Sección 3: boxplot cross-class
    box_html = build_theta_box_plotly(df, ecc_threshold, class_e_mean)

    # Tabla resumen
    display_cols = ["clase", "n", "e_mean", "theta_class_ref",
                    "theta_centered_median", "theta_centered_std",
                    "theta_centered_iqr", "theta_centered_p5",
                    "theta_centered_p95", "confiable"]
    stats_html = stats[display_cols].to_html(
        index=False, float_format="%.2f", classes="stats-table", border=0,
        formatters={"n": lambda x: f"{int(x):,}",
                    "confiable": lambda x: "✅" if x else "❌"},
    )

    # Hallazgos numéricos
    confiables_df = stats[stats["confiable"]]
    sigma_mean = confiables_df["theta_centered_std"].mean() if len(confiables_df) > 0 else 0.0
    sigma_max_row = (confiables_df.loc[confiables_df["theta_centered_std"].idxmax()]
                     if len(confiables_df) > 0 else None)
    iqr_mean = confiables_df["theta_centered_iqr"].mean() if len(confiables_df) > 0 else 0.0

    # Decisión de augmentation
    proposed = min(2.0 * sigma_mean, 15.0)
    findings_html = f"""
<h3>3.1 Hallazgos numéricos</h3>
<ul>
  <li>σ<sub>θ</sub> promedio sobre clases confiables: <b>{sigma_mean:.2f}°</b></li>
  <li>σ<sub>θ</sub> máxima entre confiables: <b>{sigma_max_row['theta_centered_std']:.2f}°</b> (clase {int(sigma_max_row['clase'])})</li>
  <li>IQR promedio sobre clases confiables: <b>{iqr_mean:.2f}°</b></li>
  <li>Umbral de confiabilidad usado: <b>e_mean ≥ {ecc_threshold:.2f}</b> → {len(confiables_df)}/{len(stats)} clases confiables</li>
</ul>
"""

    conclusion_html = f"""
{CONCLUSION_INTRO}
<ol>
  <li><b>Variación natural medida:</b> σ<sub>θ</sub> ≈ <b>{sigma_mean:.1f}°</b> en
      promedio sobre las clases confiables.</li>
  <li><b>Cota inferior razonable:</b> augmentar con un rango menor a σ<sub>θ</sub>
      replica lo que el modelo ya ve → señal de regularización débil.</li>
  <li><b>Cota superior por preservación de clase:</b> a partir de ±15° aparece
      riesgo de confusión 9↔7; ±30° riesgo 6↔9. Cota a priori: ±15°.</li>
  <li><b>Rango propuesto para augmentation</b>:
      ±min(2·σ<sub>θ</sub>, 15°) = <b>±{proposed:.1f}°</b>. Cubre 2σ de la
      variación natural (~95% si la distribución fuera gaussiana) sin entrar
      en zona de riesgo de cambio de clase.</li>
  <li><b>Hipótesis previa para anclar el experimento</b>: con este rango,
      esperamos que la augmentation suba val_acc 0.5-1 punto y, a diferencia
      del L2+gaussian noise del Paso 2 del Ej3, <em>también</em> suba
      test_acc, porque inyecta una invariancia (rotación) que el modelo
      no tenía de otro modo.</li>
</ol>
"""

    html = f"""<!doctype html>
<html lang="es"><head>
<meta charset="utf-8">
<title>Análisis de rotación — dataset combinado Ej3</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px;
          margin: 2rem auto; padding: 0 1.5rem; color: #222; line-height: 1.55; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: .3rem; }}
  h2 {{ border-bottom: 1px solid #ccc; padding-bottom: .2rem; margin-top: 2.4rem; }}
  h3 {{ margin-top: 1.8rem; }}
  code {{ background: #f4f4f4; padding: 1px 4px; border-radius: 3px; }}
  table.stats-table {{ border-collapse: collapse; margin: 1rem 0; font-size: .92rem; }}
  table.stats-table th, table.stats-table td {{ padding: .35rem .8rem;
          border-bottom: 1px solid #ddd; text-align: right; }}
  table.stats-table th {{ background: #f4f4f4; }}
  .card {{ border: 1px solid #ddd; border-radius: 6px; margin: 1.2rem 0;
           padding: .8rem 1rem; background: #fafafa; }}
  .card-header h3 {{ margin: .2rem 0; }}
  .card-header .stats-line {{ font-family: ui-monospace, Menlo, monospace;
           font-size: .88rem; color: #444; }}
  .card img {{ width: 100%; height: auto; margin-top: .6rem; }}
  .warn {{ background: #fff3cd; border-left: 4px solid #d4a000;
           padding: .35rem .7rem; margin: .4rem 0; font-size: .9rem; }}
  .demo {{ margin: 1.2rem 0; }}
  .demo img {{ max-width: 100%; height: auto; }}
</style>
</head><body>

<h1>Análisis de rotación natural — dataset combinado Ej3</h1>
<p>Dataset: <code>digits.csv</code> ({(df['source'] == 'digits').sum():,} muestras)
+ <code>more_digits.csv</code> ({(df['source'] == 'more_digits').sum():,} muestras)
= <b>{len(df):,} muestras totales</b>, {df['label'].nunique()} clases.</p>

{DESCRIPTION_HTML}

{DEMO_INTRO}
<div class="demo">
  <p><b>Clase elegida:</b> {demo_class} &nbsp;·&nbsp;
     n = {demo_n:,} &nbsp;·&nbsp; ē = {demo_e:.3f}</p>
  <img src="data:image/png;base64,{demo_b64}" alt="demo grid clase {demo_class}">
</div>

<h2>2. Los números con su variación (una tarjeta por clase)</h2>
{cards_html}

<h2>3. Resultados finales</h2>
{box_html}

<h3>3.2 Tabla resumen por clase</h3>
{stats_html}

{findings_html}

{conclusion_html}

</body></html>
"""
    output_path.write_text(html, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--first-pass", action="store_true",
                        help="Generar histograma de excentricidad para decidir umbral")
    parser.add_argument("--full", action="store_true",
                        help="Generar HTML completo (requiere ECC_THRESHOLD editado)")
    args = parser.parse_args()

    if not (args.first_pass or args.full):
        parser.error("Indicá --first-pass o --full")
    if args.first_pass and args.full:
        parser.error("--first-pass y --full son excluyentes")

    print("Cargando dataset combinado...")
    df = load_combined()
    print(f"  total: {len(df):,} muestras "
          f"(digits={int((df['source']=='digits').sum()):,}, "
          f"more_digits={int((df['source']=='more_digits').sum()):,})")

    print("Computando momentos (vectorizado)...")
    X = np.stack(df["image"].values)  # (N, 784)
    theta_deg, ecc = compute_moments(X)
    df["theta"] = theta_deg
    df["eccentricity"] = ecc
    print(f"  θ: [{theta_deg.min():.1f}°, {theta_deg.max():.1f}°] · "
          f"e: [{ecc.min():.3f}, {ecc.max():.3f}]")

    if args.first_pass:
        print(f"Generando {FIRST_PASS_HTML.name}...")
        first_pass(df, FIRST_PASS_HTML)
        print(f"OK: {FIRST_PASS_HTML}")
        print("\n→ Abrí ese HTML, decidí un umbral de excentricidad, "
              "editá ECC_THRESHOLD arriba en el script, y corré con --full.")
        return

    if args.full:
        if ECC_THRESHOLD is None:
            sys.exit("ERROR: ECC_THRESHOLD = None. Corré primero --first-pass, "
                     "mirá el histograma, y setealo en este archivo.")
        print(f"Generando {FULL_HTML.name} (umbral e_mean = {ECC_THRESHOLD})...")
        full_report(df, ECC_THRESHOLD, FULL_HTML)
        print(f"OK: {FULL_HTML}")
        print(f"OK: {STATS_CSV}")


if __name__ == "__main__":
    main()
