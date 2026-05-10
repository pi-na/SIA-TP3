"""Genera diagramas de las 4 arquitecturas del Ej2 con estilo "YouTube ML".

Estilo: fondo oscuro tipo 3Blue1Brown, neuronas como círculos, conexiones full-connected
con transparencia para sugerir densidad, capas etiquetadas con tamaño + activación.
Capas grandes (>15 neuronas) se colapsan con "..." en el medio para que se vean.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import numpy as np

OUT = Path(__file__).parent

# ---- estilo ----
BG          = "#0d1117"       # fondo oscuro tipo GitHub dark
NEURON_IN   = "#58a6ff"       # input: azul
NEURON_HID  = "#3fb950"       # hidden: verde
NEURON_OUT  = "#f78166"       # output: naranja
EDGE_COLOR  = "#30363d"       # conexiones grises tenues
TEXT_COLOR  = "#e6edf3"
LABEL_COLOR = "#8b949e"
ACTIVATION_COLOR = "#d2a8ff"

ARCHS = {
    "arch_shallow": {
        "layers": [784, 128, 10],
        "activations": [None, "ReLU", "Softmax"],
        "title": "arch_shallow",
        "subtitle": "1 capa oculta · ~101k parámetros",
        "params": 101770,
    },
    "arch_base": {
        "layers": [784, 128, 64, 10],
        "activations": [None, "ReLU", "ReLU", "Softmax"],
        "title": "arch_base",
        "subtitle": "2 capas ocultas · ~109k parámetros",
        "params": 109386,
    },
    "arch_wider": {
        "layers": [784, 256, 128, 10],
        "activations": [None, "ReLU", "ReLU", "Softmax"],
        "title": "arch_wider",
        "subtitle": "2 capas ocultas anchas · ~235k parámetros",
        "params": 235146,
    },
    "arch_deeper": {
        "layers": [784, 128, 64, 32, 10],
        "activations": [None, "ReLU", "ReLU", "ReLU", "Softmax"],
        "title": "arch_deeper",
        "subtitle": "3 capas ocultas · ~111k parámetros",
        "params": 111466,
    },
}


def neurons_to_render(n: int, max_show: int = 9) -> list[int]:
    """Devuelve los índices de neuronas a dibujar.
    Si n <= max_show, dibuja todas; si no, dibuja max_show con un gap (... en el medio)."""
    if n <= max_show:
        return list(range(n))
    half = max_show // 2
    return list(range(half)) + list(range(n - half, n))


def positions(n_to_show: int, has_gap: bool, layer_x: float, vspan: float = 5.0) -> list[tuple[float, float]]:
    """Devuelve coordenadas (x,y) para cada neurona renderizada, centradas verticalmente,
    con un hueco visible si has_gap=True."""
    if has_gap:
        # Reservamos un slot extra para el gap "..."
        positions_count = n_to_show + 1
    else:
        positions_count = n_to_show
    if positions_count == 1:
        ys = [0]
    else:
        ys = np.linspace(vspan / 2, -vspan / 2, positions_count)
    coords = []
    if has_gap:
        half = n_to_show // 2
        # mitad superior usa primeros slots, mitad inferior usa últimos
        for i in range(half):
            coords.append((layer_x, ys[i]))
        # gap en el medio
        gap_y = ys[half]
        for i in range(half, n_to_show):
            coords.append((layer_x, ys[i + 1]))
        return coords, gap_y
    return [(layer_x, y) for y in ys], None


def draw_arch(ax, arch_spec: dict, show_title: bool = True) -> None:
    layers = arch_spec["layers"]
    activations = arch_spec["activations"]
    n_layers = len(layers)

    # spaces between layers
    LAYER_GAP = 3.5
    layer_xs = [i * LAYER_GAP for i in range(n_layers)]

    NEURON_R = 0.18

    # Compute neuron positions per layer
    layer_data = []
    for i, n in enumerate(layers):
        idxs = neurons_to_render(n, max_show=9)
        has_gap = n > 9
        coords, gap_y = positions(len(idxs), has_gap, layer_xs[i], vspan=4.5)
        # color
        if i == 0:
            color = NEURON_IN
        elif i == n_layers - 1:
            color = NEURON_OUT
        else:
            color = NEURON_HID
        layer_data.append({"idxs": idxs, "coords": coords, "gap_y": gap_y, "color": color, "n": n})

    # Draw edges (fully connected) — light, with low alpha so it suggests density
    for i in range(n_layers - 1):
        for (x1, y1) in layer_data[i]["coords"]:
            for (x2, y2) in layer_data[i + 1]["coords"]:
                ax.plot([x1, x2], [y1, y2], color=EDGE_COLOR, linewidth=0.5, alpha=0.6, zorder=1)

    # Draw neurons
    for ld in layer_data:
        for (x, y) in ld["coords"]:
            circ = patches.Circle((x, y), NEURON_R, facecolor=ld["color"], edgecolor="white",
                                   linewidth=0.8, zorder=3)
            ax.add_patch(circ)
            # subtle glow
            glow = patches.Circle((x, y), NEURON_R * 1.6, facecolor=ld["color"], alpha=0.15, zorder=2)
            ax.add_patch(glow)
        # gap "..."
        if ld["gap_y"] is not None:
            x_gap = ld["coords"][0][0]
            for dy in (-0.18, 0, 0.18):
                ax.plot(x_gap, ld["gap_y"] + dy, marker="o", color=TEXT_COLOR,
                        markersize=2.5, zorder=4)

    # Layer labels (size + activation)
    for i, (ld, x) in enumerate(zip(layer_data, layer_xs)):
        # Size label below
        n = ld["n"]
        label = f"{n}"
        if i == 0:
            sublabel = "input\n(28×28)"
        elif i == n_layers - 1:
            sublabel = "output\n(10 clases)"
        else:
            sublabel = f"hidden {i}"
        ax.text(x, -2.85, label, color=TEXT_COLOR, ha="center", va="top",
                fontsize=14, fontweight="bold")
        ax.text(x, -3.25, sublabel, color=LABEL_COLOR, ha="center", va="top", fontsize=9)
        # Activation label above
        act = activations[i]
        if act:
            ax.text(x, 2.85, act, color=ACTIVATION_COLOR, ha="center", va="bottom",
                    fontsize=10, fontweight="bold",
                    bbox=dict(facecolor=BG, edgecolor=ACTIVATION_COLOR,
                              boxstyle="round,pad=0.3", linewidth=1))

    # Arrow flow
    for i in range(n_layers - 1):
        x = layer_xs[i] + LAYER_GAP / 2
        ax.annotate("", xy=(x + 0.15, 3.4), xytext=(x - 0.15, 3.4),
                    arrowprops=dict(arrowstyle="->", color=LABEL_COLOR, lw=1))

    # Title
    if show_title:
        ax.text(np.mean(layer_xs), 4.4, arch_spec["title"], color=TEXT_COLOR,
                ha="center", va="bottom", fontsize=18, fontweight="bold",
                family="monospace")
        ax.text(np.mean(layer_xs), 4.0, arch_spec["subtitle"], color=LABEL_COLOR,
                ha="center", va="bottom", fontsize=10, style="italic")

    ax.set_xlim(-0.8, layer_xs[-1] + 0.8)
    ax.set_ylim(-4.0, 5.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)


def make_individual(arch_name: str, spec: dict) -> None:
    fig = plt.figure(figsize=(11, 6.5), facecolor=BG)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    draw_arch(ax, spec)
    fig.savefig(OUT / f"{arch_name}.png", dpi=170, facecolor=BG)
    plt.close(fig)
    print(f"  → {arch_name}.png")


def make_comparison() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), facecolor=BG)
    for ax, (name, spec) in zip(axes.flatten(), ARCHS.items()):
        draw_arch(ax, spec)
    fig.suptitle("Las 4 arquitecturas del Ej2 — comparación", color=TEXT_COLOR,
                 fontsize=20, fontweight="bold", y=0.97)
    fig.savefig(OUT / "comparacion_archs.png", dpi=170, facecolor=BG)
    plt.close(fig)
    print("  → comparacion_archs.png")


def make_params_chart() -> None:
    """Bar chart de parámetros por arch para que se vea la diferencia."""
    fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
    ax.set_facecolor(BG)
    names = list(ARCHS.keys())
    params = [ARCHS[n]["params"] for n in names]
    colors = ["#3fb950", "#58a6ff", "#f78166", "#d2a8ff"]
    bars = ax.bar([n.replace("arch_", "") for n in names], params, color=colors,
                   edgecolor="white", linewidth=1.2)
    for b, p in zip(bars, params):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 3000,
                f"{p:,}".replace(",", "."), ha="center", color=TEXT_COLOR, fontsize=11,
                fontweight="bold")
    ax.set_ylabel("Parámetros entrenables", color=TEXT_COLOR, fontsize=12)
    ax.set_title("Parámetros entrenables por arquitectura",
                  color=TEXT_COLOR, fontsize=14, fontweight="bold")
    ax.tick_params(colors=LABEL_COLOR, labelsize=11)
    for spine in ax.spines.values():
        spine.set_color(LABEL_COLOR)
    ax.grid(True, axis="y", alpha=0.2, color=LABEL_COLOR)
    fig.tight_layout()
    fig.savefig(OUT / "parametros_por_arch.png", dpi=170, facecolor=BG)
    plt.close(fig)
    print("  → parametros_por_arch.png")


if __name__ == "__main__":
    print("Generando diagramas...")
    for name, spec in ARCHS.items():
        make_individual(name, spec)
    make_comparison()
    make_params_chart()
    print("Listo.")
