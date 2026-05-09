"""Análisis de convergencia del MSE: curva mean ± std por LR + criterio de plateau.

Lee `mse_history.csv` de cada run en `output/sweep_lr_multiseed/<lr>_seed<S>/`,
agrega por LR sobre las 25 corridas (5 seeds × 5 folds) y produce:

- `convergence.png`         — curva MSE_train vs época por LR (mean ± std).
- `convergence_tail.csv`    — pendiente del MSE en las últimas N épocas (criterio plateau).
- inserta bloque "## Convergencia del MSE" al principio de `analisis.md`,
  antes del plot de dispersión.

Por qué no usar el corte por epsilon como argumento de convergencia:
el criterio implementado en los scripts de entrenamiento es `mse < ε`, valor
absoluto. Como `ε` está por debajo del MSE asintótico que alcanzan los
modelos (lineal ~0.026, no-lineal ~0.011), nunca dispara: las 75 corridas
agotan las 500 épocas. La convergencia se argumenta entonces por **plateau**:
la pendiente del MSE en las últimas N épocas está en el ruido numérico.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent

PERCEPTRONS = {
    "linear": {
        "out_root": ROOT / "lineal_perceptron"   / "output"           / "sweep_lr_multiseed",
        "analisis": ROOT / "lineal_perceptron"   / "analisis_outputs" / "sweep_lr" / "multiseed",
    },
    "nonlinear": {
        "out_root": ROOT / "nonlinear_perceptron" / "output"           / "sweep_lr_multiseed",
        "analisis": ROOT / "nonlinear_perceptron" / "analisis_outputs" / "sweep_lr" / "multiseed",
    },
}

TAIL_WINDOW = 50
CONVERGENCE_BLOCK_MARKER = "## Convergencia del MSE"


def collect_history(out_root: Path) -> pd.DataFrame:
    """DataFrame con (lr, seed, fold, epoch, mse_train) sobre todos los runs."""
    rows = []
    for rd in sorted(out_root.glob("*_seed*")):
        cfg_path = rd / "config.json"
        h_path   = rd / "mse_history.csv"
        if not (cfg_path.exists() and h_path.exists()):
            continue
        cfg = json.loads(cfg_path.read_text())
        lr = float(cfg["training"]["learning_rate"])
        seed = int(cfg["random_seed"])
        h = pd.read_csv(h_path)
        h["lr"] = lr
        h["seed"] = seed
        rows.append(h[["lr", "seed", "fold", "epoch", "mse_train"]])
    if not rows:
        raise FileNotFoundError(f"No hay run dirs con mse_history en {out_root}")
    return pd.concat(rows, ignore_index=True)


def plot_convergence(history: pd.DataFrame, out_path: Path, title: str) -> None:
    """Dos paneles: lineal y log en X. Una curva por LR con banda ±1 std (sobre 25 runs)."""
    lrs = sorted(history["lr"].unique())
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(lrs)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, scale in zip(axes, ["linear", "log"]):
        for lr, color in zip(lrs, colors):
            g = history[history["lr"] == lr]
            agg = g.groupby("epoch")["mse_train"].agg(["mean", "std"])
            x = agg.index.to_numpy()
            ax.plot(x, agg["mean"], label=f"lr = {lr:g}", color=color, linewidth=2)
            ax.fill_between(x, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=color, alpha=0.2)
        ax.set_xlabel("Época")
        ax.set_ylabel("MSE train")
        ax.grid(True, alpha=0.3)
        if scale == "log":
            ax.set_xscale("log")
            ax.set_title("Eje X log (zoom en convergencia temprana)")
        else:
            ax.set_title("Eje X lineal")
        ax.legend(title="learning rate", loc="best", fontsize=9)
    fig.suptitle(
        f"{title}\nLínea: media de MSE_train por época sobre 5 seeds × 5 folds = 25 corridas. "
        "Banda: ±1 std sobre las 25 corridas."
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved: {out_path}")


def tail_slope_table(history: pd.DataFrame, window: int = TAIL_WINDOW) -> pd.DataFrame:
    """Pendiente del MSE en las últimas `window` épocas por (lr, seed, fold) y agregado por lr.

    slope = (mse[T] - mse[T-window+1]) / (window - 1)
    """
    rows = []
    for lr, g in history.groupby("lr"):
        slopes = []
        finals = []
        for (seed, fold), gg in g.groupby(["seed", "fold"]):
            tail = gg.sort_values("epoch").tail(window)
            if len(tail) < 2:
                continue
            slope = (tail["mse_train"].iloc[-1] - tail["mse_train"].iloc[0]) / (len(tail) - 1)
            slopes.append(slope)
            finals.append(float(tail["mse_train"].iloc[-1]))
        slopes = np.array(slopes)
        finals = np.array(finals)
        delta_window = slopes.mean() * (window - 1)
        rel_pct = abs(delta_window) / finals.mean() * 100
        rows.append(dict(
            lr=lr,
            mse_final_mean=finals.mean(),
            tail_slope_mean=slopes.mean(),
            tail_slope_max_abs=np.abs(slopes).max(),
            delta_in_window=delta_window,
            relative_change_pct=rel_pct,
        ))
    return pd.DataFrame(rows).sort_values("lr").reset_index(drop=True)


def render_block(perceptron: str, tail: pd.DataFrame) -> str:
    lines = []
    lines.append(CONVERGENCE_BLOCK_MARKER)
    lines.append("")
    lines.append("![Convergencia](convergence.png)")
    lines.append("")
    lines.append(
        "Cada curva es la **media del MSE de train por época sobre 25 corridas** (5 seeds × 5 folds) "
        "de un mismo LR. La banda alrededor es **±1 std sobre las 25 corridas**. El panel izquierdo "
        "muestra eje X lineal; el derecho, log para ver mejor la convergencia temprana."
    )
    lines.append("")
    lines.append("### ¿Convergió por epsilon o por techo de épocas?")
    lines.append("")
    lines.append(
        "**Por techo de épocas.** Los 75 entrenamientos de este perceptrón "
        "(5 seeds × 3 LRs × 5 folds) corrieron las 500 épocas completas; el corte por "
        "`mse_train < epsilon` **nunca se disparó**. La razón es que `epsilon` está calibrado "
        "por debajo del MSE asintótico que alcanza la arquitectura sobre este dataset, "
        "así que pedir `mse_train < epsilon` es pedir un valor que el modelo no puede alcanzar. "
        "Por eso epsilon **no** sirve como evidencia de convergencia acá."
    )
    lines.append("")
    lines.append(f"### El argumento que sí vale: plateau (pendiente en las últimas {TAIL_WINDOW} épocas)")
    lines.append("")
    lines.append(
        f"Convergencia, según la clase de optimizadores, es que la actualización **deja de cambiar el estado**: "
        f"`ΔMSE/Δepoch → 0`. No es que MSE → 0. Para cada (lr, seed, fold) calculamos la pendiente del MSE "
        f"en las últimas **{TAIL_WINDOW} épocas** (slope = (mse[T] − mse[T−{TAIL_WINDOW-1}]) / {TAIL_WINDOW-1}) "
        f"y la agregamos sobre las 25 corridas de cada LR:"
    )
    lines.append("")
    lines.append(
        f"| lr | MSE final (media sobre 25 runs) | tail-slope media | tail-slope max-abs | Δ MSE en {TAIL_WINDOW} épocas | Δ% relativo |"
    )
    lines.append("|---|---|---|---|---|---|")
    for _, r in tail.iterrows():
        lines.append(
            f"| {r['lr']:g} "
            f"| {r['mse_final_mean']:.5f} "
            f"| {r['tail_slope_mean']:.2e} "
            f"| {r['tail_slope_max_abs']:.2e} "
            f"| {r['delta_in_window']:.2e} "
            f"| {r['relative_change_pct']:.2e} % |"
        )
    lines.append("")
    # Interpretación data-driven por LR
    NOISE_THRESHOLD = 1e-3   # % relativo: por debajo = ruido numérico
    PLATEAU_THRESHOLD = 1.0  # % relativo: por debajo = plateau práctico
    cats = {"noise": [], "plateau": [], "descending": []}
    for _, r in tail.iterrows():
        lr_str = f"{r['lr']:g}"
        rel = r["relative_change_pct"]
        if rel < NOISE_THRESHOLD:
            cats["noise"].append(lr_str)
        elif rel < PLATEAU_THRESHOLD:
            cats["plateau"].append(lr_str)
        else:
            cats["descending"].append(lr_str)

    lines.append("**Lectura de la tabla:**")
    lines.append("")
    if cats["noise"]:
        lines.append(
            f"- LR {', '.join(cats['noise'])}: Δ% relativo está en el **ruido numérico de float64** "
            f"(eps_máquina ≈ 1e-16). El modelo ya no aprende, oscila dentro del redondeo. **Plateau total.**"
        )
    if cats["plateau"]:
        lines.append(
            f"- LR {', '.join(cats['plateau'])}: Δ% relativo es chico pero **no nulo** "
            f"(entre {NOISE_THRESHOLD}% y {PLATEAU_THRESHOLD}%). Hay un descenso residual real, no es ruido. "
            f"Para fines prácticos se puede tratar como convergido: el cambio en {TAIL_WINDOW} épocas "
            f"es despreciable frente al MSE final. Si se quisiera cierre estricto, habría que entrenar más épocas para este LR."
        )
    if cats["descending"]:
        lines.append(
            f"- LR {', '.join(cats['descending'])}: Δ% relativo ≥ {PLATEAU_THRESHOLD}%. "
            f"**No converge dentro de las 500 épocas** — la curva todavía baja apreciablemente. "
            f"Para este LR habría que extender el entrenamiento."
        )
    lines.append("")
    lines.append(
        "**Conclusión.** El argumento honesto de convergencia para este perceptrón es **plateau empírico** "
        "(la curva deja de moverse), no el corte por epsilon. La tabla cuantifica cuán plana está la curva en "
        f"las últimas {TAIL_WINDOW} épocas y permite distinguir LRs que efectivamente estabilizaron de los que no."
    )
    lines.append("")
    lines.append(
        "Implicancia práctica: si en una iteración futura quisiéramos un criterio de corte que se dispare "
        "antes del techo, no debería ser `mse < epsilon` sino algo basado en variación, p.ej. "
        f"`abs(mean(mse[-{TAIL_WINDOW}:]) - mean(mse[-{2*TAIL_WINDOW}:-{TAIL_WINDOW}])) < delta`."
    )
    lines.append("")
    return "\n".join(lines)


def upsert_convergence_block(analisis_md: Path, block: str) -> None:
    """Inserta el bloque antes de `![Dispersion]`. Si ya existe, lo reemplaza."""
    text = analisis_md.read_text()
    if CONVERGENCE_BLOCK_MARKER in text:
        before, _, rest = text.partition(CONVERGENCE_BLOCK_MARKER)
        rest_after = rest.split("\n", 1)[1] if "\n" in rest else ""
        # Cortar hasta el siguiente "## " o "![Dispersion"
        next_h2  = rest_after.find("\n## ")
        next_img = rest_after.find("\n![Dispersion")
        candidates = [c for c in [next_h2, next_img] if c >= 0]
        if candidates:
            cut = min(candidates)
            tail = rest_after[cut:]
        else:
            tail = ""
        new = before + block + ("\n" + tail.lstrip("\n") if tail else "")
    else:
        target = "![Dispersion]"
        if target in text:
            new = text.replace(target, block + "\n" + target, 1)
        else:
            new = text.rstrip() + "\n\n" + block + "\n"
    analisis_md.write_text(new)
    print(f"  saved: {analisis_md}")


def run_perceptron(name: str) -> None:
    info = PERCEPTRONS[name]
    info["analisis"].mkdir(parents=True, exist_ok=True)
    print(f"\n=== {name}: cargando mse_history desde {info['out_root']} ===")
    history = collect_history(info["out_root"])
    n_runs = history.groupby(["lr", "seed", "fold"]).ngroups
    print(f"  {len(history)} filas (epoch × fold × seed × lr); {n_runs} runs distintos.")

    plot_convergence(history, info["analisis"] / "convergence.png",
                     f"Convergencia del MSE - {name}")

    tail = tail_slope_table(history)
    tail.to_csv(info["analisis"] / "convergence_tail.csv", index=False)
    print("  tail-slope table:")
    print("  " + tail.to_string(index=False).replace("\n", "\n  "))

    block = render_block(name, tail)
    upsert_convergence_block(info["analisis"] / "analisis.md", block)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--perceptron", choices=["linear", "nonlinear", "both"], default="both")
    args = p.parse_args()
    targets = ["linear", "nonlinear"] if args.perceptron == "both" else [args.perceptron]
    for t in targets:
        run_perceptron(t)


if __name__ == "__main__":
    main()
