"""Versión interactiva (Plotly HTML) del grid 3D del stage 2 del cross-experiment.

Genera `grid_3d_interactive.html` con:
  - Scene 3D principal: 60 esferas rotables, hover muestra config completa + métricas
  - 3 heatmaps faceted (un panel por optimizer)
  - Bar chart ranking de top-12

Abrí el HTML con doble click — corre offline (sin servidor), Plotly embebido.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
CSV = HERE.parent / "analisis" / "cross_v1" / "stage2" / "stage2_summary.csv"
OUT = HERE / "grid_3d_interactive.html"

ARCH_ORDER = ["arch_shallow", "arch_base", "arch_wider", "arch_deeper"]
OPT_ORDER  = ["sgd", "momentum", "adam"]
LR_ORDER   = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
LR_LABEL   = ["1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]
ARCH_LABEL = [a.replace("arch_", "") for a in ARCH_ORDER]
OPT_LABEL  = ["SGD", "Momentum", "Adam"]

BG = "#0d1117"
PAPER = "#0d1117"
TEXT = "#e6edf3"
LABEL = "#8b949e"
GRID = "#30363d"


def main() -> None:
    df = pd.read_csv(CSV)
    lr_idx_map = {round(l, 6): i for i, l in enumerate(LR_ORDER)}
    df["xi"] = df["lr"].round(6).map(lr_idx_map)
    df["yi"] = df["opt"].map({o: i for i, o in enumerate(OPT_ORDER)})
    df["zi"] = df["arch"].map({a: i for i, a in enumerate(ARCH_ORDER)})
    df["lr_label"] = df["xi"].map(lambda i: LR_LABEL[int(i)])
    df["opt_label"] = df["yi"].map(lambda i: OPT_LABEL[int(i)])
    df["arch_label"] = df["zi"].map(lambda i: ARCH_LABEL[int(i)])

    vmin, vmax = df["val_acc_final_mean"].min(), df["val_acc_final_mean"].max()

    # ===== Figure with 4 subplots: 3D + 3 heatmaps row, top-12 bar bottom =====
    fig = make_subplots(
        rows=2, cols=3,
        specs=[
            [{"type": "scene", "colspan": 3}, None, None],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=("Grid 3D (rotable) — LR × Optimizer × Arquitectura",
                         "<b>SGD</b>", "<b>Momentum</b>", "<b>Adam</b>"),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
        row_heights=[0.72, 0.28],
    )

    # ----- 3D scatter -----
    sizes_3d = 6 + 22 * (df["val_acc_final_mean"] - vmin) / (vmax - vmin)
    hover = df.apply(lambda r: (
        f"<b>{r['arch']} · {OPT_LABEL[int(r['yi'])]} · LR={LR_LABEL[int(r['xi'])]}</b><br>"
        f"val_acc: <b>{r['val_acc_final_mean']:.4f}</b> ± {r['val_acc_final_std']:.4f}<br>"
        f"macro_f1: {r['macro_f1_mean']:.4f}<br>"
        f"val_loss CE: {r['val_loss_final_mean']:.4f}<br>"
        f"train_loss CE: {r['train_loss_final_mean']:.4f}<br>"
        f"best_epoch: {r['best_epoch_mean']:.1f}<br>"
        f"n: {int(r['n'])} corridas"
    ), axis=1)

    fig.add_trace(go.Scatter3d(
        x=df["xi"], y=df["yi"], z=df["zi"],
        mode="markers",
        marker=dict(
            size=sizes_3d,
            color=df["val_acc_final_mean"],
            colorscale="Viridis",
            cmin=vmin, cmax=vmax,
            line=dict(color="white", width=0.5),
            opacity=0.95,
            colorbar=dict(
                title=dict(text="val_acc", font=dict(color=TEXT)),
                tickfont=dict(color=TEXT),
                len=0.55, y=0.7, x=1.02,
                bgcolor=BG, bordercolor=GRID,
            ),
        ),
        text=hover, hoverinfo="text", name="cells",
        showlegend=False,
    ), row=1, col=1)

    # Top-3 highlight
    top3 = df.nlargest(3, "val_acc_final_mean")
    fig.add_trace(go.Scatter3d(
        x=top3["xi"], y=top3["yi"], z=top3["zi"],
        mode="markers",
        marker=dict(size=sizes_3d[top3.index] + 4,
                    color="rgba(0,0,0,0)",
                    line=dict(color="#f9e64f", width=3)),
        text=[f"#{i+1}" for i in range(len(top3))],
        hoverinfo="skip", name="top-3", showlegend=False,
    ), row=1, col=1)

    fig.update_scenes(
        xaxis=dict(tickvals=list(range(5)), ticktext=LR_LABEL,
                    title=dict(text="Learning rate", font=dict(color=TEXT)),
                    color=TEXT, backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(tickvals=list(range(3)), ticktext=OPT_LABEL,
                    title=dict(text="Optimizer", font=dict(color=TEXT)),
                    color=TEXT, backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID),
        zaxis=dict(tickvals=list(range(4)), ticktext=ARCH_LABEL,
                    title=dict(text="Arquitectura", font=dict(color=TEXT)),
                    color=TEXT, backgroundcolor=BG, gridcolor=GRID, zerolinecolor=GRID),
        bgcolor=BG, camera=dict(eye=dict(x=1.6, y=-1.7, z=1.0)),
        row=1, col=1,
    )

    # ----- 3 heatmaps -----
    for i, opt in enumerate(OPT_ORDER):
        sub = df[df["opt"] == opt]
        mat = np.full((4, 5), np.nan)
        for _, r in sub.iterrows():
            mat[int(r["zi"]), int(r["xi"])] = r["val_acc_final_mean"]
        # text on cells (val_acc without leading zero)
        text = np.where(np.isnan(mat), "", np.vectorize(lambda v: f"{v:.3f}"[1:] if not np.isnan(v) else "")(mat))
        # rich hover with the matched row
        hover_mat = np.empty_like(mat, dtype=object)
        for zi in range(4):
            for xi in range(5):
                row = sub[(sub["zi"] == zi) & (sub["xi"] == xi)]
                if row.empty:
                    hover_mat[zi, xi] = ""
                else:
                    r = row.iloc[0]
                    hover_mat[zi, xi] = (
                        f"<b>{ARCH_LABEL[zi]} · {OPT_LABEL[i]} · LR={LR_LABEL[xi]}</b><br>"
                        f"val_acc: {r['val_acc_final_mean']:.4f} ± {r['val_acc_final_std']:.4f}<br>"
                        f"macro_f1: {r['macro_f1_mean']:.4f}<br>"
                        f"val_loss CE: {r['val_loss_final_mean']:.4f}<br>"
                        f"best_epoch: {r['best_epoch_mean']:.1f}"
                    )
        fig.add_trace(go.Heatmap(
            z=mat, x=LR_LABEL, y=ARCH_LABEL,
            colorscale="Viridis", zmin=vmin, zmax=vmax,
            text=text, texttemplate="%{text}",
            textfont=dict(color="white", size=10),
            hovertext=hover_mat, hoverinfo="text",
            showscale=False,
        ), row=2, col=i + 1)
        fig.update_xaxes(title_text="LR" if i == 1 else None,
                          color=TEXT, gridcolor=GRID, row=2, col=i+1)
        fig.update_yaxes(title_text="Arch" if i == 0 else None,
                          color=TEXT, gridcolor=GRID, row=2, col=i+1)

    # ===== Layout =====
    fig.update_layout(
        title=dict(
            text=("<b>Stage 2 del Cross-experimento</b> · Grid 3D LR × Optimizer × Arquitectura<br>"
                  "<sub style='color:#8b949e'>60 celdas · 3 seeds × 5 folds por celda = 900 corridas · "
                  "batch heredado de stage 1 · ES patience=20 · arrastrá el cubo para rotar; "
                  "hover sobre cualquier celda para ver métricas</sub>"),
            font=dict(color=TEXT, size=18), x=0.02, xanchor="left",
        ),
        paper_bgcolor=PAPER, plot_bgcolor=BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif"),
        height=900, width=1500,
        margin=dict(t=130, l=20, r=80, b=40),
    )
    # subplot titles color
    for ann in fig.layout.annotations:
        ann.font.color = TEXT

    fig.write_html(str(OUT), include_plotlyjs="cdn",
                    config={"displaylogo": False, "responsive": True})
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
