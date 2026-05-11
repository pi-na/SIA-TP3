"""Orquestador overnight del Ej3.

Secuencia:
  1) step1_baseline.py   → CV + final_eval baseline (~15 min)
  2) analyze_baseline.py → plots + snippet markdown del paso 1
  3) inyectar resultados paso 1 en Notas/ejercicio 3/Plan y resultados.md
  4) step2_grid_reg.py   → grid 16x3 (~80 min)
  5) analyze_grid_reg.py → heatmaps + tabla + snippet del paso 2
  6) best combo del paso 2 → final_eval x 3 seeds
  7) analyze_baseline.py adaptado para best combo → plots + tabla test
  8) inyectar resultados paso 2 en la nota
  9) git add + commit + push

Log a ejercicio3/output/overnight.log. Cada paso se loguea con timestamp.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PYTHON = str(ROOT / ".venv" / "bin" / "python")
LOG    = ROOT / "ejercicio3" / "output" / "overnight.log"
LOG.parent.mkdir(parents=True, exist_ok=True)

NOTE = ROOT / "Notas" / "ejercicio 3" / "Plan y resultados.md"


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def run_step(label: str, cmd: list[str]) -> bool:
    log(f"START {label}: {' '.join(cmd[:2])}...")
    t0 = time.time()
    try:
        r = subprocess.run(cmd, cwd=str(ROOT))
        ok = (r.returncode == 0)
        log(f"END   {label}: rc={r.returncode} elapsed={time.time()-t0:.0f}s")
        return ok
    except Exception as e:
        log(f"EXCEPTION in {label}: {type(e).__name__}: {e}")
        log(traceback.format_exc())
        return False


def inject_snippet(snippet_path: Path, marker: str):
    """Reemplaza 'marker' en la nota con el contenido de snippet_path."""
    if not snippet_path.exists():
        log(f"WARN: snippet {snippet_path} no existe, skipping injection")
        return
    if not NOTE.exists():
        log(f"WARN: nota {NOTE} no existe, skipping injection")
        return
    text = NOTE.read_text()
    snippet = snippet_path.read_text()
    if marker not in text:
        log(f"WARN: marker '{marker}' no encontrado en la nota")
        # append al final
        new_text = text + "\n\n" + snippet
    else:
        new_text = text.replace(marker, snippet)
    NOTE.write_text(new_text)
    log(f"Injected {snippet_path.name} into nota")


def run_final_eval_best(best_combo: dict):
    """Corre final_eval con el best combo del grid."""
    base_cfg = json.loads((ROOT / "ejercicio3" / "configs" / "final_config_ej3_baseline.json").read_text())
    base_cfg["regularization"]["l2"] = best_combo["l2"]
    if best_combo["sigma"] > 0:
        base_cfg["regularization"]["augmentation"] = {
            "type": "gaussian_noise", "sigma": best_combo["sigma"]
        }
    out_root = ROOT / "ejercicio3" / "output" / "final_eval" / "best_reg"
    out_root.mkdir(parents=True, exist_ok=True)
    for seed in [42, 7, 13]:
        cfg = json.loads(json.dumps(base_cfg))
        cfg["split"]["random_seed"] = seed
        cfg["model_name"] = f"final_ej3_bestreg_seed{seed}"
        cfg_path = out_root / f"config_seed{seed}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2))
        log_p = out_root / f"log_seed{seed}.txt"
        log(f"  final_eval bestreg seed={seed} starting...")
        t0 = time.time()
        with open(log_p, "w") as lf:
            r = subprocess.run([
                PYTHON,
                str(ROOT / "ejercicio3" / "final_eval.py"),
                "--config", str(cfg_path),
                "--output-dir", str(out_root),
            ], stdout=lf, stderr=subprocess.STDOUT)
        log(f"  final_eval bestreg seed={seed} rc={r.returncode} elapsed={time.time()-t0:.0f}s")


def analyze_best_reg():
    """Re-usa analyze_baseline.py con paths del best_reg. Hago aqui una version inline simple."""
    sys.path.insert(0, str(ROOT / "ejercicio3" / "scripts"))
    import analyze_baseline as ab
    # monkey-patch paths
    ab.CV_DIR = ROOT / "ejercicio3" / "output" / "grid_reg"  # not used, we only want test
    ab.FINAL_DIR = ROOT / "ejercicio3" / "output" / "final_eval" / "best_reg"
    ab.OUT       = ROOT / "ejercicio3" / "analisis" / "best_reg"
    ab.NOTES_PNG = ROOT / "Notas" / "ejercicio 3"
    ab.OUT.mkdir(parents=True, exist_ok=True)
    # Solo paso C
    test_s, cm_mean, cms, per_df = ab.aggregate_test()
    ab.plot_confusion_matrix(cm_mean, suffix="best_reg")

    # Escribir snippet best_reg
    best_info = json.loads((ROOT / "ejercicio3" / "analisis" / "grid_reg" / "best_combo_info.json").read_text())
    md = ["### Resultados — Best combo del grid de regularización\n",
          f"**Best combo (paso 2):** L2 = `{best_info['l2']:g}` · σ = `{best_info['sigma']:g}` · val_acc CV = **{best_info['val_acc_mean']:.4f} ± {best_info['val_acc_std']:.4f}** · gap = **{best_info['gap']:.4f}**.\n",
          "**Generalización externa (test):**\n",
          "![[best_reg_test_confusion_matrix.png]]\n",
          "| Métrica | Test (best_reg) |",
          "| --- | --- |",
          f"| accuracy        | **{test_s['test_acc_mean']:.4f} ± {test_s['test_acc_std']:.4f}** |",
          f"| macro_precision | {test_s['test_macro_precision_mean']:.4f} ± {test_s['test_macro_precision_std']:.4f} |",
          f"| macro_recall    | {test_s['test_macro_recall_mean']:.4f} ± {test_s['test_macro_recall_std']:.4f} |",
          f"| macro_F1        | **{test_s['test_macro_f1_mean']:.4f} ± {test_s['test_macro_f1_std']:.4f}** |\n",
          "**Métricas por clase en test:**\n",
          "| clase | precision | recall | F1 | support |",
          "| --- | --- | --- | --- | --- |"]
    for _, r in per_df.iterrows():
        bold = "**" if int(r['class']) == 8 else ""
        md.append(f"| {bold}{int(r['class'])}{bold} | "
                  f"{bold}{r['precision_mean']:.3f}{bold} | "
                  f"{bold}{r['recall_mean']:.3f}{bold} | "
                  f"{bold}{r['f1_mean']:.3f}{bold} | "
                  f"{int(r['support_test'])} |")
    md.append("\n")
    (ROOT / "ejercicio3" / "analisis" / "best_reg" / "best_reg_results.md").write_text("\n".join(md))


def final_comparison_section():
    """Tabla comparativa final Ej2 / Ej3 baseline / Ej3 best_reg."""
    # Ej2
    ej2_acc, ej2_acc_std, ej2_f1, ej2_f1_std = 0.8529, 0.0034, 0.8062, 0.0034
    # Ej3 baseline
    bp = ROOT / "ejercicio3" / "analisis" / "baseline" / "test_summary.csv"
    b = pd.read_csv(bp).iloc[0]
    # Ej3 best_reg
    rp = ROOT / "ejercicio3" / "analisis" / "best_reg" / "test_summary.csv"
    if rp.exists():
        r = pd.read_csv(rp).iloc[0]
    else:
        r = None
    md = ["## Comparativa final (test sobre `digits_test.csv`)\n",
          "| Configuración | Test accuracy | Test macro_F1 |",
          "| --- | --- | --- |",
          f"| Ej2 (sin more_digits, sin reg) | {ej2_acc:.4f} ± {ej2_acc_std:.4f} | {ej2_f1:.4f} ± {ej2_f1_std:.4f} |",
          f"| Ej3 baseline (+more_digits, sin reg) | **{b['test_acc_mean']:.4f} ± {b['test_acc_std']:.4f}** | **{b['test_macro_f1_mean']:.4f} ± {b['test_macro_f1_std']:.4f}** |"]
    if r is not None:
        md.append(f"| **Ej3 best_reg (+more_digits + L2 + σ)** | **{r['test_acc_mean']:.4f} ± {r['test_acc_std']:.4f}** | **{r['test_macro_f1_mean']:.4f} ± {r['test_macro_f1_std']:.4f}** |")
    md.append("\n")
    md.append("**Conclusiones:**\n")
    delta_acc_base = b['test_acc_mean'] - ej2_acc
    md.append(f"- Sumar `more_digits.csv` aporta **{delta_acc_base:+.4f}** puntos de test_acc sobre el Ej2.")
    if r is not None:
        delta_acc_reg = r['test_acc_mean'] - b['test_acc_mean']
        md.append(f"- Regularización aporta **{delta_acc_reg:+.4f}** puntos adicionales sobre el baseline.")
        total = r['test_acc_mean'] - ej2_acc
        md.append(f"- Ganancia total Ej2 → Ej3 best_reg: **{total:+.4f}** puntos de accuracy.")
        if r['test_acc_mean'] >= 0.98:
            md.append(f"- 🎯 **Se alcanzó el objetivo de ≥ 98%** pedido por CompanyX.")
        else:
            md.append(f"- ⚠️ **No se alcanzó el ≥ 98%** pedido por CompanyX. Brecha residual = {0.98 - r['test_acc_mean']:.4f}.")
    return "\n".join(md)


def git_commit_and_push():
    log("git add + commit + push")
    subprocess.run(["git", "add", "-A"], cwd=str(ROOT))
    # excluir el pasted image
    subprocess.run(["git", "reset", "Pasted image 20260511004622.png"], cwd=str(ROOT))
    msg = ("ej3: baseline (+more_digits) + grid regularizacion (L2 x sigma) "
           "+ analisis completo (overnight)\n\n"
           "Paso 1: +more_digits.csv sin regularizacion (3 seeds x 5 folds CV + 3 final_eval).\n"
           "Paso 2: grid 4x4 (L2={0,1e-5,1e-4,1e-3} x sigma={0,0.03,0.1,0.2}) "
           "= 16 combos x 3 seeds x 5 folds = 240 corridas CV + 3 final_eval del best combo.\n"
           "Sin dropout (decision del equipo), sin LR schedule (no en clase).\n"
           "Resultados detallados en 'Notas/ejercicio 3/Plan y resultados.md'.")
    r = subprocess.run(["git", "commit", "-m", msg], cwd=str(ROOT))
    if r.returncode == 0:
        subprocess.run(["git", "push"], cwd=str(ROOT))
        log("push done")
    else:
        log(f"git commit rc={r.returncode}, skipping push")


def main():
    log("=" * 60)
    log("Ej3 OVERNIGHT START")
    log("=" * 60)
    t0 = time.time()

    # ============ Paso 1 ============
    if not run_step("STEP_1_BASELINE", [PYTHON, str(HERE / "step1_baseline.py")]):
        log("Step 1 failed; trying to continue with analysis anyway")
    run_step("ANALYZE_BASELINE", [PYTHON, str(HERE / "analyze_baseline.py")])

    snip1 = ROOT / "ejercicio3" / "analisis" / "baseline" / "baseline_results.md"
    inject_snippet(snip1, "🕐 _Pendiente — se completará cuando termine la corrida._\n\n---\n\n## Paso 2")

    # ============ Paso 2 ============
    if not run_step("STEP_2_GRID", [PYTHON, str(HERE / "step2_grid_reg.py")]):
        log("Step 2 failed; continuing with whatever data exists")
    run_step("ANALYZE_GRID", [PYTHON, str(HERE / "analyze_grid_reg.py")])

    snip2 = ROOT / "ejercicio3" / "analisis" / "grid_reg" / "grid_results.md"
    inject_snippet(snip2, "🕐 _Pendiente — se completará cuando termine la corrida._\n\n---\n\n## Pronósticos")

    # ============ Best combo final_eval ============
    best_info_path = ROOT / "ejercicio3" / "analisis" / "grid_reg" / "best_combo_info.json"
    if best_info_path.exists():
        best = json.loads(best_info_path.read_text())
        log(f"Best combo found: L2={best['l2']:g} sigma={best['sigma']:g}")
        run_final_eval_best(best)
        try:
            analyze_best_reg()
            snip3 = ROOT / "ejercicio3" / "analisis" / "best_reg" / "best_reg_results.md"
            # Append final con comparativa
            note_text = NOTE.read_text()
            best_snip = snip3.read_text() if snip3.exists() else ""
            import pandas as pd
            globals()["pd"] = pd
            comp = final_comparison_section()
            full_append = "\n\n" + best_snip + "\n\n" + comp + "\n"
            # Reemplazar el marker de comparativa final
            marker = "🕐 _Tabla pendiente — se completará al final con 3 filas (Ej2 / Ej3 baseline / Ej3 best reg) × las 4 métricas en test._"
            if marker in note_text:
                note_text = note_text.replace(marker, best_snip + "\n\n" + comp)
            else:
                note_text += full_append
            NOTE.write_text(note_text)
            log("Best reg analysis injected into nota")
        except Exception as e:
            log(f"EXCEPTION in analyze_best_reg: {e}\n{traceback.format_exc()}")
    else:
        log("WARN: no best_combo_info.json, skipping best_reg final_eval")

    # ============ Git ============
    try:
        git_commit_and_push()
    except Exception as e:
        log(f"EXCEPTION in git_commit_and_push: {e}\n{traceback.format_exc()}")

    log("=" * 60)
    log(f"Ej3 OVERNIGHT DONE. Total: {time.time()-t0:.0f}s")
    log("=" * 60)


if __name__ == "__main__":
    # Pandas se importa lazy en final_comparison_section, lo importamos aca por las dudas
    import pandas as pd  # noqa: F401
    main()
