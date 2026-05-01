---
name: TP3 completion plan status
description: Estado actual del plan de completion del TP3 (al 2026-05-01, después de Phase A)
type: project
originSessionId: 6a5ca89e-5f83-4840-90c3-26115775244c
---
Plan de completion del TP3 aprobado el 2026-05-01. Cubre: validación faltante (step perceptron AND, MLP XOR), Ejercicio 2 (clasificación de dígitos con MLP), y Ejercicio 3 (target ≥98%).

**Why:** TP3 ya tiene Ej1 (fraud detection) en estado de presentación. El compañero Tomás trabajó Ej1; el usuario sigue con el resto. Filosofía elegida: "híbrido pragmático" — cumplir el "como mínimo" del enunciado (LR + arquitectura + optimización) con un análisis profundo del modelo ganador. Workflow disciplinado: engine puro (cero plots adentro), Fase 1 exploratoria (k=1) → `base.json` → Fase 2 one-at-a-time (k=5) → Ej3 + Pack C si necesario → final_eval contra `digits_test.csv`.

**Estado al 2026-05-01:**
- **Phase A completa (Tasks 1-25 + 26):** engine MLP completo from-scratch con NumPy. Activations, losses, initializers, optimizers (SGD/Momentum/Adam), data utilities (parse, kfold, batch iterator), metrics, MLP class (forward/backward/fit/predict/save/load), train.py CLI con multiprocessing, README. Ej0 validation: step perceptron AND + XOR configs + tests. **78/78 tests pasan.**
- **Task 26 ✅:** 4 configs de architecture sweep para Ej2 Fase 1.1 creados.
- **Task 27 pendiente:** correr los 16 sweeps de Fase 1 (arch + opt + lr + batch), elegir base.json, correr base.json con K-fold=5. Estimado: ~60-90 min de cómputo.
- **Tasks 28-39 pendientes:** Fase 2 sweeps con k=5, plots, final_eval Ej2, Ej3 con Pack C, comparación, READMEs, docs.

**Worktree:** `.worktrees/tp3-mlp` en branch `tp3-mlp`. **Branch pusheado a `origin/tp3-mlp`** (commit head `6bbdbae`).

**Bugs reales encontrados y resueltos durante Phase A:**
- `predict()` para tanh-binario devolvía `{0,1}` en vez de bipolar `{-1,+1}` (commit `0d551d9`).
- `multiclass_metrics` con labels bipolares causaba wrap-around silencioso en confusion matrix → agregamos `_normalize_labels` helper en train.py.
- Mean/std rows del run_summary.csv perdían sus string labels por dict-spread overwrite (commit `b7cbea9`).
- numerical_grad silenciosamente devolvía garbage para int dtypes (commit `7ae9477`).
- identity activation aliasaba su input → fix en commit `30b76a0`.

**How to apply (cuando se retoma):**
- Spec: `/home/nico/Desktop/SIA/SIA-TP3/docs/superpowers/specs/2026-05-01-tp3-completion-design.md`
- Plan: `/home/nico/Desktop/SIA/SIA-TP3/docs/superpowers/plans/2026-05-01-tp3-completion.md` (4495 líneas, 39 tareas)
- Worktree: `cd /home/nico/Desktop/SIA/SIA-TP3/.worktrees/tp3-mlp` (preserva la sesión).
- Próxima tarea: **Task 27** — correr `for cfg in ejercicio2/configs/sweeps_fase1/arch_*.json; do .venv/bin/python -m mlp.train --config $cfg --csv-root . --output-dir ejercicio2/output --workers 1; done`. arch_50 ya corrió (val_acc 95.78%), faltan arch_100, arch_128_64, arch_100_50.
- **No tocar `ejercicio1/`** ni `linear_perceptron.py` / `nonlinear_perceptron.py` de Ej0.
- Engine en `mlp/` a nivel raíz. Solo NumPy/Pandas; Matplotlib solo en `analisis/` (cuando se cree).
- `digits_test.csv` se evalúa **una única vez** al final.
