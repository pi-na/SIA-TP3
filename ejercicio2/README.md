# Ejercicio 2 — Clasificación de dígitos con MLP

Clasificación de dígitos manuscritos (28×28 → 10 clases) con el MLP genérico (módulo `mlp/` en la raíz del repo).

## Workflow

### Fase 1 — Búsqueda de configuración base (k_folds=1)

Sweeps coarse-to-fine secuenciales: arquitectura → optimizador → LR → batch size. Cada sub-fase usa el ganador de la anterior.

```bash
for cfg in configs/sweeps_fase1/arch_*.json; do
    python3 -m mlp.train --config "$cfg" --csv-root .. --output-dir output --workers 1
done
# Inspeccionar val_acc_final, elegir ganadora
# Repetir con opt_*, lr_*, batch_*
```

Output: `configs/base.json` consolidado con los HP ganadores.

**Resultado Fase 1:**

| Sub-fase | Ganador | val_acc | Notas |
|---|---|---|---|
| 1.1 arquitectura | `[784, 100, 50, 10]` | 96.74% | Empate técnico con `[784, 128, 64, 10]`, elegido por parsimonia (35% más rápido, mismo macro_f1) |
| 1.2 optimizador | `adam` (lr=0.001) | 96.74% | momentum 96.34%, sgd 94.81% (no convergió en 50 ep) |
| 1.3 learning rate | `0.001` | 96.74% | lr=0.0001 no converge; lr≥0.005 overshoot |
| 1.4 batch size | `16` | 96.74% | Mejor macro_f1 (0.8646), best_ep=3, 12s |

**`base.json` con K-fold=5:** val_acc 0.9622 ± 0.0043, macro_f1 0.857 ± 0.006.

### Fase 2 — One-at-a-time con K-fold=5

Variar un solo HP a la vez sobre `base.json` para tener comparativas limpias.

```bash
for cfg in configs/sweeps_fase2/*.json; do
    python3 -m mlp.train --config "$cfg" --csv-root .. --output-dir output --workers 5
done
```

Sweeps: LR (5 valores incl. extremos 0.1 y 10), arquitectura (4: shallow, base, wider, deeper), optimizador (3: sgd/momentum/adam), inicializador (3: uniform/he/xavier).

### Plotting

```bash
python3 analisis/plot_learning_curves.py --run-dir output/base_ej2_<ts> \
    --out presentacion/01_curvas_aprendizaje_base.png
python3 analisis/plot_sweep.py --run-dirs $(ls -d output/fase2_lr_*) \
    --metric val_acc_final --label-by lr \
    --out presentacion/02_sweep_lr.png
python3 analisis/plot_confusion_matrix.py --run-dir output/base_ej2_<ts> \
    --out presentacion/04_confusion_matrix_base.png
python3 analisis/plot_per_class_metrics.py --run-dir output/base_ej2_<ts> \
    --out presentacion/05_per_class_metrics_base.png
```

### Final eval

```bash
python3 final_eval.py --config configs/base.json --csv-root .. \
    --output-dir output_final
```

**`digits_test.csv` se evalúa UNA SOLA VEZ**, con el config ganador congelado.

## Estructura de output

Cada corrida en `output/<model_name>_<ts>/` produce:

| Archivo | Contenido |
|---|---|
| `config.json` | copia del input |
| `run_summary.csv` | métricas finales por fold + filas mean/std |
| `epoch_history.csv` | métricas por época (train + val) |
| `predictions.csv` | scores out-of-fold por clase |
| `confusion_matrix.csv` | matriz por fold (formato stacked) |
| `weights.npz` | pesos finales por fold |

## Decisiones de diseño

Ver `docs/superpowers/specs/2026-05-01-tp3-completion-design.md`.

## Resultados finales

**Modelo base ganador** (`configs/base.json`):
- Arquitectura: `[784, 100, 50, 10]` con `[relu, relu, softmax]`.
- Optimizer: Adam (lr=0.001).
- Batch size: 16.
- Initializer: auto (he para relu, xavier para softmax).
- Epochs: 50, early stopping patience: 10.

**K-fold=5 sobre digits.csv:**
| Métrica | mean ± std |
|---|---|
| val_acc | 0.9622 ± 0.0043 |
| macro_f1 | 0.857 ± 0.006 |
| best_epoch | 4.6 |

**Validación de la elección — Sweeps Fase 2 (K=5):**

| Dimensión | Ganador | Empate técnico con base | Comentario |
|---|---|---|---|
| LR | 0.001 | 0.0001 (96.22% vs 95.79%) | 0.0001 cerca pero converge en 31 ep vs 5; ≥0.01 overshoot/diverge |
| Arch | wider [784,200,100,10] | base, deeper | wider 0.17pp mejor pero 3.6× más lento → base mantiene óptimo |
| Optimizer | momentum (96.29%) | adam (96.22%) | empate; adam best_ep menor → keep adam |
| Init | he/xavier/uniform | sí (todos ~96%) | empate total (std ≈0.3%) |

**Final eval (digits_test.csv, evaluado UNA SOLA VEZ):**
| Métrica | Valor |
|---|---|
| test_accuracy | **0.8630** |
| test_macro_f1 | 0.8169 |
| test_weighted_f1 | 0.8207 |

⚠️ **Brecha train→test de ~10pp** (val 96.22% → test 86.30%). Sugiere distribution shift en `digits_test.csv` respecto al train. Esto motiva Ejercicio 3 (más datos vía `more_digits.csv` + posible regularización Pack C).
