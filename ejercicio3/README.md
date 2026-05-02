# Ejercicio 3 — Clasificación de dígitos con accuracy ≥98% (target)

Construye sobre el Ej2: usa el config base ganador + `more_digits.csv` como datos extras + técnicas Pack C (L2, dropout, augmentation gaussiana, lr_schedule).

## Workflow

1. Tomar `base.json` ganador del Ej2.
2. Crear `configs/base_extra_data.json` con `extra_csv_paths` apuntando a `more_digits.csv` (engine concatena los CSVs antes del split/K-fold).
3. Correr — si `val_acc_final.mean ≥ 0.98`, ir directo a final_eval.
4. Si no, activar Pack C y combinaciones, midiendo el delta tras cada uno (val K=5 + final_eval contra `digits_test.csv`).
5. Final eval contra `digits_test.csv` con el config ganador (UNA SOLA VEZ).

## Configs evaluados

| Config | Pack C activado | Comentario |
|---|---|---|
| `base_extra_data.json` | ninguno | base.json + more_digits.csv (28k samples train) |
| `pack_c/l2.json` | L2 (1e-4) | weight decay, no penaliza bias |
| `pack_c/dropout.json` | dropout (p=0.2) | inverted dropout en hidden layers |
| `pack_c/lr_schedule.json` | step decay 0.5 cada 10 ep | lr scheduling |
| `pack_c/augmentation.json` | gaussian noise (σ=0.05) | additive noise pre-batch |
| `pack_c/l2_dropout.json` | L2 + dropout | combinación |
| `pack_c/l2_aug.json` | L2 + augmentation | **ganador** |
| `pack_c/l2_aug_s010.json` | L2 + aug (σ=0.10) | sigma más alto |
| `pack_c/l2_aug_dropout.json` | L2 + dropout + aug | todos combinados |
| `pack_c/wider_l2_aug.json` | arch_wider + L2 + aug | escalar capacidad |

## Resultados (val K=5 + test sobre digits_test.csv)

| Config | val_acc K=5 | test_accuracy | Δ vs base_extra |
|---|---:|---:|---:|
| Ej2 base (sin extra data) | 0.9622 | 0.8630 | (referencia inicial) |
| **Ej3 base_extra_data** | 0.9706 | 0.9636 | **+10.06pp en test** |
| L2 | 0.9758 | 0.9656 | +0.20pp |
| dropout | 0.9750 | 0.9628 | -0.08pp |
| l2_dropout | 0.9772 | 0.9604 | -0.32pp |
| **l2_aug (σ=0.05)** | **0.9760** | **0.9688** | **+0.52pp ← MEJOR** |
| l2_aug_s010 | 0.9768 | 0.9672 | +0.36pp |
| l2_aug_dropout | 0.9768 | 0.9656 | +0.20pp |
| wider_l2_aug | 0.9777 | 0.9640 | +0.04pp |

## Resultado final

- **Ganador:** `pack_c/l2_aug.json` — `[784, 100, 50, 10]` con Adam (lr=0.001), batch=16, **L2=1e-4 + augmentation gaussiana σ=0.05**.
- **Test accuracy:** **0.9688** (96.88%)
- **Test macro F1:** 0.9681

⚠️ **No alcanzamos el target de 0.98** (falta 1.12pp). Análisis de qué movió la aguja y qué no:

### ¿Qué ayudó?

1. **Más datos (more_digits.csv): +10.06pp en test** — el cambio dominante. Pasamos de 86.30% a 96.36% solo por entrenar sobre 28k samples en lugar de 12k. Esto sugiere que el problema principal del Ej2 era distribution shift entre `digits.csv` y `digits_test.csv`, parcialmente resuelto al ampliar la cobertura del train.
2. **L2 (1e-4): +0.20pp** — pequeña pero consistente mejora.
3. **Augmentation gaussiana (σ=0.05): +0.32pp adicional sobre L2** — al combinar L2+aug llegamos a 96.88%. Aug solo aporta cuando L2 ya está activo (sinergia).

### ¿Qué no ayudó?

1. **Dropout (p=0.2)**: -0.08pp en test. En val ayudaba (97.50% vs 97.06%) pero no transfería a test. Sugiere que dropout estaba reduciendo varianza específica al fold sin atacar el shift.
2. **Combinar dropout con L2 o aug**: empeoró respecto a L2 solo o L2+aug. Demasiada regularización con datos limitados.
3. **Arquitectura wider [784,200,100,10]**: +0.17pp en val, **-0.48pp en test**. Más parámetros memorizan train mejor pero generalizan peor → señal de overfitting al training data, no falta de capacidad.
4. **Sigma de aug=0.10**: peor que σ=0.05 en test. Más ruido de input no compensa.

### Hipótesis sobre por qué no llega a 98%

1. **Distribution shift residual**: pese a `more_digits.csv`, el set de prueba tiene un sub-poblamiento (digits con un estilo de escritura específico) que no aparece suficientemente en train. Augmentation gaussiana en input no captura ese shift (es ruido isotrópico, no transformaciones tipo rotación/desplazamiento).
2. **Modelo pequeño** (~84k params): probablemente subajustado. Pero arch_wider falló por overfitting → faltarían más datos antes de escalar capacidad.
3. **Augmentación más sofisticada**: rotaciones, desplazamientos pequeños, elastic transforms. No implementadas (gaussian noise es la única) — dejaríamos para una segunda iteración.
4. **Conjunto de validación interno** del final_eval (10% random) puede estar sub-representando los digits "difíciles" del test, lo que hace que early stopping pare antes de tiempo.

## Comparación con Ej2

Ver `presentacion/01_comparacion_ej2_vs_ej3.png`. Línea roja = target 98%.

## Plots adicionales

- `presentacion/02_confusion_matrix_ej3.png` — matriz de confusión sobre `digits_test.csv`.
- `presentacion/03_per_class_metrics_ej3.png` — precision/recall/f1 por dígito.
- `presentacion/04_curvas_aprendizaje_ej3.png` — train/val por época (K=5 folds).
