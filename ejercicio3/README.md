# Ejercicio 3 — Clasificación de dígitos con accuracy ≥98%

Construye sobre el Ej2: usa el config base ganador + `more_digits.csv` como datos extras + (si necesario) técnicas Pack C (L2, dropout, LR scheduling, ruido gaussiano).

## Workflow

1. Tomar `base.json` ganador del Ej2.
2. Crear `configs/base_extra_data.json` con `extra_csv_paths` apuntando a `more_digits.csv` (engine concatena).
3. Correr — si `val_acc_final.mean ≥ 0.98`, ir directo a final_eval.
4. Si no, activar Pack C secuencialmente (l2 → dropout → lr_schedule → augmentation), midiendo el delta tras cada uno.
5. Final eval contra `digits_test.csv`.

## Configs

| Config | Descripción | Pack C activado |
|---|---|---|
| `base_extra_data.json` | Ej2 base + more_digits.csv | ninguno |
| `pack_c/l2.json` | base + L2 weight decay (1e-4) | l2 |
| `pack_c/dropout.json` | base + dropout (p=0.2) | dropout |
| `pack_c/lr_schedule.json` | base + step decay 0.5 cada 20 ep | lr_schedule |
| `pack_c/augmentation.json` | base + gaussian noise (σ=0.05) | augmentation |

## Resultado final

- **Ganador:** _a completar_
- **Test accuracy:** _a completar_ (target ≥0.98)

## Comparación con Ej2

Ver `presentacion/01_comparacion_ej2_vs_ej3.png`.
