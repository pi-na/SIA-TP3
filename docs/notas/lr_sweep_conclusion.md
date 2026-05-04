# Learning Rate Sweep — Conclusión

## Fase 1.3 (Exploración inicial, k=1)

La tabla muestra 5 valores probados:

| LR | val_acc | macro_f1 | best_epoch | total_epochs | Comportamiento |
|---|---|---|---|---|---|
| **0.001** | **96.74%** | **0.863** | **7** | **18** | ✓ Óptimo |
| 0.0005 | 96.54% | 0.864 | 17 | 28 | Lento |
| 0.0001 | 96.22% | 0.861 | 47 | 50 | No converge |
| 0.005 | 96.02% | 0.857 | 4 | 15 | Overshoot moderado |
| 0.01 | 95.30% | 0.850 | 2 | 13 | Overshoot fuerte |

**Criterios clave:**
1. **Máximo claro en val_acc**: 96.74% (forma de parábola: sube a la izquierda, baja a la derecha)
2. **Convergencia rápida**: best_epoch=7 (versus 47 con lr=0.0001)
3. **No overshoot**: LR ≥ 0.005 diverge rápido (best_ep=2-4 son demasiado tempranos)

## Fase 2 (Validación con K-fold=5)

Se confirmó con un sweep más riguroso:

| LR | val_acc | best_epoch | Tiempo |
|---|---|---|---|
| **0.001** | **96.22%** | **4.6** | **449 s** |
| 0.0001 | 95.79% | 31.6 | 1390 s (3.1× más lento) |
| 0.01 | 94.71% | 4.6 | 455 s |
| 0.1 | 29.00% | 7.2 | catastrophic divergence |
| 10 | 12.48% | 11.6 | NaN (loss explota) |
