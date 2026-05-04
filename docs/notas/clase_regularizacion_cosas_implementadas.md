# Clase de regularización — qué implementamos vs. qué no

Esta nota cierra el matching entre el contenido del PDF `docs/clase_regularizacion/regularizacion.pdf` (la clase teórica) y lo que efectivamente implementamos en `mlp/` y usamos en el Ej 3.

## Tabla de matching 1:1

| Técnica del PDF | Slide | Implementada en `mlp/` | Usada en Ej 3 | Resultado |
|---|---|---|---|---|
| **Early stopping** | 14 | Sí — `network.py:200-210` (val_loss + patience=10) | Sí | Activo en todos los configs (Ej2 y Ej3) |
| **Augmentation — gaussiano** | 18 | Sí — `network.py:160-165` | Sí (σ = 0.05) | **+0.32 pp** sobre L2 (parte del ganador) |
| Augmentation — rotaciones | 18 | **No** | — | — |
| Augmentation — traslaciones | 18 | **No** | — | — |
| Augmentation — cambios de escala | 18 | **No** | — | — |
| **L2 / Weight Decay** | 20-25 | Sí — `network.py:168-174` (no penaliza bias) | Sí (λ = 1e-4) | **+0.20 pp** consistente |
| **Dropout** | 26 (mención) | Sí — `network.py:86-94` (inverted dropout) | Probado, descartado | Gana val, pierde test (−0.08 pp) |
| Modelos de ensamble | 26 | No (fuera de alcance) | — | — |
| Aprendizaje semi-supervisado | 26 | No (fuera de alcance) | — | — |
| Entrenamiento adversarial | 26 | No (fuera de alcance) | — | — |

## Lectura corta

**Lo que la clase enseña y nosotros usamos**:
- Early stopping (slide 14) — siempre activo.
- L2 con la fórmula exacta `E_reg = E + (1/2) λ ‖w‖²` (slides 20–25).
- Augmentation por ruido gaussiano (slide 18, una de las cuatro formas listadas).
- Dropout (slide 26, sólo mencionado en el PDF) — implementado pero descartado.

**Lo que la clase enseña y NO implementamos**:
- Tres formas de augmentation que el slide 18 lista junto al gaussiano: **rotaciones, traslaciones, cambios de escala**.
- Esto importa: el shift del Ej 2 → Ej 3 parece ser geométrico (estilos de escritura), no de ruido isotrópico. Esas tres formas son la hipótesis principal de por qué quedamos en 96.88% y no en 98%.

**Lo que usamos pero la clase no cubre**:
- `lr_schedule` (step decay): familia "modificar η durante entrenamiento" del PDF de optimizadores, pero con regla distinta a la del slide (que era `Δη = +a` o `−bη` según el error).

## Detalles de implementación (referencia rápida)

```python
# Early stopping (network.py:200-210)
if val_loss < best_val_loss:
    best_val_loss = val_loss
    best_weights = [W.copy() for W in self.weights]
    epochs_no_improvement = 0
else:
    epochs_no_improvement += 1
    if epochs_no_improvement >= patience:
        self.weights = best_weights  # restaurar mejor modelo
        break

# L2 (network.py:168-174) — bias NO penalizado
reg_grad = np.zeros_like(W)
reg_grad[:, 1:] = l2 * W[:, 1:]   # columna 0 = bias, no se toca

# Augmentation gaussiana (network.py:160-165)
if aug_sigma > 0:
    X_batch = X_batch + np.random.normal(0, aug_sigma, X_batch.shape)

# Inverted dropout (network.py:86-94)
if training and dropout_p > 0:
    mask = (np.random.rand(*a.shape) > dropout_p) / (1.0 - dropout_p)
    a = a * mask
```

## Por qué dropout no transfirió

Dropout reduce la varianza al fold de cross-validation (por eso ganó en val K-fold), pero **no corrige distribution shift**: si train y test son poblaciones distintas, dropout no pone ningún caso de test en la mesa de entrenamiento. Más detalle: `docs/notas/explicacion_distribution_shift.md`.

## Ver también

- PDF: `docs/clase_regularizacion/regularizacion.pdf`
- Backing teórico completo: `docs/notas/decisiones_y_backing_teorico.md` § 9
- Resultados detallados Ej 3: `docs/notas/resultados_ejercicio3_explicacion.md`
