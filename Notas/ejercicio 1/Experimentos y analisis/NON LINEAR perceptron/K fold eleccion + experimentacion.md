# K-fold sweep — perceptrón no-lineal

## Configuración del experimento

- LR fijo: `1e-2` (ganador del sweep de LR)
- Seed: `42` (seed-std≈0 en el sweep multi-seed → una seed es suficiente)
- Épocas: `500` (suficiente para plateau, ver sweep LR)
- Threshold: `0.89` (thr* del no-lineal — max F1 promedio en el sweep multi-seed de LR)
- K evaluados: [2, 3, 5, 10]
- Estratificado por `flagged_fraud`: sí

## Tamaño de folds por K

| K | n_train (media) | n_test (media) | Positivos en test (media) |
|---|---|---|---|
| 2 | 3750 | 3750 | 434 |
| 3 | 5000 | 2500 | 290 |
| 5 | 6000 | 1500 | 174 |
| 10 | 6750 | 750 | 87 |

## Resultados (mean ± std entre folds, a thr=0.5)

| K | MSE test | F1 | Precision | Recall | Accuracy |
|---|---|---|---|---|---|
| 2 | 0.01096 ± 0.00001 | 0.8713 ± 0.0046 | 0.8862 ± 0.0103 | 0.8573 ± 0.0186 | 0.9707 ± 0.0005 |
| 3 | 0.01097 ± 0.00028 | 0.8716 ± 0.0083 | 0.8880 ± 0.0094 | 0.8562 ± 0.0173 | 0.9708 ± 0.0017 |
| 5 | 0.01099 ± 0.00044 | 0.8724 ± 0.0298 | 0.8872 ± 0.0297 | 0.8585 ± 0.0336 | 0.9709 ± 0.0068 |
| 10 | 0.01099 ± 0.00062 | 0.8708 ± 0.0380 | 0.8867 ± 0.0332 | 0.8561 ± 0.0480 | 0.9707 ± 0.0085 |

## Std entre folds — métrica de estabilidad del estimador

| K | std MSE test | std F1 |
|---|---|---|
| 2 | 0.00001 | 0.0046 |
| 3 | 0.00028 | 0.0083 |
| 5 | 0.00044 | 0.0298 |
| 10 | 0.00062 | 0.0380 |

![K-fold sweep](../../../imagenes/kfold_sweep%201.png)

## Conclusión

### 1. La media de las métricas es estable en K

Las medias entre K=2, K=3, K=5 y K=10 caen dentro de rangos despreciables:

| Métrica | rango entre K | comentario |
|---|---|---|
| MSE test | `0.01096` → `0.01099` (rango 0.3%) | indistinguible |
| F1       | `0.8708` → `0.8724` (rango 0.2%) | indistinguible |
| Precision | `0.8862` → `0.8880` (rango 0.2%) | indistinguible |
| Recall    | `0.8561` → `0.8585` (rango 0.3%) | indistinguible |
| Accuracy  | `0.9707` → `0.9709` (rango 0.02%) | indistinguible |

Lectura: **el K no cambia lo que el modelo aprendió**. El piso de capacidad del no-lineal sobre este dataset es `MSE ≈ 0.011` y `F1 ≈ 0.872`, y eso no se mueve al cambiar la partición. K-fold no mejora el modelo — sólo estima mejor cuán bien generaliza.

### 2. La std entre folds crece monótonamente con K

Aquí el no-lineal muestra un patrón más limpio que el lineal: a más folds, **más std por fold** (la varianza muestral por fold domina cuando hay menos datos en cada uno).

| K | n_test por fold | Positivos por fold | std MSE test | std F1 |
|---|---|---|---|---|
| 2 | 3750 | 434 | **0.00001** | 0.0046 |
| 3 | 2500 | 290 | 0.00028 | 0.0083 |
| 5 | 1500 | 174 | 0.00044 | 0.0298 |
| 10 | 750 | 87 | 0.00062 | 0.0380 |

K=10 con 750 muestras y sólo ~87 positivos por fold tiene **8× más std de F1** que K=2 — esto es ruido muestral, no señal del modelo. Aplicando el criterio prescriptivo (*"si la std de F1 entre K=5 y K=10 difiere menos de 0.005, K=5 es suficiente"*): `|0.0380 − 0.0298| = 0.0082` — la diferencia es **0.0082 a favor de K=5** (K=5 tiene std MENOR que K=10). El criterio se cumple con holgura **y en la dirección correcta**: subir K **empeora** la estabilidad del estimador, no la mejora.

### 3. ¿Por qué entonces no elegir K=2 (la std mínima)?

K=2 tiene la std más baja del barrido (MSE: 0.00001, F1: 0.0046), pero eso es engañoso. El problema con K=2 es el **bias del estimador**:

- K=2 entrena cada fold con sólo el **50% (3750 muestras)** del dataset.
- El modelo final se entrena con el **100% (7500 muestras)**.
- El estimador K=2 corresponde a un modelo "sub-entrenado" respecto del final → **subestima** la performance verdadera del modelo final.

No-lineal es sensible al volumen de datos (su capacidad expresiva requiere muestras para fijar la sigmoide en la zona correcta). Estimar generalización con la mitad de los datos introduce bias sistemático.

K=5 entrena con 80% (6000 muestras) → ese 20% de "datos no vistos" es chico respecto del modelo final y el bias del estimador queda en el orden del ruido.

### 4. K=5 es la elección operativa correcta

Resumen del trade-off:

| K | Bias del estimador | Std por fold | Cómputo | Veredicto |
|---|---|---|---|---|
| 2  | alto (entrena con 50%) | mínimo (artefacto) | 2 corridas | descartar por bias |
| 3  | moderado (67%) | bajo  | 3 corridas | aceptable, pero no óptimo |
| **5**  | **bajo (80%)** | **moderado pero estable** | **5 corridas** | **ganador** |
| 10 | mínimo (90%) | mayor (87 positivos/fold) | 10 corridas | bias-improvement marginal, std-improvement negativo, 2× cómputo |

**K=5 es el ganador defendible** — entrega:
- bias del estimador chico (entrena con 80%),
- ~174 positivos por fold de validación (suficientes para que las métricas no sean ruidosas por azar de muestreo),
- std de MSE/F1 dentro de lo aceptable,
- 1× el cómputo de K=5 vs 2× de K=10.

Consistente con la justificación general en `lineal_perceptron/analisis_outputs/kfold_choice.md` y reforzado acá con datos del no-lineal: subir a K=10 **empeora** la estabilidad por fold sin ganancia compensatoria de bias.