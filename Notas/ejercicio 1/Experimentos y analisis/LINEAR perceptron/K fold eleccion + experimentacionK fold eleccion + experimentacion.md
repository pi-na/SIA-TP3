# K-fold sweep — perceptrón lineal

## Configuración del experimento

- LR fijo: `1e-4` (ganador del sweep de LR)
- Seed: `42`
- Épocas: `500` (suficiente para plateau, ver sweep LR)
- Threshold: `0.69` (thr* del lineal — max F1 promedio en el sweep multi-seed de LR)
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
| 2 | 0.02686 ± 0.00127 | 0.8820 ± 0.0036 | 0.9177 ± 0.0322 | 0.8504 ± 0.0209 | 0.9736 ± 0.0016 |
| 3 | 0.02657 ± 0.00102 | 0.8825 ± 0.0089 | 0.9125 ± 0.0223 | 0.8550 ± 0.0177 | 0.9736 ± 0.0021 |
| 5 | 0.02658 ± 0.00076 | 0.8821 ± 0.0213 | 0.9086 ± 0.0283 | 0.8585 ± 0.0368 | 0.9735 ± 0.0047 |
| 10 | 0.02665 ± 0.00206 | 0.8809 ± 0.0250 | 0.9062 ± 0.0265 | 0.8584 ± 0.0420 | 0.9732 ± 0.0054 |

## Std entre folds por K (métrica de estabilidad del estimador)

| K | std MSE test | std F1 |
|---|---|---|
| 2 | 0.00127 | 0.0036 |
| 3 | 0.00102 | 0.0089 |
| 5 | 0.00076 | 0.0213 |
| 10 | 0.00206 | 0.0250 |

![K-fold sweep](../../../imagenes/kfold_sweep.png)

## Conclusión

### 1. La media de las métricas es estable en K

Las medias entre K=2, K=3, K=5 y K=10 caen dentro de rangos muy chicos:

| Métrica | rango entre K | comentario |
|---|---|---|
| MSE test | `0.02657` → `0.02686` (rango 1.1%) | constante a los fines prácticos |
| F1       | `0.8809` → `0.8825` (rango 0.2%) | constante |
| Precision | `0.9062` → `0.9177` (rango 1.3%) | constante |
| Recall    | `0.8504` → `0.8585` (rango 1.0%) | constante |
| Accuracy  | `0.9732` → `0.9736` (rango 0.04%) | constante |

Esto dice algo importante: **el K no cambia lo que el modelo aprendió**. El piso de capacidad del Adaline es ~`MSE = 0.0266` y eso no se mueve con cuánto le rotemos la partición. K-fold no es un truco para mejorar performance, es una herramienta para **estimar mejor** la performance verdadera del modelo.

### 2. La std entre folds sí depende de K — y K=5 minimiza la std de MSE

Std del MSE entre folds:

| K | std MSE test | Lectura |
|---|---|---|
| 2  | 0.00127 | Pocas particiones (sólo 2) → cada estimación es ruidosa por la propia varianza muestral del fold |
| 3  | 0.00102 | Mejora respecto a K=2 |
| 5  | **0.00076** | **Mínima del barrido** |
| 10 | 0.00206 | Sube de nuevo: folds chicos (750 muestras, ~87 positivos por fold) → varianza muestral por fold domina |

**K=5 da el estimador más estable del MSE de generalización** sobre este dataset y modelo, en términos de std entre folds.

La std de F1 sigue otra dinámica (K=2 da la menor `0.0036` y crece con K), porque F1 es sensible a la cantidad de positivos por fold: K=2 tiene ~434 positivos por fold (muchos) y K=10 sólo ~87 (pocos → varianza alta de Precision/Recall por azar de muestreo). Pero ese "ganador" de K=2 viene con bias del estimador: entrenar con 50% del dataset no representa al modelo final que entrena con 100%.

### 3. K=5 es la elección operativa correcta

Aplicando el criterio prescriptivo (*"si la std de F1 entre K=5 y K=10 difiere menos de 0.005, K=5 es suficiente"*): `|0.0250 − 0.0213| = 0.0037 < 0.005` → **se satisface**. Subir a K=10 **no compra** estabilidad relevante en F1 y a la vez **pierde** estabilidad en MSE (`std MSE_K=10 = 0.00206`, casi 3× peor que K=5).

Trade-off final: K=5 entrega
- **menor std de MSE** del barrido,
- std de F1 dentro del ruido de K=10,
- 80% de los datos para entrenar (modelo cerca del régimen full-data → bias chico del estimador),
- 1500 muestras por fold de validación con ~174 positivos (suficientes para que las métricas de cada fold no sean ruidosas por sí solas),
- la mitad del cómputo que K=10.

**K=5 es el ganador defendible** — y la conclusión es consistente con la justificación general de K=5 en `lineal_perceptron/analisis_outputs/kfold_choice.md`.