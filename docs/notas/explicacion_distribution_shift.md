# Distribution Shift — Qué es y cómo apareció en el TP3

## Resumen rápido

**Distribution shift** = la distribución de los datos de entrenamiento es distinta de la distribución de los datos de evaluación. El modelo aprende patrones del train que no generalizan al test, no por overfitting clásico (memorizar train), sino porque **train y test son poblaciones distintas**.

En el TP3 lo vimos brutalmente al final del Ej 2:

```
val K-fold=5 (sobre digits.csv):  96.22 %
test (digits_test.csv):           86.30 %     ← caída de 10 pp
```

10 puntos de drop entre validación y test no son explicables por variabilidad estadística: hay un **shift** entre `digits.csv` y `digits_test.csv`.

---

## 1. Tipos de distribution shift

| Tipo | Definición técnica | Ejemplo en imágenes |
|---|---|---|
| **Covariate shift** | `P_train(x) ≠ P_test(x)`, pero `P(y|x)` es la misma | Train con dígitos centrados, test con dígitos rotados |
| **Label shift** | `P_train(y) ≠ P_test(y)`, pero `P(x|y)` es la misma | Train balanceado, test con clases sub-representadas |
| **Concept drift** | `P(y|x)` cambia | Cambia el significado de las clases (no aplica acá) |

En el Ej 2 vimos principalmente **covariate shift**: las features (los píxeles) son distintas en distribución entre train y test, aunque la tarea (clasificar el dígito) es la misma.

## 2. Síntomas en el TP3

### Síntoma 1 — Drop val→test asimétrico

```
val_loss y train_loss bajan juntos en train.    ← sin overfitting clásico
val K-fold=5: 96.22 %
test:         86.30 %                            ← drop fuera de varianza esperada
```

Si fuera overfitting puro, lo veríamos como `train_loss ≪ val_loss` durante el entrenamiento. No lo vimos: train y val convergieron parejos. El gap aparece **sólo** contra el test.

### Síntoma 2 — Clase 8 colapsada

La matriz de confusión del modelo base (validación) ya mostraba que la **clase 8** se predecía con precision = recall = 0. La cobertura del 8 en `digits.csv` era insuficiente, y el shift hacia `digits_test.csv` agravó el problema.

### Síntoma 3 — Errores distribuidos, no concentrados

En `digits_test.csv`, los errores no se acumularon en una sola clase: se repartieron entre varias. Cuando los errores se concentran, suele ser falta de capacidad. Cuando se reparten, suele ser shift.

## 3. Por qué pasó

`digits.csv` y `digits_test.csv` son cortes distintos del mismo problema (dígitos manuscritos 28×28), pero los autores incluyeron en el test variaciones (estilo de trazo, grosor, rotación, traslación) sub-representadas en train. La **regla de oro** del TP fue tratar `digits_test.csv` como producción: nunca se usa para validar hiperparámetros. Eso es lo correcto, pero significa que sólo en la evaluación final descubrimos el shift.

## 4. Cómo se mitigó (Ej 3)

Tres líneas de ataque, ordenadas por impacto:

| Acción | Mecanismo | Δ test |
|---|---|---|
| **`more_digits.csv`** (15.742 muestras extra) | Aumenta cobertura de la población — reduce `‖P_train − P_test‖` directamente | **+10.06 pp** ← dominante |
| **L2** (λ = 1e-4) | Reduce varianza del modelo — ayuda a generalizar bajo shift suave | +0.20 pp |
| **Augmentation gaussiana** (σ = 0.05) | Simula variabilidad → robustez ante perturbaciones isotrópicas | +0.32 pp adicional |

**El `+10` pp del baseline con extra data confirma la hipótesis de shift**: nada cambió en el modelo, sólo cubrimos más distribución, y casi todo el gap desaparece.

## 5. Por qué dropout no transfirió

Dropout en val: `+0.04` pp. En test: `−0.08` pp. Es un caso de manual: dropout reduce la varianza al fold del cross-validation (mejora val), pero **no corrige shift de distribución**. Si train y test son poblaciones distintas, el dropout no pone ningún ejemplo del test en la mesa.

Lección: una técnica que mejora val K-fold no necesariamente mejora test cuando hay shift. Siempre hay que validar contra el conjunto que más se parece a producción.

## 6. Por qué `wider` empeoró

Aumentar capacidad sin más datos = más overfitting. En presencia de shift, la red "memoriza" mejor el train (que está cerca de val por construcción del fold) pero diverge del test. Es exactamente la curva U del slide 9 del PDF de regularización: capacidad alta + datos escasos → mal generalización.

## 7. Por qué no llegamos al 98% en Ej 3

El augmentation **gaussiano** que implementamos es **isotrópico por píxel**: simula ruido de adquisición. Pero el shift que sospechamos en `digits_test.csv` es **geométrico** — rotación, traslación, escala — porque distintas personas escriben con estilos distintos.

Las tres formas de augmentation que el slide 18 del PDF de regularización menciona (rotaciones, traslaciones, cambios de escala) **no se implementaron en el TP**, y son justamente las que atacarían el shift residual. Esa es la hipótesis principal del techo en 96.88%.

## 8. Resumen para presentar

> **El drop de val 96% → test 86% en Ej 2 no es overfitting: es distribution shift. El test set tiene variaciones (estilos de escritura) sub-representadas en train. Lo confirmamos mostrando que con sólo agregar más datos (`more_digits.csv`, sin tocar el modelo) recuperamos +10 pp en test. El gap residual hasta 98% viene de augmentation geométrica que no implementamos.**

## Ver también

- `docs/notas/decisiones_y_backing_teorico.md` — sección 9 ("Pack C")
- `ejercicio3/README.md` — tabla de resultados completa
- Slide 9 del PDF de regularización (curva U capacidad/error)
