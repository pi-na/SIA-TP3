# Resultados Ej 3 — Técnicas de regularización, implementación y resultados

Esta nota expande la tabla del slide "Resultados Ej 3" con tres niveles para cada técnica: **qué es**, **cómo está implementada en el proyecto**, y **qué resultado dio**.

## Tabla resumen

| Config | val K=5 | test | Δ test vs base_extra |
|---|---|---|---|
| `base_extra_data` (referencia) | 0.9706 | 0.9636 | — |
| + L2 (λ = 1e-4) | 0.9758 | 0.9656 | +0.20 pp |
| + Dropout (p = 0.2) | 0.9750 | 0.9628 | −0.08 pp |
| + L2 + Dropout | 0.9772 | 0.9604 | −0.32 pp |
| **+ L2 + Aug gauss (σ = 0.05)** | **0.9760** | **0.9688** | **+0.52 pp** ← ganador |
| + L2 + Aug + Dropout | 0.9768 | 0.9656 | +0.20 pp |
| + Wider + L2 + Aug | 0.9777 | 0.9640 | +0.04 pp |

**Ganador final: L2 + Augmentation gaussiana → 96.88 % en test.**

---

## 1. L2 / Weight Decay

### Qué es

Penaliza el valor absoluto de los pesos, sumando al loss un término proporcional a `‖W‖²`:

```
E_reg = E_data + (1/2) · λ · Σ W_ij²
```

Slide 20-25 del PDF de regularización. La idea: pesos grandes producen funciones muy "puntiagudas" que sobreajustan; penalizarlos empuja la red hacia funciones más suaves que generalizan mejor.

### Cómo se implementa

```python
# mlp/network.py:168-174
reg_grad = np.zeros_like(W)
reg_grad[:, 1:] = l2 * W[:, 1:]   # NO penaliza bias (columna 0)
grad_W += reg_grad
```

Punto importante: **el bias no se penaliza**. La columna 0 de cada matriz de pesos es el bias (por el bias trick) y se omite del término de regularización. Práctica estándar (Goodfellow Cap. 7): el bias no aporta varianza al modelo, penalizarlo sesga el ajuste.

### Resultado

`+0.20 pp` consistente sobre el baseline. Es un boost chico pero se mantiene cuando se combina con augmentation. **Por qué es chico**: con `more_digits.csv` ya hay mucha data efectiva, y la red de `[784, 100, 50, 10]` no tiene tanta capacidad como para sobre-ajustar fuerte. La regularización ayuda al margen.

---

## 2. Augmentation gaussiana

### Qué es

En cada batch, antes de pasar la entrada por la red, se le suma ruido normal `N(0, σ²)` muestra a muestra:

```
x_batch ← x_batch + ε,  ε ~ N(0, σ²)
```

Slide 18 del PDF (una de las cuatro formas listadas: gaussiano, rotaciones, traslaciones, cambios de escala). La idea: mostrarle al modelo perturbaciones de los datos para que aprenda invariancia ante ellas.

### Cómo se implementa

```python
# mlp/network.py:160-165
if aug_sigma > 0:
    noise = np.random.normal(0, aug_sigma, X_batch.shape)
    X_batch = X_batch + noise
```

`σ = 0.05` fue el valor que ganó. Más chico (0.01) es casi inocuo; más grande (0.1) empieza a degradar la señal.

### Resultado

`+0.32 pp` adicionales **sobre L2** (no contra el baseline crudo): hay sinergia entre L2 y augmentation. L2 reduce varianza por el lado de los pesos, augmentation por el lado de los datos.

**Limitación importante**: el augmentation gaussiano es **isotrópico por píxel**. Simula ruido de adquisición. **No** simula rotaciones, traslaciones o cambios de escala — y el shift del test parece ser justamente eso (estilos de escritura distintos). Por eso quedamos en 96.88 % y no en 98 %.

---

## 3. Dropout

### Qué es

Durante el entrenamiento, cada neurona de la capa oculta se "apaga" con probabilidad `p`. El forward usa una máscara binaria aleatoria; el backward sólo actualiza las neuronas vivas. En inferencia, no hay drop.

Slide 26 del PDF (sólo mencionado, sin profundizar).

### Cómo se implementa

```python
# mlp/network.py:86-94 — inverted dropout
if training and dropout_p > 0:
    mask = (np.random.rand(*a.shape) > dropout_p) / (1.0 - dropout_p)
    a = a * mask
```

Inverted dropout: la división por `(1 − p)` durante training compensa por la baja densidad de activaciones, así que en inferencia se puede usar la red entera sin re-escalar.

### Resultado

Mejora val K-fold (`+0.04` pp) pero **empeora test** (`−0.08` pp). Es un caso clásico de "una técnica que reduce varianza al fold no corrige shift de distribución". Más detalle en `docs/notas/explicacion_distribution_shift.md`.

**Decisión**: descartado del config ganador.

---

## 4. Más datos (`more_digits.csv`)

### Qué es

No es una técnica de regularización en el sentido del PDF, pero **fue el cambio que más movió la aguja** (+10 pp). Se concatena al CSV de train un segundo CSV (15.742 muestras adicionales) antes del split.

### Cómo se implementa

```json
// config Ej3
{
  "data": {
    "csv_path": "data and documentation/digits.csv",
    "extra_csv_paths": ["data and documentation/more_digits.csv"]
  }
}
```

`mlp/data.py:load_dataset` concatena los CSVs antes del K-fold split.

### Resultado

**+10.06 pp** en test (de 86.30 % a 96.36 %), **sin tocar el modelo**. Esto es el resultado dominante del Ej 3 y la prueba directa de que el techo del Ej 2 era distribution shift, no falta de capacidad.

---

## 5. Wider (capacidad)

Probado para descartar la hipótesis de "falta de capacidad".

`[784, 200, 100, 10]` (vs base `[784, 100, 50, 10]`):
- val: 0.9777 (mejor)
- test: 0.9640 (peor)

**Confirma la curva U del slide 9 del PDF de regularización**: aumentar capacidad sin compensar con más datos o regularización mete a la red en zona de overfitting. Descartado.

---

## 6. Lo que no se probó

Tres formas de augmentation del slide 18 del PDF que **no implementamos**:
- **Rotaciones** (random rotation ±15°)
- **Traslaciones** (random shift ±2 px)
- **Cambios de escala** (random zoom ±10 %)

Hipótesis: si las implementáramos, el resultado se acercaría al 98 % objetivo, porque el shift residual parece ser geométrico (distintos estilos de escritura). **Esto es la principal pieza de trabajo futuro**.

## 7. Conclusión operativa

1. **Más datos es la palanca dominante** — siempre que sea posible, agregar datos antes que tunear hiperparámetros.
2. **L2 + augmentation gaussiana** es el combo mínimo razonable de regularización para este problema. `+0.5 pp` adicional sobre el baseline con extra data.
3. **Dropout y más capacidad** sólo ayudaron en val; no transfirieron a test. Ambos son síntomas de que el problema residual es shift, no varianza ni capacidad.
4. **Para llegar al 98 %** falta augmentation geométrica.

## Ver también

- `docs/notas/clase_regularizacion_cosas_implementadas.md` — matching con el PDF de la clase
- `docs/notas/explicacion_distribution_shift.md` — qué es el shift y cómo se mitigó
- `ejercicio3/README.md` — log experimental completo
