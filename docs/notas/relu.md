# ReLU (Rectified Linear Unit)

## Por qué ReLU: converge en menos épocas

La razón principal por la que ReLU se volvió el estándar en capas ocultas es que **hace converger la red en muchas menos épocas** que sigmoide o tanh.

### El mecanismo en una línea

Como la derivada de ReLU es **1** en la zona activa, el gradiente retropropaga **sin atenuarse** capa por capa → las capas tempranas reciben señal útil desde la primera época → todos los pesos se actualizan a buen ritmo → menos épocas para converger.

### Comparado con sigmoide

Con sigmoide la derivada máxima es 0.25. En una red de N capas, el gradiente que llega a la primera capa se multiplica por algo del orden de `0.25^N`:

- 3 capas: `≈ 0.016`
- 5 capas: `≈ 0.001`
- 10 capas: `≈ 10⁻⁶`

Las primeras capas casi no aprenden → la red tarda **mucho** más, o directamente no converge en redes profundas. Este es el problema histórico que frenó al deep learning hasta ~2012.

### Evidencia en el TP

El sweep del Ej 2 con ReLU + Adam convergió en `best_epoch = 7` sobre una red `[784, 100, 50, 10]`. Con sigmoide en las capas ocultas, en esa misma arquitectura, hubieran sido necesarias del orden de cientos de épocas (o nunca llegar a 96.7%).

### Matiz importante

ReLU acelera la convergencia **medida en épocas**, no porque cada época sea más rápida. Cada época con ReLU dura prácticamente lo mismo que con sigmoide. Lo que cambia es **cuántas épocas se necesitan** para llegar al mismo loss.

### ¿Y el costo por iteración?

Calcular la derivada de ReLU es trivial: un `if z > 0`. Comparado con:

- **Sigmoide:** `σ'(z) = σ(z)·(1−σ(z))` → requiere haber calculado la sigmoide (con un `exp`).
- **Tanh:** `tanh'(z) = 1 − tanh²(z)` → ídem, requiere `exp`.

Es más barato computacionalmente, pero en una GPU moderna esto es ruido. **Este efecto es marginal** comparado con la reducción en cantidad de épocas.

## Definición

```
ReLU(z) = max(0, z)
```

- Si z > 0 → devuelve z (pasa igual)
- Si z ≤ 0 → devuelve 0 (anula)

## ReLU como Perceptrón con Criterio de Activación

Una neurona con ReLU es esencialmente un **perceptrón simple con un criterio de activación particular**:

```
Input x → z = w·x + b → ReLU(z) = max(0, z) → output
```

El "criterio" es: **pasa el valor si es positivo, anula si es negativo**.

Comparación con otros criterios:

| Activación   | Criterio                  | Rango Output     |
|-------------|--------------------------|-----------------|
| Step        | `1 si z>0 else -1`       | {-1, 1} binario |
| Sigmoid     | `1/(1+e^(-z))`           | (0, 1) continuo |
| Identity    | `z`                      | (-∞, +∞) lineal |
| **ReLU**    | **`max(0, z)`**          | **[0, +∞)**     |

**ReLU es semi-lineal**: lineal cuando z > 0, nulo cuando z ≤ 0.

## Implementación en el Proyecto

```python
# mlp/activations.py

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_grad(z: np.ndarray, a: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)  # Gradiente: 1 si z>0, else 0
```

### En el forward pass (`mlp/network.py`, línea 82-95):

```python
for i, W in enumerate(self.weights):
    a_with_bias = self._add_bias_column(a)  # Input con bias
    z = a_with_bias @ W.T                   # z = w·x + b
    act_fn, _ = ACTIVATIONS[self.activations[i]]
    a = act_fn(z)                           # Aplicar criterio de activación
```

## Por Qué "Descarta" ~50% del Input

Las pre-activaciones `z` de capas ocultas tienen distribución aproximadamente normal con media ≈ 0 (con buena inicialización):

```
Distribución de z:
        │
        │    ╱╲
        │   ╱  ╲
────────┼──╱────╲────── z
             0
    50% negativos | 50% positivos
    → ReLU = 0    | → ReLU = z
```

Esto es aproximado y asume buena inicialización (ver He Initialization abajo).

**Efecto en el mini-batch:**

```python
# Ejemplo de un batch:
z = [-0.5, 1.2, -0.1, 2.0, -0.3, ...]
output_relu = [0, 1.2, 0, 2.0, 0, ...]  # ~50% en 0

# Gradientes para backprop:
gradients = [0, 1, 0, 1, 0, ...]  # 0 donde output era 0

# Actualización de pesos:
# w = w - lr * gradient_promedio
# Solo las muestras activas contribuyen al aprendizaje
```

## Neuronas Muertas (Dead Neurons)

Una neurona ReLU está **viva** si devuelve 0 para algunos datos pero no todos. Está **muerta** si devuelve 0 para prácticamente todos los datos.

### Neurona viva (normal):

```
Dato 1: z = -0.5 → ReLU = 0    (apagada)
Dato 2: z =  1.2 → ReLU = 1.2  (activa)
Dato 3: z = -0.1 → ReLU = 0    (apagada)
Dato 4: z =  2.0 → ReLU = 2.0  (activa)

→ Gradiente promedio ≠ 0 → la neurona aprende
```

### Neurona muerta:

```
Dato 1: z = -5.0 → ReLU = 0
Dato 2: z = -3.2 → ReLU = 0
Dato 3: z = -0.5 → ReLU = 0
...
Dato 10000: z = -1.1 → ReLU = 0

→ Gradiente promedio ≈ 0 → ¡la neurona NUNCA APRENDE!
```

### Por qué ocurre:

1. **Mala inicialización** de pesos (valores muy grandes/negativos)
2. **Learning rate muy alto** → los pesos se vuelven muy negativos
3. **Bias muy negativo**

Una vez que una neurona muere durante el entrenamiento, no tiene forma de "revivir" con ReLU porque `relu_grad(z) = 0` cuando `z ≤ 0`.

## He Initialization (Solución para ReLU)

Para evitar neuronas muertas desde el inicio, se usa **He Initialization**, que escala los pesos según el tamaño de la capa anterior:

```python
w = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
```

**¿Por qué √(2/n_in)?**
- Con ReLU, ~50% de las activaciones son 0
- La varianza de la salida de una capa se reduce a la mitad respecto a una activación lineal
- Multiplicar por √2 compensa esa pérdida de varianza
- Objetivo: mantener la varianza de las activaciones estable a lo largo de las capas

Comparación de inicializaciones:

| Inicialización | Fórmula          | Para usar con   |
|---------------|-----------------|----------------|
| He            | `√(2/n_in)`     | ReLU, Leaky ReLU |
| Xavier/Glorot | `√(1/n_in)`     | Sigmoid, Tanh   |
| Uniform       | `U(-0.1, 0.1)`  | Simple, experimental |

## Gradiente de ReLU y Backpropagation

```python
def relu_grad(z, a):
    return (z > 0).astype(float)  # 1 si activa, 0 si apagada
```

El gradiente es:
- `1` donde `z > 0` → el gradiente fluye sin cambios
- `0` donde `z ≤ 0` → el gradiente se bloquea

**Ventaja sobre sigmoid/tanh:** No hay saturación en la región positiva. Con sigmoid, valores extremos producen gradientes ≈ 0 (vanishing gradient). ReLU en su región activa siempre tiene gradiente = 1.

```
Gradiente de sigmoid en valores grandes:
  σ'(z) = σ(z)(1 - σ(z)) ≈ 0 cuando |z| >> 0  ← PROBLEMA

Gradiente de ReLU en valores positivos:
  relu'(z) = 1 siempre que z > 0               ← SIN PROBLEMA
```

## Variantes de ReLU

### Leaky ReLU

Evita neuronas muertas permitiendo un pequeño gradiente cuando z ≤ 0:

```python
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

# Gradiente:
def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)
```

### ELU (Exponential Linear Unit)

Transición suave en lugar de kink en z=0:

```python
def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))
```

### Comparación visual

```
output
  │           Identity (lineal)
  │          ╱
  │         ╱
  │        ╱
  │   ReLU ╱
──┼───────╱────── z
  │      0
  │
  │ Leaky ReLU:
──┼──╱─────────── z  (pendiente pequeña para z<0)
 ╱│ 0
╱ │
```

## Cuándo Usar ReLU

| Situación | Recomendación |
|-----------|--------------|
| Capas ocultas en MLP | ✓ ReLU (default moderno) |
| Red muy profunda | ✓ ReLU o Leaky ReLU |
| Capa de salida binaria | ✗ Usar Sigmoid |
| Capa de salida multiclase | ✗ Usar Softmax |
| Datos con valores negativos importantes | ✗ Leaky ReLU o ELU |
| Problema de neuronas muertas | → Cambiar a Leaky ReLU |

## En el Contexto del TP3

En los experimentos de dígitos (Ej2 y Ej3), ReLU se usa en capas ocultas con He initialization:

```json
{
  "architecture": {
    "layer_sizes": [784, 256, 128, 10],
    "activations": ["relu", "relu", "softmax"],
    "initializer": "auto"
  }
}
```

`initializer: "auto"` elige He para capas con ReLU y Xavier para sigmoid/tanh (ver `mlp/initializers.py`).

## Resumen

- **ReLU = perceptrón con criterio `max(0, z)`** — lineal si positivo, nulo si negativo
- **~50% de neuronas apagadas por batch** — feature, no bug (sparsity)
- **Neuronas muertas** — cuando z < 0 para casi todos los datos, los pesos nunca se actualizan
- **He initialization** — escala pesos con √(2/n_in) para compensar el 50% apagado
- **Sin vanishing gradient** — en su región activa el gradiente es siempre 1
- **Preferida en capas ocultas** sobre sigmoid/tanh por simplicidad y eficacia en redes profundas

## Ver También

- Implementación: `mlp/activations.py`
- Inicialización: `mlp/initializers.py`
- Regularización: `docs/notas/early_stopping.md`
