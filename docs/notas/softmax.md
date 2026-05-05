# Softmax — qué es y dónde aparece en la presentación

## Qué es

Softmax es la activación que convierte un vector de scores reales `z ∈ R^K` en
una **distribución de probabilidad sobre K clases** (cada componente entre 0 y
1, y todas suman 1).

Definición:

```
softmax(z)_i = exp(z_i) / Σ_j exp(z_j)
```

Implementación numéricamente estable usada en `mlp/activations.py:48`:

```python
def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)  # evita overflow en exp
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)
```

Restar el máximo por fila antes de exponenciar no cambia el resultado (la
constante se cancela en numerador y denominador) pero evita que `exp(z_i)`
desborde cuando los scores son grandes.

## Por qué es la activación correcta para clasificación multiclase

Para K-way classification queremos una distribución de probabilidad sobre las
K clases, no K probabilidades independientes. Las propiedades clave de
softmax son:

- **Salidas en [0, 1]**: cada `softmax(z)_i` es no-negativa y ≤ 1.
- **Suman 1**: `Σ_i softmax(z)_i = 1` por construcción.
- **Diferenciable**: gradiente bien definido en todo R^K.
- **Preserva orden**: la clase con el `z` más grande también tiene la
  probabilidad más grande (es invariante a traslaciones).

Sigmoides independientes en cada output también dan valores en [0, 1], pero
**no garantizan que sumen 1** y tratan las clases como independientes — lo
que para clasificación multiclase es incorrecto: los dígitos 0-9 son
mutuamente excluyentes, una imagen es exactamente uno.

## Softmax + cross-entropy: el "atajo" del gradiente

Cuando se usa softmax como activación de salida y cross-entropy como pérdida,
el gradiente conjunto se simplifica enormemente. Mostrado en
`mlp/losses.py:35-39`:

```python
def cross_entropy_grad_with_softmax(y_true_onehot, y_pred_softmax):
    """∂CE/∂z = softmax(z) - y_true.
    Se cancelan softmax_grad y log_grad."""
    return (y_pred_softmax - y_true_onehot) / y_true_onehot.shape[0]
```

Sin el atajo, calcular `∂CE/∂z` requeriría componer la jacobiana de softmax
(matriz K×K) con el gradiente de log. Como esos dos términos se cancelan
algebraicamente, el módulo evita el costo y la inestabilidad numérica
calculando directamente `softmax(z) - y_true`.

Es por esto que `mlp/network.py:39-44` valida el binding mutuo:

- `loss="cross_entropy"` requiere `activations[-1]="softmax"`.
- `activations[-1]="softmax"` requiere `loss="cross_entropy"`.

Y `mlp/network.py:46-48` exige que softmax solo aparezca como activación
**final** (no en capas ocultas).

## Dónde aparece softmax en la presentación

### 1. Slide "Ej 2 — El problema" (arquitectura final)

```
- Input: 784
- Hidden 1: 100 (ReLU)
- Hidden 2: 50 (ReLU)
- Output: 10 (Softmax)
```

La arquitectura final del Ej 2 (heredada al Ej 3) usa softmax en la capa de
salida porque tenemos 10 clases mutuamente excluyentes (dígitos 0-9). Cada
output del modelo es la probabilidad estimada de que la imagen sea ese
dígito; la predicción es `argmax` sobre los 10 outputs.

### 2. Slide "Función de activación — cuál y por qué"

> Ej 2 / Ej 3 — capa de salida: **softmax**.
> "Clasificación multiclase: devuelve distribución de probabilidad sobre las
> 10 clases."

Acá softmax se contrasta con sigmoide (Ej 1) y ReLU (capas ocultas). La
elección por capa depende del rol:

| Capa | Activación | Razón |
|---|---|---|
| Ej 1 — output (1 neurona) | sigmoide | target en [0,1] |
| Ej 2/3 — ocultas | ReLU | gradiente fuerte, no satura |
| Ej 2/3 — output (10 neuronas) | softmax | distribución sobre 10 clases |

### 3. Slide "Inicialización de pesos"

> Xavier: distribución $\mathcal{N}(0, \sqrt{1/\text{fan\_in}})$ — capas con
> **tanh / sigmoide / softmax**: mantiene varianza balanceada.

Para la última capa antes de softmax usamos inicialización **Xavier**, no
He. Razón: He está calibrada para ReLU (compensa que ReLU descarta la mitad
del input); softmax/sigmoide/tanh tienen comportamiento simétrico alrededor
del cero y se benefician de la varianza de Xavier. El selector
`initializer="auto"` en `mlp/initializers.py:31` mapea
`"softmax" → "xavier"` automáticamente.

## Por qué softmax es la elección estándar (vs alternativas)

| Alternativa | Problema |
|---|---|
| **K sigmoides independientes** | No suman 1; trata clases como no excluyentes (multilabel, no multiclass) |
| **Identity + argmax** | No diferenciable; sin probabilidades calibradas |
| **Hardmax (one-hot del max)** | No diferenciable; gradiente = 0 en todo el dominio |
| **Normalización L1 de scores positivos** | Necesita scores no-negativos; comportamiento del gradiente menos elegante |

Softmax + cross-entropy es la combinación canónica para clasificación
multiclase porque (a) produce probabilidades válidas, (b) el gradiente
combinado es trivial de computar, y (c) la pérdida es convexa en los
scores `z` para clasificación.

## Resumen ejecutivo

- **Qué hace**: convierte K scores reales en una distribución de
  probabilidad sobre K clases (suma 1).
- **Dónde se usa en este TP**: capa de salida del MLP en Ej 2 y Ej 3
  (10 clases de dígitos).
- **Con qué pérdida**: cross-entropy. La combinación tiene gradiente
  cerrado simple `softmax(z) - y_onehot`.
- **Con qué inicialización**: Xavier (no He).
- **Restricción**: solo válida como activación final; nunca en capas
  ocultas.
