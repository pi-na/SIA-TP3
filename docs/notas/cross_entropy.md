# Cross-Entropy Loss

## Contexto

En el TP3 cambiamos la función de costo según el problema:

| Ejercicio | Salida | Loss |
|---|---|---|
| Ej 0 (XOR) | continua en {-1,+1} | MSE |
| Ej 1 (distillation) | continua en [0,1] | MSE |
| **Ej 2 / Ej 3** | **distribución multiclase (10 dígitos)** | **Cross-Entropy + Softmax** |

Esta nota explica por qué cross-entropy es la elección estándar para clasificación multiclase y cómo se combina con softmax.

---

## 1. Definición

Para una muestra con etiqueta one-hot `y ∈ {0,1}^K` (K clases) y predicción `O ∈ [0,1]^K` (probabilidades, suma 1):

```
H(y, O) = - Σ_k  y_k · log(O_k)
```

Como `y` es one-hot, sólo sobrevive el término de la clase correcta `c`:

```
H(y, O) = - log(O_c)
```

**Lectura intuitiva**: el costo es chico si la red asigna alta probabilidad a la clase correcta (`O_c → 1 ⇒ -log(1) = 0`) y crece sin límite si asigna baja probabilidad (`O_c → 0 ⇒ -log(0) = +∞`).

Para un batch de N muestras tomamos el promedio:

```
E = - (1/N) Σ_i  log(O_{i, c_i})
```

## 2. Por qué se usa con softmax

La capa de salida en Ej 2/3 es **softmax**:

```
O_k = exp(z_k) / Σ_j exp(z_j)
```

Convierte logits en probabilidades. La combinación softmax + cross-entropy tiene un gradiente especialmente limpio:

```
∂E / ∂z_k  =  O_k - y_k
```

Es decir: el gradiente con respecto a la pre-activación de la última capa es **(predicción − target)**, sin pasar por derivadas de la sigmoide ni del logaritmo. Esto:

1. **Evita problemas numéricos**: no hay que dividir por `O_k`, que podría ser cercano a 0.
2. **Es barato de calcular**: una sola resta por clase.
3. **Tiene escala unitaria**: el gradiente no se satura aunque la red esté muy mal calibrada.

En `mlp/network.py` hay un guard: si la última activación es `softmax`, el loss debe ser `cross_entropy`. La combinación está implementada como un único bloque que computa directamente `O - y` para la backward, sin recorrer las dos derivadas por separado.

## 3. Comparación con MSE

Si usáramos MSE con softmax (`E = ‖O − y‖²`), la derivada arrastra el Jacobiano de softmax (matriz K×K) y los términos `O_k(1 − O_k)` saturan: cuando la red está muy confiada en la respuesta equivocada, el gradiente se hace pequeño y el aprendizaje se frena. Cross-entropy no sufre eso — penaliza fuerte la confianza en la clase incorrecta.

| Loss | Saturación cuando la red está mal | Gradiente |
|---|---|---|
| MSE + softmax | Sí (gradiente ~0 en errores grandes) | Complicado, depende del Jacobiano |
| **Cross-entropy + softmax** | **No** | **`O − y`, limpio** |

## 4. Cross-entropy binaria (BCE)

Cuando `K = 2` y la salida es una sola sigmoide en `[0,1]`:

```
BCE = -[ y · log(O) + (1 − y) · log(1 − O) ]
```

Implementada en `mlp/losses.py` pero **no usada en el TP** (Ej 1 usa MSE porque entrena con probabilidades continuas, no etiquetas binarias).

## 5. Resumen

- **Cross-entropy** mide la divergencia entre la distribución predicha y la real.
- En clasificación multiclase es la elección estándar: penaliza fuerte la confianza incorrecta.
- Combinada con **softmax**, da un gradiente analítico limpio: `∂E/∂z = O − y`.
- En `mlp/`: implementada como bloque single-pass softmax+CE para estabilidad numérica.
