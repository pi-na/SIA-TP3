# Cross-entropy en el Ej2 — explicación intuitiva

Nota explicativa sobre **cómo y por qué la cross-entropy castiga lo que castiga**. Pensada para defensa oral y para intuición rápida — referencias formales en [`metricas.pdf`](metricas.pdf) y [`Segunda tanda de experimentos/Justificacion_metricas.md`](Segunda%20tanda%20de%20experimentos/Justificacion_metricas.md).

---

## La fórmula y qué hace

En Ej2 con targets one-hot, la CE por muestra colapsa a:

$$\mathrm{CE}_i = -\log(q_{c^*_i})$$

donde $q_{c^*_i}$ es la **probabilidad que el modelo le asignó a la clase verdadera** (`c*`). El resto del vector softmax no entra en la cuenta.

Es decir, **la loss de cada muestra depende sólo de cuánta probabilidad le dio el modelo a la respuesta correcta**.

---

## ¿Qué filas suman a la loss?

Las filas que más aportan al promedio total son:

1. **Las que el modelo erró** — porque le asignó poca probabilidad a la clase verdadera.
2. **Las que acertó pero con poca confianza** — incluso si argmax fue correcto, si $q_{c^*}$ era 0.4, la CE sigue siendo significativa.

Las filas que el modelo "ya tiene bien y confiado" (probabilidad alta a la clase correcta) prácticamente **no aportan** a la loss.

### Tabla de contribuciones

| Situación | $q_{c^*}$ | $-\log(q_{c^*})$ | Contribución |
|---|---|---|---|
| Acertó MUY confiado | 0.99 | 0.010 | ≈ 0 (ignorada) |
| Acertó confiado | 0.90 | 0.105 | pequeña |
| Acertó "a medias" | 0.50 | 0.693 | moderada |
| Acertó por la mínima | 0.20 | 1.609 | grande |
| Erró pero algo le asignó | 0.10 | 2.303 | grande |
| Erró bastante | 0.01 | 4.605 | enorme |
| Erró desastrosamente | 0.001 | 6.908 | catastrófica |

**Patrón clave:** la curva $-\log$ es **convexa**. La penalización crece **muchísimo más rápido cerca de 0** que cerca de 1. Castiga exponencialmente la confianza errada.

---

## Por qué esto le sirve al optimizer

El gradiente combinado de CE + softmax respecto a los logits es:

$$\frac{\partial \mathrm{CE}}{\partial z_i} = q_i - y_i$$

Para la clase verdadera $c^*$:

$$\frac{\partial \mathrm{CE}}{\partial z_{c^*}} = q_{c^*} - 1$$

Entonces:

- Si $q_{c^*} = 0.99$ → gradiente = $-0.01$ (casi nulo — el optimizer **no actualiza** esa muestra)
- Si $q_{c^*} = 0.50$ → gradiente = $-0.5$ (empuje moderado)
- Si $q_{c^*} = 0.01$ → gradiente = $-0.99$ (empuje máximo posible)

**El modelo se concentra en aprender lo que NO sabe bien.** Las muestras donde ya está confiado y acertando contribuyen casi cero al gradiente — el optimizer no las "toca". Las que mueven los pesos son las **erradas** y las **acertadas con poca confianza** (las que están cerca del límite de decisión).

Implementación en el repo: `mlp/losses.py:35-39`

```python
def cross_entropy_grad_with_softmax(y_true_onehot, y_pred_softmax):
    return (y_pred_softmax - y_true_onehot) / N
```

Es decir, literalmente `predicción − target`. El "truco" de combinar softmax+CE es que sus derivadas se cancelan en el chain rule y queda esta forma mínima — sin overflow numérico, sin código complicado.

---

## La dinámica típica de entrenamiento

Esto se ve en las curvas de `convergence_val.png` del cross-experiment:

1. **Primeras épocas:** CE alta (~2.3, igual a $-\log(0.1)$, que es lo que da el modelo random sobre 10 clases uniformes). **Todas las muestras aportan mucho** porque el modelo arranca confundido en todas.

2. **Épocas intermedias:** CE baja rápido. El modelo aprende primero las muestras "fáciles" (donde estaba errado con baja prob asignada a la verdadera). Esas dejan de aportar y se va concentrando en las difíciles.

3. **Épocas finales:** CE baja lento. Solo quedan las muestras **ambiguas** (acc alta, pero la confianza no termina de subir del todo). En este punto:
   - `train_loss` sigue bajando lentamente — el modelo se va volviendo más confiado en train.
   - `val_loss` puede **empezar a subir** — el modelo se vuelve confiado en errores de val (overfitting de calibración).
   - `val_acc` queda estable mientras `val_loss` empeora.

Esto es exactamente la señal que early stopping detecta en `mlp/network.py:fit()`.

---

## El caso curioso para defensa oral

> **"¿Puede ser que dos modelos tengan exactamente la misma val_acc pero distinta val_loss?"**

**Sí, y pasa seguido.** Ejemplo:

- **Modelo A:** predice 60% prob a la clase correcta en cada muestra, acierta el argmax el 60% del tiempo. Acc = 0.60.
- **Modelo B:** predice 99% prob a la clase ganadora en cada muestra, acierta el argmax el 60% del tiempo. Acc = 0.60.

Misma accuracy. Pero la cross-entropy:

| Modelo | CE en muestras correctas | CE en muestras erradas | CE promedio |
|---|---|---|---|
| A | $-\log(0.60) = 0.51$ | $-\log(0.40) = 0.92$ | $\approx 0.67$ |
| B | $-\log(0.99) = 0.010$ | $-\log(0.01) = 4.6$ | $\approx 1.84$ |

**Modelo B tiene CE muchísimo más alta** — está "seguro y errado" en las muestras en las que se equivoca. La CE detecta eso; la accuracy lo ignora.

Por eso CE como **tie-breaker entre dos cells con val_acc similar** te dice algo distinto y relevante: mide **calibración**, no solo aciertos.

---

## Resumen para la defensa

- En Ej2 con one-hot targets, **CE por muestra es simplemente $-\log(q_{c^*})$**.
- La loss **se concentra en lo que el modelo no sabe bien**: errores + aciertos con poca confianza.
- **Castiga exponencialmente la confianza errada** (por la convexidad del log negativo).
- El gradiente combinado de CE+softmax es literalmente `predicción − target` — el más simple posible (ver `mlp/losses.py`).
- Dos modelos con misma val_acc pueden tener distinta val_loss → CE mide **calibración** además de aciertos. Por eso la usamos como tie-breaker en el cross-experiment.

---

## Referencias

- [`metricas.pdf`](metricas.pdf) — infografía de todas las métricas del TP.
- [`Segunda tanda de experimentos/Justificacion_metricas.md`](Segunda%20tanda%20de%20experimentos/Justificacion_metricas.md) — por qué reportamos val_acc + macro_f1, distinción CE vs P/R/F1.
- [Clase de métricas y sobreajuste](../../docs/clase_metricas_sobreajuste/metricas_sobreajuste.pdf) — fundamento teórico.
- `mlp/losses.py` — implementación de `cross_entropy` y su gradiente combinado.
