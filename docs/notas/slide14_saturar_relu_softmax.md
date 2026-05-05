# Slide 14 — Notas sobre función de activación

Comentarios originalmente incluidos en la slide 14
("Función de activación — cuál y por qué") y movidos acá para no
recargar la slide.

---

## 1. ¿Qué es "saturar"?

Una activación **satura** cuando su pendiente $f'(z)$ se aproxima a $0$
para entradas grandes en magnitud.

Sigmoide y tanh saturan en sus extremos: el gradiente que llega a las
capas previas se desvanece (*vanishing gradient*) y la red deja de
aprender.

### ¿Por qué importa para la elección de activación?

- **Sigmoide / tanh** en capas ocultas profundas → riesgo de vanishing
  gradient. Por eso quedan reservadas para casos puntuales (por ejemplo
  Ej 1, donde el target es probabilidad y la red es de un solo nodo).
- **ReLU** $= \max(0, z)$: derivada $1$ en zona positiva, $0$ en
  negativa. Sin saturación en su lado activo, el gradiente fluye sin
  atenuarse a través de muchas capas. Es la activación estándar moderna
  para capas ocultas (Ej 2 y Ej 3).
- **Softmax** se usa solo en la capa de salida multiclase: convierte
  logits en una distribución de probabilidad sobre las 10 clases.

### Resumen oral (un párrafo)

> Saturación significa que la derivada de la activación se va a cero;
> cuando eso pasa en muchas capas consecutivas el gradiente se desvanece
> y la red deja de aprender. Por eso usamos ReLU en las capas ocultas:
> en su zona activa la derivada vale 1 y el gradiente viaja sin
> atenuarse. Sigmoide y softmax quedan reservadas para las salidas, que
> es donde nos interesa interpretar el resultado como probabilidad.

---

## 2. Softmax — qué es, por qué y cómo lo usamos

### Qué es

Softmax convierte un vector de scores reales $z \in \mathbb{R}^K$ en
una **distribución de probabilidad sobre $K$ clases**:

$$
\mathrm{softmax}(z)_i \;=\; \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
$$

Cada componente queda en $[0,1]$ y todas suman 1. En `mlp/activations.py`
se implementa restando el máximo por fila antes de exponenciar (truco
estándar de estabilidad numérica: evita que `exp(z_i)` desborde sin
cambiar el resultado, porque la constante se cancela arriba y abajo).

### Por qué la elegimos para Ej 2 / Ej 3

Las 10 clases de dígitos son **mutuamente excluyentes**: una imagen es
exactamente un dígito. Eso descarta usar 10 sigmoides independientes,
que tratarían las clases como un problema multilabel y no garantizarían
que las probabilidades sumen 1.

Las propiedades clave que necesitábamos:

- **Salidas en $[0,1]$** y **suman 1** ⇒ probabilidades válidas sobre
  las 10 clases.
- **Diferenciable** en todo $\mathbb{R}^K$ ⇒ se puede entrenar por
  backprop.
- **Preserva el orden**: la clase con el $z$ más grande también tiene la
  probabilidad más grande, así que la predicción es simplemente
  `argmax` sobre los 10 outputs.

Backing en la cátedra: softmax **no aparece explícitamente en los PDFs**
de la materia (ver `docs/notas/decisiones_y_backing_teorico.md` §1).
Es la elección estándar de la literatura para clasificación multiclase
y se justifica por el atajo de gradiente con cross-entropy (siguiente
punto).

### Cómo lo usamos en el código

1. **Arquitectura final del Ej 2** (heredada al Ej 3):

   ```
   Input 784  →  Hidden 100 (ReLU)  →  Hidden 50 (ReLU)  →  Output 10 (Softmax)
   ```

2. **Pareja softmax + cross-entropy** (binding mutuo en
   `mlp/network.py`):
   - `loss="cross_entropy"` requiere `activations[-1]="softmax"`.
   - `activations[-1]="softmax"` requiere `loss="cross_entropy"`.
   - softmax solo puede aparecer como activación **final**, nunca en
     capas ocultas.

   El motivo es algebraico: con esa combinación el gradiente se
   simplifica a

   $$
   \frac{\partial L}{\partial z} \;=\; \mathrm{softmax}(z) \,-\, y_{\text{onehot}}
   $$

   Sin el atajo habría que componer la jacobiana $K \times K$ de
   softmax con el gradiente del log; al usarlas juntas, esos términos se
   cancelan y queda una resta. Más limpio numéricamente y más rápido.

3. **Inicialización**: la última capa usa **Xavier**, no He. He está
   calibrada para ReLU (compensa que ReLU descarta la mitad del input);
   softmax tiene comportamiento simétrico y se beneficia de la varianza
   de Xavier. El selector `initializer="auto"` en `mlp/initializers.py`
   mapea `"softmax" → "xavier"` automáticamente.

### Resumen oral (un párrafo)

> Para clasificar 10 dígitos necesitamos una distribución de
> probabilidad sobre las 10 clases, no 10 probabilidades independientes.
> Softmax hace exactamente eso: exponencia los scores y los normaliza
> para que sumen 1. Lo usamos en la capa de salida del MLP del Ej 2 y
> Ej 3, emparejado con cross-entropy: esa combinación tiene un
> gradiente cerrado simple, $\mathrm{softmax}(z) - y$, que es lo que
> implementa el módulo. La predicción final es el `argmax` sobre las
> 10 salidas.
