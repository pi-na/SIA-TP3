# Ejercicio 3 — Regularización: resumen de la clase y lectura de las diapos 33+

Nota de estudio: resumen de la clase de regularización de la cátedra y cómo se conecta con las diapositivas 33 en adelante de `docs/slides_presentacion/slides_presentacion.tex`.

---

## 1. Resumen de la clase de regularización

**Definición de la cátedra:** "Conjunto de técnicas diseñadas para reducir el error de test (o validación)."

### Concepto base que motiva todo

- Un modelo tiene una **capacidad** (qué tan complejas son las funciones que puede aproximar).
- Capacidad muy baja → **underfitting** (no aprende ni el train).
- Capacidad muy alta → **overfitting** (aprende ruido del train, no generaliza).
- El error de generalización (≈ test) tiene forma de U respecto a la capacidad: hay una capacidad óptima.
- Regularización = empujar al modelo hacia esa capacidad óptima sin tener que cambiar la arquitectura.

### Las técnicas que enumera la clase, en orden

1. **Early stopping.** Cortar el entrenamiento cuando el error de validación deja de bajar. Es la forma más barata de regularización: evita que el modelo siga ajustándose al ruido del train.
2. **Data augmentation.** Inyectar variabilidad sintética al input durante training. La clase lista cinco formas:
   - Ruido gaussiano (lo que usamos en Ej 3).
   - Rotaciones.
   - Traslaciones.
   - Cambios de escala.
   - Etc.
   - Advertencia explícita: *"cuidado con no modificar el dato"* — un 6 rotado 180° es un 9, no se puede rotar libre.
3. **L2 / Weight Decay.** Sumás λ · ||W||² al costo. El gradiente extra λW "tira" los pesos hacia cero en cada update. Limita la magnitud de los pesos → menos capacidad efectiva. La justificación matemática formal está en Goodfellow Cap. 7.1.1 (Taylor, Hessiano, autovalores) — la clase no la desarrolla, solo cita.
4. **Otras** (mencionadas pero no desarrolladas):
   - Dropout.
   - Modelos de ensamble.
   - Aprendizaje semi-supervisado.
   - Entrenamiento adversarial.

### Puntos importantes que repite la profesora

- Regularización mejora error de generalización, aunque a veces suba el error de entrenamiento.
- *"Se ocupa del gap que existe entre train y val/test."*
- Se aplica recién cuando el modelo ya está bajando bien el error de train pero empieza a sobreajustar.

---

## 2. Lectura de las diapos 33 en adelante

### Slide 33 — Curvas de aprendizaje del modelo ganador (Ej 3)

El modelo ganador (L2 + augmentation gaussiano) converge **más lento** que el base del Ej 2 (best_ep ~10 vs ~5). Eso es esperable: tanto L2 como el augmentation hacen el problema más difícil de minimizar (L2 penaliza pesos grandes, augmentation cambia el input cada batch). A cambio, las curvas de train y val no se separan → **sin overfitting visible**.

### Slide 34 — Qué ayudó y qué no

**Ayudó:**
- **Más datos: +10 pp.** El movimiento dominante. No es regularización en sentido estricto, pero ataca la misma raíz (cobertura).
- **L2 (λ=1e-4): +0.20 pp** consistente.
- **Aug gaussiano (σ=0.05): +0.32 pp adicional sobre L2** → hay sinergia, no son redundantes.

**No ayudó:**
- **Dropout (p=0.2):** mejora val pero empeora test. Reduce varianza dentro del K-fold pero el problema real era cobertura desigual de las clases 5 y 8, no varianza del modelo.
- **wider [784, 200, 100, 10]:** mejor en val, peor en test → más capacidad ≠ mejor generalización. Confirma que no faltaba modelo.
- **Combinar todo (L2+dropout+aug):** peor que L2+aug solo. Más regularización no es estrictamente mejor; sobre-restringe.

### Slide 35 — Por qué no llegamos al 98%

Mejor resultado: 96.88% en test. Faltan 1.12 pp.

Hipótesis:
- El **augmentation gaussiano** simula **ruido por píxel**, pero la diferencia que castiga al test parece **geométrica** (distintas personas escriben con rotación, traslación, grosor de trazo distintos).
- **La clase enumeró cuatro formas de augmentation; sólo implementamos una** (gaussiano). Faltaron rotación, traslación, escala — exactamente las que atacarían la varianza geométrica entre escritores.
- **wider empeoró** → no es problema de capacidad del modelo, es problema de cobertura del dataset.

### Slide 36 — Cierre

Pantalla de "Gracias".

---

## 3. Conexión clase ↔ implementación

| Técnica de la clase | ¿La usamos? | Dónde |
|---|---|---|
| Early stopping | Sí | Ya venía del Ej 2 (patience=10 sobre val_loss) |
| Augmentation — ruido gaussiano | Sí | Ej 3 ganador (σ=0.05) |
| Augmentation — rotación | No | Faltó |
| Augmentation — traslación | No | Faltó |
| Augmentation — escala | No | Faltó |
| L2 / weight decay | Sí | Ej 3 ganador (λ=1e-4) |
| Dropout | Probado, descartado | p=0.2 empeoraba test |
| Ensamble / semi-supervisado / adversarial | No | Fuera de alcance |

**Conclusión:** usamos 3 técnicas de las listadas (early stopping, L2, augmentation gaussiano) y probamos dropout. El gap final hasta 98% es atribuible a no haber implementado las otras formas de augmentation que la profesora mencionó (rotación, traslación, escala) — que son las que atacarían la varianza geométrica entre escritores que parece ser la fuente del error residual.
