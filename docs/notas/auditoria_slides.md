# Auditoría de `slides_presentacion.tex` (estado al 2026-05-04)

Lista accionable de gaps en la presentación, cruzando contra (a) las decisiones del Doc 1 (`decisiones_y_backing_teorico.md`) y (b) las preguntas de `ppt_notes.md`.

**Resumen ejecutivo:** la presentación cubre bien las decisiones de **activación, init, optimizador, LR y función de costo** (sección "Decisiones de diseño", slides D1-D6). El bloque Ej0/1/2/3 también está sólido. Los **gaps mayores** son tres: (1) no hay slide explícito sobre **bias** y **convergencia/epsilon**; (2) el slide D3 sobre init no menciona el mismatch de la fórmula Xavier vs. el HTML; (3) la auditoría del **Pack C vs. la clase de regularización** está implícita en los slides "lo que ayudó / no ayudó" pero no hay un slide de **mapeo 1:1 técnica-de-clase ↔ implementación**, que es justo lo que pide la última línea de `ppt_notes.md`.

---

## Tabla 1 — Cruce contra `ppt_notes.md` (lista de hiperparámetros que el TP pide tener clara)

| Hiperparámetro / decisión | Estado en slides | Gap | Acción concreta |
|---|---|---|---|
| **Learning rate** | ✓ Slide D5 (decisión) + slide 15 (sweep Fase 2 con extremos catastróficos) | Ninguno significativo. | — |
| **M cantidad de capas** | △ Implícito en slide 16 (sweep arquitectura: shallow/base/deeper/wider) | No se discute "cómo elegimos M=2 capas ocultas" como decisión separada. | Opcional: agregar una bullet en slide 16 explicitando "fijamos 2 capas ocultas porque shallow=1 subajusta y deeper=3 no aporta". |
| **Cantidad de neuronas por capa** | ✓ Slide 16 (sweep arch) + slide D1 (lista) | Ninguno. | — |
| **Cantidad de epochs** | △ Aparece en tablas (best_epoch, total_ep) pero **no hay slide** sobre la decisión "50 epochs + early stopping patience=10" | Falta justificar de dónde sale el 50, qué pasa con el patience. | **Agregar 1 bullet** al slide D5 ("Convergencia") o a un nuevo slide D7. |
| **Estrategia online / batch / mini-batch** | ✓ Slide "Cómo se entrena: mini-batch SGD" (después del slide del problema Ej2) | Ninguno significativo. | — |
| **Funciones de activación** | ✓ Slide D2 (decisión por ejercicio) | Ninguno. | — |
| **Inicialización de pesos** | ✓ Slide D3 (decisión + tabla He/Xavier/Uniform con fórmulas) | **Falta declarar la fuente** ("LinkedIn article que la cátedra mandó como referencia") y el mismatch en la fórmula de Xavier (`sqrt(1/fan_in)` vs. el `sqrt(2/(n_in+n_out))` que dice el HTML). | **Agregar nota al pie** en D3 con la fuente y el matiz de fórmula. |
| **Optimizador** | ✓ Slide D4 (decisión + sweep) + slide 17 (sweep Fase 2) | Ninguno. | — |
| **Cálculo de convergencia / ε / tamaño de ε** | ✗ **No hay slide** | El TP pide explícitamente discutir esto. Está mencionado en pasada pero no hay un slide dedicado. | **Agregar slide nuevo "D7 — Criterio de convergencia"**: epsilon en Ej1, early stopping (val_loss + patience=10) en Ej2/3, con justificación. |
| **Cómo se manejó el bias** | ✗ **No hay slide** | El TP lo pide explícitamente. | **Agregar slide nuevo "D8 — Bias"**: bias trick (columna de 1's, peso adicional), no se penaliza con L2 en Pack C. Una sola página. |

---

## Tabla 2 — Cruce contra Doc 1 (decisiones que tomamos pero no están bien defendidas en slides)

| Decisión del Doc 1 | Estado en slides | Acción |
|---|---|---|
| **Mismatch de Xavier (`sqrt(1/fan_in)` vs HTML `sqrt(2/(n_in+n_out))`)** | No mencionado | Nota al pie en slide D3, una línea. |
| **`auto` initializer mode** (relu→he, tanh/sigmoid/softmax→xavier) | Mencionado en footnote D3 ("modo `auto` que elige el inicializador según la activación") | OK. |
| **No implementamos η adaptativo del PDF, sino step decay** | No mencionado (el slide "qué seguimos / qué difiere" no lo cubre) | Agregar bullet al slide "decisiones distintas (y por qué)" mencionando que **`lr_schedule` step decay** está en la familia de η adaptativo del slide pero no es la regla específica del PDF. |
| **Por qué Adam es el ganador** (no solo el resultado, sino que Adam combina RMSProp+Momentum y el PDF lo dice "muy usado en la práctica") | △ Implícito en slide D4 (takeaway box) | Slide D4 puede agregar una línea: "El PDF de optimizadores combina RMSProp + Momentum → Adam, lo recomienda explícitamente como 'muy usado en la práctica'". |
| **Capacidad / curva U como respaldo del wider→peor en test** | △ Implícito en slide 26 ("descarta falta de capacidad") | Opcional: en slide 26, agregar una mini-cita ("consistente con la curva U del PDF de regularización slide 9: más capacidad → más overfitting si no se regulariza"). |
| **Pack C — mapeo 1:1 técnica-de-clase ↔ implementación** | ✗ **No hay slide** | Es el gap más importante. Ver Tabla 3. |

---

## Tabla 3 — El gap mayor: matching Pack C ↔ clase de regularización

`ppt_notes.md` última línea: *"HACER UN MATCHING ACA DE TEORIA Y LO HECHO"*. Esto NO está en los slides actualmente. El slide "¿Qué es Pack C?" explica la nomenclatura interna, pero no hay un slide que cruce las técnicas mencionadas en el PDF de regularización con lo que implementamos.

### Propuesta — slide nuevo después del slide "¿Qué es Pack C?"

**Título sugerido:** "Pack C ↔ Clase de Regularización (mapeo 1:1)"

**Contenido sugerido (tabla):**

| Técnica del PDF (slide) | Implementada | Resultado |
|---|---|---|
| Early stopping (slide 14) | ✓ | Activo en todos los configs |
| Augmentation — gaussiano (slide 18) | ✓ σ=0.05 | +0.32 pp (parte del ganador) |
| Augmentation — rotaciones (slide 18) | ✗ no implementada | — |
| Augmentation — traslaciones (slide 18) | ✗ no implementada | — |
| Augmentation — cambios de escala (slide 18) | ✗ no implementada | — |
| L2 / Weight Decay (slides 20-25) | ✓ λ=1e-4 | +0.20 pp |
| Dropout (slide 26 — mención) | ✓ p=0.2 | val sí, test no (-0.08 pp) |
| Modelos de ensamble | — | fuera de alcance |
| Semi-supervisado | — | fuera de alcance |
| Adversarial | — | fuera de alcance |

**Bullet de cierre:** "Las **tres formas de augmentation que NO implementamos** (rotaciones, traslaciones, escalas) son la hipótesis principal de por qué no llegamos al 98% — coherente con que el shift parece geométrico, no de ruido isotrópico."

Este slide cierra dos cosas a la vez: (a) la pregunta de `ppt_notes.md` ("hacer matching"), y (b) la hipótesis del slide actual "Por qué no llegamos al 98%" (que ya menciona augmentation geométrica).

---

## Tabla 4 — Slides que están bien y no requieren cambios

Para no perderlos de vista al editar:

- **Portada y sección "Decisiones de diseño" (D1-D6)**: estructura sólida, una decisión por slide.
- **Bloque Ej0 (AND, XOR convergencia, interpretación)**: completo.
- **Bloque Ej1 (setup, MSE comparison, threshold sweep, ROC/PR, recomendación)**: muy sólido.
- **Bloque Ej2 (problema, mini-batch, workflow, sweeps LR/arch/opt, curvas, matriz, final eval, matriz test)**: completo y bien encadenado.
- **Bloque Ej3 (plan, primer dominó, qué es Pack C, tabla resultados, comparación visual, curvas, matriz, qué movió la aguja, por qué no llegamos al 98%)**: completo.
- **Slides "Alineación con la clase — lo que seguimos / lo que difiere"**: ya existen. **Pero la sección "decisiones distintas" merece agregar el bullet del lr_schedule vs. η adaptativo** (Tabla 2).

---

## Acciones priorizadas (por impacto)

1. **[ALTO]** Slide nuevo "Pack C ↔ clase de regularización" (Tabla 3). Cierra la pregunta más explícita de `ppt_notes.md`.
2. **[ALTO]** Slide nuevo "Convergencia y bias" (combina los dos puntos faltantes de la lista de `ppt_notes.md`). Una sola página alcanza:
   - Convergencia: Ej1 = epsilon-train; Ej2/3 = early stopping val_loss patience=10.
   - Bias: bias trick (columna 1's), peso normal, no penalizado por L2.
3. **[MEDIO]** Nota al pie en slide D3 (init): fuente del HTML + mismatch de Xavier (`sqrt(1/fan_in)` vs `sqrt(2/(n_in+n_out))`).
4. **[BAJO]** Bullet en slide "decisiones distintas": `lr_schedule` step decay ≠ η adaptativo del PDF (en la misma familia, pero distinta regla).
5. **[BAJO]** Bullet en slide D4 (optimizador): el PDF de optimizadores recomienda Adam textual ("muy usado en la práctica").
6. **[BAJO]** Mini-cita en slide 26 ("¿Por qué no llegamos al 98%"): conectar con la curva U del slide 9 del PDF de regularización.

Si solo se hacen las acciones 1 y 2, los slides quedan completos respecto a las preguntas explícitas de `ppt_notes.md`.
