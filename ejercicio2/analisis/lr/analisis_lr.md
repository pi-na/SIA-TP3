# Análisis del sweep de learning rates — Ejercicio 2

**Experimento:** 4 arquitecturas × 5 LR × 5 seeds × 5 folds = 500 corridas totales.
**Fijo en todas:** Adam, batch=32, z-score, 50 épocas, early stopping patience=10.
**Datos crudos:** `raw.csv` | **Curvas por época:** `epoch_history.csv` | **Resumen:** `summary.csv`

---

## LR explorados

| LR | Etiqueta |
|---|---|
| 0.0001 | 1e-4 |
| 0.0005 | 5e-4 |
| 0.001  | 1e-3 (baseline del sweep de arq.) |
| 0.005  | 5e-3 |
| 0.01   | 1e-2 |

---

## Resultados (media ± std sobre 25 corridas — 5 seeds × 5 folds)

| Arquitectura | LR | CE train final | CE val final | Accuracy val | F1 macro |
|---|---|---|---|---|---|
| shallow | 1e-4 | 0.0090 | 0.1778 | 0.9551 ± 0.0056 | 0.8498 ± 0.0080 |
| shallow | 5e-4 | 0.0016 | 0.1880 | 0.9568 ± 0.0054 | 0.8515 ± 0.0077 |
| shallow | 1e-3 | 0.0034 | 0.2030 | 0.9576 ± 0.0061 | 0.8522 ± 0.0089 |
| shallow | 5e-3 | 0.0855 | 0.7094 | 0.9366 ± 0.0137 | 0.8308 ± 0.0159 |
| shallow | 1e-2 | 0.1762 | 1.1022 | 0.9375 ± 0.0107 | 0.8305 ± 0.0124 |
| base    | 1e-4 | 0.0055 | 0.1983 | 0.9532 ± 0.0044 | 0.8478 ± 0.0060 |
| base    | 5e-4 | 0.0009 | 0.2111 | 0.9551 ± 0.0049 | 0.8498 ± 0.0064 |
| base    | 1e-3 | 0.0032 | 0.2272 | 0.9554 ± 0.0059 | 0.8500 ± 0.0078 |
| base    | 5e-3 | 0.0762 | 0.4491 | 0.9409 ± 0.0078 | 0.8337 ± 0.0095 |
| base    | 1e-2 | 0.1874 | 0.5282 | 0.9202 ± 0.0123 | 0.8098 ± 0.0177 |
| wider   | 1e-4 | 0.0025 | 0.1945 | 0.9551 ± 0.0054 | 0.8501 ± 0.0068 |
| wider   | 5e-4 | 0.0051 | 0.2159 | 0.9573 ± 0.0058 | 0.8535 ± 0.0072 |
| wider   | 1e-3 | 0.0066 | 0.2511 | 0.9557 ± 0.0062 | 0.8505 ± 0.0075 |
| wider   | 5e-3 | 0.1052 | 0.4525 | 0.9406 ± 0.0092 | 0.8335 ± 0.0098 |
| wider   | 1e-2 | 0.3794 | 0.6590 | 0.9071 ± 0.0105 | 0.7954 ± 0.0164 |
| deeper  | 1e-4 | 0.0045 | 0.2130 | 0.9507 ± 0.0047 | 0.8446 ± 0.0065 |
| deeper  | 5e-4 | 0.0012 | 0.2275 | 0.9525 ± 0.0063 | 0.8472 ± 0.0082 |
| deeper  | 1e-3 | 0.0070 | 0.2498 | 0.9522 ± 0.0059 | 0.8471 ± 0.0079 |
| deeper  | 5e-3 | 0.0549 | 0.3378 | 0.9456 ± 0.0083 | 0.8403 ± 0.0090 |
| deeper  | 1e-2 | 0.1347 | 0.4135 | 0.9332 ± 0.0102 | 0.8196 ± 0.0196 |

---

## Observaciones

### 1. Zona "buena": lr entre 1e-4 y 1e-3

Los tres LR bajos (1e-4, 5e-4, 1e-3) producen resultados similares en accuracy y F1 macro — la diferencia máxima entre ellos es de ~0.5pp en accuracy y ~0.003 en F1, dentro del std de cada configuración. No hay diferencia estadísticamente significativa en las métricas de clasificación en ese rango.

Sin embargo, la **CE de validación** sí diferencia: lr=1e-4 tiene la val_loss más baja en casi todos los casos (ej. shallow: 0.178 vs 0.203 con lr=1e-3). El modelo con lr más chico converge más lento pero llega a una solución que se ajusta menos a los datos de entrenamiento.

### 2. Zona "mala": lr ≥ 5e-3

Con lr=5e-3 y lr=1e-2 la performance cae notablemente:

| Zona | Accuracy val (shallow) | CE val (shallow) |
|---|---|---|
| lr=1e-4 a 1e-3 | 0.955–0.958 | 0.178–0.203 |
| lr=5e-3 | 0.937 | 0.709 |
| lr=1e-2 | 0.938 | 1.102 |

La caída es especialmente visible en la CE de validación, que se dispara. En las curvas de convergencia (`convergence_val.png`) se ve que con lr=5e-3 y 1e-2 el entrenamiento oscila o no converge limpiamente — el early stopping para antes de que el modelo haya encontrado un mínimo estable.

Esto es consistente con lo de la clase de optimizadores: con lr muy alto, los pasos de Adam son tan grandes que el modelo salta por encima de los mínimos en lugar de converger hacia ellos.

### 3. El sobreajuste persiste en toda la zona "buena"

Incluso con los LR bajos, la brecha train/val sigue siendo grande:

| LR | CE train (shallow) | CE val (shallow) | Ratio |
|---|---|---|---|
| 1e-4 | 0.0090 | 0.178 | ~20× |
| 5e-4 | 0.0016 | 0.188 | ~117× |
| 1e-3 | 0.0034 | 0.203 | ~60× |

Con lr=1e-4 el ratio baja a ~20×, pero sigue siendo un sobreajuste claro. El LR no resuelve el sobreajuste — solo lo modera. Esto es coherente con la clase de regularización: la herramienta correcta para cerrar la brecha train/val es la **regularización** (L2, dropout), no el LR.

### 4. Las curvas de convergencia (`convergence_gap.png`) lo muestran

La brecha val_loss − train_loss abre rápido con todos los LR en la zona buena. Con lr=1e-4 la apertura es más gradual (convergencia más lenta), con lr=1e-3 abre más rápido porque el modelo baja el train_loss más agresivamente. Con lr=5e-3 y 1e-2 la brecha es errática (oscilaciones).

---

## Decisión de LR base

Se elige **lr=1e-4** como LR base para el siguiente sweep (regularización) porque:

- Tiene la **CE de validación más baja** en todas las arquitecturas (mejor generalización medida por la loss de entrenamiento)
- Las métricas de clasificación (accuracy, F1) son equivalentes a lr=5e-4 y lr=1e-3 (diferencias dentro del std)
- La convergencia es más estable (menos oscilación en las curvas de época)
- El sobreajuste sigue presente pero es menos severo (~20× vs ~60–117×), lo que da más margen para que la regularización lo corrija

La diferencia entre lr=1e-4 y lr=1e-3 en accuracy (~0.3pp para shallow) es ruido estadístico. Lo que importa para el siguiente paso es partir de la configuración con menor brecha train/val, para que la regularización tenga el mayor efecto posible.

---

## Próximo paso

Con arch_base [784, 128, 64, 10] + lr=1e-4 + Adam como base:
- Sweep de **regularización**: L2 (λ ∈ {0, 1e-4, 1e-3, 1e-2}) y dropout (p ∈ {0, 0.1, 0.3, 0.5})
- Objetivo: cerrar la brecha train/val sin sacrificar accuracy ni F1 macro
