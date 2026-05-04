# Early Stopping

## ¿Qué es?

**Early Stopping** es una técnica de regularización simple que detiene el entrenamiento de un modelo cuando el error de validación deja de mejorar. El objetivo es evitar **overfitting** manteniéndose en el punto óptimo de capacidad del modelo.

## Intuición Teórica

Durante el entrenamiento de una red neuronal, existe un **punto óptimo de capacidad** donde el modelo:
- Aprende bien los datos de entrenamiento
- Generaliza correctamente a datos nuevos

### Dos regímenes problemáticos:

1. **Underfitting** (capacidad insuficiente)
   - Error de entrenamiento no baja
   - Solución: aumentar capacidad del modelo (arquitectura más grande, función de activación no lineal)

2. **Overfitting** (capacidad excesiva)
   - Error de entrenamiento sigue bajando
   - Pero error de validación empieza a subir
   - Solución: **aquí entra early stopping**

### Visualización:

```
Error
  │     Training error ↓
  │    ╱╲
  │   ╱  ╲_____ (continúa bajando)
  │  ╱
  │ ╱ Validation error ↓ luego ↑
  │╱_╲_╱╲__
  │      ╲_____(empieza a subir → OVERFITTING)
  │
  └────────────────────→ Épocas
     ↑ Punto óptimo (aquí deberíamos parar)
```

## Cómo se Aplica

### Concepto

1. Entrenas época a época
2. En cada época, evalúas el error en validación
3. Haces seguimiento del **mejor error de validación visto hasta ahora**
4. Si el error no mejora durante N épocas consecutivas (`patience`), **detenes el entrenamiento**
5. **Restauras los pesos del mejor modelo** (no del último)

### Parámetro clave

- **`early_stopping_patience`** (ej: 5, 10, 15)
  - Número de épocas consecutivas sin mejora antes de parar
  - Rango típico: 5-20 épocas

### Ejemplo con patience=5:

```
Época | Val Error | Mejora? | Contador | Acción
------|-----------|---------|----------|--------
  1   |   0.50    |   ✓     |    0     | Guardar pesos
  2   |   0.45    |   ✓     |    0     | Guardar pesos
 ...
  50  |   0.40    |   ✓     |    0     | Guardar pesos (MEJOR)
  51  |   0.41    |   ✗     |    1     | 
  52  |   0.42    |   ✗     |    2     | 
  53  |   0.43    |   ✗     |    3     | 
  54  |   0.44    |   ✗     |    4     | 
  55  |   0.45    |   ✗     |    5     | STOP → Restaurar pesos de época 50
```

## Implementación en el Proyecto

### 1. Configuración en `train.py` (línea 152)

```python
mlp.fit(
    X_train, y_train, X_val, y_val,
    epochs=train_cfg["epochs"],
    batch_size=train_cfg["batch_size"],
    early_stopping_patience=train_cfg.get("early_stopping_patience"),  # ← desde config.json
    callback=on_epoch,
)
```

El parámetro se obtiene del archivo de configuración JSON (ej: `base.json`).

### 2. Lógica en `mlp/network.py`, método `fit()` (líneas 199-210)

```python
# Inicialización (líneas 152-154)
best_val_loss = float("inf")      # Mejor error de validación visto
epochs_no_improvement = 0         # Contador de épocas sin mejora
best_weights = None               # Almacena pesos del mejor modelo

# En cada época (líneas 200-210)
if early_stopping_patience is not None:
    if val_loss < best_val_loss:
        # ✓ Mejora encontrada
        best_val_loss = val_loss
        epochs_no_improvement = 0
        best_weights = [W.copy() for W in self.weights]  # Guardar pesos
    else:
        # ✗ Sin mejora
        epochs_no_improvement += 1
        if epochs_no_improvement >= early_stopping_patience:
            # Restaurar mejor modelo y detener
            if best_weights is not None:
                self.weights = best_weights
            break
```

### 3. Registro en `train.py` (línea 171)

Se calcula el epoch con mejor validación loss:

```python
"best_epoch": int(np.argmin([h["val_loss"] for h in history_compact])),
```

Este valor aparece en `run_summary.csv` para validar que early stopping funcionó.

## Cuándo Usar Early Stopping

✓ **Úsalo cuando:**
- Ya estás entrenando bien los datos (error de entrenamiento baja)
- Ves que el error de validación empieza a crecer (overfitting)
- Quieres automatizar la detención sin adivinar el número de épocas

✗ **NO lo uses cuando:**
- El error de entrenamiento no baja (underfitting)
  - Solución: aumentar capacidad del modelo, no regularizar

## Configuración Recomendada

Para `config.json`:

```json
{
  "training": {
    "epochs": 1000,
    "batch_size": 32,
    "early_stopping_patience": 10,
    ...
  }
}
```

- **`epochs`**: Máximo alto (ej: 1000) — early stopping parará antes si es necesario
- **`early_stopping_patience`**: 5-15 épocas según dataset
  - Datasets pequeños: patience=5
  - Datasets grandes: patience=10-15

## Ventajas

1. ✓ **Simple**: solo 1 parámetro (`patience`)
2. ✓ **Automático**: no necesitas saber cuántas épocas exactas
3. ✓ **Evita overfitting**: detiene antes de degradarse
4. ✓ **Ahorra tiempo**: puede entrenar menos que el máximo de épocas

## Ver También

- Clase de regularización: `docs/clase_regularizacion/regularizacion parte 1.VTT` (minutos 19:36 - 34:00)
- Otros métodos de regularización: L2/Weight Decay, Data Augmentation, Dropout
