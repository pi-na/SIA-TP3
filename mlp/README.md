# `mlp/` — Perceptrón Multicapa Genérico

Módulo NumPy-only para entrenar redes feedforward configurables. Diseñado para
ser invocado con `config.json` y producir CSVs por corrida (sin plots).

## Uso

```bash
python3 -m mlp.train \
    --config ejercicio2/configs/sweeps_fase1/arch_100_50.json \
    --csv-root . \
    --output-dir ejercicio2/output \
    --workers 5
```

Output: carpeta `output/<model_name>_<timestamp>/` con:

- `config.json` — copia íntegra del input.
- `run_summary.csv` — una fila por fold + filas `mean`/`std`.
- `epoch_history.csv` — métricas train+val por época.
- `predictions.csv` — predicciones out-of-fold con scores por clase.
- `confusion_matrix.csv` — formato stacked (fold, true, pred, count).
- `weights.npz` — pesos finales por fold.

## API mínima

```python
from mlp.network import MLP
from mlp.optimizers import Adam

mlp = MLP(
    layer_sizes=[784, 100, 10],
    activations=["relu", "softmax"],
    loss="cross_entropy",
    optimizer=Adam(lr=0.001),
    initializer="auto",
    seed=42,
)
history = mlp.fit(X_train, y_train_onehot, X_val, y_val_onehot,
                  epochs=50, batch_size=64, early_stopping_patience=10)
preds = mlp.predict(X_test)
mlp.save("model.npz")
loaded = MLP.load("model.npz")
```

## Schema del config

Ver `docs/superpowers/specs/2026-05-01-tp3-completion-design.md` §3.6.

## Cómo agregar un optimizador nuevo

1. Subclasear `Optimizer` en `mlp/optimizers.py` con `__init__` y `step()`.
2. Registrar en `OPTIMIZERS = {...}`.
3. Agregar test en `mlp/tests/test_optimizers.py` (toy quadratic problem).
4. Actualizar la lista de validación en `train.py:load_and_validate_config`.

## Cómo agregar una activación nueva

1. Definir `act(z)` y `act_grad(z, a)` en `mlp/activations.py`.
2. Registrar en `ACTIVATIONS = {...}`.
3. Agregar caso en `auto_pick` en `mlp/initializers.py`.
4. Test parametrizado en `mlp/tests/test_activations.py`.

## Tests

```bash
python3 -m pytest mlp/tests/ -v
```

| Test file | Qué valida |
|---|---|
| `test_activations.py` | Gradientes vs diferenciación numérica |
| `test_losses.py` | Gradientes vs diferenciación numérica |
| `test_initializers.py` | Distribución de pesos correcta + reproducibilidad |
| `test_optimizers.py` | SGD/Momentum/Adam reducen loss en problema cuadrático |
| `test_data.py` | Stratified K-fold + parsing de stringified arrays |
| `test_metrics.py` | Accuracy + macro/weighted + confusion matrix |
| `test_network.py` | Init + forward + backward (gradient check) |
| `test_xor.py` | Convergencia XOR (regression test del integration) |
| `test_save_load.py` | Round-trip preserva pesos |
| `test_train_cli.py` | Validación de config |
| `test_train_smoke.py` | E2E sobre datos sintéticos chicos |
