# Ejercicio 1 — Detección de fraude (Knowledge Distillation)

## Objetivo

Entrenar un **TinyModel** (perceptrón simple) que replique la salida de probabilidad de fraude de un **BigModel**.

## Dataset

- `../data and documentation/fraud_dataset.csv`
- **Target de entrenamiento**: `big_model_fraud_probability` (float en `[0, 1]`).
- **`flagged_fraud` NO se usa para entrenar** — es ground truth para evaluación final.
- **Features**: `timestamp`, `amount_usd`, `quantity_purchased`, `session_duration_seconds`, `days_since_last_purchase`, `account_age_days`, `device_screen_resolution`, `time_since_last_login_s`, `items_viewed_before_purchase`.

## Tareas

1. Comparar perceptrón simple lineal vs no lineal (underfitting, saturación de capacidad).
2. Estudio de generalización del perceptrón seleccionado: métricas, estrategia de train/test split, recomendación de umbral de fraude.

## Notas de implementación

- La salida debe estar en `[0, 1]` → activación sigmoide/logística.
- Implementación desde cero con NumPy (sin sklearn/PyTorch/TensorFlow).
