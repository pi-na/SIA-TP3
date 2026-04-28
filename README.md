# SIA TP3 — Perceptrones (ITBA, 1Q 2026)

Implementación de perceptrones desde cero con NumPy.

## Estructura

- [`ejercicio0/`](ejercicio0/) — Validación: perceptrón lineal (Adaline) y perceptrón no lineal (tanh).
- [`ejercicio1/`](ejercicio1/) — Detección de fraude (Knowledge Distillation: TinyModel replica BigModel).
- `data and documentation/` — Datasets y documentación del enunciado.
- `docs/` — Notas de skills/superpowers.
- `Enunciado TP3 - 1Q 2026.pdf` — Consigna oficial.
- `apunte perceptron.md` — Apunte teórico.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib pytest
```

## Tests

Cada ejercicio tiene su propio `conftest.py` y suite de tests. Para correr los del ejercicio 0:

```bash
cd ejercicio0 && ../.venv/bin/pytest -v
```
