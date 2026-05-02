# SIA TP3 — Perceptrones (ITBA, 1Q 2026)

Implementación desde cero con NumPy.

## Estructura

- [`mlp/`](mlp/) — **Módulo genérico de MLP** (config-driven, NumPy-only). Usado por ej0 (XOR), ej2 (dígitos), ej3 (≥98%).
- [`ejercicio0/`](ejercicio0/) — Validación: escalón (AND), lineal 1D, no-lineal 1D, MLP (XOR).
- [`ejercicio1/`](ejercicio1/) — Detección de fraude (knowledge distillation: TinyModel ≈ BigModel).
- [`ejercicio2/`](ejercicio2/) — Clasificación de dígitos (MLP, sweeps Fase 1+2).
- [`ejercicio3/`](ejercicio3/) — Dígitos con accuracy ≥98% (more_digits.csv + Pack C si necesario).
- [`docs/`](docs/) — Specs, plans, guía para compañeros.
- `data and documentation/` — Datasets oficiales del enunciado.

## Setup

```bash
# Opción A: instalar deps system-wide
pip install --user numpy pandas matplotlib pytest

# Opción B: venv
python3 -m venv .venv
.venv/bin/pip install numpy pandas matplotlib pytest
```

## Tests

```bash
python3 -m pytest mlp/tests/ ejercicio0/tests/ -v
```

## Workflow para experimentos nuevos

Ver [`docs/guia_compañeros.md`](docs/guia_compañeros.md) para el end-to-end.

## Resultados finales

| Ejercicio | val K-fold=5 | test (digits_test.csv) | Comentario |
|---|---:|---:|---|
| Ej2 — base.json | 0.9622 | **0.8630** | distribution shift fuerte train→test |
| Ej3 — base_extra_data | 0.9706 | 0.9636 | +10pp test solo por more_digits.csv |
| Ej3 — l2_aug (ganador) | 0.9760 | **0.9688** | L2 + gaussian noise σ=0.05 |

⚠️ Ej3 target 98% no alcanzado (mejor 96.88%). Análisis completo en [`ejercicio3/README.md`](ejercicio3/README.md).

## Spec y plan

- Spec: `docs/superpowers/specs/2026-05-01-tp3-completion-design.md`
- Plan: `docs/superpowers/plans/2026-05-01-tp3-completion.md`

(Las carpetas `docs/superpowers/` están en `.gitignore` — solo viven en la copia local.)
