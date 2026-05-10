# Ejercicio 2 — Experimentación MLP

Carpeta de trabajo para todos los experimentos del Ej2 (MLP sobre `digits.csv`).

## Estructura

```
ejercicio2_experimentacion/
├── scripts/         # runners de experimentos (uno por sweep)
├── configs/         # configs JSON: arquitecturas, sweeps, base.json
├── output/          # raw outputs de cada corrida (CSVs, weights, history)
├── analisis/        # análisis de los outputs, una subcarpeta por experimento
├── analisis_dataset/
└── presentacion/
```

## Flujo de un experimento

1. **Config** → crear el JSON correspondiente en `configs/` (arquitectura,
   sweep, hiperparámetros). Reusar `configs/base.json` como punto de partida
   y modificar one-at-a-time.
2. **Script** → armar un nuevo runner en `scripts/` copiando
   `scripts/runner_ejemplo_multiprocess.py` como plantilla. Solo hay que
   adaptar `_build_cfg_for_combo` y la construcción de `jobs` al grid del
   sweep nuevo. El resto (ProcessPoolExecutor, OMP=1, escritura de CSVs)
   queda igual.
3. **Ejecutar** → el runner escribe los raw outputs en `output/<nombre>/`.
4. **Análisis** → crear `analisis/<nombre_experimento>/` con los plots,
   tablas y un `analisis.md` que interprete los resultados. **Cada decisión
   tiene que tener hipótesis previa, experimento y interpretación** (regla 2
   del CLAUDE.md raíz).

## Reglas que aplican (ver CLAUDE.md raíz)

- Promedios: explicitar **qué** se promedia y **sobre qué eje**
  (seeds vs folds vs épocas). Nombrar columnas tipo `acc_mean_seedsfolds`.
- Reportar **el set completo de métricas**: train/val loss + accuracy +
  macro precision/recall/F1. Multiclase → macro-average por default.
- `digits_test.csv` **NO se toca** durante búsqueda de hiperparámetros.
- Toda la HP search vive en `digits.csv` con CV interno (k=5).
