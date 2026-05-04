# docs/ — Índice

Documentación, presentaciones e informe del TP3.

## Estructura

Cada documento `.tex` vive en su propia carpeta junto al `.pdf` compilado:

| Carpeta | Contenido |
|---|---|
| `slides_presentacion/` | **Presentación final** (36 slides, versión completa) |
| `slides_resumida/` | Versión resumida de la presentación (21 slides) |
| `informe_tp3/` | Informe escrito del TP |
| `guion_presentacion/` | Guion para la exposición |
| `bitacora_experimentacion/` | Bitácora cronológica de experimentación |
| `notas/` | Notas conceptuales en Markdown |
| `clase_optimizadores/` | Material de cátedra: PDF + VTT |
| `clase_regularizacion/` | Material de cátedra: PDF + VTTs |
| `weight_initialization/` | Material de cátedra: HTML sobre init de pesos |

## Notas conceptuales (`notas/`)

| Archivo | Tema |
|---|---|
| `decisiones_y_backing_teorico.md` | Backing teórico completo: para cada decisión, qué dice la cátedra y qué hicimos |
| `cross_entropy.md` | Cross-entropy + softmax: por qué y cómo funciona el gradiente |
| `early_stopping.md` | Early stopping: definición, implementación, configuración |
| `relu.md` | ReLU: definición, neuronas muertas, He init, vs sigmoid/tanh |
| `lr_sweep_conclusion.md` | Tablas resumidas del sweep de learning rate (Fase 1 y Fase 2) |
| `explicacion_distribution_shift.md` | Qué es distribution shift y cómo apareció en el TP3 |
| `clase_regularizacion_cosas_implementadas.md` | Matching 1:1 entre PDF de regularización y lo implementado |
| `resultados_ejercicio3_explicacion.md` | Cada técnica de regularización: qué es, cómo se implementa, qué dio |
| `auditoria_slides.md` | Auditoría de los slides contra `ppt_notes.md` |
| `guia_compañeros.md` | Guía para los compañeros del grupo |

## Compilar un documento

Cada documento se compila desde su propia carpeta:

```bash
cd docs/slides_presentacion
pdflatex slides_presentacion.tex
```

Los `.tex` que incluyen imágenes referencian `../../ejercicio*/...` (dos niveles arriba), así que `pdflatex` debe correrse desde la carpeta del `.tex`.
