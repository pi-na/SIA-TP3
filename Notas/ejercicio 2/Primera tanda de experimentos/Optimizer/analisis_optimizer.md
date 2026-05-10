# Análisis del sweep de optimizadores — Ejercicio 2

**Experimento:** 3 optimizadores (SGD, Momentum, Adam) × 5 LR × 5 seeds × 5 folds.
**SGD:** 50 épocas (experimento anterior). **Momentum y Adam:** 70 épocas.
**Fijo:** arch_base [784, 128, 64, 10], batch=32, sin early stopping, z-score.
**Datos:** `raw.csv` | `epoch_history.csv` | `summary.csv`

---

## Curvas de convergencia por LR

### lr = 1e-4
![convergence_lr1e-4](convergence_lr1e-4.png)

### lr = 5e-4
![convergence_lr5e-4](convergence_lr5e-4.png)

### lr = 1e-3
![convergence_lr1e-3](convergence_lr1e-3.png)

### lr = 5e-3
![convergence_lr5e-3](convergence_lr5e-3.png)

### lr = 1e-2
![convergence_lr1e-2](convergence_lr1e-2.png)

---

## Resultados finales — media ± std sobre 25 corridas (5 seeds × 5 folds)

| Optimizador | LR   | CE train | CE val | Accuracy val  | F1 macro      |
| ----------- | ---- | -------- | ------ | ------------- | ------------- |
| SGD         | 1e-4 | 0.679    | 0.716  | 0.825 ± 0.010 | 0.668 ± 0.010 |
| SGD         | 5e-4 | 0.234    | 0.307  | 0.915 ± 0.005 | 0.797 ± 0.008 |
| SGD         | 1e-3 | 0.150    | 0.251  | 0.932 ± 0.004 | 0.821 ± 0.007 |
| SGD         | 5e-3 | 0.023    | 0.212  | 0.949 ± 0.004 | 0.842 ± 0.005 |
| SGD         | 1e-2 | 0.006    | 0.226  | 0.951 ± 0.004 | 0.845 ± 0.006 |
| Momentum    | 1e-4 | 0.116    | 0.233  | 0.937 ± 0.004 | 0.828 ± 0.005 |
| Momentum    | 5e-4 | 0.013    | 0.218  | 0.950 ± 0.004 | 0.843 ± 0.004 |
| Momentum    | 1e-3 | 0.004    | 0.235  | 0.951 ± 0.004 | 0.845 ± 0.005 |
| Momentum    | 5e-3 | 0.0003   | 0.271  | 0.955 ± 0.004 | 0.849 ± 0.005 |
| Momentum    | 1e-2 | 0.0001   | 0.280  | 0.957 ± 0.005 | 0.852 ± 0.005 |
| Adam        | 1e-4 | ~0       | 0.319  | 0.957 ± 0.005 | 0.851 ± 0.006 |
| Adam        | 5e-4 | ~0       | 0.361  | 0.961 ± 0.004 | 0.856 ± 0.005 |
| Adam        | 1e-3 | ~0       | 0.350  | 0.964 ± 0.005 | 0.859 ± 0.006 |
| Adam        | 5e-3 | 0.038    | 0.674  | 0.950 ± 0.005 | 0.847 ± 0.006 |
| Adam        | 1e-2 | 0.216    | 0.655  | 0.896 ± 0.020 | 0.789 ± 0.022 |

---

## Observaciones

### 1. SGD necesita LR alto o muchas épocas para converger

Con lr=1e-4, SGD llega a epoch 50 con CE de entrenamiento en 0.679 — prácticamente sin haber aprendido. Con lr=1e-2 converge en 50 épocas y da los mejores resultados para SGD (accuracy 0.951). El problema de SGD es que da pasos de tamaño fijo en la misma dirección del gradiente, sin memoria ni adaptación. Para aprender bien necesita o un LR alto o muchas más épocas.

### 2. Momentum converge más rápido que SGD con el mismo LR

En el gráfico de lr=1e-4 se ve claramente: Momentum (naranja) baja mucho más rápido que SGD (azul). Con lr=5e-4, Momentum llega a CE de train ~0.013 en 70 épocas mientras SGD queda en ~0.234. Momentum acumula velocidad en la dirección del gradiente (como una pelota que rueda cuesta abajo), lo que le permite avanzar más rápido en zonas con gradiente consistente.

### 3. Adam converge drásticamente más rápido con LR bajos

Con lr=1e-4, Adam (verde) baja su train loss a casi 0 en las primeras 10 épocas, mientras SGD todavía está en 2.0 y Momentum en 0.5. Adam adapta el tamaño efectivo del paso para cada parámetro por separado — donde el gradiente es pequeño da pasos más grandes, donde es grande los reduce. Esto le permite avanzar rápido incluso con LR nominalmente chico.

### 4. Adam es inestable con LR altos

Con lr=5e-3 y lr=1e-2, las curvas de Adam oscilan violentamente — la train loss fluctúa y en algunos casos la val loss sube en lugar de bajar. Con lr=1e-2 la val_acc cae a 0.896 con std de 0.020, mucho peor que con lr bajos. Esto ocurre porque Adam ya adapta internamente el tamaño de paso — si encima le das un LR alto, los pasos efectivos son demasiado grandes y el modelo salta por encima de los mínimos.

### 5. Val loss de Adam sube con el tiempo a LR intermedios

Con lr=1e-3, la val loss de Adam (gráfico derecho) baja rápido en las primeras épocas pero después empieza a subir gradualmente desde la época ~15 en adelante. La train loss en cambio sigue bajando hasta casi 0. Esto es la señal de sobreajuste: el modelo terminó de aprender la distribución general y empieza a memorizar el training set. Con SGD y Momentum este fenómeno es menos pronunciado porque convergen más lento.

### 6. La zona segura de Adam es lr entre 1e-4 y 1e-3

| LR | Accuracy val (Adam) | F1 macro (Adam) | Estable |
|---|---|---|---|
| 1e-4 | 0.957 ± 0.005 | 0.851 | Sí |
| 5e-4 | 0.961 ± 0.004 | 0.856 | Sí |
| 1e-3 | 0.964 ± 0.005 | 0.859 | Sí |
| 5e-3 | 0.950 ± 0.005 | 0.847 | No (oscila) |
| 1e-2 | 0.896 ± 0.020 | 0.789 | No (diverge) |

---

## Decisión de optimizador y LR base

Se elige **Adam con lr=1e-3** como configuración base para los próximos pasos porque:

- Tiene la mejor accuracy (0.964) y F1 macro (0.859) de todas las combinaciones probadas
- Converge en pocas épocas — la curva de train loss llega a casi 0 antes de la época 20
- Es la combinación que mejor explota la capacidad de Adam (adaptación por parámetro) sin entrar en la zona inestable
- La val loss con lr=1e-3 sube levemente después de la época 15, lo que indica que el próximo paso a estudiar es la regularización para controlar ese sobreajuste
