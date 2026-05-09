# Preguntas abiertas
## Experimentacion, variables
Por ejemplo, en el analisis de LR [[ejercicio1/nonlinear_perceptron/analisis_outputs/sweep_lr/multiseed/analisis|analisis multiseed non linear]]
V## Decisiones de diseño, hiperparametros
#### Normalización de la data
Antes de hacer K-folding, normalizamos la informacion haciendo z-score.
Para cada feature:
$$(x - μ) / σ$$Entonces cada feature queda con media 0 y std 1. Con z-score bien aplicado, todas las features entran al perceptrón en escala comparable.
#### K FOLD
Hacemos K-Fold estratificado. Hay que explicar como hacemos el K-fold (y explicar por que es "estratificado"... es lo de la división de clases previo a la partición).
Hay que explicar por que elegimos K = 5. Hay que tener algun tipo de justificacion, que esta faltante. 
La justificacion **probablemente** la consigamos por medio de experimentación, variando valores de K. Quizás hasta llegamos a la conclusión de que conviene otro K.
 1. Construcción de los folds (make_stratified_folds / make_simple_folds)                                     
  - En el estratificado: se mezclan por separado los índices de fraude y los de no-fraude (rng.shuffle(pos_idx), rng.shuffle(neg_idx) — linear:126-127, nonlinear:131-132) antes de repartirlos en los K folds.                                  
  - En el simple: se mezclan todos los índices (rng.shuffle(idx) — linear:151, nonlinear:154).         
  - Usa el seed "raíz" del config: seed = config["random_seed"] (linear:340, nonlinear:403). 
#### Estrategia de entrenamiento
Hacemos entrenamiento **online**, no intentamos implementar batch o mini batch para el lineal.
En clase vimos que el lineal se entrena con entrenamiento online.
#### Inicializacion de los pesos
Es igual para el perceptron lineal y el no lineal.

La inicializacion de los pesos se hace con valor aleatorio distribución uniforme. 

``` python
rng = np.random.default_rng(seed)                                                
weights = rng.uniform(-0.1, 0.1, size=n + 1) 
```
- Distribución uniforme en [-0.1, 0.1].                                                                                         
  - Tamaño n + 1 donde n = len(feature_cols). La componente weights[0] es el bias, las restantes son los pesos de cada feature.   
  - Seed viene del config (random_seed); cada fold usa seed + k_i (linear_perceptron.py:349, nonlinear_perceptron.py:417), así    
  dentro de una corrida los folds tienen inicializaciones distintas pero reproducibles.                   
#### Criterio de corte
Usamos un epsilon, pero en todos los experimentos que hicimos (vease los analisis multiseed de LR para ambos perceptrones), encontramos que los perceptrones terminan por limite de epocas, y convergen (se nota por la pendiente en el grafico de convergencia) bastante antes del limite de epocas. 
### Hiperparametros
Los hiperparametros del perceptron lineal ADALINE y no-lineal son 
#### Learning rate
Experimentamos con 3 LR, el analisis de resultados para cada uno esta en 
- [[ejercicio1/nonlinear_perceptron/analisis_outputs/sweep_lr/multiseed/analisis|analisis multiseed NO LINEAL]]
- [[ejercicio1/lineal_perceptron/analisis_outputs/sweep_lr/multiseed/analisis|analisis multiseed LINEAL]]
Para el perceptrton lineal elegimos LR = 10^-3
Para el perceptrton no lineal elegimos LR = 10^-2
#### epochs
Cuando hicimos los experimentos de LR vimos que dependiendo el LR, 150 epocas o 500 epocas sobra para que converge.
#### epsilon — criterio de parada anticipada         
#### threshold
Lo que hacemoss para el threshold es medir precision recall accuracy F1 y encontrar el balance que no parezca. En los analisis de LR para el LINEAL lo explicamos:

> Criterio para elegir un threshold: nos interesa conseguir la mejor recall posible sin dejar de lado la precision; Notamos que con thresholds bajos la recall queda alta, pero con poca precision. Estamos flaggeando demasiadas compras como fraude, y asi es facil agarrar la mayor cantidad de fraudes posible. Entonces queremos un **balance** entre precision y recall -> usamos F1 para la decisión.

--- 
## Resultados
Una vez definidos los hiperparametros, pasamos a analizar los resultados del modelo.
### Generalizacion
Tenemos que mostrar los resultados de la generalización.

> Generalización = qué tan bien el modelo predice en datos que no vio durante el entrenamiento.

Un modelo que memoriza el train set (overfitting) tiene MSE train bajo pero MSE test alto. Un modelo que generaliza tiene MSE similares en ambos.                                            

Para medirlo usamos K-fold cross-validation: entrenamos K veces en particiones distintas y promediamos las métricas de test, así la estimación no depende de una sola división suerte/mala suerte.

Hay que mostrar generalización para el perceptron lineal Y TAMBIÉN para el no lineal.
#### **Perceptron lineal**
Hola completame!
#### **Perceptron no lineal**
**No se observa overfitting**, vemos que la linea de error para el training set y para el test set estan pisadas. ACA FALTA UN GRAFICO.
### Convergencia - subajuste?
Como medimos si el modelo ajusta correctamente al problema? Tenemos el MSE. En clase dijeron MSE alto en training set -> underfitting. Pero como juzgamos si un MSE es alto? Cual es la referencia?

Lo que hicimos fue analizar el dataset [[informe_analisis_dataset.pdf]]. Encontramos reglas básicas que dividen el dataset a partir de un umbral en 3 features, y calculamos métricas para el sistema que únicamente hace esta división por umbrales. Los resultados fueron precision=100% / recall=80% / acc=97.68%.

Definimos que hay **underfitting** si el perceptrón en train no iguala (acc ≥ 97.68%, precision ≥ 100% sobre lo que predice, recall ≥  80%).

Al menos, queremos que iguale o mejore el **recall** para considerar que vale la pena usar el perceptron.

Predicción teórica para el lineal: las 3 reglas son discontinuidades (salto ~6% → 100% en un entero). Geométricamente imposibles para una recta + sigmoide saturada → el lineal debería underfittear estructuralmente, y el no-lineal acercarse más.

Tenemos que calcular estas metricas para el lineal y el no lineal con los hiperparametros que elegimos antes, y compararlas con el baseline.
