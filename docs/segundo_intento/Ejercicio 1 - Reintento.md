## Decisiones de diseño, hiperparametros
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
#### Hiperparametros
>[!warning] HAY QUE JUSTIFICAR LOS CUATRO HP!!

Los hiperparametros del perceptron lineal ADALINE y no-lineal son 
  - learning rate
	  - Justificar elección del learning rate con convergencia.
  - epochs — más épocas para LRs pequeños (actualmente 7500, podría probar 15000)
  - epsilon — criterio de parada anticipada         
  - threshold

  >[!warning] evaluacion del threshold
  > HAY QUE REHACER LA EVALUACION!! HACER MEJORES EXPERIMENTOS Y DAR UNA JUSTIFICACION LIMPIA SOBRE LA DECISION

**Experimentacion para decisión del LEARNING RATE en el perceptron NO LINEAL:**
Armé plots de convergencia para cada learning rate:

![[lr_0_001.png|609]]
![[lr_0_01.png|574]]
![[lr_0_0001.png|672]]

Las tres tasas (0.01, 0.001, 0.0001) producen resultados  prácticamente idénticos (MSE 0.0110-0.0113). La diferencia esta en la cantidad de epocas que les toma converger a cada una: lr=0.01 es más rápido (~30 épocas) que lr=0.001 (~100).

>[!warning] CAMBIOS EN ESTA EJECUCION
>DEJE EJECUTANDO OTRO SCRIPT Q PRUEBA MUCHOS SEED Y CALCULA POSTA PROMEDIO Y DESVIO DE MSE, JUNTO CON MAS METRICAS, PARA EVALUAR BIEN 


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
