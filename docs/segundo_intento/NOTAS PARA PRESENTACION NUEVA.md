# Ejercicio 1 perceptron lineal y no lineal
## Decisiones de diseño, hiperpametros
#### Normalización de la data
Antes de hacer K-folding, normalizamos la informacion haciendo z-score.
Para cada feature:
$$(x - μ) / σ$$Entonces cada feature queda con media 0 y std 1. Con z-score bien aplicado, todas las features entran al perceptrón en escala comparable
#### K FOLD
Hacemos K-Fold estratificado. Hay que explicar como hacemos el K-fold (y explicar por que es "estratificado"... es lo de la división de clases previo a la partición), y por que elegimos K = 5. Hay que tener algun tipo de justificacion, que esta faltante. 
La justificacion **probablemente** la consigamos por medio de experimentación, variando valores de K. Quizás hasta llegamos a la conclusión de que conviene otro K.
#### Estrategia de entrenamiento
Hacemos entrenamiento **online**, no intentamos implementar batch o mini batch para el lineal.
Si preguntan decimos que no quisimos darle tanto tiempo al ej1 y que nos parecio suficiente para la complejidad del asunto.
#### Inicializacion de los pesos
La inicializacion de los pesos se hace con valor aleatorio distribución uniforme.
Si preguntan decimos que no quisimos darle tanto tiempo al ej1 y que nos parecio suficiente para la complejidad del asunto.
#### Hiperparametros
Los hiperparametros del perceptron lineal ADALINE y no-lineal son 
  - learning rate
	  - Justificar elección del learning rate con convergencia.
  - epochs — más épocas para LRs pequeños (actualmente 7500, podría probar 15000)
  - epsilon — criterio de parada anticipada         
  - threshold
	  - analizar bien como lo vamos a evaluar, ahora tenemos ese grafico con 3 curvas precision / recall / f1 **que fue super confuso en la presentacion**

**Experimentacion para decisión del LEARNING RATE en el perceptron NO LINEAL**
Armé plots de convergencia para cada learning rate:

![[lr_0_001.png|450]]
![[lr_0_01.png|451]]
![[lr_0_0001.png|456]]
![[sweep_all_zoom50.png|469]]

Las tres tasas (0.01, 0.001, 0.0001) producen resultados  prácticamente idénticos (MSE 0.0110-0.0113). La diferencia esta en la cantidad de epocas que les toma converger a cada una: lr=0.01 es más rápido (~30 épocas) que lr=0.001 (~100).

Hay que elegir entre 0.01 y 0.001; Podriamos decir que LR=0.001 nos 

#### Regularización
No implementamos regularización (tipo L2 decay). Tema complejidad del ejercicio1.

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

Predicción teórica para el lineal: las 3 reglas son discontinuidades (salto ~6% → 100% en un entero). Geométricamente imposibles para una recta + sigmoide saturada → el lineal debería underfittear estructuralmente, y el no-lineal acercarse más.

Tenemos que calcular estas metricas para el lineal y el no lineal con los hiperparametros que elegimos antes, y compararlas con el baseline.

# Correcciones especificas de la presentacion actual
slide 3
Eliminar "regla del enunciado"

