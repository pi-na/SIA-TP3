**JUSTIFICACION DECISIONES DE VALORES DE MOMENTUM Y ADAM**
[[clase optimizadores.pdf#page=13]]
queda justificar CROSS ENTROPY y softmax

**Arquitectura**                                                            
  - Número de capas ocultas                                               
  - Número de neuronas por capa                             
  - Función de activación por capa (sigmoid, tanh, ReLU)
  **Entrenamiento**                                                           
  - Tasa de aprendizaje (learning rate)
  - Cantidad de épocas                                                    
  - Tamaño del batch (batch size) — determina cuántas muestras ve el
  modelo antes de actualizar los pesos
  **Optimizador** (de la clase de optimizadores)
  - Tipo: SGD / SGD con momentum / Adam                                   
  - Parámetros internos del optimizador: β para momentum; β₁, β₂ para Adam
  **Regularización** (de la clase de regularización)                          
  - L2 — coeficiente λ                                                    
  - Dropout — probabilidad p de apagado de neuronas                       
  - Early stopping — paciencia (cuántas épocas sin mejora se toleran antes de frenar)                                                             
  **Inicialización de pesos**                                                 
  - Uniforme / He / Xavier                                               
  **Preprocesamiento**
  - Normalización de los inputs (z-score por feature)
# Pre-analisis del dataset
aca lo normalizamos

# Decisión de arquitectura Base

Con respecto a la arquitectura lo que a mi se me ocurrio es generar dos capas ocultas que lo que hagan es: 
- en la primer capa oculta, se dividan en pequeños pedacitos los números
- en la segunda capa oculta, se combinen esas estructuras (loops, lineas verticales y cruces) 
La capa de salida me matchee esa combinación con el numero de salida

![[Pasted image 20260509151211.png]]![[Pasted image 20260509151252.png|451]]

En cuanto al tamaño de cada capa, elegimos un tamaño que sea razonable en cuanto a la cantidad de entradas que tenemos: [784, 128, 64, 10]
# Variación de arquitecturas

![[Pasted image 20260509152531.png]]

 - **Shallow** debería tener más error que Base — le falta un nivel de jerarquía para combinar las partes que detectó la primera capa
  - **Wider** debería ser similar o levemente mejor que Base — más neuronas dan más capacidad pero a esta escala puede no hacer diferencia          
  - **Deeper** puede ser mejor o igual que Base, pero también puede tener gradientes más pequeños en las primeras capas → converge más lento
# función de activacion de la arq

 **Sigmoid y tanh:**                                                            
  - Sigmoid: para entradas muy positivas la salida se acerca a 1, para muy negativas a 0. En esas zonas la curva es casi plana → el gradiente es casi 0.                           
  - Tanh: igual pero entre -1 y 1.                                           
  - Cuando el gradiente es casi 0 en una capa, el gradiente que llega a la capa anterior es todavía más chico (se multiplica en cada capa hacia atrás). Con 2 capas ocultas esto ya empieza a ser un problema — la primera capa aprende muy lento o directamente no aprende.
  **ReLU:**                                                     
  - Para entradas positivas la salida es la entrada misma → el gradiente es exactamente 1, no se aplana.                                               
  - Para entradas negativas la salida es 0 → esa neurona no contribuye al gradiente en esa pasada, pero no lo aplana para las otras.                
  - Resultado: el gradiente fluye sin atenuarse por las capas.               

  **¿Importa esto para nuestro problema específico?**                              
Con z-score los inputs quedan centrados en 0, así que al inicio del  entrenamiento las activaciones de las capas ocultas también van a estar cerca de 0. En esa zona sigmoid y tanh tienen gradiente razonable. El problema aparece a medida que los pesos crecen y las pre-activaciones se alejan del 0 — ahí sigmoid y tanh empiezan a saturar. Con solo 2 capas ocultas el efecto no es tan dramático como con redes más profundas. Por eso es esperable que las tres funciones lleguen a resultados similares, pero ReLU debería **converger más rápido** porque el gradiente fluye mejor.
Para la capa de salida no hay decisión: tiene que ser **softmax** porque estamos haciendo clasificación multiclase y necesitamos que las 10 salidas sumen 1 y sean interpretables como probabilidades. 
# Decision Kfold
Aca lo que hicimos fue basarnos en la justificación del ejercicio 1 sobre el 80-20.
K=5 da 80% de datos para train por fold, suficientes positivos de la clase 5 en el fold de test (~54 por fold), y 5 estimados para promediar.
# Estrategia de entrenamiento: Batch, mini batch, online
HAY QUE JUSTIFICAR ESTO
# Experimento de lr vs epocas para todas las arquitecturas

Aca evaluamos las 4 arquitecturas, con distintos lr y sus funciones de activacion correspondientes para cada capa oculta.
Usamos batch-size de 32 y un maximo de epocas de 50, 5seeds y Kfold = 5. La primer corrida de ahora es con la actualizacion de pesos normal a la cual se le resta el lr x gradiente. 

A partir del analisis_lr, vimos que no hay variacion entre las arquitecturas.
El proximo experimento lo vamos a hacer en base a la arch base: adam vs momentum


