Comentarios generales:
- El scope de lo que intentaron abarcar estuvo bien, el problema principal es que no hubo justificación desde la parte teórica ni desde la perspectiva de resultados  (corrección que ya se dio en otro TP).
- Traten de equilibrar la distribución en la presentación o, ante preguntas, que conteste quien tiene menos asignado (corrección que ya se dio en otro TP).
- Se percibió que gran parte de las cosas escritas en la PPT no las entendían del todo y/o no recordaban cómo se habían implementado. 
- El uso de modelos para soporte de código, armado de PPTs, discusión de temas, etc. está perfecto pero, en última instancia, uds. son responsables de las decisiones que toman y lo que deciden incluir en su TP. Los términos y experimentos incluidos deberían entenderlos.
- Sobre esto, una aclaración, porque dar una presentación abierta es, en cierto nivel, algo subjetivo: les puede ocurrir que quieran explorar herramientas que no están seguros del impacto que deberían tener o no tienen una intuición al respecto. Esto está OK, ya que recién estan abordando la temática. Es válido experimentar con alguna herramienta sin estar del todo seguros de cómo funciona por detrás (nos pueden preguntar) pero de todas formas el por qué lo hacen debe estar correctamente justificado.

Ejercicio 1: 

- Mencionan la regla del enunciado, pero ¿entienden por qué está la regla? ¿Qué pasa si yo entreno con “flagged_fraud” como columna para tratar de predecir la probabilidad?
- No explican si realizan pre-procesamiento de los datos (ej: normalización)
- Bien que analizan las variables y cómo se relacionan con la salida de interés: la probabilidad de fraude.
- ¿A qué le llaman “S marginal”?
- ¿A qué le llaman “aplica una sola S en la direccion w”?
    
- Dicen que el lineal queda por fuera de [0,1] en un 7% de los casos, ¿de dónde sale este dato si todavía no muestran ningún tipo de entrenamiento? ¿esto ocurre en la mejor instancia de entrenamiento o en un promedio?
    

- No explican por qué usan K-Fold=5. La sección de aprendizaje no requería separación en train y test, si bien podían hacerlo ¿Por qué deciden hacerlo así? ¿Por qué van con K-Fold=5? Nada de esto está definido.
    
- ¿Por que F1 les parece la mejor métrica para decidir el umbral óptimo? Dicen que es “decisión más de negocio” ¿por qué?
    
- IMPORTANTE: Los hiperparametros deben estar definidos y, los que les parezcan importantes, analizados y justificados (ejemplo: tasa de aprendizaje, épocas, entre otros).
    

  

Ejercicio 2:

- Mencionan el uso de K-Fold=1. Conceptualmente, implica no tener cross-validation pero al preguntar no lo tenían muy claro.
    
- Bien que esquematizan como quieren llevar adelante la experimentación (slide 11)
    
- Sweep de arquitectura: falta indicar hiperparámetros, mostrar resultados y fundamentar las decisiones. Todas las conclusiones realizadas no derivan de los resultados mostrados en la presentación (convergencia más lenta, empate técnico, subajuste).
    

- Algo para remarcar que sí estuvo bien es la conclusión de elección de arquitectura, si es que todo lo anterior se sostiene por resultados.
    

- En relación a la elección de optimizador y función de activación, aplican los mismos comentarios que arriba.
    
- Prueban otras funciones de activación y distintas formas de inicialización, lo cual esta bueno, pero no hay resultados ni comentarios al respecto.
    
- Introducen el término de “overshoot” ¿A qué le llaman overshoot? Mismo comentario que antes, no muestran ningún resultado del cual se desprenden los comentarios que introducen en la slide 19.
    
- Bien que inspeccionan el conjunto de datos y la distribución de las clases en el mismo.
    
- Mencionan divergencia catastrófica cuando lo que están mostrando puede ser overfitting. No es lo mismo. Tampoco se puede entender con precisión en qué escenario se encuentran (slide 20).
    

  

Ejercicio 3:

- Excelente que plantean la hipótesis de la cual parten, consecuencia del ejercicio 2, para plantear un plan secuencial de pasos que intenta resolver el gap que les falta.
    
- Introducen de una varias técnicas de regularización, pero no explican bien por qué deciden ir por regularización y no, por ejemplo, cambios de arquitectura o mayor ajuste de hiperparámetros.
    
- Bien las reflexiones que hacen sobre por qué les falta 1.12% para llegar al 98% esperado. De todas formas, si bien las reflexiones como tal están OK, no están los resultados para respaldarlas