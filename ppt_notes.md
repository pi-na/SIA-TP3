# Notas generales sobre MLP
Hay que saber que experimentos se realizaron y decisiones se tomaron sobre los hiperparametros:
- learning rate
- M cantidad de capas
- Cantidad de neuronas por capa
- Candiad de epochs
- Estrategia: online-batch-mini batch
- Funciones de activación de las neuronas
- Inicializacion de los pesos
- Optimizador elegido
- Calculo de la convergencia - Epsilon - Tamaño de epsilon
- Como se manejó el BIAS

## Clase vs implementación
Narias ya validó la clase de MLP
Falta validar con la clase de optimizadores

# Ejercicio 0
EXCLUIR

# Ejercicio 1
"Aclaración: Es de suma importancia que exploren el conjunto de datos antes de trabajar con él: ¿qué dice la documentación de cada columna? ¿en qué rangos se mueven las distintas columnas? ¿de qué está compuesto mi dataset? ¿los datos están limpios?"
> Con esta directiva pienso que realmente vale la pena incluir el algoritmo de baseline para el ej1, resumidamente

# Ejercicio 2
- ¿cómo evalúo el desempeño de mi sistema?
- (b) ¿qué variantes realizo para encontrar la solución?
- variantes de tasa de aprendizaje
- variantes de arquitectura
- variantes de mecanismos de optimización
Que otra exploración se hizo?
Cuales fueron todas las decisiones de diseño, arquitectura, hiperparametros, y como fue la experimentación?

## Eleccion de hiperparametros
Se hizo un sweep que probó todo. Esta en [[ejercicio2/README.md]].
Para la arquitectura:
**Notación `[784, 100, 50, 10]`**: números entre corchetes indican la topología:
- 784: capa de entrada (28×28 píxeles de MNIST)
- 100, 50: capas ocultas con esas neuronas
- 10: capa de salida (10 clases, dígitos 0-9)

**"Empate técnico"** → dos arquitecturas rinden igual (96.74%). Se elige por **parsimonia**: menos neuronas = 35% más rápido sin perder accuracy. Regla de Occam: modelo más simple gana si el rendimiento es equivalente.

## Funcion de activacion 
Ya tenemos en ejercicio2/Readme.md explicacion de las comparativas entre funciones de activacion.

## Initializer PENDIENTE
También acá parece que claude eligio funciones initializer conocidas, de donde las sacó?
Pasarle a claude el paper que nos mandaron, y el transcript de la clase optimizador para ver si coincide con algo de ahi y si no justificar...

# Ejercicio 3
(a) ¿Cuál es el mejor resultado que pudieron obtener con este nuevo conjunto de datos?
(b) ¿Qué técnicas utilizaron para mejorar el rendimiento con respecto al caso anterior?
(c) Además de sus propias técnicas, ¿existen otros factores que influyeron en el cambio de
rendimiento entre este ejercicio y el anterior?

