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
EXCLUIR todo lo q sea del ej0


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


## Arquitectura
Incluir lo que explica en el doc de decisiones sobre los experimentos, y que cuando hicimos el wider sin mas datos terminamos, se aumento la capacidad pero terminamos en overfitting.

## Funcion de activacion 
Ya tenemos en ejercicio2/Readme.md explicacion de las comparativas entre funciones de activacion.
Claude nos recomendó ReLu y después hicimos pruebas que llegaron a que era mejor
Incluir una diapo que introduzca relu

## Estrategia
Explicar: en clase dijeron que esta bueno mini batch *apunte ipad* y que por eso lo elegimos
Contar el testing que hicimos sobre el batch size *ver pdf decisiones*

# Ejercicio 3
(a) ¿Cuál es el mejor resultado que pudieron obtener con este nuevo conjunto de datos?
(b) ¿Qué técnicas utilizaron para mejorar el rendimiento con respecto al caso anterior?
(c) Además de sus propias técnicas, ¿existen otros factores que influyeron en el cambio de
rendimiento entre este ejercicio y el anterior?

Aca se hizo lo de "Pack C" de estrategias. Son lo explicado en la clase de regularizacion. HACER UN MATCHING ACA DE TEORIA Y LO HECHOO

# PRESENTACION
1) ok
2) ok
slide 3 eliminar
4) sacar ej 0 y poner un poco mas de info de la funcion de activacion
que es satura? y sacar el comentario de lo que descarta la clase
AGREGAR 2 SLIDES NUEVAS, una para cada tabla que hay en la nota "docs/lr_sweep_conclusion.md"
5) sacar la seccion  "regla del apunte"
sacar todo el comentario de pie
6) ok - obs: aprender como funciona adam y momentum
7) cambiar naming "fase" 1.3 y fase 2, por "Fase 1 exploratoria k = 1" y "fase 2 k=5 validacion".
El comentario al pie debe ser unicamente Elegimos LR = 10^-3 con Adam.
8) eliminar esta slide. Crear un note .md que explique como funciona cross entropy 
9) Sacar el comentario de practica standard. Sacar bullet de "El bias NO se penaliza..."; Modificar comentario de "Early stopping..." paraque quede unicamente "Early stopping detecta overfitting que Epsilon solo no detecta".
Slides 10 a 15 -> ELIMINAR
17) Modificar seccion "Ground truth" para que diga "Criterio de evaluación".
Agregar una diapo para incluir lo que se encontró sobre criterio algoritmico para encontrar fraude. La diapo debe armarse a partir de la informacion de la pagina 2 del documento "ejercicio1/PRESENTACION_LEER/informe_analisis_dataset.pdf; La diapo debe incluir el umbral de los OR, y las tablas de las seccion 3.2. Luego OTRA diapositiva con el cuadrito de conclusion de "El 80% del fraude se etiqueta...".
19) ELIMINAR
20) Cambiar titulo a "Pruebas sobre threshold.
21) ELIMINAR
22) eliminar columna "Caso de uso". Eliminar el comentario de "El modelo entrega scores..."
24) ok
25) ELIMINAR
26) Cambiar titulo por "Fases de experimentacion"
27) eliminar
28) eliminar
AGREGAR NUEVA DIAPO - arquitectura, con la tabla de la seccion 6 de "notas/decisiones_y_backing_teorico.md" , y una explicacion de la conclusion
29) ELIMINAR
AGREGAR NUEVA DIAPO - La tabla de seccion 3 "Optimizador" del doc "notas/decisiones_y_backing_teorico.md" y una breve explicacion de la conclusion
30) eliminar
31) ok
32) ok
33) ok
CREAR NUEVA NOTA QUE EXPLIQUE EN DETALLE A QUE SE REFIERE CON DISTRBUTION SHIFT en "docs/explicacion_distribution_shift.md"
34) ok
35) Cambiar nombre de la seccion por "Ejercicio 3 - Conclusiones"
36) Editar el 2. para que No diga "Pack C" que diga solo "Se emplea regularizacion..." ; Eliminar el comentario
37) ok
MOVER DIAPO 43 PARA QUE ESTE JUSTO DESPUES DE LA 37
38) eliminar
39) eliminar pero crear una nueva nota "docs/clase_regularizacion_cosas_implementadas.md" con toda esta data.
40) Cambiar titulo a "Resultados" ; Crear nota "docs/resultados_ejercicio3_explicacion" Que explique que es cada tecnica de regularizacion, como se implementa en el proyecto, y los reusltados en si.
41) ok
42) ok
43) MOVER COMO SE EXPLICO ANTES
44) cambiar titulo: Resultados
45) MODIFICAR DIAPO: Sacar conclusiones mas simples pq no entendemos las que hay
46) eliminar
47) eliminar
48) Cambiar "parsimonia" por "Hay que elegir por simplicidad, tener criterio de complejidad en tiempo y espacio, no solo por pequeñas diferencias de resultados pp"
49) Sacar el "Preguntas" solo Gracias.

