# Stage 1
Analisis de los resultados obtenidos en la etapa de "Pre_LR_Batch_Opt", resultados mas finos explicados en [[Notas/ejercicio 2/Segunda tanda de experimentos/Pre_LR_Batch_Opt/analisis|analisis]]

En nuestros primeros experimentos del ejercicio 2, fuimos modificando los hiperparametros 1 a la vez. Comenzamos definiendo 4 arquitecturas y variando el lr -> vemos que ENTRE las arquitecturas no hay variacion en cuanto a los lr. Ambas curvas de convergencia (train loss y val loss) muestran el mismo comportamiento.
![convergence_train](convergence_train.png)
Continuando con el experimento, fijamos el batch-size en 32, pero fuimos variando el optimizer y lr, empezamos a ver como a medida que fuimos agrandando el lr, iban empeorando los resultados para cada optimizer. Todas las corridas fueron con 5 seeds y 5 folds.
Los lr utilizados fueron 1e-4, 5e-4, 1e-3, 5e-3, 1e-2.
![convergence_lr1e-4](convergence_lr1e-4.png)
![convergence_lr1e-3](convergence_lr1e-3.png)
![convergence_lr5e-3](convergence_lr5e-3.png)
Del optimizer sweep (batch=32 fijo) sacamos una observación: Adam se desestabiliza con LR ≥ 5e-3, mientras que SGD/Momentum siguen tolerando LR  alto. Esa observación + la teoría de la cátedra (la regla LR↔batch lineal)  nos generó la sospecha de que el techo de Adam podría no ser intrínseco al  optimizador, sino al producto LR×batch. Para confirmarlo no nos alcanzaba con  el optimizer sweep — necesitábamos variar también el batch.
![[Pasted image 20260510171351.png]]

El techo de Adam (lr < 5e-3 con batch=32) no es un límite del optimizador sino del producto LR×batch. El pre-experimento confirma que Adam@5e-3 con batch=256 es estable (val_loss=0.191) mientras que con batch=16 no convergio (val_loss=0.550). La regla de escalado lineal LR↔batch de la clase de optimizadores predice exactamente esto. Para val_loss, la mejor combinación encontrada es  Adam con lr=5e-4×batch=16 o lr=1e-3×batch=64 (ambas dan val_loss ≈ 0.170). SGD y Momentum son mucho  menos sensibles al batch — su val_loss varía menos entre batch=16 y batch=256 — pero parten de valores de val_loss más altos que Adam en la zona óptima.

![[lr_batch_relationship.png]]

  En el optimizer sweep con batch=32 fijo, Adam fue el único optimizador que mostró comportamiento drásticamente distinto según el LR: estable y rápido con lr≤1e-3, inestable y oscilante con lr≥5e-3. SGD y Momentum toleraron LR altos sin colapsar — sus curvas de convergencia degradaban gradualmente, sin el quiebre abrupto que mostró Adam.
  Ese comportamiento exagerado de Adam no es un defecto — es consecuencia de su mecanismo de adaptación por parámetro. Adam ajusta internamente el tamaño de paso, lo que lo hace muy sensible al producto LR×batch. Si ese producto está fuera de rango, los pasos efectivos son demasiado grandes y el modelo no converge.
  El gráfico muestra los dos efectos en simultáneo. Para Adam con lr=5e-3 (inestable a batch=32), aumentar el batch de 16 a 256 baja la CE de 0.55 a 0.19 y sube la val_acc de 0.934 a 0.954 — las dos métricas mejoran juntas a medida que la relación LR×batch se acerca al rango adecuado. Para lr=5e-4 y lr=1e-3 (ya estables), las curvas son planas: el batch no mueve la aguja porque el LR ya era el correcto para esos valores. SGD y Momentum no requieren este análisis porque su sensibilidad al batch es mucho menor — para ellos, fijar batch=32 no introduce el mismo sesgo en la comparación.
![[ejercicio2_experimentacion/analisis/cross_v1/stage2/stage2_val_acc_vs_lr_per_opt.png]]

# Stage 2
![[grid_3d_static.png]]
sexyyy

> [!warning] Que significan las métricas! Importante!!
> Importantisimo tener visto [[Notas/ejercicio 2/Segunda tanda de experimentos/Cross_LR_Opt_Arch/analisis#Qué significa cada columna|explicación de métricas usadas en los experimentos]]

