- TP2  MNA (Kuramto-Sivashinsky)

En este trabajo expondremos la implementación de la solución aproximada a las ecuaciones de Kuramoto-Sivashinsky.

- Implementación

Se utilizó matlab para la totalidad del proyecto y también se utilizó google sheets en conjunto para el análisis de los datos.

- Funcionalidad

Por una parte, está el archivo SequentialMovementRun.m, en donde hay ciertas variables que se pueden modificar para cambiar la funcionalidad del código:

N --> los puntos a discretizar sobre la función
delta_t --> Paso de integración
q --> orden de los métodos afines
perturbación --> 0 si no se quiere perturbar al sistema y 1 si se quiere perturbarlo
frames --> la cantidad de frames que se quiere para la graficación de la perturbación
IntStart, IntFin --> el intervalo de la función a simular
method --> método a utilizar para la implementación del sistema

Siendo los valores de method los siguientes:
1 - Lie Trotter
2 - Strang
3 - Ruth
4 - Neri
5 - Affine Symmetric
6 - Affine Asymmetric


Por otra parte, se encuentra el archivo ParallelRun que se utiliza para ejecutar la implementación de métodos afines paralelos y que mantiene los parámetros anteriores con diferencia de method siendo sus valores:

1 - Affine Symmetric Parallel
2 - Affine Asymmetric Parallel


Por último, se utilizó el archivo ComparisonMain.m que se utilizó para generar los archivos de post procesamiento de los resultados que utiliza los mismos parámtros que el primer archivo de ejecución.