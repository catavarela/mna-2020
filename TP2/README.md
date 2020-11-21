## TP MNA 2

En este repositorio se encuentra la implementación del TP2 de Métodos Numéricos Avanzados.

## Implementación

El modelo fue implementado en Matlab en su totalidad, y se utilizó python 3 para la generación de visualizaciones de los resultados.

## Ejecución

El programa se ejecuta desde el archivo `Main.m` o bien `ComparisonMain.m` (Desde ahora, los `Mains`). El primero corre el programa y genera la visualización de la solución a la ecuación KS, el segundo en cambio genera un archivo CSV con los errores entre distintas corridas.

Desde los `Mains` puede definirse la cantidad de muestras tomadas (`N`), el paso (`h`), y el integrador utilizado. Para esto, debe utilizarse alguno de los integradores definidos: `LieTrotter`, `Strang`, `Ruth`, `Neri`, `AfinSimetrico`, `AfinAsimetrico`, `AfinSimetricoParalelo`, `AfinAsimetricoParalelo`. Y deben llamarse utilizando la siguiente interfaz:

```
Solver(h,x,k,q,p,@integrador);
```

Donde los parámetros son:
- `h`: Paso de integración
- `x`: Valores de x
- `k`: Valores de k
- `q`: Orden del integrador. Los SSMs comunes no utilizan este parametro, pero debe pasarse igual para respetar la interfaz
- `p`: 1 si se quiere perturtbación aleatoria en la condicion inicial, 0 si no.
- `integrador`: Alguno de los integradores listados previamente.

De esta manera, un ejemplo de ejecución es:

```
% x y k definidos previamente en el
Solver(0.002,x,k,4,0,@AfinAsimetrico);
```

## Visualización de resultados

Para la visualización de los errores se utilizó el script de python definido en el directorio `analysis`. Para ejecutarlo debe ejecutarse:

```
pip install -r requirements.txt
python errors.py
```

Las últimas 2 líneas del script definen el análisis que se quiere realizar y el directorio de dónde obtener los archivos para comparar. Por defecto se calcula el error global de los SSMs comúnes a partir de lo calculado en el directorio `results/data/error_global/comunes`.