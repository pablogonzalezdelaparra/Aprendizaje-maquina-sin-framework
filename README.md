# Momento de Retroalimentación: Módulo 2 Implementación de una técnica de aprendizaje máquina sin el uso de un framework. (Portafolio Implementación)

# Datos de la entrega
Nombre: Pablo González de la Parra

Matrícula: A01745096

Fecha: 28/08/2023

# Descripción del repositorio
De acuerdo con lo establecido en la actividad, el repositorio contiene el código fuente que implemente un algoritmo de aprendizaje máquina en Python sin utilizar ningún framework o librería dedicada para su funcionamiento.

En este caso se decidió implementar el algoritmo denominado <b>árbol de decisión</b> (Decision Tree) de manera que clasifique la información de un dataset en específico.

En el archivo denominado ```decision_tree.py``` se encuentra la implementación del algoritmo en cuestión, el cual posee tanto el algoritmo como el entrenamiento y validación de este.

El algoritmo funciona con la división del dataset en 80% de entrenamiento y 20% de testing. De igual manera, el 80% de entrenamiento se divide en 20% de validación, el cual permite modificar los hiperparámetros para encontrar la solución más optima.

La solución es impresa a la terminal, la cual consiste de dos partes:
1. Accuracy score sobre el dataset de validación
2. Accuracy score sobre el dataset de testing

Ejemplo:
```
Puntuación de precisión en set de validación:  0.9375
Puntuación de precisión en set de testing:  0.85
```

# Manual de ejecución
Para poder ejecutar el algoritmo se debe de tener instalado Python 3.8 o superior, y seguir los siguienets pasos:

1. Clonar el repositorio en la computadora local.
2. Abrir una terminal en la carpeta del repositorio.
4. Instalar las librerias y dependenicas utilizando <b>pip</b> y los siguientes comandos.

```
pip install pandas

pip install numpy

pip install sklearn

pip install scikit-learn
```

3. Ejecutar el comando ```python decision_tree.py```

# Referencias
Origen del dataset utilizado:

### Drugs A, B, C, X, Y for Decision Trees

https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees


