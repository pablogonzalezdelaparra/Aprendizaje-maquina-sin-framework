############## SECCIÓN DE PREPARACIÓN DE DATASET ##############
# Importar librerias
import numpy as np
import pandas as pd

# Cargar el dataset
# Origen: https://www.kaggle.com/datasets/pablomgomez21/drugs-a-b-c-x-y-for-decision-trees
data = pd.read_csv("./dataset/drug200.csv")
data.head()

# Cambiar los valores de las columnas categoricas a numericas.
# Esto se hace para que el algoritmo pueda trabajar con los datos categoricos 
# y participen en la prediccion de clasificacion.
data['Sex'] = data['Sex'].map({'F': 0, 'M': 1})
data['BP'] = data['BP'].map({'LOW': 0, 'NORMAL': 1, 'HIGH': 2})
data['Cholesterol'] = data['Cholesterol'].map({'NORMAL': 0, 'HIGH': 1})

############## SECCIÓN DE ESTRUCTURA DEL MODELO ##############
class Node():
    def __init__(self, best_characteristic=None, best_division_criteria=None, 
                 left_child=None, right_child=None, value=None):
        """ Clase para representar un nodo de un arbol de decision. El nodo
        puede ser un nodo de decision o un nodo hoja. Si el nodo es un nodo de
        decision, tiene un valor de caracteristica y un criterio de division.
        Si el nodo es un nodo hoja, tiene un valor de prediccion. """
        
        # Para nodos de decision
        self.best_characteristic = best_characteristic
        self.best_division_criteria = best_division_criteria
        self.left = left_child
        self.right = right_child
        
        # Para nodos hoja
        self.value = value

def split_data(data):
    """ Funcion para dividir los datos en dos grupos. El primer grupo contiene
    los datos de entrada y el segundo grupo contiene los datos de salida.
    Los datos de salida son los valores de la columna objetivo. """

    # Se obtienen las dimensiones de los datos de entrada
    x_rows_size, x_columns_size = np.shape(data[:, :-1])

    # Se obtienen los valores de la columna objetivo (Y)
    y_values= data[:, -1]

    return x_rows_size, x_columns_size, y_values

def get_mode(y_values):
    """ Funcion para obtener la moda de los valores de un subset de datos.
    Se utiliza la moda para predecir la clase de un conjunto de datos. """

    # Obtener los valores unicos y sus conteos
    unique_values, value_counts = np.unique(y_values, return_counts=True)
    
    # Encontrar el indice del valor con mayor conteo
    mode_index = np.argmax(value_counts)

    # Encontrar la moda
    mode = unique_values[mode_index]
    
    return mode

def build_tree(data, max_depth, current_depth=0):
    """ Funcion recursiva para construir el arbol de decision.
    El arbol de decision se construye usando el algoritmo de encontrar
    la mejor división al recorrer cada caracteristica y cada criterio. """

    # Se obtienen las dimensiones de los datos de entrada y salida
    x_rows_size, x_columns_size, y_values = split_data(data)

    # Se realiza la division de los datos .Si el numero de muestras es mayor a 
    # 1 y la profundidad actual es menor a la maxima profundidad
    if x_rows_size>=2 and current_depth<=max_depth:
        # Encuentra la mejor division (y criterio) para maximizar la ganancia 
        # de informacion
        best_column, best_division_criteria, left, right, \
            information_gain = get_best_parameters(data, x_columns_size)

        # Revisar si la ganancia de informacion es positiva
        # Esto es importante porque si la ganancia de informacion es negativa, 
        # no se puede dividir mas
        if information_gain>0:
            # Obtenga los datos de los nodos izquierdo y derecho
            left_child = build_tree(left, max_depth, current_depth=+1)
            right_child = build_tree(right, max_depth, current_depth=+1)
            # Devuelve un nodo de decision
            return Node(best_column, best_division_criteria,
                        left_child, right_child)

    mode = get_mode(y_values)
    leaf_value = mode

    # Devuelve un nodo hoja
    return Node(value=leaf_value)

def get_division_criteria(data, column, criteria):
    """ Función para obtener los datos de los nodos izquierdo y derecho.
    Los nodos izquierdo y derecho se obtienen usando un criterio de 
    division. El criterio de division es un valor unico de una columna."""

    left_data = []
    right_data = []

    # Se recorren los posibles criterios (valores unicos)
    for row in data:
        value = row[column]
        if value <= criteria:
            # Se obtienen los datos de la izquierda
            left_data.append(row)
        else:
            # Se obtienen los datos de la derecha
            right_data.append(row)

    return np.array(left_data), np.array(right_data)

def get_information_gain(left_data, right_data):
    """ Función para calcular la ganancia de información.
    La ganancia de información se calcula usando el indice de Gini. """

    # Se obtienen los valores de la columna objetivo para cada hijo
    left_y_values= left_data
    right_y_values = right_data

    # Se obtienen los valores unicos de la columna objetivo para cada hijo
    total_y_values = np.concatenate((left_y_values, right_y_values))

    # Se calcula el peso de cada hijo
    left_weight = len(left_y_values) / len(total_y_values)
    right_weight = len(right_y_values) / len(total_y_values)

    # Se calcula la ganancia de informacion usando el indice de Gini
    total_gini_index = get_gini_index(total_y_values)
    current_gini_index = (left_weight * get_gini_index(left_y_values)) + \
        (right_weight * get_gini_index(right_y_values))
    information_gain = total_gini_index - current_gini_index

    return information_gain

def get_best_parameters(data, x_columns_size):
    """ Función para calcular la mejor division de los datos de entrada.
    La mejor division se calcula usando la ganancia de informacion. """

    # Se inicializan las variables
    max_information_gain = -float("inf")
    
    # Se recorre cada columna para encontrar la mejor division
    for column in range(x_columns_size):

        # Se obtienen los valores de una columna
        selected_column_values = data[:, column]

        # Se obtienen los valores unicos de una columna
        possible_criterias = np.unique(selected_column_values)

        # Se recorren los posibles criterios (valores unicos)
        for criteria in possible_criterias:
            ## Se obtienen los datos de los nodos izquierdo y derecho
            left_data, right_data = get_division_criteria(
                data, column, criteria)

            # Es importante revisar que los hijos no esten vacios, ya que si 
            # estan vacios no se puede calcular la ganancia de informacion
            if len(left_data)>0 and len(right_data)>0:

                # Se calcula la ganancia de informacion
                information_gain = get_information_gain(left_data[:,-1], 
                                                        right_data[:, -1])

                # Si la ganancia de informacion es mayor a la ganancia de 
                # informacion maxima, se actualizan los valores
                if information_gain>max_information_gain:
                    best_column = column
                    best_division_criteria = criteria
                    left = left_data
                    right = right_data

                    # Se actualiza la ganancia de informacion maxima
                    max_information_gain = information_gain

    # Se retornan las caracteristicas de la mejor division
    return best_column, best_division_criteria, left, right, \
        max_information_gain

def get_gini_index(y_values):
    """ Función para calcular el indice de Gini.
    El indice de Gini se calcula usando la entropia. """

    # Se inicializa el indice de Gini
    gini_index = 0

    # Se obtienen los valores unicos de la columna objetivo
    y_unique_values = np.unique(y_values)

    # Se recorren los valores unicos de la columna objetivo
    for y_value in y_unique_values:
        # Se calcula el indice de Gini para cada valor unico
        gini_index = 1 - np.log2((len(y_values[y_values == y_value]) /\
                                   len(y_values))**2)

    return gini_index

############## SECCIÓN DE EJECUCIÓN DE MODELO ##############
def train_model(X, Y, max_depth=3):
    """ Función para entrenar el modelo.
    El modelo se entrena usando el algoritmo de 
    construir el arbol de decision. """
    
    # Se concatenan los datos de entrada y salida
    dataset = np.column_stack((X, Y))

    # Se construye el arbol de decision
    tree = build_tree(dataset, max_depth=max_depth)

    return tree

def predict_value(tree, X):
    """ Función para predecir la clase de los datos de entrada.
    La clase de los datos de entrada se predice usando el arbol 
    de decision. """
        
    # Se inicializa la lista de predicciones
    predictions = []

    # Se recorren los datos de entrada
    for row in X:
        # Se inicializa el nodo raiz
        temp_tree = tree
        while temp_tree.value is None:
            # Se obtiene el valor de la caracteristica
            characteristic = row[temp_tree.best_characteristic]
            # Se revisa si el valor de la caracteristica es menor o igual 
            # al criterio de division
            if characteristic <= temp_tree.best_division_criteria:
                temp_tree = temp_tree.left
            else:
                temp_tree = temp_tree.right
        # Se agrega la prediccion a la lista de predicciones
        predictions.append(temp_tree.value)

    return predictions
    
############## SECCIÓN DE ENTRENAMIENTO Y PRUEBA DE MODELO ##############
# ----- PREPARACIÓN DE DATOS -----
# Importar librerias
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Caracteristicas (X)
characteristic_rows = data.iloc[:, :-1].values
# Resultados esperados (Y)
result_rows = data.iloc[:, -1].values.reshape(-1,1)

# ----- DIVISIÓN INICIAL EN ENTRENAMIENTO Y PRUEBA -----
# Dividir el dataset en 80% entrenamiento y 20% prueba
X_train_full, X_test, Y_train_full, Y_test = train_test_split(
    characteristic_rows, result_rows, test_size=0.2, random_state=36)

# ----- DIVISIÓN DEL 80% DE ENTRENAMIENTO EN ENTRENAMIENTO Y VALIDACIÓN -----
# Dividir el 80% de entrenamiento en 80% entrenamiento y 20% validación
X_train, X_val, Y_train, Y_val = train_test_split(
    X_train_full, Y_train_full, test_size=0.2, random_state=42)

# ----- ENTRENAMIENTO DEL MODELO -----
# Entrenar el arbol de decision usando el 80% de entrenamiento
tree = train_model(X_train, Y_train, max_depth=3)

# ----- PREDICCIÓN DEL MODELO EN LOS DATOS DE VALIDACIÓN -----
# Predecir los resultados de valores de validación
Y_val_pred = predict_value(tree, X_val)
print("Puntuación de precisión en set de validación: ", 
      accuracy_score(Y_val, Y_val_pred))

# ----- PREDICCIÓN DEL MODELO EN LOS DATOS DE PRUEBA -----
# Predecir los resultados de valores de prueba
Y_test_pred = predict_value(tree, X_test)
print("Puntuación de precisión en set de testing: ", 
      accuracy_score(Y_test, Y_test_pred))