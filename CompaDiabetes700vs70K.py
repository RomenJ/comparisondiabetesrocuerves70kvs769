import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats

datos = pd.read_csv('diabetes.csv')

X1 = datos[['Glucose','DiabetesPedigreeFunction','Pregnancies', 'Insulin','SkinThickness']]

Y1 = datos['Outcome']
print('Tamaño Muestral Grupo 1: N=768')
print(len(datos)) 
# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.3, random_state=42)

# Imputa los valores faltantes en Y_train
imputer = SimpleImputer(strategy='most_frequent')
Y_train1 = imputer.fit_transform(Y_train1.values.reshape(-1, 1))
Y_train1 = Y_train1.flatten()

# Imputa los valores faltantes en Y_test
Y_test1 = imputer.transform(Y_test1.values.reshape(-1, 1))
Y_test1 = Y_test1.flatten()

# Imputa los valores faltantes en X_train
imputer1 = SimpleImputer(strategy='mean')
X_train1 = imputer1.fit_transform(X_train1)

# Imputa los valores faltantes en X_test
X_test1 = imputer1.transform(X_test1)

# Crear y entrenar el modelo de regresión logística
model1 = LogisticRegression()
model1.fit(X_train1, Y_train1)

# Realizar predicciones en el conjunto de prueba
Y_pred1 = model1.predict(X_test1)

# Calcular el grado de error general en los datos de prueba
error_general1 = 1 - metrics.accuracy_score(Y_test1, Y_pred1)

# Calcular el porcentaje de acierto en los datos de prueba
acierto1 = metrics.accuracy_score(Y_test1, Y_pred1)

# Calcular las métricas de matriz de confusión
confusion_matrix1 = metrics.confusion_matrix(Y_test1, Y_pred1)
verdaderos_positivos1 = confusion_matrix1[1, 1]
falsos_negativos1 = confusion_matrix1[1, 0]
falsos_positivos1 = confusion_matrix1[0, 1]
verdaderos_negativos1 = confusion_matrix1[0, 0]

# Calcular sensibilidad y especificidad
sensibilidad1 = verdaderos_positivos1 / (verdaderos_positivos1 + falsos_negativos1)
especificidad1 = verdaderos_negativos1 / (verdaderos_negativos1 + falsos_positivos1)

# Imprimir resultados
print("Grado de error general en los datos de prueba N=768:", error_general1)
print("Porcentaje de acierto en los datos de prueba N=768:", acierto1)
print("Porcentaje de error para Verdaderos Positivos N=768:", falsos_negativos1 / (verdaderos_positivos1 + falsos_negativos1))
print("Porcentaje de acierto para Verdaderos Positivos N=768:", verdaderos_positivos1 / (verdaderos_positivos1 + falsos_negativos1))
print("Porcentaje de error para Falsos Negativos N=768:", falsos_positivos1 / (verdaderos_negativos1 + falsos_positivos1))
print("Porcentaje de acierto para Falsos Negativos N=768:", verdaderos_negativos1 / (verdaderos_negativos1 + falsos_positivos1))
print("Porcentaje de error para Falsos Positivos N=768:", falsos_positivos1 / (verdaderos_negativos1 + falsos_positivos1))
print("Porcentaje de acierto para Falsos Positivos N=768:", verdaderos_negativos1 / (verdaderos_negativos1 + falsos_positivos1))
print("Porcentaje de error para Verdaderos Negativos N=768:", falsos_negativos1 / (verdaderos_positivos1 + falsos_negativos1))
print("Porcentaje de acierto para Verdaderos Negativos N=768:", verdaderos_positivos1 / (verdaderos_positivos1 + falsos_negativos1))
print("Sensibilidad N=768:", sensibilidad1)
print("Especificidad N=768:", especificidad1)


# Obtener las probabilidades predichas para la clase positiva
Y_prob1 = model1.predict_proba(X_test1)[:, 1]

# Calcular la curva ROC
fpr1, tpr1, thresholds1 = roc_curve(Y_test1, Y_prob1)

# Calcular el área bajo la curva ROC (AUC)
roc_auc1 = auc(fpr1, tpr1)

# Graficar la curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr1, tpr1, color='darkorange', lw=2, label=f'AUC N768 = {roc_auc1:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad) N=768')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad) N=768')
plt.title('Curva ROC N=768')
plt.legend(loc='lower right')
plt.show()


datos = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
print('Tamaño Muestral Grupo 1: N= 70692')
print(len (datos))

X = datos[['HighChol','BMI', 'Fruits','Age','Income','HighBP', 'GenHlth','Sex','Education']]
#X = datos[['Glucose','BloodPressure']]

Y = datos['Diabetes_binary']

# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Imputa los valores faltantes en Y_train
imputer = SimpleImputer(strategy='most_frequent')
Y_train = imputer.fit_transform(Y_train.values.reshape(-1, 1))
Y_train = Y_train.flatten()

# Imputa los valores faltantes en Y_test
Y_test = imputer.transform(Y_test.values.reshape(-1, 1))
Y_test = Y_test.flatten()

# Imputa los valores faltantes en X_train
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Imputa los valores faltantes en X_test
X_test = imputer.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, Y_train)

# Realizar predicciones en el conjunto de prueba
Y_pred = model.predict(X_test)

# Calcular el grado de error general en los datos de prueba
error_general = 1 - metrics.accuracy_score(Y_test, Y_pred)

# Calcular el porcentaje de acierto en los datos de prueba
acierto = metrics.accuracy_score(Y_test, Y_pred)

# Calcular las métricas de matriz de confusión
confusion_matrix = metrics.confusion_matrix(Y_test, Y_pred)
verdaderos_positivos = confusion_matrix[1, 1]
falsos_negativos = confusion_matrix[1, 0]
falsos_positivos = confusion_matrix[0, 1]
verdaderos_negativos = confusion_matrix[0, 0]

# Calcular sensibilidad y especificidad
sensibilidad = verdaderos_positivos / (verdaderos_positivos + falsos_negativos)
especificidad = verdaderos_negativos / (verdaderos_negativos + falsos_positivos)

# Imprimir resultados
print("Grado de error general en los datos de prueba Nº de casos: 70692:", error_general)
print("Porcentaje de acierto en los datos de prueba Nº de casos: 70692:", acierto)
print("Porcentaje de error para Verdaderos Positivos Nº de casos: 70692:", falsos_negativos / (verdaderos_positivos + falsos_negativos))
print("Porcentaje de acierto para Verdaderos Positivos Nº de casos: 70692:", verdaderos_positivos / (verdaderos_positivos + falsos_negativos))
print("Porcentaje de error para Falsos Negativos Nº de casos: 70692:", falsos_positivos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de acierto para Falsos Negativos Nº de casos: 70692:", verdaderos_negativos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de error para Falsos Positivos Nº de casos: 70692:", falsos_positivos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de acierto para Falsos Positivos Nº de casos: 70692:", verdaderos_negativos / (verdaderos_negativos + falsos_positivos))
print("Porcentaje de error para Verdaderos Negativos Nº de casos: 70692:", falsos_negativos / (verdaderos_positivos + falsos_negativos))
print("Porcentaje de acierto para Verdaderos Negativos Nº de casos: 70692:", verdaderos_positivos / (verdaderos_positivos + falsos_negativos))
print("Sensibilidad:", sensibilidad)
print("Especificidad:", especificidad)


# Obtener las probabilidades predichas para la clase positiva
Y_prob = model.predict_proba(X_test)[:, 1]

# Calcular la curva ROC
fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)

# Calcular el área bajo la curva ROC (AUC)
roc_auc = auc(fpr, tpr)

# Graficar la curva ROC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC 70692 = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('Tasa de Falsos Positivos (1 - Especificidad)Nº de casos: 70692')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)Nº de casos: 70692')
plt.title('Curva ROC Nº de casos: 70692')
plt.legend(loc='lower right')
plt.show()


