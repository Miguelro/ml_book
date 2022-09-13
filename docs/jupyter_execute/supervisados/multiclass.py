#!/usr/bin/env python
# coding: utf-8

# # Multiclass clasificación.
# (multiclases)=
# ## Introducción.
# ```{index} One-vs-Rest, One-vs-One
# ```
# 
# No todos los modelos predictivos de clasificación, soportan una clasificación multiclase, y en este sentido algoritmos muy importantes por su uso como Perceptron, Regresión logística y Support Vector Machine fueron diseñados para hacer una clasificación binaria, es decir con tan sólo dos clases a predecir.
# 
# Debido a esta restricción que se tiene para este tipo de algoritmos, se puede diseñar una aproximación a una clasificación multiclase utilizando clasificación binaria, dividiendo la clasificación múltiple en varias clasificaciones binarias y hacer un ajuste binario sobre estas últimas. Para hacer esto último, se han ideado dos enfoques o estrategias diferentes denominados One-vs-Rest y One-vs-One. En este apartado vamos a desarrollar estas dos estrategias, y se desarrollará algún ejemplo que permita clarificar cómo se pueden usar.
# 
# ## One-Vs-Rest para clasificación múltiple.
# 
# La clasificación One-Vs-Rest (de forma abreviada OvR) también es conocida como One-vs-All (de forma abreviada OvA), se utiliza para resolver un problema de clasificación múltiple mediante varios problemas de clasificación binaria. En este sentido lo que se hace es entrenar un modelo de  clasificación binaria de cada clase sobre el resto, es decir si por ejemplo la variable de clasificación está constituida por tres clases: rojo, verde y azul; entonces se generarían tres modelo de clasificación binaria:
# 
# * Clasificación binaria de rojo frente al resto [verde, azul]
# 
# * Clasificación binaria de verde frente al resto [rojo, azul]
# 
# * Clasificación binaria de azul frente al resto [verde, rojo]
# 
# Por lo tanto en este tipo de problemas, se requiere ajustar un modelo por cada clase constituyente de la variable de clasificación, lo cual puede ocasionar una ralentización importante del proceso.
# 
# Con este enfoque lo que se requiere es que cada uno de los modelos, prediga una probabilidad de pertenencia de una observación a una determinada clase, y entonces la decisión a tomar es clasificar una observación en la clase que se haya obtenido una mayor probabilidad. Por lo tanto, este procedimiento puede ser utilizado en los algoritmos que predigan una probabilidad de pertenencia a una clase, como pueden ser la *regresión logística* o *Perceptron*. 
# 
# La clase de scikit Learn *LogisticRegression* tiene un hiparámetro denominado *multiclase* que si se le da un valor de 'ovr' realiza una clasificación multiclase de tipo One-vs-Rest. Veamos a continuación un ejemplo utilizando un conjunto de datos generados de forma artificial con la función *make_classification*. 

# In[1]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# generamos los datos con un total de tres clases (n_classes=3)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
n_classes=3, random_state=1)
# definimos el modelo para clasificación múltiple multi_class='ovr'
model = LogisticRegression(multi_class='ovr')
# definimos el porcedimiento de evaluación el modelo
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# Imprimimos la acuracidad del modelo obtenido
print('Acuracidad media: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# ### La clase OneVsRestClassifier.
# 
# Scikit learn tiene una clase propia para hacer este tipo de clasificaciones múltiples: <a href="" target="_blank">OneVsRestClassifier</a>, mediante la cual se puede utilizar esta estrategia con cualquier clasificador  de los comunmente utilizados. A continuación se muestra un ejemplo de su uso, utilizando una regresión logística como clasificador.

# In[2]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
# importamos OneVsRestClassifier
from sklearn.multiclass import OneVsRestClassifier

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
n_classes=3, random_state=1)

model = LogisticRegression()
# indicamos a la clase OneVsRestClassifier el modelo a utilizar
ovr = OneVsRestClassifier(estimator = model)
# hacemos la evaluación del modelo
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(ovr, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Mean Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# Cuando se usa el modelo OvR, se puede utilizar el mismo para hacer predicciones de forma similar a como se utiliza cualquier otro clasificador, utilizando para ello el método *predict*. 

# In[3]:


from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
n_classes=3, random_state=1)
# modelo base
model = LogisticRegression()
# 
ovr = OneVsRestClassifier(model)
# ajuste del modelo
ovr.fit(X, y)
# valores para predicción
row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223,
1.80998823, 0.58810926, -0.02542177, -0.52835426]
yhat = ovr.predict([row])
print('clase predicha: %d' % yhat[0])


# ## One-Vs-One para clasificación múltiple (OvO).
# 
# Constituye otro método heurístico para hacer clasificaciones múltiples vía clasificaciones binarias. En esta ocasión, las clasificaciones binarias se realizan utilizando todos los pares posibles de clases que se puedan hacer, y en este sentido si las clases de la variable clasificadora son rojo, azul y amarillo, entonces las clasificaciones binarias a realizar serían los pares siguientes:
# 
# * rojo vs azul
# 
# * rojo vs amarillo
# 
# * azul vs amarillo
# 
# En este sentido y teniendo esto en cuenta, el número de modelos a ajustar sería combinaciones de m tomados de dos en dos, siendo m el número de clases que tiene la variable clasificadora. Ese valor por lo tanto sería igual a m*(m-1)/2.
# 
# La decisión final que se toma con este método, sería bien la clase más votada o bien si el método utilizado produce un scores ( o probabilidad) entonces la clase elegida será la que tenga mayor suma de scores. 
# 
# En scikit learn, se puede ver que la clase SVC (support vector machine) tiene un parámetro para indicar que se quiere utilizar este tipo de clasificación. Esto se puede ver en el siguiente ejemplo.

# In[4]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
n_classes=3, random_state=1)
# definimos el modelo e indicamos que utilice una clasificación 
model = SVC(decision_function_shape='ovo')

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# evaluamos el modelo
n_scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# resumen de la acuracidad
print('Acuracidad media: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# Al igual que ocurría con la clasificación OvR, scikit learn tiene una clase genérica que permite hacer clasificaciones del tipo One vs One, la clase se denomina OneVsOneClassifier, y a continuación se muestra un ejemplo, explicativo de su uso.

# In[5]:


from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier


X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
n_classes=3, random_state=1)


model = SVC()
# definimos la estrategia OvO
ovo = OneVsOneClassifier(model)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

n_scores = cross_val_score(ovo, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

print('Acuracidad Media: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# Como es de esperar también se pueden hacer predicciones, utilizando para ello el método *predict*.

# In[6]:


from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier

X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5,
n_classes=3, random_state=1)

model = SVC()

ovo = OneVsOneClassifier(model)

ovo.fit(X, y)
# tomamos dato para hacer predicción
row = [1.89149379, -0.39847585, 1.63856893, 0.01647165, 1.51892395, -3.52651223,
1.80998823, 0.58810926, -0.02542177, -0.52835426]
yhat = ovo.predict([row])
print('Clase predicha: %d' % yhat[0])


# In[ ]:




