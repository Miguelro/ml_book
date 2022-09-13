#!/usr/bin/env python
# coding: utf-8

# # Validación de modelos.
# 
# ```{index} validación modelos
# ```
# (validacionModelos)=
# ## Introducción.
# 
# Una vez se tenga implementado un modelo en scikit learn, con los hiperparámetros correspondientes, es necesario proceder a ver y estudiar qué calidad tiene ese modelo, incluso si se quieren comparar resultados obtenidos mediante cambios en los hiperparámetros del modelo, habría que comparar los resultados obtenidos y ver cual es el modelo que mejor se ajusta a los datos. Para hacer todo esto scikit learn nos ofrece una serie de herramientas muy interesantes para facilitarnos estas tareas.
# 
# Otro de los aspectos importantes a tener en cuenta cuando queremos desarrollar un modelo de machine learning, es cómo probar nuestro modelo, es decir comprobar que el mismo también se ajusta bien a unos datos de prueba que no han intervenido para el entrenamiento del modelo.En este sentido hay que tener en cuenta que ajustar los parámetros de una función de predicción y además probarla con los mismos datos es un gran error, ya que este tipo de modelo tendría una validación de los datos casi perfecta, pero no nos aseguraríamos que el modelo es lo suficiente general como para poder hacer la predicción de otro tipo de datos. Un modelo que no se generaliza de forma adecuada a otros datos, diremos que está **sobreajustado** y no serviría para conseguir que propósito que nosotros queremos.
# 
# Para evitar lo anterior lo que normalmente se hace es dividir el conjunto de datos en dos grupos, uno servirá para entrenar el modelo ( train en inglés) y otros que normalmente de denomina test y que servirá para testear si el modelo se puede generalizar o no a cualquier conjunto de datos, es decir sería el **conjunto de prueba** para comprobar la fiabilidad del modelo.
# 
# ```{index} train_test_split
# ```
# 
# Este método de dividir todos los datos en dos subconjunto (train y test), se puede hacer fácilmente en scikit learn por medio de la función auxiliar <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html" target="_blank">train_test_split</a>. Veamos un ejemplo indicativo de cómo poder conseguir esto. 

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm

X, y = datasets.load_iris(return_X_y=True)
X.shape, y.shape


# Vamos a quedarnos con un 40 por ciento de los datos para poder evaluar el modelo

# In[2]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=0)

X_train.shape, y_train.shape

X_test.shape, y_test.shape


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print("Acuracidad datos entrenamiento: ", clf.score(X_train, y_train))
print("Acuracidad datos de test (prueba): ", clf.score(X_test, y_test))


# No obstante y para asegurar la generalización del modelo, otro procedimiento que se sigue en ocasiones para conseguir esto es  dividir el conjunto de datos en tres grupos de observaciones (no en dos como se ha comentado antes): 
# 
# * Grupo train ( de entrenamiento)
# 
# * Grupo de validación (el nuevo grupo)
# 
# * Grupo de test (el de chequeo final de los datos)
# 
# Con estos tres grupos de datos el procedimiento que se sigue es el siguiente: se entrena el modelo con el grupo train, después de realiza la evaluación con el grupo de evaluación y por último y cuando el experimento parece tener éxito se hace la evaluación final con el grupo de test.
# 
# Como puede entenderse, el procedimiento descrito anteriormente es muy difícil de implementar cuando el conjunto de datos no es lo suficientemente amplio como para poder elegir de forma aleatoria y representativa los tres conjuntos de datos, de tal forma que en estos casos los resultados pueden depender de una elección aleatoria particular para el par de conjuntos (entrenamiento, validación). 
# 
# ```{index} validación cruzada, cross-validation, k-fold
# ```
# 
# ## Validación cruzada
# Debido al inconveniente anterior, existe también otro procedimiento para validar de una forma más adecuada un modelo. Se trata del procedimiento denominado *validación cruzada* (de forma abreviada CV). Con este procedimiento se elimina el grupo de validación y se sigue manteniendo el conjunto de test. El procedimiento, denominado k-fold CV, que se sigue con este método es el se describe a continuación.
# 
# El conjunto de datos se divide en k subconjuntos disjuntos, y entonces **para cada uno de estos grupos o folds**, se hace lo siguiente:
# 
# * El  modelo se entrena usando k-1 grupos.
# 
# * El modelo resultantes se valida o testea con el grupo que no se ha utilizado en el entrenamiento del modelo.
# 
# De esta manera se obtienen k scores (uno por cada modelo obtenido con este procedimiento), y entonces la validación del modelo se realiza con la media de los k escores obtenidos, medida a la que se suele añadir la desviación típica de los esos datos. Este enfoque puede ser computacionalmente costos, pero no desperdicia demasiados datos.
# 
# Una explicación gráfica de este procedimiento se puede ver en el siguiente gráfico.
# 
# ![k_fold](figuras/k-fold.PNG)
# 
# ```{index} cross_val_score
# ```
# ### corss_val_score
# 
# Uno de los procedimientos más sencillos y cómodos para usar la validación cruzada en scikip learn es utilizando la función auxiliar <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html#sklearn.model_selection.cross_val_score" target="_blank">cross_val_score </a>en el estimador y el conjunto de datos. 
# 
# Para ver un ejemplo de cómo poder utilizar esta función, a continuación se utiliza el conjunto de datos iris cargado anteriormente y se comprueba el ajuste del modelo mediante un una validación cruzada de un total de 4 grupos de datos o folds 

# In[3]:


from sklearn.model_selection import cross_val_score
clf = svm.SVC(kernel='linear', C=1, random_state=42)
scores = cross_val_score(clf, X, y, cv=4) # cv=4 se toman 4 folds
print("Los escores obtenidos son los siguientes")
scores


# Una vez obtenidos esos valores, lo que se suele hacer es calcular la puntuación media y su desviación estándar, como se muestra a continuación.

# In[4]:


print("%0.2f acuracidad del modelo con una desviación estándar de %0.2f" % (scores.mean(), scores.std()))


# Como puede observarse, de forma predeterminada, *corss_val_score* utiliza como métrica de cálculo el método *score* del modelo. Este comportamiento predeterminado se puede cambiar sin más que utilizar el parámetro *scoring* y darle como valor una regla de evaluación del modelo de las que se utilizan en scikit learn y cuya relación se puede ver en <a href="https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter" target="_blank">este enlace</a>. Igualmente, en el tema [medidas de la Bondad del Ajuste](medidasBondadAjuste), se puede var de forma pormenorizada muchas de las métricas que se utilizan para comprobar el ajuste realizado.

# In[5]:


from sklearn import metrics
scores = cross_val_score( clf, X, y, cv=4, scoring='f1_macro')
scores


# En los ejemplos anteriores se ha proporcionado al parámetro cv un valor entero, por lo que en estos casos se utilizan las estrategias de partición de datos ([más adelante se explican este tipo de estrategias](estrategiasvalidacion) ) denominadas KFold o StratifiedKFold. Si el estimador es un clasificador y el valor de y es una clasificación binaria o multiclase, se utiliza StratifiedKFold, en otro caso se usa KFold.
# 
# Pero este comportamiento se puede modificar y utilizar otro tipo de estrategias de validación cruzada, debiendo pasar para ello al parámetro *cv* un [iterador de validación cruzada](estrategiasvalidacion), como puede verse en el siguiente ejemplo.

# In[6]:


from sklearn.model_selection import ShuffleSplit
n_samples = X.shape[0]
# Definimos la estrategia de partición de los datos
cv = ShuffleSplit(n_splits=4, test_size=0.3, random_state=0)
cross_val_score(clf, X, y, cv=cv)


# Se pueden también definir funciones a medida para pasar su nombre al parámetro *cv*.

# In[7]:


def custom_cv_2folds(X):
    n = X.shape[0]
    i = 1
    while i <= 2:
        idx = np.arange(n * (i - 1) / 2, n * i / 2, dtype=int)
        yield idx, idx
        i += 1

custom_cv = custom_cv_2folds(X)
cross_val_score(clf, X, y, cv=custom_cv)


# ### cross_validate
# 
# ```{index} cross_validate
# ```
# 
# la función <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate" target="_blank"> cross_validate </a> hace las mismas funciones de *cross_val_score* pero puede devolver más información.
# 
# * Permite especificar múltiples métricas para la evaluación del modelo.
# 
# * Devuelve un diccionario que contiene tiempos de ajustes, tiempos de puntuación y opcionalmente puntuaciones de entrenamiento así como de estimadores ajustados, además de la puntuación  de prueba.
# 
# Para la evaluación de múltiples métricas, el valor devuelto es un diccionario con las siguientes claves: ['test_<scorer1_name>', 'test_<scorer2_name>', 'test_<scorer...>', 'fit_time', 'score_time'].
# 
# El parámetro *return_train_score* por defecto tiene un valor de False con la finalidad de ahorrar tiempo de cálculo, entonces para evaluar también las puntuaciones en el conjunto de entrenamiento, debe establecerse a un valor de  True.
# 
# Las métricas múltiples se pueden especificar como una lista, una tupla o un conjunto de nombres de marcador predefinidos:

# In[8]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
scoring = ['precision_macro', 'recall_macro']
clf = svm.SVC(kernel='linear', C=1, random_state=0)
scores = cross_validate(clf, X, y, scoring=scoring)
sorted(scores.keys())

scores['test_recall_macro']


# También se pueden definir como un nombre de anotador de asignación en un diccionario a una función de puntuación predefinida o personalizada:

# In[9]:


from sklearn.metrics import make_scorer
scoring = {'prec_macro': 'precision_macro',
           'rec_macro': make_scorer(recall_score, average='macro')}
scores = cross_validate(clf, X, y, scoring=scoring,
                        cv=5, return_train_score=True)
sorted(scores.keys())


scores['train_rec_macro']


# A continuación se muestra un ejemplo de *cross_validate* con una sola métrica.

# In[10]:


scores = cross_validate(clf, X, y,
                        scoring='precision_macro', cv=5,
                        return_estimator=True)
sorted(scores.keys())


# (estrategiasvalidacion)=
# ## Estrategias de partición de datos.
# ```{index} iteradores validación cruzada, cross validation iterators
# ```
# Este tipo de estrategias también asumen la denominación de Iteradores de Validación cruzada (Cross validation iterators) y dependiendo del conjunto de datos con los que se esté trabajando, se pueden asumir diferentes estrategias:
# 
# ### Datos igualmente distribuidos.
# 
# #### K-Fold
# 
# Con <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold" target="_blank">esta estrategia</a> se divide todo el conjunto de datos en k grupos de igual tamaño, llamados también flods (si k=n entonces esta estrategia es igual que la denominada *Leave One Out* que se verá posteriormente ). En este caso la función de predicción aprende de los k-1 folds y el folds restante se utiliza como prueba de test.
# 
# #### Repated K-Fold.
# 
# Con <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html#sklearn.model_selection.RepeatedKFold" target="_blank">esta estrategia</a>, lo que se hace es repetir el procedimiento anterior de división en k grupos de igual tamaño, n veces.
# 
# #### Leave One Out (LOO).
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html#sklearn.model_selection.LeaveOneOut" target="_blank">En este caso </a>generamos una validación cruzada, pero de tal manera que en cada iteración se deja un elemento fuera  que servirá de test del modelo, el resto de datos servirá para entrenar el modelo. 
# 
# #### Leave P Out (LPO).
# 
# Con <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePOut.html#sklearn.model_selection.LeavePOut" target="_blank">este modelo de selección de muestras</a>, se eliminan cada vez p datos del conjunto original que servirán para hacer el test y el resto de datos se utilizan para entrenar el modelo. Por lo tanto es un método muy similar a LOO visto antes, y en total el número de muestras de entrenamiento será igual a  $\binom{n}{p}$.
# 
# #### Permutaciones aleatorias o Shuffle & Split.
# 
# El iterador <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit" target="_blank">ShuffleSplit</a> genera un número de divisiones del conjunto de datos definido por el usuario de forma totalmente independiente en generación de los datos. En este caso las muestras se "barajan" y después se dividen en conjunto de train y de test.
# 
# A continuación se muestra esto con un ejemplo

# In[11]:


from sklearn.model_selection import ShuffleSplit
X = np.arange(15)
ss = ShuffleSplit(n_splits=5, test_size=0.25, random_state=0)
for train_index, test_index in ss.split(X):
    print("%s %s" % (train_index, test_index))


# Podemos observar que como el 25 por ciento de n=15 es 3.75, las muestras de test están formadas por un total de cuatro elementos. Con el parámetro n_split=5 se indica que queremos obtener un total de 5 pares de muestras train-test.

# ### Datos con clases no equilibradas.
# 
# Algunos problemas de clasificación pueden exhibir un gran desequilibrio en la distribución de las clases objetivo: por ejemplo, podría haber varias veces más muestras negativas que muestras positivas. 
# 
# En tales casos, se recomienda utilizar el muestreo estratificado implementado en StratifiedKFold y StratifiedShuffleSplit para garantizar que las frecuencias de clase relativas se conserven aproximadamente en cada conjunto de pares train-test
# 
# #### K-fold estratificado.
# 
# Este método de partición de la muestra, <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold">StratifiedKFoldy</a>, conserva las proporciones de clase, tanto en el conjunto de datos para entrenamiento como en el de test. Similar a K-Fold visto antes pero manteniendo las proporciones de la clase que hay en el conjunto de datos.
# 
# #### División aleatoria estratificada.
# 
# El método <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html#sklearn.model_selection.StratifiedShuffleSplit" target="_blank"> StratifiedShuffleSplit</a> es una variación de ShuffleSplit visto antes, pero teniendo en cuenta  que mantiene las proporciones de la clase de clasificación tanto en el conjunto de prueba como en el de test. 

# ### Datos agrupados.
# 
# Estaremos en esta situación cuando tengamos un conjunto de datos de tal forma que una serie de datos tengan el mismo identificador. Un ejemplo de esta situación se puede dar por ejemplo si tenemos datos médicos de determinados pacientes, en este caso cada paciente tendrá un identificador único y este dato sería el identificador del grupo.
# 
# En este caso, nos gustaría saber si un modelo entrenado en un conjunto particular de grupos se generaliza bien a los grupos no vistos y que estarían en el grupo de test. Para medir esto, debemos asegurarnos de que todas las muestras en el pliegue de validación provengan de grupos que no están representados en absoluto en el pliegue de entrenamiento emparejado.
# 
# Scikit learn nos proporciona las siguientes herramientas para poder conseguir esto. El identificador de agrupación de datos se indica mediante el parámetros *groups*.
# 
# #### Group k_Fold.
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html#sklearn.model_selection.GroupKFold" target="_blank">GroupKFold </a> es una variación de K-fold visto anteriormente pero teniendo en cuenta el grupo de pertenencia, de esta manera se garantiza que un mismo grupo no esté tanto en el entrenamiento como en el test.
# 
# Por ejemplo, supongamos que tenemos cuatro pacientes cuya identificación es 1,2,3 y 4 entonces se puede implementar esta estrategia de la siguiente manera.

# In[12]:


from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10, 3, 5, 7]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d", "d","e","c"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))


# Como puede verse en la salida anterior, una misma persona no está al mismo tiempo en el grupo de entrenamiento y de test, y así de esta manera tenemos perfectamente separados los grupos.
# 
# #### StratifiedGroupKFold
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html#sklearn.model_selection.StratifiedGroupKFold" target="blank">StratifiedGroupKFold </a> es similar a GroupKFold, lo único que mantiene las proporciones de clase de la muestra total en los conjuntos de train y de test.
# 
# #### LeaveOneGroupOut 
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html#sklearn.model_selection.LeaveOneGroupOut" target="_blank">LeaveOneGroupOut </a> similar LOO visto anteriormente, pero teniendo en cuenta el grupo de pertenencia.

# In[13]:


from sklearn.model_selection import LeaveOneGroupOut

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10, 3, 5, 7]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d", "d","e","c"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4]

logo = LeaveOneGroupOut()
for train, test in logo.split(X, y, groups=groups):
    print("%s %s" % (train, test))


# #### Dejar p grupos fuera (LeavePGroupsOut)
# 
# <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeavePGroupsOut.html#sklearn.model_selection.LeavePGroupsOut" target="_blank">LeavePGroupsOut </a> es similar a LPO, pero teniendo en cuenta que un grupo no puede estar al mismo tiempo en el grupo de train y de test.
# 
# #### División aleatoria grupal.
# 
# El iterador <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html#sklearn.model_selection.GroupShuffleSplit" target ="_blank">GroupShuffleSplit</a>  se comporta como una combinación de ShuffleSplity LeavePGroupsOut.

# In[14]:


from sklearn.model_selection import GroupShuffleSplit

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 0.001]
y = ["a", "b", "b", "b", "c", "c", "c", "a"]
groups = [1, 1, 2, 2, 3, 3, 4, 4]
gss = GroupShuffleSplit(n_splits=4, test_size=0.5, random_state=0)
for train, test in gss.split(X, y, groups=groups):
    print("%s %s" % (train, test))


# ## Valiación cruzada para las series temporales.
# 
# Los datos de series temporales se caracterizan por la correlación entre observaciones cercanas en el tiempo ( autocorrelación ). Sin embargo, las técnicas clásicas de validación cruzada, como KFoldy ShuffleSplitsuponen que las muestras son independientes y están distribuidas de manera idéntica, y darían como resultado una correlación irrazonable entre las instancias de entrenamiento y prueba (lo que produce estimaciones deficientes del error de generalización) en los datos de series temporales. Por lo tanto, es muy importante evaluar nuestro modelo para datos de series temporales sobre las observaciones "futuras" menos parecidas a las que se utilizan para entrenar el modelo. Para lograr esto, una solución es proporcionada por <a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit" target="_blank">TimeSeriesSplit</a>.

# ## Bibliografia.
# 
# * https://es.wikipedia.org/wiki/Validaci%C3%B3n_cruzada

# In[ ]:




