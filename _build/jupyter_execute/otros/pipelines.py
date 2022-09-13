#!/usr/bin/env python
# coding: utf-8

# ## Pipelines , canalizaciones o tuberías.
# ```{index} pipelines o canalizaciones o tuberías
# ```
# 
# ### Introducción
# Esta herramienta que ofrece Scikit Learn se utiliza mucho en ciencia de datos para poder incorporar en una sola instrucción una serie de transformadores que se combinan con regresores, clasificadores u otros estimadores para obtener con una sola instrucción de scickit learn los resultados que se buscan en un determinado trabajo.
# 
# Los pipelines o tuberías se pueden utilizar posteriormente como si fuesen un estimador más. Lo que permite emplearlas en clases como `GridSearchCV` para seleccionar los parámetros e hiperparametros de los modelos mediante validación cruzada.
# ```{index} GridSearchCV
# ```
# Uno de los primeros pasos que normalmente se hacen en los procesos de análisis e inferencia sobre los datos es la transformación de los mimos para adaptarlos a las necesidades que el investigador necesita y así poder obtener conclusiones fiables. Para poder realizar estas tareas previas, Scikit Learn posee los denominados transformadores, que se pasan a exponer en el siguiente apartado. 
# 
# ### Los transformadores.
# ```{index} transformadores, estimadores
# ```
# 
# En aprendizaje automático o análisis de datos, la creación de un modelo es un proceso complejo que requiere llevar a cabo múltiples pasos. Siendo la preparación de los datos uno de los que más tiempo requiere. Tras la obtención de un conjunto de datos es necesario aplicarle a este diferentes operaciones antes de poder utilizar un estimador. A modo de ejemplo algunas de las operaciones más habituales son: limpieza de datos, extracción de las características, normalización de las características y reducción de la dimensionalidad.
# 
# Scikit Learn cuenta con una buena colección de transformadores, que [se pueden encontrar en este enlace](https://scikit-learn.org/stable/data_transforms.html), los cuales permiten hacer las transformaciones oportunas de los datos para que una vez depurados convenientemente o se realicen las acciones indicadas en el párrafo anterior, pasen a ser analizados mediante las diversas técnicas de análisis de datos que ofrece Scikit Learn.
# 
# Es conjunto de transformadores, puede por ejemplo limpiar los datos iniciales ([preprocesamiento de los datos](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)), reducir la dimensión de los mismos, es decir reducir el número de variables a estudiar por otra cantidad inferior pero que conservan la mayor cantidad de información posible del conjunto inicial de los datos ([reducción de la dimensionalidad](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#data-reduction)) o [extraer features](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction).
# 
# Al igual que otros estimadores, estos están representados por clases con un método *fit*, que aprende los parámetros del modelo (p. ej., media y desviación estándar para la normalización) de un conjunto de entrenamiento, y un método *transform*  que aplica este modelo de transformación a datos no vistos. El método *fit_transform* puede ser más conveniente y eficiente para modelar y transformar los datos de entrenamiento simultáneamente.
# 
# En Scikit Learn hay que distinguir entre los conceptos de transformadores y estimadores. De forma resumida, se pueden distinguir estas características utilizando el siguiente criterio:
# 
# * Un **Transformador** se refiere a un objeto con los métodos fit() y transform(), que permiten limpiar, reducir, expandir o generar nueva features. De esta manera, los transformadores ayudan a modificar los datos para que puedan ser pasados a un algoritmo de machine learning. Ejemplos de estos transformadores pueden ser *OneHotEncoder* y *MinMaxScaler*.
# 
# * Un **Estimador** hace referencia a un modelo de machine learning, y en este sentido será un objeto con los métodos *fit()* y *predict()*.
# 
# ### Construcción y uso de las tuberías.
# 
# En este apartado, vamos a ver cómo se construyen estas tuberías desde Scikit Learn, y para ello es preciso tener en cuenta que en estas tuberías todos los estimadores, excepto el último, deben ser transformadores (es decir, deberán tener el método transform()), mientras que el último puede ser de cualquier tipo (transformador, clasificador, etc.).
# 
# Estos objetos pipelines, se construyen utilizando una lista de pares del tipo (clave,valor), de tal manera que clave es una cadena que contiene el nombre específico que se quiere dar a este paso, y valor será el nombre del estimador utilizado.
# 
# Para entender mejor este concepto veamos a continuación un pequeño ejemplo, en el que se hace primero un análisis de componentes principales y después se calcula un ajuste mediante regresión lineal.

# In[1]:


from sklearn.pipeline import Pipeline
from sklearn import linear_model
from sklearn.decomposition import PCA
estimadores = [('reduce_dim', PCA()), ('regresion', linear_model.LinearRegression())]
pipe = Pipeline(estimadores)
pipe


# Los transformadores utilizados en la canalización se pueden almacenar en caché usando el argumento *memory*.
# 
# La lista de parámetros que se pueden utilizar con esta clase son los siguientes:
#     
# * **steps** : Es una lista de tuplas que serán encadenadas en el orden en el que se han declarado en este parámetro.
#     
# * **memory** : Se utiliza para poder almacenar en caché los transformadores utilizados en las tuberías.
#     
# * **verbose** : Es un valor booleano, que por defecto tiene el valor de False. Si se le cambia a True, se va a mostrar el tiempo empleado en cada paso.
# 
# Estos parámetros y atributos de la clase  [se pueden ver en este enlace](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html). 
# 
# ```{index} make_pipeline
# ```
# 
# Scikit Learn ofrece una función de utilidad denominada [*make_pipeline*](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html) que permite también construir tuberías, de tal forma que esta función toma un número variable de estimadores para devolver una canalización, y los nombres los construye de forma automática. Veamos el ejemplo anterior, pero utilizando este formato abreviado.

# In[2]:


from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.decomposition import PCA
#estimadores = [('reduce_dim', PCA()), ('regresion', linear_model.LinearRegression())]
pipe2 = make_pipeline(PCA(),linear_model.LinearRegression())
pipe2


# Los estimadores que están en una canalización, se almacenan como una lista en el atributo *steps* y se puede acceder a ellos por el índice:

# In[3]:


pipe.steps[0]


# Y también se puede acceder mediante el nombre dado a la canalización:

# In[4]:


pipe2.named_steps["pca"]


# #### Definiendo los parámetros de los elementos de una canalización.
# 
# Ya sabemos que las clases que nos ofrece Scikit Learn, tienen una serie de parámetros que permiten hacer ajustes de tal manera que se obtengan modelos los más ajustado posible a los datos con los que se tarabaja.
# 
# Se pueden acceder a los parámetros de estos estimadores para su modificación en la canalización utilizando para ello la siguiente sintaxis: <estimador>__<parametro>.
#     
# En el ejemplo anterior, se ha utilizado un análisis de componentes principales, pero se han utilizado para ello los parámetros que por defecto tienen la clase PCA(), y el número de componentes que se extraen es el que por defecto ofrece esa clase. Si quisiéramos modificar este parámetro, por ejemplo a dos componentes lo haríamos de la siguiente manera: 

# In[5]:


pipe.set_params(reduce_dim__n_components = 2)


# Esta posibilidad de poder los valores de los parámetros es muy importante, sobre todo si queremos hacer ajustes del modelo mediante búsquedas de cuadrículas, mediante la clase GridSearchCV. Para este caso, se puede utilizar la línea marcada en el siguiente ejemplo:

# In[6]:


from sklearn.model_selection import GridSearchCV

param_grid = dict(reduce_dim__n_components=[2, 5, 10],
                  regresion__fit_intercept=[True,False])

grid_search = GridSearchCV(pipe, param_grid=param_grid)



# ### Utilización de la caché.
# 
# La utilización de transformadores puede ser muy costosa en términos de cálculos que pueden hacer. Para intentar mejorar el tiempo de ejecución, la clase pipeline permite almacenar en la caché cada transformador después de llamar al método *fit*. Esta función es utilizada con la finalidad de evitar calcular los transformadores de ajuste dentro de una tubería siempre que los parámetros y los datos de entrada sean idénticos. Esto se puede conseguir gracias a la utilización de parámetro *memory*, el cual puede ser una cadena que contenga el directorio donde almacenar en cache la información de los transformadores o un objeto de tipo *joblib.Memory*. Veamos a continuación un ejemplo:

# In[7]:


from tempfile import mkdtemp
from shutil import rmtree
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
estimators = [('reduce_dim', PCA()), ('clf', SVC())]
cachedir = mkdtemp()
pipe = Pipeline(estimators, memory=cachedir)
pipe


# In[8]:


# borramos el directorio de caché para cuando ya no se necesita
rmtree(cachedir)


# Este tipo de tuberías, se apoyan en muchas ocasiones en dos clases que sirven para facilitar el trabajo de preparación de los datos en una determinada operación de machine learning. Estas clases son ColumnTranformer y FeatureUnion, que pasamos a verlas a continuación-

# ### La clase ColumnTransformer .
# 
# ```{index} ColumnTransformer (clase)
# ```
# 
# Como su propio nombre indica, se utiliza esta clase para hacer una transformación previa de los datos que aparecen en las columnas del dataset (array o dataframe de pandas)sobre el que estamos trabajando.
# 
# Gracias a esta clase, de una manera separada y mediante una sola instrucción se puede proceder a transformar en diferentes formatos las columnas de nuestro dataset. Además como esta transformación se hace de forma independiente para cada columna, se puede utilizar una programación en paralelo, con la finalidad de agilizar las transformación, lo cual es bastante interesante en los casos en los que trabajemos con ficheros muy voluminosos.
# 
# Este procedimiento es muy útil para conjunto de datos que tienen columnas heterogéneas de tipos de datos, ya que de esta manera se integran en un sólo transformador diferentes mecanismos de extracción o transformación de información.
# 
# Los parámetros más importantes de esta clase son los siguientes:
# 
# * **transformers**. Es una lista de tuplas, de forma que cada tupla tiene el siguiente formato: (nombre, transformador, columnas). El significado de cada elemento de la tupla es el siguiente:
#     * **nombre**. Es el nombre que asignamos a esta transformación en concreto. Nos servirá  para asignar determinados parámetros al transformador mediante la expresión *set_params*, de forma similar a como se hace con los pipelines ya vistos anteriormente.
#     * **transformador**. Puede tomar los valores de 'drop', 'passthrough' o un determinado estimador. Este estimador debe admitir los parámetros *fit* y *transform*. Con la opción 'drop' se eliminará la columna, y con 'passthrough', se mantiene la columna tal y como está.
#     * **columnas**. Hace referencia a la o las columnas sobre las que queremos hace la transformación. Esta referencia a esas columnas, se puede hacer mediante una lista de enteros indicando las posiciones de las columnas, o bien mediante una lista de literales que indican los nombre de las columnas a las que queremos hacer la transformación.
#     
# * **remainder*. Con este parámetro lo que se indica es qué hacer con las columnas sobre las que no se ha indicado ningún tipo de transformación, ya que por defecto lo que se hace es prescindir de ese tipo de columnas. El valor por defecto de este parámetro es 'drop', por ese motivo ocurre lo comentado anteriormente. Sin embargo si a este parámetro le asignamos el valor 'passthrough' todas las columnas no especificadas en el parámetro *transformers* se mantienen.
# 
# * **n_jobs**. Es un parámetro muy indicado para cuando se esté trabajando con un conjunto de datos de mucho peso, ya que con él se está indicando el número de núcleos que se quieren utilizar de nuestro ordenador para hacer un procesamiento en paralelo.
# 
# ```{index} make_column_transformer  (clase)
# ```
# 
# Existe una clase denominada *[make_column_transformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.make_column_transformer.html#sklearn.compose.make_column_transformer)*  que está indicada para realizar una función similar a ColumnTransformer pero de una forma más fácil para el usuario de la misma.
# 
# A continuación pasamos a mostrar un ejemplo sobre la utilización de esta herramienta proporcionada por Scikit Learn. 

# In[9]:


from sklearn.datasets import fetch_openml
# Cargamos el conjunto de datos de titanic
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)


# Veamos un poco el contenido de este fichero.

# In[10]:


X.head()


# Como podemos observar este conjunto de datos presenta campos de muy diversa índole, y así por ejemplo tenemos variables de tipo categórico: pclass, sex, embarked. También hay variables de tipo numérico como age, ticket.
# 
# Algunas características de estas variables, se pueden obtener de forma cómoda con el método *describe()*.

# In[11]:


X.describe()


# También podemos obtener información del tipo de variable con el que estamos trabajando con la siguiente instrucción:

# In[12]:


X.info()


# De acuerdo con la información obtenida en los pasos anteriores, vamos a proceder a hacer lo siguiente con todas las variables que conforman nuestro conjunto de datos:
# 
# * name, cabin, boat, body, home.dest, se van a borrar de la base de datos.
# 
# * sex, embarked se van a recodificar
# 
# * age, sibsp, fare, se van a escalar.

# ```{index} get_dummies()  (pandas)
# ```
# 
# Veamos inicialmente cómo poder hacer todas estas transformaciones de una forma secuencial. En primer lugar vamos a recodificar como variables de tipo dummies las indicadas anteriormente, para lo cual se empleará el método get_dummies() de pandas.

# In[13]:


import pandas as pd

X = pd.get_dummies(X, columns=['sex', 'embarked'], drop_first=True)

X.head()


# A continuación procedemos a escalar (es decir convertirlas en variables con media igual a cero y varianza igual a 1) las variables  age, sibsp,y fare, para lo cual empleamos la clase StandardScaler.

# In[14]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_X = scaler.fit_transform(X.loc[:,['age', 'sibsp', 'fare']])

X2 = pd.DataFrame(scaled_X, columns=['age', 'sibsp', 'fare'])
X2.head()


# Como puede verse, para realizar el paso anterior hemos tenido que seleccionar previamente las columnas que queremos estandarizar y al final obtenemos un conjunto de datos separado del inicial. Si no lo hacemos de esta manera y trabajamos son todo el conjunto de datos inicial, obtendríamos un error de python que nos diría por ejemplo que las variables de tipo alfanumérico no las puede estandarizar.
# 
# Entonces para hacer este procedimiento de manera secuencial, lo que se puede hacer es dividir el conjunto de datos inicial en subconjuntos a los que se les aplica la transformación que queremos hacer y después unir todos ellos en un sólo conjunto de datos.
# 
# Para una mayor claridad de este procedimiento, vamos a ir paso a paso haciendo esto con nuestro conjunto de datos con los que estamos trabajando.

# In[15]:


# Cargamos el conjunto de datos de titanic
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# separamos los datos
categoricas = X.loc[:,['sex', 'embarked']]
numericas = X.loc[:,['age', 'sibsp', 'fare']]
borrar = X.loc[:,
               [y for y in X.columns if y not in list(categoricas.columns)+list(numericas.columns)]]


# Una vez tenemos separados los datos podemos hacer las transformaciones pertinente en cada uno de estos subconjuntos.
# 
# ```{index} OneHotEncoder
# ```
# 
# Comenzamos por aplicar la clase OneHotEncoder a las variables de tipo categórico.

# In[16]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(dtype='int', drop='first')
encoded_raw = encoder.fit_transform(categoricas)

categoricas = pd.DataFrame(
    encoded_raw.toarray(), 
    columns=encoder.get_feature_names_out()
)
categoricas.head()


# Ahora procedemos a estanadarizar las variables numéricas elegidas

# In[17]:


scaler = StandardScaler()
scaled_raw = scaler.fit_transform(numericas)

numericas = pd.DataFrame(
    scaled_raw, 
    columns=numericas.columns
)
numericas.head()


# Y por último procederíamos a integrar estos subconjuntos en uno sólo

# In[18]:


final = pd.concat([categoricas,numericas], axis='columns')
final.head()


# Todos los procedimientos que se han realizado anteriormente son correctos, pero para hacerlos hemos necesitado hacer una serie de pasos previos que ralentiza el procedimiento. A continuación vamos a hacer lo anterior pro utilizando la clase *ColumnTransformer*, gracias a la cual podemos integrar todo lo anterior en pocas lineas de código, y además podremos contar con la ventaja de hacer una ejecución en paralelo, ya que las transformaciones que se hacen sobre cada una de las columnas son independientes. Veamos cómo hacer todo esto a continuación.

# In[19]:


transformaciones = [
('codificacion', OneHotEncoder(dtype='int',drop='first'),['sex', 'embarked']), 
('scale', StandardScaler(), ['age', 'sibsp', 'fare'])
]


# In[20]:


from sklearn.compose import ColumnTransformer

# Cargamos de nuevo el conjunto de datos de titanic
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

column_transformer = ColumnTransformer(transformaciones,
                                       remainder='drop', # la opción por defecto
                                      n_jobs = 4 #para utilizar 4 cores en paralelo
                                      )
# Aplicamos el modelo anterior a nuestros datos
X_transformados = column_transformer.fit_transform(X)

# Obtenemos el conjunto de datos final
X_final = pd.DataFrame(
    X_transformados,
    columns = column_transformer.get_feature_names_out()
)
X_final.head()


# Como puede verse el resultado final es el mismo que el obtenido anteriormente, pero se cuenta con la ventaja de su sencillez y rapidez de cálculo (esto último no se aprecia en este ejemplo ya que son pocos los datos tratados, pero sin embargo sí sería un aspecto a tener en cuenta si trabajamos con datos muy voluminosos).

# ### La clase FeatureUnion.
# 
# ```{index} FeatureUnion (clase)
# ```
# Esta clase es muy común utilizarla en las tuberías de Scikit Learn, por lo que se ha creído conveniente abrir un apartado en este momento para hacer una presentación de la misma. 
# 
# La clase [FeatureUnion](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.FeatureUnion.html#sklearn.pipeline.FeatureUnion) permite combinar una lista de objetos transformadores en un nuevo transformador combinando las salidas de cada uno de ellos. Las transformaciones individuales se hacen de forma independiente y por lo tanto se pueden hacer en paralelo lo que permite aligerar considerablemente los tiempos de cálculo cuando estamos trabajando con conjunto de datos muy voluminosos.
# 
# Veamos a continuación un sencillo ejemplo sobre el uso de esta clase para entender mejor su significado y ver cómo funciona. 
# 

# In[21]:


# Cargamos el conjunto de datos de titanic
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# Me quedo sólo con algunas columnas de dataset X
X = X.loc[:,['pclass','age','sibsp','parch','fare','body']]


# Vemos cuantos valores nulos hay
X.isna().sum()


# Como la variable body tiene muchos datos sin valor la excluimos del análisis, y para el resto procedemos a imputar los valores faltantes, ya que de no hacerlo, el procedimiento nos generaría un error insalvable

# In[22]:


X = X.loc[:,['pclass','age','sibsp','parch','fare']]
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(X)
X = imp.transform(X)
X = pd.DataFrame(X, columns = ['pclass','age','sibsp','parch','fare'])

# Veamos si se han eliminado los valores faltantes
X.isna().sum()


# Ahora que hemos imputados los datos faltantes, procedemos a utilizar la clase FeatureUnion

# In[62]:


from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import PCA, TruncatedSVD

# definimos dos conjuntos de transformormadores
union = FeatureUnion([("pca", PCA(n_components=2)),
                      ("svd", TruncatedSVD(n_components=2))])



# Generamos ahora el nuevo conjunto de datos con las transformaciones indicadas antes
resul = union.fit_transform(X)
resul


# In[63]:


resul.shape


# Como podemos comprobar en la salida anterior, tenemos como resultado un objeto de tipo numpyarray, con el mismo número de filas que el conjunto original y cuatro columnas, de tal manera que las dos primeras columnas se corresponden a la salida de un procedimiento PCA, pues le hemos indicado que conserve las dos primeras componentes, y las otras dos columnas se corresponden a la salida de la clase TruncatedSVD a la que hemos indicado también que conserve 2 componentes.
