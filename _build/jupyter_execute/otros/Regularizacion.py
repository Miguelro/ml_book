#!/usr/bin/env python
# coding: utf-8

# ## Regularización
# 
# ```{index} regularización (lasso ridge y ElasticNet)
# ```
# 
# ### Introducción.
# 
# En muchas técnicas de machine learning, de lo que se trata de encontrar una serie de coeficientes de tal manera que se minimice una determinada función de coste. En este sentido, la regularización consiste en añadir una determinada penalización a la función de coste. De esta manera, al añadir estas penalizaciones lo que conseguimos es que el modelo obtenido generalice mejor la predicción. Las regularizaciones más utilizadas dentro de la ciencia de datos son las tres siguientes:
# 
# * Lasso (también conocida como L1)
# 
# * Ridge (también conocida como L2)
# 
# * ElasticNet que combina las dos penalizaciones anteriores.
# 
# ### Funcionamiento de la regularización.
# 
# En la mayor parte de lo problemas de regresión, de lo que se trata es de minimizar la función de coste definida por el error cuadrático medio J.
# 
# $$ J=MSE$$
# 
# Cuando se incluye una regularización, lo que se hace es añadir un término a la función de coste que penaliza la complejidad del modelo, y entonces en el caso de MSE se tendrá:
# 
# $$ J=MSE+\alpha \cdot C$$
# 
# de esta manera C es la medida de complejidad del modelo. Dependiendo de cómo se mida la complejidad, se tendrán diferentes tipos de regularización. El hiperperámetro $\alpha$ indica cómo de importante es para nosotros que el modelo sea simple en relación a cómo de importante es su rendimiento.
# 
# Cuando usamos regularización lo que realmente se está haciendo es minimizar la complejidad del modelo a la vez que se minimiza la función de coste. Esto genera modelos más simples que tienden a generalizar mejor. Tener en cuenta que los modelos excesivamente complejos tiende a hacer un sobreajuste del modelo, es decir, a encontrar una solución que funciona muy bien para los datos de entrenamiento pero muy mal para datos nuevos. Lo que interesa es encontrar modelos que además de aprender bien, también tengan una buena capacidad de predicción para datos nuevos.
# 
# ### Regularización Lasso (L1).
# 
# En este tipo de regularización, la complejidad de C se mide como la media del valor absoluto de los coeficientes del modelo y se puede aplicar a regresiones lineales, polinómicas, regresión logística, redes neuronales, máquinas de vectores de soporte, etc. la expresión mátemática a utilizar sería:
# 
# $$C=\frac{1}{N}\sum\left|W_{i}\right| $$
# 
# Para el caso del error cuadrático medio, a continuación se muestra cual es el desarrollo completo para una regularización Lasso.
# 
# $$ C=\frac{1}{M}\sum_{i=1}^{M}(real_{i}-estimado_{i})^{2}+\alpha\cdot\frac{1}{N}\sum_{j=1}^{N}\left|W_{j}\right|$$
# 
# Esta regularización es útil cuando sospechemos que varios de los atributos de entrada (features) sean irrelevantes. Al usar Lasso, estamos fomentando que la solución sea poco densa. Es decir, favorecemos que algunos de los coeficientes acaben valiendo 0. Esto puede ser útil para descubrir cuáles de los atributos de entrada son relevantes y en general para obtener un modelo que generalice mejor. Lasso nos puede ayudar en este sentido, a hacer la selección de atributos de entrada. Lasso funciona mejor cuando los atributos no están muy correlados entre ellos.
# 
# ### Regularizacón Ridge (L2).
# 
# En la regularización Ridge, la complejidad C se mide como la media del cuadrado de los coeficientes del modelo. Al igual que ocurría en Lasso, la regularización Ridge se puede aplicar a varias técnicas de aprendizaje automático. La expresón matemática es la siguiente:
# 
# $$C=\frac{1}{2N}\sum_{j=1}^{N}W_{i}^{2} $$
# 
# Para el caso del error cuadrático medio, el desarrollo completo de Lasso es:
# 
# $$C=\frac{1}{M}\sum_{i=1}^{M}(real_{i}-estimado_{i})^{2}+\alpha\cdot\frac{1}{2N}\sum_{j=1}^{N}W_{i}^{2} $$
# 
# La regularización Ridge es útil cuando se sospeche que varios de los atributos de entrada (features) estén correlados entre ellos. Con Ridge lo que se consigue es que esos coeficientes acaben siendo pequeños. Esta disminución de los coeficientes minimiza el efecto de la correlación entre los atributos de entrada y hace que el modelo generalice mejor. Ridge funciona mejor cuando la mayoría de los atributos son relevantes.
# 
# ### Regularización ElasticNet (L1 y L2)
# 
# ElasicNet combina las regularizaciones L1 y L2, además con el parámetro r podemos indicar qué importancia relativa tiennen Lasso y Ridge respectivamente. La expresión matemática que rije esta regularización es:
# 
# $$ C=r \cdot Lasso +(1-r) \cdot Ridge $$
# 
# Desarrollando y utilizando como función de pérdida el error cuadrático medio, se tendrá lo siguiente:
# 
# $$C=\frac{1}{M}\sum_{i=1}^{M}(real_{i}-estimado_{i})^{2}+r\cdot\alpha\cdot\frac{1}{N}\sum_{j=1}^{N}\left|W_{j}\right|+(1-r)\cdot\alpha\cdot\frac{1}{2N}\sum_{j=1}^{N}W_{i}^{2}
#  $$
# 
# La regularización ElasticNet se utilizará cuando se tenga un gran número de atributos, algunos serán irrelevantes y otros estarán correlados entre ellos.

# In[ ]:




