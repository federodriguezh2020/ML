##### Carga de datos y algunos comentarios adicionales.
##### descompite el fichero .zip y corre la linea:
rm(list=ls())
train <- read.table('C:/Users/julie/OneDrive/Escritorio/Di Tella/Modulo 2/Machine Learning/TP/datosTP2020/train.csv', header = T, sep =',', dec = '.')

#############################################################
### COMENTARIOS BREVES: #####################################
#############################################################

#@ El objetivo es modelar la variable TARGET (1 = cliente insatisfecho).
#@ No hay un diccionario de variables <data set anonimizado>, sin embargo
#@ se pueden deducir algunas varias cosas del data set:
######          el prefijo 'imp' y 'saldo'  indican importes y saldos (var continuas).
######          el prefijo 'num' indica número de cosas (contactos a call center, cantidad de productos contratados, etc.)
######          el prefijo 'in'  indican parecería ser un flag (variable categórica).
######          el prefijo 'delta' indica cambios en el tiempo de alguna de las variables.
######          Aparecen algunas variables misteriosas como var 36, 21 y 38.

# A trabajar en el TP.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# EXPLORAMOS UN POCO
head(train)

# ¿Cuántas observaciones y features tenemos?
dim(train) # 33008 x 312

# Exploramos un poco, Cómo es la estructura de los datos?:
str(train)
summary(train)

# Que proporcion tenemos de cada tipo de churn:
prop.table(table(train$TARGET)) # Data set desbalanceado.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ANALISIS UNIVARIADO

#@ Notar que si bien algunas variables tienen indicado datos faltantes, en algunas otras (como 'age') es 
# evidente que se han imputado muchos datos perdidos con algún valor de referencia. Mira su distribución:
hist(train$age) # Desde los 23 a los 27 aprox.

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# INGENIERÍA DE ATRIBUTOS

# Como vimos en la exploracion de datos hay muchas variables en las que el % de valores distintos de 0 es muy baja.
# Decidimos probar que sucede al eliminarlas. 

table(train$num_aport_var13_hace3)

table(train$num_op_var40_efect_ult1, train$TARGET) # Casi no hay valores distintos de 0. 
options(scipen = 999)
prop.table(table(train$num_aport_var17_ult1, train$TARGET))
table(train$num_var1_0, train$TARGET) 
table(train$num_var1, train$TARGET)