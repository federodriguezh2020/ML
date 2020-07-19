rm(list=ls())
library(rpart)
library(dplyr)
library(rpart)
library(rpart.plot)
library(ROCR)
library(glmnet)
library(imbalance)
library(caret)
library(ggplot2)
library(randomForest)
library(scatterplot3d)
library(e1071)
library(corrplot)
library(ggplot2)
library(gbm)
library(MASS)
options(scipen = 999)

train <- read.table("/Users/julietadeantonio/Desktop/train.csv", header = T, sep =',', dec = '.')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# EXPLORANDO EL DATA SET ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
head(train)

# ¿Cuántas observaciones y features tenemos?
dim(train) # 33008 x 312

# Exploramos un poco, Cómo es la estructura de los datos?:
head(str(train),312)
summary(train)

# Que proporcion tenemos de cada tipo de churn:
prop.table(table(train$TARGET)) # Data set desbalanceado.

summary(train)

table(is.na(train))

# Ponemos TARGET como factor:
train$TARGET <- as.factor(ifelse(train$TARGET == 0, "no", "yes"))

# GRAFICAMOS ALGUNAS RELACIONES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ggplot(train) +
  geom_bar(aes(x = TARGET, fill = TARGET)) +
  ggtitle("Cantidad de usuarios por clase") +
  xlab("Churn vs. No Churn") +
  ylab("# de usuarios") + 
  scale_fill_manual(values=c("#E69F00", "darkgrey"))

var <- c(colnames(train[startsWith(colnames(train), prefix = "var")]))

par(mfcol = c(2, 3))
for (k in 1:3) {
  j0 <- var[k]
  br0 <- seq(min(train[var[k]]), max(train[var[k]]), le = 11)
  x0 <- seq(min(train[var[k]]), max(train[var[k]]), le = 50)
  for (i in 1:2) {
    i0 <- levels(train$TARGET)[i]
    x <- train[train$TARGET == i0, j0]
    hist(x, br = br0, proba = T, col = grey(0.8), main = i0,
         xlab = j0)
    lines(x0, dnorm(x0, mean(x), sd(x)), col = "red", lwd = 2)}
}

imp_aport <- c(colnames(train[startsWith(colnames(train), prefix = "imp_ap")]))

par(mfcol = c(2, 6))
for (k in 1:6) {
  j0 <- imp_aport[k]
  br0 <- seq(min(train[imp_aport[k]]), max(train[imp_aport[k]]), le = 11)
  x0 <- seq(min(train[imp_aport[k]]), max(train[imp_aport[k]]), le = 50)
  for (i in 1:2) {
    i0 <- levels(train$TARGET)[i]
    x <- train[train$TARGET == i0, j0]
    hist(x, br = br0, proba = T, col = grey(0.8), main = i0,
         xlab = j0)
    lines(x0, dnorm(x0, mean(x), sd(x)), col = "red", lwd = 2)}
}

age <- c(colnames(train[startsWith(colnames(train), prefix = "age")]))

par(mfcol = c(2, 1))
for (k in 1:1) {
  j0 <- age[k]
  br0 <- seq(min(train[age[k]]), max(train[age[k]]), le = 11)
  x0 <- seq(min(train[age[k]]), max(train[age[k]]), le = 50)
  for (i in 1:2) {
    i0 <- levels(train$TARGET)[i]
    x <- train[train$TARGET == i0, j0]
    hist(x, br = br0, proba = T, col = grey(0.8), main = i0,
         xlab = j0)
    lines(x0, dnorm(x0, mean(x), sd(x)), col = "red", lwd = 2)}
}

# INGENIERIA DE ATRIBUTOS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Eliminamos los NA:
train <- train %>%
  filter(complete.cases(.))

# Dado que la variable var36 tiene mas del 40% como NaN y se han imputado como el numero 99, 
# cambiaremos ese numero por una media ponderada:

var36_99 <- sample(c(0,1,2,3), size = nrow(train %>% filter(var36 == 99)), 
       prob = c(0.0099, 0.3255, 0.1932, 0.4714), replace = TRUE)

# Chequeamos que nos queden bien los %. Las prob salen de train %>% filter(var36 != 99).
train$var36 <- ifelse(train$var36 == 99, var36_99 , train$var36)
prop.table(table(train$var36)) 

# Eliminamos todas aquellas variables con mas del 95% de las observaciones con el mismo valor.
for (i in 1:(ncol(train)-1)){
  maximo <- train[i] == as.numeric(names(which.max(table(train[i]))))
  total <- sum(maximo, na.rm = TRUE)
  
  if (total/nrow(train) > 0.95){
    train[i] = NULL
  }
}
maximo <- NULL

# Vemos las variables que pasaron el filtro:
str(train)
summary(train)

# Eliminamos la variable ID ya que no tiene poder explicativo:
train$ID <- NULL

# Cambiamos la clase de las variables:
vector_nombre_ind <- c(colnames(train[startsWith(colnames(train), prefix = "ind")]))

for (i in 1:length(vector_nombre_ind)){
  train[[vector_nombre_ind[i]]] <- as.factor(train[,vector_nombre_ind[i]])
}

# Le aplicamos logaritmo a varibales para reescalarlas:
vector_nombre_imp <- c(colnames(train[startsWith(colnames(train), prefix = "imp")]))

for (i in 1:length(vector_nombre_imp)){
  train[[vector_nombre_imp[i]]] <- log(train[,vector_nombre_imp[i]]+1)
}

train$var38 <- log(train$var38)

# Dividimos en entrenamiento y testeo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(777)
samp <- sample(c(0.20*nrow(train)))

train <- train[-samp,]

test <- train[samp,]

# Vemos que sucede con la variable age, en la que creemos que se imputo el valor 23 a todos los NA:
# UNDERSAMPLING en la variable age.
ggplot(train) +
  geom_histogram(aes(x = age, fill = TARGET)) +
  ggtitle("Distribucion de la variable age modificada") +
  xlab("Valores") +
  ylab("# de usuarios") +
  scale_fill_manual(values=c("#E69F00", "darkgrey"))

prop.table(table(train$age == 23, train$TARGET))

train_borrador <- train %>% filter(age == 23 & TARGET == "no")
id = sample(nrow(train_borrador), 0.5*nrow(train_borrador))
train_borrador <- train_borrador[id, ]

train_borrador2 <- train %>% filter(age != 23)
train_borrador3 <- train %>% filter(age == 23 & TARGET == "yes")

train <- rbind(train_borrador, train_borrador2, train_borrador3)

# PASAMOS TODO A MATRIZ, EN LUGAR DE DATA FRAME
matrix_train = model.matrix( ~ . - 1, train)
matrix_test = model.matrix(~. -1 , test)

x_train <- as.matrix(matrix_train[,-186]) # 186 es la variable TARGET. 
y_train = as.matrix(matrix_train[,186]) # 0/1 flag.

x_test <- as.matrix(matrix_test[,-186])
y_test<- as.matrix(matrix_test[,186])

######### REGRESION LOGISTICA SIN REGULARIZACION #######
######################################################

# Entrenamos el modelo Regresion Logistica (LASSO) - Sin Regularizacion
lasso.sin.reg = glmnet(x = x_train , 
                       y = y_train,
                       family = 'binomial', 
                       alpha = 1 , 
                       lambda = 0, # sin regularizacion
                       standardize = TRUE)

summary(lasso.sin.reg)

# Predicciones sobre datos TRAIN 
pred_tr = predict(lasso.sin.reg, s = 0 , newx = x_train, type = 'response')

# Predicciones sobre datos TEST 
pred = predict(lasso.sin.reg, s = 0 , newx = x_test, type = 'response')
head(pred) # s = 0 (sin regularizacion)

# Tabla de frecuencias (churn y no-churn)
freq_table <- table(y_train) / dim(y_train)[1]
prob_priori <- as.numeric(freq_table[1])
freq_table
prob_priori # la vamos a usar como punto de corte.
table(y_train) #/ dim(y_train)

######################
# Performance en TRAIN
######################

# Clasificamos en [0,1] en funcion de una probababilidad a priori
y_hat = as.numeric(pred_tr >= prob_priori) 
head(y_hat)

# Matriz de Confusion
matriz.confusion = table(y_hat, y_train)
matriz.confusion

# Matriz expresada en prob.
prop.table(table(y_hat, y_train))

# Tasa de error 
# estimacion puntual con datos de test
tasa_error_tr <- 1-sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_tr # 11.49 %

##### Curva ROC y Area baja la curva ROC (AUC)
### Primero obtenemos las propabilidades a posteriori, luego:
pred2 <- prediction(pred_tr, y_train)
perf <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf, main="Curva ROC", colorize=T)

auc <- performance(pred2, measure = "auc")
auc@y.values[[1]] # 0.79 (aprox)

######################
# Performance en TEST
######################

# Clasificamos en [0,1] en funcion de una probababilidad a priori
y_hat = as.numeric(pred >= prob_priori) 

# Matriz de Confusion
matriz.confusion = table(y_hat, y_test)
matriz.confusion

# Matriz expresada en prob.
prop.table(matriz.confusion)

# Tasa de error 
tasa_error_rl <- 1-sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_rl # 0.08861912

##### Curva ROC y Area baja la curva ROC (AUC)
#dev.off();
pred2_rl <- prediction(pred, y_test)
perf_rl <- performance(pred2_rl,"tpr","fpr")

# Graficamos la curva ROC
plot(perf_rl, main="Curva ROC", colorize=T)

auc_rl <- performance(pred2_rl, measure = "auc")
auc_rl@y.values[[1]] # 0.8016736 (aprox)

######### REGRESION LOGISTICA + REGULARIZACION #######
######################################################

# Creamos una secuencia de valores para lambda
grid.l =exp(seq(1 , -10 , length = 20)) # Secuencia de lamdas.
grid.l

# Entrenamos el modelo Regularizacion Logistica - Regularizacion
lasso.reg = glmnet(x = x_train , 
                   y = y_train,
                   family = 'binomial', 
                   alpha = 1 , 
                   lambda = grid.l,
                   standardize = TRUE)

# Analizamos Df (grados de libertad a medida que aumenta lambda)
lasso.reg # Tasa de error para cada valor de laamda.

# Visualuzamos la evolucion de los valores de las variables 
# en funcion de los distintos lambda log(??)
plot(lasso.reg, xvar = "lambda", label = TRUE) # Vemos que muchos parametros tienden a 0, se eliminan.

# Predicciones sobre datos TEST 
pred = predict(lasso.reg, s = 0.008 , newx = x_test, type = 'response')
head(pred) # s = 0 (con regularizacion)

# Evaluando PERFORNANCE del modelo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Tabla de frecuencias (churn y no-churn)
freq_table <- table(y_train) / dim(y_train)[1]
prob_priori <- as.numeric(freq_table[1])
freq_table
prob_priori
table(y_train) #/ dim(y_train)

# Clasificamos en [0,1] en funcion de una probababilidad a priori
y_hat = as.numeric(pred >= prob_priori) 

# Comparamos los predicho vs. real para generar matriz de confusion
# Repaso de conceptos: FP y FN
matriz.confusion = table(y_hat, y_test)

# Matriz de confusion
matriz.confusion

# Matriz expresada en prob.
prop.table(matriz.confusion)

# Tasa de error
tasa_error <- 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error # 0.08892261

# Curva ROC y AUC
dev.off();
pred2 <- prediction(pred, y_test)
perf <- performance(pred2,"tpr","fpr")

# Graficamos la curva ROC
plot(perf, main="Curva ROC", colorize=T)

auc <- performance(pred2, measure = "auc")
auc@y.values[[1]] # 0.7754554

###############################################################
###     Ajustando lambda via 5-fold cross-validation.      ####  
###############################################################
cv.out = cv.glmnet(x_train, y_train, 
                   family = 'binomial', # Modelo logistico.
                   type.measure="auc", # Metrica del CV (no depende del umbral).
                   lambda = grid.l, 
                   alpha = 1, # LASSO.
                   nfolds = 5)

cv.out
# Graficamos como varia AUC en funcion del valor de lambda 
plot (cv.out)
bestlam = cv.out$lambda.min
bestlam # poca regularizacion, nos da la idea de que no ganamos mucho con la regularizacion.

# Lambda con 1 Stev
cv.out$lambda.1se

# Predicciones sobre el conjunto de TEST con el mejor lambda
pred = predict(lasso.reg, s = bestlam , newx = x_test, type = 'response')
y_hat = as.numeric(pred>= prob_priori)

# Evaluamos Performance en TEST
# Matriz de confusion 
matriz.confusion = table(y_hat, y_test)
matriz.confusion

# Tasa de Error
tasa_error_rlr <- 1 - sum(diag(matriz.confusion))/sum(matriz.confusion)
tasa_error_rlr*100 # 8.861912

# Curva ROC y AUC
pred2_rlr <- prediction(pred, y_test)
perf_rlr <- performance(pred2_rlr,"tpr","fpr")
auc_rlr <- performance(pred2_rlr,"auc")
auc_rlr@y.values[[1]] # 0.8007087

########################################################
############## Clasificacion R. Forest: ################
########################################################

churn.forest = randomForest(TARGET~.,
                            data=train,
                            mtry = floor(sqrt(ncol(train))), # numero de variables candidatas para corte --> sqrt(p) --> Numero recomendado inicial.
                            ntree=5000,                       # B: numero de arboles bootstrap 
                            sample = 0.15*floor(train),       # tamanio de cada re-muestra bootstrap.
                            maxnodes = 10,                   # cantidad maxima de nodos terminales en c/ arbol.
                            nodesize = 8,                   # cantidad minima de datos en nodo terminal.
                            importance=T,                    # Computar importancia de c/ covariable.
                            proximity =F,                     # computa la matriz de proximidad entre observaciones.
                            classwt = c(0.8849329, 0.1150671)
                            # na.action =  na.roughfix       # imputa perdidos con mediana / moda. 
                            # na.action = rfImpute           # imputa perdidos con datos proximos.
                            # na.action = na.omit            # descarta valores perdidos. 
)

# Respecto del Fitting del modelo: 
head(churn.forest$votes,3)     # estimaciones para cada observacion (OOB)
head(churn.forest$predicted,3) # (Idem pero catagorias).
head(churn.forest$oob.times,3) # nro de veces q cada obs queda out-of-bag. # Numero de veces que cada observacion queda outofbag. Nos sirve para optimizar el RF.
churn.forest$confusion         # Matriz de confusuion (OOB) 

# Performance: (estimaciones OOB)
churn.forest 

# Importancia de cada variable: (+ alto = mas contribucion).
head(churn.forest$importance,5)

dev.off()
varImpPlot(churn.forest, main ='Importancia de cada feature')

# Performance fuera de la muestra
pred.rfor.test = predict(churn.forest,newdata=test)
matriz.conf = table(pred.rfor.test,test$TARGET)
matriz.conf
t_error <- 1-sum(diag(matriz.conf))/sum(matriz.conf) 
paste(c("Tasa Error % (TEST) = "),round(t_error,4)*100,sep="")

# Curva ROC sobre test
pred = predict(churn.forest, newdata = test ,type='prob')[,1]
pred2 <- prediction(pred, test$TARGET)
auc.rf <- performance(pred2, measure = "auc")@y.values[[1]]
paste(c("AUC % (TEST) = "),round(auc.rf,6)*100,sep="")

############### Sintonia fina de hiper-parametros (OOB):
valores.m = c(82,86,90,94)  # valores de "m", numero de variables candidatas para corte
valores.maxnode = c(34, 36, 40)  # Complejidad de Arboles en el bosque.
parametros = expand.grid(valores.m = valores.m,valores.maxnode = valores.maxnode) 
# En la practica elegimos una grilla mas grande.
head(parametros,6) 
dim(parametros)

te = c() # Tasa de error estimada por OOB.
for(i in 1:dim(parametros)[1]){ # i recorre la grilla de parametros.
  forest.oob  = randomForest(TARGET~.,
                             data=train,
                             mtry = parametros[i,1], # m
                             ntree=5000,               
                             sample = 0.15*nrow(train), 
                             maxnodes = parametros[i,2], # complejidad 
                             nodesize = 8, 
                             proximity =F,
                             classwt = c(0.8849329, 0.1150671))
  te[i] = 1 - sum(diag(forest.oob$confusion[,-3]))/sum(forest.oob$confusion[,-3])
  print(i)
}

forest.oob

which(min(te)==te)
parametros[which(min(te)==te),] 
best_m <- parametros[which(min(te)==te),][1][2,1]
best_maxnode <- parametros[which(min(te)==te),][2][2,1]

# Aca podemos analizar como varia la T.Error
# en funcion de la grilla de hiperparametros que entrenamos
scatterplot3d(cbind(parametros,te),type = "h", color = "blue")

# Re-entrenamaos con m* y maxnodes*:
modelo.final = randomForest(TARGET~.,
                            data=train,
                            mtry = 86, # best_m,     # Estimacion de m*
                            ntree = 5000,              
                            sample = 0.15*nrow(train), 
                            maxnodes = 36, # best_maxnode, # Estimacion de maxnodes*
                            nodesize = 8,
                            importance=F, 
                            proximity =T,
                            classwt = c(0.8849329, 0.1150671)
)  


dim(modelo.final$proximity) # Matriz de proximidades. 

# Performance en TEST
pred.rfor.test = predict(modelo.final,newdata=test)
matriz.conf = table(pred.rfor.test,test$TARGET)
matriz.conf
t_error_rf <- 1-sum(diag(matriz.conf))/sum(matriz.conf) # 8.79% 
paste(c("Tasa Error % (TEST) = "),round(t_error_rf,4)*100,sep="")

# Curva ROC sobre test
pred_rf = predict(modelo.final, newdata = test ,type='prob')[,2]
pred2_rf <- prediction(pred_rf, test$TARGET)
perf_rf <- performance(pred2_rf,"tpr","fpr")
auc.rf <- performance(pred2_rf, measure = "auc")@y.values[[1]] # 76.944
paste(c("AUC % (TEST) = "),round(auc.rf,6)*100,sep="")

########################################################
######################## SVM ###########################
########################################################

# Curva ROC
rocplot = function (pred , truth , ...) {
  predob = prediction(pred , truth )
  perf = performance(predob ,"tpr" , "fpr")
  auc =  performance(predob , measure = "auc")
  plot ( perf ,...) 
  return(auc@y.values[[1]])
}

########################################################
# SVM GAUSSIANO
svm.rad <- svm(TARGET ~ ., 
               data = train, 
               kernel = "radial", 
               cost = 50,
               gamma = 1,
               scale = TRUE)
svm.rad

summary(svm.rad)

svm.rad$index[1:3] # Indice de los datos de train que resultaron vectores soportes.
train[10,] # Primer vector soporte.

head(svm.rad$SV) # coordenadas de los primeros vectores soporte.

svm.rad$cost

svm.rad$fitted[1:5] # Estimaciones en la muestra de entrenamiento

# Predicciones en test
pred.outsample = predict(svm.rad, newdata = test)
matriz.conf <- table(pred.outsample, test$TARGET)
t_error <- 1-sum(diag(matriz.conf))/sum(matriz.conf) 
paste(c("Tas Error % (TEST) = "),round(t_error,4)*100,sep="")

# Curva ROC sobre test
fitted = attributes(predict(svm.rad,test,decision.values=T))$decision.values
head(fitted, 3) # Distancia al hiperplano.

#dev.off()
#x11()
par(mfrow = c(1,2))
auc.svm.rad <- rocplot(-1*fitted, test$TARGET , main ="TEST Data")
plot(fitted, pch = 20, col = as.factor(test$TARGET))
paste(c("AUC % (TEST) = "),round(auc.svm.rad,6)*100,sep="")

# Optimizacion del parametro COSTO y GAMMA por VC
tune.out <- tune.svm(TARGET ~ ., data = train, 
                     cross = 5, 
                     kernel = "radial",  
                     cost= c(75,100,125),
                     gamma = c(0.005,0.01, 0.015),
                     scale = TRUE)
summary(tune.out)

plot(tune.out)

opt_cost <- tune.out$best.model$cost
opt_gamma <- tune.out$best.model$gamma

opt_cost; opt_gamma

summary(tune.out$best.model)

# Optimizados los parametros

svm.rad <- svm(TARGET ~ ., 
               data = train, 
               kernel = "radial", 
               cost = opt_cost,
               gamma = opt_gamma,
               scale = TRUE)

# Predicciones en test
pred.outsample = predict(svm.rad, newdata = test)
matriz.conf <- table(pred.outsample, test$TARGET)
t_error <- 1-sum(diag(matriz.conf))/sum(matriz.conf) 
paste(c("Tasa Error % (TEST) = "),round(t_error,4)*100,sep="")
#T.Error = 2.55%

pred_svm = predict(svm.rad, newdata = test)[,2]
pred2_svm <- prediction(ifelse(pred.outsample == "no", 0, 1), test$TARGET)
perf_svm <- performance(pred2_svm,"tpr","fpr")

# Curva ROC sobre test
fitted = attributes(predict(svm.rad,test,decision.values=T))$decision.values
head(fitted, 3) # Distancia al hiperplano.

dev.off()
x11()
par(mfrow = c(1,2))
auc.svm.rad <- rocplot(-1*fitted, test$TARGET , main ="TEST Data")
plot(fitted, pch = 20, col = as.factor(test$TARGET))
paste(c("AUC % (TEST) = "), round(auc.svm.rad,6)*100,sep="")
# AUC = 93.25

#########################################################################################################
# GRAFICAMOS TODAS LAS CURVAS ROC
plot(perf_rl, col = "red", lwd = 2, main="Curvas ROC") + plot(perf_rlr, add = TRUE, col = "blue",  lwd = 2) +
  plot(perf_rf, add = TRUE, col = "green",  lwd = 2) + plot(perf_svm, add = TRUE, col = "orange",  lwd = 2) +
  legend("bottomright", legend=c("Regresion Logistica s/ Regularizacion", "Regresion Logistica c/ Regularizacion", "Random Forest", "SVM radial"),
         col=c("red", "blue", "green", "orange"), lty=1:1, cex=1)
  
