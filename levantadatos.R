rm(list=ls())
library(rpart)
library(dplyr)
library(rpart)
library(rpart.plot)
library(ROCR)
library(glmnet)
library(imbalance)
library(caret)
options(scipen = 999)

train <- read.table('C:/Users/julie/OneDrive/Escritorio/Di Tella/Modulo 2/Machine Learning/TP/datosTP2020/train.csv', header = T, sep =',', dec = '.')

# INGENIERIA DE ATRIBUTOS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#train_new <- train %>% filter(TARGET == 1)
#train_new <- train %>% filter(var36 != 99)
#train_new2 <- train %>% filter(var36 == 99)

#train_oversampling <- rbind(train, train_new) # Hacemos oversampling

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
  max_elem <- train[i] == as.numeric(names(which.max(table(train[i]))))
  total_cases <- sum(max_elem, na.rm = TRUE)
  
  if (total_cases/nrow(train) > 0.95){
    train[i] = NULL
  }
}
max_elem <- NULL

# Vemos las variables que pasaron el filtro:
str(train)
summary(train)

# Eliminamos la variable ID ya que no tiene poder explicativo:
train$ID <- NULL

# Vemos que sucede con la variable age, en la que creemos que se imputo el valor 23 a todos los NA:
# UNDERSAMPLING en la variable age.
hist(train$age, breaks = 40)
prop.table(table(train$age == 23, train$TARGET))

train_borrador <- train %>% filter(age == 23 & TARGET == 0)
id = sample(nrow(train_borrador), 0.75*nrow(train_borrador))
train_borrador <- train_borrador[id, ]

train_piola <- train %>% filter(age != 23)
train_avion <- train %>% filter(age == 23 & TARGET == 1)

train <- rbind(train_borrador, train_piola, train_avion)

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

# Ponemos TARGET como factor:
train$TARGET <- as.factor(ifelse(train$TARGET == 0, "No churn", "Churn"))

# Dividimos en entrenamiento y testeo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
set.seed(777)
samp <- sample(c(0.2*nrow(train)))

train <- train[-samp,]

test <- train[samp,]

# PROBANDO MODELOS
ctrl <- trainControl(method = "boot", # Usamos el metodo bootstrap
                     number = 5, 
                     verboseIter = FALSE,
                     sampling = "down") # Hacemos undersampling

model_rf_under <- caret::train(TARGET ~ .,
                               data = train,
                               method = 'rpart',
                               trControl = ctrl)

final_under <- data.frame(actual = test$TARGET,
                          predict(model_rf_under, newdata = test, type = "prob"))

final_under$predict <- ifelse(final_under$Churn > 0.65, "Churn", "No churn")

cm_under <- confusionMatrix(as.factor(final_under$predict), test$TARGET)
