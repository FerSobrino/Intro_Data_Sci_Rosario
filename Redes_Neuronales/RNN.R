library(tidyverse)
library(tensorflow)
library(keras)
library(cowplot)

bank <- read_csv('/Users/fernandasobrino/Documents/GitHub/Intro_Data_Sci_Rosario/Redes_Neuronales/banknotes.txt')

dim(bank)
names(bank)
head(bank)

bank$classification <- as.factor(bank$classification)

ggplot(bank) + geom_point(aes(variance, entropy, colour = classification))

## dividir mi muestra en train y test 
idx <- sample(seq(1,2), size = nrow(bank), replace = TRUE, prob = c(0.8,0.2))
train <- bank[idx == 1, ]
test <- bank[idx == 2, ]
x_train <- data.matrix(train[,c("variance","entropy")])
x_test <- data.matrix(test[,c("variance", "entropy")])
y_train <- as.numeric(train$classification)
y_test <- as.numeric(test$classification)

plot1 <- ggplot(train) + geom_bar(aes(classification, fill = classification)) +
  theme(legend.position = "none")
plot2 <- ggplot(test) + geom_bar(aes(classification, fill = classification)) +
  theme(legend.position = "none")

plot_grid(plot1, plot2, labels = "AUTO")


## Activacion y costos 

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

# Feed Forward : 
## z1 = xW1
## h = sigma(z1)
## z2 = hw2
## y_hat = sigma(z2)


feedforward <- function(x, w1, w2) {
  z1 <- cbind(1, x) %*% w1
  h <- sigmoid(z1)
  z2 <- cbind(1, h) %*% w2
  list(output = sigmoid(z2), h = h)
}


## funcion de costos cuadratica 
## gradient descent w* = w - a gradiente(C(w))
# y_hat = sigma(sigma(xw1)w2)
## dw2 = partial C / partial w2 = partial C / partial y_hat * partial y_hat/ partial w2
## dw2 = (y_hat - y) *h * y_hat(1-y_hat)

## dw1 = partial C / partial w1 = partial C / partial y_hat * partial y_hat/ partial h * 
## partial h/ partial w1 
## dw1 = (y_hat -y) y_hat(1-y_hat)w2*x*sigma(xw1)(1-sigma(xw1))

backpropagate <- function(x,y,y_hat,w1,w2,h,a){
  dw2 <- t(cbind(1,h)) %*% (y_hat*(1 - y_hat)*(y_hat - y))
  dh <- (y_hat - y) %*% t(w2[-1, ,drop = FALSE])
  dw1 <- t(cbind(1,x)) %*% (h* (1 - h) * dh)
  w1 <- w1 - a*dw1
  w2 <- w2 - a*dw2
  list(w1 = w1 , w2 = w2)
}


## Entrenar la red 

learn <- function(x, y , hidden = 5, a = 0.01, iterations = 1000){
  d <- ncol(x) + 1
  w1 <- matrix(rnorm(d*hidden), d, hidden)
  w2 <- as.matrix(rnorm(hidden + 1))
  for (i in 1:iterations){
    ff <- feedforward(x, w1, w2)
    bp <- backpropagate(x, y,
                        y_hat = ff$output,
                        w1, w2,
                        h = ff$h, 
                        a = a)
    w1 <- bp$w1
    w2 <- bp$w2
  }
  list(output = ff$output , w1 = w1, w2 = w2)
}


x <- data.matrix(bank[,c("variance","entropy")])
y <- as.numeric(bank$classification)

nn <- learn(x, y, hidden = 5, iterations = 100000)
mean((nn$output > .5) == y)




########################################################### 
library(tidyverse)
library(keras)
library(tensorflow)


raw_data <- read_csv("/Users/fernandasobrino/Documents/GitHub/Intro_Data_Sci_Rosario/Redes_Neuronales/mnist_train.csv", col_names = FALSE)
dim(raw_data)
head(raw_data)

barplot(table(raw_data[,1]), col = rainbow(10,0.5), main = 'Digitos')


idx <- sample(seq(1,2), size = nrow(raw_data), replace = TRUE, prob = c(0.8, 0.2))
train <- raw_data[idx == 1, ]
test <- raw_data[idx == 2, ]
x_train <- as.matrix(train[,2:785])
#x_train <- train %>% select(-X1) %>% as.matrix()
x_test <- as.matrix(test[,2:785])
y_train <- as.matrix(train$X1)
y_test <- as.matrix(test$X1)


## 2 --> (0,0,1,0,0,0,0,0,0,0)T

y_train <- to_categorical(y_train,10)
y_test <- to_categorical(y_test,10)


#### Keras 

model <- keras_model_sequential()

model %>%
  layer_dense(units = 15, activation = "sigmoid", input_shape = c(784)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

model %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd",
    metrics = c('accuracy')
  )

fit_model <- model %>%
  fit(x_train,y_train,
      epochs = 15, 
      batch_size = 128,
      validation_split = 0.2)

model %>% evaluate(x_train, y_train)
model %>% evaluate(x_test, y_test)


## Deep Learning 

model_nn <- keras_model_sequential()

model_nn %>%
  layer_dense(units = 15, activation = "sigmoid",input_shape = c(784)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 128, activation = "sigmoid") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 20, activation = "sigmoid") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 10, activation = "softmax")

model_nn %>%
  compile(
    loss = "categorical_crossentropy",
    optimizer = "sgd",
    metrics = c('accuracy')
  )

model_nn %>% fit(
  x_train, y_train, 
  epochs = 15,
  batch_size = 128,
  validation_split = 0.2
)

model_nn %>% evaluate(x_test,y_test)


## CNN 

train_x_cnn <- array_reshape(x_train,
                             dim = c(nrow(x_train), 28, 28, 1))

test_x_cnn <- array_reshape(x_test,
                             dim = c(nrow(x_test), 28, 28, 1))


model_cnn <- keras_model_sequential()
model_cnn %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = "sigmoid", 
                input_shape = c(28,28,1)) %>%
  layer_max_pooling_2d(pool_size = c(3,3)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 15, activation = "sigmoid") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units  = 10, activation = "softmax")

model_cnn %>% compile(
  loss = "categorical_crossentropy", 
  optimizer = "adam", 
  metrics = "accuracy"
)


model_cnn %>% fit(
  train_x_cnn, y_train, 
  epoch =  5,
  batch_size = 128, 
  validatio_split  = 0.2
)
