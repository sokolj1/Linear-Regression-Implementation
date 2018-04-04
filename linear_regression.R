# Author: John Sokol
# Machine Learning in R
# 25 February 2018

# Linear Regression implementation

# Algorithm steps: 
# 1. Fit the model by minimize the sum of squared errors (SSE or RSS)
# 2. Utilize matrix algebra to accomplish task 1: b_hat <- solve(t(X) * X) * t(X) * y_train
# 3. b_hat is now considered the "weights" or beta to predict the test regression values

# Test the model: 
# 4. Matrix multiply the x_test features by b_hat to obtain regression prediction values
# 5. Calculate model prediction accuracy by comparing coefficients to built in lm(~) function and 
# and the mulitple and adjusted R^2 values. 

# setwd("./Desktop")

# Kaggle training and test datasets
# train_linreg <- read.csv("train_linreg.csv", sep = ",")
# test_linreg <- read.csv("test_linreg.csv", sep = ",")

# ** Please disregard: Used this code for Kaggle dataset **
# check for NA values
# numberOfNA = length(which(is.na(train_linreg) == TRUE))
# if(numberOfNA > 0) {
#   cat('Number of missing values found: ', numberOfNA)
#   cat('\nRemoving missing values...')
#   train_linreg = train_linreg[complete.cases(train_linreg), ]
# }

# import dataset
baldwin_data <- read.csv("baldwin_lingreg_train.csv", sep = ",")

# divide dataset into "train" and "test" data frames
indexes <- sample(1:nrow(baldwin_data), size=0.8*nrow(baldwin_data))
train_baldwin <- baldwin_data[indexes,]
test_baldwin <- baldwin_data[-indexes,]


# linear regression function. returns the following: 
# 1. b_hat coefficient that is fitted by train dataset for test dataset y predictions. 
# 2. multiple R squared: measures the amount of variation that can be explained 
#    by the predictor variables
# 3. adjusted R squared: modified version of R squared that accounts for the number 
#    of features in the model 
# 4. linreg_plot: ggplot of actual Y labels vs. predicted Y labels

# accepts two arguments:
# 1. train dataset organized such that X[1], X[2], X[3]... X[i], Y
# 2. test dataset organized such that X[1], X[2], X[3]... X[i], Y
lin_reg <- function(train, test) {
    
    # train model by determining OLS estimates of linear regression coefficients (b_hat)
    # store train dataset bias [,1] (set to 1 by default) and independent (x) values [,2]
    train_features <- as.matrix(cbind(1, train[, 1:ncol(train) - 1]))
    
    # store train dataset dependent prediction (y) values 
    train_y <- as.matrix(train[,ncol(train)])
    
    # calculate the B value weights for each independent x value 
    # solve(a,b) function solves the equation a %*% x = b, where b can be either a vector or a matrix;
    # so this function executes inverse matrix multiplication. When solve.default is run for function documentation,
    # solve() actually performs WR decomposition to solve inverse matrix linear algebra 
    b_hat <- as.matrix(solve(t(train_features) %*% train_features) %*% t(train_features) %*% train_y)
    
    # test model by computing predictions for y with test dataset
    # test dataset bias [,1] (set to 1 by default) and independent (x) values [,2]
    test_features <- as.matrix(cbind(1, test[, 1:ncol(test) - 1]))
    
    # test dataset dependent prediction (y) values 
    test_y <- as.matrix(test[,ncol(test)])
    
    # computes test dependent variable prediction from B, which was determined from training data
    y_prediction <- test_features %*% b_hat
    
    # calculate the residual sum of squared errors 
    # initialize rss to 0
    rss <- 0
    for (i in 1:nrow(test)){
        rss <- rss + (test_y[i] - y_prediction[i])^2
    }
    
    # caluclate the total sum of squared errors
    # initialize tss to 0
    tss <- 0 
    
    # mean prediction value for tss calculation
    y_prediction_mean <- mean(y_prediction)
    
    for (i in 1:nrow(test)){
        tss <- tss + (test_y[i] - y_prediction_mean)^2
    }
    
    # compute multiple r squared value 
    multi_r_squared <- 1 - (rss / tss)
    
    # compute adjusted r squared value
    adj_r_squared <- 1 - (((rss / (nrow(train) - (ncol(train) - 1) - 1))) / ((tss / nrow(train) - 1)))
    
    # concatenated dataframe of the acutal y labels and the predicted y labels
    y_dataframe <- data.frame(test_y, y_prediction)
    
    linreg_plot <- ggplot(y_dataframe, aes(x = test_y,  y = y_prediction)) + xlab("Actual Y labels") + 
    ylab("Predicted Y labels") + ggtitle("Linear Regression Implementation") + geom_point()

    # returns linear regression coefficients and r squared value for test dataset
    return(list(coefficients = b_hat, multi_r_squared = multi_r_squared, adj_r_squared = adj_r_squared, linreg_plot = linreg_plot))
}

# call implemented linear regression function 
lin_reg_instan <- lin_reg(train_baldwin, test_baldwin)

# coefficients 
lin_reg_instan$coefficients

# mulitple R squared
lin_reg_instan$multi_r_squared

# adjusted R squared
lin_reg_instan$adj_r_squared

# ggplot of implemented linear regression function
lin_reg_instan$linreg_plot

# call built in linear regression function 
summary(lm(test_baldwin$Y ~ test_baldwin$X1 + test_baldwin$X2 + test_baldwin$X3 + test_baldwin$X4 + test_baldwin$X5))
