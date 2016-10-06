##############################################
# CMPT 459 Programming Assignment 2
# Ivy Tse
##############################################

#***** QUESTION 1 *****#
titanic <- read.csv("titanic.csv")
titanic[titanic==""] <- NA # set blank values to NA
set.seed(1)
data <- sort(sample(nrow(titanic), nrow(titanic)*0.8))
training <- titanic[data,] # 80% training data
test <- titanic[-data,] # 20% test data
 
#***** QUESTION 2 *****#
colSums(is.na(training)) # number of missing values in training
colSums(is.na(test))# number of missing values in test

#***** QUESTION 3 *****#
#(a) Insert the mean age for all missing ages in the column

#(b) Delete the rows of the missing Embarked values since there 
#    are only 2 rows with that missing value

#(c) Delete the entire Cabin column since there are two many 
#    missing values

#(d) There are no missing attributes with the Name and PassengerID.

training$Age[is.na(training$Age)] <- mean(training$Age, na.rm=TRUE)
test$Age[is.na(test$Age)] <- mean(test$Age, na.rm=TRUE)

training <- training[!is.na(training$Embarked),]
test <- test[!is.na(test$Embarked),]

training$Cabin <- NULL
test$Cabin <- NULL

#***** QUESTION 4 *****#
# Use logistic regression based on "Survival" attribute.
# "PassengerID", "Name", and "Ticket" attributes are not used in the
# logistic regression since all those values are unique.
new_training <- training[c("Survived", "Pclass", "Sex", "Age", "SibSp",
"Parch", "Fare", "Embarked")]

new_test <- test[c("Survived", "Pclass", "Sex", "Age", "SibSp","Parch", 
"Fare", "Embarked")]

logistic_reg <- glm(Survived ~.,family=binomial(link="logit"),new_training)

summary(logistic_reg)

# Based on the coefficient rankings from the summary, the three most
# significant attributes are "Pclass", "Sex", and "Age".

#***** QUESTION 5 *****#
test$Prediction <- round(predict(logistic_reg, new_test, type="response"))

Actual_Results <- test$Survived
Predicted_Results <- test$Prediction
confusion_matrix <- table(Actual_Results,Predicted_Results)
confusion_matrix
plot(confusion_matrix)

mean(Actual_Results == Predicted_Results) # accuracy of model = 0.7988827

#***** QUESTION 6 *****#
library(ROCR)
predicted <- predict(logistic_reg, new_test, type="response")
predicted_real <- prediction(predicted, test$Survived)
perf <- performance(predicted_real, measure="tpr", x.measure="fpr")
plot(perf)

auc <- performance(predicted_real, measure = "auc")
auc@y.values[[1]] # auc of regression model is 0.8417511

#***** QUESTION 7 *****#
library('e1071')

linear_tune <- tune(svm, Survived ~., data=new_test, kernel="linear")
radial_tune <- tune(svm, Survived ~., data=new_test, kernel="radial")
polynomial_tune <- tune(svm, Survived ~., data=new_test, kernel="polynomial")
sigmoid_tune <- tune(svm, Survived ~., data=new_test, kernel="sigmoid")

Results <- c()
Results$SVM_linear <- round(predict(linear_tune$best.model, new_test))
Results$SVM_radial <- round(predict(radial_tune$best.model, new_test))
Results$SVM_polynomial <- round(predict(polynomial_tune$best.model, new_test))
Results$SVM_sigmoid <- round(predict(sigmoid_tune$best.model, new_test))

SVM_linear_results <- Results$SVM_linear
SVM_radial_results <- Results$SVM_radial
SVM_polynomial_results <- Results$SVM_polynomial
SVM_sigmoid_results <- Results$SVM_sigmoid

mean(Actual_Results == SVM_linear_results)
mean(Actual_Results == SVM_radial_results)

radial_tune$best.parameters

# For the linear kernels, predictions are not as accurate as the radial
# kernels. This effect happens because the the radial kernel is has a better
# fit for the test data. The best parameter for the radial kernal is with a
# dummy parameter of 0.

#***** QUESTION 8 *****#
mean(Actual_Results == SVM_linear_results) # accuracy = 0.7932961 
mean(Actual_Results == SVM_radial_results) # accuracy = 0.8324022
mean(Actual_Results == SVM_polynomial_results) # accuracy = 0.7486034
mean(Actual_Results == SVM_sigmoid_results) # accuracy = 0.7486034

# Based on the accuracy results of the 4 models, the one with the best result
# is when kernel = radial

radial_confusion <- table(Actual_Results, SVM_radial_results)
radial_confusion
plot(radial_confusion)

#***** QUESTION 9 *****#
radial_pred <- predict(radial_tune$best.model, new_test, type="responce")
radial_pred_real <- prediction(radial_pred, test$Survived)
radial_perf <- performance(radial_pred_real, measure="tpr", x.measure="fpr")
plot(radial_perf)

auc_radial <- performance(radial_pred_real, measure = "auc")
auc_radial@y.values[[1]] # auc of regression model is 0.8709355

# The AUC of my SVM method is about 0.87, which is higher than the AUC for
# the logistic regression method (rather than lower). The classification for
# my chosen SVM model is also higher than the classification accuracy of the
# logistic regression model. The accuracy of my chosen SVM model is about 
# 0.83 whereas the accuracy of the logistic regression model is about 0.80

