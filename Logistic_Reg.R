######## Housing Prices ####

#CLASSIFICATION TASK ON LOGISTIC REGRESSION

#### Variable to Predict: Basement (binary) ######

# Libraries to Load 

library(tidymodels)
library(sqldf)
library(dplyr)
library(splitTools)
library(rpart.plot)
library(baguette)
library(caret)
library(yardstick)
library(e1071) #SVM LIBRARY
library(randomForest)
library(pROC)
install.packages('pROC')

# Import Data

Prices <- read.csv('https://raw.githubusercontent.com/ryankysar/BUAN-381/main/Housing_Price_Data.csv')

View(Prices)


# Check for NA Values

sum(is.na(Prices))

# Summary of the Data

summary(Prices)



####################### Cleaning ########################

# Rename the Columns

colnames(Prices)[colnames(Prices)=="price"] <- "Price"

colnames(Prices)[colnames(Prices)=="area"] <- "Total_Area"

colnames(Prices)[colnames(Prices)=="bedrooms"] <- "Bedrooms"

colnames(Prices)[colnames(Prices)=="stories"] <- "Stories"

colnames(Prices)[colnames(Prices)=="mainroad"] <- "Mainroad"

colnames(Prices)[colnames(Prices)=="guestroom"] <- "GuestRoom"

colnames(Prices)[colnames(Prices)=="basement"] <- "Basement"

colnames(Prices)[colnames(Prices)=="hotwaterheating"] <- "HotWaterHeating"

colnames(Prices)[colnames(Prices)=="airconditioning"] <- "AC"

colnames(Prices)[colnames(Prices)=="parking"] <- "Parking_Spots"

colnames(Prices)[colnames(Prices)=="prefarea"] <- "PrefArea"

colnames(Prices)[colnames(Prices)=="furnishingstatus"] <- "Furnished"

colnames(Prices)[colnames(Prices)=="bathrooms"] <- "Bathrooms"


#### Adding Non-Linear Transformations to the Data

Prices$Total_Area2 <- Prices$Total_Area^2 #Quadratic Transformation
Prices$Total_Area3 <- Prices$Total_Area^3 #Cubic Transformation
Prices$Total_Area4 <- Prices$Total_Area^4 #Fourth Order Transformation


# Changing Variable Furnished to be a Factor

class(Prices$Furnished)

Prices$Furnished <- as.factor(Prices$Furnished)

levels(Prices$Furnished)


# Creating Binary Columns for Yes & No Answers, 1 = Yes, 0 = No 

Prices$Mainroad[Prices$Mainroad == "no"] <- 0

Prices$Mainroad[Prices$Mainroad == "yes"] <- 1

Prices$GuestRoom[Prices$GuestRoom == "no"] <- 0

Prices$GuestRoom[Prices$GuestRoom == "yes"] <- 1

Prices$Basement[Prices$Basement == "no"] <- 0

Prices$Basement[Prices$Basement == "yes"] <- 1

Prices$HotWaterHeating[Prices$HotWaterHeating == "no"] <- 0

Prices$HotWaterHeating[Prices$HotWaterHeating == "yes"] <- 1

Prices$AC[Prices$AC == "no"] <- 0

Prices$AC[Prices$AC == "yes"] <- 1

Prices$PrefArea[Prices$PrefArea == "no"] <- 0

Prices$PrefArea[Prices$PrefArea == "yes"] <- 1




## Converting Binary variables to factors

Prices$Basement <- as.factor(Prices$Basement)

Prices$Stories <- as.numeric(Prices$Stories)

Prices$AC <- as.numeric(Prices$AC)

Prices$Parking_Spots <- as.numeric(Prices$Parking_Spots)

Prices$PrefArea <- as.numeric(Prices$PrefArea)

Prices$GuestRoom <- as.numeric(Prices$GuestRoom)

Prices$Bedrooms <- as.numeric(Prices$Bedrooms)

Prices$Total_Area <- as.numeric(Prices$Total_Area)

Prices$HotWaterHeating <- as.numeric(Prices$HotWaterHeating)

Prices$Mainroad <- as.numeric(Prices$Mainroad)

Prices$Bathrooms <- as.numeric(Prices$Bathrooms)

Prices1 <- Prices[, c("Price", "Total_Area", "Bedrooms", "Bathrooms", "AC", "HotWaterHeating", "GuestRoom", "Parking_Spots", "Basement", "PrefArea", "Stories", "Mainroad")]



###################### Data Partitioning ###########################

# Partitioning the data into Training, Testing, and Holdout #

#Split the data frame into partitions 
set.seed(678)
Sets <- partition(Price, p = c(train = 0.70, Hold = 0.15, test = 0.15))
str(Sets)


# Setting the seed so it reproduces the same results
set.seed(678)
Train <- Prices1[Sets$train,]
Hold <- Prices1[Sets$Hold,]
Test <- Prices1[Sets$test,]


dim(Train)
dim(Hold)
dim(Test)

summary(Train$Basement)
summary(Hold$Basement)
summary(Test$Basement)



#################### Single Classification Model ###########################

#### Variable to Predict: Basement (binary) ######


kern_type<-"polynomial" #SPECIFY KERNEL TYPE

######################### Support Vector Machine ##############################


#BUILD SVM CLASSIFIER
SVM_Model_1<- svm(Basement ~ ., 
                data = Train, 
                type = "C-classification", #set to "eps-regression" for numeric prediction
                kernel = kern_type,
                cost=100,                   #REGULARIZATION PARAMETER
                gamma = 1/(ncol(Train)-1), #DEFAULT KERNEL PARAMETER
                coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model_1) #DIAGNOSTIC SUMMARY


# Pre tuning Predictions

# Make predictions on the testing data
predictions_test_svm <- predict(SVM_Model_1, Test)

# Create confusion matrix for out-of-sample predictions
confusion_test_svm <- table(predictions_test_svm, Test$Basement)

confusionMatrix(confusion_test_svm)


# Make predictions on the training data
predictions_train_svm <- predict(SVM_Model_1, Train)

# Create confusion matrix for in-sample predictions
confusion_train_svm <- table(predictions_train_svm, Train$Basement)


confusionMatrix(confusion_train_svm)


SVM_OUT <- .381
SVM_IN <- .3481


######################## TUNING SVM MODEL #############################

#TUNING THE SVM BY CROSS-VALIDATION
tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(10) 
TUNE_SVM_Model_1 <- tune.svm(x = Train[,-9],
                 y = Train[,9],
                 type = "C-classification",
                 kernel = kern_type,
                 tunecontrol=tune_control,
                 cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                 gamma = 1/(ncol(Train)-1), #KERNEL PARAMETER
                 coef0 = 0,           #KERNEL PARAMETER
                 degree = 2)          #POLYNOMIAL KERNEL PARAMETER

print(TUNE_SVM_Model_1) #OPTIMAL TUNING PARAMETERS FROM VALIDATION PROCEDURE


#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
SVM_Retune_1<- svm(Basement ~ ., 
                 data = Train, 
                 type = "C-classification", 
                 kernel = kern_type,
                 degree = TUNE_SVM_Model_1$best.parameters$degree,
                 gamma = TUNE_SVM_Model_1$best.parameters$gamma,
                 coef0 = TUNE_SVM_Model_1$best.parameters$coef0,
                 cost = TUNE_SVM_Model_1$best.parameters$cost,
                 scale = FALSE)

print(SVM_Retune_1) #DIAGNOSTIC SUMMARY


# Make predictions on the testing data
predictions_test_svm_tune <- predict(SVM_Retune_1, Test)

# Create confusion matrix for out-of-sample predictions
confusion_test_svm_tune <- table(predictions_test_svm_tune, Test$Basement)

confusionMatrix(confusion_test_svm_tune)


# Make predictions on the training data
predictions_train_svm_tune <- predict(SVM_Retune_1, Train)

# Create confusion matrix for in-sample predictions
confusion_train_svm_tune <- table(predictions_train_svm_tune, Train$Basement)


confusionMatrix(confusion_train_svm_tune)


SVM_OUT_TUNE <- .4048
SVM_IN_TUNE <- .3727




################ PREDICTION W/ LOGISTIC REGRESSION ####################


#build the model with the training partition
Model_LOG<-glm(Basement ~ Total_Area + Bedrooms + Stories + Mainroad + GuestRoom + HotWaterHeating + AC + Parking_Spots + PrefArea,
               data = Train, family = binomial(link="logit"))
summary(Model_LOG)

#### STORIES, BEDROOMS, AND GUESTROOM ARE STATISTCALLY SIGNIFICANT

#takes the coefficients to the base e for odds-ratio interpretation
exp(cbind(Model_LOG$coefficients, confint(Model_LOG)))

#generating predicted probabilities
Predictions<-predict(Model_LOG, Train, type="response")


#converts predictions to boolean TRUE (1) or FALSE (0) based on 1/2 threshold on output probability
Binpredict <- (Predictions >= .5)
View(Binpredict)



#build confusion matrix based on binary prediction in-sample
Confusion<-table(Binpredict, Train$Basement == 1)
Confusion


################### Accuracy From Logistic Regression #################

#builds the confusion matrix to look at accuracy on testing data out-of-sample
confusionMatrix(table(predict(Model_LOG, Test, type="response") >= 0.5,
                      Test$Basement == 1), positive = 'TRUE')

#builds the confusion matrix to look at accuracy on testing data out-of-sample
confusionMatrix(table(predict(Model_LOG, Train, type="response") >= 0.5,
                      Train$Basement == 1), positive = 'TRUE')


LOG_OUT <- .7619
LOG_IN <- .7638



######################### Probit Model #################################


# Build the model with the training partition
Model_PROBIT <- glm(Basement ~ Total_Area + Bedrooms + Stories + Mainroad + GuestRoom + HotWaterHeating + AC + Parking_Spots + PrefArea,
                    data = Train, family = binomial(link = "probit"))
summary(Model_PROBIT)

# Display coefficients and their odds ratio interpretation
exp(cbind(Model_PROBIT$coefficients, confint(Model_PROBIT)))

# Generating predicted probabilities
Predictions_PROBIT <- predict(Model_PROBIT, Train, type = "response")

# Convert predictions to binary (TRUE or FALSE) based on 0.5 threshold on output probability
Binpredict_PROBIT <- (Predictions_PROBIT >= 0.5)

# Build confusion matrix based on binary prediction in-sample
Confusion_PROBIT <- table(Binpredict_PROBIT, Train$Basement == 1)
Confusion_PROBIT

# Display summary analysis of confusion matrix in-sample
confusionMatrix(Confusion_PROBIT, positive = 'TRUE')

################ Accuracy From Probit Regression ##########

# Accuracy from Probit Regression on testing data out-of-sample
confusionMatrix(table(predict(Model_PROBIT, Test, type = "response") >= 0.5,
                      Test$Basement == 1), positive = 'TRUE')

# Accuracy from Probit Regression on training data out-of-sample
confusionMatrix(table(predict(Model_PROBIT, Train, type = "response") >= 0.5,
                      Train$Basement == 1), positive = 'TRUE')


PROB_IN <- .7638
PROB_OUT <- .7619


##################### Classification Tree ###################


# Build the classification tree
Tree_Model <- rpart(factor(Basement) ~ Total_Area + Bedrooms + Stories + Mainroad + GuestRoom + HotWaterHeating + AC + Parking_Spots + PrefArea,
                        data = Train, method = "class")

# Plot the classification tree
rpart.plot(Tree_Model, extra = 101, under = TRUE, cex = 0.8)


# CART MODEL
#SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec)

class_fmla <- Basement ~ .

class_tree <- class_spec %>%
  fit(formula = class_fmla, data = Train)
print(class_tree)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree$fit)

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_cart <- predict(class_tree, new_data = Test, type="class") %>%
  bind_cols(Test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Generate in-sample predictions on the training set and combine with training data
pred_cart_train <- predict(class_tree, new_data = Train, type = "class") %>%
  bind_cols(Train) # Add class predictions directly to train data

pred_prob <- predict(class_tree, new_data = Test, type="prob") %>%
  bind_cols(Test) #ADD PROBABILITY PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE CONFUSION MATRIX AND DIAGNOSTICS
cart_confusion <- table(pred_cart$.pred_class, pred_cart$Basement)
confusionMatrix(cart_confusion, positive='1') #FROM CARET PACKAGE

# Generate confusion matrix and diagnostics for in-sample predictions
cart_confusion_train <- table(pred_cart_train$.pred_class, pred_cart_train$Basement)
confusionMatrix(cart_confusion_train, positive = '1') # From caret package

CART_IN <- .7979
CART_OUT <- .6786

##############################################################################
########################## Create an ROC Curve ##########################
##############################################################################

# Calculate ROC AUC
roc_curve_data <- roc(Basement ~ pred_prob, data = data.frame(pred_prob, Basement))
roc_auc <- auc(roc_curve_data)

# Plot ROC curve
autoplot(roc_curve_data)


# Print ROC AUC
print(auc(roc_auc))


#######################################################################
############################### TUNE THIS MODEL ########################## 
################################## #################### #################### 




##################################### Random Forest #########################################

# Define predictor variables
predictor_rand <- c("Total_Area", "Bedrooms", "Stories", "Mainroad", "GuestRoom", "HotWaterHeating", "AC", "Parking_Spots", "PrefArea")

# Build the Random Forest model
Model_RF <- randomForest(factor(Basement) ~ .,
                         data = Train[, c(predictor_rand, "Basement")],
                         ntree = 500, mtry = sqrt(length(predictor_rand)))

# Print the model summary
print(Model_RF)


# Make predictions on training and testing datasets
predictions_train_rand <- as.numeric(as.character(predict(Model_RF, Train[, predictor_rand], type = "response")))
predictions_test_rand <- as.numeric(as.character(predict(Model_RF, Test[, predictor_rand], type = "response")))

# Convert predicted probabilities to binary predictions (0 or 1) using a threshold of 0.5
binary_predictions_train <- ifelse(predictions_train_rand > 0.5, 1, 0)
binary_predictions_test <- ifelse(predictions_test_rand > 0.5, 1, 0)

# Create confusion matrices
confusion_train <- table(binary_predictions_train, Train$Basement)
confusion_test <- table(binary_predictions_test, Test$Basement)

print(confusion_train)


print(confusion_test)


confusionMatrix(confusion_train) #FROM CARET PACKAGE

confusionMatrix(confusion_test)



RAND_IN <- .9213
RAND_OUT <- .6667


# CREATE A TABLE

RAND_IN
RAND_OUT
SVM_IN
SVM_OUT
SVM_IN_TUNE
SVM_OUT_TUNE
LOG_IN
LOG_OUT
PROB_IN
PROB_OUT

# Create a matrix with the provided data
TABLE_VAL <- matrix(c(RAND_IN, RAND_OUT, SVM_IN, SVM_OUT, SVM_IN_TUNE, SVM_OUT_TUNE, LOG_IN, LOG_OUT, PROB_IN, PROB_OUT, CART_IN, CART_OUT),
                    ncol = 2, byrow = TRUE)

# Set column names and row names
colnames(TABLE_VAL) <- c('IN', 'OUT')
rownames(TABLE_VAL) <- c('Random Forest', 'SVM', 'SVM (Tuned)', 'Logistic Regression', 'Probit Regression', "CART")

# Print the table
print(TABLE_VAL)


###############################################################################
################# Multi Class Prediction Variable #########################
################################################################################

####### Variable: Furnished

class(Prices$Parking_Spots)

Train$Parking_Spots <- as.factor(Train$Parking_Spots)

Test$Parking_Spots <- as.factor(Test$Parking_Spots)

Hold$Parking_Spots <- as.factor(Hold$Parking_Spots)

Train$Basement <- as.numeric(Train$Parking_Spots)

Test$Basement <- as.numeric(Test$Basement)

Hold$Basement <- as.numeric(Hold$Basement)


#BUILD SVM CLASSIFIER
SVM_Model_2<- svm(Parking_Spots ~ ., 
                  data = Train, 
                  type = "C-classification", #set to "eps-regression" for numeric prediction
                  kernel = kern_type,
                  cost=100,                   #REGULARIZATION PARAMETER
                  gamma = 1/(ncol(Train)-1), #DEFAULT KERNEL PARAMETER
                  coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                  degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                  scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model_2) #DIAGNOSTIC SUMMARY


# Pre tuning Predictions

# Make predictions on the testing data
predictions_test_svm_2 <- predict(SVM_Model_2, Test)

# Create confusion matrix for out-of-sample predictions
confusion_test_svm_2 <- table(predictions_test_svm_2, Test$Parking_Spots)

confusionMatrix(confusion_test_svm_2)


# Make predictions on the training data
predictions_train_svm_2 <- predict(SVM_Model_2, Train)

# Create confusion matrix for in-sample predictions
confusion_train_svm_2 <- table(predictions_train_svm_2, Train$Parking_Spots)


confusionMatrix(confusion_train_svm_2)


SVM_M_OUT <- .5714
SVM_M_IN <- .5827


######################## TUNING SVM MODEL #############################

#TUNING THE SVM BY CROSS-VALIDATION
tune_control<-tune.control(cross=10) #SET K-FOLD CV PARAMETERS
set.seed(10) 
TUNE_SVM_Model_2 <- tune.svm(x = Train[,-8],
                             y = Train[,8],
                             type = "C-classification",
                             kernel = kern_type,
                             tunecontrol=tune_control,
                             cost=c(.01, .1, 1, 10, 100, 1000), #REGULARIZATION PARAMETER
                             gamma = 1/(ncol(Train)-1), #KERNEL PARAMETER
                             coef0 = 0,           #KERNEL PARAMETER
                             degree = 2)          #POLYNOMIAL KERNEL PARAMETER

print(TUNE_SVM_Model_2) #OPTIMAL TUNING PARAMETERS FROM VALIDATION PROCEDURE


#RE-BUILD MODEL USING OPTIMAL TUNING PARAMETERS
SVM_Retune_2<- svm(Parking_Spots ~ ., 
                   data = Train, 
                   type = "C-classification", 
                   kernel = kern_type,
                   degree = TUNE_SVM_Model_1$best.parameters$degree,
                   gamma = TUNE_SVM_Model_1$best.parameters$gamma,
                   coef0 = TUNE_SVM_Model_1$best.parameters$coef0,
                   cost = TUNE_SVM_Model_1$best.parameters$cost,
                   scale = FALSE)

print(SVM_Retune_2) #DIAGNOSTIC SUMMARY


# Make predictions on the testing data
predictions_test_svm_tune_2 <- predict(SVM_Retune_2, Test)

# Create confusion matrix for out-of-sample predictions
confusion_test_svm_tune_2 <- table(predictions_test_svm_tune_2, Test$Parking_Spots)

confusionMatrix(confusion_test_svm_tune_2)


# Make predictions on the training data
predictions_train_svm_tune_2 <- predict(SVM_Retune_2, Train)

# Create confusion matrix for in-sample predictions
confusion_train_svm_tune_2 <- table(predictions_train_svm_tune_2, Train$Parking_Spots)


confusionMatrix(confusion_train_svm_tune_2)


SVM_OUT_M_TUNE <- .5476
SVM_IN_M_TUNE <- .5722




# CART MODEL
#SPECIFYING THE CLASSIFICATION TREE MODEL
class_spec_2 <- decision_tree(min_n = 20 , #minimum number of observations for split
                            tree_depth = 30, #max tree depth
                            cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("classification")
print(class_spec_2)

class_fmla_2 <- Parking_Spots ~ .

class_tree_2 <- class_spec_2 %>%
  fit(formula = class_fmla_2, data = Train)
print(class_tree_2)

#VISUALIZING THE CLASSIFICATION TREE MODEL:
class_tree_2$fit %>%
  rpart.plot(type = 4, extra = 2, roundint = FALSE)

plotcp(class_tree_2$fit)


#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_cart_2 <- predict(class_tree_2, new_data = Test, type="class") %>%
  bind_cols(Test) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Generate in-sample predictions on the training set and combine with training data
pred_cart_train_2 <- predict(class_tree_2, new_data = Train, type = "class") %>%
  bind_cols(Train) # Add class predictions directly to train data

pred_prob <- predict(class_tree_2, new_data = Test, type="prob") %>%
  bind_cols(Test) #ADD PROBABILITY PREDICTIONS DIRECTLY TO TEST DATA

#GENERATE CONFUSION MATRIX AND DIAGNOSTICS
cart_confusion_2 <- table(pred_cart_2$.pred_class, pred_cart_2$Parking_Spots)
confusionMatrix(cart_confusion_2, positive='1') #FROM CARET PACKAGE

# Generate confusion matrix and diagnostics for in-sample predictions
cart_confusion_train_2 <- table(pred_cart_train_2$.pred_class, pred_cart_train_2$Parking_Spots)
confusionMatrix(cart_confusion_train_2, positive = '1') # From caret package

CART_M_IN <- 1
CART_M_OUT <- .381


############################ TUNE THIS MODEL ##################################


#BUILD A CLASS TREE ##################


########################### Random Forest ###############################

rf_model_2 <- randomForest(Parking_Spots ~ ., data = Train, ntree = 500)


# Make predictions on the test data
predictions_rfmodel2 <- predict(rf_model_2, newdata = Test)

predictions_rfmodel2in <- predict(rf_model_2, newdata = Train)

confusion_rf <- table(predictions_rfmodel2, Test$Parking_Spots)

confusion_rf_in <- table(predictions_rfmodel2in, Train$Parking_Spots)

confusionMatrix(confusion_matrix_RF)

confusionMatrix(confusion_rf_in)



M_Rand_IN <- 1

M_RAND_OUT <- .381







M_RAND_OUT
M_Rand_IN
CART_M_IN
CART_M_OUT
SVM_IN_M_TUNE
SVM_OUT_M_TUNE
SVM_M_IN
SVM_M_OUT






