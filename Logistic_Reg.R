######## Housing Prices ####

#CLASSIFICATION TASK ON LOGISTIC REGRESSION

#### Variable to Predict: Basement (binary) ######

# Libraries to Load 

library(tidymodels)
library(sqldf)
library(splitTools)
library(rpart.plot)
library(baguette)
library(caret)
library(yardstick)
library(e1071) #SVM LIBRARY

# Import Data

Prices <- read.csv('https://raw.githubusercontent.com/ryankysar/BUAN-381/main/Housing_Price_Data.csv')

View(Prices)



summary(Prices)

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



# Variable Type

class(Prices$Furnished)

Prices$Furnished <- as.factor(Prices$Furnished)


# Creating Binary Columns for Yes No Answers, 1 = Yes, 0 = No 

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


 Prices1 <- Prices %>%
  select(Price, Total_Area, Bedrooms, Bathrooms, AC, HotWaterHeating, GuestRoom, Parking_Spots, Basement, PrefArea, Stories, Mainroad)


#Split the data frame into partitions 
set.seed(123)
split<-initial_split(Prices1, .7, strata=Basement) #CREATE THE SPLIT
Train<-training(split) #TRAINING PARTITION
Test<-testing(split) #test PARTITION




#VERIFY STRATIFIED SAMPLING YIELDS EQUALLY SKEWED PARTITIONS
mean(Train$Basement==1)
mean(Test$Basement==1)


kern_type<-"polynomial" #SPECIFY KERNEL TYPE

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


#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY)
(E_IN_M1<-1-mean(predict(SVM_Model_1, Train)==Train$Basement))
(E_OUT_M1<-1-mean(predict(SVM_Model_1, Test)==Test$Basement))



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
                 degree = TUNE$best.parameters$degree,
                 gamma = TUNE$best.parameters$gamma,
                 coef0 = TUNE$best.parameters$coef0,
                 cost = TUNE$best.parameters$cost,
                 scale = FALSE)

print(SVM_Retune_1) #DIAGNOSTIC SUMMARY


#REPORT IN AND OUT-OF-SAMPLE ERRORS (1-ACCURACY) ON RETUNED MODEL
(E_IN_RETUNE<-1-mean(predict(SVM_Retune_1, Train)==Train$Basement))
(E_OUT_RETUNE<-1-mean(predict(SVM_Retune_1, Test)==Test$Basement))

#SUMMARIZE RESULTS IN A TABLE:
TUNE_TABLE <- matrix(c(E_IN_M1, 
                       E_IN_RETUNE,
                       E_OUT_M1,
                       E_OUT_RETUNE),
                     ncol=2, 
                     byrow=TRUE)

colnames(TUNE_TABLE) <- c('UNTUNED', 'TUNED')
rownames(TUNE_TABLE) <- c('E_IN', 'E_OUT')
TUNE_TABLE #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS

################ 5/6/24 ###################




###PREDICTION W/ LOGISTIC REGRESSION###

sapply(Train, class)

sapply(Test, class)

#build the model with the training partition
Model_LOG<-glm(Basement ~ Total_Area + Bedrooms + Stories + Mainroad + GuestRoom + HotWaterHeating + AC + Parking_Spots + PrefArea,
               data = Train, family = binomial(link="logit"))
summary(Model_LOG)

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

#display summary analysis of confusion matrix in-sample
confusionMatrix(Confusion, positive='TRUE') #need to load the library e1071

#builds the confusion matrix to look at accuracy on testing data out-of-sample
confusionMatrix(table(predict(Model_LOG, Test, type="response") >= 0.5,
                      Test$Basement == 1), positive = 'TRUE')


#builds the confusion matrix to look at accuracy on valid data

confusionMatrix(table(predict(Model_LOG, Hold, type="response") >= 0.5,
                      Hold$Basement == 1), positive = 'TRUE')





