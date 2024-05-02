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


#### Adding Non-Linear Transformations to the Data

Prices$Total_Area2 <- Prices$Total_Area^2 #Quadratic Transformation
Prices$Total_Area3 <- Prices$Total_Area^3 #Cubic Transformation
Prices$Total_Area4 <- Prices$Total_Area^4 #Fourth Order Transformation



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

Prices$Stories <- as.factor(Prices$Stories)

Prices$AC <- as.factor(Prices$AC)

Prices$Parking_Spots <- as.numeric(Prices$Parking_Spots)

Prices$PrefArea <- as.factor(Prices$PrefArea)

Prices$GuestRoom <- as.factor(Prices$GuestRoom)

Prices$Bedrooms <- as.numeric(Prices$Bedrooms)

Prices$Total_Area <- as.numeric(Prices$Total_Area)

Prices$HotWaterHeating <- as.factor(Prices$HotWaterHeating)


#Split the data frame into partitions 
set.seed(123)
Sets <- partition(Price, p = c(train = 0.70, Hold = 0.15, test = 0.15))
str(Sets)


# Setting the seed so it reproduces the same results
set.seed(123)

Train <- Prices[Sets$train,]
Hold <- Prices[Sets$Hold,]
Test <- Prices[Sets$test,]


dim(Train)
dim(Hold)
dim(Test)

summary(Train$Price)
summary(Hold$Price)
summary(Test$Price)





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





