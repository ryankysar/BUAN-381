######## Housing Prices ####

#### Variable to Predict: Price of Houses ######

# Libraries to Load 

library(tidymodels)
library(sqldf)
library(splitTools)
library(rpart.plot)
library(baguette)
library(caret)

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

# Checking Classes

class(Price)
Price <- as.numeric(Price)
class(Total_Area)
Total_Area <- as.numeric(Total_Area)


# Find the correlation of the variables ******************

cor(Price, Total_Area)

plot(Price, Total_Area)


# Partitioning the data into Training, Testing, and Holdout ##

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



# Verify a strong relaionship between variables

M1 <- lm(Price ~ Total_Area, Train)
summary(M1)

Pred_1_IN <- predict(M1, Train)
Pred_1_IN
M1$fitted.values

Pred_1_OUT <- predict(M1, Test)

(RMSE_1_IN<-sqrt(sum((Pred_1_IN-Train$Price)^2)/length(Pred_1_IN))) #computes in-sample error
(RMSE_1_OUT<-sqrt(sum((Pred_1_OUT-Test$Price)^2)/length(Pred_1_OUT))) #computes out-of-sample 


# Did you do non-linear transformations on your variables??
# squaring and cubing them?

M2 <- lm(Price ~ Total_Area + Total_Area2, Train)
summary(M2)

Pred_2_IN <- predict(M2, Train)

Pred_2_OUT <- predict(M2, Test)

(RMSE_2_IN<-sqrt(sum((Pred_2_IN-Train$Price)^2)/length(Pred_2_IN))) #computes in-sample error
(RMSE_2_OUT<-sqrt(sum((Pred_2_OUT-Test$Price)^2)/length(Pred_2_OUT))) #computes out-of-sample 


M3 <- lm(Price ~ Total_Area + Total_Area2 + Total_Area3, Train)

Pred_3_IN <- predict(M3, Train)
Pred_3_OUT <- predict(M3, Test)


(RMSE_3_IN<-sqrt(sum((Pred_3_IN-Train$Price)^2)/length(Pred_3_IN))) #computes in-sample error
(RMSE_3_OUT<-sqrt(sum((Pred_3_OUT-Test$Price)^2)/length(Pred_3_OUT))) #computes out-of-sample 


M4 <- lm(Price ~ Total_Area + Total_Area2 + Total_Area3 + Total_Area4, Train)

Pred_4_IN <- predict(M4, Train)
Pred_4_OUT<- predict(M4, Test)

(RMSE_4_IN<-sqrt(sum((Pred_4_IN-Train$Price)^2)/length(Pred_4_IN))) #computes in-sample error
(RMSE_4_OUT<-sqrt(sum((Pred_4_OUT-Test$Price)^2)/length(Pred_4_OUT))) #computes out-of-sample 


# Model Comparison

TABLE_VAL_LM <- as.table(matrix(c(RMSE_1_IN, RMSE_2_IN, RMSE_3_IN, RMSE_4_IN, RMSE_1_OUT, RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_VAL_LM) <- c('LINEAR', 'QUADRATIC', 'CUBIC', '4th ORDER')
rownames(TABLE_VAL_LM) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_LM #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS








###################### STOPPING POINT ####################







Prices_reg <- decision_tree(min_n = 20 , #minimum number of observations for split
              tree_depth = 30, #max tree depth
              cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("regression")

print(Prices_reg)

#Esimating the Model

Prices_fmla <- Price ~ .
Prices_tree <- Prices_reg %>%
  fit(formula = Prices_fmla, data = Train)
print(Prices_tree)


# Predicting the Prices
#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_prices <- predict(Prices_tree, new_data = Test) %>%
  bind_cols(Test) 

#GENERATE CONFUSION MATRIX AND DIAGNOSTICS
confusion <- table(pred_prices$.pred, pred_prices$Price)


rmse(pred_prices, estimate=.pred, truth=Price)




#VISUALIZING THE REGRESSION TREE
Prices_tree$fit %>%
  rpart.plot(type = 2, roundint = FALSE)

#change the numbers out of scientific notation
