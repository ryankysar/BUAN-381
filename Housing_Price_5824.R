######## Housing Prices ####

#### Variable to Predict: Price of Houses ######

# Libraries to Load 

library(tidymodels)
library(sqldf)
library(splitTools)
library(rpart.plot)
library(baguette)
library(caret)
library(yardstick)
library(lmridge)
library(broom) 
library(MASS) 
library(ggplot2) 
library(caret) 
library(mgcv) 




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

class(Prices$Price)
Prices$Price <- as.numeric(Prices$Price)
class(Prices$Total_Area)
Prices$Total_Area <- as.numeric(Prices$Total_Area)


# Find the correlation of the variables ******************

correlation <- cor(Price, Total_Area)
correlation

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

# Bivariate Model using Training Data

M1 <- lm(Price ~ Total_Area, Train)
summary(M1)

Pred_1_IN <- predict(M1, Train)
Pred_1_IN
M1$fitted.values

Pred_1_OUT <- predict(M1, Test)

(RMSE_1_IN<-sqrt(sum((Pred_1_IN-Train$Price)^2)/length(Pred_1_IN))) #computes in-sample error
(RMSE_1_OUT<-sqrt(sum((Pred_1_OUT-Test$Price)^2)/length(Pred_1_OUT))) #computes out-of-sample 


#Quadratic model from training data


M2 <- lm(Price ~ Total_Area + Total_Area2, Train)
summary(M2)

Pred_2_IN <- predict(M2, Train)

Pred_2_OUT <- predict(M2, Test)

(RMSE_2_IN<-sqrt(sum((Pred_2_IN-Train$Price)^2)/length(Pred_2_IN))) #computes in-sample error
(RMSE_2_OUT<-sqrt(sum((Pred_2_OUT-Test$Price)^2)/length(Pred_2_OUT))) #computes out-of-sample 

#Cubic model from training data

M3 <- lm(Price ~ Total_Area + Total_Area2 + Total_Area3, Train)

Pred_3_IN <- predict(M3, Train)
Pred_3_OUT <- predict(M3, Test)


(RMSE_3_IN<-sqrt(sum((Pred_3_IN-Train$Price)^2)/length(Pred_3_IN))) #computes in-sample error
(RMSE_3_OUT<-sqrt(sum((Pred_3_OUT-Test$Price)^2)/length(Pred_3_OUT))) #computes out-of-sample 

#4th order polynomial from Training Data

M4 <- lm(Price ~ Total_Area + Total_Area2 + Total_Area3 + Total_Area4, Train)

Pred_4_IN <- predict(M4, Train)
Pred_4_OUT<- predict(M4, Test)

(RMSE_4_IN<-sqrt(sum((Pred_4_IN-Train$Price)^2)/length(Pred_4_IN))) #computes in-sample error
(RMSE_4_OUT<-sqrt(sum((Pred_4_OUT-Test$Price)^2)/length(Pred_4_OUT))) #computes out-of-sample 


#lm model against bivariate models with comparisons
TABLE_VAL_1 <- as.table(matrix(c(RMSE_1_IN, RMSE_2_IN, RMSE_3_IN, RMSE_4_IN, RMSE_1_OUT, RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_VAL_1) <- c('LINEAR', 'QUADRATIC', 'CUBIC', '4th ORDER')
rownames(TABLE_VAL_1) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_1 


#regularize the model 
reg_mod<-lm.ridge(Price ~ ., Prices, lambda=seq(0,.5,.01))
summary_reg <- tidy(reg_mod)
head(summary_reg, 11)


#Poisson Regression Model??


#Quasi Possion Regression + benchmarking against lm and bivariate
Model5 <- gam(Price ~ Total_Area, data = Train, family = 'quasipoisson')
summary(Model5)
PRED_5_IN <- predict(Model5, Train, type = 'response')
PRED_5_OUT <- predict(Model5, Test, type = 'response')

(RMSE_5_IN<-sqrt(sum((PRED_5_IN-Train$Price)^2)/length(PRED_5_IN)))  
(RMSE_5_OUT<-sqrt(sum((PRED_5_OUT-Test$Price)^2)/length(PRED_5_OUT))) 

#SPLINE
Model6 <- gam(Price ~ Total_Area, data = Train, family = 'gaussian')
summary(Model6)
PRED_6_IN <- predict(Model6, Train, type = 'response')
PRED_6_OUT <- predict(Model6, Test, type = 'response')

(RMSE_6_IN<-sqrt(sum((PRED_6_IN-Train$Price)^2)/length(PRED_6_IN)))  
(RMSE_6_OUT<-sqrt(sum((PRED_6_OUT-Test$Price)^2)/length(PRED_6_OUT))) 


#Model Comparisons
TABLE_VAL_2 <- as.table(matrix(c(RMSE_1_IN, RMSE_2_IN, RMSE_3_IN, RMSE_4_IN, RMSE_5_IN, RMSE_6_IN, RMSE_1_OUT, RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT, RMSE_5_IN, RMSE_6_OUT), ncol=6, byrow=TRUE))
colnames(TABLE_VAL_2) <- c('LINEAR', 'QUADRATIC', 'CUBIC', '4th ORDER','QUASIPOISSON', 'SPLINE')
rownames(TABLE_VAL_2) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_2 #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS


# Multivariate Model

#fmla DESCRIPTION:
fm <- Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots 

  
#### Bagged Tree Model
  
  spec_bagg <- bag_tree(min_n = 20 , #minimum number of observations for split
                          tree_depth = 30, #max tree depth
                          cost_complexity = 0.01, #regularization parameter
                          class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("regression") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=300) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagg


#FITTING THE MODEL
set.seed(456)
bagg_forest <- spec_bagg %>%
  fit(formula = fm, data = Train)
print(bagg_forest)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_class_in <- predict(bagg_forest, new_data = Train) %>%
  bind_cols(Train$Price) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Calculate RMSE
rmsee_in <- sqrt(mean((pred_class_in$...2 - pred_class_in$.pred)^2))
rmsee_in

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_class_out <- predict(bagg_forest, new_data = Test) %>%
  bind_cols(Test$Price) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Calculate RMSE
rmsee_out <- sqrt(mean((pred_class_out$...2 - pred_class_out$.pred)^2))
rmsee_out

#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE VALID SET AND COMBINE WITH VALID DATA
pred_class_Hold<- predict(bagg_forest, new_data = Hold) %>%
  bind_cols(Hold$Price) #ADD CLASS PREDICTIONS DIRECTLY TO VALID DATA

# Calculate RMSE
rmsee_v <- sqrt(mean((pred_class_Hold$...2 - pred_class_Hold$.pred)^2))
rmsee_v

#Tuning
spec_bagg <- bag_tree(min_n = tune() , #minimum number of observations for split
                        tree_depth = tune(), #max tree depth
                        cost_complexity = tune(), #regularization parameter
                        class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("regression") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=300) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagg

tree_grid <- grid_regular(hardhat::extract_parameter_set_dials(spec_bagg), levels = 3)


#TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
set.seed(456) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tuned_results <- tune_grid(spec_bagg,
                          fm, #MODEL FORMULA
                          resamples = vfold_cv(Train, v=3), #RESAMPLES / FOLDS
                          grid = tree_grid, #GRID
                          metrics = metric_set(yardstick::rmse)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_param <- select_best(tuned_results)

#FINALIZE THE MODEL SPECIFICATION
final_specc <- finalize_model(spec_bagg, best_param)


#FIT THE FINALIZED MODEL
final_modell <- final_specc %>% fit(fm, Train)

pred_class_inn <- predict(final_modell, new_data = Train) %>%
  bind_cols(Train$Price)

#RMSE Tuned
rmse_intunedd <- sqrt(mean((pred_class_inn$...2 - pred_class_inn$.pred)^2))
rmse_intunedd

pred_class_outt <- predict(final_modell, new_data = Test) %>%
  bind_cols(Test$Price)
rmse_outtunedd <- sqrt(mean((pred_class_outt$...2 - pred_class_outt$.pred)^2))
rmse_outtunedd

pred_class_hold <- predict(final_modell, new_data = Hold) %>%
  bind_cols(Hold$Price)
rmse_vtunedd <- sqrt(mean((pred_class_hold$...2 - pred_class_hold$.pred)^2))
rmse_vtunedd


rmsee_in
rmse_intunedd
rmsee_out
rmse_outtunedd
rmsee_v
rmse_vtunedd

rmsee_in - rmse_intunedd

rmsee_out - rmse_outtunedd

###################### STOPPING POINT ####################

