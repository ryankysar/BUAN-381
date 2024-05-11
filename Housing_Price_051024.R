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
library(mgcv) 
library(e1071)
library(glmnet)
install.packages('glmnet')



# Import Data

Prices <- read.csv('https://raw.githubusercontent.com/ryankysar/BUAN-381/main/Housing_Price_Data.csv')

View(Prices)



# Check for NA Values

sum(is.na(Prices))

# Summary of the Data

summary(Prices)

# Identifying the class of each variable

# Using sapply()
sapply(Prices, class)



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

#### Adding Non-Linear Transformations to the Data

Prices$Total_Area2 <- Prices$Total_Area^2 #Quadratic Transformation
Prices$Total_Area3 <- Prices$Total_Area^3 #Cubic Transformation
Prices$Total_Area4 <- Prices$Total_Area^4 #Fourth Order Transformation



# Checking Classes of the Variables

class(Prices$Price)
Prices$Price <- as.numeric(Prices$Price)
class(Prices$Total_Area)
Prices$Total_Area <- as.numeric(Prices$Total_Area)

# Making sure the variables used are numeric

Prices$Total_Area <- as.numeric(Prices$Total_Area)

Prices$Bedrooms <- as.numeric(Prices$Bedrooms)

Prices$Bathrooms <- as.numeric(Prices$Bathrooms)

Prices$Stories <- as.numeric(Prices$Stories)

Prices$Parking_Spots <- as.numeric(Prices$Parking_Spots)


############### Finding the Correlation Between Variables ##############

attach(Prices)

# Find the correlation of the variables

data <- data.frame(Price, Total_Area, Bedrooms, Bathrooms, Stories, Parking_Spots)


cm <- cor(data, use = "complete.obs")

print(cm)



# Create the heatmap (let me know if this runs for you, it wasnt running for me)
heatmap(cm, 
        Rowv = NA, 
        Colv = NA, 
        col = colorRampPalette(c("blue", "white", "red"))(20), 
        xlab = "Variables", 
        ylab = "Variables", 
        main = "Correlation Heatmap")



correlation_Area <- cor(Price, Total_Area)
correlation_Area

correlation_bath <- cor(Price, Bathrooms)
correlation_bath

correlation_stories <- cor(Price, Stories)
correlation_stories



# Create a scatter plot
PLOTS <- plot(Total_Area, Price, 
     xlab = "Total Area", 
     ylab = "Price",
     main = "Total Area vs Price",
     col = "red", 
     pch = 16, 
     cex = 1.0) 

# Add a trend line (linear regression)
fit <- lm(Price ~ Total_Area) 
abline(fit, col = "blue") 

library(ggplot2)

ggsave("Scatter.png",plot = PLOTS,   width = 5, height = 5)



###################### Data Partitioning ###########################
library(caret) 

# Partitioning the data into Training, Testing, and Holdout #

#Split the data frame into partitions 
set.seed(456)
Sets <- partition(Price, p = c(train = 0.70, Hold = 0.15, test = 0.15))
str(Sets)


# Setting the seed so it reproduces the same results
set.seed(456)
Train <- Prices[Sets$train,]
Hold <- Prices[Sets$Hold,]
Test <- Prices[Sets$test,]


dim(Train)
dim(Hold)
dim(Test)

summary(Train$Price)
summary(Hold$Price)
summary(Test$Price)



####################### Bivariate Model ###########################

# Bivariate Model using Training Data

M1 <- lm(Price ~ Total_Area, Train)
summary(M1)

PRED_1_IN <- predict(M1, Train)
PRED_1_IN

M1$fitted.values

PRED_1_OUT <- predict(M1, Test)

(RMSE_1_IN<-sqrt(sum((PRED_1_IN-Train$Price)^2)/length(PRED_1_IN))) #computes in-sample error
(RMSE_1_OUT<-sqrt(sum((PRED_1_OUT-Test$Price)^2)/length(PRED_1_OUT))) #computes out-of-sample 


#Quadratic model from training data

M2 <- lm(Price ~ Total_Area + Total_Area2, Train)
summary(M2)

PRED_2_IN <- predict(M2, Train)

PRED_2_OUT <- predict(M2, Test)

(RMSE_2_IN<-sqrt(sum((PRED_2_IN-Train$Price)^2)/length(PRED_2_IN))) #computes in-sample error
(RMSE_2_OUT<-sqrt(sum((PRED_2_OUT-Test$Price)^2)/length(PRED_2_OUT))) #computes out-of-sample 

#Cubic model from training data

M3 <- lm(Price ~ Total_Area + Total_Area2 + Total_Area3, Train)

PRED_3_IN <- predict(M3, Train)
PRED_3_OUT <- predict(M3, Test)


(RMSE_3_IN<-sqrt(sum((PRED_3_IN-Train$Price)^2)/length(PRED_3_IN))) #computes in-sample error
(RMSE_3_OUT<-sqrt(sum((PRED_3_OUT-Test$Price)^2)/length(PRED_3_OUT))) #computes out-of-sample 

#4th order polynomial from Training Data

M4 <- lm(Price ~ Total_Area + Total_Area2 + Total_Area3 + Total_Area4, Train)

PRED_4_IN <- predict(M4, Train)
PRED_4_OUT<- predict(M4, Test)

(RMSE_4_IN<-sqrt(sum((PRED_4_IN-Train$Price)^2)/length(PRED_4_IN))) #computes in-sample error
(RMSE_4_OUT<-sqrt(sum((PRED_4_OUT-Test$Price)^2)/length(PRED_4_OUT))) #computes out-of-sample 


#lm model against bivariate models with comparisons
TABLE_VAL_1 <- as.table(matrix(c(RMSE_1_IN, RMSE_2_IN, RMSE_3_IN, RMSE_4_IN, RMSE_1_OUT, RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT), ncol=4, byrow=TRUE))
colnames(TABLE_VAL_1) <- c('LINEAR', 'QUADRATIC', 'CUBIC', '4th ORDER')
rownames(TABLE_VAL_1) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_1 



################## Regularization ################################


#Regularization of Model 4

reg_mod<-lmridge(Price ~ Total_Area + Total_Area2 + Total_Area3 + Total_Area4, Train, lambda=seq(0,.5,.01))
summary(reg_mod)

PRED_reg_IN <- predict(reg_mod, Train)
PRED_reg_OUT<- predict(reg_mod, Test)

(RMSE_reg_IN<-sqrt(sum((PRED_reg_IN-Train$Price)^2)/length(PRED_reg_IN))) #computes in-sample error
(RMSE_reg_OUT<-sqrt(sum((PRED_reg_OUT-Test$Price)^2)/length(PRED_reg_OUT))) #computes out-of-sample 


##### Tuning Regularization: Finding the Best Lamda values

predictors <- c("Total_Area", "Total_Area2", "Total_Area3", "Total_Area4")
target <- "Price"

x_train <- as.matrix(Train[, predictors])
x_test <- as.matrix(Test[, predictors])
y_train <- Train[, target]
y_test <- Test[, target]

cv_model <- cv.glmnet(x_train, y_train)
plot(cv_model)

best_lambda <- cv_model$lambda.min
reg_mod <- glmnet(x_train, y_train, alpha = 0, lambda = best_lambda)

pred_train <- predict(reg_mod, s = best_lambda, newx = x_train)
pred_test <- predict(reg_mod, s = best_lambda, newx = x_test)
RMSE_reg_tune_IN <- sqrt(mean((pred_train - y_train)^2))
RMSE_reg_tune_OUT <- sqrt(mean((pred_test - y_test)^2))

cat("In-sample RMSE:", RMSE_reg_IN, "\n")
cat("Out-of-sample RMSE:", RMSE_reg_OUT, "\n")


#Quasi Possion Regression + benchmarking against lm and bivariate

M5 <- gam(Price ~ Total_Area, data = Train, family = 'quasipoisson')
summary(M5)

PRED_5_IN <- predict(M5, Train, type = 'response')
PRED_5_OUT <- predict(M5, Test, type = 'response')

(RMSE_5_IN<-sqrt(sum((PRED_5_IN-Train$Price)^2)/length(PRED_5_IN)))  
(RMSE_5_OUT<-sqrt(sum((PRED_5_OUT-Test$Price)^2)/length(PRED_5_OUT))) 


###############SPLINE
M6 <- gam(Price ~ Total_Area, data = Train, family = 'gaussian')
summary(M6)

PRED_6_IN <- predict(M6, Train, type = 'response')
PRED_6_OUT <- predict(M6, Test, type = 'response')

(RMSE_6_IN<-sqrt(sum((PRED_6_IN-Train$Price)^2)/length(PRED_6_IN)))  
(RMSE_6_OUT<-sqrt(sum((PRED_6_OUT-Test$Price)^2)/length(PRED_6_OUT))) 

#Apply LASSO regularization to the SPLINE Model
Spline_Mult <- gam(Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots, 
                   data = Train, 
                   family = 'gaussian',
                   method = "REML")
summary(Spline_Mult)

PRED_Mult_Spl_IN <- predict(Spline_Mult, Train, type = 'response')
PRED_Mult_Spl_OUT <- predict(Spline_Mult, Test, type = 'response')

(RMSE_SPl_MULT_IN <- sqrt(sum((PRED_Mult_Spl_IN - Train$Price)^2) / length(PRED_Mult_Spl_IN)))  
(RMSE_Spl_MULT_OUT <- sqrt(sum((PRED_Mult_Spl_OUT - Test$Price)^2) / length(PRED_Mult_Spl_OUT)))

#Model Comparisons
TABLE_VAL_2 <- as.table(matrix(c(RMSE_1_IN, RMSE_2_IN, RMSE_3_IN, RMSE_4_IN,RMSE_reg_IN, RMSE_5_IN, RMSE_6_IN,RMSE_reg_tune_IN, RMSE_1_OUT, RMSE_2_OUT, RMSE_3_OUT, RMSE_4_OUT, RMSE_reg_OUT, RMSE_5_OUT, RMSE_6_OUT, RMSE_reg_tune_OUT), ncol=8, byrow=TRUE))
colnames(TABLE_VAL_2) <- c('LINEAR', 'QUADRATIC', 'CUBIC', '4th ORDER', 'Regularized','QUASIPOISSON', 'SPLINE', 'Tuned Regularization')
rownames(TABLE_VAL_2) <- c('RMSE_IN', 'RMSE_OUT')
TABLE_VAL_2 #REPORT OUT-OF-SAMPLE ERRORS FOR BOTH HYPOTHESIS


##################################################################################
############### ###Creating a plot using the Predictions#####################
################## NEED TO DO #########################################
##################################################################################

Pred_1_IN
Pred_1_OUT
Pred_2_IN
Pred_2_OUT
Pred_3_IN
Pred_4_IN
Pred_4_OUT
PRED_5_IN
pred_train
pred_test
PRED_6_IN
PRED_6_OUT


# Combine actual prices and predicted prices into separate data frames for "IN" and "OUT"
predictions_IN <- data.frame(
  Actual = Train$Price,
  PRED_1_IN = PRED_1_IN,
  PRED_2_IN = PRED_2_IN,
  PRED_3_IN = PRED_3_IN,
  PRED_4_IN = PRED_4_IN,
  PRED_5_IN = PRED_5_IN,
  pred_train = pred_train,
  PRED_6_IN = PRED_6_IN
)


predictions_OUT <- data.frame(
  Actual = Test$Price,
  PRED_1_OUT = PRED_1_OUT,
  PRED_2_OUT = PRED_2_OUT,
  PRED_4_OUT = PRED_4_OUT,
  pred_test = pred_test,
  PRED_6_OUT = PRED_6_OUT
)




# Choosing the best out of Sample performance

# Using the Model 4, run with the hold set and report the RMSE

Final_biv_mod<-lm(Price ~ Total_Area + Total_Area2 + Total_Area3 + Total_Area4, Hold)


PRED_Final_BIV <- predict(M4, Hold)

(RMSE_FINAL_BIV<-sqrt(sum((PRED_Final_BIV-Hold$Price)^2)/length(PRED_Final_BIV))) 

### Result: 1,460,789





##########################################################################################
##########################################################################################
############################# Multivariate Model ################################
##########################################################################################
##########################################################################################

#fmla DESCRIPTION:
MULT_M1 <- lm(Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots, Train)


Pred_in_Mult_M1 <- predict(MULT_M1, Train, type = 'response')
Pred_out_Mult_M1 <- predict(MULT_M1, Test, type = 'response')


(RMSE_Mult_M1_IN<-sqrt(sum((Pred_in_Mult_M1-Train$Price)^2)/length(Pred_in_Mult_M1)))  
(RMSE_Mult_M1_OUT<-sqrt(sum((Pred_out_Mult_M1-Test$Price)^2)/length(Pred_out_Mult_M1))) 



# Regularize this model #
reg_mult_mod<-lmridge(Price ~  Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots, Train, lambda=seq(0,.5,.01))
summary(reg_mult_mod)

PRED_reg_mult_mod_IN <- predict(reg_mult_mod, Train)
PRED_reg_mult_mod_OUT<- predict(reg_mult_mod, Test)

(RMSE_reg_mult_IN<-sqrt(sum((PRED_reg_mult_mod_IN-Train$Price)^2)/length(PRED_reg_mult_mod_IN))) #computes in-sample error
(RMSE_reg_mult_OUT<-sqrt(sum((PRED_reg_mult_mod_OUT-Test$Price)^2)/length(PRED_reg_mult_mod_OUT))) #computes out-of-sample 


##################################################################################
##################### HOW TO TUNE REGULARIZED MODEL: lambda values ####################
######################################### need to do  #########################################
#################################################################################

fmla <- Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots

MULT_M1 <- lm(fmla, data = Train)

Pred_in_Mult_M1 <- predict(MULT_M1, newdata = Train)
Pred_out_Mult_M1 <- predict(MULT_M1, newdata = Test)

RMSE_Mult_M1_IN <- sqrt(mean((Pred_in_Mult_M1 - Train$Price)^2))
RMSE_Mult_M1_OUT <- sqrt(mean((Pred_out_Mult_M1 - Test$Price)^2))

cv_model <- cv.glmnet(x = model.matrix(fmla, data = Train), y = Train$Price)

best_lambda <- cv_model$lambda.min

reg_mult_mod <- glmnet(x = model.matrix(fmla, data = Train), y = Train$Price, lambda = best_lambda)

PRED_reg_mult_mod_IN <- predict(reg_mult_mod, s = best_lambda, newx = model.matrix(fmla, data = Train))
PRED_reg_mult_mod_OUT <- predict(reg_mult_mod, s = best_lambda, newx = model.matrix(fmla, data = Test))

RMSE_reg_mult_IN <- sqrt(mean((PRED_reg_mult_mod_IN - Train$Price)^2))
RMSE_reg_mult_OUT <- sqrt(mean((PRED_reg_mult_mod_OUT - Test$Price)^2))

cat("RMSE for multivariate model without regularization (training):", RMSE_Mult_M1_IN, "\n")
cat("RMSE for multivariate model without regularization (testing):", RMSE_Mult_M1_OUT, "\n")
cat("RMSE for regularized multivariate model (training):", RMSE_reg_mult_IN, "\n")
cat("RMSE for regularized multivariate model (testing):", RMSE_reg_mult_OUT, "\n")

##################### SPLINE MODEL #####################

Spline_Mult <- gam( Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots, Train, family = 'gaussian')
summary(Spline_Mult)

PRED_Mult_Spl_IN <- predict(Spline_Mult, Train, type = 'response')
PRED_Mult_Spl_OUT <- predict(Spline_Mult, Test, type = 'response')

(RMSE_SPl_MULT_IN<-sqrt(sum((PRED_Mult_Spl_IN-Train$Price)^2)/length(PRED_Mult_Spl_IN)))  
(RMSE_Spl_MULT_OUT<-sqrt(sum((PRED_Mult_Spl_OUT-Test$Price)^2)/length(PRED_Mult_Spl_OUT))) 


########################## SVM MODEL ########################

kern_type = 'radial'

#BUILD SVM CLASSIFIER
SVM_Model_Mult<- svm(Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots, 
                  data = Train, 
                  type = "eps-regression", #set to "eps-regression" for numeric prediction
                  kernel = kern_type,
                  cost=100,                   #REGULARIZATION PARAMETER
                  gamma = 1/(ncol(Train)-1), #DEFAULT KERNEL PARAMETER
                  coef0 = 0,                    #DEFAULT KERNEL PARAMETER
                  degree=2,                     #POLYNOMIAL KERNEL PARAMETER
                  scale = FALSE)                #RESCALE DATA? (SET TO TRUE TO NORMALIZE)

print(SVM_Model_Mult) #DIAGNOSTIC SUMMARY


PRED_Mult_SVM_IN <- predict(SVM_Model_Mult, Train, type = 'response')
PRED_Mult_SVM_OUT <- predict(SVM_Model_Mult, Test, type = 'response')

(RMSE_SVM_IN<-sqrt(sum((PRED_Mult_SVM_IN-Train$Price)^2)/length(PRED_Mult_SVM_IN)))  
(RMSE_SVM_MULT_OUT<-sqrt(sum((PRED_Mult_SVM_OUT-Test$Price)^2)/length(PRED_Mult_SVM_OUT))) 

##########################TUNING OF SVM MODEL############################
cost_range <- 10^seq(-2, 2, by = 0.5)  # Example range for the cost parameter
gamma_range <- 10^seq(-2, 2, by = 0.5)  # Example range for the gamma parameter

tuned_svm <- tune(svm, 
                  Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots, 
                  data = Train, 
                  type = "eps-regression", 
                  kernel = "radial", 
                  ranges = list(cost = cost_range, gamma = gamma_range))

best_svm <- tuned_svm$best.model
print(best_svm)

PRED_Mult_SVM_IN <- predict(best_svm, Train)
PRED_Mult_SVM_OUT <- predict(best_svm, Test)

RMSE_SVM_IN <- sqrt(mean((PRED_Mult_SVM_IN - Train$Price)^2))
RMSE_SVM_OUT <- sqrt(mean((PRED_Mult_SVM_OUT - Test$Price)^2))

print(paste("RMSE on training set:", RMSE_SVM_IN))
print(paste("RMSE on test set:", RMSE_SVM_OUT))

##########################################################################################
############################## CAN YOU TUNE THIS MODEL ##############################
##########################################################################################


fmla <- Price ~ Total_Area + Bedrooms + Bathrooms + Stories + Parking_Spots



########################## Bagged Tree Model #############################
  
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
  fit(formula = fmla, data = Train)
print(bagg_forest)

#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_in_Bag <- predict(bagg_forest, new_data = Train) %>%
  bind_cols(Train$Price) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Calculate RMSE
rmse_in_bag <- sqrt(mean((pred_in_Bag$...2 - pred_in_Bag$.pred)^2))
rmse_in_bag


#GENERATE OUT-OF-SAMPLE PREDICTIONS ON THE TEST SET AND COMBINE WITH TEST DATA
pred_out_Bag <- predict(bagg_forest, new_data = Test) %>%
  bind_cols(Test$Price) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Calculate RMSE
rmse_out_bag <- sqrt(mean((pred_out_Bag$...2 - pred_out_Bag$.pred)^2))
rmse_out_bag

#################### TUNING OF THE MODEL ###########################

#Tuning
spec_bagg <- bag_tree(min_n = tune() , #minimum number of observations for split
                      tree_depth = tune(), #max tree depth
                      cost_complexity = tune(), #regularization parameter
                      class_cost = NULL)  %>% #for output class imbalance adjustment (binary data only)
  set_mode("regression") %>% #can set to regression for numeric prediction
  set_engine("rpart", times=300) #times = # OF ENSEMBLE MEMBERS IN FOREST
spec_bagg

tree_grid <- grid_regular(hardhat::extract_parameter_set_dials(spec_bagg), levels = 3)


#################### TUNING OF THE MODEL ###########################



#TUNING THE MODEL ALONG THE GRID W/ CROSS-VALIDATION
set.seed(456) #SET SEED FOR REPRODUCIBILITY WITH CROSS-VALIDATION
tuned_results <- tune_grid(spec_bagg,
                           fmla, #MODEL FORMULA
                           resamples = vfold_cv(Train, v=3), #RESAMPLES / FOLDS
                           grid = tree_grid, #GRID
                           metrics = metric_set(yardstick::rmse)) #BENCHMARK METRIC

#RETRIEVE OPTIMAL PARAMETERS FROM CROSS-VALIDATION
best_param <- select_best(tuned_results)

#FINALIZE THE MODEL SPECIFICATION
final_specc <- finalize_model(spec_bagg, best_param)


#FIT THE FINALIZED MODEL
final_modell <- final_specc %>% fit(fmla, Train)

pred_class_bag_tune <- predict(final_modell, new_data = Train) %>%
  bind_cols(Train$Price)

#RMSE Tuned
rmse_bag_tuned <- sqrt(mean((pred_class_bag_tune$...2 - pred_class_bag_tune$.pred)^2))
rmse_bag_tuned

pred_class_bag_tune_o <- predict(final_modell, new_data = Test) %>%
  bind_cols(Test$Price)

rmse_bag_tuned_o <- sqrt(mean((pred_class_bag_tune_o$...2 - pred_class_bag_tune_o$.pred)^2))
rmse_bag_tuned_o



RMSE_Mult_M1_IN
RMSE_Mult_M1_OUT
RMSE_SPl_MULT_IN
RMSE_Spl_MULT_OUT
RMSE_SVM_IN
RMSE_SVM_MULT_OUT
rmse_in_Rand
rmse_out_Rand
rmse_in_bag
rmse_out_bag
rmse_bag_tuned
rmse_bag_tuned_o

 
TABLE_VAL_MULT <- matrix(c(RMSE_Mult_M1_IN, RMSE_Mult_M1_OUT, RMSE_SPl_MULT_IN, RMSE_Spl_MULT_OUT,
                           RMSE_SVM_IN, RMSE_SVM_MULT_OUT, rmse_in_Rand, rmse_out_Rand,
                           rmse_in_bag, rmse_out_bag, rmse_bag_tuned, rmse_bag_tuned_o),
                         ncol = 2, byrow = TRUE)
colnames(TABLE_VAL_MULT) <- c('IN', 'OUT')
rownames(TABLE_VAL_MULT) <- c('RMSE_Mult_M1', 'RMSE_SPl_MULT', 'RMSE_SVM', 'rmse_in_Rand', 'rmse_in_bag', 'rmse_bag_tuned')

# Print the updated table
print(TABLE_VAL_MULT)



#Choose the best out of sample performance and run it using the hold set
## RMSE bag tuned is best right now ############



#GENERATE IN-SAMPLE PREDICTIONS ON THE TRAIN SET AND COMBINE WITH TRAIN DATA
pred_Final_Mult <- predict(random_forest, new_data = Hold) %>%
  bind_cols(Hold$Price) #ADD CLASS PREDICTIONS DIRECTLY TO TEST DATA

# Calculate RMSE
rmse_final_mult <- sqrt(mean((pred_Final_Mult $...2 - pred_Final_Mult $.pred)^2))
rmse_final_mult

##### Result: 1,242,096




###########################################################################


###################### STOPPING POINT ####################

