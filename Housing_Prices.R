######## Housing Prices ####

#### Variable to Predict: Price of Houses ######

# Libraries to Load 

library(tidymodels)
library(sqldf)
library(splitTools)
library(rpart.plot)
library(baguette)

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



# Partitioning the data into Training, Testing, and Holdout ##

#Split the data frame into partitions 
Sets <- partition(Prices$Total_Area, p = c(train = 0.70, Hold = 0.15, test = 0.15))
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



Prices_reg <- decision_tree(min_n = 20 , #minimum number of observations for split
              tree_depth = 30, #max tree depth
              cost_complexity = 0.01)  %>% #regularization parameter
  set_engine("rpart") %>%
  set_mode("regression")

print(Prices_reg)

#Esimating the Model

Prices_fmla <- Price ~ .
Prices_tree <- Prices_reg %>%
  fit(formula = Prices_fmla, data = Prices)
print(Prices_tree)

#VISUALIZING THE REGRESSION TREE
Prices_tree$fit %>%
  rpart.plot(type = 2, roundint = FALSE)

#change the numbers out of scientific notation


# Univariate Model

M1 <- lm(Price ~ Total_Area, Prices)
summary(M1)


# Bivariate Regression Modeling

M2 <- lm(Price ~ Total_Area + Bedrooms, Prices)
summary(M2)

M3 <- lm(Price ~ Total_Area + PrefArea, Prices)
summary(M3)

M4 <- lm(Price ~ Total_Area + AC, Prices)
summary(M4)

# Multivariate Regression Modeling

M5 <- lm(Price ~ Total_Area + Parking_Spots + Stories, Prices)
summary(M5)

M6 <- lm(Price ~ Total_Area + Bedrooms + Stories + Bathrooms, Prices)
summary(M6)

M7 <- lm(Price ~ Total_Area + PrefArea + Bedrooms + Stories, Prices)
summary(M7)


