######## Most Profitable Movies of All Time ####

#### Variable to Predict: Worldwide Gross ######


# Libraries to Load 

library(tidymodels)
library(sqldf)


# Import Data

Movie <- read.csv("https://raw.githubusercontent.com/ryankysar/BUAN-381/main/Movies.csv")
View(Movie)

summary(Movie)

# Partitioning the data into Training and Testing ##

p <-.7 # 70% in sample and 30% out of sample

Movie_Count <- dim(Movie)[1]

Training_size <- floor(p * Movie_Count)

Training_size


# Setting the seed so it reproduces the same results

set.seed(123)

Train_DF <- sample(Movie_Count, size = Training_size) # Run 32 and 34 together



Training_M <- Movie[Train_DF, ] #PULLS RANDOM ROWS FOR TRAINING
Testing_M <- Movie[-Train_DF, ] #PULLS RANDOM ROWS FOR TESTING


dim(Training_M)

dim(Testing_M)



