# Data Preprocessing Template

# Importing the dataset
dataset = read.csv('Data.csv')

# Take care of missing data
dataset$Age = ifelse(is.na(dataset$Age), 
                     ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Age)

dataset$Salary = ifelse(is.na(dataset$Salary), 
                     ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                     dataset$Salary)

# Encode categorical variables
dataset$Country = factor(dataset$Country,
                         levels = c('France', 'Spain', 'Germany'), 
                         labels = c(1,2,3))

dataset$Purchased = factor(dataset$Purchased,
                         levels = c('No', 'Yes'), 
                         labels = c(0,1))

# Splitting the dataset into the Training set and Test set
# install.packages("caTools")
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
    # dataset$dependentVariable, splitRatio = percent goes to training set
    # TRUE = row goes to training_set below
    # FALSE = row goes to testing_set below
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling 
  # can only do on numerical values, so columns 2,3
training_set[, 2:3] = scale(training_set[, 2:3])
test_set[, 2:3] = scale(test_set[, 2:3])