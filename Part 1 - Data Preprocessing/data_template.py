# Data Preprocessing Template
# numpy = contains math tools
# matplotlib.pyplot = importing subplot pyplot of library matplotlib
    # to plot visualizations
# pandas = to import datasets and manage them

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # independent variable vector
    # : means take all columns
    # -1 means except last one
X
y = dataset.iloc[:, 3].values # dependent variable vector (Purchased)
    # take all rows in 3rd column (start counting )
y
  
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # train_test_split(independent, depdendent, % test size)
    # random_state is how you sample

# Feature Scaling
    # age and salary are not on same scale
    # problem is ML algorithms depend on Euclidian distance, so feature scale! 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()     # create object of the class for X
X_train = sc_X.fit_transform(X_train) # fit transform on training set
X_test = sc_X.transform(X_test) # then transform after fitting
sc_y = StandardScaler()     # do same for y
y_train = sc_y.fit_transform(y_train)"""
