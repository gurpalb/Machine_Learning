# Data Preprocessing Template
# numpy = contains math tools
# matplotlib.pyplot = importing subplot pyplot of library matplotlib
    # to plot visualizations
# pandas = to import datasets and manage them

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Working directories must always contain our datasets
    # use file explorer

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # independent variable vector
    # : means take all columns
    # -1 means except last one
X
y = dataset.iloc[:, 3].values # dependent variable vector (Purchased)
    # take all rows in 3rd column (start counting )
y

# Taking care of missing data - replace NaN with mean of column
    # import Inputer class from sklearn library
from sklearn.preprocessing import Imputer 
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# now fit imputer into our data
imputer = imputer.fit(X[:, 1:3])  # fit to columns missing data only (age and salary, 2nd
    # and third columns & go until 3 because count at 0 and upper bound excl.)
# now replace missing data with mean of column
X[:, 1:3] = imputer.transform(X[:, 1:3])
X    

# Encoding categorical variables (replace text with numbers)
# Country field
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X
    # but, 0, 1, and 2 could imply 2 > 1; so have dummy encoding of 
    # 3 columns for this (i.e. 000, 001, 010 etc.)
# Create dummy variables using OneHotEncoder
onehotencoder= OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X

# Purchased field
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    # train_test_split(independent, depdendent, % test size)
    # random_state is how you sample

# Feature Scaling
    # age and salary are not on same scale
    # problem is ML algorithms depend on Euclidian distance, so feature scale! 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()     # create object of the class for X
X_train = sc_X.fit_transform(X_train) # fit transform on training set
X_test = sc_X.transform(X_test) # then transform after fitting
sc_y = StandardScaler()     # do same for y
y_train = sc_y.fit_transform(y_train)
    # Do we need to do feature scaling of dummy variables? Depends
        # we may need to make interpretations (like France not 1 anymore)
    # Do we need to do feature scaling for y_test? 
        # No, if categorical (0 and 1)
        # Yes, if regression (numerical)