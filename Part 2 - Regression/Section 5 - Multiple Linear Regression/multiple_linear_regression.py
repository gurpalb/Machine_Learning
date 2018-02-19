# Multiple Linear Regression
    # venture capitalist problem
    # dataset rows = companies
    # dependent = profit
    # independent = R&D Spend, Administration, Marketing Spend, State

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values # all but last columns
y = dataset.iloc[:, 4].values # y is last column, counting start at 0 in python

# Encoding categorical data
# State field
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3]) # State field column
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:] # removed first column from X
    # reminder you may need to do with other algorithms too
    # but for MLR, the library does that for you

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Cannot make 2D plot given so many independent variables

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# Add intercept = add a column of ones
    # np.ones(50,1) because that's the size for X
    # as.type(int) to give type to column of ones
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# optimal matrix of variables X_opt
    # idea to remove 1-by-1 if variable not statistically significant
    # we pick significance level, so pick SL = 0.05
    # OLS= ordinary least squares
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
    # x2 p-value = highest greater than 0.05 so remove
    # p-value = 0.990
    # remove and fit model again
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
    # remove x1
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
    # remove x2
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
    # remove x2 ()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
    # done