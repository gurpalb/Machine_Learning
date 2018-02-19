# SVR
    # same as polynomial regression, but not fit through "outliers" depending
    # on what kernal we choose. In this example, kernel = 'rbf' 

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
    # need feature scaling for SVR class 
from sklearn.preprocessing import StandardScaler
    # create 2 separate objects, 1 for X, 1 for y
sc_X = StandardScaler() # scale X
sc_y = StandardScaler() # scale y
X = sc_X.fit_transform(X) # fit transformed X
# y = sc_y.fit_transform(y) # fit transformed y BUT this line does not work
 y = np.ravel(sc_y.fit_transform(y.reshape(-1, 1)))  
     # this line is a quick fix for unworking line before it

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf') # poly or rbf can work, not linear ones
regressor.fit(X, y)

# Predicting a new result
y_pred = regressor.predict(6.5) # get non-scaled version of salary
y_pred
y_pred = sc_y.inverse_transform(y_pred) # to get original scale of salary
y_pred

# Visualising the SVR results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()