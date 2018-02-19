# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
    # dependent variable = Salary
    # independent = Level
    # plot y vs. x looks exponentially increasing salary per level
X = dataset.iloc[:, 1:2].values
    # really just need X[:, 1] but that is a vector, ML needs matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""
    # here, we don't really need to split because we have small n

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
    # do not need feature scaling, SLR and MLR used libraries that 
    # did not need it

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression 
    # get class sklearn.linear_model, and object LinearRegression
lin_reg = LinearRegression() # create our object
lin_reg.fit(X, y) # fit our data to our object

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
    # get class sklearn.preprocessing, and object PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4) # creates X vector into new columns
    # new columns are up to degree specified
    # ex. change intercept | X to X | X^2 | X^3 ...
        # recall intercept is column of 1's
X_poly = poly_reg.fit_transform(X)
    # method fit_transform(X) works
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict(6.5)

# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))