# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values # columns 2, 3 -- start count at 0
y = dataset.iloc[:, 4].values # column 4 (0 or 1)

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
    # 400 obs, so 75% in training 

# Feature Scaling
    # do it because we want accurate results
    # age and salary on order of magnitude different scales
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
    # don't need to scale y_train because already 0 or 1

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
    # library = sklearn.linear model (logistic regression is linear model)
    # Class = LogisticRegression
    # Object = classifier (i.e. instance of the class)
classifier = LogisticRegression(random_state = 0)
    # many parameters, seen with Ctrl+i
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test) # vector of predictions
y_pred
    # get a bunch of 0, 1 as your answer on the x_test
    # can now compare to y_test

# Making the Confusion Matrix 
    # = to test how many correct predictions our classifer made
from sklearn.metrics import confusion_matrix
    # library = sklearn.metrics
    # function = confusion_matrix
        # function instead of class because not capital in front
    # Object = cm (i.e. instance of the class)
cm = confusion_matrix(y_test, y_pred)
    # compare real (y_test) and predicted (y_pred)
cm
    # array([[65,  3],
       # [ 8, 24]], dtype=int64)
    # correct = 65 + 24 = 89
    # incorrect = 3 + 8 = 11

# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    # pixel points with 0.01 resolution
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
    # prediction boundary is straight line because logistic regression
    # is a linear prediction model
    # note: our users are not linearly changing, so green in red and likewise

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()