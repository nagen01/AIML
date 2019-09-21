#Polinomial linear regression

#Importing the important libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading the csv data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

#Creating training and test data
"""
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, _test = train_test_split(X, y, test_size= 0.2, randon_state=0)
"""

#Create your regression here

#Predict result
y_pred = regressor.predict(X)

#visualize the result with regression
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualize the result with regression with higher resolution
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()
