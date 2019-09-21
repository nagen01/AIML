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
#Creatting the linear regresson model and fit it
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualize the result with linear regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear regression)') 
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualize the result with Polynomial regression
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Polynomial regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

#visualize the result with with linear and Polynomial regression
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'green')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Truth or Bluff (Linear and Polynomial regression)')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


#Predicting the new value with linear regression
lin_reg.predict(6.5)
#Predicting the new value with polynomial regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))
