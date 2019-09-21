#Multiple Leniar regression

#Importing important files
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Reading the CSV data set
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

#Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#Removing the dummy variable
X = X[:, 1:]

#Splitting dataset in training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Fitting the leniar regressor in training data set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#Making predictions
y_pred = regressor.predict(X_test)

#------------------------------------------------------------
#Bulding the optimal model using Backword elemination process
#------------------------------------------------------------
#             adding the constant variable  
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#      ######  Backword elemination  #######
#                    1st step
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#      Remove the index containing highest P value greater than 0.05
#                    2nd step
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()  
#      Remove the index containing highest P value greater than 0.05
#                    3rd step
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#      Remove the index containing highest P value greater than 0.05
#                    3rd step
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary() 
#      Remove the index containing highest P value greater than 0.05
#                    3rd step
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()    