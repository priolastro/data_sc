"""
Time series problem using regression for Stock Price Prediction
"""

import numpy as np
import pandas as pd 
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score 
from sklearn.linear_model import Lasso, Ridge 
from sklearn.svm import SVR 
from sklearn.model_selection import GridSearchCV 
import joblib  




def download_stock(Code):
    # stocks = input("Enter the code of the stock:- ") 
    stocks = Code 
    data = yf.download(stocks, "2008-01-01", "2021-01-18", auto_adjust=True) 
    return data

code = 'F'
data_frame = download_stock(code)
print(data_frame.head())
print("The shape of the data is:\n", data_frame.shape)
print("Data info: \n", data_frame.info())
print("Data facts: \n", data_frame.describe())

data_frame.Close.plot(figsize=(10, 7), color='r')
plt.ylabel("{} Prices".format(code))
plt.title("{} Price Series".format(code))
plt.savefig('Close.pdf')
plt.close()

sns.distplot(data_frame["Close"])
plt.savefig('Close_dist.pdf')
plt.close()
sns.distplot(data_frame["Open"])
plt.savefig('Open_dist.pdf')
plt.close()
sns.distplot(data_frame["High"])
plt.savefig('High_dist.pdf')
plt.close()


"""
    model setup
"""
X = data_frame.drop("Close", axis='columns')
y = data_frame["Close"] 
X_train, X_test, y_train, y_test  = train_test_split(X,y,test_size=0.2, random_state=0) 
print(X_train.shape) 
print(X_test.shape) 
print(y_train.shape) 
print(y_test.shape) 

#linear model
lr = LinearRegression() 
lr.fit(X_train, y_train) 
pred1 = lr.predict(X_test)
#evaluation linear model
mse = mean_squared_error(y_test, pred1) 
rmse = np.sqrt(mse) 
r2_scors = r2_score(y_test, pred1) 

print("MSE:- ", mse) 
print("RMSE:- ", rmse) 
print("R2_score:- ", r2_scors) 

#Lasso and Ridge linear models
la = Lasso().fit(X_train, y_train )
ri = Ridge().fit(X_train, y_train ) 
la_p = la.predict(X_test) 
ri_p = ri.predict(X_test)
##evaluate Lasso
mse = mean_squared_error(y_test, la_p) 
rmse = np.sqrt(mse) 
r2_scors = r2_score(y_test, la_p) 
print("MSE:- ", mse) 
print("RMSE:- ", rmse) 
print("R2_score:- ", r2_scors) 
##evaluate Ridge
mse = mean_squared_error(y_test, ri_p) 
rmse = np.sqrt(mse) 
r2_scors = r2_score(y_test, ri_p) 
print("MSE:- ", mse) 
print("RMSE:- ", rmse) 
print("R2_score:- ", r2_scors) 


#SVR supporting vectors regression

svr = SVR() 
param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}   
grid = GridSearchCV(SVR(), param_grid, refit=True, verbose=3)    
grid.fit(X_train, y_train)

svr = SVR(C=10, gamma=0.01, kernel='rbf') 
svr.fit(X_train, y_train) 
svr_pred = svr.predict(X_test) 

joblib.dump(ri, 'model.pkl') 

ridge_from_joblib = model = joblib.load("model.pkl")