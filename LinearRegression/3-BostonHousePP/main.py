import numpy as np 
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
import plotly.io as pio
pio.templates
import seaborn as sns 
import matplotlib.pyplot as plt 
from scipy import stats
from scipy.stats import norm, skew 
# matplotlib inline 
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
from fpdf import FPDF



def load_data_set(load_boston):
    load_boston = load_boston()
    X = load_boston.data
    y = load_boston.target
    data = pd.DataFrame(X, columns=load_boston.feature_names) 
    data["SalePrice"] = y  
    return data, X, y



data_frame, X, y = load_data_set(load_boston)

print("The shape of the data is:\n", data_frame.shape)
print("Data info: \n", data_frame.info())
print("Data description: \n", load_boston().DESCR)
print("Data facts: \n", data_frame.describe())

# ##to save the data in a compressed filed
# compression_opts = dict(method='zip',archive_name='out.csv')  
# data_frame.to_csv('out.zip', index=False,compression=compression_opts) 
##check if null data in the data frame
print(data_frame.isnull().sum())

##plot pair plots
# sns.pairplot(data_frame, height=2.5)
plt.tight_layout()
# plt.savefig("pair_plot.pdf", dpi=150)

##plot distribution plot target data
sns.distplot(data_frame['SalePrice'])
plt.savefig("distribution_target.pdf", dpi=150)
print("Skewness: %f" % data_frame['SalePrice'].skew())
print("Kurtosis: %f" % data_frame['SalePrice'].kurt())


sns.distplot(data_frame['SalePrice'] , fit=norm)
(mu, sigma) = norm.fit(data_frame['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)])
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.savefig("Normal_target.pdf", dpi=150)
plt.close()
# res = stats.probplot(data_frame['SalePrice'], plot=plt)

data_frame["SalePrice"] = np.log1p(data_frame["SalePrice"])
print("Skewness: %f" % data_frame['SalePrice'].skew())
print("Kurtosis: %f" % data_frame['SalePrice'].kurt())
sns.distplot(data_frame['SalePrice'] , fit=norm)

(mu, sigma) = norm.fit(data_frame['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
plt.savefig("Normal_target_log.pdf", dpi=150)
# res = stats.probplot(data_frame['SalePrice'], plot=plt)



## Plot data correlation
plt.figure(figsize=(10,10))
cor = data_frame.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.PuBu)
plt.savefig("correlation.pdf", dpi=150)

cor_target = abs(cor["SalePrice"]) # absolute value of the correlation 

relevant_features = cor_target[cor_target>0.2] # highly correlated features 

names = [index for index, value in relevant_features.iteritems()] # getting the names of the features 

names.remove('SalePrice') # removing target feature 

print(names) # printing the features 
print(len(names))

## build model
X = data_frame.drop("SalePrice", axis=1) 
y = data_frame["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

lr = LinearRegression() 
lr.fit(X_train, y_train)
predictions = lr.predict(X_test)  

print("Actual value of the house: ", y_test[0]) 
print("Model Predicted Value: ", predictions[0])

mse = mean_squared_error(y_test, predictions) 
rmse = np.sqrt(mse)
print("RMSE=", rmse)