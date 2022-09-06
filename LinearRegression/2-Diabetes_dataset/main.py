import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

#load dataset and split it
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_training = diabetes_X[:-20]
diabetes_X_testing = diabetes_X[-20:]
diabetes_y_training = diabetes_y[:-20]
diabetes_y_testing = diabetes_y[-20:]
#create linear regression model
regression_model = linear_model.LinearRegression()
#train model with the training set
regression_model.fit(diabetes_X_training, diabetes_y_training)
#make the prediciton using the testing set
diabetes_y_pred = regression_model.predict(diabetes_X_testing)

#print data prediction of the model
COEFFICIENT = regression_model.coef_
MSE = mean_squared_error(diabetes_y_testing, diabetes_y_pred)
R2SCORE = r2_score(diabetes_y_testing, diabetes_y_pred)

print("The coefficient is %.5f\nThe MSE is %.5f\nThe coefficient of determination (if 1 perfect prediction) is %.5f "%(COEFFICIENT, MSE, R2SCORE))

#plot 
plt.scatter(diabetes_X_testing, diabetes_y_testing, color="black")
plt.plot(diabetes_X_testing, diabetes_y_pred, color="blue", linewidth=3)
plt.xlabel('BMI (body mass index)')
plt.ylabel('Disease progression(one year after baseline)')

plt.xticks(())
plt.yticks(())
plt.show()

