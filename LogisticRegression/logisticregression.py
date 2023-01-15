from scipy import optimize
import numpy as np

class LinearRegressionUsingGD:
    """ 
    using GD with scipy function
    Not using regularization, result compared to sklearn could be different
    """ 

    def __init__(self, learning_rate = 0.05, n_iterations = 1000):
        self.learning_rate = learning_rate 
        self.n_iterations = n_iterations

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def hypothesis(self, theta, x):
        return self.sigmoid(np.dot(x,theta))

    def cost_function(self, theta, x, y):
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.hypothesis(theta, x)) + (1 - y) * np.log(
                1 - self.hypothesis(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(np.dot(x,theta)) - y)

    def fit(self, x, y, theta):
        opt_weights = optimize.fmin_tnc(func=self.cost_function, x0=theta,
        fprime=self.gradient,args=(x, y.flatten()))
        self.parameters = opt_weights[0]
        return self
    
    def predict(self, x):
        parameters = self.parameters
        return self.hypothesis(parameters, x) 

        




