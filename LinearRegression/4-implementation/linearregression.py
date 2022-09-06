'''
Class to compute Linear regression manually 
'''


import numpy as np

class LinearRegressionUsingGD:

    def __init__(self, alpha=0.05, n_iterations=1000):
        self.alpha = alpha
        self.n_iterations = n_iterations

    def fit(self, x, y):
        self.cost_ = []
        self.w_ = np.zeros((x.shape[1], 1))
        print('Vettore iniziale: ', self.w_)
        m = x.shape[0]
    
        for _ in range(self.n_iterations):
            y_pred = np.dot(x, self.w_)
            residuals = y_pred - y
            gradient_vector = np.dot(x.T, residuals)
            self.w_ -= (self.alpha / m) * gradient_vector
            cost = np.sum((residuals ** 2)) / (2 * m)
            self.cost_.append(cost)   
        print('Intercept = %0.2f\nCoefficient = %0.2f' % (self.w_[0], self.w_[1]))
        return self
    
    def predict(self, x):
        return np.dot(x, self.w_)