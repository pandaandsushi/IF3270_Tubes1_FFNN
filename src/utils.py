import numpy as np

class ActivationFunction:
    def __init__(self):
        pass
    
    @staticmethod
    def linear(self, x):
        return x
    
    @staticmethod
    def relu(self, x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    # Bonus: Swish, softplus, and ELU
    @staticmethod
    def swish(self, x):
        return x * ActivationFunction.sigmoid(x)
    
    @staticmethod
    def softplus(self, x):
        return np.log(1 + np.exp(x))
    
    @staticmethod
    def elu(self, x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
class LossFunction:
    def __init__(self):
        pass

    @staticmethod
    def mse(self, yi, y_hat):
        return np.mean((yi - y_hat) ** 2)
    
    @staticmethod
    def binCrossEntropy(self, yi, y_hat):
        return -np.mean(yi * np.log(y_hat) + (1 - yi) * np.log(1 - y_hat))
    
    @staticmethod
    def catCrossEntropy(self, yi, y_hat):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
    
        n = yi.shape[0]
        
        loss = -np.sum(yi * np.log(y_hat)) / n

        return loss

class Derivative:
    def __init__(self):
        pass

    @staticmethod
    def linear(self, x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(self, x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(self, x):
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(self, x):
        return (2 / (np.exp(x) - np.exp(-x)))**2
    
    @staticmethod
    def softmax(self, x):
        s = ActivationFunction.softmax(x)
        return np.outer(s, s) - np.diag(s)
    
    # Bonus: Swish, softplus, and ELU
    @staticmethod
    def swish(self, x):
        s = ActivationFunction.sigmoid(x)
        return s + x * s * (1 - s)
    
    @staticmethod
    def softplus(self, x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))
    