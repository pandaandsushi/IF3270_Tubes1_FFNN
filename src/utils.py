import numpy as np

class ActivationFunction:
    def __init__(self):
        pass
    
    @staticmethod
    def linear(x):
        return x
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def tanh(x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    
    # Bonus: Swish, softplus, and ELU
    @staticmethod
    def swish(x):
        return x * ActivationFunction.sigmoid(x)
    
    @staticmethod
    def softplus(x):
        return np.log(1 + np.exp(x))
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))
    
class LossFunction:
    def __init__(self):
        pass

    @staticmethod
    def mse(yi, y_hat):
        return np.mean((yi - y_hat) ** 2)
    
    @staticmethod
    def binCrossEntropy(yi, y_hat):
        return -np.mean(yi * np.log(y_hat) + (1 - yi) * np.log(1 - y_hat))
    
    @staticmethod
    def catCrossEntropy(yi, y_hat):
        eps = 1e-15
        y_hat = np.clip(y_hat, eps, 1 - eps)
    
        n = yi.shape[0]
        
        loss = -np.sum(yi * np.log(y_hat)) / n

        return loss

class Derivative:
    def __init__(self):
        pass

    @staticmethod
    def linear(x):
        return np.ones_like(x)
    
    @staticmethod
    def relu(x):
        return np.where(x > 0, 1, 0)
    
    @staticmethod
    def sigmoid(x):
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        return 1 - np.tanh(x)**2
    
    @staticmethod
    def softmax(x):
        s = ActivationFunction.softmax(x)
        return np.outer(s, s) - np.diag(s)
    
    # Bonus: Swish, softplus, and ELU
    @staticmethod
    def swish(x):
        s = ActivationFunction.sigmoid(x)
        return s + x * s * (1 - s)
    
    @staticmethod
    def softplus(x):
        return 1 / (1 + np.exp(-x))
    
    @staticmethod
    def elu(x, alpha=1.0):
        return np.where(x > 0, 1, alpha * np.exp(x))
    