from utils import ActivationFunction, Derivative, LossFunction
import numpy as np


class FFNN:
    def __init__(self, layers, activations, loss_function, method='uniform', seed=None, low_bound = -1, up_bound = 1, mean = 0.0, variance = 0.1):
        self.layers = layers
        self.activations = activations
        self.loss_function = loss_function
        self.num_layers = len(layers) - 1
        self.seed = seed
        self.low_bound = low_bound
        self.up_bound = up_bound
        self.mean = mean
        self.variance = variance

        self.weights = []
        self.biases = []
        self.gradients = []

        # agar hasil random bisa sama untuk seed sama
        if seed is not None:
            np.random.seed(seed)

        self.weights, self.biases = self.initialize_weights(method)

    def initialize_weights(self, method):
        weights = []
        biases = []
        for i in range (self.num_layers):
            if (method == "zero"):
                w = np.zeros((self.layers[i],self.layers[i+1]))
                b = np.zeros(1,self.layers[i+1])
            elif (method == "uniform"):
                w = np.random.uniform(self.low_bound, self.up_bound, (self.layers[i],self.layers[i+1]))
                b = np.random.uniform(self.low_bound, self.up_bound, (1,self.layers[i+1]))
            elif (method == "normal"):
                deviation = np.sqrt(self.variance)
                w = np.random.normal(self.mean, deviation, (self.layers[i],self.layers[i+1]))
                b = np.random.normal(self.mean, deviation, (1, self.layers[i+1]))
            elif method == 'xavier':
                fan_in = self.layers[i]
                fan_out = self.layers[i+1]
                limit = np.sqrt(2 / (fan_in + fan_out))
                w = np.random.normal(0, limit, (self.layers[i], self.layers[i+1]))
                b = np.zeros((1, self.layers[i+1]))
            elif method == 'he':
                fan_in = self.layers[i]
                limit = np.sqrt(2 / float(fan_in))
                w = np.random.normal(0, limit, (self.layers[i], self.layers[i+1]))
                b = np.zeros((1, self.layers[i+1]))
            else:
                raise ValueError("Method initialize weight unknown")

            weights.append(w)
            biases.append(b)
        return weights, biases
    
    def forward(self, input):
        a = input
        activations = [a]
        pre_activations = []
        
        for i in range(self.num_layers):
            sigma = np.dot(a, self.weights[i]) + self.biases[i]
            pre_activations.append(sigma)
            
            if self.activations[i] == 'linear':
                a = ActivationFunction.linear(sigma)
            elif self.activations[i] == 'relu':
                a = ActivationFunction.relu(sigma)
            elif self.activations[i] == 'sigmoid':
                a = ActivationFunction.sigmoid(sigma)
            elif self.activations[i] == 'tanh':
                a = ActivationFunction.tanh(sigma)
            elif self.activations[i] == 'softmax':
                a = ActivationFunction.softmax(sigma)
            elif self.activations[i] == 'swish':
                a = ActivationFunction.swish(sigma)
            elif self.activations[i] == 'softplus':
                a = ActivationFunction.softplus(sigma)
            elif self.activations[i] == 'elu':
                a = ActivationFunction.elu(sigma)
            else:
                raise ValueError("Method activation unknown")
            activations.append(a)
        
        return activations, pre_activations

    
