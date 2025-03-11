from utils import ActivationFunction, Derivative, LossFunction
import numpy as np
import json

class FFNN:
    def __init__(self, layers, activation_functions, loss_function, weight_method='uniform', seed=None, low_bound = -1, up_bound = 1, mean = 0.0, variance = 0.1):
        self.layers = layers
        self.activation_functions = activation_functions
        self.loss_function = loss_function
        self.weight_method = weight_method
        # karena tidak menghitung layer input
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

        self.weights, self.biases = self.initialize_weights(self.weight_method)

    def save_model(self, file_path):
        model_data = {
            "layers": self.layers,
            "activation_functions": self.activation_functions,
            "loss_function": self.loss_function,
            "weight_method": self.weight_method,
            "mean": self.mean,
            "variance": self.variance,
            "low_bound": self.low_bound,
            "up_bound": self.up_bound,
            "seed": self.seed,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f)
        print("Model saved successfully to "+file_path)

    def load_model(self, file_path):
        with open(file_path, "r") as f:
            model_data = json.load(f)

        self.layers = model_data["layers"]
        self.activation_functions = model_data["activation_functions"]
        self.loss_function = model_data["loss_function"]
        self.weight_method = model_data["weight_method"]
        self.mean = model_data["mean"]
        self.variance = model_data["variance"]
        self.low_bound = model_data["low_bound"]
        self.up_bound = model_data["up_bound"]
        self.seed = model_data["seed"]
        self.weights = [np.array(w) for w in model_data["weights"]]
        self.biases = [np.array(b) for b in model_data["biases"]]

        print("Model loaded successfully from "+file_path)


    def initialize_weights(self, method):
        weights = []
        biases = []
        for i in range (self.num_layers):
            if (method == "zero"):
                w = np.zeros((self.layers[i], self.layers[i+1]))
                b = np.zeros((1, self.layers[i+1]))
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
        activations = [a] #net
        pre_activations = [] #out
        
        for i in range(self.num_layers):
            sigma = np.dot(a, self.weights[i]) + self.biases[i]
            pre_activations.append(sigma)
            
            if self.activation_functions[i] == 'linear':
                a = ActivationFunction.linear(sigma)
            elif self.activation_functions[i] == 'relu':
                a = ActivationFunction.relu(sigma)
            elif self.activation_functions[i] == 'sigmoid':
                a = ActivationFunction.sigmoid(sigma)
            elif self.activation_functions[i] == 'tanh':
                a = ActivationFunction.tanh(sigma)
            elif self.activation_functions[i] == 'softmax':
                a = ActivationFunction.softmax(sigma)
            elif self.activation_functions[i] == 'swish':
                a = ActivationFunction.swish(sigma)
            elif self.activation_functions[i] == 'softplus':
                a = ActivationFunction.softplus(sigma)
            elif self.activation_functions[i] == 'elu':
                a = ActivationFunction.elu(sigma)
            else:
                raise ValueError("Method activation unknown")
            activations.append(a)
        
        return activations, pre_activations

    def count_loss(self, observe, pred):
        if self.loss_function == "mse":
            return LossFunction.mse(observe, pred)
        elif self.loss_function == "binary_crossentropy":
            return LossFunction.binCrossEntropy(observe, pred)
        elif self.loss_function == "categorical_crossentropy":
            return LossFunction.catCrossEntropy(observe, pred)
        else:
            raise ValueError("Loss method unknown")

    def backward(self, input, output, activations, pre_activations):
        batch_size = input.shape[0]

        delta_loss = activations[-1] - output

        weight_gradient = []
        biases_gradient = []

        for i in range (self.num_layers - 1, -1, -1):
            prev_activation = activations[i]

            wg = np.dot(prev_activation.T, delta_loss) / batch_size
            bg = np.mean(delta_loss, axis=0)

            weight_gradient.append(wg)
            biases_gradient.append(bg)

            if i > 0:
                z = pre_activations[i-1]
                activation = self.activation_functions[i-1]

                if activation == 'sigmoid':
                    sigma_prime = Derivative.sigmoid(z)
                elif activation == 'relu':
                    sigma_prime = Derivative.relu(z)
                elif activation == 'tanh':
                    sigma_prime = Derivative.tanh(z)
                elif activation == 'linear':
                    sigma_prime = Derivative.linear(z)
                elif activation == 'swish':
                    sigma_prime = Derivative.swish(z)
                elif activation == 'softplus':
                    sigma_prime = Derivative.softplus(z)
                elif activation == 'elu':
                    sigma_prime = Derivative.elu(z, alpha=1.0)
                elif activation == 'softmax':
                    sigma_prime = Derivative.softmax(z)
                else:
                    raise ValueError("Unknown Activation Method")
                
                delta = np.dot(delta, self.weights[i].T) * sigma_prime

        return weight_gradient, biases_gradient

    def update_weights(self, weight_gradient, biases_gradient, learning_rate):
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * weight_gradient[i]
            self.biases[i] -= learning_rate * biases_gradient[i]

    def train(self, x_train, y_train, epoch, learning_rate):
        for i in range (epoch):
            act, pre_act = self.forward(x_train)
        
            loss = self.count_loss(y_train, act[-1])
            
            #wg: weight gradient, bg: bias gradient
            wg, bg = self.backward(x_train, y_train, act, pre_act)

            self.update_weights(wg, bg, learning_rate)

