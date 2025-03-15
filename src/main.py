from utils import ActivationFunction, Derivative, LossFunction
import numpy as np
import json

class FFNN:
    def __init__(self, layers, activation_functions, loss_function="mse", weight_method='uniform', seed=None, low_bound = -1, up_bound = 1, mean = 0.0, variance = 0.1):
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

        self.weights, self.biases = self.initialize_weights(weight_method)

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
                w = np.zeros((self.layers[i],self.layers[i+1]))
                b = np.zeros((1,self.layers[i+1]))
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
        print("Starting forward")
        print(input)

        a = input
        activations = [a] #net
        pre_activations = [] #out
        
        for i in range(self.num_layers):
            sigma = np.dot(a, self.weights[i]) + self.biases[i]
            pre_activations.append(sigma)
            
            if self.activation_functions[i] == 'linear':
                a = ActivationFunction.linear(sigma)
            elif self.activation_functions[i] == 'relu':
                print("Applying ReLU")
                print("sigma shape:", sigma)
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

    def backward(self, input, output, activations, pre_activations, learning_rate):
        weight_gradient = []
        biases_gradient = []

        # BACKWARD OUTPUT
        # ∂Err/∂Out
        if self.loss_function == "mse":
            dErr_dOut = activations[-1] - output 
        elif self.loss_function == "binary_crossentropy":
            dErr_dOut = activations[-1] - output
        elif self.loss_function == "categorical_crossentropy":
            dErr_dOut = activations[-1] - output
        else:
            raise ValueError("Loss method unknown")

        # ∂Out/∂Net
        # NEXT: bingung
        # if self.activation_functions[-1] == "sigmoid":
        #     dOut_dNet = activations * (1 - activations)

        if self.activation_functions[-1] == 'sigmoid':
            dOut_dNet = Derivative.sigmoid(pre_activations[-1])
        elif self.activation_functions[-1] == 'relu':
            dOut_dNet = Derivative.relu(pre_activations[-1])
        elif self.activation_functions[-1] == 'tanh':
            dOut_dNet = Derivative.tanh(pre_activations[-1])
        elif self.activation_functions[-1] == 'linear':
            dOut_dNet = Derivative.linear(pre_activations[-1])
        elif self.activation_functions[-1] == 'swish':
            dOut_dNet = Derivative.swish(pre_activations[-1])
        elif self.activation_functions[-1] == 'softplus':
            dOut_dNet = Derivative.softplus(pre_activations[-1])
        elif self.activation_functions[-1] == 'elu':
            dOut_dNet = Derivative.elu(pre_activations[-1], alpha=1.0)
        elif self.activation_functions[-1] == 'softmax':
            dOut_dNet = Derivative.softmax(pre_activations[-1])
        else:
            raise ValueError("Unknown Activation Method")

        # ∂Net/∂W
        dNet_dW = activations[-2]

        # Count weight and bias
        delta = dErr_dOut * dOut_dNet  # This is dE/dNet
        weight_gradient.append(np.dot(dNet_dW.T, delta))
        biases_gradient.append(np.sum(delta, axis=0, keepdims=True))

        # BACKWARD HIDDEN
        # output layer already processed, start from num_layers-2
        for l in range(self.num_layers - 2, -1, -1):
            activation_func = self.activation_functions[l]

            # ∂Err/∂Net
            delta = np.dot(delta, self.weights[l+1].T) * getattr(Derivative, activation_func)(pre_activations[l])

            # ∂Net/∂W
            dNet_dW = activations[l]

            # Compute gradients, insert in beginning since we went from behind
            weight_gradient.insert(0, np.dot(dNet_dW.T, delta))
            biases_gradient.insert(0, np.sum(delta, axis=0, keepdims=True))

        for l in range(self.num_layers):
            self.weights[l] -= learning_rate * weight_gradient[l]
            self.biases[l] -= learning_rate * biases_gradient[l]
        return weight_gradient, biases_gradient
        
    def train(self, X_train, y_train, epochs=50, learning_rate=0.01, batch_size=32, verbose=True):
        num_samples = X_train.shape[0]
        loss_history = []

        for epoch in range(epochs):
            # Shuffle data at each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                # Mini-batch selection
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                print("Done Forward" + epoch+1 + "times")
                # Forward pass
                activations, pre_activations = self.forward(X_batch)

                # Compute loss
                loss = self.count_loss(y_batch, activations[-1])
                epoch_loss += np.mean(loss)

                # Backpropagation
                self.backward(X_batch, y_batch, activations, pre_activations, learning_rate)

            # Average loss per epoch
            epoch_loss /= (num_samples / batch_size)
            loss_history.append(epoch_loss)

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        return loss_history
    def check(self):
        print("Check>>>>>>>>>>>>>>>>>>>>>>>>")
        return