from sklearn.preprocessing import OneHotEncoder
from utils import ActivationFunction, LossFunction, Derivative, RMSNorm
import numpy as np
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import networkx as nx

class FFNN:
    def __init__(self, layers, activation_functions, loss_function="mse", weight_method='xavier', seed=None, low_bound = -1, up_bound = 1, mean = 0.0, variance = 0.1, regularization=None, lambda_reg=0.01, rms_norm=False):
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
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.rms_norm = rms_norm

        self.weights = []
        self.biases = []
        self.w_gradients = []
        self.b_gradients = []
        self.rms_layer = []

        # agar hasil random bisa sama untuk seed sama
        if seed is not None:
            np.random.seed(seed)

        self.weights, self.biases = self.initialize_weights(weight_method)

        if self.rms_norm:
            self.rms_layer = [RMSNorm(dim=self.layers[i+1]) for i in range (self.num_layers)]
        else:
            self.rms_layer = None

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
        print("Model saved successfully to " + file_path)

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

        print("Model loaded successfully from " + file_path)


    def initialize_weights(self, method):
        weights = []
        biases = []
        for i in range (self.num_layers):
            if (method == "zero"):
                w = np.zeros((self.layers[i],self.layers[i+1]))
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
        activations = [a] #out sudah dikasih aktivasi
        pre_activations = [] #net

        for i in range(self.num_layers):
            # print("Ini bias shape")
            # print(a.shape)
            # print(self.weights[i].shape)
            sigma = np.dot(a, self.weights[i]) + self.biases[i]

            if self.rms_norm:
                sigma = self.rms_layer[i](sigma)

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
        base_loss = None
        if self.loss_function == "mse":
            base_loss = LossFunction.mse(observe, pred)
        elif self.loss_function == "binary_crossentropy":
            base_loss = LossFunction.binCrossEntropy(observe, pred)
        elif self.loss_function == "categorical_crossentropy":
            base_loss = LossFunction.catCrossEntropy(observe, pred)
        else:
            raise ValueError("Loss method unknown")

        if self.regularization == "l1":
            penalty = self.lambda_reg * sum([np.sum(np.abs(w)) for w in self.weights])
            return base_loss + penalty
        elif self.regularization == "l2":
            penalty = self.lambda_reg * sum([np.sum(w ** 2) for w in self.weights])
            return base_loss + penalty
        else:
            return base_loss

        

    def backward(self, input, output, activations, pre_activations, learning_rate):
        weight_gradient = []
        biases_gradient = []

        # BACKWARD OUTPUT
        # ∂Err/∂Out
        # print("Counting Err/Out\n")
        if self.loss_function == "mse":
            dErr_dOut = activations[-1] - output
        elif self.loss_function == "binary_crossentropy":
            dErr_dOut = activations[-1] - output
        elif self.loss_function == "categorical_crossentropy":
            dErr_dOut = activations[-1] - output
        else:
            raise ValueError("Loss method unknown")

        # ∂Out/∂Net
        # print("Counting Out/Net\n")
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
            # print(pre_activations[-1].shape)
            s = ActivationFunction.softmax(pre_activations[-1])
            # print(s.shape)
            dOut_dNet = Derivative.softmax(pre_activations[-1])
        else:
            raise ValueError("Unknown Activation Method")

        # ∂Net/∂W
        # print("Counting Net/W\n")
        dNet_dW = activations[-2]

        # Count weight and bias
        if (self.activation_functions[-1] == 'softmax' and self.loss_function == 'categorical_crossentropy'):
            delta = dErr_dOut
        else:
            delta = dErr_dOut * dOut_dNet

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
            if self.regularization == "l1":
                weight_gradient[l] += self.lambda_reg * np.sign(self.weights[l])
            elif self.regularization == "l2":
                weight_gradient[l] += 2 * self.lambda_reg * self.weights[l]
            self.weights[l] -= learning_rate * weight_gradient[l]
            self.biases[l] -= learning_rate * biases_gradient[l]
        
        self.w_gradients = weight_gradient
        self.b_gradients = biases_gradient

    def plot_model_structure(self, show_weights=True, show_gradients=True):
        G = nx.DiGraph()

        edge_labels = {}
        # # Input nodes
        # for i in range(self.layers[0]):
        #     G.add_node(f"x{i+1}", layer="input")

        # Hidden layer nodes
        for hidden_idx in range(1, self.num_layers):
            for neuron_idx in range(self.layers[hidden_idx]):
                G.add_node(f"h{hidden_idx}-{neuron_idx+1}", layer="hidden")
            G.add_node(f"b{hidden_idx}", layer="bias")

        # Output nodes
        for neuron_idx in range(self.layers[-1]):
            G.add_node(f"o{neuron_idx+1}", layer="output")

        # # Edge dari input ke h-1
        # for i in range (self.layers[0]):
        #     for j in range (self.layers[1]):
        #         G.add_edge(f"x{i+1}", f"h1-{j+1}")

        # Edge dari setiap hidden layer
        for i in range(1, self.num_layers-1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i+1]):
                    # G.add_edge(f"h{i}-{j+1}", f"h{i+1}-{k+1}")
                    from_node = f"h{i}-{j+1}"
                    to_node = f"h{i+1}-{k+1}"
                    G.add_edge(from_node, to_node)

                    if show_weights:
                        weight = self.weights[i][j][k]
                        label = f"W:{weight:.2f}"
                        if show_gradients and len(self.w_gradients) > i:
                            grad = self.w_gradients[i][j][k]
                            label += f"\nG:{grad:.2f}"
                        edge_labels[(from_node, to_node)] = label

        # # Edge dari hidden terakhir ke output
        # for i in range (self.layers[-2]):
        #     for j in range (self.layers[-1]):
        #         G.add_edge(f"h{self.num_layers-1}-{i+1}", f"o{j+1}")

        # Edge dari hidden terakhir ke output
        last_hidden = self.num_layers - 1
        for i in range(self.layers[-2]):
            for j in range(self.layers[-1]):
                from_node = f"h{last_hidden}-{i+1}"
                to_node = f"o{j+1}"
                G.add_edge(from_node, to_node)

                if show_weights:
                    weight = self.weights[-1][i][j]
                    label = f"W:{weight:.2f}"
                    if show_gradients and len(self.w_gradients) > last_hidden - 1:
                        grad = self.w_gradients[-1][i][j]
                        label += f"\nG:{grad:.2f}"
                    edge_labels[(from_node, to_node)] = label

        # # Edge dari bias ke masing-masing hidden layer
        # for i in range (self.num_layers-1):
        #     for j in range (self.layers[i+1]):
        #         G.add_edge(f"b{i+1}", f"h{i+1}-{j+1}")

        # Edge dari bias ke masing-masing hidden layer
        for i in range(1, self.num_layers):
            for j in range(self.layers[i]):
                from_node = f"b{i}"
                to_node = f"h{i}-{j+1}"
                G.add_edge(from_node, to_node)
                # Bias weight is just bias value
                if show_weights:
                    bias_val = self.biases[i-1][0][j]
                    label = f"B:{bias_val:.2f}"
                    if show_gradients and len(self.b_gradients) > i-1:
                        bias_grad = self.b_gradients[i-1][0][j]
                        label += f"\nG:{bias_grad:.2f}"
                    edge_labels[(from_node, to_node)] = label


        pos = {}
        vertical_spacing = 15000
        horizontal_spacing = 50000

        for hidden_idx in range(1, self.num_layers):
            for neuron_idx in range(self.layers[hidden_idx]):
                pos[f"h{hidden_idx}-{neuron_idx+1}"] = (hidden_idx * horizontal_spacing, neuron_idx * vertical_spacing)


        for neuron_idx in range(self.layers[-1]):
            pos[f"o{neuron_idx+1}"] = ((self.num_layers) * horizontal_spacing, neuron_idx * vertical_spacing)

        for hidden_idx in range(1, self.num_layers):
            pos[f"b{hidden_idx}"] = ((hidden_idx) * (horizontal_spacing) - 30000, -15000)


        # Visualisasi graph menggunakan matplotlib
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        # Tampilkan plot
        plt.title("FFNN Graph")
        plt.show()

    def plot_loss_curve(self, loss_history):
        plt.plot(loss_history, label="Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()
        plt.show()

    def weight_dist_plot(self, layer_idx=None):
        if layer_idx is None:
            layer_idx = range(self.num_layers)

        for i in layer_idx:
            plt.figure(figsize=(5, 3))
            plt.hist(self.weights[i].flatten(), bins=50)
            plt.title(f'Weight Distribution for Layer {i+1}')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')
            plt.show()

    def grad_dist_plot(self, layer_idx=None):
        if layer_idx is None:
            layer_idx = range(self.num_layers)

        for i in layer_idx:
            plt.figure(figsize=(5, 3))
            if i < len(self.w_gradients): 
                plt.hist(self.w_gradients[i].flatten(), bins=50)
                plt.title(f'Gradient Distribution for Layer {i+1}')
                plt.xlabel('Gradient Value')
                plt.ylabel('Frequency')
                plt.show()

    def train(self, X_train, y_train, epochs, learning_rate, batch_size, verbose):
        num_samples = X_train.shape[0]
        loss_history = []  # Menyimpan loss training history

        encoder = OneHotEncoder(sparse_output=False)
        y_train_encoded = encoder.fit_transform(y_train.reshape(-1, 1)) 
            

        for epoch in range(epochs):
            # Shuffle data at each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train, y_train_encoded = X_train[indices], y_train_encoded[indices]

            epoch_loss = 0
            for i in range(0, num_samples, batch_size):
                # Mini-batch selection
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train_encoded[i:i + batch_size]

                # Forward Propagation
                activations, pre_activations = self.forward(X_batch)

                # Compute loss
                loss = self.count_loss(y_batch, activations[-1])
                epoch_loss += np.mean(loss)

                # Backpropagation
                self.backward(X_batch, y_batch, activations, pre_activations, learning_rate)

            # Average loss per epoch
            epoch_loss /= (num_samples / batch_size)
            loss_history.append(epoch_loss)
            
            if verbose == 1:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}")

        # Display the loss curve after training
        self.plot_loss_curve(loss_history)

