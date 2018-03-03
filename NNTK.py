import numpy as np

def relu(data):
    return np.maximum(data, 0)


def relu_dirivative(data):
    return 1. * (data > 0)


def sigmoid(data):
    return 1 / (1 + np.exp(-data))


def sigmoid_derivative(data):
    return data * (1 - data)

# Work in progress
def debug(self):
    import os
    if self.error < self.minimum_error:
        print("MINIMUM ERROR")
        self.minimum_error = self.error
    else:
        print()
    if self.error > self.previous_error:
        print("ERROR INCREASED")
    else:
        print("ERROR DECREASED")
    print("ERROR: " + str(self.error))
    print("MINIMUM ERROR: " + str(self.minimum_error))
    print("TARGET  | OUTPUT | ERROR")
    for i in range(50):
        print(int(self.target[i][0] * 9), " | ", np.round(self.output[i][0] * 9, 10), " | ", self.output_error[i][0])
    os.system("clear")

class Layer:
    def __init__(self, n_neurons=4, non_lin=sigmoid, non_lin_derivative=sigmoid_derivative):
        self.n_neurons = n_neurons
        self.non_lin = non_lin
        self.non_lin_derivative = non_lin_derivative
        self.dim = np.ndarray
        self.weights = np.ndarray
        self.input = np.ndarray
        self.weighted_sum = np.ndarray
        self.output = np.ndarray
        self.delta = np.ndarray


    def build_layer(self, input_dim):
        self.dim = input_dim
        self.dim.append(self.n_neurons)
        self.weights = 2 * np.random.random(self.dim) - 1
        return self.dim[1::]

    def forward(self, input):
        self.input = input
        self.weighted_sum = self.input.dot(self.weights)
        self.output = self.non_lin(self.weighted_sum)
        return self.output

    def backward(self, delta):
        self.delta = delta
        delta_dot_weight = delta.dot(self.weights.T)
        next_delta = delta_dot_weight * self.non_lin_derivative(self.input)
        return next_delta

    def update(self, learning_rate=0.01, momentum=1):
        new_weights = self.input.T.dot(self.delta)
        self.weights = (self.weights * momentum) + (new_weights * learning_rate)


class NN:
    def __init__(self, dataset, layers, learning_rate=0.01, momentum=1, tolerance=0.1, debug=debug):
        self.dataset = dataset
        self.input = self.normalize(self.dataset.data)
        self.target = self.normalize(self.dataset.target)
        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.tolerance = tolerance
        self.error = 1
        self.previous_error = 0
        self.minimum_error = 1
        self.output = list
        self.output_error = np.ndarray
        self.debug = debug

    @staticmethod
    def normalize(data):
        max_range_list = []
        min_range_list = []
        if type(data[0]) == np.ndarray:
            normalized_data = np.empty([len(data), len(data[0])], dtype=float)
            for i in range(len(normalized_data)):
                max_range_list.append(max(data[i]))
                min_range_list.append(min(data[i]))
            max_range = max(max_range_list)
            min_range = min(min_range_list)
            for i in range(len(normalized_data)):
                for j in range(len(normalized_data[0])):
                    normalized_data[i][j] = data[i][j] / float(max_range - min_range)
        else:
            normalized_data = np.empty([len(data), 1], dtype=float)
            max_range = max(data)
            min_range = min(data)
            for i in range(len(normalized_data)):
                normalized_data[i] = data[i] / float(max_range - min_range)
        return normalized_data

    def build_nn(self):
        input_dim = list(self.input.shape[1::])
        for layer in self.layers:
            input_dim = layer.build_layer(input_dim)

    def feed_forward(self):
        input = self.input
        for layer in self.layers:
            input = layer.forward(input)
        self.output = self.layers[-1].output

    def back_propagation(self):
        self.output_error = self.target - self.layers[-1].output
        last_delta = self.output_error
        self.error = np.mean(np.abs(last_delta))
        for layer in self.layers[::-1]:
            last_delta = layer.backward(last_delta)
            layer.update(self.learning_rate)

    def fit(self):
        while self.error >= self.tolerance:
            self.feed_forward()
            self.back_propagation()
            self.debug(self)
            self.previous_error = self.error

    def predict(self, data):
        input = data
        for layer in self.layers:
            input = layer.forward(input)
        print(self.layers[-1].output)

    def save_weights(self):
        np.save("weights", [layer.weights for layer in self.layers])

    def load_weigths(self):
        all_weights = np.load("weights.npy")
        for i, layer in enumerate(self.layers):
            layer.weights = all_weights[i]
