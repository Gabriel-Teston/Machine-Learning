import numpy as np


class NeuralNetwork:
    default_format = [2, 4, 1]

    def __init__(self, dataset, layers_format=default_format, learning_rate=0.1, momentum=0.5, tol=0.1):
        self.dataset = dataset
        self.training_data = self.normalize(self.dataset.data)
        self.target = self.normalize(self.dataset.target)
        self.input_size = len(self.training_data[0])
        self.layers_format = layers_format
        self.layers_count = len(self.layers_format)
        self.weights = [2 * np.random.random((self.layers_format[i], self.layers_format[i + 1])) - 1 for i in
                        range(len(self.layers_format) - 1)]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.tol = tol
        self.sinapse_weighted_sum = []
        self.sinapse_results = []
        self.output_error = []
        self.cost = 1
        self.output = []
        self.previous_cost = 0
        self.delta = []

    @staticmethod
    def normalize(data):
        max_range = 0
        min_range = 1
        if type(data[0]) == np.ndarray:
            normalized_data = np.empty([len(data), len(data[0])], dtype=float)
            for i in range(len(normalized_data)):
                for j in range(len(normalized_data[0])):
                    if min_range > data[i][j]:
                        min_range = data[i][j]
                    if max_range < data[i][j]:
                        max_range = data[i][j]
            for i in range(len(normalized_data)):
                for j in range(len(normalized_data[0])):
                    normalized_data[i][j] = data[i][j] / float(max_range - min_range)
        else:
            normalized_data = np.empty([len(data), 1], dtype=float)
            for i in range(len(normalized_data)):
                if min_range > data[i]:
                    min_range = data[i]
                if max_range < data[i]:
                    max_range = data[i]
            for i in range(len(normalized_data)):
                normalized_data[i] = data[i] / float(max_range - min_range)
        return normalized_data

    @staticmethod
    def softsign(soma):
        return soma / (1 + np.abs(soma))

    @staticmethod
    def softsign_derivada(soma):
        return 1 / (np.power(1 + abs(soma), 2))

    def feedfoward(self):
        w_s = self.training_data.dot(self.weights[0])
        self.sinapse_weighted_sum = [w_s]
        last_result = self.softsign(w_s)
        self.sinapse_results = [last_result]
        for i, w in enumerate(self.weights[1::]):
            w_s = np.dot(last_result, w)
            self.sinapse_weighted_sum.append(w_s)
            last_result = self.softsign(w_s)
            self.sinapse_results.append(last_result)
        self.output = self.sinapse_results[-1]

    def cost_function(self):
        self.output_error = self.target - self.output
        self.cost = np.mean(np.abs(self.output_error))
        if self.cost < self.previous_cost:
            print("Cost increased")

    def gradient_descend(self):
        last_delta = self.output_error * self.softsign_derivada(self.sinapse_weighted_sum[-1])
        self.delta = [last_delta]
        for i, w in enumerate(self.weights[:0:-1]):
            weight_t = w.T
            delta_dot_weight = np.dot(last_delta, weight_t)
            last_delta = delta_dot_weight * self.softsign_derivada(self.sinapse_weighted_sum[-(i + 2)])
            self.delta.append(last_delta)

    def back_propagation(self):
        backwards_weight = self.weights[::-1]
        new_weights = []
        for i, r in enumerate(self.sinapse_results[-2::-1]):
            result_t = r.T
            new_weight = np.dot(result_t, self.delta[i])
            new_weights.append((backwards_weight[i] * self.momentum) + (new_weight * self.learning_rate))
        result_t = self.training_data.T
        new_weight = np.dot(result_t, self.delta[-1])
        new_weights.append((self.weights[0] * self.momentum) + (new_weight * self.learning_rate))
        self.weights = new_weights[::-1]

    def fit(self):
        while self.cost >= self.tol:
            self.feedfoward()
            self.cost_function()
            self.gradient_descend()
            self.back_propagation()
            print("Cost: " + str(self.cost))
