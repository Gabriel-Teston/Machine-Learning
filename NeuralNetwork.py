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
        self.lowest_cost = 1

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

    @staticmethod
    def softsign(soma):
        return soma / (1 + np.abs(soma))

    @staticmethod
    def softsign_derivative(soma):
        return 1 / (np.power(1 + abs(soma), 2))

    # Not implemented yet
    @staticmethod
    def add_bias(data):
        bias_array = np.ones([len(data), 1])
        bias_data = np.append(data, bias_array, axis=1)
        return bias_data

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
        if self.cost > self.previous_cost:
            print("Cost increased")
        if self.cost < self.lowest_cost:
            self.lowest_cost = self.cost
            # This print helps to track if the NN is learning.
            print("Lowest cost: " + str(self.lowest_cost) + "########################################")
        self.previous_cost = self.cost

    def gradient_decend(self):
        last_delta = self.output_error * self.softsign_derivative(self.sinapse_weighted_sum[-1])
        self.delta = [last_delta]
        for i, w in enumerate(self.weights[:0:-1]):
            weight_t = w.T
            delta_dot_weight = np.dot(last_delta, weight_t)
            last_delta = delta_dot_weight * self.softsign_derivative(self.sinapse_weighted_sum[-(i + 2)])
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
        self.weights = np.array(new_weights[::-1])

    def fit(self):
        count = 1000
        while self.cost >= self.tol:
            self.feedfoward()
            self.cost_function()
            self.gradient_decend()
            self.back_propagation()
            if count >= 1000:
                for i in range(len(self.target)):
                    print(self.target[i], self.output[i], self.output_error[i])
                count = 0
            print("Cost: " + str(self.cost))
            count += 1
        for i in range(len(self.target)):
            print(self.target[i], self.output[i], self.output_error[i])

    def predict(self, data):
        w_s = data.dot(self.weights[0])
        last_result = self.softsign(w_s)
        for i, w in enumerate(self.weights[1::]):
            w_s = np.dot(last_result, w)
            last_result = self.softsign(w_s)
        print(last_result)
    def save_weights(self):
        np.save("weights",self.weights)
    def load_weigths(self):
        self.weights = np.load("weights.npy")
