from NeuralNetwork import *

# Just a XOR example and how should be the training dataset. Sklearn datasets follow bassicaly this pattern
class Dataset:
    def __init__(self):
        self.data = np.array([[0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1]])
        self.target = np.array([[0], [1], [1], [0]])

dataset = Dataset()
nl = NeuralNetwork(dataset=dataset, layers_format=[2, 10, 10, 1], learning_rate=0.5, momentum=1, tol=0.01)
nl.fit()
print(nl.output)
