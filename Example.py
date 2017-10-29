import NeuralNetwork as NL
from sklearn.datasets import load_digits


# XOR training dataset example and how should be the training dataset.
# Sklearn datasets follow bassicaly this pattern.
class Dataset:
    def __init__(self):
        self.data = np.array([[0, 0],
                              [0, 1],
                              [1, 0],
                              [1, 1]])
        self.target = np.array([[0], [1], [1], [0]])


dataset = load_digits()
nl = NL.NeuralNetwork(dataset=dataset, layers_format=[64, 100, 100, 1], learning_rate=0.001, momentum=1, tol=0.01)
nl.fit()
for i in range(len(nl.target)):
    print(nl.target[i], nl.output[i], nl.output_error[i])
nine = np.array([0, 0, 0, 1, 1, 0, 0, 0,
                 0, 0, 1, 0, 0, 1, 0, 0,
                 0, 0, 1, 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 0, 0, 1, 0, 0,
                 0, 0, 0, 1, 1, 0, 0, 0,
                 0, 0, 0, 0, 0, 0, 0, 0])
print(nl.predict(nine))

# Running in the terminal:
# 1- Be sure that you have the libs installed:
#   -- Numpy
#   -- sklearn
# 2- Go to the NeuralNetwork.py folder.
# 3- Run python, python3 or etc.
# 4- >>>from sklearn.datasets import load_digits
# 5- >>>import NeuralNetwork as NL
# 6- >>>dataset = load_digits()
# 7- >>>nl = NL.NeuralNetwork(dataset=dataset...)
# 8- >>>nl.fit()
# 9- Let it training some time
# 10- ctrl + c pause the training process but you can run #8 any time
# 11- >>>nl.predict(<some digit to predict>)
# 12- the training digit may follow the nine example above
