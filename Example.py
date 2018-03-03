# Importing the neural network toolkit.
from NNTK import *

# Importing the sklearn model dataset.
# -for more infos go to "http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html".
from sklearn.datasets import load_digits

# Building the NN layers
# - The number of neurons in the first layers should be equal to the number of inputs of the data set.
layer0 = Layer(n_neurons=100)
layer1 = Layer(n_neurons=100)
layer2 = Layer(n_neurons=1)

# Building the dataset.
dataset = load_digits()

# Building the NN and setting the threshold values.
nn = NN(dataset, [layer0, layer1, layer2], learning_rate=0.001, momentum=0.5, tolerance=0.01)
nn.build_nn()

# Loading the weights of an already trained nn.
nn.load_weigths("weigths.npy")

# Starting to train the NN.
nn.fit()

# It is recommended that training be done from the terminal to be possible
# to execute commands during training.

# After some trainig...
# Building an eight model.
eight = np.array([0, 0, 0, 1, 1, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 0, 1, 1, 0, 0, 0,
                  0, 0, 0, 1, 1, 0, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 1, 0, 0, 1, 0, 0,
                  0, 0, 0, 1, 1, 0, 0, 0])

# Executing the prediction method.
# All the output values are constrained into the interval of 0 to 1,
# so the output of the prediction should be multiplied for 9 to get the real value.
nn.predict(eight)*9