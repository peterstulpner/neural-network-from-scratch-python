import numpy as np

X = [[1, 2, 3, 2.5],
     [2, 5, -1, 2],
     [-1.5, 2.7, 3.3, -0.8]]

# Saving a model is simply tracking the weights and the biases
# Weights are usually randomised between -1 and 1, but want to have small values
# Smaller values => small increments through the network so that an input doesn't explode in the output
# Biases are usually set to 0, however there are some times where you would not want this,
# mainly when the network is not outputting anything

np.random.seed(0)


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Layer 1 is the input to layer 2, hence layer 2 needs to have shape 5 X m
layer1 = Layer_Dense(len(X[0]), 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
print(f"Layer 1 output: {layer1.output}")
layer2.forward(layer1.output)
print(f"Layer 2 output: {layer2.output}")
