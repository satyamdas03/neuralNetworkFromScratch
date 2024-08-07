import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()  # Setting up a default datatype for the numpy use

X, y = spiral_data(100, 3)  # 100 feature sets of 3 classes

# Define the Layer_Dense class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define the Activation_ReLU class
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Initialize the layers
layer1 = Layer_Dense(2, 5)  # 2 input features (matches the dataset), 5 neurons
activation1 = Activation_ReLU()

# Perform a forward pass
layer1.forward(X)
activation1.forward(layer1.output)

# Print the output of the first layer
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
## GONNA HAVE TO UPDATE THE README FILE

#coding up the softmax function(following the steps inside the documents.txt)

import numpy as np
import nnfs
nnfs.init()
layer_outputs = [[4.8,1.21,2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]
exp_values = np.exp(layer_outputs)
norm_values = exp_values/np.sum(exp_values, axis=1,keepdims=True) 
##np.sum(exp_values, axis=1, keepdims=True) 
##axis=1 --> for row wise sums, keepdims=True --> for printing the sums output in the layer_outputs shape
print(norm_values)
