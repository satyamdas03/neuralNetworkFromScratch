#using numpy to make the calculations
#using numpy will eliminate the use of for loops

# if we use np.dot(inputs, weights) --> error
# adjusting weights and biases can give us completely different and they are associated with individual neurons

#the layer1_output is the output for layer 1 which is then the input of layer2
# hidden layers are called so because, we as programmers are not really in charge of how that layer changes

## coding up the above code, converting the layers into objects

import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init() #setting up a default datatype for the numpy use

X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
          ] #--> features from a single sample

X, y = spiral_data(100,3) #100 feature sets of 3 classes
#gonna use oops
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # we are gonna pass the shape inside this parenthesis, we need to know kind of 2 things, whats the size of the input coming in and how many neurons are we gonna have
        self.biases = np.zeros((1, n_neurons)) #since the biases is a 1d array of the number of neurons
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases

#coding up the activation function
class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0, inputs)

# in the case of neural networks there are some ways we are gonna initialise a layer, first is we are gonna have to train the model that we have saved and we want to load in that model 
# the output of the first layer is the input of the second layer
layer1 = Layer_Dense(4,5) #4--> input size ; 5 is the random value
layer2 = Layer_Dense(5,2) #5--> output of one layer is input of another
layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)


