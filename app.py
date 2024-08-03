#using numpy to make the calculations
#using numpy will eliminate the use of for loops

import numpy as np
inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]
          ] #--> features from a single sample
weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]

weights2 = [[0.1,-0.14,0.5],
           [-0.5,0.12,-0.33],
           [-0.44,0.73,-0.13]]
biases2 = [-1,2,-0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2
print(layer2_outputs)



# if we use np.dot(inputs, weights) --> error
# adjusting weights and biases can give us completely different and they are associated with individual neurons

#the layer1_output is the output for layer 1 which is then the input of layer2
# hidden layers are called so because, we as programmers are not really in charge of how that layer changes

## coding up the above code, converting the layers into objects

import numpy as np
np.random.seed(0)
X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
          ] #--> features from a single sample

#gonna use oops
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # we are gonna pass the shape inside this paranthesis, we need to know kind of 2 things, whats the size of the input coming in and how many neurons are we gonna have
        self.biases = np.zeros(1, n_neurons)
    def forward(self):
        pass
# in the case of neural networks there are some ways we are gonna initialise a layer, first is we are gonna have to train the model that we have saved and we want to load in that model 

