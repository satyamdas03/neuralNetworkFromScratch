inputs = [1,2,3,2.5] #unique inputs
#adding extra unique weight sets
#adding extra unique biases
#using list of lists
weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-.26, -0.27, 0.17, 0.87]
           ]
biases = [2,3,0.5]
#pretty much the output for the neural network for now
# using loop structure to simplify the calculation
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases): # zip combines 2 lists into a list of lists
    neuron_output = 0
    for n_input , weight in zip(inputs,neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
print(layer_outputs)
## this for loop will be transitioned by using numpys and all


# the input values can be either just values or the layer
# the weights are given for the synopsys
# the bias is given for the neuron
# we are going to use 4 inputs and 3 neurons, that means there are going to be 3 unique weight sets, and each weight set is going to have 4 unique values, we are going to need unique biases 

# the next steps will be working the DOT PRODUCT
    #--code --> ditching the repetative code 
    #--weight and bias
    #--shape
    #--dot product

# NOTES: weights and biases are used to tune the outcome 

# the concept of SHAPE
# shape is basically at each dimension, what's the size of 
# that dimension, suppose we have got a list of 4 elements : 
# list : [1,5,6,2] ; shape is (4,) ; type : 1Darray, vector
# list of list --> 2Darray, matrix
# tensor is an object that can be represented as an object that can be represented as an array
# we want to multiply the weights and inputs