inputs = [1,2,3,2.5] #unique inputs
weights = [0.2,0.8,-0.5,1.0]
bias = 2 
#pretty much the output for the neural network for now
output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias
print(output) 


# the input values can be either just values or the layer
# the weights are given for the synopsys
# the bias is given for the neuron
# we are going to use 4 inputs and 3 neurons, that means there are going to be 3 unique weight sets, and each weight set is going to have 4 unique values

