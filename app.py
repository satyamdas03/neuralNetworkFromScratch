inputs = [1,2,3,2.5] #unique inputs
#adding extra unique weight sets
weights1 = [0.2,0.8,-0.5,1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-.26, -0.27, 0.17, 0.87]
#adding extra unique biases
bias1 = 2 
bias2 = 3
bias3 = 0.5
#pretty much the output for the neural network for now
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
        ]
print(output) 


# the input values can be either just values or the layer
# the weights are given for the synopsys
# the bias is given for the neuron
# we are going to use 4 inputs and 3 neurons, that means there are going to be 3 unique weight sets, and each weight set is going to have 4 unique values, we are going to need unique biases 

# the next steps will be working the DOT PRODUCT
    #--code
    #--weight and bias
    #--shape
    #--dot product

