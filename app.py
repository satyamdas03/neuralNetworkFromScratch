#using numpy to make the calculations
#using numpy will eliminate the use of for loops

import numpy as np
inputs = [1,2,3,2.5] #--> features from a single sample
weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-.26, -0.27, 0.17, 0.87]]
biases = [2,3,0.5]
output = np.dot(weights,inputs) + biases
print(output)


# if we use np.dot(inputs, weights) --> error
# adjusting weights and biases can give us completely different
