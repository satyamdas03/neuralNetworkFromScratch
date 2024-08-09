# Constructing a Neural Network From Scratch

## Overview

By doing this project I am aiming to construct a neural network from scratch by understanding all its functionalities.

## Versions and Changes

### Version 1: Initial Implementation

- **Inputs**: Unique inputs provided as a list.
- **Weights**: Separate unique weight sets for each neuron.
- **Biases**: Separate unique biases for each neuron.
- **Output Calculation**: Used individual calculations for each neuron.
- **Code**:
  ```python
  inputs = [1,2,3,2.5] # unique inputs
  weights1 = [0.2,0.8,-0.5,1.0]
  weights2 = [0.5, -0.91, 0.26, -0.5]
  weights3 = [-.26, -0.27, 0.17, 0.87]
  bias1 = 2 
  bias2 = 3
  bias3 = 0.5

  output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
            inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
            inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3
           ]
  print(output)
  ```

### Version 2: Loop Implementation

- **Improvement**: Simplified the output calculation using loop structures.
- **Weights and Biases**: Used list of lists for weights and a separate list for biases.
- **Code**:
  ```python
  inputs = [1,2,3,2.5] # unique inputs
  weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-.26, -0.27, 0.17, 0.87]]
  biases = [2,3,0.5]

  layer_outputs = []
  for neuron_weights, neuron_bias in zip(weights, biases):
      neuron_output = 0
      for n_input, weight in zip(inputs, neuron_weights):
          neuron_output += n_input * weight
      neuron_output += neuron_bias
      layer_outputs.append(neuron_output)
  print(layer_outputs)
  ```

### Version 3: Using NumPy for Efficiency

- **Improvement**: Eliminated the use of for loops by leveraging NumPy for matrix operations.
- **Code**:
  ```python
  import numpy as np
  inputs = [1,2,3,2.5]
  weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-.26, -0.27, 0.17, 0.87]]
  biases = [2,3,0.5]
  output = np.dot(weights, inputs) + biases
  print(output)
  ```

  ### Version 4: Using The Transpose Method

- **Improvement**: to fix the error we are going to use the transpose method
we are going to transpose the weight array, using the numpy method. So first we are going to convert the weights array to numpy array and then use .T to make it into a transpose. Using numpy to make the calculations. Using numpy will eliminate the use of for loops
- **Code**:
  ```python
  import numpy as np
  inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]
          ] #--> features from a single sample
  weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-.26, -0.27, 0.17, 0.87]]
  biases = [2,3,0.5]
  output = np.dot(inputs, np.array(weights).T) + biases
  print(output)
  ```

  ### Version 5: Adding a New Layer

- **Code**:
  ```python
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
  ```
- The layer1_output is the output for layer 1 which is then the input of layer2


### Version 6: Converting the object of layers to objects

- **Code**:
  ```python
  import numpy as np
  np.random.seed(0)
  X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]
          ] #--> features from a single sample

  #gonna use oops
  class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons) # we are gonna pass the shape inside this parenthesis, we need to know kind of 2 things, whats the size of   the input coming in and how many neurons are we gonna have
        self.biases = np.zeros((1, n_neurons)) #since the biases is a 1d array of the number of neurons
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights)+self.biases
  # in the case of neural networks there are some ways we are gonna initialise a layer, first is we are gonna have to train the model that we have saved and we want to load   in that model 
  # the output of the first layer is the input of the second layer
  layer1 = Layer_Dense(4,5) #4--> input size ; 5 is the random value
  layer2 = Layer_Dense(5,2) #5--> output of one layer is input of another
  layer1.forward(X)
  print(layer1.output)
  layer2.forward(layer1.output)
  print(layer2.output)
  ```


### Version 7: Will be coding up an activation function

- **Improvement**: Will be coding up an activation function, can be using a varity of functions the are examples that can be used,
- **Explanation For Function1**: step function == if x>0 , y==1 else y==0 . We will be using the step function as an activation function
- **Working Principle**: each neuron in the hidden layer and the output layer will have this activation function and this comes into play when we are using the inputs, the weights and the bias, when we tweak the weights and biases, the activation function will always give values between 0 and 1
- **Explanation For Function2**: next type of function is the sigmoid function, the basic reason of using a sigmoid function, is that it gives us a more granular output, which tells us what is the impact of the weights and the biases on the activation function
- **Explanation For Function3**: next type of function is the rectifier function, which states that if x>0 --> y==x else y==0, better than sigmoid function
- **Installation of an extra package**: gonna use the nnfs package to install that
  ```python
  pip install nnfs
  ```
- **Code**:
  ```python
  import numpy as np
  import nnfs
  from nnfs.datasets import spiral_data

  nnfs.init()  # Setting up a default datatype for the numpy use

  X = [[1,2,3,2.5],
    [2.0,5.0,-1.0,2.0],
    [-1.5,2.7,3.3,-0.8]]

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
  ```

  ### Version 8: Broader Aspects of the steps to code up the softmax function
  - *exponatiating every output value of the output layer*
  - *then normalizing the exponatiated values*
  - **Without using numpy**
    ```python
      import math
      layer_outputs = [4.8,1.21,2.385]
      E = math.e
      exp_values = []
      for output in layer_outputs:
        exp_values.append(E**output)
      print(exp_values)
      ## next we will be normalizing the exponitiated values
      norm_base = sum(exp_values)
      norm_values = []
      for value in exp_values:
        norm_values.append(value/norm_base)
    print(norm_values)
    print(sum(norm_values)) ##this will round up to 1
    ```
  - **with using numpy(using single layer of output)**
    ```python
    import numpy as np
    layer_outputs = [4.8,1.21,2.385]
    exp_values = np.exp(layer_outputs)
    norm_values = exp_values/np.sum(exp_values)
    print(norm_values)
    print(sum(norm_values))
    ```
  - **with using numpy(using multiple layer of output)**
    ```python
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
    ```

   ### Version 9: Adding the concept of softmax function to the neural network code
   - **Code**:
     ```python
     import numpy as np
     import nnfs
     from nnfs.datasets import spiral_data

     nnfs.init()  # Setting up a default datatype for the numpy use

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

     class Activation_Softmax:
        def forward(self, inputs):
          exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True)) #avoiding overflow since we are using the exponential function
          probabilities = exp_values/np.sum(exp_values,axis=1,keepdims=True)
          self.output = probabilities

     X,y = spiral_data(samples=100,classes=3)
     dense1 = Layer_Dense(2,3)
     activation1 = Activation_ReLU()

     dense2 = Layer_Dense(3,3)
     activation2 = Activation_Softmax()

     dense1.forward(X)
     activation1.forward(dense1.output)

     dense2.forward(activation1.output)
     activation2.forward(dense2.output)

     print(activation2.output[:5])
     ```


## Concepts Covered

- **Inputs**: Represents the data being fed into the neural network.
- **Weights**: Values that are adjusted during training to tune the model.
- **Biases**: Added to the weighted sum of inputs to introduce non-linearity.
- **Dot Product**: Efficiently computes the weighted sum of inputs using NumPy.


  

