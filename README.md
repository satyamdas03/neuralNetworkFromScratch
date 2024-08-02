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

## Concepts Covered

- **Inputs**: Represents the data being fed into the neural network.
- **Weights**: Values that are adjusted during training to tune the model.
- **Biases**: Added to the weighted sum of inputs to introduce non-linearity.
- **Dot Product**: Efficiently computes the weighted sum of inputs using NumPy.


  

