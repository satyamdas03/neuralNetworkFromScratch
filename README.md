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
