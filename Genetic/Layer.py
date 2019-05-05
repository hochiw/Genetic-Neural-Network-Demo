import random
import numpy as np

# Layers in the neural network
class Layer:

    def __init__(self,numL,numN,activation=None,weights=None,bias=None):

        # If weights are not provided then generate a random set of weights
        if not weights:
            self.weights = np.random.rand(numL,numN)
        else:
            self.weights = weights

        # if bias values are not provided then generate a random set of bias values
        if not bias:
            self.bias = np.random.rand(numN)
        else:
            self.bias = bias

        # Initialize other values
        self.numL = numL
        self.numN = numN
        self.activation = activation
        self.error = None
        self.delta = None

    # Activation functions
    def tanh(x):
        return np.tanh(x)

    def sigmoid(x):
        return 1/(1+np.exp(-x))

    def reLu(x):
        return np.maximum(0,x)

    # Function that Initializes the data for activation
    def activate(self,x):
        # Multiply the weights with the input data then add the bias values
        dt = np.dot(x,self.weights) + self.bias
        return self.activate_i(dt)

    # A helper function that feeds the modified data through the layer
    def activate_i(self,x):

        # Simply return the data if an activation function is not provided
        if not self.activation:
            return x

        # Feed the data through the provided activation function
        if self.activation == "sigmoid":
            return Layer.sigmoid(x)

        if self.activation == "relu":
            return Layer.reLu(x)

        if self.activation == "tanh":
            return Layer.tanh(x)
        return x
