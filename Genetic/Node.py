import random
import numpy as np

# Nodes in the neural network layer
class Node:

    def __init__(self,  activation=None):
        self.value = -1
        self.activation = self.activation_select(activation)

    def getValue(self):
        if self.activation:
            value = self.value
            self.value = -1
            return self.activation(value)
        else:
            value = self.value
            self.value = -1
            return self.value

    def feed(self, value):
        self.value += value

     # Activation functions
    def tanh(self,x):
        return np.tanh(x)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def reLu(self,x):
        return np.maximum(0,x)

    def activation_select(self,type_i):
        if type_i == None:
            return None

        if type_i == "tanh":
            return self.tanh
        if type_i == "sigmoid":
            return self.sigmoid
        if type_i == "relu":
            return self.reLu
