import random
import numpy as np
from Node import Node
# Layers in the neural network
class Layer:

    def __init__(self,layer_num,numN,activation=None,weights=None,bias=None):
        self.connections = {}
        self.layer_num = layer_num
        self.nodes = self.init_nodes(numN, activation)
        

        if bias == None:
            self.bias = np.random.uniform(-1,1)
        else:
            self.bias = bias

    def init_nodes(self,numN, activation):
        return [Node(self.layer_num,activation) for _ in range(numN)]

    def create_connection(self, nodeA, nodeB, weight):
        if nodeA not in self.connections.keys():
            self.connections[nodeA] = [(nodeB,weight)]

        for i in self.connections[nodeA]:
            if i[0] == nodeB:
                return
            
        self.connections[nodeA].append((nodeB, weight))
        return

    def activate(self):
        for node in self.nodes:
            if node in self.connections.keys():
                for data in self.connections[node]:
                    other_node = data[0]
                    weight = data[1]
                    other_node.feed((node.getValue() * weight) + self.bias)

   
