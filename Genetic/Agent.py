import random
import numpy as np
from Node import Node

# Agents that are used to simulate randomized neural network
class Agent:
    def __init__(self, inp ,output, connection_chance, max_L, max_N, empty_layers = False):
        act_list = ['tanh','sigmoid','relu']
        self.connection_chance = connection_chance
        self.nodes = {}
        self.id = id(self)
        self.fitness = 0
        self.true_num = -1
        self.guess = -1
        self.connections = {}
        self.max_L = max_L
        self.inp = inp
        self.output = output

        # If the agent isn't a children then feed it with random layers
        self.nodes[0] = [Node(random.choice(act_list)) for _ in range(self.inp)]
        self.nodes[max_L+1] = [Node(random.choice(act_list)) for _ in range(output)]
        self.init_connections()

    # Function that handles print(Agent)
    def __str__(self):
        return "Agent: " + str(id(self)) + " Fitness: " + str(self.fitness)

    def init_connections(self):
        for i in self.nodes.keys():
            for j in self.nodes.keys():
                if i < j:
                    for node in self.nodes[i]:
                        for node_i in self.nodes[j]:
                            self.create_connection(node,node_i,np.random.uniform(-1,1))

    def create_connection(self, nodeA, nodeB, weight):
        if nodeA not in self.connections.keys():
            self.connections[nodeA] = [(nodeB,weight)]
        else:
            self.connections[nodeA].append((nodeB,weight))

    # Function that feeds the data through the neural nerwork
    def feed_forward(self, x):
        if len(x) == len(self.nodes[0]):
            for i in range(len(x)):
                self.nodes[0][i].value = x[i]
        else:
            return None

        for key in self.nodes.keys():
            for node in self.nodes[key]:
                if node in self.connections.keys():
                    for other_node in self.connections[node]:
                        value = other_node[1] * node.getValue()
                        node_i = other_node[0]
                        node_i.feed(value)

        return [x.getValue() for x in self.nodes[self.max_L + 1]]

    # Function that handles the result
    def predict(self, x):
        result = self.feed_forward(x)
        return np.argmax(result)
