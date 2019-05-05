import random
import numpy as np
from Layer import Layer

# Agents that are used to simulate randomized neural network
class Agent:
    def __init__(self, inp ,output, connection_chance, max_L, max_N, empty_layers = False):
        act_list = ['tanh','sigmoid','relu']
        self.connection_chance = connection_chance

        # If the agent isn't a children then feed it with random layers
        self.layers = []
        if not empty_layers:
            self.layers = self.feed_layer(inp, output, 1, max_L, 1, max_N)

        self.create_connections(self.connection_chance)

        #print("Agent: {0}, Layers: {1}, Nodes: {2}, Connections: {3}".format(id(self),len(self.layers),len([node for layer in self.layers for node in layer.nodes]),
                                                                             #len([connection for layer in self.layers for connection in layer.connections])))

        # Initialize data
        self.id = id(self)
        self.fitness = 0
        self.true_num = -1
        self.guess = -1


    # Function that handles print(Agent)
    def __str__(self):
        return "Agent: " + str(id(self)) + " Fitness: " + str(self.fitness)

    # Function that adds the given layer to the layer list
    def add_layer(self,layer):
        self.layers.append(layer)

    # Function that generate random layers with random number of neurons given a upper and lower limit
    def feed_layer(self, inp, output, min_num_layers, max_num_layers, min_num_neuron, max_num_neuron):
        layer_num = 0
        layers = []

        # List of the activation functions
        act_list = ['tanh','sigmoid','relu']

        # Append the in layer
        layers.append(Layer(layer_num,inp,random.choice(act_list)))
        layer_num += 1

        # Append random hidden layers in between
        for _ in range(random.randint(min_num_layers,max_num_layers)):
            layers.append(Layer(layer_num, random.randint(min_num_neuron,max_num_neuron), random.choice(act_list)))
            layer_num += 1
        # Append the output layer
        layers.append(Layer(layer_num, output, random.choice(act_list)))
        layer_num += 1

        return layers

    def create_connections(self, connection_chance):
        num_layers = len(self.layers)
        for layer in range(num_layers):
            next_layers = self.layers[layer + 1: num_layers]
            if len(next_layers) == 0:
                return 
            for node in self.layers[layer].nodes:
                for i in range(random.randint(1,20)):
                    if np.random.uniform(0,1) <= connection_chance:
                        random_layer = random.choice(next_layers)
                        random_node = random.choice(random_layer.nodes)
                        self.layers[layer].create_connection(node,random_node,np.random.uniform(-1,1))
                    else:
                        for output in self.layers[-1].nodes:    
                            self.layers[layer].create_connection(node,output,np.random.uniform(-1,1))

                

    # Function that feeds the data through the neural nerwork
    def feed_forward(self, x):
        if len(x) == len(self.layers[0].nodes):
            for i in range(len(x)):
                self.layers[0].nodes[i].value = x[i]
        else:
            return None
        for layer in self.layers:
            layer.activate()
        return [x.getValue() for x in self.layers[-1].nodes]

    # Function that handles the result
    def predict(self, x):
        result = self.feed_forward(x)
        return np.argmax(result)
