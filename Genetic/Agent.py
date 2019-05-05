import random
import numpy as np
from Layer import Layer

# Agents that are used to simulate randomized neural network
class Agent:
    def __init__(self, inp ,output, empty_layers = False):
        act_list = ['tanh','sigmoid','relu']

        # If the agent isn't a children then feed it with random layers
        self.layers = []
        if not empty_layers:
            self.layers = self.feed_layer(inp, output, 1, 100, 1, 500)

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
        layers = []

        # List of the activation functions
        act_list = ['tanh','sigmoid','relu']

        # Generate random number of neurons
        rdn_in = random.randint(min_num_neuron,max_num_neuron)
        rdn_out = random.randint(min_num_neuron,max_num_neuron)

        # Append the in layer
        layers.append(Layer(inp, rdn_in, random.choice(act_list)))

        # Append random hidden layers in between
        for _ in range(random.randint(min_num_layers,max_num_layers)):
            layers.append(Layer(rdn_in, rdn_out, random.choice(act_list)))
            rdn_in = rdn_out
            rdn_out = random.randint(min_num_neuron, max_num_neuron)

        # Append the output layer
        layers.append(Layer(rdn_in, output, random.choice(act_list)))

        return layers

    # Function that feeds the data through the neural nerwork
    def feed_forward(self, x):
        for layer in self.layers:
            x = layer.activate(x)
        return x

    # Function that handles the result
    def predict(self, x):
        result = self.feed_forward(x)
        return int(np.amax(result))

    # Function and prints the configuation of a neural network
    def getConfig(self):
        print("The best fit neural network has {0} layers with the fitness of {1}.\n".format(len(self.layers),self.fitness))
        for layer in range(len(self.layers)):
            print("Layer {0} has input: {1} and output: {2}\n".format(layer,self.layers[layer].numL,self.layers[layer].numN))

