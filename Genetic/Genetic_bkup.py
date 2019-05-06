import random
import numpy as np
from Agent import Agent
from Layer import Layer

# The main genetic algorithm class
class Genetic:
    # Initialize data
    def __init__(self, population, generations, selection_percentage, mutation_rate, connection_chance,maxL,maxN,inp, output):
        # Number of inputs and outputs
        self.inp = inp
        self.output = output

        # Number of agents
        self.population = population

        # Number of generations
        self.generations = generations

        # How many percent of the top agents we select
        self.selection_percentage = selection_percentage

        # How often does a mutation occur
        self.mutation_rate = mutation_rate

        self.connection_chance = connection_chance

        self.maxL = maxL

        self.maxN = maxN

        # Agent with the highest fitness are stored here
        self.best_fit = None

        # Initialize agents
        self.agents = self.init_agents()

    def init_agents(self):
        # Create agents according to the given data
        return [Agent(self.inp, self.output, self.connection_chance,self.maxL,self.maxN) for _ in range(self.population)]

    # Function that computes the fitness of all the agents
    def fitness(self, agents):
        for agent in agents:
            # Calculate the error between the true and the output value
            error = abs(agent.guess - agent.true_num)

            # Return 1 if the agent has it correctly or else return the error
            if error != 0:
                if error > 1:
                    agent.fitness = 1/error
                else:
                    agent.fitness = error
            else:
                agent.fitness = 1

        return agents

    # Selection function that selects top scoring agents
    def selection(self, agents):
        # Sort the agents by their fitness value
        agents = sorted(self.agents, key=lambda x: x.fitness, reverse=True)

        # Select agents according to the given percentage
        agents = self.agents[:int(self.selection_percentage * len(self.agents))]

        # Print out the top 10 agents
        print('\n'.join(map(str,sorted(agents[:10],key=lambda x:x.fitness,reverse=True))))
        print("Average fitness: {0}".format(sum(i.fitness for i in agents)/len(agents)))

        return agents

    # Function that selects two random parents from the list for repopulation
    def crossover(self, agents):
        offspring = []
        act_list = ['tanh','sigmoid','relu']

        # Repopulate enough to fulfill the desired population
        for _ in range(int((self.population - len(self.agents))/ 2)):
            # Select two random parents
            parent_1 = random.choice(agents)
            parent_2 = random.choice(agents)

            # Create two empty agents
            child_1 = Agent(self.inp, self.output, self.connection_chance,self.maxL, self.maxN, True)
            child_2 = Agent(self.inp, self.output, self.connection_chance,self.maxL, self.maxN, True)

            split_1 = random.randint(1,len(parent_1.layers))
            split_2 = random.randint(0,len(parent_2.layers) - 1)

            child_1.layers = parent_1.layers[0:split_1] + parent_2.layers[split_2: len(parent_2.layers)]

            split_1 = random.randint(0, len(parent_1.layers) - 1)
            split_2 = random.randint(1,len(parent_2.layers))

            child_2.layers = parent_2.layers[0:split_2] + parent_1.layers[split_1:len(parent_1.layers)]

            # Attempt to reconnect the links
            for layer in child_1.layers:
                for node in layer.nodes:
                    for layer_i in parent_1.layers:
                        for layer_f in parent_2.layers:
                            if node in layer_i.connections.keys():
                                for item in layer_i.connections[node]:
                                    if item[0] in layer.nodes:
                                        layer.create_connection(node,item[0],item[1])

                            elif node in layer_f.connections.keys():
                                for item in layer_f.connections[node]:
                                    if item[0] in layer.nodes:
                                        layer.create_connection(node,item[0],item[1])

                            else:
                                for output in child_1.layers[-1].nodes:
                                    layer.create_connection(node,output,np.random.uniform(-1,1))

            for layer in child_2.layers:
                for node in layer.nodes:
                    for layer_i in parent_1.layers:
                        for layer_f in parent_2.layers:
                            if node in layer_i.connections.keys():
                                for item in layer_i.connections[node]:
                                    if item[0] in layer.nodes:
                                        layer.create_connection(node,item[0],item[1])

                            elif node in layer_f.connections.keys():
                                for item in layer_f.connections[node]:
                                    if item[0] in layer.nodes:
                                        layer.create_connection(node,item[0],item[1])

                            else:
                                for output in child_2.layers[-1].nodes:
                                    layer.create_connection(node,output,np.random.uniform(-1,1))


            # Put the children in the list
            offspring.append(child_1)
            offspring.append(child_2)

        # Extend the original agents list with the generated offspring list
        agents.extend(offspring)
        return agents

    # Function that handles random mutation of an agent
    def mutation(self, agents):

        # Iterate through each layer of each agent
        for agent in agents:
            num_layers = len(agent.layers)

            for layer in range(num_layers):
                next_layers = agent.layers[layer + 1: num_layers]
                if len(next_layers) == 0:
                    continue
                for node in agent.layers[layer].nodes:

                    # Decides whether a mutation occurs or not
                    if random.uniform(0.0, 1.0) <= self.mutation_rate:

                        random_layer = random.choice(next_layers)
                        random_node = random.choice(random_layer.nodes)
                        agent.layers[layer].create_connection(node,random_node,np.random.uniform(-1,1))

                    if random.uniform(0.0, 1.0) <= self.mutation_rate:
                        agent.layers[layer].bias = np.random.uniform(-1,1)


        return agents

    # Function that handles the simulation, can be a game or simple functions
    # In my case the system generates two numbers and simply mutiply them together
    # then see if the neural network can figure it's a multiplication with the
    # two numbers fed through it
    def simulate(self, i, agent):

        # Generate 2 random numbers
        num = random.randint(1,101)

        true = -1
        if 1 <= num <= 33:
            true = 0
        elif 34 <= num <= 66:
            true = 1
        elif 67 <= num <= 100:
            true = 2


        # Feed the two numbers through the network
        guess = agent.predict([num])

        # Store both numbers
        # (Storing the true number may not be good for an algorithm like this)
        # (But I couldn't come up with a better fitness function so I had no choice)
        if true == guess:
            return True
        return False

    # Function that encapsulate all the process of the genetic algorithm
    def start(self):
        # Rerun the processes for n generations
        for generation in range(self.generations):
            print("Generation " + str(generation))


            # Run through the simulation process for each agent
            for agent in range(len(self.agents)):
                count = 0
                while(self.simulate(agent,self.agents[agent])):
                    self.agents[agent].fitness += 1
                    count += 1
                    if count % 10 == 0:
                        print("Agent {0} has the fitness of {1}".format(self.agents[agent].id,self.agents[agent].fitness))
                self.agents[agent].fitness -= 1



            # Run through all the processes
            #self.agents = self.fitness(self.agents)

            self.agents = self.selection(self.agents)

            self.agents = self.crossover(self.agents)

            self.agents = self.mutation(self.agents)


            self.best_fit= max(self.agents, key=lambda x:x.fitness)

        while(True):
            x = int(input("Input X: "))
            result = self.best_fit.predict([x])
            print("Best fit thinks it is : {0}".format(result))
