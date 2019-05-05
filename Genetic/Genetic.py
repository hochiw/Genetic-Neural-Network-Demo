import random
import numpy as np
from Agent import Agent
from Layer import Layer

# The main genetic algorithm class
class Genetic:
    # Initialize data
    def __init__(self, population, generations, selection_percentage, mutation_rate,  inp, output):
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

        # Agent with the highest fitness are stored here
        self.best_fit = None

        # Initialize agents
        self.agents = self.init_agents()

    def init_agents(self):
        # Create agents according to the given data
        return [Agent(self.inp, self.output) for _ in range(self.population)]

    # Function that computes the fitness of all the agents
    def fitness(self, agents):
        for agent in agents:
            # Calculate the error between the true and the output value
            error = abs(agent.guess - agent.true_num)

            # Return 1 if the agent has it correctly or else return the error
            if error != 0:
                agent.fitness = 1/error
            else:
                agent.fitness += 1

        return agents

    # Selection function that selects top scoring agents
    def selection(self, agents):
        # Sort the agents by their fitness value
        agents = sorted(self.agents, key=lambda x: x.fitness, reverse=True)

        # Print out the top 10 agents
        print('\n'.join(map(str,agents[:10])))

        # Select agents according to the given percentage
        agents = self.agents[:int(self.selection_percentage * len(self.agents))]

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
            child_1 = Agent(self.inp, self.output, True)
            child_2 = Agent(self.inp, self.output, True)

            # Randomize a range for layer extraction
            split_1 = random.randint(0, len(parent_1.layers) - 1)
            split_2 = random.randint(0,len(parent_2.layers) - 1)

            # Extract the first portion of parent 1 and second portion of parent 2
            p1 = parent_1.layers[0:split_1]
            p2 = parent_2.layers[split_2:len(parent_2.layers)]

            # Check if the portion is empty
            if len(p1) == 0 or len(p2) == 0:
                continue

            # If the first portion have more then one value then select the last one
            # or else select the first one (which is the only one)
            first = p1[0]
            if len(p1) > 1:
                first = p1[-1]

            # Select the first value of portion 2
            second = p2[0]

            # Create a hidden layer that connects two portions
            hidden = Layer(first.numN, second.numL,random.choice(act_list))

            # Append the layer to the end of the first portion
            p1.append(hidden)

            # Combine everything
            child_1.layers = p1  + p2

            # Extract first portion from parent 2 and second portion from parent 1
            p1 = parent_2.layers[0:split_2]
            p2 = parent_1.layers[split_1:len(parent_1.layers)]

            # Check if the portions are empty
            if len(p1) == 0 or len(p2) == 0:
                continue

            # If the first portion have more then one value then select the last one
            # or else select the first one (which is the only one)
            first = p1[0]
            if len(p1) > 1:
                first = p1[-1]

            # Select the first value of portion 2
            second = p2[0]

            # Create a hidden layer that connects two portions
            hidden = Layer(first.numN, second.numL,random.choice(act_list))

            # Append the layer to the end of the first portion
            p1.append(hidden)

            # Combine everything
            child_2.layers = p1 +  p2

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
            for layer in agent.layers:

                # Decides whether a mutation occurs or not
                if random.uniform(0.0, 1.0) <= self.mutation_rate:

                    # Select random weights from the weight list
                    mask = mask = np.random.randint(0,2,size=layer.weights.shape).astype(np.bool)

                    # Generate random weights
                    r = np.random.rand(*layer.weights.shape)*np.max(layer.weights)

                    # Replace the selected weights
                    layer.weights[mask] = r[mask]

        return agents

    # Function that handles the simulation, can be a game or simple functions
    # In my case the system generates two numbers and simply mutiply them together
    # then see if the neural network can figure it's a multiplication with the
    # two numbers fed through it
    def simulate(self, i, agent):

        # Generate 2 random numbers
        num_1 = random.randint(0,100)
        num_2 = random.randint(0,100)

        # Multiply them
        true = num_1 * num_2

        # Feed the two numbers through the network
        guess = agent.predict([num_1, num_2])

        # Store both numbers
        # (Storing the true number may not be good for an algorithm like this)
        # (But I couldn't come up with a better fitness function so I had no choice)
        agent.true_num = true
        agent.guess = guess
        return

    # Function that encapsulate all the process of the genetic algorithm
    def start(self):
        # Rerun the processes for n generations
        for generation in range(self.generations):
            print("Generation " + str(generation))

            # Run through the simulation process for each agent
            for agent in range(len(self.agents)):
                self.simulate(agent,self.agents[agent])

            # Run through all the processes
            self.agents = self.fitness(self.agents)
            self.agents = self.selection(self.agents)
            self.agents = self.crossover(self.agents)
            self.agents = self.mutation(self.agents)

            # Replace the existing best fit agent with a better one
            if self.best_fit:

                # Instead of selecting a best one from each generation
                # We select the best one from all generations
                if max(self.agents, key=lambda x:x.fitness).fitness >= self.best_fit.fitness:
                    self.best_fit = max(self.agents, key=lambda x:x.fitness)
            else:

                # Select the best one from the first generation
                self.best_fit= max(self.agents, key=lambda x:x.fitness)
