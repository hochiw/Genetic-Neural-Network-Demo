from Genetic import Genetic

def main():

    # All the required data for the genetic algorithm
    population = 100
    generations = 1000
    selection_percentage = 0.2
    mutation_rate = 0.3
    connection_chance = 0.5
    max_layers = 10
    max_nodes = 10
    inp = 1
    output = 3
    # 1-33: 0 34-66: 1 67 - 100: 2

    # Initialize and run the genetic algorithm
    GA = Genetic(population,generations,selection_percentage,mutation_rate,connection_chance,max_layers,max_nodes,inp,output)
    GA.start()



if __name__ == '__main__':
    main()
