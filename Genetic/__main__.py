from Genetic import Genetic 

def main():

    # All the required data for the genetic algorithm
    population = 100
    generations = 100
    selection_percentage = 0.1
    mutation_rate = 0.2
    inp = 2
    output = 1

    # Initialize and run the genetic algorithm
    GA = Genetic(population,generations,selection_percentage,mutation_rate,inp,output)
    GA.start()
    GA.best_fit.getConfig()


if __name__ == '__main__':
    main()
