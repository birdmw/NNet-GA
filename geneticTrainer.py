from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainerHelper import *

def evolve(population):
    for G in range (GENERATIONS):
        print "GENERATION: ",G
        printFinalOuts(population)
        #population.mutate()
        population.train([WARS, BATTLES])
        population.prune()
        population.repopulate()

if __name__ == "__main__":
    CREATURE_COUNT = 40
    NEURON_COUNT= 3
    INPUT_COUNT = 1
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = NEURON_COUNT +1
    GENERATIONS = 13
    WARS = 7
    BATTLES = CREATURE_COUNT**2

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve(population)

    printFinalOuts(population)
    population.train([WARS, BATTLES])
    printFinalOuts(population)
    population.train([WARS, BATTLES])
    printFinalOuts(population)

