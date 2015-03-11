from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainer_extra import *

def evolve():
    for G in range (GENERATIONS):
        print "GENERATION: ",G
        print "Top MU: ",population.creatureList[0].ELO.mu,"Top sigma: ",population.creatureList[0].ELO.sigma, "Top fitness: ",population.creatureList[0].fitness
        #population.mutate()
        population.train(TRAINING_SETS, BATTLES)
        population.prune()
        population.repopulate()

if __name__ == "__main__":
    CREATURE_COUNT = 40
    NEURON_COUNT= 2
    INPUT_COUNT = 1
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = 2
    GENERATIONS = 10
    TRAINING_SETS = 7
    BATTLES = CREATURE_COUNT**2

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve()
    printFinalOuts(population)
    population.train(TRAINING_SETS, BATTLES)
    printFinalOuts(population)
    population.train(TRAINING_SETS, BATTLES)
    printFinalOuts(population)

