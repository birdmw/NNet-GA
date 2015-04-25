from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainerHelper import *

def evolve(population):
    for G in range (GENERATIONS):
        print "GENERATION: ",G
        bestID = findBestId(population)
        printFinalOuts(population, bestID)
        population.mutate()
        population.train([WARS, BATTLES])
        population.prune()
        population.repopulate()

if __name__ == "__main__":
    CREATURE_COUNT = 200
    NEURON_COUNT= 4
    INPUT_COUNT = 1
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = NEURON_COUNT*2
    GENERATIONS = 1000
    WARS = 1
    BATTLES = CREATURE_COUNT**2

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve(population)
    population.train([WARS, BATTLES])
    bestID = findBestId(population)
    for i in range (30):
        population.train([WARS, BATTLES])
        printFinalOuts(population, bestID)
