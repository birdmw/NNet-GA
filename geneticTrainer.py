from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainerHelper import *

def evolve(population):
    for G in range (GENERATIONS):
        percentComplete = G/float(GENERATIONS)
        percentLeft = 1.0 - percentComplete
        print "GENERATION: ",G
        bestID = findBestId(population)
        printFinalOuts(population, bestID)
        MUTATE_AMOUNT = INITIAL_MUTATE * percentLeft
        population.mutate(constant = MUTATE_AMOUNT)
        population.train([WARS, BATTLES])
        population.prune(KILL_PERCENT)
        MUTATE_AMOUNT = .01 * INITIAL_MUTATE * percentLeft
        population.repopulate(MATE_PERCENT, MUTATE_AMOUNT)

if __name__ == "__main__":
    CREATURE_COUNT = 100
    NEURON_COUNT= 20
    INPUT_COUNT = 1
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = NEURON_COUNT*2
    GENERATIONS = 100
    WARS = 6
    BATTLES = CREATURE_COUNT
    INITIAL_MUTATE = 10
    KILL_PERCENT = .75
    MATE_PERCENT = .50

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve(population)
    population.train([WARS, BATTLES])
    bestID = findBestId(population)
    for i in range (3):
        population.train([1, 0])
        printFinalOuts(population, bestID)
