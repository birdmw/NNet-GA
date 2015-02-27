from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainer_extra import *

def evolve():
    for G in range (GENERATIONS):
        print "GENERATION: ",G
        population.creatureList[0].printMe()
        population.mutate()
        population.train(TRAINING_SETS)
        population.prune()
        population.repopulate()

if __name__ == "__main__":
    CREATURE_COUNT = 1000
    NEURON_COUNT= 4
    INPUT_COUNT = 1
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = 2
    GENERATIONS = 5
    TRAINING_SETS = 4

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve()

    bestCreature = population.creatureList[0]
    population.train(TRAINING_SETS)
    bestCreature.run()
    printFinalOuts(population)


