from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainerHelper import *

def evolve(population, generations, trainSets, battles, pattLength=0):
    for G in range (generations):
        print "GENERATION: ",G
        #print "Top MU: ",population.creatureList[0].ELO.mu,"Top sigma: ",population.creatureList[0].ELO.sigma, "Top fitness: ",population.creatureList[0].fitness
        #population.mutate()
        population.train([trainSets, battles,pattLength])
        population.prune()
        population.repopulate()


def main():
    CREATURE_COUNT = 100
    NEURON_COUNT= 10
    INPUT_COUNT = 0
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = 1
    GENERATIONS = 10
    TRAINING_SETS = 7
    BATTLES = CREATURE_COUNT*2

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve(population, GENERATIONS, TRAINING_SETS, BATTLES, pattLength=50)
    
    #printFinalOuts(population)
    #population.train([TRAINING_SETS, BATTLES])
    #printFinalOuts(population)
    #population.train(TRAINING_SETS, BATTLES)
    printFinalOuts(population)


if __name__ == "__main__":
    main()
