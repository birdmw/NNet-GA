from population import *
from multiprocessing import Pool, cpu_count, Process
from geneticTrainerHelper import *

def evolve(population, generations, trainSets, battles, pattLength=0,killPercent=0.5,matePerc,maxMutate):
    for G in range (generations):
        print "GENERATION: ",G
        percentComplete = G/float(generations)
        percentLeft = 1.0 - percentComplete
        bestID = findBestId(population)
        printFinalOuts(population, bestID)
        #print "Top MU: ",population.creatureList[0].ELO.mu,"Top sigma: ",population.creatureList[0].ELO.sigma, "Top fitness: ",population.creatureList[0].fitness
        #population.mutate()
        
        population.train([trainSets, battles,pattLength])
        population.prune(killPercent)
        mutateMagnitude = .01 * maxMutate * percentLeft
        population.repopulate(matePerc, mutateMagnitude)


def main():
    CREATURE_COUNT = 100
    NEURON_COUNT= 10
    INPUT_COUNT = 0
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = 1
    GENERATIONS = 10
    WARS = 7
    BATTLES = CREATURE_COUNT*2
    INITIAL_MUTATE = 1
    KILL_PERCENT = .50

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    evolve(population, GENERATIONS, WARS, BATTLES, pattLength=50,KILL_PERCENT)
    
    bestID = findBestId(population)
    population.train([WARS, BATTLES])
    for i in range (3):
        population.train([1, 0])
        printFinalOuts(population, bestID)
    #printFinalOuts(population)
    #population.train([TRAINING_SETS, BATTLES])
    #printFinalOuts(population)
    #population.train(TRAINING_SETS, BATTLES)
    printFinalOuts(population)

if __name__ == "__main__":
    main()
