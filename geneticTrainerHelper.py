from copy import *

def printFinalOuts( population ):
    print "training outs:"
    for o in range (len(population.trainingCreature.output)):
        print population.trainingCreature.output[o].outbox
    populationCopy = deepcopy(population)
    while len (populationCopy.creatureList) > 1:
        populationCopy.sortByMu()
        populationCopy.creatureList.pop()
        if len ( populationCopy.creatureList ) > 1:
            populationCopy.sortBySigma()
            populationCopy.creatureList.reverse()
            populationCopy.creatureList.pop()
    print "best creature outs: "
    for o in range ( len ( populationCopy.creatureList[0].output ) ): 
        print populationCopy.creatureList[0].output[o].outbox

    print "MU: ",populationCopy.creatureList[0].ELO.mu,"Sigma: ",populationCopy.creatureList[0].ELO.sigma, "fitness: ",populationCopy.creatureList[0].fitness
    
