from copy import *

def findBestId(population):
    populationCopy = deepcopy(population)
    while len (populationCopy.creatureList) > 1:
        populationCopy.sortByMu()
        populationCopy.creatureList.pop()
        if len ( populationCopy.creatureList ) > 1:
            populationCopy.sortBySigma()
            populationCopy.creatureList.reverse()
            populationCopy.creatureList.pop()
    return populationCopy.creatureList[0].ID      
  


def printFinalOuts( population, ID ):
    print "training outs:"
    for o in range (len(population.trainingCreature.output)):
        print population.trainingCreature.output[o].outbox
    for c in population.creatureList:
        if c.ID == ID:
            index = population.creatureList.index(c)
    print "best creature outs: "
    for o in range ( len ( population.creatureList[index].output ) ): 
        print population.creatureList[index].output[o].outbox

    print "ID", population.creatureList[index].ID  ,"MU: ",population.creatureList[index].ELO.mu,"Sigma: ",population.creatureList[index].ELO.sigma, "fitness: ",population.creatureList[index].fitness
    
