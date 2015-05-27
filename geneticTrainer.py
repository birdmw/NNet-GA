from population import *
from geneticHelper import *
from environment import *
from random import *

def evolve(population, trainData, generations = 10):#
    for G in range (generations):#one training set per generation
        print "GENERATION: ",G#
        trainPopulation(population, trainData)#
        battle(population)#
        prune(population)#
        mutate(population, population.repopulate())#

def prune ( pop , killPercent = .50 ):
    saveIDs = list()
    saveCount = len ( pop.creatureList ) * max(min(1.0-killPercent,1.0),0.0)
    while len(saveIDs) < (saveCount):
        pop.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
        i=0
        while (pop.creatureList[i].ID in saveIDs):
            i+=1
        saveIDs.append(pop.creatureList[i].ID)
        if len(saveIDs) < (saveCount):
            pop.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
            i=0
            while (pop.creatureList[i].ID in saveIDs):
                i+=1
            saveIDs.append(pop.creatureList[i].ID)
    finalCreatureList = []
    for creature in pop.creatureList:
      if (creature.ID in saveIDs):
        finalCreatureList.append(creature)
    pop.creatureList = finalCreatureList
    pop.sortByID()

def mutate (pop, mutateIDs, mutateAmount = .01):
    for ID in mutateIDs:
        index = pop.IDToIndex(ID)
        for s in range(len(pop.creatureList[index].synapseList)):
            for p in range(len(pop.creatureList[index].synapseList[s].propertyList)):
                pop.creatureList[index].synapseList[s].propertyList[p] = max(min(gauss( pop.creatureList[index].synapseList[s].propertyList[p] , mutateAmount),1000),-1000)
        for n in range(len(pop.creatureList[index].neuronList)):
            for p in range(len(pop.creatureList[index].neuronList[n].propertyList)):
                pop.creatureList[index].neuronList[n].propertyList[p] = max(min(gauss( pop.creatureList[index].neuronList[n].propertyList[p] , mutateAmount),1000),-1000)

def battle( pop, battles = None ):
    if battles == None:
        battles = int(random()*len(pop.creatureList)**2)
    for b in range(battles):
        creature1 = choice( pop.creatureList )
        creature2 = choice( pop.creatureList )
        updateELO(creature1, creature2)

def updateELO( creature1, creature2 ):
    if creature1.fitness > creature2.fitness:
        creature1.ELO,creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO)
    elif creature2.fitness > creature1.fitness:
        creature2.ELO,creature1.ELO = rate_1vs1(creature2.ELO,creature1.ELO)
    else:
        creature1.ELO, creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO, drawn=True)

def main(population = None, trainData = None): #trainData is docy() type
    if population == None:
        population = Population()
    if trainData == None:
        trainData = docy()
        trainData.generateSin(len(population.creatureList[0].input), len(population.creatureList[0].output))
    evolve(population, trainData)

if __name__ == "__main__":
    main()
