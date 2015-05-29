from copy import *
import numpy as np
from math import *

def findBestCreature(population):
    population.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
    return population.creatureList[0]
  


def printFinalOuts( population, ID ):
    tempList2 = []
    tempList3 = []
    for p in range ( len ( population.creatureList[0].synapseList[0].propertyList ) ):
        for s in range ( len ( population.creatureList[0].synapseList ) ):
            tempList3 = []
            for c in range ( len ( population.creatureList ) ):
                tempList3.append(population.creatureList[c].synapseList[s].propertyList[p])
            a = np.array(tempList3)
            std3 = np.std(a)
            tempList2.append(std3)

    for p in range ( len ( population.creatureList[0].neuronList[0].propertyList ) ):
        for n in range ( len ( population.creatureList[0].neuronList ) ):
            tempList3 = []
            for c in range ( len ( population.creatureList ) ):
                tempList3.append(population.creatureList[c].neuronList[n].propertyList[p])
            a = np.array(tempList3)
            std3 = np.std(a)
            tempList2.append(std3)
    print "standard deviation of creature properties is: ", sum(tempList2) / float(len(tempList2))
            

    for c in population.creatureList:
        if c.ID == ID:
            index = population.creatureList.index(c)
            break
    
    print "exected outs:"
    for o in range (len(population.creatureList[index].expectedOutputs)):
        print population.creatureList[index].expectedOutputs[o]

    print "best creature outs: "
    for o in range ( len ( population.creatureList[index].output ) ): 
        print population.creatureList[index].output[o].outbox

    print "ID", population.creatureList[index].ID  ,"MU: ",population.creatureList[index].ELO.mu,"Sigma: ",population.creatureList[index].ELO.sigma, "fitness: ",population.creatureList[index].fitness
    
