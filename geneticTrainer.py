from population import *
from geneticHelper import *
from trainer import *
from random import *
from Tkinter import *
import creatureGUI_2 as cg2
from trueskill import Rating, quality_1vs1, rate_1vs1
import numpy as np
import pylab as P

def evolve(population, trainData, generations=10, setsPerGen=1,battles = "Random"):
    for G in range (generations):
        print "GENERATION: ",G
        for t in range(setsPerGen):
            print "  set: ", t
            trainPopulation(population, trainData, setsPerGen)
            battle(population,battles)
        prune(population)
        mutateIDs = population.repopulate()
        mutate(population, mutateIDs)

def prune ( pop , killPercent = .50 ):
##    print "before prune statistics:"
##    pop.printAverages()
    saveIDs = list()
    saveCount = int(len ( pop.creatureList ) * max(min(1.0-killPercent,1.0),0.0))
    while len(saveIDs) < (saveCount):
        pop.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
        i=0
        while (pop.creatureList[i].ID in saveIDs):
            i+=1
        saveIDs.append(pop.creatureList[i].ID)
        #saveTheChildren
##        pop.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
##        if len(saveIDs) < (saveCount):
##            i=0
##            while (pop.creatureList[i].ID in saveIDs):
##                i+=1
##            saveIDs.append(pop.creatureList[i].ID)

    finalCreatureList = []
    for creature in pop.creatureList:
      if (creature.ID in saveIDs):
        finalCreatureList.append(creature)
    pop.creatureList = finalCreatureList
    pop.sortByID()
##    print "after prune statistics:"
##    pop.printAverages()

def mutate (pop, mutateIDs, mutateAmount = .01):

    for ID in mutateIDs:
        index = pop.IDToIndex(ID)

        #on average mutate one property of one synapse
        for s in range(len(pop.creatureList[index].synapseList)):
            if random()< 1/len(pop.creatureList[index].synapseList):
                for p in range(len(pop.creatureList[index].synapseList[s].propertyList)):
                    if random()< 1/len(pop.creatureList[index].synapseList[s].propertyList):
                        propertyMutateAmount = p*mutateAmount
                        pop.creatureList[index].synapseList[s].propertyList[p] = max(min(gauss( pop.creatureList[index].synapseList[s].propertyList[p] , propertyMutateAmount),1000),-1000)
        #on average mutate one property of one neuron
        for n in range(len(pop.creatureList[index].neuronList)):
            if random()<1/len(pop.creatureList[index].neuronList):
                for p in range(len(pop.creatureList[index].neuronList[n].propertyList)):
                    if random()<1/len(pop.creatureList[index].neuronList[n].propertyList):
                        propertyMutateAmount = p*mutateAmount
                        pop.creatureList[index].neuronList[n].propertyList[p] = max(min(gauss( pop.creatureList[index].neuronList[n].propertyList[p] , propertyMutateAmount),1000),-1000)
##    print "after mutate statistics:"
##    pop.printAverages()

def battle( pop, battles = "Random" ):
    if battles == "Random":
        battles = min(int(len(pop.creatureList)/2+(random()*len(pop.creatureList)**2)),10000)
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

def trainPopulation(population, trainData, setsPerGen, tSetIndex = None):
    for c in range(len(population.creatureList)):
        #trainCreature(population, c, trainData, setsPerGen, tSetIndex)
        trainCreature_InvertedAbsDist(population, c, trainData, setsPerGen, tSetIndex)

def trainCreature(population, c, trainData, setsPerGen, tSetIndex = None, huntWindow = 4):
    #accepts a creature and a training set
    #runs the creature for the length of the dataset
    #sets the creatures fitness using hunt
    if tSetIndex == None:
        tSet = trainData.randomSet()
        tSetIndex = trainData.data.index(tSet)
    else:
        tSet = trainData.data[tSetIndex]
    cycleFitnessList = []
    creatureOutputArray = []
    for cyc in range(len(tSet[1][0])): # for each cycle
        inputs = []
        for i in range(len(tSet[0])): # for each input
            inputs.append(tSet[0][i][cyc]) # make a list of input neuron inputs
        population.creatureList[c].run(1, inputs) #
        cycleFitnessList.append( judgeFitnessWithHunt( population.creatureList[c] , trainData, cyc, tSetIndex, huntWindow ) ) #judge
    newAvgFit = sum(cycleFitnessList)/float(len(cycleFitnessList)) #then average all together for the creature
    population.creatureList[c].fitness = ( ( setsPerGen - 1 ) * population.creatureList[c].fitness + newAvgFit) / setsPerGen


def trainCreature_InvertedAbsDist(population, c, trainData, setsPerGen, tSetIndex = None):


    #!!!!!!!!!!!!!!!!!!!WARNING ONLY WORKS WITH ONE TRAINING SET!!!!!!!!!!!


    #accepts a creature and a training set
    #runs the creature for the length of the dataset
    #sets the creatures fitness using hunt
    if tSetIndex == None:
        tSet = trainData.randomSet()
        tSetIndex = trainData.data.index(tSet)
    else:
        tSet = trainData.data[tSetIndex]
    totalFitness = 0
    creatureOutputArray = []
    cycles =len(tSet[1][0])
    if cycles == 0:
        print 'AHHHH NO VALUES IN DOCY OUTPUT AHHHHHHHHHHHHHHHHHHHHHHHHHHHHH'
        return
    for cyc in range(cycles): # for each cycle
        inputs = []
        for i in range(len(tSet[0])): # for each input
            inputs.append(tSet[0][i][cyc]) # make a list of input neuron inputs

        population.creatureList[c].run(1, inputs) #
        totalFitness+=judgeCycleFitnessInvertedAbsDistance(population.creatureList[c] , trainData, cyc, tSetIndex)

    newAvgFit = totalFitness/cycles
    population.creatureList[c].fitness = newAvgFit


def judgeFitnessWithHunt(creature, trainData, cyc, tSetIndex, huntWindow=2):
    neuronDiffList = []
    for outputIndex in range(len(creature.output)): #for each output
        windowIndex = 0
        minDiff = abs(creature.output[outputIndex].outbox - trainData.data[tSetIndex][1][outputIndex][cyc])#initialize minDiff to prevent calling something that doesnt exist
        while windowIndex <= abs(huntWindow): #for each window
            if cyc+windowIndex < len(trainData.data[tSetIndex][1][outputIndex]):# if it didnt roll off either end
                minDiff = min( minDiff, abs(creature.output[outputIndex].outbox - trainData.data[tSetIndex][1][outputIndex][cyc+windowIndex])) #find the minimum
            if cyc-windowIndex >= 0:
                minDiff = min( minDiff, abs(creature.output[outputIndex].outbox - trainData.data[tSetIndex][1][outputIndex][cyc-windowIndex]))
            windowIndex+=1
        neuronDiffList.append(minDiff)
    avgDiff = sum(neuronDiffList)/float(len(neuronDiffList)) #and average
    return myGauss(avgDiff)

def judgeCycleFitnessInvertedAbsDistance(creature, trainData, cyc, tSetIndex):
    neuronDiffList = []
    totalDiff=1 #Minimum distance is one (Prevents divide by zero, and infinity fitness
    for outputIndex in range(len(creature.output)): #for each output
        totalDiff += abs(creature.output[outputIndex].outbox - trainData.data[tSetIndex][1][outputIndex][cyc])
    return 1/totalDiff

def arrayAbsSum(array):
    total = 0.0
    for a in array:
        total += abs(array)
    return total

def arrayAbsDifference (arrayOne,arrayTwo):
    array=[]
    for i in range( len(arrayTwo) ):
        array.append( abs(arrayOne[i] - arrayTwo[i]) )
    return array

def myGauss(x,mu=0.0,sig=1.0):
    '''
    Uses mu and sig to create a gaussian, then uses x as an input to the gaussian, returning the probability that x would be seen in the gaussian
    '''
    if sig == 0.0:
        if x==mu:
            return 1.0
        else:
            return 0.0
    p1 = -np.power(x-mu,2.)
    p2 = 2*np.power(sig,2.)
    g = np.exp(p1/p2)
    return g

def main(): #trainData is docy() type
    #root = Tk()
    population = DummyPopulation(CreatureCount=500, NeuronCount=3, InputCount=1, OutputCount=1)
    trainData = docy()
    #generateSinTracker(self, inputCount, outputCount, cycleCount=360, a=1, b=1, c=0, reps=1)
    cycleCount=360
    a=1
    b=1
    c=0
    trainData.generateSinTracker(len(population.creatureList[0].input), len(population.creatureList[0].output),cycleCount,a,b,c)
    #trainData.generateConstant(len(population.creatureList[0].input), len(population.creatureList[0].output), constantIn=1, constantOut=5)
##    print "ins"
##    print trainData.data[0][0]
##    print "outs"
##    print trainData.data[0][1]

    generations=50
    setsPerGen=1
    battles = 5000
    #evolve(population, trainData, generations=3, setsPerGen=1,battles = "Random")
    evolve(population, trainData, generations, setsPerGen,battles)
    print 'Training newbies...'
    trainPopulation(population, trainData, setsPerGen)
    battle(population,battles)
    battle(population,battles)
    battle(population,battles)

    bestCreature = findBestCreature(population)

    print 'Best creatures offset:', bestCreature.offset

    population.creatureList.sort(key = lambda x: abs(x.offset), reverse=False)
    sortedBestOffset = population.creatureList[0].offset

    print 'Best offset:', sortedBestOffset

    offsetList = []
    for creat in population.creatureList:
        offsetList.append(creat.offset)

    P.figure()
    bins = np.linspace(-2.0, 2.0, num=25)
    # the histogram of the data with histtype='step'
    n, bins, patches = P.hist(offsetList, bins, histtype='bar', rwidth=1)


    print 'PRUNING...'
    prune(population)

    bestCreature = findBestCreature(population)

    print 'Best creatures offset:', bestCreature.offset

    population.creatureList.sort(key = lambda x: abs(x.offset), reverse=False)
    sortedBestOffset = population.creatureList[0].offset

    print 'Best offset:', sortedBestOffset

    offsetList = []
    for creat in population.creatureList:
        offsetList.append(creat.offset)

    P.figure()
    bins = np.linspace(-2.0, 2.0, num=25)
    # the histogram of the data with histtype='step'
    n, bins, patches = P.hist(offsetList, bins, histtype='bar', rwidth=1)
    P.setp(patches, 'facecolor', 'r')


    P.show()
    #docy.data[set][io][put][cycle]

    #gui = cg2.CreatureGUI_Beta(root,bestCreature,trainData.data[0][0])
    #root.geometry("900x500+300+300")
    #root.mainloop()


##    print "ins"
##    print trainData.data[0][0]
##    print "outs"
##    print trainData.data[0][1]

if __name__ == "__main__":
    main()
