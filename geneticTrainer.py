from population import *
from geneticHelper import *
from trainer import *
from random import *
from Tkinter import *
import creatureGUI_2 as cg2

def evolve(population, trainData, generations=10, setsPerGen=1):
    for G in range (generations):
        print "GENERATION: ",G
        for t in range(setsPerGen):
            trainPopulation(population, trainData, setsPerGen)
            battle(population)
        prune(population)
        mutateIDs = population.repopulate()
        mutate(population, mutateIDs)

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

        #on average mutate one property of one synapse
        for s in range(len(pop.creatureList[index].synapseList)):
            #if random()< 1/len(pop.creatureList[index].synapseList):
                for p in range(len(pop.creatureList[index].synapseList[s].propertyList)):
                    #if random()< 1/len(pop.creatureList[index].synapseList[s].propertyList):
                        propertyMutateAmount = p*mutateAmount
                        pop.creatureList[index].synapseList[s].propertyList[p] = max(min(gauss( pop.creatureList[index].synapseList[s].propertyList[p] , propertyMutateAmount),1000),-1000)
        #on average mutate one property of one neuron
        for n in range(len(pop.creatureList[index].neuronList)):
            #if random()<1/len(pop.creatureList[index].neuronList):
                for p in range(len(pop.creatureList[index].neuronList[n].propertyList)):
                    #if random()<1/len(pop.creatureList[index].neuronList[n].propertyList):
                        propertyMutateAmount = p*mutateAmount
                        pop.creatureList[index].neuronList[n].propertyList[p] = max(min(gauss( pop.creatureList[index].neuronList[n].propertyList[p] , propertyMutateAmount),1000),-1000)

def battle( pop, battles = "Random" ):
    if battles == "Random":
        battles = min(int(random()*len(pop.creatureList)**2),10000)
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

def trainPopulation(pop, trainData, setsPerGen, tSetIndex = None):
    for c in pop.creatureList:
        trainCreature(c, trainData, setsPerGen)

def trainCreature(creature, trainData, setsPerGen, tSetIndex = None, huntWindow = 4):
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
    for cyc in range(len(tSet[1][0])): #for each cycle
        for inp in range(len(tSet[0])): #set the inputs
            creature.input[inp].inbox = [tSet[0][inp][cyc]]
        creature.run(1)
        cycleFitnessList.append( judgeFitnessWithHunt( creature, trainData, cyc, tSetIndex, huntWindow ) ) #judge
    newAvgFit = sum(cycleFitnessList)/float(len(cycleFitnessList)) #then average all together for the creature
    creature.fitness = ( ( setsPerGen - 1 ) * creature.fitness + newAvgFit) / setsPerGen

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
    root = Tk()
    population = Population(CreatureCount=1000, NeuronCount=15, InputCount=1, OutputCount=1)
    trainData = docy()
    #generateSinTracker(self, inputCount, outputCount, cycleCount=360, a=1, b=1, c=0, reps=1)
    trainData.generateSinTracker(len(population.creatureList[0].input), len(population.creatureList[0].output),cycleCount=360)
    print "ins"
    print trainData.data[0][0]
    print "outs"
    print trainData.data[0][1]

    #evolve(population, trainData, generations=3, setsPerGen=1)
    evolve(population, trainData, generations=20, setsPerGen=1)

    bestCreature = findBestCreature(population)

    #docy.data[set][io][put][cycle]
    gui = cg2.CreatureGUI_Beta(root,bestCreature,trainData.data[0][0])
    root.geometry("900x500+300+300")
    root.mainloop()
    print "ins"
    print trainData.data[0][0]
    print "outs"
    print trainData.data[0][1]

if __name__ == "__main__":
    main()
