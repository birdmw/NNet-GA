from random import *
from math import *
import numpy as np

#note: hunting is where each value, at each cycle, each neuron, is allowed to seach through the local expected outputs for a best value
#this is different from phase searching where we wait till the end and "shift" our data looking for a best result. Personally I like the
#idea of hunting a little more. It is more flexible and it solves the "punished for overflow" issue. Only hunting works right now.


class docy:
    def __init__(self, data=[], huntWindow = 1): #for now it will only work with a hunt window, working on phase shift
        #docy.data[set][io][put][cycle]
        self.data = data

    def addSet(self, tset):
        #[io][put][cycle]
        data.append(tset)
        
    def removeSet(self, index = -1 ):
        #index
        if (abs(index) > len(self.data)):
            print "set does not exist, cannot remove"
        else:
            data.pop(index)
            
    def randomSet(self):
        return choice(self.data)
    
    def generateSin(self, inputCount, outputCount, cycleCount=2, a=1, b=1, c=1, reps=1):
        inputList, outputList = [] , []
        inputs , outputs = [] , []
        for x in range(int(cycleCount * reps)):
                inputs.append(x*pi/180)
                outputs.append(a*sin(b*inputs[-1]+c))
        for y in range(inputCount):
                inputList.append(inputs)
                outputList.append(outputs)
        self.data = [[inputList, outputList]]

def trainPopulation(pop, trainData, tSetIndex = None):
    for c in pop.creatureList:
        trainCreature(c, trainData)

def trainCreature(creature, trainData, tSetIndex = None, huntWindow = 2):
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
            cycleFitnessList.append( judgeFitnessWithHunt( creature, trainData, cyc, tSetIndex, huntWindow ) ) #judge
        creature.run(1)
    creature.fitness = sum(cycleFitnessList)/float(len(cycleFitnessList)) #then average all together for the creature

def judgeFitnessWithHunt(creature, trainData, cyc, tSetIndex, huntWindow):
    neuronDiffList = []
    for neuronIndex in range(len(creature.output)): #for each output
        windowIndex = 0
        minDiff = abs(creature.output[neuronIndex].outbox - trainData.data[tSetIndex][1][neuronIndex][cyc])
        while windowIndex <= abs(huntWindow): #for each window
            if cyc+windowIndex < len(trainData.data[tSetIndex][1][neuronIndex]):# if it didnt roll off either end
                minDiff = min( minDiff, abs(creature.output[neuronIndex].outbox - trainData.data[tSetIndex][1][neuronIndex][cyc+windowIndex])) #find the minimum
            if cyc-windowIndex >= 0:
                minDiff = min( minDiff, abs(creature.output[neuronIndex].outbox - trainData.data[tSetIndex][1][neuronIndex][cyc-windowIndex]))
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

def myGauss(x,mu=1.0,sig=1.0):
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
    
