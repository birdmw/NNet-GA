from random import *
from math import *

#note: hunting is where each value, at each cycle, each neuron, is allowed to seach through the local expected outputs for a best value
#this is different from phase searching where we wait till the end and "shift" our data looking for a best result. Personally I like the
#idea of hunting a little more. It is more flexible and it solves the "punished for overflow" issue. Only hunting works right now.


class docy:
    def __init__(self, data=[], huntWindow = 4): #for now it will only work with a hunt window, working on phase shift
        #docy.data[set][io][put][cycle]
        self.data = data
        #cycleWindow is the window in which outputs are allowed to hunt for a best result
        self.huntWindow = huntWindow
        self.maxCycleShift = maxCycleShift
        
    def addSet(tset):
        #[io][put][cycle]
        data.append(tset)
        
    def removeSet( index = -1 ):
        #index
        if (abs(index) > len(self.data)):
            print "set does not exist, cannot remove"
        else:
            data.pop(index)
    def randomSet():
        return choice(self.data)
    
    def generateSin(inputCount, outputCount, cycleCount=360, a=1, b=1, c=1, reps=1):
        inputList, outputList = [] , []
        inputs , outputs = [] , []
        for x in range(int(cycleCount * reps)):
                inputs.append(x*pi/180)
                outputs.append(a*sin(b*x+c))
        for y in range(inputCount):
                inputList.append(inputs)
                outputList.append(outputs)
        self.data = [[inputList, outputList]]

def trainCreature(creature, docy, tSetIndex = None):
    #accepts a creature and a training set
    #runs the creature for the length of the dataset
    #sets the creatures fitness using "forgiveness" tricks
    if tSetIndex == None:
        tSet = docy.data.randomSet()
        tSetIndex = docy.data.index(tset)
    else:
        tSet = docy.data[tSetIndex]
    cycleFitnessList = []
    creatureOutputArray = []
    cycles = len(docy.data[0][1][0])
    for cyc in range(cycles):
        for inp in range(len(creature.input)):
            creature.input[inp].inbox = tSet[0][inp][cyc]
        if docy.huntWindow != 0: #hunting enabled, no phase shifting
            cycleFitnessList.append( judgeFitnessWithHunt( creature, docy, cyc, tSetIndex ) )
        else: #phase shift mode
            outputArray = []
            for oup in range(len(creature.oup)):
                outputArray.append( creature.outbox[oup].outbox )
            creatureOutputArray.append(outputArray)
        creature.run(1)
    if docy.huntWindow != 0:
        creature.fitness = sum(cycleFitnessList)/float(len(cycleFitnessList))
    else:
        creature.fitness = judgeFitnessWithCycleShift(zip(creatureOutputArray), docy, tSetIndex)

def judgeFitnessWithHunt(creature, docy, cyc, tSetIndex):
    neuronDiffList = []
    for neuronIndex in range(len(creature.output)):
        minDiff = abs(creature.output[neuronIndex].outbox - docy.data[tSetIndex][1][neuronIndex][cyc])
        windowIndex = 1
        while windowIndex <= docy.huntWindow:
            if cyc+windowIndex > 0 and cyc+windowIndex < len(docy.data[tSetIndex][1][neuronIndex]):
                minDiff = min( minDiff, abs(creature.output[neuronIndex].outbox - docy.data[tSetIndex][1][neuronIndex][cyc+windowIndex]))
            if cyc-windowIndex > 0 and cyc-windowIndex < len(docy.data[tSetIndex][1][neuronIndex]):
                minDiff = min( minDiff, abs(creature.output[neuronIndex].outbox - docy.data[tSetIndex][1][neuronIndex][cyc-windowIndex]))
            windowIndex+=1
        neuronDiffList.append(minDiff)
    avgDiff = sum(neuronDiffList)/float(len(neuronDiffList))
    return myGauss(avgDiff)

'''
def judgeFitnessWithCycleShift(outputArray, docy, tSetIndex):
    for neuronIndex in range(len(outputArray)):
        cycleShift = -len(outputArray[neuronIndex])
        while cycleShift<len(outputArray[neuronIndex]):
            if cycleShift<0:
                Overflow = outputArray[neuronIndex][len(outputArray)-abs(
            else:
            
            cycleShift +=1
'''
            
            
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
    
