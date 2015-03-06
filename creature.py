from random import *
from math import *
from copy import *
from pickle import *
from creatureGUI import *

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount):
        self.neuronCount = neuronCount
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.neuronList  = []
        self.synapseList = []
        self.fitness = 0.1
        self.input = []
        self.output = []
        self.cycles=0
        self.age = 0

        #Best parameters are used in the lessons to allow improvement only

        self.bestFitness = 0.05
        self.bestNeuronList = []
        self.bestSynapseList = []

        #Property count is used for calculating the percentage of traits to mutate - evaluated at initialization
        #self.propertyCount = 0

        for n in range(self.neuronCount):
            self.neuronList.append(Neuron())
            #self.propertyCount+=1

        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])

        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])

        for n1 in self.neuronList:
            for n2 in self.neuronList:
                self.synapseList.append(Synapse(n1, n2,self.neuronCount))
                #self.propertyCount+=4
        #self.updateBest()


    def evaluateBest(self):
        '''
        Calculates the fitness of this creature, and determines if this creature should update or revert it's neurons, synapses and fitness
        '''
        if self.bestFitness > self.fitness:
            self.revertBest()
        else:
            self.updateBest()


    def revertBest(self):
        '''
        Reverts this creatures fitness, neuronList, and synapseList to their previous best.
        '''
        self.fitness = deepcopy(self.bestFitness)
        self.neuronList = deepcopy(self.bestNeuronList)
        self.synapseList = deepcopy(self.bestSynapseList)

    def updateBest(self):
        '''
        Updates this creatures bestFitness, bestNeuronList and bestSynapselist.
        '''
        self.bestFitness = deepcopy(self.fitness)
        self.bestNeuronList = deepcopy(self.neuronList)
        self.bestSynapseList = deepcopy(self.synapseList)

    def run(self,inputSet,cycles):
        '''
        Let each creature run 'lessons' number of 'cycles' length tests on the current training set
        parameters:
           population: The population this creature lives in. This can be replaced with trainingCreature
           cycles: The number of cycles that this creature is allowed to run before it's outputs are read.
        returns:
            none
        '''
        for r in range( cycles ):
            for i in range ( self.inputCount ):
                self.input[i].inbox.append(inputSet[i])
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

    def run_untilConverged(self,inputSet,maxCycles):
        '''
        Let each creature run 'lessons' number of 'cycles' length tests on the current training set
        parameters:
           population: The population this creature lives in. This can be replaced with trainingCreature
           cycles: The number of cycles that this creature is allowed to run before it's outputs are read.
        returns:
            none
        '''
        self.cycles = 0
        outputTracker = []
        for cycle in range( maxCycles ):
            for i in range ( self.inputCount ):
                self.input[i].inbox.append(inputSet[i])
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()


            self.cycles+=1

            if cycle <= (self.neuronCount**2+10):#*(2.0/3.0):
                outputTracker.append([])
                for o in self.output:
                    outputTracker[-1].append(o.outbox)
            else:
                newVal=[]
                for o in self.output:
                    newVal.append(o.outbox)
                outputTracker = outputTracker[1:]+outputTracker[:1]
                outputTracker[-1] = newVal

                if checkConvergence(outputTracker):
                    break


class Neuron:
    def __init__(self):
        minChoice = -15 #-5
        maxChoice = 15 #5
        self.maxVal = 1000
        self.threshold = random()*choice([minChoice,maxChoice])
        self.inbox = []
        self.value = random()*choice([minChoice,maxChoice])
        self.outbox = random()*choice([minChoice,maxChoice])
        self.prevOutbox = random()*choice([minChoice,maxChoice])

    def run(self):
        self.prevOutbox = self.outbox
        avgInput = sum(self.inbox)/float(len(self.inbox)+1)
        self.value += min(self.maxVal,max(avgInput,-1*self.maxVal))
        if (self.value >= self.threshold):
            self.outbox = self.value
            self.value = 0.0
        self.inbox = []

class Synapse:
    def __init__(self, n1, n2,neuronCount):
        minChoice = -10 #-2
        maxChoice = 10 #2
        self.maxVal = 1000
        self.a = random()*choice([minChoice,maxChoice])
        self.b = random()*choice([minChoice*pi,maxChoice*pi])
        self.c = random()*choice([minChoice*pi,maxChoice*pi])
        self.d = random()*choice([minChoice,maxChoice]) #/neuronCount
        self.n1 = n1
        self.n2 = n2

    def run(self):
        if abs(self.a)+abs(self.d) < 0.001:
            print 'dead synapse: a = ',self.a,' d = ',self.d
            #This synapses has been evolved/initialized to be not useful. Don't bother running.
            return
        try:
            self.n2.inbox.append(min(self.maxVal, max(self.a * sin(self.b * self.n1.outbox + self.c) + self.d * (self.n1.prevOutbox - self.n1.outbox),-1*self.maxVal)))
        except Exception as e:
            print'!!!!!! WARNING: Synapse broke math!!!!!!!'
            print e
            pass


def checkConvergence(outputLists):
    '''
    Takes the given list of output lists, finds the min and max value for each output
    For all output sets:
        If the difference between the min and max value is greater than 0.01 % of the max value, return False

    Return true if all outputs pass

    outputLists = [[cycleA_output0,cycleA_output1,...],[cycleB_output0,cycleB_output1,...],...]
    '''

    percentageDifferenceToBeConverged = 0.01 #0.0001
    for outInd in range(len(outputLists[0])):
        reorderedOutputs = []
        for pt in outputLists:
            reorderedOutputs.append(pt[outInd])

        minOut = min(reorderedOutputs)
        maxOut = max(reorderedOutputs)
        diff = abs(maxOut)-abs(minOut)
        if diff > abs(maxOut *percentageDifferenceToBeConverged):
            return False

    return True



def testCreatureRepeatability(creature,inputSets,runs,cycles):
    print 'Creature fitness = ',creature.fitness
    for inputSet in inputSets:
        print 'Inputs: ',inputSet
        for r in range(runs):
            creature.run(inputSet,cycles)
            outputs = []
            for outp in creature.output:
                outputs.append(outp.outbox)

            print '  Run',r,' Outputs: ',outputs

def save_creature(creature,fileName):
    fCreature = open(fileName,'wb')
    dump(creature,fCreature)
    fCreature.close()
    return fileName

def load_creature(fileName):
    fCreature = open(fileName,'r')
    creat = load(fCreature)
    fCreature.close()
    for n in range ( creat.neuronCount ):
        if n < creat.inputCount:
            creat.input[n] = creat.neuronList[n]
        if (creat.neuronCount - n-1) < creat.outputCount:
            creat.output[creat.neuronCount -n-1] = creat.neuronList[n]

    #print 'Loaded Creatures outputs:'

    return creat


def main():
    neuronCount = 21
    inputCount =2
    outputCount = 1

    inputSet = [0,1]
    cycles = 45

    runs =6

    demoCreature = Creature(neuronCount, inputCount, outputCount)

    filename = r'C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\bestie4lyfe_2015_2_17_12_17_35'
    demoCreature = load_creature(filename)


    print 'Creature Description:'
    print '  Number of neurons:',len(demoCreature.neuronList)
    print '  Number of synapses:',len(demoCreature.synapseList)
    print '  Number of inputs:',len(demoCreature.input)
    print '  Number of outputs:',len(demoCreature.output)
    print '  Cycles to run:',cycles
    print '  Inputs: ',inputSet

    demoCreature.run(inputSet,cycles)
    outputs = []
    for outp in demoCreature.output:
        outputs.append(outp.outbox)
    print '  Outputs: ',outputs

    print ''
    print 'Repeatability test:'
    inputSets = [[0,0],[0,1],[1,0],[1,1]]
    testCreatureRepeatability(demoCreature,inputSets,runs,cycles)


    seeCreature(demoCreature)

if __name__ == '__main__':
    main()
