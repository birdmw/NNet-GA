from random import *
from math import *
from copy import *
from pickle import *

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
        self.updateBest()


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
        for i in range ( self.inputCount ):
            self.input[i].inbox = inputSet[i]
        for r in range( cycles ):
            if r != 0:
                for i in range ( self.inputCount ):
                    self.input[i].inbox += inputSet[i]
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

class Neuron:
    def __init__(self):
        self.threshold = random()*randint(-1,1)
        self.inbox = random()*randint(-1,1)
        self.value = random()*randint(-1,1)
        self.outbox = random()*randint(-1,1)
        self.prevOutbox = random()*randint(-1,1)

    def run(self):
        self.prevOutbox = self.outbox
        #self.outbox = 0.0
        self.value += self.inbox
        if (self.value >= self.threshold):
            self.outbox = self.value
        self.value = 0.0
        self.inbox = 0.0

class Synapse:
    def __init__(self, n1, n2,neuronCount):
        self.a = random()*randint(-1,1)
        self.b = random()*randint(-1,1)*pi
        self.c = random()*randint(-1,1)*pi
        self.d = random()*randint(-1,1)/neuronCount
        self.n1 = n1
        self.n2 = n2

    def run(self):
        try:
            self.n2.inbox += self.a * sin(self.b * self.n1.outbox + self.c) + self.d * (self.n1.prevOutbox - self.n1.outbox)
        except:
            print'!!!!!! WARNING: Synapse broke math!!!!!!!'
            pass


def testCreatureRepeatability(creature,inputSets,runs,cycles):
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
            creat.neuronList[n] = creat.input[n]
        if (creat.neuronCount - n-1) < creat.outputCount:
            creat.neuronList[n] = creat.output[creat.neuronCount -n-1]
    #print 'Loaded Creatures outputs:'

    return creat


def main():
    neuronCount = 6
    inputCount =2
    outputCount = 1

    inputSet = [0,1]
    cycles = 45

    runs =6

    #demoCreature = Creature(neuronCount, inputCount, outputCount)

    filename = r'C:\Users\chris.nelson\Desktop\NNet\ExhaustiveTrainingPerGen\MaNigga2015_2_12_20_0_45'
    demoCreature = load_creature()


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





if __name__ == '__main__':
    main()
