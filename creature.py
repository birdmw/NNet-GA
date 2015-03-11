from neuron import *
from synapse import *
from trueskill import Rating, quality_1vs1, rate_1vs1
from NNetLibrary import *
import numpy as np

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount,maxCycles=50):
        self.neuronCount, self.inputCount, self.outputCount = neuronCount, inputCount, outputCount
        self.maxCycles = maxCycles
        self.cycles = 0
        self.neuronList, self.input, self.output, self.synapseList  = [], [], [], []
        self.fitness = 0.0
        self.expectedOutputs = []
        for n in range(self.neuronCount):
            self.neuronList.append(Neuron())
        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])
            self.input[-1].isInput = 1
        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])
            self.output[-1].isOutput = 1
        for n1 in self.neuronList:
            for n2 in self.neuronList:
                self.synapseList.append( Synapse(n1, n2, len(self.neuronList) ) )
        self.ELO = Rating()

    def myGauss(self, mu,sig,x):
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

    def run( self ): #no cycles or population, that info is internal to creature now
        runFitness = 0.0
        outputTracker = []
        self.cycles = 0
        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

            self.cycles +=1

            totalCreatureOutputDifference = 0.0
            for Out in range(len(self.output)):
                tOut = self.expectedOutputs[Out]
                cOut = self.output[Out].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)
            runFitness = (runFitness + self.myGauss(0,1,totalCreatureOutputDifference) ) / 2

            if cyc <= (self.neuronCount**2+10):#*(2.0/3.0):
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

        self.fitness = (self.fitness + runFitness ) / 2



def main():
    neuronCount = 2
    inputCount =1
    outputCount = 1
    MaxCycles = 45

    inputSet = [0]
    expOut = [0]

    runs =6

    demoCreature = Creature(neuronCount, inputCount, outputCount,MaxCycles)
    for i in range(len(inputSet)):
        demoCreature.input[i].inbox = [inputSet[i]]

    demoCreature.expectedOutputs = expOut
##    filename = r'C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\bestie4lyfe_2015_2_17_12_17_35'
##    demoCreature = load_creature(filename)


    print 'Creature Description:'
    print '  Number of neurons:',len(demoCreature.neuronList)
    print '  Number of synapses:',len(demoCreature.synapseList)
    print '  Number of inputs:',len(demoCreature.input)
    print '  Number of outputs:',len(demoCreature.output)
    print '  Max cycles to run:',demoCreature.cycles
    print '  Initial ELO: ',demoCreature.ELO
    print '  Inputs: ',inputSet
    print '  Expected Outputs: ',demoCreature.expectedOutputs

    demoCreature.run()

    outputs = []
    for outp in demoCreature.output:
        outputs.append(outp.outbox)
    print '  Outputs: ',outputs
    print '  Cycles ran: ',demoCreature.cycles


    print ''
    print 'Repeatability test:'
    inputSets = [[0],[1]]
    testCreatureRepeatability(demoCreature,inputSets,runs)


    #seeCreature(demoCreature)

if __name__ == '__main__':
    main()
