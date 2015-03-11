from neuron import *
from synapse import *
from trueskill import Rating, quality_1vs1, rate_1vs1
import numpy as np
from random import *

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount):
        self.neuronCount, self.inputCount, self.outputCount = neuronCount, inputCount, outputCount
        self.neuronList, self.input, self.output, self.synapseList  = [], [], [], []
        self.fitness = 0.0
        self.rank = random()
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
                if not n2 in self.output and not n1 in self.input and not n1==n2:
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

    def run( self ):
        runFitness = 0.0
        for r in range( self.cycles ):

            '''
            for i in range ( self.inputCount ):
                self.input[i].inbox = population.trainingCreature.input[i].inbox
            '''
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()
            totalCreatureOutputDifference = 0.0
            for Out in range(len(self.output)):
                tOut = self.expectedOutputs[Out]
                cOut = self.output[Out].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)
            runFitness = (runFitness + self.myGauss(0,10,totalCreatureOutputDifference) ) / 2
        self.fitness = (self.fitness + runFitness ) / 2

