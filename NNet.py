from math import *
from random import *
from copy import *
import numpy as np
import matplotlib.pyplot as plt
from trueskill import Rating, quality_1vs1, rate_1vs1
from multiprocessing import Pool, cpu_count, Process

class Population:
    def __init__(self, CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN ):
        self.cycles = CYCLES_PER_RUN
        self.creatureList = []
        self.creatureCount = CREATURE_COUNT
        self.neuronCount = NEURON_COUNT
        self.inputCount = INPUT_COUNT
        self.outputCount = OUTPUT_COUNT
        self.trainingCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )
        self.trainingCreature.cycles = CYCLES_PER_RUN
        self.rollingMaxOutput = 0.0
        for out in self.trainingCreature.output:
            out.outbox = gauss(0,1)
        self.synapseCount = len ( self.trainingCreature.synapseList )
        self.populate()

    def pruneByELO ( self ):
        half = len(self.creatureList)/2
        self.sortBySigma()
        highSigmaCreatureList = self.creatureList[half:]
        self.sortByMu()
        index = 0
        for k in range(half):
            if self.creatureList[-1-index] in highSigmaCreatureList:
                self.creatureList.pop(-1-index)
            else:
                index += 1

    def populate( self ):
         while (len(self.creatureList) < self.creatureCount):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))

    def repopulate( self ):
         while (len(self.creatureList) < 2):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount):
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              if not(mother == father):
                child = mate( mother , father )
                self.creatureList.append( child )

    def mutateBySigma( self ):
        maxOut = 0
        for i in range( len(self.trainingCreature.output)):
            maxOut = max(maxOut,self.trainingCreature.output[i].outbox, abs(self.trainingCreature.output[i].outbox))

        self.rollingMaxOutput = ( 9 * self.rollingMaxOutput +  maxOut ) / 10
        for creature in self.creatureList:
            for n in creature.neuronList:
                n.threshold = max(min(gauss( n.threshold , creature.ELO.sigma*.12*self.rollingMaxOutput),1000000),-1000000)
            for s in creature.synapseList:
                s.a = max(min(gauss( s.a , creature.ELO.sigma*.12*self.rollingMaxOutput ),1000000),-1000000)
                s.b = max(min(gauss( s.b , creature.ELO.sigma*.12*self.rollingMaxOutput ),1000000),-1000000)
                s.c = max(min(gauss( s.c , creature.ELO.sigma*.12*self.rollingMaxOutput ),1000000),-1000000)
                s.d = max(min(gauss( s.d , creature.ELO.sigma*.12*self.rollingMaxOutput ),1000000),-1000000)

    def setTrainingConstant( self, const=1 ):
        for i in self.trainingCreature.input:
            i.inbox = [const]
        for o in self.trainingCreature.output:
            o.outbox = const

    def setTrainingBools ( self ):
        for i in self.trainingCreature.input:
            i.inbox = [float(bool(getrandbits(1)))]
        for o in self.trainingCreature.output:
            if self.trainingCreature.output.index(o)%4==0:
                self.trainingCreature.output[0].outbox = float(  bool(self.trainingCreature.input[0].inbox[0]) ^ bool(self.trainingCreature.input[1].inbox[0]))##<---xor for inputs 0 and 1
            elif self.trainingCreature.output.index(o)%4==1:
                self.trainingCreature.output[1].outbox = float(  bool(self.trainingCreature.input[0].inbox[0]) & bool(self.trainingCreature.input[1].inbox[0]))##<---and for inputs 0 and 1
            elif self.trainingCreature.output.index(o)%4==2:
                self.trainingCreature.output[2].outbox = float(  bool(self.trainingCreature.input[0].inbox[0]) or bool(self.trainingCreature.input[1].inbox[0]))##<---or for inputs 0 and 1
            elif self.trainingCreature.output.index(o)%4==3:
                self.trainingCreature.output[3].outbox = float(~(bool(self.trainingCreature.input[0].inbox[0]) & bool(self.trainingCreature.input[1].inbox[0])))##<---nand for inputs 0 and 1

    def setPuts ( self ):
        self.expectedOutputs = []
        for c in self.creatureList:
            for i in range(len(c.input)):
                c.input[i].inbox=self.trainingCreature.input[i].inbox
            for j in range(len(c.output)):
                c.expectedOutputs.append(self.trainingCreature.output[j].outbox)
            c.cycles = self.cycles

    def runPopulation( self, CYCLES_PER_RUN ):
        for creature in self.creatureList:
            creature.run(self, CYCLES_PER_RUN)

    def sortByMu( self ):
        self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
        return self

    def sortBySigma( self ):
        self.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
        return self

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount):
        self.neuronCount, self.inputCount, self.outputCount = neuronCount, inputCount, outputCount
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

    def setFitness( self ):
        totalCreatureOutputDifference = 0.0
        for Out in range(len(self.output)):
            tOut = self.expectedOutputs[Out]
            cOut = self.output[Out].outbox
            totalCreatureOutputDifference += abs(tOut-cOut)
        self.fitness = myGauss(0,1,totalCreatureOutputDifference)

    def run( self ): #no cycles or population, that info is internal to creature now
        for r in range( self.cycles ):
            '''
            for i in range ( self.inputCount ):
                self.input[i].inbox = population.trainingCreature.input[i].inbox
            '''
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

class Neuron:
    def __init__(self):
        self.inbox = [gauss(0,1)]
        self.value, self.threshold, self.outbox, self.prevOutbox = gauss(0,1), gauss(0,1), gauss(0,1), gauss(0,1)
        self.isInput, self.isOutput = 0, 0

    def run(self):
        self.prevOutbox = self.outbox
        if self.isOutput == 0:#If not an output neuron
            self.outbox = 0.0
        self.value += sum(self.inbox) / (float(len(self.inbox)))
        if (self.value >= self.threshold):
            self.outbox = max(min(self.value,1000000),-1000000)
            self.value = 0.0
        if self.isInput == 0:#If not an input neuron
            self.inbox = [0.0]

class Synapse:
    def __init__(self, n1, n2, neuronCount):
        self.a = gauss(0,1) / neuronCount
        self.b = gauss(0,1) / neuronCount
        self.c = gauss(0,1) / neuronCount
        self.d = gauss(0,1) / neuronCount
        self.n1, self.n2 = n1, n2

    def run(self):
        if self.n1.isOutput == 0 and self.n2.isInput == 0: #If not an input and not an output
            sinFxn = max(min(self.a * sin(self.b * self.n1.outbox + self.c),1000000),-1000000)
            diffFxn = max(min(self.d * (self.n1.prevOutbox - self.n1.outbox),1000000),-1000000)
            self.n2.inbox.append( sinFxn + diffFxn )

def mate (mother, father):
     child = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
     for i in range(len(child.neuronList)):
          if getrandbits(1):
              child.neuronList[i].threshold = father.neuronList[i].threshold
          else:
              child.neuronList[i].threshold = mother.neuronList[i].threshold
     for i in range(len(child.synapseList)):
          if getrandbits(1):
              child.synapseList[i].a = father.synapseList[i].a
          else:
              child.synapseList[i].a = mother.synapseList[i].a
          if getrandbits(1):
              child.synapseList[i].b = father.synapseList[i].b
          else:
              child.synapseList[i].b = mother.synapseList[i].b
          if getrandbits(1):
              child.synapseList[i].c = father.synapseList[i].c
          else:
              child.synapseList[i].c = mother.synapseList[i].c
          if getrandbits(1):
              child.synapseList[i].d = father.synapseList[i].d
          else:
              child.synapseList[i].d = mother.synapseList[i].d
     return child

def updateELO( creature1, creature2 ):
      if creature1.fitness > creature2.fitness:
        creature1.ELO,creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO)
      elif creature2.fitness > creature1.fitness:
        creature2.ELO,creature1.ELO = rate_1vs1(creature2.ELO,creature1.ELO)
      else:
        creature1.ELO, creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO, drawn=True)

def myGauss(mu,sig,x):
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

def printCreature ( creature ):
    print "  -Creature", creature
    print "  --",creature.neuronCount," neurons, ",creature.inputCount," inputs, ",creature.outputCount," outputs, ", len(creature.synapseList)," synapses."
    print "  --",creature.ELO.mu," mu, ",creature.ELO.sigma," sigma "

def randomTrials( population, TRIALS ):
    creatureList = population.creatureList
    p=Pool()
    creatureList = p.map(parallelCreatureRun,creatureList)
    for T in range(TRIALS):
        creature1 = choice( creatureList )
        creature2 = choice( creatureList )
        while creature1 == creature2:
            creature2 = choice( creatureList )
        creature1.setFitness()
        creature2.setFitness()
        updateELO(creature1, creature2)

def parallelCreatureRun( creature ):
    creature.run()
    return creature

def printFinalOuts( population ):
    print "training outs:"
    for o in range (len(population.trainingCreature.output)):
        print population.trainingCreature.output[o].outbox
    print "best creature outs:"
    for c in range (len(population.creatureList[0].output)):
        print population.creatureList[0].output[c].outbox

def f(x):
    return x*x

if __name__ == "__main__":
    CREATURE_COUNT = 100
    NEURON_COUNT= 15
    INPUT_COUNT = 2
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = NEURON_COUNT + 1
    GENERATIONS = 6
    TRAINING_SETS = 4
    TRIALS = 100
    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT, CYCLES_PER_RUN )
    for G in range (GENERATIONS):
        print "GENERATION: ",G
        population.mutateBySigma()
        for s in range(TRAINING_SETS):
            population.setTrainingBools()
            population.setPuts()
            randomTrials( population, TRIALS )

        population.pruneByELO()
        printCreature(population.creatureList[0])
        population.repopulate()
    bestCreature = population.creatureList[0]
    population.setTrainingBools()
    bestCreature.run(  )
    printFinalOuts(population)


