from math import *
from random import *
from copy import *
import numpy as np
import matplotlib.pyplot as plt

class Population:
    def __init__(self, CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT ):
        self.creatureList = []
        self.creatureCount = CREATURE_COUNT
        self.neuronCount = NEURON_COUNT
        self.inputCount = INPUT_COUNT
        self.outputCount = OUTPUT_COUNT
        self.trainingCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        for out in self.trainingCreature.output:
            out.outbox = gauss(0,1)
        self.synapseCount = len ( self.trainingCreature.synapseList )
        self.populate()


    def populate( self ):
         while (len(self.creatureList) < 2):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount):
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              if not (mother == father):
                child = mate( mother , father )
                self.creatureList.append( child )

    def mutateByFitness (self, multiplier):
        for creature in self.creatureList:
            error = multiplier/creature.fitness#-multiplier
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold ,error )
            for s in range ( len ( creature.synapseList ) ):

                creature.synapseList[s].a = gauss( creature.synapseList[s].a , error )
                creature.synapseList[s].b = gauss( creature.synapseList[s].b , error )
                creature.synapseList[s].c = gauss( creature.synapseList[s].c , error )
                creature.synapseList[s].d = gauss( creature.synapseList[s].d , error )

    def mutateAbs( self, absVal ):
        half = len(self.creatureList)/2
        for creature in self.creatureList[:half]:
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold , absVal )
            for s in range ( len ( creature.synapseList ) ):

                creature.synapseList[s].a = gauss( creature.synapseList[s].a , absVal )
                creature.synapseList[s].b = gauss( creature.synapseList[s].b , absVal )
                creature.synapseList[s].c = gauss( creature.synapseList[s].c , absVal )
                creature.synapseList[s].d = gauss( creature.synapseList[s].d , absVal )

    def setTraining1Constant( self, constVal ):
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))
        self.trainingCreature.output[0].outbox = constVal

    def setTrainingMultiplyInputs ( self, multiplier ):
        # Random inputs
        for i in self.trainingCreature.input:
            i.inbox = random()*multiplier

        self.trainingCreature.output[0].outbox = float( self.trainingCreature.input[0].inbox * self.trainingCreature.input[1].inbox)


    def setTraining1Bool ( self ):
        # Random inputs
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))

        self.trainingCreature.output[0].outbox = float(  bool(self.trainingCreature.input[0].inbox) ^ bool(self.trainingCreature.input[1].inbox))

    def setTraining2Bools( self ):

		# Random inputs
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))

		# 4 output: xor, and, or, nand of 2 input
        self.trainingCreature.output[0].outbox = float(  bool(self.trainingCreature.input[0].inbox) ^ bool(self.trainingCreature.input[1].inbox))##<---xor for inputs 0 and 1
        self.trainingCreature.output[1].outbox = float(  bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox))##<---and for inputs 0 and 1

    def setTraining4Bools( self ):

		# Random inputs
        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))

		# 4 output: xor, and, or, nand of 2 input
        self.trainingCreature.output[0].outbox = float(  bool(self.trainingCreature.input[0].inbox) ^ bool(self.trainingCreature.input[1].inbox))##<---xor for inputs 0 and 1
        self.trainingCreature.output[1].outbox = float(  bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox))##<---and for inputs 0 and 1
        self.trainingCreature.output[2].outbox = float(  bool(self.trainingCreature.input[0].inbox) or bool(self.trainingCreature.input[1].inbox))##<---or for inputs 0 and 1
        self.trainingCreature.output[3].outbox = float(~(bool(self.trainingCreature.input[0].inbox) & bool(self.trainingCreature.input[1].inbox)))##<---nand for inputs 0 and 1

    def train( self, CYCLES_PER_RUN ):
        for creature in self.creatureList:
            creature.run(self, CYCLES_PER_RUN)

    def judge( self ):
        for creature in self.creatureList:

            totalCreatureOutputDifference = 0.0
            for Out in range(len(creature.output)):
                tOut = self.trainingCreature.output[Out].outbox
                cOut = creature.output[Out].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)
            #print "totalCreatureOutputDifference", totalCreatureOutputDifference
            creature.fitness = 1 / log( totalCreatureOutputDifference + exp(1) , exp(1) )

    def sort( self ):
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

    def prune ( self ):
        creatureCount = len(self.creatureList)
        for i in range (creatureCount/2):
            self.creatureList.pop()

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount):
        self.neuronCount = neuronCount
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.neuronList  = []
        self.synapseList = []
        self.fitness = random()
        self.input = []
        self.output = []

        for n in range(self.neuronCount):
            self.neuronList.append(Neuron())

        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])

        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])
        for n1 in self.neuronList:
            for n2 in self.neuronList:

                self.synapseList.append( Synapse(n1, n2, len(self.neuronList) ) )

    def run( self, population, cycles ):
        for r in range( cycles ):

            for i in range ( self.inputCount ):
                self.input[i].inbox += population.trainingCreature.input[i].inbox

            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

class Neuron:
    def __init__(self):
        self.threshold = gauss(0,1)
        self.inbox = gauss(0,1)
        self.value = gauss(0,1)
        self.outbox = gauss(0,1)
        self.prevOutbox = gauss(0,1)

    def run(self):
        self.prevOutbox = self.outbox
        #self.outbox = 0.0
        self.value += self.inbox
        if (self.value >= self.threshold):
            self.outbox = min(self.value,1000000)
            self.value = 0.0
        self.inbox = 0.0

class Synapse:
    def __init__(self, n1, n2, neuronCount):
        self.a = gauss(0,1) / neuronCount
        self.b = gauss(0,1) / (2*pi) / neuronCount
        self.c = gauss(0,1) * pi / neuronCount
        self.d = gauss(0,1) / neuronCount
        self.n1 = n1
        self.n2 = n2

    def run(self):

        sinFxn = min(self.a * sin(self.b * self.n1.outbox + self.c),1000000)

        diffFxn = min(self.d * (self.n1.prevOutbox - self.n1.outbox),1000000)

        self.n2.inbox += sinFxn + diffFxn

def mate (mother, father):
     child = deepcopy( mother )
     for i in range(len(child.neuronList)):
          if getrandbits(1):
              child.neuronList[i].threshold = father.neuronList[i].threshold
     for i in range(len(child.synapseList)):
          if getrandbits(1):
              child.synapseList[i].a = father.synapseList[i].a
          if getrandbits(1):
              child.synapseList[i].b = father.synapseList[i].b
          if getrandbits(1):
              child.synapseList[i].c = father.synapseList[i].c
          if getrandbits(1):
              child.synapseList[i].d = father.synapseList[i].d
     return child

def printPopulation ( population ):
    print "==SIGMA CREATURE:"
    printCreature ( population.sigmaCreature )
    print "==AVERAGE WINNING CREATURE:"
    printCreature ( population.avgWinningCreature )
    print "==AVERAGE LOSING CREATURE:"
    printCreature ( population.avgLosingCreature )
    print "==TRAINING CREATURE:"
    printCreature ( population.trainingCreature )
    for creature in population.creatureList:
        print "=Population ", len ( population.creatureList )
        print "  ",population.creatureCount," creatureCount, ", population.neuronCount, " neuronCount, ",population.inputCount," inputCount, ", population.outputCount, " outputCount, ",population.synapseCount," synapseCount"

def printCreature ( creature ):
        #print "  -Creature"
        #print "  --",creature.neuronCount," neurons, ",creature.inputCount," inputs, ",creature.outputCount," outputs, ", len(creature.synapseList)," synapses."
        print "  --",creature.fitness," fitness "

def printSynapse ( synapse ):
        print "    ~Synapse"
        print "    ~~ a = ",synapse.a,", b = ",synapse.b,", c = ",synapse.c,", d = ",synapse.d

def printNeuron ( neuron ):
        print "    *Neuron"
        print "    ** inbox = ",neuron.inbox,", value = ", neuron.value, ", outbox = ", neuron.outbox, ", threshold = ",neuron.threshold,", prevOutbox = ", neuron.prevOutbox

def printPopOuts ( population ):
        print "::::::Training Outputs::::::::::::::"
        c = population.trainingCreature
        for o in c.output:
            printNeuron ( o )
        print "::::::Population Outputs::::::::::::"
        for c in population.creatureList:
            for o in c.output:
                printNeuron ( o )

if __name__ == "__main__":
    CREATURE_COUNT = 100
    NEURON_COUNT= 12
    INPUT_COUNT = 2
    OUTPUT_COUNT = 1
    CYCLES_PER_RUN = NEURON_COUNT*2
    GENERATIONS = 10

    for i in range(1):

        population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )

        for G in range (GENERATIONS):
            print "GENERATION: ",G
            population.populate()
            population.mutateByFitness( .1 )
            population.setTrainingMultiplyInputs(  1 )
            population.train( CYCLES_PER_RUN )
            population.judge()
            population.sort()
            #population.prune()
            printCreature( population.creatureList[0] )

        for o in range(OUTPUT_COUNT):
            name = 'Output '+str(o)

    print "training outs:"
    for o in range (len(population.trainingCreature.output)):
        print population.trainingCreature.output[o].outbox
    print "best creature outs:"
    for c in range (len(population.creatureList[0].output)):
        print population.creatureList[0].output[c].outbox
    print

