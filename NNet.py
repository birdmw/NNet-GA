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
        self.sigmaCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )
        self.avgWinningCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        self.avgLosingCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        self.trainingCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )
        self.deltaCreature = Creature( NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT  )

        for out in self.deltaCreature.output:
            out = 1.0
        for out in self.trainingCreature.output:
            out = 0.5

        self.synapseCount = len ( self.sigmaCreature.synapseList )
        self.populate()


    def populate( self ):
         while (len(self.creatureList) < 2): #If there is only one left, create a random new creature.
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount): #Breed until full population
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              if not (mother == father):
                child = mate( mother , father )
                self.creatureList.append( child )
         self.mutate()

    def mutate( self ):
        for creature in self.creatureList:
            for n in range( len ( creature.neuronList ) ):
                creature.neuronList[n].threshold = gauss( creature.neuronList[n].threshold , self.sigmaCreature.neuronList[n].threshold)
            for s in range ( len ( creature.synapseList ) ):
                creature.synapseList[s].a = gauss( creature.synapseList[s].a , self.sigmaCreature.synapseList[s].a )
                creature.synapseList[s].b = gauss( creature.synapseList[s].b , self.sigmaCreature.synapseList[s].b )
                creature.synapseList[s].c = gauss( creature.synapseList[s].c , self.sigmaCreature.synapseList[s].c )
                creature.synapseList[s].d = gauss( creature.synapseList[s].d , self.sigmaCreature.synapseList[s].d )

    def setTrainingCreature( self ):

        for i in self.trainingCreature.input:
            i.inbox = float(bool(getrandbits(1)))##random bool

        self.trainingCreature.output[0].outbox = float(bool(self.trainingCreature.input[0].inbox)^bool(self.trainingCreature.input[1].inbox))##<---xor for inputs 0 and 1
        self.trainingCreature.output[1].outbox = not(float(bool(self.trainingCreature.input[0].inbox) and  bool(self.trainingCreature.input[1].inbox)))##<---xor for inputs 0 and 1

        '''
        for i in self.trainingCreature.input:
            i.inbox = 0.5
        for o in self.trainingCreature.output:
            o.inbox = 0.5

        '''
    def compete( self, CYCLES_PER_RUN ):
        for creature in self.creatureList:
            #Create new thread HERE. Please god...

            creature.run(self, CYCLES_PER_RUN)

    def resolve( self ):

        #Calculate fitness

        sumFitness=0
        for creature in self.creatureList:
            fitSum = 0
            for out in range(self.outputCount):
                mu = self.trainingCreature.output[out].outbox
                sigma = self.deltaCreature.output[out].outbox
                fitSum += myGauss(mu,sigma,creature.output[out].outbox)
            creature.fitness = fitSum/self.outputCount
            sumFitness += creature.fitness
        averageFitness = sumFitness / self.creatureCount

        sumFitness = 0.0
        for creature in self.creatureList:
            sumFitness += creature.fitness
        averageFitness = sumFitness / self.creatureCount
        winningCreatures = []
        losingCreatures = []
        for creature in self.creatureList:
            #printCreature ( creature )
            if (creature.fitness >= averageFitness):
                winningCreatures.append(creature)
            else:
                losingCreatures.append(creature)
        if (len ( losingCreatures) != 0 and len(winningCreatures) !=0):

            print len(losingCreatures)," losers, ",len(winningCreatures)," winners, ","average fit = ",averageFitness

            self.avgWinningCreature.fitness = sum( w.fitness for w in winningCreatures) / len(winningCreatures)
            for i in range ( self.neuronCount ):
                self.avgWinningCreature.neuronList[i].threshold = sum( w.neuronList[i].threshold for w in winningCreatures) / len(winningCreatures)
                self.avgWinningCreature.neuronList[i].outbox = sum( w.neuronList[i].outbox for w in winningCreatures) / len(winningCreatures)

                self.avgLosingCreature.neuronList[i].threshold = sum( l.neuronList[i].threshold for l in losingCreatures ) / len(losingCreatures)

            for i in range ( self.synapseCount ):
                self.avgWinningCreature.synapseList[i].a = sum( w.synapseList[i].a for w in winningCreatures ) / len(winningCreatures)
                self.avgWinningCreature.synapseList[i].b = sum( w.synapseList[i].b for w in winningCreatures ) / len(winningCreatures)
                self.avgWinningCreature.synapseList[i].c = sum( w.synapseList[i].c for w in winningCreatures ) / len(winningCreatures)
                self.avgWinningCreature.synapseList[i].d = sum( w.synapseList[i].d for w in winningCreatures ) / len(winningCreatures)

                self.avgLosingCreature.synapseList[i].a = sum( l.synapseList[i].a for l in losingCreatures ) / len(losingCreatures)
                self.avgLosingCreature.synapseList[i].b = sum( l.synapseList[i].b for l in losingCreatures ) / len(losingCreatures)
                self.avgLosingCreature.synapseList[i].c = sum( l.synapseList[i].c for l in losingCreatures ) / len(losingCreatures)
                self.avgLosingCreature.synapseList[i].d = sum( l.synapseList[i].d for l in losingCreatures ) / len(losingCreatures)

            for i in range ( self.neuronCount ):
                self.sigmaCreature.neuronList[i].threshold = self.avgWinningCreature.neuronList[i].threshold - self.avgLosingCreature.neuronList[i].threshold
                self.deltaCreature.neuronList[i].outbox = self.trainingCreature.neuronList[i].outbox - self.avgWinningCreature.neuronList[i].outbox

            for i in range ( self.synapseCount ):
                self.sigmaCreature.synapseList[i].a = self.avgWinningCreature.synapseList[i].a - self.avgLosingCreature.synapseList[i].a
                self.sigmaCreature.synapseList[i].b = self.avgWinningCreature.synapseList[i].b - self.avgLosingCreature.synapseList[i].b
                self.sigmaCreature.synapseList[i].c = self.avgWinningCreature.synapseList[i].c - self.avgLosingCreature.synapseList[i].c
                self.sigmaCreature.synapseList[i].d = self.avgWinningCreature.synapseList[i].d - self.avgLosingCreature.synapseList[i].d

        for creature in self.creatureList:
            if (creature.fitness < averageFitness):
                self.creatureList.remove(creature)

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
            self.neuronList.append(Neuron(random()))

        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])

        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])
        for n1 in self.neuronList:
            for n2 in self.neuronList:

                self.synapseList.append(Synapse(n1, n2, random() / self.neuronCount, random() / (2*pi) / self.neuronCount, random() * pi / self.neuronCount, random() / self.neuronCount))

    def run( self, population, cycles ): #GOOD TIME TO APPLY PARALLEL PROCESSING
        for r in range( cycles ):
            for i in range ( len ( self.input ) ):
                self.input[i].inbox = population.trainingCreature.input[i].inbox
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()



class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.inbox = random()
        self.value = random()
        self.outbox = random()
        self.prevOutbox = random()

    def run(self):
        self.prevOutbox = self.outbox
        self.outbox = 0.0
        self.value += self.inbox
        if (self.value >= self.threshold):
            self.outbox += self.value
            self.value = 0.0
        self.inbox = 0.0

class Synapse:
    def __init__(self, n1, n2, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n1 = n1
        self.n2 = n2

    def run(self):
        self.n2.inbox += self.a * sin(self.b * self.n1.outbox + self.c) + self.d * (self.n1.prevOutbox - self.n1.outbox)

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
    for creature in population.creatureList:
        print "=Population ", len ( population.creatureList )
        print "  ",population.creatureCount," creatureCount, ", population.neuronCount, " neuronCount, ",population.inputCount," inputCount, ", population.outputCount, " outputCount, ",population.synapseCount," synapseCount"
        print "==SIGMA CREATURE:"
        printCreature ( population.sigmaCreature )
        print "==AVERAGE WINNING CREATURE:"
        printCreature ( population.avgWinningCreature )
        print "==AVERAGE LOSING CREATURE:"
        printCreature ( population.avgLosingCreature )
        print "==TRAINING CREATURE:"
        printCreature ( population.trainingCreature )

def printCreature ( creature ):
        print "  -Creature"
        print "  --",creature.neuronCount," neurons, ",creature.inputCount," inputs, ",creature.outputCount," outputs, ", len(creature.synapseList)," synapses."
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

def myGauss(mu,sig,x):
        return np.exp(-np.power(x - mu, 2.) / 2 * np.power(sig, 2.))

if __name__ == "__main__":
    CREATURE_COUNT = 10
    NEURON_COUNT= 20
    INPUT_COUNT = 2
    OUTPUT_COUNT = 2
    CYCLES_PER_RUN = 5
    GENERATIONS = 200
    naughtyWinnersFits=[]
    for i in range(1):
        naughtyWinnersFits.append([])
        population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )
        for G in range (GENERATIONS):
            print "|||||||||||||||||||||||| GENERATION: ",G,"||||||||||||||||||||||||"
            #printPopulation (population)
            #printCreature(population.creatureList[0])
            #printSynapse(population.creatureList[0].synapseList[0])
            naughtyWinnersFits[-1].append(population.avgWinningCreature.fitness)
            population.populate()
            population.setTrainingCreature()
            population.compete( CYCLES_PER_RUN )
            population.resolve()


    for i in range(len(naughtyWinnersFits)):
        name = 'Round: '+str(i)
        plt.figure(name, figsize=(8,8))
        plt.plot(naughtyWinnersFits[i])

    plt.show()

