from math import *
from random import *
from copy import *

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
        self.synapseCount = len ( self.avgWinningCreature.synapseList )
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
        for o in self.trainingCreature.output:
            o.outbox = float(bool(self.trainingCreature.input[0])^bool(self.trainingCreature.input[1]))##<---xor for inputs 0 and 1

    def compete( self, CYCLES_PER_RUN ):
        for creature in self.creatureList:
            creature.run(self, CYCLES_PER_RUN)

    def resolve( self ):
        sumFitness = 0.0
        for creature in self.creatureList:
            sumFitness += creature.fitness
        averageFitness = sumFitness / self.creatureCount
        winningCreatures = []
        losingCreatures = []
        for creature in self.creatureList:
            if (creature.fitness >= averageFitness):
                print "pass ",creature.fitness
                winningCreatures.append(creature)
            else:
                losingCreatures.append(creature)
                print "fail ",creature.fitness

        self.avgWinningCreature.fitness = sum( w.fitness for w in winningCreatures) / self.creatureCount
        for i in range ( self.neuronCount ):
            self.avgWinningCreature.neuronList[i].threshold = sum( w.neuronList[i].threshold for w in winningCreatures) / self.neuronCount

        for i in range ( self.synapseCount ):
            self.avgWinningCreature.synapseList[i].a = sum( w.synapseList[i].a for w in winningCreatures ) / self.synapseCount
            self.avgWinningCreature.synapseList[i].b = sum( w.synapseList[i].b for w in winningCreatures ) / self.synapseCount
            self.avgWinningCreature.synapseList[i].c = sum( w.synapseList[i].c for w in winningCreatures ) / self.synapseCount
            self.avgWinningCreature.synapseList[i].d = sum( w.synapseList[i].d for w in winningCreatures ) / self.synapseCount

        for i in range ( self.neuronCount ):
            self.avgLosingCreature.neuronList[i].threshold = sum( l.neuronList[i].threshold for l in losingCreatures ) / self.neuronCount

        for i in range ( self.synapseCount ):
            self.avgLosingCreature.synapseList[i].a = sum( l.synapseList[i].a for l in losingCreatures ) / self.synapseCount
            self.avgLosingCreature.synapseList[i].b = sum( l.synapseList[i].b for l in losingCreatures ) / self.synapseCount
            self.avgLosingCreature.synapseList[i].c = sum( l.synapseList[i].c for l in losingCreatures ) / self.synapseCount
            self.avgLosingCreature.synapseList[i].d = sum( l.synapseList[i].d for l in losingCreatures ) / self.synapseCount

        for i in range ( self.neuronCount ):
            self.sigmaCreature.neuronList[i].threshold = abs( self.avgWinningCreature.neuronList[i].threshold - self.avgLosingCreature.neuronList[i].threshold  )

        for i in range ( self.synapseCount ):
            self.sigmaCreature.synapseList[i].a = abs( self.avgWinningCreature.synapseList[i].a - self.avgLosingCreature.synapseList[i].a  )
            self.sigmaCreature.synapseList[i].b = abs( self.avgWinningCreature.synapseList[i].b - self.avgLosingCreature.synapseList[i].b  )
            self.sigmaCreature.synapseList[i].c = abs( self.avgWinningCreature.synapseList[i].c - self.avgLosingCreature.synapseList[i].c  )
            self.sigmaCreature.synapseList[i].d = abs( self.avgWinningCreature.synapseList[i].d - self.avgLosingCreature.synapseList[i].d  )

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
        self.error = 1.0
        self.fitness = 0.0
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
        error = 0.0
        for i in range ( len ( self.output ) ):
            try:
                error += abs( self.output[i].outbox - population.trainingCreature.output[i].outbox ) / abs( population.trainingCreature.output[i].outbox ) 
            except:
                absoluteError = abs( self.output[i].outbox - population.trainingCreature.output[i].outbox )
                if ( absoluteError<= 1.0 ):
                    error += absoluteError
                else:
                    pass
        error = error / self.outputCount
        self.error += ( self.outputCount * self.error + error ) / ( self.outputCount + 1)
        self.fitness = 1.0 - self.error

class Neuron:
    def __init__(self, threshold):
        self.threshold = threshold
        self.inbox = 0.0
        self.value = 0.0
        self.outbox = 0.0
        self.prevOutbox = 0.0

    def run(self):
        self.prevOutbox = self.outbox
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
        self.n1.outbox = 0.0

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


if __name__ == "__main__":
    CREATURE_COUNT = 20
    NEURON_COUNT= 6
    INPUT_COUNT = 2
    OUTPUT_COUNT = 2
    CYCLES_PER_RUN = 4
    GENERATIONS = 10

    population = Population ( CREATURE_COUNT, NEURON_COUNT, INPUT_COUNT, OUTPUT_COUNT )

    for G in range (GENERATIONS):
        print "GENERATION: ",G
        population.populate()
        population.setTrainingCreature()
        population.compete( CYCLES_PER_RUN )
        population.resolve()

