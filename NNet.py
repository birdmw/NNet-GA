from math import *
verbosity = 1
from random import *
import pygame
import sys

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount):
        self.neuronCount = neuronCount
        self.inputCount = inputCount
        self.outputCount = outputCount
        self.neuronList  = []
        self.synapseList = []
        self.outputList  = [] #contains values
        self.fitness = 0.0
##        generalA = random() / self.neuronCount
##        generalB = random() / (2*pi) / self.neuronCount
##        generalC = random() * pi / self.neuronCount
##        generalD = random() / self.neuronCount
##        generalThreshold = random()



        #temporary input method
        self.inputValues = []
        for i in range(self.inputCount):
            self.inputValues.append(random())
        if verbosity == 1:
            print 'Inputs:',self.inputValues

        for n in range(neuronCount):
            self.neuronList.append(Neuron(random()))

            if n < inputCount:
                self.neuronList[n].inbox = self.inputValues[n]

        if verbosity == 1:
            print 'number of neurons:',len(self.neuronList)

        for n1 in self.neuronList:
            for n2 in self.neuronList:
                self.synapseList.append(Synapse(n1, n2, random() / self.neuronCount, random() / (2*pi) / self.neuronCount, random() * pi / self.neuronCount, random() / self.neuronCount))

        if verbosity == 1:
            print 'number of synapses:', len(self.synapseList)

    def run(self,cycles):
        for i in range(cycles):
            #Should inject values into the INPUTS at this point
            ''' for each input neuron set the value to the desired values'''
            #for inputsInd in range(self.inputCount):
            #   neuronList[inputsInd].inbox = 0.1
            #run each neuron GOOD TIME TO APPLY PARALLEL PROCESSING
            outputs = []
            if verbosity:
                print 'cycle:',i
            #print self.neuronList[0].inbox
            for n in self.neuronList:
                ind = self.neuronList.index(n)
                if ind < self.inputCount:
                   n.inbox = self.inputs[ind] ##<--------------CHANGED THIS TO BBE SELF REFFERENTIAL
                n.run()
                if verbosity:
                    outputs.append(n.outToWorld)
            if verbosity:
                self.outputList.append(outputs)
                #print 'Outputs'
                #print self.outputList
            #run each synapse
            for s in self.synapseList:
                s.run()

    def setInputs(self,inputs):
        self.inputs = inputs

    def setTargets(self,targets):
        self.targets = targets

    def setFitness(self, fitness):
        self.fitness = fitness


class Neuron:
    def __init__(self, threshold):
        #self.index = index
        self.type = type
        self.threshold = threshold
        self.inbox = 0.0
        self.value = 0.0
        self.outbox = 0.0
        self.prevOutbox = 0.0
        self.outToWorld = 0.0


    def run(self):
        if verbosity:
            print round(self.inbox,2),'->',round(self.value,2),'/',round(self.threshold,2),'->',round(self.outbox,2)
        self.value += self.inbox
        self.outbox = 0.0
        if (abs(self.value) >= abs(self.threshold)):
            self.outbox += self.value
            self.outToWorld = self.value
            self.value = 0.0

        self.prevOutbox = self.outbox
        self.inbox = 0.0

    def clearValue():
        self.value = 0.0

class Synapse:
    def __init__(self, n1, n2, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.n1 = n1
        self.n2 = n2

    def run(self):
        out = self.a * sin(self.b * self.n1.outbox + self.c) + self.d * (self.n1.prevOutbox - self.n1.outbox) #NOTE: synapses fire a constant with empty inboxes this way (Dc offsets )
        self.n2.inbox += out

def see(creature):
    for n in creature.neuronList:
        print n.inbox , n.value , n.outbox

#MATE:
#hand this function a couple creatures and it will return an offspring of theirs
def mate (mother, father):
     child = Creature(mother.neuronCount, mother.inputCount, mother.outputCount)
     n = child.neuronList
     m = mother.neuronList
     f = father.neuronList
     for i in range(len(child.neuronList)):
          n[i].threshold = choice( [m[i].threshold, f[i].threshold] )
     s = child.synapseList
     m = mother.synapseList
     f = father.synapseList
     for i in range(len(child.synapseList)):
          s[i].a = choice( [m[i].a, f[i].a] )
          s[i].b = choice( [m[i].b, f[i].b] )
          s[i].c = choice( [m[i].c, f[i].c] )
          s[i].d = choice( [m[i].d, f[i].d] )
     return child

#GENERATE STARTING POPULATION:
#used to create an initial randomized population
def generateStartingPopulation(creatureCount, neuronCount, inputCount, outputCount):
    population = list()
    for c in range (creatureCount):
        population.append(Creature(neuronCount, inputCount, outputCount))
    return population

#GENERATE TRAINING SET:
#accepts an example creature and the number of sets to create
#trainingSet = generateTrainingSet(CreepyCreature)
#trainingSet[i][j][k] ##Let's call this a "Static Training Set"
# i: training set number
# j: input/output as 0/1
# k: input neuron # / output neuron #
#
#NOTE: We can technically add another dimension called cycles, l to catch dynamic input/output relationships
#Lets call this type a "Dynamic Training Set" and we wont use it yet

def generateTrainingSet(creature, noOfSets):
    trainingSet = []

    tempCombined = []
    for Set in range(noOfSets):
        temp1 = []
        temp2 = []
        for inputIndex in range(creature.inputCount):
            #What should the inputs be?
            temp1.append(float(bool(getrandbits(1))))#<---random bools
        for outputIndex in range(creature.outputCount):
            #what should the outputs be? Must have at least one
            if outputIndex == 0: #first output
                temp2.append(float(bool(temp1[0])^bool(temp1[1])))#<---xor for inputs 0 and 1
        trainingSet.append([temp1,temp2])
    return trainingSet

#FITNESS FUNCTION:
#returns a creatures fitness
#0.0 is a perfect fit
def fitnessFxn (creature):
    difference = 0
    if not(len(creature.targets)==creature.outputCount):
        print "output neuron list is not the same length as target list length"
    else:
        for i in range ( creature.outputCount ):
            output = creature.neuronList[ creature.neuronCount - creature.outputCount + i ].outbox
            target = creature.targets[i]
            difference = abs(target-output)
    fitness = ( difference / creature.outputCount )
    return fitness

#PRUNE:
#removes creatures that perform below average
def prune(population, cycles):
    avgHealth = 0
    for creature in population:

        creature.health = fitness(creature, targetOuts, cycles)
        avgHealth += creature.health
    avgHealth /= len( population )
    for creature in population:
        if (creature.health > avgHealth):
            population.remove(creature)
    return population

#REPOPULATE:
#population is the population of creatures
#creatureCount is the number of creatures to repopulate to
def repopulate(population, creatureCount):
     p = list(population)
     while (len(population) < creatureCount):
          mother = choice( p )
          father = choice( p )
          if not (mother == father):
            population.append( mate( mother , father ) )
     return population

#MUTATE POPULATION
#chanceOfMutation from 0.0 to 1.0
#ammountOfMutation is randomized between 0.0 and itself
def mutatePopulation (population, chanceOfMutation, ammountOfMutation):
     for c in population:
          for n in c.neuronList:
                if (random() <= chanceOfMutation):
                    n.threshold *= ammountOfMutation * random()
          for s in c.synapseList:
                if (random() <= chanceOfMutation):
                    s.a *= ammountOfMutation * random()
                if (random() <= chanceOfMutation):
                    s.b *= ammountOfMutation * random()
                if (random() <= chanceOfMutation):
                    s.c *= ammountOfMutation * random()
                if (random() <= chanceOfMutation):
                    s.d *= ammountOfMutation * random()
     return population

'''OVER-ARCHING CONCEPT:
i. generate a random population
  -For G generations:
    1. generate a training set of inputs and targets
    2. train population:
        I.  Set the inputs and outputs with creature.setInputs and creature.setTargets (each creature could technically have different targets herbavore / carnivore / plant / virus etc.)
        II. for each creature
            a. .run it for some number of cycles of interest
                i.  NOTE: we are assuming the trainingSet is of the Static variety for now
            b. at the end of the cycles use the fitness function to set the creature.fitness
                i.  NOTE: we could check at every cycle, and we would need to with a Dynamic Training Set
                ii. NOTE: You must always re-test old creatures (a rolling average would imply a "wisdom" honoring system)
                    -There is also an effect where older creatures have values (pun intended?)
    3. prune the unfit creatures
    4. repopulate a desired population size
    5. apply a tiny mutation to all creatures
    6. repeat with step 1
'''
TRAINING_SET_SIZE = 1
CYCLES_PER_RUN = 10
C = Creature(6,2,1)
trainingSet = generateTrainingSet(C,TRAINING_SET_SIZE)
for S in range ( TRAINING_SET_SIZE ):
    totalDifference=0
    C.setInputs(trainingSet[S][0])
    C.setTargets(trainingSet[S][1])
    C.run(CYCLES_PER_RUN)
    totalDifference += fitnessFxn(C)
C.setFitness(totalDifference/TRAINING_SET_SIZE)
print "Expected output", C.targets
print "Output",C.neuronList[-C.outputCount].outbox
print "fitness =",C.fitness