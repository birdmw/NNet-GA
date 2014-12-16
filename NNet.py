from math import *
verbosity = 1
from random import *
import numpy as np
import sys

class population:
    def __init__(self, trainingSet, creatureList):
        self.tSet = trainingSet
        self.cList = creatureList
        self.outputList=[]
        self.fitnessList= []
        self.creatureCount = len(creatureList)
        self.inputCount = creatureList[0].inputCount
        self.neuronCount = creatureList[0].neuronCount
        self.outputCount = creatureList[0].outputCount
        self.tCount = len(trainingSet)
        self.masterOutputList = []
        self.sigmas = []
        self.targets = []
        for r in range(self.outputCount):
            self.sigmas.append(1.0)
            self.targets.append([])
            for t in range (len (tSet)):
                self.targets[-1].append(tSet[i][1][outputCount-r])
    def myGauss(mu,sig,val):
        a=1/(sig*(sqrt(2*pi)))
        return a*exp(-((val-mu)**2)/(2*sig**2))
    
    def calculateFitnessOfCreature(outputList):
        self.masterOutputList.append(outputList)
        perCreatFitnesses=[]
        Avs = []
        for o in range(self.outputCount):
            perCreatFitnesses.append([])
            for t in range (len (self.tSet)):
                perCreatFitnesses[-1].append(self.myGauss(self.targets[o][t],self.sigma[o],outputList[t][o]))
            Avs.append(sum(perCreatFitnesses[-1])/len(perCreatFitnesses[-1]))
        fitness = sum(Avs)/len(Avs)
        return fitness

    def updateSigma():
        #for each output
            #build a histogram of output
        for r in range(self.outputCount):
            self.sigmas[r] = np.std(np.array(self.masterOutputList[:][:][r]))
        self.masterOutputList=[]   
        
        '''for n in range len creatureList
        temp = []
        for each creature temp.append creature.output[n]
        self.fitnessList[n] = np.gauss(self.outputList[n][:][])'''
        #Calculate fitness for each creature for each output
        #For each training set, for each output, for each creature, calculate sigma per output
        #Of all creatures per training set

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
# k: input neuron # "or" output neuron #
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

#REPOPULATE:
#population is the population of creatures
#creatureCount is the number of creatures to repopulate to
def repopulate(population, creatureCount):
     p = list(population)
     while (len(population) < creatureCount):
          mother = choice( p )
          father = choice( p )
          if not (mother == father):
            child=mate( mother , father )
            mutateCreature(child)
            population.append( child )
     return population

def mutateCreature (c):
        for n in c.neuronList:
            if (random() <= chanceOfMutation):
                n.threshold *= amountOfMutation * random()
        for s in c.synapseList:
            if (random() <= chanceOfMutation):
                s.a *= amountOfMutation * random()
            if (random() <= chanceOfMutation):
                s.b *= amountOfMutation * random()
            if (random() <= chanceOfMutation):
                s.c *= amountOfMutation * random()
            if (random() <= chanceOfMutation):
                s.d *= amountOfMutation * random()
    return creature


#MUTATE POPULATION
#chanceOfMutation from 0.0 to 1.0
#ammountOfMutation is randomized between 0.0 and itself
def mutatePopulation (population, chanceOfMutation, amountOfMutation):
     for c in population:
          for n in c.neuronList:
                if (random() <= chanceOfMutation):
                    n.threshold *= amountOfMutation * random()
          for s in c.synapseList:
                if (random() <= chanceOfMutation):
                    s.a *= amountOfMutation * random()
                if (random() <= chanceOfMutation):
                    s.b *= amountOfMutation * random()
                if (random() <= chanceOfMutation):
                    s.c *= amountOfMutation * random()
                if (random() <= chanceOfMutation):
                    s.d *= amountOfMutation * random()
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
TRAINING_SET_SIZE = 10
CYCLES_PER_RUN = 10
chanceOfMutation = .01
amountOfMutation = .1
Creepy = Creature(6,2,1)
ceatureList.append(Creepy)
trainingSet = generateTrainingSet(Creepy,TRAINING_SET_SIZE)
P = population (trainingSet, creatureList)

for G in range (generations):
    P.fitnessList=[]
    for C in creatureList:
        OutList=[]
        for T in trainingSet:
            OutList.append([])
            
            C.setInputs(T[0])
            
            C.run(CYCLES_PER_RUN)
            for outNeuron in C.neuronList[:-C.outputCount]:
                OutList[-1].append(outNeuron.outbox)
            
        C.setFitness = P.calculateFitnessOfCreature(OutList)
        P.fitnessList.append(C.fitness)
    P.updateFitness()
    creatureList = P.naturalSelection() #inludes prune and repopulate

#C.setFitness(totalDifference/TRAINING_SET_SIZE)
print "Expected output", C.targets
print "Output",C.neuronList[-C.outputCount].outbox
print "fitness =",C.fitness
