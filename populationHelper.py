from math import *
from random import *

#only used in repeatability:
#from datetime import *
import time
import csv



def setTrainingTimes5( self ):
    inVal=random()
    for i in self.trainingCreature.input:
        i.inbox = [inVal]
    for o in self.trainingCreature.output:
        o.outbox = inVal*5.0

def setTrainingTimes1( self, inputSet=None):
    if inputSet == None:
        inVal=[random()] #In = [0,1]
    else:
        inVal = inputSet

    for i in self.trainingCreature.input:
        i.inbox = inVal
    for o in self.trainingCreature.output:
        o.outbox = inVal[0]

def setTrainingMinus1(self, inputSet=None):
    if inputSet == None:
        inVal=[random()*2-1] # In = [-1,1]
    else:
        inVal = inputSet

    for i in self.trainingCreature.input:
        i.inbox = inVal
    for o in self.trainingCreature.output:
        o.outbox = inVal[0]-1


def setTrainingTimes1Negative( self, inputSet=None):
    if inputSet == None:
        inVal=[random()*2-1]# In = [-1,1]
    else:
        inVal = inputSet

    for i in self.trainingCreature.input:
        i.inbox = inVal
    for o in self.trainingCreature.output:
        o.outbox = inVal[0]

def setTrainingConstant( self, const = 1.0 ):
    for i in self.trainingCreature.input:
        i.inbox = [const]
    for o in self.trainingCreature.output:
        o.outbox = const

def setTrainingSin( self ):
    randVal = 2*pi*random()
    for i in self.trainingCreature.input:
        i.inbox = [randVal]
    for o in self.trainingCreature.output:
        o.outbox = sin(randVal)

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
    #print "expected:", self.expectedOutputs
    for c in self.creatureList:
        #c.expectedOutputs = []
        for i in range(len(c.input)):
            c.input[i].inbox=self.trainingCreature.input[i].inbox
        for j in range(len(c.output)):
            c.expectedOutputs[j]=(self.trainingCreature.output[j].outbox)
        c.cycles = self.cycles

def testCreatureRepeatability(creature,inputSets,outputSets,runs,saveData=0,scribe='', verbosity=0):
    if verbosity:
        print 'Creature: fit=',creature.fitness, '   ELO=',creature.ELO

    repFit=0
    for inputSet in inputSets:
        inputs = []
        for i in range(len(inputSet)):
            creature.input[i].inbox = [inputSet[i]]
            inputs.append(creature.input[i].inbox[0])
        
        creature.expectedOutputs = outputSets[inputSets.index(inputSet)]

        if verbosity:
            print 'Inputs: ',inputs

        for r in range(runs):
            creature.run()
            outputs = []
            for outp in creature.output:
                outputs.append(outp.outbox)

            if verbosity:
                print '  Run',r,' Outputs: ',outputs, '    Cycles: ', creature.cycles, '   Fitness: ',creature.fitness
            if saveData:
                toWrite = [creature.ID,creature.ELO.sigma,creature.ELO.mu,r]
                scribe.writerow(toWrite+inputs+outputs+[creature.fitness,creature.cycles])

#def printPopulation(numCreatures):
