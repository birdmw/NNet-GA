from creature import *
from multiprocessing import Pool, cpu_count, Process
from math import *
from random import *

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

    def prune ( self ):
        self.pruneByELO()
        #self.pruneByMu()

    def mutate ( self ):
        self.mutateBySigma()

    def train ( self, TRAINING_SETS, BATTLES ):
        for s in range(TRAINING_SETS):
            #print "set:",s

            #self.setTrainingConstant(-10)
            #self.setTrainingSin()
            self.setTrainingTimes5()
            #self.setTrainingTimes1()
            self.setPuts()
            #parallel code - broke for now
            '''
            p=Pool()
            self.creatureList = p.map(runCreature, creatureList)
            '''
            #serial code
            for c in self.creatureList:
                runCreature(c)

            self.battle( BATTLES )

    '''
    def avgFitness(self):
        for c in self.creatureList:
            c.fitness = sum(c.fitnessList)/(float(len(c.fitnessList)))
    '''

    def battle( self, pairings ):
        #print "battle"
        creatureList = self.creatureList
        for p in range(pairings):
            #print p
            creature1 = choice( creatureList )
            creature2 = choice( creatureList )
            while creature1 == creature2:
                creature2 = choice( creatureList )

            self.updateELO(creature1, creature2)
        #print "battle - end"

    def pruneByELO ( self ):
        avgRank=0.0
        for c in self.creatureList:
            c.rank = c.ELO.mu / c.ELO.sigma
            avgRank += c.rank
        avgRank = avgRank / float(len(self.creatureList))
        count = len ( self.creatureList)
        index = 0
        while index < len(self.creatureList):
            #print "index:",index
            #print "len(self.creatureList):", len(self.creatureList)
            if self.creatureList[index].rank < avgRank:
                self.creatureList.pop(self.creatureList.index(self.creatureList[index]))
            else:
                index += 1
        self.sortByMu()

    def pruneByMu (self):
        self.sortByMu()
        half = len(self.creatureList)/2
        for k in range(half):
            self.creatureList.pop()

    def updateELO(self,  creature1, creature2 ):
      if creature1.fitness > creature2.fitness:
        creature1.ELO,creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO)
      elif creature2.fitness > creature1.fitness:
        creature2.ELO,creature1.ELO = rate_1vs1(creature2.ELO,creature1.ELO)
      else:
        creature1.ELO, creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO, drawn=True)

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
                child = self.mate( mother , father )
                self.creatureList.append( child )

    def mutateBySigma( self ):
        half = len(self.creatureList)/2

        maxOut = 0.0
        for i in range( len(self.trainingCreature.output)):
            maxOut = max(maxOut,self.trainingCreature.output[i].outbox, abs(self.trainingCreature.output[i].outbox))
            print maxOut
        self.rollingMaxOutput = ( self.rollingMaxOutput +  maxOut ) / 2
        #print " rolling max out", self.rollingMaxOutput
        mutateAmount = .0012*self.rollingMaxOutput

        for creature in self.creatureList:
            #print "mutating by:", creature.ELO.sigma*mutateAmount
            for n in creature.neuronList:
                n.threshold = max(min(gauss( n.threshold , creature.ELO.sigma*mutateAmount),1000000),-1000000)
            for s in creature.synapseList:
                #print "mutating synapse by:", creature.ELO.sigma*mutateAmount
                s.a = max(min(gauss( s.a , creature.ELO.sigma*mutateAmount),1000000),-1000000)
                s.b = max(min(gauss( s.b , creature.ELO.sigma*mutateAmount ),1000000),-1000000)
                s.c = max(min(gauss( s.c , creature.ELO.sigma*mutateAmount ),1000000),-1000000)
                s.d = max(min(gauss( s.d , creature.ELO.sigma*mutateAmount ),1000000),-1000000)

    def setTrainingTimes5( self ):
        inVal=random()
        for i in self.trainingCreature.input:
            i.inbox = [inVal]
        for o in self.trainingCreature.output:
            o.outbox = inVal*5.0

    def setTrainingTimes1( self ):
        inVal=random()
        for i in self.trainingCreature.input:
            i.inbox = [inVal]
        for o in self.trainingCreature.output:
            o.outbox = inVal

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
        self.expectedOutputs = []
        #print "expected:", self.expectedOutputs
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

    def mate (self, mother, father):
         child = Creature( self.neuronCount, self.inputCount, self.outputCount  )
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

def runCreature( creature ):
    creature.run()
    return creature
