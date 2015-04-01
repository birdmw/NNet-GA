from creature import *
from multiprocessing import Pool, cpu_count, Process
from math import *
from random import *

class Population:

    def __init__(self, CreatureCount, NeuronCount, InputCount, OutputCount,Cycles):
        self.cycles = Cycles
        self.creatureList = []
        self.creatureCount = CreatureCount
        self.neuronCount = NeuronCount
        self.inputCount = InputCount
        self.outputCount = OutputCount
        self.speciesFitness = 0
        self.rollingMaxOutput = 0.0

        #Creature pseudo-creature data structures
        self.trainingCreature = Creature( self.neuronCount, self.inputCount , self.outputCount )      
        self.trainingCreature.cycles = self.cycles
        for out in self.trainingCreature.output:
            out.outbox = gauss(0,1)
        self.synapseCount = len ( self.trainingCreature.synapseList )

        self.statsCreature = Creature( self.neuronCount, self.inputCount , self.outputCount  )
        for inIndex in range(InputCount):
            self.statsCreature.input[inIndex].inbox = []

        self.statsCreature.fitness = []
        
        #once we start adding/deleting synapses/neurons, these two will either need to be updated with changes, or removed from the code:
        self.synapseCount = len ( self.trainingCreature.synapseList )
        self.neuronCount = NeuronCount


        #Generate the seed population
        self.populate()
         

    def prune ( self ):
        self.pruneByELO()
        #self.pruneByMu()
        #self.pruneByFitness


    def mutate ( self ):
        self.mutateBySigma()

    def train ( self, args ):
        self.trainByELO(args[0],args[1])

    def populate( self ):
         while (len(self.creatureList) < self.creatureCount):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))

    def repopulate( self ):
         self.repopulateSimple()
         #self.repopulateRandomInjections()

    def repopulateSimple(self):
         while (len(self.creatureList) < 2):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount):
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              if not(mother == father):
                child = self.mate( mother , father )
                self.creatureList.append( child )

    def repopulateRandomInjections( self ):
        while (len(self.creatureList) < 2): #If there is less than two creatures left, create a random new creature.
            self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
        while (len(self.creatureList) < self.creatureCount): #Breed until full population
            if randint(1,50) == 1:
                self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
            else:
                mother = choice( self.creatureList )
                father = choice( self.creatureList )
                if not (mother == father):
                    child = mate( mother , father )
                    self.creatureList.append( child )

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

    def pruneByFitness(self):
        '''
        Will delete bottom half of creature list. And any creatures with extremely low fitness
        '''
        self.sortByFitness()
        startLen = len(self.creatureList)
        toBeRemoved = []
        percentToPrune = 0.5 #Can be adjusted to kill more or less creatures

        self.creatureList = self.creatureList[:int(percentToPrune*(startLen))]


        if len(self.creatureList)==0:
            print '======== WARNING: ALL CREATURES DIED ========'
            self.populate()
            print '======== !!RANDOMLY REPOPULATED!! ========'

    def updateELO(self,  creature1, creature2 ):
      if creature1.fitness > creature2.fitness:
        creature1.ELO,creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO)
      elif creature2.fitness > creature1.fitness:
        creature2.ELO,creature1.ELO = rate_1vs1(creature2.ELO,creature1.ELO)
      else:
        creature1.ELO, creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO, drawn=True)


    def mutateBySigma( self ):
        half = len(self.creatureList)/2

        maxOut = 0
        for i in range( len(self.trainingCreature.output)):
            maxOut = max(maxOut,self.trainingCreature.output[i].outbox, abs(self.trainingCreature.output[i].outbox))

        self.rollingMaxOutput = ( self.rollingMaxOutput +  maxOut ) / 2
        #print " rolling max out", self.rollingMaxOutput
        
        
        
        #0.0012 is magic number. Evaluate
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

    def trainByELO(self,rounds,battles):
        creatureList = self.creatureList
        for s in range(rounds):

            #self.setTrainingConstant()
            #self.setTrainingSin()
            #self.setTrainingTimes5()
            self.setTrainingTimes1()
            self.setPuts()
            #parallel code - broke for now
            '''
            p=Pool()
            self.creatureList = p.map(runCreature, creatureList)
            '''
            #serial code
            self.runPopulation()
            #for c in self.creatureList:
            #    self.runCreature(c)

            self.battle( battles )

    def battle( self, pairings ):
        #print "battle"
        creatureList = self.creatureList
        for p in range(pairings):
            creature1 = choice( creatureList )
            creature2 = choice( creatureList )
            while creature1 == creature2:
                creature2 = choice( creatureList )

            self.updateELO(creature1, creature2)
        #print "battle - end"

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
        #print "expected:", self.expectedOutputs
        for c in self.creatureList:
            c.expectedOutputs = []
            for i in range(len(c.input)):
                c.input[i].inbox=self.trainingCreature.input[i].inbox
            for j in range(len(c.output)):
                c.expectedOutputs.append(self.trainingCreature.output[j].outbox)
            c.cycles = self.cycles

    def runPopulation( self):
        for creature in self.creatureList:
            creature.run(self, self.cycles)

    def sortByFitness( self ):
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        return self

    def sortByMu( self ):
        self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
        return self

    def sortBySigma( self ):
        self.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
        return self

    def mate (self, mother, father):
        child = Creature( self.neuronCount, self.inputCount, self.outputCount  )
        for nInd in range(len(child.neuronList)):
            for propInd in range(len(child.neuronList[nInd].propertyList)):
                if getrandbits(1):
                    child.neuronList[nInd].propertyList[propInd] = father.neuronList[nInd].propertyList[propInd]
                else:
                    child.neuronList[nInd].propertyList[propInd] = mother.neuronList[nInd].propertyList[propInd]
        
        for sInd in range(len(child.synapseList)):
            for propInd in range(len(child.synapseList[sInd].propertyList)):
                if getrandbits(1):
                    child.synapseList[sInd].propertyList[propInd] = father.synapseList[sInd].propertyList[propInd]
                else:
                    child.synapseList[sInd].propertyList[propInd] = mother.synapseList[sInd].propertyList[propInd]

        return child

    def runCreature(self, creature ):
        creature.run()
        return creature

def main():
    CreatureCount = 50
    NeuronCount = 3
    MaxCycles = 600
    TrainingSets = 2
    Battle = CreatureCount**2
    '''
    Lessons = 1
    LessonMutationDivider = 1
    GenerationMutationDivider = 10
    MaxValue=2000000

    runs = 5
    '''
    
##    trainingSetInputs = [[0,0],[0,1],[1,0],[1,1]]
##    trainingSetOutputs = [[0,0],[0,1],[1,0],[1,1]]
    trainingSetInputs = [[-2],[0],[2]]
    trainingSetOutputs = [[-2],[0],[2]]


    InputCount = len(trainingSetInputs[0])
    OutputCount = len(trainingSetOutputs[0])

    #inSetCopies = deepcopy(trainingSetInputs)
    #outSetCopies = deepcopy(trainingSetOutputs)
    
    print 'Creating population...'
    demoPop =  Population(CreatureCount, NeuronCount, InputCount, OutputCount,MaxCycles) #, Lessons, LessonMutationDivider,GenerationMutationDivider,MaxValue)
    
    print 'Population information:'
    print '  Number of creatures:',demoPop.creatureCount
    print "  Top MU: ",demoPop.creatureList[0].ELO.mu
    print "  Top sigma: ",demoPop.creatureList[0].ELO.sigma
    print "  Top fitness: ",demoPop.creatureList[0].fitness

    print 'Training...'
    demoPop.train([TrainingSets, Battle])
    print 'Training Complete!'
    print 'Population information:'
    print '  Number of creatures:',demoPop.creatureCount
    print "  Top MU: ",demoPop.creatureList[0].ELO.mu
    print "  Top sigma: ",demoPop.creatureList[0].ELO.sigma
    print "  Top fitness: ",demoPop.creatureList[0].fitness
    print 'Pruning...'
    demoPop.prune()
    print 'Population information:'
    print '  Number of creatures:',demoPop.creatureCount
    print "  Top MU: ",demoPop.creatureList[0].ELO.mu
    print "  Top sigma: ",demoPop.creatureList[0].ELO.sigma
    print "  Top fitness: ",demoPop.creatureList[0].fitness
    print 'Repopulating...'
    demoPop.repopulate()
    print 'Population information:'
    print '  Number of creatures:',demoPop.creatureCount
    print "  Top MU: ",demoPop.creatureList[0].ELO.mu
    print "  Top sigma: ",demoPop.creatureList[0].ELO.sigma
    print "  Top fitness: ",demoPop.creatureList[0].fitness
    
    print '--FINISHED--'

    '''
    badStart = True
    print 'Finding valid starting population...'
    while(badStart):
        demoPop =  Population(CreatureCount, NeuronCount, InputCount, OutputCount,MaxCycles, Lessons, LessonMutationDivider,GenerationMutationDivider,MaxValue)

        #demoPop.run_generation_runUntilConverged_randHybridFitness(trainingSetInputs,trainingSetOutputs)
        #demoPop.run_generation_randHybridFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)
        demoPop.run_generation_gaussDistFitness_randCreatInjection(trainingSetInputs,trainingSetOutputs)



        if (demoPop.fitness > 1e-5):
            if (len(demoPop.creatureList) > CreatureCount/100):
                badStart = False
            else:
                print 'Failed. Retrying...'
        else:
            print 'Failed. Retrying...'
    '''


    '''
    localtime = time.localtime(time.time())
    Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
    Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])

    filename = r"C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\bestie4lyfe_"+Date+'_'+Time
    save_creature(demoPop.creatureList[0],filename)


    seeCreature(demoPop.creatureList[0] )
    '''

if __name__ == '__main__':
    main()
