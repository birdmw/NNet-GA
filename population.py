from creature import *
from multiprocessing import Pool, cpu_count, Process
from math import *
from random import *
import populationHelper as pHelp
from copy import *
from trueskill import *

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

        if self.creatureCount < 400:
            self.hiSigmaThreshold = 4
            self.loSigmaThresholdReset = 1.15
        else:
            self.hiSigmaThreshold = 4.5
            self.loSigmaThresholdReset = 1.45

        self.loSigmaThreshold = self.loSigmaThresholdReset
        self.avgMaxSigma = 10

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
        #self.pruneByMaxSigLowMu()
        #self.pruneByLowSigLowMu()
        #self.pruneByRank()
        #self.pruneByMu()
        #self.pruneByFitness


    def mutate ( self ):
        self.mutateByConstant()
        #self.mutateBySigma()

    def train ( self, args ):
        self.trainByELO(args[0],args[1])

    def populate( self ):
         while (len(self.creatureList) < self.creatureCount):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount,self.cycles))
              self.creatureList[-1].ID = len(self.creatureList)-1

    def repopulate( self ):
         self.repopulateSimple()
         #self.repopulateRandomInjections()

    def repopulateSimple(self):
         while (len(self.creatureList) < 2):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
         while (len(self.creatureList) < self.creatureCount):
              mother = choice( self.creatureList )
              father = choice( self.creatureList )
              child = self.mate( mother , father )
              child.ID = self.creatureList[-1].ID+1
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

    def pruneByELO ( self, killPercent = .50 ):
        copyCreatureList = deepcopy(self.creatureList)
        #print "creatureCount =", len ( self.creatureList )
        saveIDs = list()
        creatureCount = len ( self.creatureList )
        saveCount = creatureCount * max(min(1.0-killPercent,1.0),0.0)
        #print "savecount", saveCount
        while len(saveIDs) < (saveCount):
            self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
            i=0
            while (self.creatureList[i].ID in saveIDs):
                i+=1
            saveIDs.append(self.creatureList[i].ID)
            if len(saveIDs) < (saveCount):
                self.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
                i=0
                while (self.creatureList[i].ID in saveIDs):
                    i+=1
                saveIDs.append(self.creatureList[i].ID)
        #print pruneIDs
        for creature in self.creatureList:
                if not (creature.ID in saveIDs):
                    #print "removing, ", creature.ID
                    self.creatureList.remove(creature)
        #print "creatureCount =", len ( self.creatureList )
        #print "--------------------"
        #for c in self.creatureList:
            #print c.ID, "id"
        self.sortByID()
        
        for c in copyCreatureList:
            if not (c.ID in saveIDs):
                print c.ID, "id ", round(c.ELO.mu,1), "mu  ", round(c.ELO.sigma,2), "sigma", "--PRUNED--"
            else:
                print c.ID, "id ", round(c.ELO.mu,1), "mu  ", round(c.ELO.sigma,2), "sigma"
                

    def pruneByRank ( self ):
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

    def pruneByLowSigLowMu ( self ):
        avgMu=0.0
        avgSig=0

        for c in self.creatureList:
            avgMu+=c.ELO.mu
            avgSig+=c.ELO.sigma

        avgMu = avgMu / float(len(self.creatureList))
        avgSig = avgSig / float(len(self.creatureList))
        maxSig = max(c.ELO.sigma for c in self.creatureList)

        if avgSig > self.hiSigmaThreshold:#Enforces somesort of basic confidence of the creature
            avgSig = self.hiSigmaThreshold


        self.avgMaxSigma = (self.avgMaxSigma+2*maxSig)/3 #Weighted rolling average
        if round(self.avgMaxSigma,1)==round(maxSig,1): #If max sigma is not changing ( Your least certain creature is not getting more confident)
            self.loSigmaThreshold = maxSig + 1 #Force the prune condition
            
        print '   Avg Max sigma = ',self.avgMaxSigma
        print '   Max sigma = ',maxSig
        pruneCandidates = []
        for c in self.creatureList:
            if maxSig < self.loSigmaThreshold: #If below this threshold, all creatures become prune candidates
                if c.ELO.mu < avgMu: #Creature is relatively 'worse'
                    print '   Pruning creature:', c.ID
                    print '   average: sig=',avgSig,' mu=',avgMu
                    print '   creature: sig=',c.ELO.sigma ,' mu=',c.ELO.mu
                    self.creatureList.pop(self.creatureList.index(c))
                    self.avgMaxSigma = 10 # Just a reset value for the rolling average
                    self.loSigmaThreshold = self.loSigmaThresholdReset

            elif c.ELO.sigma < avgSig: #If we are confident about the creature (relative to other creatures)
                if c.ELO.mu < avgMu: #Creature is relatively 'worse'
                    print '   Pruning creature: ', c.ID
                    print '   average: sig=',avgSig,' mu=',avgMu
                    print '   creature: sig=',c.ELO.sigma ,' mu=',c.ELO.mu
                    self.creatureList.pop(self.creatureList.index(c))
                    self.avgMaxSigma = 10 # Reset value of the rolling average since new creatures have been added, and we are no longer confident of the entire population
                    self.loSigmaThreshold = self.loSigmaThresholdReset #Just incase, but this shouldn't be needed


    def pruneByMaxSigLowMu ( self ):
        avgMu=0.0
        avgSig=0

        for c in self.creatureList:
            avgMu+=c.ELO.mu
            avgSig+=c.ELO.sigma

        avgMu = avgMu / float(len(self.creatureList))
        avgSig = avgSig / float(len(self.creatureList))
        maxSig = max(c.ELO.sigma for c in self.creatureList)

        self.avgMaxSigma = (self.avgMaxSigma+3*maxSig)/4 #Weighted rolling average: If population is not progressing, start killing punks
        if round(self.avgMaxSigma,1)==round(maxSig,1): #If max sigma is not changing ( Your least certain creature is not getting more confident) (Think weakest link)
            self.loSigmaThreshold = maxSig + 1 #Force the prune condition
            
        print '   Avg Max sigma = ',self.avgMaxSigma
        print '   Max sigma = ',maxSig

        if maxSig < self.loSigmaThreshold: 
            for c in self.creatureList:
                if c.ELO.mu < avgMu: #Creature is relatively 'worse'
                    print '   Pruning creature:', c.ID
                    print '   average: sig=',avgSig,' mu=',avgMu
                    print '   creature: sig=',c.ELO.sigma ,' mu=',c.ELO.mu
                    self.creatureList.pop(self.creatureList.index(c))
                    self.avgMaxSigma = 10 # Just a reset value for the rolling average.
                    self.loSigmaThreshold = self.loSigmaThresholdReset

            #elif c.ELO.sigma < avgSig: #If we are confident about the creature (relative to other creatures)
            #    if c.ELO.mu < avgMu: #Creature is relatively 'worse'
            #        print '   Pruning creature: ', c.ID
            #        print '   average: sig=',avgSig,' mu=',avgMu
            #        print '   creature: sig=',c.ELO.sigma ,' mu=',c.ELO.mu
            #        self.creatureList.pop(self.creatureList.index(c))
            #        self.avgMaxSigma = 10 # Reset value of the rolling average since new creatures have been added, and we are no longer confident of the entire population
            #        sel



    def updateELO(self,  creature1, creature2 ):
      if creature1.fitness > creature2.fitness:
        creature1.ELO,creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO)
      elif creature2.fitness > creature1.fitness:
        creature2.ELO,creature1.ELO = rate_1vs1(creature2.ELO,creature1.ELO)
      else:
        creature1.ELO, creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO, drawn=True)

    def mutateByConstant ( self, const = 1.0):
        mutateAmount = .012*const
        for creature in self.creatureList:
            #print "mutating by:", creature.ELO.sigma*mutateAmount
            for n in creature.neuronList:
                for p in n.propertyList:
                    p = max(min(gauss( p , mutateAmount),1000),-1000)
            for s in creature.synapseList:
                for p in s.propertyList:
                    #print "mutating synapse by:", creature.ELO.sigma*mutateAmount
                    p = max(min(gauss( p , mutateAmount),1000),-1000)
    

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
                for p in n.propertyList:
                    p = max(min(gauss( p , creature.ELO.sigma*mutateAmount),1000),-1000)
            for s in creature.synapseList:
                for p in s.propertyList:
                    #print "mutating synapse by:", creature.ELO.sigma*mutateAmount
                    p = max(min(gauss( p , creature.ELO.sigma*mutateAmount),1000),-1000)

    def trainByELO(self,rounds,battles):
        creatureList = self.creatureList
        for s in range(rounds):

            #pHelp.setTrainingConstant(self, 1.0)
            pHelp.setTrainingSin(self)
            #self.setTrainingTimes5()
            #pHelp.setTrainingTimes1(self)
            #pHelp.setTrainingTimes1Negative(self)
            #pHelp.setTrainingMinus1(self)
            pHelp.setPuts(self)
            #parallel code - broke for now
            '''
            p=Pool()
            self.creatureList = p.map(runCreature, creatureList)
            '''
            #serial code
            print '   Round: ',s
            print '   Running...'
            self.runPopulation()
            #for c in self.creatureList:
            #    self.runCreature(c)
            print '   Battling...'
            self.battle( battles )
            #self.battle_ffa()

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

   
    def battle_ffa( self ):
        #self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        fits = []
        CreatELO_ffaList=[]
        #TrueSkill(backend="scipy")
        for c in self.creatureList:
            fits.append(round((1-c.fitness)*10,2)) #For TrueSkill, 0 is the best.    #Yes, 'magic numbers' but all they change is the scale and resolution of fitness (11 digits)
            CreatELO_ffaList.append([c.ELO])

        try:
            newELO = rate(CreatELO_ffaList,fits) #Perform a ffa update on all creatures' ELO
        except:
            setup(backend="mpmath") #Increase calculation resolution. Errors occur with large populations
            newELO = rate(CreatELO_ffaList,fits) #Perform a ffa update on all creatures' ELO

        for c in self.creatureList:
            c.ELO = newELO[self.creatureList.index(c)][0]  #TrueSkill spits answers out in lenth (team size, so ffa = 1) tuples, needs to not be a tuple.

        #self.creatureList.sort(key = lambda x: x.ID, reverse=False)


    def runPopulation( self):
        for creature in self.creatureList:
            creature.run()

    def sortByFitness( self ):
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)
        return self

    def sortByMu( self ):
        self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
        return self

    def sortByID( self ):
        self.creatureList.sort(key = lambda x: x.ID, reverse=False)
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

    def removeCreature(self, creature ):
        self.creatureList.pop(creature)

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
