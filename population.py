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
        for out in range(len(self.trainingCreature.output)):
            self.trainingCreature.output[out].outbox = gauss(0,1)
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
         

    def prune ( self , killPercent = .50 ):
        self.pruneByELO(killPercent)
        #self.pruneByMaxSigLowMu()
        #self.pruneByLowSigLowMu()
        #self.pruneByRank()
        #self.pruneByMu()
        #self.pruneByFitness


    def mutate ( self, constant = 1.0 ):
        self.mutateByConstant(constant)
        #self.mutateBySigma()

    def train ( self, args ):
        #self.trainByELO(args[0],args[1])
        self.trainByELO_patterns(args[0],args[1],args[2])

    def populate( self ):
         while (len(self.creatureList) < self.creatureCount):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount,self.cycles))
              self.creatureList[-1].ID = len(self.creatureList)-1

    def repopulate( self, matePercent = .25, mutateAmount = .01 ):
         self.repopulateByElo(matePercent, mutateAmount)
         #self.repopulateSimple()
         #self.repopulateRandomInjections()

    def repopulateByElo(self, matePercent, mutateAmount):
        print "repopulatebyelo", matePercent
        copyCreatureList = deepcopy(self.creatureList)#for print
        nonBreederIDs = []
        creatureCount = len ( self.creatureList )
        nonBreederCount = creatureCount * int(max(min(1-matePercent,1.0),0.0))
        #print "nonBreederCount: ", nonBreederCount
        #print "len(self.creatureList)", len(self.creatureList)
        #print "len(nonBreederIDs)", len(nonBreederIDs)
        
        while len(nonBreederIDs) < (nonBreederCount):
            self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=False)
            for c in self.creatureList:
                if not (c.ID in nonBreederIDs):
                    nonBreederIDs.append(self.creatureList[i].ID)
                    break
            if len(nonBreederIDs) < (nonBreederCount):
                self.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
                for c in self.creatureList:
                    if not (c.ID in nonBreederIDs):
                        nonBreederIDs.append(self.creatureList[i].ID)
                        break

        print "nonBreederIDs: ", len(nonBreederIDs)
        while (len(self.creatureList) < self.creatureCount):
            motherID,fatherID,father,mother = None, None, None, None
            while mother == None or father == None:
                for creature in self.creatureList:
                        if not(creature.ID in nonBreederIDs): #is a breeder
                            if ( random() < ( 1.0 / float( len( self.creatureList ) - nonBreederCount ) ) ) and mother==None: #random chance
                                mother = creature
                            if ( random() < ( 1.0 / float( len( self.creatureList ) - nonBreederCount ) ) ) and father==None: #random chance
                                father = creature        
            child = self.mate( mother , father )
            if mother == father:
                self.mutateCreatureByConstant(child, mutateAmount)
            child.ID = self.creatureList[-1].ID+1
            self.creatureList.append( child )
        self.sortByID()
        '''
        for c in copyCreatureList:
            if not (c.ID in nonBreederIDs):
                print c.ID, "id ", round(c.ELO.mu,1), "mu  ", round(c.ELO.sigma,2), "sigma", "--CANDIDATE BREEDER--"
            else:
                print c.ID, "id ", round(c.ELO.mu,1), "mu  ", round(c.ELO.sigma,2), "sigma"
        '''

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

    def pruneByELO ( self, killPercent ):
        print "prunebyelo", killPercent
        copyCreatureList = deepcopy(self.creatureList)
        saveIDs = list()
        creatureCount = len ( self.creatureList )
        saveCount = creatureCount * max(min(1.0-killPercent,1.0),0.0)
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
        for creature in self.creatureList:
                if not (creature.ID in saveIDs):
                    self.creatureList.remove(creature)
        self.sortByID()
        '''
        for c in copyCreatureList:
            if not (c.ID in saveIDs):
                print c.ID, "id ", round(c.ELO.mu,1), "mu  ", round(c.ELO.sigma,2), "sigma", "--PRUNED--"
            else:
                print c.ID, "id ", round(c.ELO.mu,1), "mu  ", round(c.ELO.sigma,2), "sigma"
        '''
                
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
      if creature1.averageFitness > creature2.averageFitness:
        creature1.ELO,creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO)
      elif creature2.averageFitness > creature1.averageFitness:
        creature2.ELO,creature1.ELO = rate_1vs1(creature2.ELO,creature1.ELO)
      else:
        creature1.ELO, creature2.ELO = rate_1vs1(creature1.ELO,creature2.ELO, drawn=True)

    def mutateByConstant ( self, constant):
        mutateAmount = constant
        print "mutateAmount", mutateAmount
        print "before", self.creatureList[0].neuronList[0].propertyList[0]
        for creature in range(len(self.creatureList)):
            for s in range(len(self.creatureList[creature].synapseList)):
                for p in range(len(self.creatureList[creature].synapseList[s].propertyList)):
                    self.creatureList[creature].synapseList[s].propertyList[p] = max(min(gauss( self.creatureList[creature].synapseList[s].propertyList[p] , mutateAmount),1000),-1000)
            for n in range(len(self.creatureList[creature].neuronList)):
                for p in range(len(self.creatureList[creature].neuronList[n].propertyList)):
                    self.creatureList[creature].neuronList[n].propertyList[p] = max(min(gauss( self.creatureList[creature].neuronList[n].propertyList[p] , mutateAmount),1000),-1000)
        print "final", self.creatureList[0].neuronList[0].propertyList[0]

    def mutateCreatureByConstant ( self, creature, constant = 1.0):
        mutateAmount = constant
        print "mutateAmount", mutateAmount
        for s in range(len(creature.synapseList)):
            for p in range(len(creature.synapseList[s].propertyList)):
                creature.synapseList[s].propertyList[p] = max(min(gauss( creature.synapseList[s].propertyList[p] , mutateAmount),1000),-1000)
        for n in range(len(creature.neuronList)):
            for p in range(len(creature.neuronList[n].propertyList)):
                creature.neuronList[n].propertyList[p] = max(min(gauss( creature.neuronList[n].propertyList[p] , mutateAmount),1000),-1000)
        #print "final", self.creatureList[0].neuronList[0].propertyList[0]


    def trainByELO(self,wars,battles):
        creatureList = self.creatureList
        for s in range(wars):
            for i in range ( len ( self.creatureList ) ):
                #pHelp.setTrainingConstant(self, 1.0)
                pHelp.setTrainingSin(self)
                #pHelp.setTrainingTimes5(self)
                #pHelp.setTrainingTimes1(self)
                #pHelp.setTrainingTimes1Negative(self)
                #pHelp.setTrainingMinus1(self)
                pHelp.setCreaturePuts(self,i)
            #parallel code - broke for now
            '''
            p=Pool()
            self.creatureList = p.map(runCreature, creatureList)
            '''
            #serial code
            print '   War: ',s
            print '   Running...'
            self.runPopulation()
            #for c in self.creatureList:
            #    self.runCreature(c)
            print '   Battling...'
            self.battle( battles )
            #self.battle_ffa()

    def trainByELO_patterns(self,rounds,battles,patternLength):
        creatureList = self.creatureList
        for s in range(rounds):
            print '   Round: ',s
            print '   Running...'

            for cInd in range(len(self.creatureList)):
                self.creatureList[cInd].fitness = 0

            for pattPosition in range(patternLength):
                pHelp.setTraining_sineGenerator(self, pattPosition)
                pHelp.setPuts(self)
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


    def runPopulation( self, ):
        for creature in self.creatureList:
            creature.run()
        for creature in self.creatureList:
            creature.averageFitness = sum(creature.fitnessList)/float(len(creature.fitnessList))

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
