from creature import *
from math import *
from random import *
import populationHelper as pHelp
from copy import *
from trueskill import *

class Population:

    def __init__(self, CreatureCount=50, NeuronCount=50, InputCount=1, OutputCount=1):
        self.creatureList = []
        self.creatureCount = CreatureCount
        self.neuronCount = NeuronCount
        self.inputCount = InputCount
        self.outputCount = OutputCount
        self.issueID = 0
        #Generate the seed population
        self.populate()

    def populate( self ):
         while (len(self.creatureList) < self.creatureCount):
              self.creatureList.append(Creature(self.neuronCount,self.inputCount,self.outputCount))
              self.creatureList[-1].ID = self.issueID
              self.issueID += 1

    def repopulate(self, matePercent = .25, asexualChance = 0.1):#returns a list of IDs for children spawned asexually
        newCreatureIDList = []
        nonBreederIDs = []
        creatureCount = len ( self.creatureList )
        nonBreederCount = creatureCount * int(max(min(1-matePercent,1.0),0.0))

        #BUILD A LIST OF IDs FOR NON BREEDERS
        while len(nonBreederIDs) < (nonBreederCount):
            self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=False)
            i=0
            while (self.creatureList[i].ID in nonBreederIDs):
                i+=1
            nonBreederIDs.append(self.creatureList[i].ID)

            if len(nonBreederIDs) < (nonBreederCount):
                self.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)
                i=0
                while (self.creatureList[i].ID in nonBreederIDs):
                    i+=1
                nonBreederIDs.append(self.creatureList[i].ID)

        #FIND PARENTS FROM BREEDERS

        #LOOP OVER THIS##################################################
        breederIDs = []
        for creature in self.creatureList:
            if not (creature.ID in nonBreederIDs):
                breederIDs.append(creature.ID)


        #PICK PARENTS
        asexualOffspringList = []
        while (len(self.creatureList) < self.creatureCount):
            motherID = choice(breederIDs)
            fatherID = choice(breederIDs)
            if random() < asexualChance:
                fatherID = motherID

            for creature in self.creatureList:
                if creature.ID == motherID:
                    mother = creature
                if creature.ID == fatherID:
                    father = creature

            #can be optimized for asexual breeding

            child = self.mate( mother , father )
            child.ID = self.issueID
            self.issueID += 1
            if mother == father:
                asexualOffspringList.append(child.ID)
            self.creatureList.append( child )
        self.sortByID()
        return asexualOffspringList

    def mate (self, mother, father):
        child = Creature( self.neuronCount, self.inputCount, self.outputCount  )
        for nInd in range(len(child.neuronList)):
                if getrandbits(1):
                    child.neuronList[nInd] = father.neuronList[nInd]
                else:
                    child.neuronList[nInd] = mother.neuronList[nInd]
        for sInd in range(len(child.synapseList)):
                if getrandbits(1):
                    child.synapseList[sInd] = father.synapseList[sInd]
                else:
                    child.synapseList[sInd] = mother.synapseList[sInd]
        return child

    def sortByFitness( self ):
        self.creatureList.sort(key = lambda x: x.fitness, reverse=True)

    def sortByMu( self ):
        self.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)

    def sortBySigma( self ):
        self.creatureList.sort(key = lambda x: x.ELO.sigma, reverse=True)

    def sortByID( self ):
        self.creatureList.sort(key = lambda x: x.ID, reverse=False)

    #MAKE DICTIONARY FOR ID VS INDEX
    def IDToIndex (self, ID):
        for creature in self.creatureList:
            if creature.ID == ID:
                return self.creatureList.index(creature)
        print "No creature with ID ", ID

    def indexToID (self, index):
        return self.creatureList[index].ID
        print "No creature with index ", index
