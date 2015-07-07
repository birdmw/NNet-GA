from population import *
from sobol_lib_NoNumpy import *
import sobolTrainerHelper as sHelp
import creatureHelper as cHelp
import populationHelper as pHelp

#only used in repeatability:
import time
import csv
from math import *

class sobolTrainer(object):
    """description of class"""
    def __init__(self, PopSize, NeuronCount, InputCount, OutputCount,MaxCycles,TrainingSets):
        self.popSize = PopSize
        self.inputCount = InputCount
        self.outputCount = OutputCount
        self.maxCycles = MaxCycles
        self.population = Population(PopSize,NeuronCount,InputCount,OutputCount,MaxCycles)
        self.trainingSets = TrainingSets
        #self.battles = Battles
        self.neuronCount = NeuronCount
        self.synCount = len(self.population.creatureList[0].synapseList)
        self.neurPropCount = len(self.population.creatureList[0].neuronList[0].propertyList)
        self.synPropCount =  len(self.population.creatureList[0].synapseList[0].propertyList)

        #!!!!!KNOBS!!!!!
        self.minPropertyValue = -1E2
        self.maxPropertyValue = 1E2


        self.sobolPoints, self.nextSeed = sHelp.generatePopulationSobolPoints(self.popSize,self.neuronCount,self.synCount,self.neurPropCount ,self.synPropCount,self.minPropertyValue,self.maxPropertyValue)

        for creatInd in range(self.popSize):
            self.population.creatureList[creatInd] = sHelp.assignCreatureProps(self.population.creatureList[creatInd],self.sobolPoints[creatInd])

    def testPopulation(self,battles=0):
        args = [self.trainingSets,battles]

        self.population.train(args)

        self.population.prune()

    def sobolRepopulate(self):
        newCount = self.popSize - len(self.population.creatureList)
        thisSeed = self.nextSeed
        print '  repopulating starting at seed:',self.nextSeed
        newPoints, self.nextSeed = sHelp.generatePopulationSobolPoints(newCount,self.neuronCount,self.synCount,self.neurPropCount ,self.synPropCount,self.minPropertyValue,self.maxPropertyValue,startSeed=self.nextSeed )
        print '  ending before seed:',self.nextSeed

        self.sobolPoints= self.sobolPoints +  newPoints

        for newC in range(newCount):
            self.population.creatureList.append(sHelp.assignCreatureProps(Creature(self.neuronCount,self.inputCount,self.outputCount),newPoints[newC]))
            self.population.creatureList[-1].ID = thisSeed+newC
            
        
        testList = []
        for c in self.population.creatureList:
            testList.append(c.ID)
        print '   New pop size: ', len(self.population.creatureList)
        print testList

    def purifyPopulation(self,inputSets,outputSets):
        for c in self.population.creatureList:
            if not sHelp.doesCreatureChange(c,inputSets,outputSets):                   
                print '   Culling creature:', c.ID
                print '   creature: sig=',c.ELO.sigma ,' mu=',c.ELO.mu
                self.population.creatureList.pop(self.population.creatureList.index(c))
                #sobTrain.population.removeCreature(c)

    def sobolEugenicRepopulate(self,inputSets,outputSets):
        newCount = self.popSize - len(self.population.creatureList)
        newCList=[]
        print '  repopulating starting at seed:',self.nextSeed
        while len(newCList) < (newCount - 4):
            thisSeed = self.nextSeed
            newPoints, self.nextSeed = sHelp.generatePopulationSobolPoints(newCount,self.neuronCount,self.synCount,self.neurPropCount ,self.synPropCount,self.minPropertyValue,self.maxPropertyValue,startSeed=self.nextSeed )
        
            for newC in range(newCount):
                newCList.append(sHelp.assignCreatureProps(Creature(self.neuronCount,self.inputCount,self.outputCount),newPoints[newC]))
                newCList[-1].ID = thisSeed+newC
                c=newCList[-1]
                if not sHelp.doesCreatureChange(c,inputSets,outputSets):                   
                    print '   Culling creature:', c.ID
                    #print '   creature: sig=',c.ELO.sigma ,' mu=',c.ELO.mu
                    newCList.pop(-1)
        print '  ending at seed:',self.nextSeed-1
        #while self.popSize > len(self.population.creatureList)+5: 
        #    self.sobolRepopulate()
        #    self.purifyPopulation(inputSets,outputSets)


def main():
    
    neuronCount = 10
    folder=r'C:\Users\chris.nelson\Desktop\NNet\SobolStructureCharacterizations\7N'

    inputCount = 1
    outputCount = 1
    MaxCycles = 1

    creatureCount = 200

    Generations= 20

    trainingSets = 25#50

    test_num = 11 #Don't forget to change the population trainer.

    battles = creatureCount*3#2
    
    
    #Used for repeatability test and ZE CULLING
    inputSets = [[-pi],[-pi/2],[0],[pi],[pi]]
    expectedOutputSets=[[-4],[1],[0],[1],[4]]
    
    for index in range(len(inputSets)):
        for index2 in range(len(inputSets[index])):
            if test_num == 11:
                expectedOutputSets[index][index2]=sin(inputSets[index][index2])



    print 'Battles = ',battles

    t_start = time.time()

    print 'Creating sobol population...'
    sobTrain = sobolTrainer(creatureCount, neuronCount, inputCount, outputCount,MaxCycles,trainingSets)
    
    sobTrain.purifyPopulation(inputSets,expectedOutputSets)
    sobTrain.sobolEugenicRepopulate(inputSets,expectedOutputSets)

    for g in range(Generations):
        print 'Generation: ',g+1
        print 'Training...'
        sobTrain.testPopulation(battles) #trains and prunes

        print 'Repopulating...'
        #sobTrain.sobolRepopulate()
        sobTrain.sobolEugenicRepopulate(inputSets,expectedOutputSets)

    print 'Training...' #One more for the road
    sobTrain.testPopulation(battles) #trains and prunes


    sobTrain.population.creatureList.sort(key = lambda x: x.ELO.mu, reverse=True)
    numCreatToInspect = len(sobTrain.population.creatureList)
    creats=[]
    c_ins=[]
    c_outs=[]
    c_mus=[]
    c_sigs=[]
    c_cycles=[]


    for i in range(numCreatToInspect):
        if sobTrain.population.creatureList[i].ELO.sigma < 60:
            creats.append(sobTrain.population.creatureList[i])
            c_ins.append([])
            c_outs.append([])
            c_mus.append(creats[-1].ELO.mu)
            c_sigs.append(creats[-1].ELO.sigma)
            c_cycles.append(creats[-1].cycles)

            for inI in range(len(creats[-1].output)):
                c_ins[-1].append(creats[-1].input[inI].inbox[0])

            for outI in range(len(creats[-1].output)):
                c_outs[-1].append(creats[-1].output[outI].outbox)
        i-=1


    #print 'Top Creatures Description:'
    #print '  Mu:',c_mus
    #print '  Sig:',c_sigs
    #print '  Last Cycles:',c_cycles
    #print '  Last Inputs: ',c_ins
    #print '  Last Outputs: ',c_outs
    
    inputSets = [[-2],[-1],[0],[1],[2]]
    expectedOutputSets=[[4],[1],[0],[1],[4]]


    
    fileName= str(neuronCount)+'N_test'+str(test_num)+'_RepeatabilityTest'
    localtime = time.localtime(time.time())
    Date = str(localtime[0])+'_'+str(localtime[1])+'_'+str(localtime[2])
    Time = str(localtime[3])+'_'+str(localtime[4])+'_'+str(localtime[5])
    file_name=folder+'\\'+fileName+'_'+str(Date)+'_'+str(Time)+'.csv'
    fdata = open(file_name,'a')
    scribe= csv.writer(fdata, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    scribe.writerow(['Creature ID',	'Sigma',	'Mu',	'Run',	'Inputs',	'Outputs',	'Fitness',	'Cycles'])

    runs = 4
    print 'Creature Repeatability tests...'
    for i in range(numCreatToInspect):
        #print '================================================'
        pHelp.testCreatureRepeatability(creats[i],inputSets,expectedOutputSets,runs,1,scribe)
        #print '================================================'
    
    duration = time.time() - t_start
    print 'Test duration:',duration
    scribe.writerow(['','','','','','','','','Test Duration:',duration])
    fdata.close()
    
    print 'theorhetical max creature ID:', len(sobTrain.sobolPoints)
    #seeCreature(demoCreature)

if __name__ == '__main__':
    main()
