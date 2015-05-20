from neuron import *
from synapse import *
from trueskill import Rating, quality_1vs1, rate_1vs1
import creatureHelper as cHelp
import numpy as np

class Creature:
    def __init__(self , neuronCount, inputCount, outputCount,maxCycles=50):
        self.neuronCount, self.inputCount, self.outputCount = neuronCount, inputCount, outputCount
        self.maxCycles = maxCycles
        self.cycles = 0
        self.neuronList, self.input, self.output, self.synapseList  = [], [], [], []
        self.fitness = 0.0
        self.rank = random()
        self.ELO = Rating()
        self.expectedOutputs = None
        self.ID = ''

        for n in range(self.neuronCount):
            self.neuronList.append(Neuron(n))

        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])
            self.input[-1].isInput = 1

        for o in range (self.outputCount):
            index = self.neuronCount - self.outputCount + o
            self.output.append(self.neuronList[index])
            self.output[-1].isOutput = 1
            #self.expectedOutputs.append('')

        createdSynapses = 0
        for n1 in self.neuronList:
            for n2 in self.neuronList:
                if not n1 in self.output and not n2 in self.input:# and not n1==n2: #No feedback
                    self.synapseList.append( Synapse(n1, n2, len(self.neuronList),createdSynapses ) )
                    n2.inputSynapseCount+=1
                    n2.synapseList.append(self.synapseList[-1])
                    createdSynapses+=1


    def run( self ): #no cycles or population, that info is internal to creature now
        #self.run_until_converged()
        self.run_maxCycles()

    def run_maxCycles(self):
        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

        if self.expectedOutputs != None:
            totalCreatureOutputDifference=0
            for OutInd in range(len(self.output)):
                    tOut = self.expectedOutputs[OutInd]
                    cOut = self.output[OutInd].outbox
                    totalCreatureOutputDifference += abs(tOut-cOut)

            mu=0
            stdev = 1
            self.fitness = (self.fitness + cHelp.myGauss(mu,stdev,totalCreatureOutputDifference) ) / 2

    def run_pattern_1cycle(self):
        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

        totalCreatureOutputDifference=0
        for OutInd in range(len(self.output)):
                tOut = self.expectedOutputs[OutInd]
                cOut = self.output[OutInd].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)

        mu=0
        stdev = 1
        self.fitness = (self.fitness + cHelp.myGauss(mu,stdev,totalCreatureOutputDifference) ) / 2

    def run_until_converged(self):
        runFitness = 0.0
        outputTracker = []
        self.cycles = 0

        for cyc in range( self.maxCycles):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()

            self.cycles +=1

            totalCreatureOutputDifference = 0.0
            for OutInd in range(len(self.output)):
                tOut = self.expectedOutputs[OutInd]
                cOut = self.output[OutInd].outbox
                totalCreatureOutputDifference += abs(tOut-cOut)


            #10 is a magic number. Determine experimentally  THIS IS THE PID ALTERNATIVE TO PLAY WITH!!!
            #runFitness = (runFitness + cHelp.myGauss(0,10,totalCreatureOutputDifference) ) / 2
            mu=0
            stdev = 1
            self.fitness = cHelp.myGauss(mu,stdev,totalCreatureOutputDifference)

            #Number of starting cycles is magic number. Determine experimentally
            if cyc <= (self.neuronCount**2+10):#*(2.0/3.0):
                outputTracker.append([])
                for o in self.output:
                    outputTracker[-1].append(o.outbox)
            else:
                newVal=[]
                for o in self.output:
                    newVal.append(o.outbox)
                outputTracker = outputTracker[1:]+outputTracker[:1]
                outputTracker[-1] = newVal

                if cHelp.checkConvergence(outputTracker):
                    break
        # THIS IS THE PID ALTERNATIVE TO PLAY WITH!!!
        #self.fitness = (self.fitness + runFitness ) / 2



def main():
    neuronCount = 17
    inputCount =3
    outputCount = 5
    MaxCycles = 6
    inputSet=[]
##    expOut=[]
    for i in range(inputCount):
        inputSet.append(randint(0,10))

    demoCreature = Creature(neuronCount, inputCount, outputCount,MaxCycles)
    for i in range(len(inputSet)):
        demoCreature.input[i].inbox = [inputSet[i]]

    demoCreature.ID = randint(0,1000)
##    filename = r'C:\Users\chris.nelson\Desktop\NNet\CreatureDebugging\bestie4lyfe_2015_2_17_12_17_35'
##    demoCreature = load_creature(filename)


    print 'Creature Description:'
    print '  Number of neurons:',len(demoCreature.neuronList)
    print '  Number of synapses:',len(demoCreature.synapseList)
    print '  Number of inputs:',len(demoCreature.input)
    print '  Number of outputs:',len(demoCreature.output)
    print '  Max cycles to run:',demoCreature.cycles
    print '  Initial ELO: ',demoCreature.ELO
    print '  Inputs: ',inputSet
    print '  Expected Outputs: ',demoCreature.expectedOutputs

    demoCreature.run()

    outputs = []
    for outp in demoCreature.output:
        outputs.append(outp.outbox)
    print '  Outputs: ',outputs
    print '  Cycles ran: ',demoCreature.cycles


    #print ''
    #print 'Repeatability test:'
    #inputSets = [[0],[1]]
    #testCreatureRepeatability(demoCreature,inputSets,runs)

    fileLoc = "C:\Users\chris.nelson\Desktop\NNet\DemoCreatures_5_14_2015\\"+"DemoCreature_"+str(neuronCount)+"N_ID"+str(demoCreature.ID)

    cHelp.save_creature(demoCreature,fileLoc)

    #seeCreature(demoCreature)

if __name__ == '__main__':
    main()
