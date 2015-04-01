from population import *
from sobol_lib_NoNumpy import *


class sobolTrainer(object):
    """description of class"""
    def __init__(self, PopSize, NeuronCount, InputCount, OutputCount,MaxCycles):
        self.popSize = PopSize
        self.population = Population(PopSize,NeuronCount,InputCount,OutputCount,MaxCycles)
        self.neuronCount = NeuronCount
        self.SynCount = len(self.population.creatureList[0].synapseList)
        self.NeurPropCount = len(self.population.creatureList[0].neuronList[0].propertyList)
        self.SynPropCount =  len(self.population.creatureList[0].synapseList[0].propertyList)
        self.minPropertyValue = -1E4
        self.maxPropertyValue = 1E4
        self.sobolPoints, self.nextSeed = generatePopulationSobolPoints(PopSize,NeurCount,SynCount,NeurPropCount,SynPropCount,minPropertyValue,maxPropertyValue)

        for creatInd in range(self.popSize):
            self.population.creatureList[creatInd] = assignCreatureProps(self.population.creatureList[creatInd],self.sobolPoints[creatInd])

    def testPopulation(self):
        battles = self.popSize**2
        trainingSets = 100
        args = [trainingSets,battles]

        self.population.train(args)

        self.population.prune()

    def sobolRepopulate(self):
        newCount = self.popSize - len(self.population.creatureList)
        newPoints, self.nextSeed = generatePopulationSobolPoints(PopSize,NeurCount,SynCount,NeurPropCount,SynPropCount,minPropertyValue,maxPropertyValue,startSeed=self.nextSeed )

        self.sobolPoints= self.sobolPoints +  newPoints

        for newC in range(newCount):
            self.population.creatureList.append(assignCreatureProps(Creature(self.neuronCount,self.inputCount,self.outputCount),newPoints[newC]))

def main():
    neuronCount = 2
    inputCount =1
    outputCount = 1
    MaxCycles = 45

    inputSet = [0]
    expOut = [0]

    runs =6

    demoCreature = Creature(neuronCount, inputCount, outputCount,MaxCycles)
    for i in range(len(inputSet)):
        demoCreature.input[i].inbox = [inputSet[i]]

    demoCreature.expectedOutputs = expOut
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


    print ''
    print 'Repeatability test:'
    inputSets = [[0],[1]]
    testCreatureRepeatability(demoCreature,inputSets,runs)


    #seeCreature(demoCreature)

if __name__ == '__main__':
    main()
