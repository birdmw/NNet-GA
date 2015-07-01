from neuron import *
from synapse import *
from trueskill import Rating, quality_1vs1, rate_1vs1
import creatureHelper as cHelp

class Creature:
    def __init__(self , neuronCount=10, inputCount=1, outputCount=1):
        self.neuronCount, self.inputCount, self.outputCount = neuronCount, inputCount, outputCount
        self.neuronList, self.input, self.output, self.synapseList  = [], [], [], []
        self.fitness = 0.0
        self.ELO = Rating()
        self.ID = ''
        self.age = 0
        self.battleCount = 0

        #MAKE NEURONS
        for n in range(self.neuronCount):
            self.neuronList.append(Neuron(n))
        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])
            self.input[-1].isInput = 1
        for o in range (self.outputCount):
            self.output.append(self.neuronList[self.neuronCount - self.outputCount + o])
            self.output[-1].isOutput = 1

        #MAKE SYNAPSES
        createdSynapses = 0
        for n1 in self.neuronList:
            for n2 in self.neuronList:
                if not(n1 in self.output) and not(n2 in self.input ):
                    self.synapseList.append( Synapse(n1, n2, len(self.neuronList),createdSynapses ) )
                    n2.synapseList.append(self.synapseList[-1])
                    n2.inputSynapseCount += 1
                    createdSynapses+=1

    def run( self, cycles = 1, inputs = None ): #no cycles or population, that info is internal to creature now
        if inputs == None:
            for c in range( cycles ):
                for n in self.neuronList:
                    n.run()
                for s in self.synapseList:
                    s.run()
        else: #inputs is passed
            self.setInputs(inputs)
            for c in range( cycles ):
                for n in self.neuronList:
                    n.run()
                for s in self.synapseList:
                    s.run()

    def setInputs(self, inputs):
        for n in range(len(self.neuronList)):
            if self.neuronList[n].isInput == 1:
                self.neuronList[n].setNeuronInput(inputs[n])
            else:
                break


class DummyCreature:
    def __init__(self , neuronCount=10, inputCount=1, outputCount=1):
        self.neuronCount, self.inputCount, self.outputCount = neuronCount, inputCount, outputCount
        self.neuronList, self.input, self.output, self.synapseList  = [], [], [], []
        self.fitness = 0.0
        self.ELO = Rating()
        self.ID = ''
        self.offset = (random()-0.5)*4

        #MAKE NEURONS
        for n in range(self.neuronCount):
            self.neuronList.append(DummyNeuron(n))
        for i in range (self.inputCount):
            self.input.append(self.neuronList[i])
            self.input[-1].isInput = 1
        for o in range (self.outputCount):
            self.output.append(self.neuronList[self.neuronCount - self.outputCount + o])
            self.output[-1].isOutput = 1

        #MAKE SYNAPSES
        createdSynapses = 0
        for n1 in self.neuronList:
            for n2 in self.neuronList:
                if not(n1 in self.output) and not(n2 in self.input ):
                    self.synapseList.append( DummySynapse(n1, n2, len(self.neuronList),createdSynapses ) )
                    n2.synapseList.append(self.synapseList[-1])
                    n2.inputSynapseCount += 1
                    createdSynapses+=1

    def run( self, cycles = 1, inputs = [0], ForceTracking = True ): #no cycles or population, that info is internal to creature now
        self.setInputs(inputs)
        for c in range( cycles ):
            for n in self.neuronList:
                n.run()
            for s in self.synapseList:
                s.run()
        if ForceTracking:
            self.bypassNet(inputs)

    def setInputs(self, inputs):
##        for n in range(len(self.input)):
##            self.input[n].setNeuronInput(inputs[n])

        for n in range(len(self.neuronList)):
            if self.neuronList[n].isInput == 1:
                self.neuronList[n].setNeuronInput(inputs[n])
            else:
                break

    def bypassNet(self,outputs):
        inputList = []
##        for n in range(len(self.output)):
##            self.output[n].setNeuronInput(outputs[n]+self.offset)

        index = len(self.neuronList)-1
        for n in range(len(self.neuronList)):
            if self.neuronList[index-n].isOutput == 1:
                self.neuronList[index-n].setNeuronInput(outputs[-1*n-1]+self.offset)
            else:
                break



def main():
    neuronCount = 5
    inputCount = 1
    outputCount = 1
    cycles = 100
    inputSets=[]
    for i in range(inputCount):
        inputSets.append([])
        for c in range(cycles):
            inputSets[-1].append(2*random()-1)

    #demoCreature = Creature(neuronCount, inputCount, outputCount)
    demoCreature = DummyCreature(neuronCount, inputCount, outputCount)
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
    print '  Initial ELO: ',demoCreature.ELO
    print '  Inputs: ',inputSet
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

    #fileLoc = "C:\Users\chris.nelson\Desktop\NNet\DemoCreatures_5_14_2015\\"+"DemoCreature_"+str(neuronCount)+"N_ID"+str(demoCreature.ID)

    #cHelp.save_creature(demoCreature,fileLoc)

    #seeCreature(demoCreature)

if __name__ == '__main__':
    main()

